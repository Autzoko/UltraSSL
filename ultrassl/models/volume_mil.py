"""
Volume-level MIL classifier components for breast ultrasound classification.

Shared building blocks used by all three volume classifiers:
  - joint_volume_classifier.py    (binary + multi-class)
  - lesion_presence_classifier.py (binary only)
  - lesion_subtype_classifier.py  (multi-class only)

Architecture:
    Per-slice CLS embeddings (B, K, 768)
    -> MIL aggregation (gated attention or top-k)
    -> Volume-level feature (B, 256)
    -> Task heads: binary (B, 1) and/or multi-class (B, n_classes)
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("ultrassl")


# ── Loss functions ────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """Binary focal loss (Lin et al. 2017).

    Reduces contribution of easy-to-classify examples, focusing training
    on hard negatives. Particularly useful for imbalanced has_lesion detection.

    Args:
        alpha: Weighting factor for positives (1-alpha for negatives).
        gamma: Focusing parameter. Higher = more focus on hard examples.
        pos_weight: Optional additional positive class weight tensor.
    """

    def __init__(self, alpha=0.25, gamma=2.0, pos_weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.register_buffer(
            "pos_weight",
            torch.tensor([pos_weight], dtype=torch.float32) if pos_weight is not None else None,
        )

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, 1) raw logits.
            targets: (B, 1) binary labels (0 or 1).

        Returns:
            Scalar focal loss.
        """
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none", pos_weight=self.pos_weight,
        )
        p = torch.sigmoid(logits)
        pt = p * targets + (1 - p) * (1 - targets)
        focal_weight = (1 - pt) ** self.gamma
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * focal_weight * bce
        return loss.mean()


class MultiClassFocalLoss(nn.Module):
    """Multi-class focal loss for lesion subtype classification.

    Args:
        gamma: Focusing parameter.
        weight: Per-class weight tensor (n_classes,).
        ignore_index: Label value to ignore (default -1).
    """

    def __init__(self, gamma=2.0, weight=None, ignore_index=-1):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        if weight is not None:
            self.register_buffer("weight", weight)
        else:
            self.weight = None

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, C) raw logits.
            targets: (B,) class indices.

        Returns:
            Scalar focal loss.
        """
        mask = targets != self.ignore_index
        if not mask.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        logits = logits[mask]
        targets = targets[mask]

        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean()


# ── MIL Aggregation ──────────────────────────────────────────────────

class GatedAttentionPool(nn.Module):
    """Gated attention MIL pooling (Ilse et al. 2018).

    Two parallel paths (Tanh and Sigmoid) produce gated attention scores.
    Supports masking for padded volumes.

    Args:
        embed_dim: Input embedding dimension.
        hidden_dim: Hidden dimension for attention networks.
        dropout: Dropout on attention weights.
    """

    def __init__(self, embed_dim=768, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.attention_V = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Tanh(),
        )
        self.attention_U = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.attention_W = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, embeddings, mask=None):
        """
        Args:
            embeddings: (B, K, D) per-slice embeddings.
            mask: (B, K) attention mask, 1=valid slice, 0=padding.

        Returns:
            pooled: (B, D) aggregated volume representation.
            attn_weights: (B, K) normalized attention weights.
        """
        V = self.attention_V(embeddings)    # (B, K, hidden)
        U = self.attention_U(embeddings)    # (B, K, hidden)
        scores = self.attention_W(V * U)    # (B, K, 1)
        scores = scores.squeeze(-1)         # (B, K)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=1)  # (B, K)
        attn_weights = self.dropout(attn_weights)

        pooled = torch.bmm(attn_weights.unsqueeze(1), embeddings).squeeze(1)  # (B, D)
        return pooled, attn_weights


class TopKPool(nn.Module):
    """Top-K MIL pooling: learn a scorer, select top-k slices, average embeddings.

    Args:
        embed_dim: Input embedding dimension.
        topk: Number of top slices to select.
    """

    def __init__(self, embed_dim=768, topk=8):
        super().__init__()
        self.scorer = nn.Linear(embed_dim, 1)
        self.topk = topk

    def forward(self, embeddings, mask=None):
        """
        Args:
            embeddings: (B, K, D) per-slice embeddings.
            mask: (B, K) attention mask, 1=valid, 0=padding.

        Returns:
            pooled: (B, D) aggregated volume representation.
            attn_weights: (B, K) indicator weights (1/k for selected, 0 otherwise).
        """
        scores = self.scorer(embeddings).squeeze(-1)  # (B, K)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        k = min(self.topk, embeddings.shape[1])
        topk_vals, topk_idx = torch.topk(scores, k, dim=1)

        # Gather top-k embeddings
        topk_idx_exp = topk_idx.unsqueeze(-1).expand(-1, -1, embeddings.shape[-1])
        topk_embeds = torch.gather(embeddings, 1, topk_idx_exp)  # (B, k, D)

        pooled = topk_embeds.mean(dim=1)  # (B, D)

        # Build pseudo attention weights for interpretability
        attn_weights = torch.zeros_like(scores)
        attn_weights.scatter_(1, topk_idx, 1.0 / k)

        return pooled, attn_weights


# ── Volume Classifier ────────────────────────────────────────────────

class VolumeClassifier(nn.Module):
    """Volume-level MIL classifier with optional binary + multi-class heads.

    Takes pre-extracted CLS token embeddings from a frozen DINO backbone
    and aggregates them via MIL pooling to produce volume-level predictions.

    Architecture:
        slice_embeds (B, K, embed_dim)
        -> MIL pool -> (B, embed_dim)
        -> projector -> (B, hidden_dim)
        -> binary_head -> (B, 1)        [if enable_binary]
        -> multiclass_head -> (B, C)    [if enable_multiclass]

    Args:
        embed_dim: DINO embedding dimension (768 for ViT-B).
        mil_type: "gated_attention" or "topk".
        hidden_dim: Dimension of the shared projection layer.
        n_classes: Number of subtype classes (3 for Class2/3/4).
        enable_binary: Include binary has_lesion head.
        enable_multiclass: Include multi-class subtype head.
        topk: Slices to select for topk pooling.
        dropout: Dropout rate in projector.
    """

    def __init__(
        self,
        embed_dim=768,
        mil_type="gated_attention",
        hidden_dim=256,
        n_classes=3,
        enable_binary=True,
        enable_multiclass=True,
        topk=8,
        dropout=0.25,
    ):
        super().__init__()
        self.enable_binary = enable_binary
        self.enable_multiclass = enable_multiclass

        # MIL pooling
        if mil_type == "gated_attention":
            self.pool = GatedAttentionPool(embed_dim, hidden_dim=128, dropout=0.1)
        elif mil_type == "topk":
            self.pool = TopKPool(embed_dim, topk=topk)
        else:
            raise ValueError(f"Unknown mil_type: {mil_type}")

        # Shared projector
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Task heads
        if enable_binary:
            self.binary_head = nn.Linear(hidden_dim, 1)
        if enable_multiclass:
            self.multiclass_head = nn.Linear(hidden_dim, n_classes)

        n_params = sum(p.numel() for p in self.parameters())
        logger.info(
            f"VolumeClassifier: mil={mil_type}, hidden={hidden_dim}, "
            f"binary={enable_binary}, multiclass={enable_multiclass} "
            f"({n_params:,} params)"
        )

    def forward(self, embeddings, mask=None):
        """
        Args:
            embeddings: (B, K, embed_dim) per-slice CLS tokens.
            mask: (B, K) attention mask (1=valid, 0=padding).

        Returns:
            dict with:
                binary_logit: (B, 1) if enable_binary
                class_logits: (B, n_classes) if enable_multiclass
                attn_weights: (B, K) MIL attention weights
                volume_embed: (B, hidden_dim) projected volume feature
        """
        pooled, attn_weights = self.pool(embeddings, mask)  # (B, D), (B, K)
        projected = self.projector(pooled)                   # (B, hidden_dim)

        out = {"attn_weights": attn_weights, "volume_embed": projected}

        if self.enable_binary:
            out["binary_logit"] = self.binary_head(projected)
        if self.enable_multiclass:
            out["class_logits"] = self.multiclass_head(projected)

        return out

    def get_trainable_params(self):
        """Return all parameters (everything is trainable; backbone is external)."""
        return list(self.parameters())
