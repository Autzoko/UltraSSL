"""
Patch-level lesion classifier using frozen DINOv2 backbone features.

Architecture:
    Input (B, 3, 224, 224)
    → Frozen DINO backbone → patch tokens (B, N_patches, embed_dim)
    → Patch classification head → per-patch logits (B, N_patches, 1)
    → MIL aggregation → image-level logit (B, 1)

Supports two MIL pooling strategies:
    - "topk": average of top-k% patch scores
    - "attention": learnable attention-weighted pooling
"""

import logging
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure dinov2 is importable
_project_root = Path(__file__).resolve().parent.parent.parent
_dinov2_root = _project_root / "dinov2"
if str(_dinov2_root) not in sys.path:
    sys.path.insert(0, str(_dinov2_root))
os.environ.setdefault("XFORMERS_DISABLED", "1")

from ultrassl.models.backbone import build_backbone

logger = logging.getLogger("ultrassl")


class PatchLesionClassifier(nn.Module):
    """Patch-level lesion detection using frozen DINOv2 features.

    The backbone extracts per-patch features. A lightweight head produces
    per-patch lesion scores. MIL pooling aggregates patch scores into an
    image-level prediction for reducing false positives.
    """

    def __init__(
        self,
        backbone_checkpoint: str,
        arch: str = "vit_base",
        patch_size: int = 14,
        img_size: int = 224,
        head_type: str = "linear",
        mil_type: str = "topk",
        topk_ratio: float = 0.1,
        mlp_hidden_dim: int = 256,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.img_size = img_size
        self.grid_h = img_size // patch_size
        self.grid_w = img_size // patch_size
        self.n_patches = self.grid_h * self.grid_w
        self.mil_type = mil_type
        self.topk_ratio = topk_ratio

        # Build and freeze backbone
        self.backbone, embed_dim = build_backbone(
            model_name=arch,
            patch_size=patch_size,
            pretrained=backbone_checkpoint,
            img_size=img_size,
            drop_path_rate=0.0,  # no stochastic depth at inference
        )
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

        # Classification head
        self.head = self._build_head(embed_dim, head_type, mlp_hidden_dim)

        # Attention pooling (if used)
        if mil_type == "attention":
            self.attn_pool = nn.Sequential(
                nn.Linear(embed_dim, 128),
                nn.Tanh(),
                nn.Linear(128, 1),
            )

        n_head_params = sum(p.numel() for p in self.head.parameters())
        logger.info(f"PatchLesionClassifier: {arch}, head={head_type} ({n_head_params} params), "
                    f"mil={mil_type}, grid={self.grid_h}x{self.grid_w}")

    def _build_head(self, embed_dim, head_type, mlp_hidden_dim):
        if head_type == "linear":
            return nn.Linear(embed_dim, 1)
        elif head_type == "mlp":
            return nn.Sequential(
                nn.Linear(embed_dim, mlp_hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(mlp_hidden_dim, 1),
            )
        else:
            raise ValueError(f"Unknown head_type: {head_type}")

    def train(self, mode=True):
        """Override to keep backbone always in eval mode."""
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, images):
        """Forward pass.

        Args:
            images: (B, 3, img_size, img_size) tensor.

        Returns:
            patch_logits: (B, N_patches, 1) per-patch classification logits.
            image_logit: (B, 1) MIL-aggregated image-level logit.
        """
        with torch.no_grad():
            out = self.backbone(images, is_training=True)
            patch_tokens = out["x_norm_patchtokens"]  # (B, N_patches, embed_dim)

        patch_logits = self.head(patch_tokens)  # (B, N_patches, 1)
        image_logit = self._mil_aggregate(patch_logits, patch_tokens)  # (B, 1)
        return patch_logits, image_logit

    def _mil_aggregate(self, patch_logits, patch_tokens):
        """Aggregate patch scores to image-level prediction."""
        if self.mil_type == "topk":
            k = max(1, int(self.n_patches * self.topk_ratio))
            scores = patch_logits.squeeze(-1)  # (B, N)
            topk_vals, _ = torch.topk(scores, k, dim=1)
            return topk_vals.mean(dim=1, keepdim=True)  # (B, 1)

        elif self.mil_type == "attention":
            # Attention weights from patch features (not from logits)
            attn_scores = self.attn_pool(patch_tokens.detach())  # (B, N, 1)
            attn_weights = F.softmax(attn_scores.squeeze(-1), dim=1)  # (B, N)
            weighted = (attn_weights.unsqueeze(-1) * patch_logits).sum(dim=1)  # (B, 1)
            return weighted

        else:
            raise ValueError(f"Unknown mil_type: {self.mil_type}")

    def get_trainable_params(self):
        """Return parameters that require gradients (head + optional attn_pool)."""
        params = list(self.head.parameters())
        if self.mil_type == "attention":
            params.extend(self.attn_pool.parameters())
        return params


# ── Patch-level label assignment ──────────────────────────────────────

def assign_patch_labels(
    bboxes: list,
    img_width: int,
    img_height: int,
    grid_h: int = 16,
    grid_w: int = 16,
    patch_size: int = 14,
    img_size: int = 224,
    ignore_margin: float = 0.0,
) -> torch.Tensor:
    """Compute per-patch binary labels from bounding boxes.

    Maps bboxes from original image coordinates to the 224x224 resized space,
    then assigns each patch as positive/negative based on whether its center
    falls inside any bbox.

    Args:
        bboxes: List of [x1, y1, x2, y2] in original pixel coordinates.
        img_width: Original image width.
        img_height: Original image height.
        grid_h: Patch grid height (default 16 for 224/14).
        grid_w: Patch grid width.
        patch_size: Patch size in pixels (14 for ViT-B/14).
        img_size: Resized image size (224).
        ignore_margin: Pixels in resized space near bbox borders to mark as
            ignore (label=-1). Set to 0 to disable.

    Returns:
        labels: (grid_h * grid_w,) tensor with values:
            0.0 = negative, 1.0 = positive, -1.0 = ignore (near border)
    """
    labels = torch.zeros(grid_h * grid_w, dtype=torch.float32)

    if not bboxes:
        return labels

    # Scale factors from original image space to resized space
    sx = img_size / img_width
    sy = img_size / img_height

    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        # Transform to resized coordinates
        rx1, ry1, rx2, ry2 = x1 * sx, y1 * sy, x2 * sx, y2 * sy

        for r in range(grid_h):
            for c in range(grid_w):
                # Patch center in resized image space
                cx = (c + 0.5) * patch_size
                cy = (r + 0.5) * patch_size
                idx = r * grid_w + c

                if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
                    labels[idx] = 1.0
                elif ignore_margin > 0 and labels[idx] == 0.0:
                    # Check if within margin of bbox border
                    if (rx1 - ignore_margin <= cx <= rx2 + ignore_margin and
                            ry1 - ignore_margin <= cy <= ry2 + ignore_margin):
                        labels[idx] = -1.0

    return labels
