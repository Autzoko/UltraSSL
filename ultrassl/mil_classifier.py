"""
Volume-level MIL classification: dataset, models, losses, and metrics.

Supports joint training of:
    1. Binary has_lesion detection (Focal Loss)
    2. Subtype classification (e.g., Class3 vs Class4, weighted CE)

Key components:
    - FocalLoss: binary focal loss for imbalanced has_lesion detection
    - GatedAttentionPool: gated attention MIL aggregation (Ilse et al. 2018)
    - TopKPool: top-k average MIL aggregation
    - JointVolumeClassifier: shared MIL + projection + binary/subtype heads
    - VolumeShardDataset: loads volume slices from WebDataset tar shards
    - CachedVolumeDataset: loads pre-extracted CLS embeddings from disk
    - scan_shard_volumes: scans tar shards to build volume-level index
    - volume_train_val_split: stratified train/val split by dataset
    - compute_binary_metrics / compute_subtype_metrics: evaluation utilities
"""

import glob
import io
import json
import logging
import os
import random
import tarfile
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

logger = logging.getLogger("mil_classifier")


# ============================================================================
# Losses
# ============================================================================


class FocalLoss(nn.Module):
    """Binary focal loss (Lin et al. 2017) for imbalanced detection.

    Reduces loss contribution from easy examples, focusing training on
    hard positives and negatives. Effective for the sparse has_lesion task.

    Args:
        alpha: Weighting factor for the positive class [0, 1].
        gamma: Focusing parameter — higher = more focus on hard examples.
        pos_weight: Additional multiplicative weight for positive samples.
    """

    def __init__(self, alpha=0.25, gamma=2.0, pos_weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, 1) raw logits.
            targets: (B, 1) binary labels {0, 1}.
        Returns:
            Scalar focal loss.
        """
        pw = (
            torch.tensor([self.pos_weight], device=logits.device)
            if self.pos_weight
            else None
        )
        bce = F.binary_cross_entropy_with_logits(
            logits, targets.float(), reduction="none", pos_weight=pw,
        )
        p = torch.sigmoid(logits)
        pt = targets * p + (1 - targets) * (1 - p)
        focal_weight = (1 - pt) ** self.gamma
        alpha_weight = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        return (alpha_weight * focal_weight * bce).mean()


# ============================================================================
# MIL Pooling
# ============================================================================


class GatedAttentionPool(nn.Module):
    """Gated attention MIL pooling (Ilse et al. 2018).

    Uses separate tanh (attention) and sigmoid (gate) paths whose
    element-wise product is projected to a scalar attention score per
    instance.  Softmax over instances yields the bag-level embedding.

    Args:
        embed_dim: Input embedding dimension (768 for ViT-B).
        hidden_dim: Bottleneck dimension for attention paths.
        dropout: Dropout on the gated attention features.
    """

    def __init__(self, embed_dim=768, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.attn_V = nn.Sequential(nn.Linear(embed_dim, hidden_dim), nn.Tanh())
        self.attn_U = nn.Sequential(nn.Linear(embed_dim, hidden_dim), nn.Sigmoid())
        self.attn_w = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, embeddings, mask=None):
        """
        Args:
            embeddings: (B, K, D) slice embeddings.
            mask: (B, K) bool — True for valid positions.
        Returns:
            pooled: (B, D) bag-level embedding.
            attn_weights: (B, K) attention weights (sum to 1 over valid).
        """
        V = self.attn_V(embeddings)  # (B, K, H)
        U = self.attn_U(embeddings)  # (B, K, H)
        scores = self.attn_w(self.dropout(V * U)).squeeze(-1)  # (B, K)

        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))

        attn_weights = F.softmax(scores, dim=1)  # (B, K)
        pooled = torch.bmm(attn_weights.unsqueeze(1), embeddings).squeeze(1)
        return pooled, attn_weights


class TopKPool(nn.Module):
    """Top-K MIL pooling: learn a scorer, select top-k, average.

    Args:
        embed_dim: Input embedding dimension.
        topk: Number of instances to select.
    """

    def __init__(self, embed_dim=768, topk=8):
        super().__init__()
        self.scorer = nn.Linear(embed_dim, 1)
        self.topk = topk

    def forward(self, embeddings, mask=None):
        """
        Args:
            embeddings: (B, K, D).
            mask: (B, K) bool.
        Returns:
            pooled: (B, D) average of top-k embeddings.
            weights: (B, K) indicator weights (1/k for selected, 0 otherwise).
        """
        scores = self.scorer(embeddings).squeeze(-1)  # (B, K)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))

        k = min(self.topk, embeddings.size(1))
        _, topk_idx = scores.topk(k, dim=1)  # (B, k)

        idx_exp = topk_idx.unsqueeze(-1).expand(-1, -1, embeddings.size(-1))
        pooled = torch.gather(embeddings, 1, idx_exp).mean(dim=1)  # (B, D)

        weights = torch.zeros_like(scores)
        weights.scatter_(1, topk_idx, 1.0 / k)
        return pooled, weights


# ============================================================================
# Classifier
# ============================================================================


class JointVolumeClassifier(nn.Module):
    """Joint binary (has_lesion) + subtype classifier with shared MIL.

    Architecture:
        CLS tokens (B, K, 768) → MIL pool → projection (768→hidden_dim)
        → binary_head (hidden_dim→1)  +  subtype_head (hidden_dim→n_subtypes)

    Args:
        embed_dim: Backbone CLS token dimension.
        hidden_dim: Shared projection bottleneck.
        n_subtypes: Number of subtype classes (2 for Class3/Class4).
        mil_type: "gated_attention" or "topk".
        topk: K for top-k pooling.
        dropout: Dropout in projection layer.
    """

    def __init__(
        self,
        embed_dim=768,
        hidden_dim=256,
        n_subtypes=2,
        mil_type="gated_attention",
        topk=8,
        dropout=0.25,
    ):
        super().__init__()

        if mil_type == "gated_attention":
            self.mil_pool = GatedAttentionPool(embed_dim, hidden_dim=128, dropout=0.1)
        elif mil_type == "topk":
            self.mil_pool = TopKPool(embed_dim, topk=topk)
        else:
            raise ValueError(f"Unknown mil_type: {mil_type}")

        self.projector = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.binary_head = nn.Linear(hidden_dim, 1)
        self.subtype_head = nn.Linear(hidden_dim, n_subtypes)

    def forward(self, cls_tokens, mask=None):
        """
        Args:
            cls_tokens: (B, K, embed_dim) per-slice CLS embeddings.
            mask: (B, K) bool — True for real slices, False for padding.
        Returns:
            dict with binary_logit (B,1), subtype_logits (B, n_subtypes),
            attn_weights (B, K), volume_embed (B, hidden_dim).
        """
        pooled, attn = self.mil_pool(cls_tokens, mask)
        proj = self.projector(pooled)
        return {
            "binary_logit": self.binary_head(proj),
            "subtype_logits": self.subtype_head(proj),
            "attn_weights": attn,
            "volume_embed": proj,
        }


# ============================================================================
# Shard scanning
# ============================================================================


def scan_shard_volumes(shard_dir, datasets=("Class3", "Class4")):
    """Scan WebDataset shards and build a volume-level index.

    Reads JSON annotations from all ``shard-*.tar`` files under each
    dataset directory and groups slices by ``sample_id`` (volume).

    Args:
        shard_dir: Root directory containing dataset subdirectories.
        datasets: Which dataset folders to include.

    Returns:
        dict: ``{sample_id: {"dataset", "has_lesion", "n_slices",
               "slices": [{"shard_path", "key_prefix", "slice_idx",
                            "has_lesion"}, ...]}}``.
    """
    volumes = {}
    for ds_name in datasets:
        ds_path = os.path.join(shard_dir, ds_name)
        if not os.path.isdir(ds_path):
            logger.warning(f"Dataset directory not found: {ds_path}")
            continue
        shard_files = sorted(glob.glob(os.path.join(ds_path, "shard-*.tar")))
        logger.info(f"  {ds_name}: scanning {len(shard_files)} shards ...")

        for shard_path in shard_files:
            with tarfile.open(shard_path, "r") as tar:
                for member in tar.getmembers():
                    if not member.name.endswith(".json"):
                        continue
                    f = tar.extractfile(member)
                    if f is None:
                        continue
                    meta = json.loads(f.read().decode("utf-8"))
                    sid = meta["sample_id"]

                    if sid not in volumes:
                        volumes[sid] = {
                            "dataset": meta.get("dataset", ds_name),
                            "has_lesion": 0,
                            "slices": [],
                        }

                    sl_hl = int(meta.get("has_lesion", 0))
                    volumes[sid]["has_lesion"] = max(volumes[sid]["has_lesion"], sl_hl)
                    volumes[sid]["slices"].append({
                        "shard_path": shard_path,
                        "key_prefix": member.name.rsplit(".", 1)[0],
                        "slice_idx": meta.get("slice_idx", 0),
                        "has_lesion": sl_hl,
                    })

    # Sort slices within each volume and record count
    for vol in volumes.values():
        vol["slices"].sort(key=lambda s: s["slice_idx"])
        vol["n_slices"] = len(vol["slices"])

    # Summary
    for ds in datasets:
        ds_vols = [v for v in volumes.values() if v["dataset"] == ds]
        n_pos = sum(v["has_lesion"] for v in ds_vols)
        logger.info(
            f"  {ds}: {len(ds_vols)} volumes "
            f"({n_pos} positive, {len(ds_vols) - n_pos} negative)"
        )
    logger.info(f"Total: {len(volumes)} volumes")
    return volumes


def volume_train_val_split(volume_index, val_ratio=0.15, seed=42):
    """Stratified train/val split by dataset.

    Uses deterministic seeding so all ranks get the same split.

    Returns:
        (train_ids, val_ids) — lists of sample_id strings.
    """
    rng = random.Random(seed)
    by_dataset = defaultdict(list)
    for vid, info in volume_index.items():
        by_dataset[info["dataset"]].append(vid)

    train_ids, val_ids = [], []
    for ds_name in sorted(by_dataset.keys()):
        vids = sorted(by_dataset[ds_name])
        rng.shuffle(vids)
        n_val = max(1, int(len(vids) * val_ratio))
        val_ids.extend(vids[:n_val])
        train_ids.extend(vids[n_val:])
        logger.info(f"  Split {ds_name}: {len(vids) - n_val} train, {n_val} val")

    return train_ids, val_ids


# ============================================================================
# Dataset — on-the-fly from shards
# ============================================================================


class VolumeShardDataset(Dataset):
    """Map-style dataset that loads volume slices from WebDataset tar shards.

    For each volume, randomly samples up to ``max_slices`` slices, loads their
    PNG images from the tar files, and returns the batch for backbone forward.

    Args:
        volume_index: Output of ``scan_shard_volumes()``.
        volume_ids: List of sample_id strings to include.
        max_slices: Maximum slices per volume (pad shorter, subsample longer).
        class_map: ``{dataset_name: int}`` mapping for subtype labels.
        img_size: Resize images to this size.
    """

    def __init__(
        self,
        volume_index,
        volume_ids,
        max_slices=32,
        class_map=None,
        img_size=224,
    ):
        self.volume_index = volume_index
        self.volume_ids = list(volume_ids)
        self.max_slices = max_slices
        self.class_map = class_map or {"Class3": 0, "Class4": 1}
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.volume_ids)

    def __getitem__(self, idx):
        vol_id = self.volume_ids[idx]
        vol = self.volume_index[vol_id]
        slices = vol["slices"]

        # Sample or use all slices
        if len(slices) > self.max_slices:
            selected = random.sample(slices, self.max_slices)
            n_valid = self.max_slices
        else:
            selected = list(slices)
            n_valid = len(selected)

        mask = torch.zeros(self.max_slices, dtype=torch.bool)
        mask[:n_valid] = True

        # Group by shard path for efficient tar access
        images = torch.zeros(self.max_slices, 3, 224, 224)
        shard_groups = defaultdict(list)
        for i, s in enumerate(selected):
            shard_groups[s["shard_path"]].append((i, s))

        for shard_path, items in shard_groups.items():
            with tarfile.open(shard_path, "r") as tar:
                for i, s in items:
                    try:
                        f = tar.extractfile(f"{s['key_prefix']}.png")
                        if f is not None:
                            img = Image.open(io.BytesIO(f.read())).convert("RGB")
                            images[i] = self.transform(img)
                    except Exception as e:
                        logger.debug(f"Failed to read slice: {e}")

        return {
            "images": images,
            "mask": mask,
            "has_lesion": torch.tensor(vol["has_lesion"], dtype=torch.float32),
            "subtype": torch.tensor(
                self.class_map.get(vol["dataset"], -1), dtype=torch.long,
            ),
            "volume_id": vol_id,
        }


# ============================================================================
# Dataset — cached embeddings
# ============================================================================


class CachedVolumeDataset(Dataset):
    """Load pre-extracted CLS embeddings from cache directory.

    Expects per-volume ``.pt`` files with keys:
    ``cls_tokens`` (N, 768), ``has_lesion`` (int), ``dataset`` (str).

    Args:
        cache_dir: Directory containing ``embeddings/<sample_id>.pt``.
        volume_ids: List of sample_id strings.
        max_slices: Pad/subsample to this many slices.
        class_map: ``{dataset_name: int}`` for subtype labels.
    """

    def __init__(self, cache_dir, volume_ids, max_slices=32, class_map=None):
        self.cache_dir = cache_dir
        self.volume_ids = list(volume_ids)
        self.max_slices = max_slices
        self.class_map = class_map or {"Class3": 0, "Class4": 1}

    def __len__(self):
        return len(self.volume_ids)

    def __getitem__(self, idx):
        vol_id = self.volume_ids[idx]
        path = os.path.join(self.cache_dir, "embeddings", f"{vol_id}.pt")
        data = torch.load(path, map_location="cpu", weights_only=True)

        tokens = data["cls_tokens"]  # (N, D)
        N = tokens.size(0)

        if N > self.max_slices:
            idx_sel = torch.randperm(N)[: self.max_slices]
            tokens = tokens[idx_sel]
            n_valid = self.max_slices
        else:
            n_valid = N

        padded = torch.zeros(self.max_slices, tokens.size(1))
        padded[:n_valid] = tokens[:n_valid]
        mask = torch.zeros(self.max_slices, dtype=torch.bool)
        mask[:n_valid] = True

        return {
            "cls_tokens": padded,
            "mask": mask,
            "has_lesion": torch.tensor(
                float(data.get("has_lesion", 0)), dtype=torch.float32,
            ),
            "subtype": torch.tensor(
                self.class_map.get(data.get("dataset", ""), -1), dtype=torch.long,
            ),
            "volume_id": vol_id,
        }


# ============================================================================
# Collation
# ============================================================================


def collate_volumes(batch):
    """Custom collate that stacks tensors and lists strings."""
    result = {}
    for key in batch[0]:
        if key == "volume_id":
            result[key] = [b[key] for b in batch]
        else:
            result[key] = torch.stack([b[key] for b in batch])
    return result


# ============================================================================
# Backbone embedding helper
# ============================================================================


def extract_cls_tokens(backbone, images, device, chunk_size=64):
    """Extract CLS tokens from a batch of volume images.

    Processes images through the frozen backbone in chunks to manage memory.

    Args:
        backbone: Frozen ViT backbone (eval mode, no grad).
        images: (B, K, 3, H, W) tensor of volume slice images.
        device: Torch device for computation.
        chunk_size: Forward-pass batch size to limit peak memory.

    Returns:
        cls_tokens: (B, K, embed_dim) on the same device.
    """
    B, K, C, H, W = images.shape
    flat = images.view(B * K, C, H, W).to(device)

    parts = []
    with torch.no_grad():
        for i in range(0, flat.size(0), chunk_size):
            out = backbone(flat[i : i + chunk_size], is_training=True)
            parts.append(out["x_norm_clstoken"])

    return torch.cat(parts, dim=0).view(B, K, -1)


# ============================================================================
# Metrics
# ============================================================================


def compute_binary_metrics(probs, labels):
    """Compute binary classification metrics.

    Args:
        probs: (N,) numpy array of predicted probabilities.
        labels: (N,) numpy array of ground truth {0, 1}.

    Returns:
        dict with accuracy, sensitivity, specificity, f1, auroc.
    """
    preds = (probs >= 0.5).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())

    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    prec = tp / max(tp + fp, 1)
    f1 = 2 * prec * sens / max(prec + sens, 1e-8)

    try:
        from sklearn.metrics import roc_auc_score

        auroc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else float("nan")
    except ImportError:
        auroc = float("nan")

    return {
        "accuracy": round(acc, 4),
        "sensitivity": round(sens, 4),
        "specificity": round(spec, 4),
        "f1": round(f1, 4),
        "auroc": round(auroc, 4) if not np.isnan(auroc) else auroc,
    }


def compute_subtype_metrics(probs, labels, class_names):
    """Compute multi-class subtype metrics.

    Args:
        probs: (N, C) numpy array of predicted probabilities.
        labels: (N,) numpy array of ground-truth class indices.
        class_names: List of class name strings, length C.

    Returns:
        dict with accuracy, per_class_f1, macro_f1, confusion_matrix.
    """
    preds = probs.argmax(axis=1)
    acc = float((preds == labels).mean())

    n_classes = len(class_names)
    per_class_f1 = {}
    for c in range(n_classes):
        tp = int(((preds == c) & (labels == c)).sum())
        fp = int(((preds == c) & (labels != c)).sum())
        fn = int(((preds != c) & (labels == c)).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        per_class_f1[class_names[c]] = round(
            2 * prec * rec / max(prec + rec, 1e-8), 4,
        )

    macro_f1 = float(np.mean(list(per_class_f1.values())))

    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(labels, preds):
        cm[int(t)][int(p)] += 1

    return {
        "accuracy": round(acc, 4),
        "per_class_f1": per_class_f1,
        "macro_f1": round(macro_f1, 4),
        "confusion_matrix": cm.tolist(),
    }
