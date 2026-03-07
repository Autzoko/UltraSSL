"""
Lesion center heatmap localizer using frozen DINOv2 patch tokens.

Architecture:
    Input (B, 3, 224, 224)
    → Frozen DINO backbone → patch tokens (B, 256, 768) → reshape (B, 768, 16, 16)
    → LocalizationHead (~1.4M params) → heatmap logits (B, 1, 64, 64)

Components:
    - LesionLocalizer: full model with frozen backbone + learnable head
    - LocalizationHead: conv decoder from patch features to heatmap
    - CenterNetFocalLoss: modified focal loss for Gaussian heatmap targets
    - generate_gaussian_heatmap: bbox → Gaussian center heatmap
    - detect_peaks: NMS-based peak detection on predicted heatmaps
    - compute_localization_metrics: center_hit_rate, recall, distance, FPR
    - build_heatmap_wds_pipeline: WebDataset loader with balanced sampling
"""

import glob
import json
import logging
import math
import os
import random
import tarfile
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import webdataset as wds

from ultrassl.models.backbone import build_backbone

logger = logging.getLogger("ultrassl")


# ============================================================================
# Gaussian heatmap generation
# ============================================================================


def generate_gaussian_heatmap(
    bboxes_normalized: list,
    heatmap_size: int = 64,
    min_sigma: float = 1.5,
) -> torch.Tensor:
    """Generate a Gaussian center heatmap from normalized bounding boxes.

    Args:
        bboxes_normalized: List of [nx1, ny1, nx2, ny2] in [0, 1] coords.
            Empty list → all-zero heatmap (negative slice).
        heatmap_size: Output heatmap spatial size (H = W).
        min_sigma: Minimum Gaussian sigma in heatmap pixels.

    Returns:
        (1, H, H) float32 tensor with values in [0, 1].
    """
    H = heatmap_size
    heatmap = torch.zeros(1, H, H, dtype=torch.float32)

    if not bboxes_normalized:
        return heatmap

    # Coordinate grids
    ys = torch.arange(H, dtype=torch.float32).unsqueeze(1)  # (H, 1)
    xs = torch.arange(H, dtype=torch.float32).unsqueeze(0)  # (1, H)

    for bbox in bboxes_normalized:
        nx1, ny1, nx2, ny2 = bbox

        # Center in heatmap space — snap to integer grid so peak == 1.0 exactly.
        # Required for CenterNet focal loss (pos_mask = target.eq(1)).
        cx = int(round((nx1 + nx2) / 2.0 * H))
        cy = int(round((ny1 + ny2) / 2.0 * H))
        cx = max(0, min(H - 1, cx))
        cy = max(0, min(H - 1, cy))

        # Bbox dims in heatmap space
        w_h = (nx2 - nx1) * H
        h_h = (ny2 - ny1) * H

        sigma = max(min(w_h, h_h) / 6.0, min_sigma)

        # Gaussian within 3-sigma radius for efficiency
        radius = int(math.ceil(3 * sigma))
        y_lo = max(0, int(cy) - radius)
        y_hi = min(H, int(cy) + radius + 1)
        x_lo = max(0, int(cx) - radius)
        x_hi = min(H, int(cx) + radius + 1)

        if y_lo >= y_hi or x_lo >= x_hi:
            continue

        local_ys = ys[y_lo:y_hi, :]
        local_xs = xs[:, x_lo:x_hi]

        gaussian = torch.exp(
            -((local_xs - cx) ** 2 + (local_ys - cy) ** 2) / (2 * sigma ** 2)
        )

        # Element-wise max for overlapping bboxes
        heatmap[0, y_lo:y_hi, x_lo:x_hi] = torch.maximum(
            heatmap[0, y_lo:y_hi, x_lo:x_hi], gaussian
        )

    return heatmap


# ============================================================================
# CenterNet-style focal loss
# ============================================================================


class CenterNetFocalLoss(nn.Module):
    """Modified focal loss for continuous Gaussian heatmap targets.

    Handles three cases:
        - Positive pixels (target == 1): -(1-p)^alpha * log(p)
        - Near-center pixels (0 < target < 1): down-weighted by (1-target)^beta
        - Background pixels (target == 0): -p^alpha * log(1-p)

    Loss is normalized by N = number of center points in the batch.

    Args:
        alpha: Focusing parameter for hard examples (default 2.0).
        beta: Down-weighting for near-center Gaussian tail pixels (default 4.0).
    """

    def __init__(self, alpha: float = 2.0, beta: float = 4.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute CenterNet focal loss.

        Args:
            pred_logits: (B, 1, H, H) raw logits (before sigmoid).
            target: (B, 1, H, H) Gaussian heatmap targets in [0, 1].

        Returns:
            Scalar loss.
        """
        # Force float32 — in AMP float16, 1-1e-6 rounds to 1.0 → log(0) = -inf
        pred = torch.sigmoid(pred_logits.float())
        pred = pred.clamp(1e-6, 1 - 1e-6)
        target = target.float()

        pos_mask = target.eq(1).float()
        neg_mask = target.lt(1).float()

        # Positive: -(1-p)^alpha * log(p)
        pos_loss = -torch.pow(1 - pred, self.alpha) * torch.log(pred) * pos_mask

        # Negative: -(1-target)^beta * p^alpha * log(1-p)
        neg_loss = (
            -torch.pow(1 - target, self.beta)
            * torch.pow(pred, self.alpha)
            * torch.log(1 - pred)
            * neg_mask
        )

        # Normalize by number of center points (at least 1)
        n_pos = pos_mask.sum()
        n_centers = max(n_pos.item(), 1.0)

        loss = (pos_loss.sum() + neg_loss.sum()) / n_centers
        return loss


# ============================================================================
# Localization head
# ============================================================================


class LocalizationHead(nn.Module):
    """Conv decoder from patch features (B, D, 16, 16) to heatmap (B, 1, 64, 64).

    Architecture:
        Conv2d(D→256, 1×1) + BN + ReLU        16×16
        Conv2d(256→256, 3×3, pad=1) + BN + ReLU 16×16
        ConvTranspose2d(256→128, 4, s=2, p=1)  32×32 + BN + ReLU
        ConvTranspose2d(128→64, 4, s=2, p=1)   64×64 + BN + ReLU
        Conv2d(64→1, 1×1)                      64×64 logits
    """

    def __init__(self, embed_dim: int = 768):
        super().__init__()

        self.layers = nn.Sequential(
            # 16×16 → 16×16
            nn.Conv2d(embed_dim, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # 16×16 → 32×32
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # 32×32 → 64×64
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # 64×64 → 64×64 (logits)
            nn.Conv2d(64, 1, kernel_size=1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        # Bias init for final conv: log(1/99) ≈ -2.19 so initial sigmoid ≈ 0.01
        final_conv = self.layers[-1]
        nn.init.constant_(final_conv.bias, -2.19)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, embed_dim, 16, 16) patch feature maps.

        Returns:
            (B, 1, 64, 64) raw logits.
        """
        return self.layers(x)


# ============================================================================
# Full localizer model
# ============================================================================


class LesionLocalizer(nn.Module):
    """Frozen DINO backbone + learnable localization head.

    Args:
        backbone_checkpoint: Path to pretrained weights or DINOv2 hub name.
        arch: Backbone architecture (default "vit_base").
        patch_size: Patch size (default 14).
        img_size: Input image size (default 224).
        unfreeze_last_n: Number of last transformer blocks to unfreeze (default 0).
    """

    def __init__(
        self,
        backbone_checkpoint: str,
        arch: str = "vit_base",
        patch_size: int = 14,
        img_size: int = 224,
        unfreeze_last_n: int = 0,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.img_size = img_size
        self.grid_size = img_size // patch_size  # 16 for 224/14

        # Build backbone
        self.backbone, embed_dim = build_backbone(
            model_name=arch,
            patch_size=patch_size,
            pretrained=backbone_checkpoint,
            img_size=img_size,
            drop_path_rate=0.0,
        )

        # Freeze backbone
        for p in self.backbone.parameters():
            p.requires_grad = False

        # Optionally unfreeze last N blocks
        if unfreeze_last_n > 0:
            for block in self.backbone.blocks[-unfreeze_last_n:]:
                for p in block.parameters():
                    p.requires_grad = True
            logger.info(f"Unfroze last {unfreeze_last_n} backbone blocks")

        self.backbone.eval()

        # Localization head
        self.head = LocalizationHead(embed_dim)

        n_head_params = sum(p.numel() for p in self.head.parameters()) / 1e6
        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6
        logger.info(
            f"LesionLocalizer: {arch}, head={n_head_params:.2f}M params, "
            f"trainable={n_trainable:.2f}M, grid={self.grid_size}x{self.grid_size}"
        )

    def train(self, mode=True):
        """Override to keep backbone always in eval mode."""
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            images: (B, 3, img_size, img_size) input images.

        Returns:
            (B, 1, 64, 64) raw heatmap logits.
        """
        B = images.shape[0]
        G = self.grid_size

        with torch.no_grad():
            out = self.backbone(images, is_training=True)
            patch_tokens = out["x_norm_patchtokens"]  # (B, G*G, D)

        # Reshape to spatial grid
        D = patch_tokens.shape[-1]
        features = patch_tokens.transpose(1, 2).reshape(B, D, G, G)  # (B, D, G, G)

        return self.head(features)

    def get_trainable_params(self):
        """Return only parameters that require gradients."""
        return [p for p in self.parameters() if p.requires_grad]


# ============================================================================
# Peak detection
# ============================================================================


def detect_peaks(
    heatmap: torch.Tensor,
    threshold: float = 0.3,
    nms_kernel: int = 3,
    max_detections: int = 5,
) -> list:
    """Detect center peaks from a predicted heatmap using NMS.

    Args:
        heatmap: (1, H, H) or (H, H) sigmoid-activated heatmap.
        threshold: Minimum confidence for a detection.
        nms_kernel: Kernel size for max-pool NMS.
        max_detections: Maximum number of peaks to return.

    Returns:
        List of dicts: [{"x": float, "y": float, "confidence": float}, ...]
        Coordinates are normalized to [0, 1].
    """
    if heatmap.dim() == 2:
        heatmap = heatmap.unsqueeze(0)  # (1, H, H)

    H = heatmap.shape[-1]
    hm = heatmap.unsqueeze(0)  # (1, 1, H, H) for max_pool2d

    # NMS: keep only local maxima
    pad = nms_kernel // 2
    pooled = F.max_pool2d(hm, kernel_size=nms_kernel, stride=1, padding=pad)
    keep = (hm == pooled) & (hm >= threshold)

    # Extract peak coordinates
    keep = keep.squeeze(0).squeeze(0)  # (H, H)
    ys, xs = torch.where(keep)
    scores = heatmap[0, ys, xs]

    if len(scores) == 0:
        return []

    # Sort by confidence, take top-k
    order = scores.argsort(descending=True)[:max_detections]

    peaks = []
    for idx in order:
        peaks.append({
            "x": (xs[idx].item() + 0.5) / H,
            "y": (ys[idx].item() + 0.5) / H,
            "confidence": scores[idx].item(),
        })

    return peaks


# ============================================================================
# Localization metrics
# ============================================================================


def compute_localization_metrics(
    pred_heatmaps: torch.Tensor,
    annotations: list,
    threshold: float = 0.3,
    nms_kernel: int = 3,
    max_detections: int = 5,
) -> dict:
    """Compute localization metrics over a batch.

    Args:
        pred_heatmaps: (B, 1, H, H) sigmoid-activated predicted heatmaps.
        annotations: List of B annotation dicts, each with "bboxes_normalized"
            and "has_lesion".

    Returns:
        Dict with:
            - center_hit_rate: fraction of positive slices where top-1 peak
              falls inside any GT bbox
            - top_k_recall: fraction of GT bboxes that have a peak inside them
            - mean_center_distance: mean L2 between predicted peak and nearest
              GT center (normalized coords, positive slices only)
            - false_positive_rate: fraction of negative slices with any detection
    """
    B = pred_heatmaps.shape[0]

    n_pos = 0
    n_hits = 0
    n_gt_bboxes = 0
    n_gt_recalled = 0
    total_distance = 0.0
    n_neg = 0
    n_false_pos = 0

    for i in range(B):
        ann = annotations[i]
        has_lesion = ann.get("has_lesion", 0)
        bboxes = ann.get("bboxes_normalized", [])
        hm = pred_heatmaps[i]  # (1, H, H)

        peaks = detect_peaks(hm, threshold, nms_kernel, max_detections)

        if has_lesion and bboxes:
            n_pos += 1

            # Center hit rate: top-1 peak inside any GT bbox
            if peaks:
                top_peak = peaks[0]
                px, py = top_peak["x"], top_peak["y"]
                hit = any(
                    b[0] <= px <= b[2] and b[1] <= py <= b[3]
                    for b in bboxes
                )
                if hit:
                    n_hits += 1

                # Mean center distance: nearest GT center
                gt_centers = [
                    ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in bboxes
                ]
                min_dist = min(
                    math.sqrt((px - gcx) ** 2 + (py - gcy) ** 2)
                    for gcx, gcy in gt_centers
                )
                total_distance += min_dist

            # Top-k recall: fraction of GT bboxes with a peak inside
            for bbox in bboxes:
                n_gt_bboxes += 1
                recalled = any(
                    bbox[0] <= p["x"] <= bbox[2] and bbox[1] <= p["y"] <= bbox[3]
                    for p in peaks
                )
                if recalled:
                    n_gt_recalled += 1

        else:
            # Negative slice
            n_neg += 1
            if peaks:
                n_false_pos += 1

    return {
        "center_hit_rate": n_hits / max(n_pos, 1),
        "top_k_recall": n_gt_recalled / max(n_gt_bboxes, 1),
        "mean_center_distance": total_distance / max(n_pos, 1),
        "false_positive_rate": n_false_pos / max(n_neg, 1),
        "n_positive_slices": n_pos,
        "n_negative_slices": n_neg,
    }


# ============================================================================
# Volume-aware train/val split
# ============================================================================


def scan_volume_ids(shard_dir: str, datasets: list = None) -> dict:
    """Scan shard JSONs and collect unique volume IDs with metadata.

    Args:
        shard_dir: Root directory containing <dataset_name>/shard-*.tar.
        datasets: Dataset names to include (None = all).

    Returns:
        dict: {sample_id: {"dataset": str, "has_lesion": int, "n_slices": int}}
    """
    volumes = {}
    for entry in sorted(os.listdir(shard_dir)):
        subdir = os.path.join(shard_dir, entry)
        if not os.path.isdir(subdir):
            continue
        if datasets is not None and entry not in datasets:
            continue
        shard_files = sorted(glob.glob(os.path.join(subdir, "shard-*.tar")))
        if not shard_files:
            continue
        logger.info(f"  Scanning {entry}: {len(shard_files)} shards ...")

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
                            "dataset": meta.get("dataset", entry),
                            "has_lesion": 0,
                            "n_slices": 0,
                        }

                    volumes[sid]["has_lesion"] = max(
                        volumes[sid]["has_lesion"], int(meta.get("has_lesion", 0))
                    )
                    volumes[sid]["n_slices"] += 1

    for ds in (datasets or sorted(set(v["dataset"] for v in volumes.values()))):
        ds_vols = [v for v in volumes.values() if v["dataset"] == ds]
        n_pos = sum(v["has_lesion"] for v in ds_vols)
        logger.info(
            f"  {ds}: {len(ds_vols)} volumes "
            f"({n_pos} positive, {len(ds_vols) - n_pos} negative)"
        )
    logger.info(f"Total: {len(volumes)} volumes")
    return volumes


def volume_train_val_split(volume_info: dict, val_ratio: float = 0.15, seed: int = 42):
    """Stratified train/val split by dataset.

    Deterministic seeding ensures all DDP ranks get the same split.

    Args:
        volume_info: Output of scan_volume_ids().
        val_ratio: Fraction of volumes to hold out for validation.
        seed: Random seed for reproducibility.

    Returns:
        (train_ids, val_ids) — sets of sample_id strings.
    """
    rng = random.Random(seed)
    by_dataset = defaultdict(list)
    for vid, info in volume_info.items():
        by_dataset[info["dataset"]].append(vid)

    train_ids, val_ids = [], []
    for ds_name in sorted(by_dataset.keys()):
        vids = sorted(by_dataset[ds_name])
        rng.shuffle(vids)
        n_val = max(1, int(len(vids) * val_ratio))
        val_ids.extend(vids[:n_val])
        train_ids.extend(vids[n_val:])
        logger.info(f"  Split {ds_name}: {len(vids) - n_val} train, {n_val} val")

    return set(train_ids), set(val_ids)


# ============================================================================
# WebDataset pipeline for heatmap training
# ============================================================================


def _discover_shards_balanced(shard_dir: str, datasets: list = None) -> list:
    """Find shards with replication for dataset balance.

    Replicates smaller datasets' shard lists to match the largest dataset.
    """
    by_dataset = {}
    for entry in sorted(os.listdir(shard_dir)):
        subdir = os.path.join(shard_dir, entry)
        if not os.path.isdir(subdir):
            continue
        if datasets is not None and entry not in datasets:
            continue
        found = sorted(glob.glob(os.path.join(subdir, "shard-*.tar")))
        if found:
            by_dataset[entry] = found
            logger.info(f"  {entry}: {len(found)} shards")

    if not by_dataset:
        return []

    max_shards = max(len(v) for v in by_dataset.values())
    balanced = []
    for name, shards in sorted(by_dataset.items()):
        n_reps = max(1, math.ceil(max_shards / len(shards)))
        replicated = (shards * n_reps)[:max_shards]
        balanced.extend(replicated)
        logger.info(f"  {name}: replicated {len(shards)} -> {len(replicated)} shards")

    return balanced


def build_heatmap_wds_pipeline(
    shard_dir: str,
    datasets: list = None,
    heatmap_size: int = 64,
    img_size: int = 224,
    batch_size: int = 24,
    num_workers: int = 8,
    shuffle_buffer: int = 5000,
    epoch_length: int = 1000,
    oversample_positive: float = 3.0,
    balance_datasets: bool = True,
    min_sigma: float = 1.5,
    volume_ids: set = None,
):
    """Build a WebDataset pipeline for heatmap localization training.

    Args:
        shard_dir: Root directory containing <dataset_name>/shard-*.tar.
        datasets: List of dataset names to include (None = all).
        heatmap_size: Heatmap output size (default 64).
        img_size: Input image resize (default 224).
        batch_size: Batch size per GPU.
        num_workers: DataLoader workers.
        shuffle_buffer: Samples to buffer for shuffling.
        epoch_length: Number of batches per epoch.
        oversample_positive: Positive enrichment factor (e.g. 3.0 → keep 1/3 negatives).
        balance_datasets: Replicate smaller datasets to match largest.
        min_sigma: Minimum Gaussian sigma in heatmap pixels.
        volume_ids: If provided, only include slices whose sample_id is in this set.
            Used for volume-aware train/val splitting.

    Returns:
        (dataset, loader) tuple.
    """
    if balance_datasets:
        shards = _discover_shards_balanced(shard_dir, datasets)
    else:
        shards = []
        for entry in sorted(os.listdir(shard_dir)):
            subdir = os.path.join(shard_dir, entry)
            if not os.path.isdir(subdir):
                continue
            if datasets is not None and entry not in datasets:
                continue
            found = sorted(glob.glob(os.path.join(subdir, "shard-*.tar")))
            shards.extend(found)
            if found:
                logger.info(f"  {entry}: {len(found)} shards")

    if not shards:
        raise FileNotFoundError(f"No shard files found under {shard_dir}")

    if num_workers > len(shards):
        logger.info(f"Reducing num_workers {num_workers} -> {len(shards)}")
        num_workers = len(shards)

    logger.info(f"Heatmap WDS: {len(shards)} shards from {shard_dir}")

    detection_transform = transforms.Compose([
        transforms.Resize((img_size, img_size),
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    def decode_heatmap(sample):
        img = sample["png"]
        if img.mode != "RGB":
            img = img.convert("RGB")

        ann = sample["json"]
        if isinstance(ann, bytes):
            ann = json.loads(ann.decode())
        elif isinstance(ann, str):
            ann = json.loads(ann)

        tensor = detection_transform(img)
        bboxes = ann.get("bboxes_normalized", [])
        heatmap = generate_gaussian_heatmap(bboxes, heatmap_size, min_sigma)

        return tensor, heatmap, ann

    pipeline = (
        wds.WebDataset(shards, shardshuffle=True, nodesplitter=wds.split_by_node,
                       empty_check=False)
        .shuffle(shuffle_buffer)
        .decode("pil")
    )

    # Volume-aware filtering: only keep slices belonging to allowed volumes
    if volume_ids is not None:
        _vol_ids = volume_ids  # capture in closure

        def volume_filter(sample):
            ann = sample.get("json", b"{}")
            if isinstance(ann, bytes):
                ann = json.loads(ann.decode())
            elif isinstance(ann, str):
                ann = json.loads(ann)
            return ann.get("sample_id", "") in _vol_ids

        pipeline = pipeline.select(volume_filter)
        logger.info(f"Heatmap WDS: filtering to {len(volume_ids)} volumes")

    # Balanced sampling: keep all positives, subsample negatives
    if oversample_positive > 1.0:
        keep_prob = 1.0 / oversample_positive

        def balanced_filter(sample):
            ann = sample.get("json", b"{}")
            if isinstance(ann, bytes):
                ann = json.loads(ann.decode())
            elif isinstance(ann, str):
                ann = json.loads(ann)
            if ann.get("has_lesion", 0) == 1:
                return True
            return random.random() < keep_prob

        pipeline = pipeline.select(balanced_filter)

    dataset = pipeline.map(decode_heatmap)

    def collate_heatmap(samples):
        images = torch.stack([s[0] for s in samples])
        heatmaps = torch.stack([s[1] for s in samples])
        annotations = [s[2] for s in samples]
        return images, heatmaps, annotations

    loader = wds.WebLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        collate_fn=collate_heatmap,
    )

    loader = loader.repeat(2).slice(epoch_length)

    logger.info(
        f"Heatmap WDS: batch_size={batch_size}, "
        f"oversample_positive={oversample_positive}, "
        f"epoch={epoch_length} batches"
    )
    return dataset, loader
