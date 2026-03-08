"""
Patch-level lesion detection using frozen DINOv2 patch tokens.

Architecture:
    Input (B, 3, 224, 224)
    → Frozen DINO backbone → patch tokens (B, 256, 768)
    → MLP head → per-patch logits (B, 256)
    → Reshape → patch score map (B, 1, 16, 16)

Components:
    - PatchDetector: frozen backbone + MLP classification head
    - assign_three_region_patch_labels: shrunk/ignore/negative labeling from bboxes
    - PatchFocalLoss: focal loss with ignore masking + hard negative mining
    - build_patch_detection_pipeline: WebDataset loader with balanced sampling
    - patches_to_regions: connected component analysis for region proposals
    - compute_patch_metrics: patch-level accuracy/precision/recall/F1/AUROC
    - compute_region_metrics: region-level recall/precision/F1 against GT bboxes
    - visualize_patch_heatmap: three-panel visualization
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
# Three-region patch labeling
# ============================================================================


def assign_three_region_patch_labels(
    bboxes: list,
    img_width: int,
    img_height: int,
    grid_h: int = 16,
    grid_w: int = 16,
    patch_size: int = 14,
    img_size: int = 224,
    shrink_ratio: float = 0.7,
    expand_ratio: float = 1.3,
    pos_iou_thresh: float = 0.5,
    ignore_iou_thresh: float = 0.1,
) -> torch.Tensor:
    """Compute per-patch labels using three-region overlap-based assignment.

    For each bbox, computes a shrunk core (positive), expanded ring (ignore),
    and everything else (negative). Overlap ratio between patch footprint and
    bbox regions determines assignment.

    Args:
        bboxes: List of [x1, y1, x2, y2] in original pixel coordinates.
        img_width: Original image width.
        img_height: Original image height.
        grid_h: Patch grid height (16 for 224/14).
        grid_w: Patch grid width.
        patch_size: Patch size in pixels (14 for ViT-B/14).
        img_size: Resized image size (224).
        shrink_ratio: Scale factor for high-confidence core box (0.7 = 70%).
        expand_ratio: Scale factor for ignore zone boundary (1.3 = 130%).
        pos_iou_thresh: Overlap ratio with shrunk box for positive (0.5).
        ignore_iou_thresh: Overlap ratio with expanded box for ignore (0.1).

    Returns:
        (grid_h * grid_w,) tensor with values:
            1.0 = positive, 0.0 = negative, -1.0 = ignore
    """
    n_patches = grid_h * grid_w
    labels = torch.zeros(n_patches, dtype=torch.float32)

    if not bboxes:
        return labels

    # Pre-compute all patch footprints in resized image space
    patches = torch.zeros(n_patches, 4, dtype=torch.float32)
    for r in range(grid_h):
        for c in range(grid_w):
            idx = r * grid_w + c
            patches[idx, 0] = c * patch_size       # x1
            patches[idx, 1] = r * patch_size       # y1
            patches[idx, 2] = (c + 1) * patch_size  # x2
            patches[idx, 3] = (r + 1) * patch_size  # y2

    patch_area = float(patch_size * patch_size)

    # Scale factors from original image to resized space
    sx = img_size / img_width
    sy = img_size / img_height

    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        # Transform to resized coordinates
        rx1, ry1, rx2, ry2 = x1 * sx, y1 * sy, x2 * sx, y2 * sy

        # Box center and dimensions in resized space
        cx = (rx1 + rx2) / 2.0
        cy = (ry1 + ry2) / 2.0
        w = rx2 - rx1
        h = ry2 - ry1

        if w <= 0 or h <= 0:
            continue

        # Shrunk box (positive core)
        sw, sh = w * shrink_ratio, h * shrink_ratio
        shrunk = torch.tensor([cx - sw / 2, cy - sh / 2, cx + sw / 2, cy + sh / 2])

        # Expanded box (ignore boundary)
        ew, eh = w * expand_ratio, h * expand_ratio
        expanded = torch.tensor([cx - ew / 2, cy - eh / 2, cx + ew / 2, cy + eh / 2])

        # Compute overlap with shrunk box (vectorized)
        inter_x1 = torch.max(patches[:, 0], shrunk[0])
        inter_y1 = torch.max(patches[:, 1], shrunk[1])
        inter_x2 = torch.min(patches[:, 2], shrunk[2])
        inter_y2 = torch.min(patches[:, 3], shrunk[3])
        inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
        overlap_shrunk = inter_area / patch_area  # (n_patches,)

        # Compute overlap with expanded box (vectorized)
        inter_x1 = torch.max(patches[:, 0], expanded[0])
        inter_y1 = torch.max(patches[:, 1], expanded[1])
        inter_x2 = torch.min(patches[:, 2], expanded[2])
        inter_y2 = torch.min(patches[:, 3], expanded[3])
        inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
        overlap_expanded = inter_area / patch_area  # (n_patches,)

        # Assign labels: positive > ignore > negative (max across bboxes)
        for i in range(n_patches):
            if overlap_shrunk[i] >= pos_iou_thresh:
                labels[i] = 1.0  # positive always wins
            elif overlap_expanded[i] >= ignore_iou_thresh and labels[i] == 0.0:
                labels[i] = -1.0  # ignore (don't overwrite existing positive)

        # Fallback for tiny lesions: if no patch is positive, assign closest patch
        if (labels == 1.0).sum() == 0 and w > 0 and h > 0:
            patch_centers_x = (patches[:, 0] + patches[:, 2]) / 2
            patch_centers_y = (patches[:, 1] + patches[:, 3]) / 2
            dist = (patch_centers_x - cx) ** 2 + (patch_centers_y - cy) ** 2
            closest_idx = dist.argmin().item()
            labels[closest_idx] = 1.0

    return labels


# ============================================================================
# Focal loss with ignore masking and hard negative mining
# ============================================================================


class PatchFocalLoss(nn.Module):
    """Binary focal loss for patch-level predictions with ignore masking
    and within-slice hard negative mining.

    Args:
        alpha: Weighting for positive class (default 0.75).
            Higher alpha gives more weight to positives.
        gamma: Focusing parameter (default 2.0).
        neg_subsample_ratio: Max negatives per positive patch per slice (default 3.0).
            Set 0 to disable subsampling.
        neg_patches_per_neg_slice: Max negative patches to keep per all-negative
            slice (default 10). Prevents easy negatives from dominating.
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0,
                 neg_subsample_ratio: float = 3.0,
                 neg_patches_per_neg_slice: int = 10):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.neg_subsample_ratio = neg_subsample_ratio
        self.neg_patches_per_neg_slice = neg_patches_per_neg_slice

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, N_patches) raw logits.
            labels: (B, N_patches) with 1.0 (positive), 0.0 (negative), -1.0 (ignore).

        Returns:
            Scalar loss.
        """
        # Work in float32 for numerical stability
        logits = logits.float()
        labels = labels.float()

        # Ignore mask
        valid_mask = (labels >= 0)  # True for positive and negative

        # Compute per-element focal loss
        targets = labels.clamp(min=0)  # map -1 → 0 for BCE computation
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        p = torch.sigmoid(logits)
        pt = targets * p + (1 - targets) * (1 - p)
        focal_weight = (1 - pt) ** self.gamma
        alpha_weight = targets * self.alpha + (1 - targets) * (1 - self.alpha)

        per_element_loss = alpha_weight * focal_weight * bce  # (B, N)

        # Apply ignore mask
        per_element_loss = per_element_loss * valid_mask

        # Hard negative mining: per-slice subsampling
        if self.neg_subsample_ratio > 0:
            subsample_mask = torch.ones_like(labels, dtype=torch.bool)
            B = logits.shape[0]
            for i in range(B):
                pos_mask_i = (labels[i] == 1.0)
                neg_mask_i = (labels[i] == 0.0)
                n_pos = pos_mask_i.sum().item()
                n_neg = neg_mask_i.sum().item()

                if n_neg == 0:
                    continue

                if n_pos > 0:
                    # Positive slice: keep n_pos * ratio hard negatives
                    n_neg_keep = max(1, int(n_pos * self.neg_subsample_ratio))
                else:
                    # Negative slice: keep only a small fixed number of
                    # hard negatives to prevent easy negatives dominating
                    n_neg_keep = self.neg_patches_per_neg_slice

                if n_neg > n_neg_keep:
                    # Keep top-loss negatives (hard negative mining)
                    neg_idx = torch.where(neg_mask_i)[0]
                    neg_losses_only = per_element_loss[i, neg_idx]
                    _, topk_order = neg_losses_only.sort(descending=True)
                    # Drop negatives beyond the top n_neg_keep
                    drop_idx = neg_idx[topk_order[n_neg_keep:]]
                    subsample_mask[i, drop_idx] = False

            per_element_loss = per_element_loss * subsample_mask

        # Normalize
        n_valid = per_element_loss.gt(0).sum().clamp(min=1)
        return per_element_loss.sum() / n_valid


# ============================================================================
# Model
# ============================================================================


class PatchDetector(nn.Module):
    """Frozen DINO backbone + MLP head for patch-level lesion detection.

    Produces a per-patch lesion score map (16×16 for ViT-B/14 at 224×224).

    Args:
        backbone_checkpoint: Path to pretrained DINO weights or hub model name.
        arch: Backbone architecture (default "vit_base").
        patch_size: Patch size (default 14).
        img_size: Input image size (default 224).
        head_hidden_dim: MLP hidden dimension (default 256).
        unfreeze_last_n: Number of backbone blocks to unfreeze (default 0).
    """

    def __init__(
        self,
        backbone_checkpoint: str,
        arch: str = "vit_base",
        patch_size: int = 14,
        img_size: int = 224,
        head_hidden_dim: int = 256,
        unfreeze_last_n: int = 0,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.img_size = img_size
        self.grid_h = img_size // patch_size
        self.grid_w = img_size // patch_size
        self.n_patches = self.grid_h * self.grid_w

        # Build and freeze backbone
        self.backbone, embed_dim = build_backbone(
            model_name=arch,
            patch_size=patch_size,
            pretrained=backbone_checkpoint,
            img_size=img_size,
            drop_path_rate=0.0,
        )
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

        # Optional: unfreeze last N blocks
        if unfreeze_last_n > 0:
            self._unfreeze_blocks(unfreeze_last_n)

        # MLP head: embed_dim → hidden → 1
        self.head = nn.Sequential(
            nn.Linear(embed_dim, head_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(head_hidden_dim, 1),
        )

        n_head_params = sum(p.numel() for p in self.head.parameters())
        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            f"PatchDetector: {arch}, head={n_head_params} params, "
            f"trainable={n_trainable}, grid={self.grid_h}x{self.grid_w}"
        )

    def _unfreeze_blocks(self, n_blocks):
        """Unfreeze last N transformer blocks for fine-tuning."""
        for block in self.backbone.blocks[-n_blocks:]:
            for p in block.parameters():
                p.requires_grad = True
        logger.info(f"Unfroze last {n_blocks} backbone blocks")

    def train(self, mode=True):
        """Keep backbone in eval mode (BatchNorm, Dropout frozen)."""
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, images):
        """Forward pass.

        Args:
            images: (B, 3, img_size, img_size) tensor.

        Returns:
            patch_logits: (B, N_patches) per-patch logits.
            patch_map: (B, 1, grid_h, grid_w) spatial logit map.
        """
        with torch.no_grad():
            out = self.backbone(images, is_training=True)
            patch_tokens = out["x_norm_patchtokens"]  # (B, N_patches, embed_dim)

        logits = self.head(patch_tokens).squeeze(-1)  # (B, N_patches)
        B = logits.shape[0]
        patch_map = logits.view(B, 1, self.grid_h, self.grid_w)
        return logits, patch_map

    def get_trainable_params(self):
        """Return parameters that require gradients."""
        return [p for p in self.parameters() if p.requires_grad]

    def unfreeze_backbone_blocks(self, n_blocks):
        """Called mid-training to enable fine-tuning of last N blocks."""
        self._unfreeze_blocks(n_blocks)


# ============================================================================
# Volume-aware data scanning and splitting
# ============================================================================


def scan_volume_ids(shard_dir: str, datasets: list = None) -> dict:
    """Scan shard JSONs and collect volume metadata.

    Returns:
        {sample_id: {"dataset": str, "has_lesion": int, "n_slices": int,
                      "n_positive_slices": int}}
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
                            "n_positive_slices": 0,
                        }

                    has_les = int(meta.get("has_lesion", 0))
                    volumes[sid]["has_lesion"] = max(volumes[sid]["has_lesion"], has_les)
                    volumes[sid]["n_slices"] += 1
                    if has_les:
                        volumes[sid]["n_positive_slices"] += 1

    for ds in sorted(set(v["dataset"] for v in volumes.values())):
        ds_vols = [v for v in volumes.values() if v["dataset"] == ds]
        n_pos = sum(v["has_lesion"] for v in ds_vols)
        n_slices = sum(v["n_slices"] for v in ds_vols)
        n_pos_slices = sum(v["n_positive_slices"] for v in ds_vols)
        logger.info(
            f"  {ds}: {len(ds_vols)} volumes ({n_pos} pos), "
            f"{n_slices} slices ({n_pos_slices} pos)"
        )
    logger.info(f"Total: {len(volumes)} volumes")
    return volumes


def volume_three_way_split(
    volume_info: dict,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
):
    """Stratified three-way split by dataset.

    Returns:
        (train_ids, val_ids, test_ids) — sets of sample_id strings.
    """
    rng = random.Random(seed)
    by_dataset = defaultdict(list)
    for vid, info in volume_info.items():
        by_dataset[info["dataset"]].append(vid)

    train_ids, val_ids, test_ids = [], [], []
    for ds_name in sorted(by_dataset.keys()):
        vids = sorted(by_dataset[ds_name])
        rng.shuffle(vids)
        n_test = max(1, int(len(vids) * test_ratio))
        n_val = max(1, int(len(vids) * val_ratio))
        test_ids.extend(vids[:n_test])
        val_ids.extend(vids[n_test:n_test + n_val])
        train_ids.extend(vids[n_test + n_val:])
        logger.info(
            f"  Split {ds_name}: {len(vids) - n_test - n_val} train, "
            f"{n_val} val, {n_test} test"
        )

    return set(train_ids), set(val_ids), set(test_ids)


# ============================================================================
# WebDataset pipeline
# ============================================================================


def _discover_shards_balanced(shard_dir: str, datasets: list = None) -> list:
    """Find shards with replication for dataset balance."""
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


def build_patch_detection_pipeline(
    shard_dir: str,
    datasets: list = None,
    img_size: int = 224,
    batch_size: int = 24,
    num_workers: int = 8,
    shuffle_buffer: int = 5000,
    epoch_length: int = 1000,
    oversample_positive: float = 3.0,
    balance_datasets: bool = True,
    volume_ids: set = None,
):
    """Build a WebDataset pipeline for patch detection training.

    Returns:
        (dataset, loader) tuple. Loader yields (images, annotations) where
        images is (B, 3, img_size, img_size) and annotations is a list of dicts.
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

    if not shards:
        raise FileNotFoundError(f"No shard files found under {shard_dir}")

    if num_workers > len(shards):
        logger.info(f"Reducing num_workers {num_workers} -> {len(shards)}")
        num_workers = len(shards)

    logger.info(f"Patch detection WDS: {len(shards)} shards from {shard_dir}")

    detection_transform = transforms.Compose([
        transforms.Resize(
            (img_size, img_size),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    def decode_sample(sample):
        img = sample["png"]
        if img.mode != "RGB":
            img = img.convert("RGB")

        ann = sample["json"]
        if isinstance(ann, bytes):
            ann = json.loads(ann.decode())
        elif isinstance(ann, str):
            ann = json.loads(ann)

        tensor = detection_transform(img)
        return tensor, ann

    pipeline = (
        wds.WebDataset(shards, shardshuffle=True, nodesplitter=wds.split_by_node,
                       empty_check=False)
        .shuffle(shuffle_buffer)
        .decode("pil")
    )

    # Volume-aware filtering
    if volume_ids is not None:
        _vol_ids = volume_ids

        def volume_filter(sample):
            ann = sample.get("json", b"{}")
            if isinstance(ann, bytes):
                ann = json.loads(ann.decode())
            elif isinstance(ann, str):
                ann = json.loads(ann)
            return ann.get("sample_id", "") in _vol_ids

        pipeline = pipeline.select(volume_filter)
        logger.info(f"Patch detection WDS: filtering to {len(volume_ids)} volumes")

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

    dataset = pipeline.map(decode_sample)

    def collate_fn(samples):
        images = torch.stack([s[0] for s in samples])
        annotations = [s[1] for s in samples]
        return images, annotations

    loader = wds.WebLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        collate_fn=collate_fn,
    )

    loader = loader.repeat(2).slice(epoch_length)

    logger.info(
        f"Patch detection WDS: batch_size={batch_size}, "
        f"oversample_positive={oversample_positive}, "
        f"epoch={epoch_length} batches"
    )
    return dataset, loader


# ============================================================================
# Region extraction (post-processing)
# ============================================================================


def patches_to_regions(
    patch_scores: torch.Tensor,
    threshold: float = 0.5,
    min_area_patches: int = 2,
    max_regions: int = 5,
    grid_h: int = 16,
    grid_w: int = 16,
) -> list:
    """Convert patch score map to candidate bounding box regions.

    Steps:
        1. Threshold scores to get binary map
        2. Connected component analysis
        3. Filter small components
        4. Sort by mean score, take top-K
        5. Convert to normalized coordinates

    Args:
        patch_scores: (grid_h, grid_w) sigmoid scores in [0, 1].
        threshold: Score threshold for binary map.
        min_area_patches: Minimum component area in patches.
        max_regions: Maximum number of regions to return.
        grid_h: Patch grid height.
        grid_w: Patch grid width.

    Returns:
        List of dicts: [{"bbox_normalized": [nx1,ny1,nx2,ny2],
                         "center_normalized": [cx, cy],
                         "mean_score": float, "max_score": float,
                         "area_patches": int}, ...]
    """
    scores_np = patch_scores.detach().cpu().numpy()
    binary = (scores_np >= threshold).astype(np.int32)

    if binary.sum() == 0:
        return []

    # Connected component analysis
    try:
        from scipy.ndimage import label as scipy_label
        labeled, n_components = scipy_label(binary)
    except ImportError:
        # Fallback: simple BFS-based connected components
        labeled, n_components = _bfs_connected_components(binary)

    regions = []
    for comp_id in range(1, n_components + 1):
        mask = (labeled == comp_id)
        area = mask.sum()
        if area < min_area_patches:
            continue

        ys, xs = np.where(mask)
        r_min, r_max = ys.min(), ys.max()
        c_min, c_max = xs.min(), xs.max()

        # Normalized bbox coordinates
        nx1 = c_min / grid_w
        ny1 = r_min / grid_h
        nx2 = (c_max + 1) / grid_w
        ny2 = (r_max + 1) / grid_h

        # Center
        cx = (nx1 + nx2) / 2
        cy = (ny1 + ny2) / 2

        mean_score = float(scores_np[mask].mean())
        max_score = float(scores_np[mask].max())

        regions.append({
            "bbox_normalized": [nx1, ny1, nx2, ny2],
            "center_normalized": [cx, cy],
            "mean_score": mean_score,
            "max_score": max_score,
            "area_patches": int(area),
        })

    # Sort by mean score descending, take top-K
    regions.sort(key=lambda r: r["mean_score"], reverse=True)
    return regions[:max_regions]


def _bfs_connected_components(binary: np.ndarray) -> tuple:
    """Simple BFS-based connected components (fallback if scipy unavailable)."""
    H, W = binary.shape
    labeled = np.zeros_like(binary, dtype=np.int32)
    comp_id = 0

    for r in range(H):
        for c in range(W):
            if binary[r, c] == 1 and labeled[r, c] == 0:
                comp_id += 1
                queue = [(r, c)]
                labeled[r, c] = comp_id
                while queue:
                    cr, cc = queue.pop(0)
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < H and 0 <= nc < W:
                            if binary[nr, nc] == 1 and labeled[nr, nc] == 0:
                                labeled[nr, nc] = comp_id
                                queue.append((nr, nc))

    return labeled, comp_id


# ============================================================================
# Metrics
# ============================================================================


def compute_patch_metrics(all_logits: list, all_labels: list) -> dict:
    """Compute patch-level classification metrics.

    Filters out ignore labels (-1) before computing metrics.

    Args:
        all_logits: List of (N_patches,) tensors (raw logits).
        all_labels: List of (N_patches,) tensors with 1.0/0.0/-1.0.

    Returns:
        Dict with accuracy, precision, recall, f1, auroc, and counts.
    """
    if not all_logits:
        return {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0, "auroc": 0}

    logits_cat = torch.cat(all_logits)
    labels_cat = torch.cat(all_labels)

    # Filter out ignore
    valid = labels_cat >= 0
    logits_valid = logits_cat[valid]
    labels_valid = labels_cat[valid].long()

    if len(logits_valid) == 0:
        return {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0, "auroc": 0}

    probs = torch.sigmoid(logits_valid).numpy()
    targets = labels_valid.numpy()
    preds = (probs >= 0.5).astype(int)

    n_pos = (targets == 1).sum()
    n_neg = (targets == 0).sum()

    tp = ((preds == 1) & (targets == 1)).sum()
    fp = ((preds == 1) & (targets == 0)).sum()
    fn = ((preds == 0) & (targets == 1)).sum()

    accuracy = float((preds == targets).mean())
    precision = float(tp / max(tp + fp, 1))
    recall = float(tp / max(tp + fn, 1))
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    # AUROC
    auroc = 0.0
    if n_pos > 0 and n_neg > 0:
        try:
            from sklearn.metrics import roc_auc_score
            auroc = float(roc_auc_score(targets, probs))
        except Exception:
            auroc = _manual_auroc(targets, probs)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auroc": auroc,
        "n_total_patches": len(targets),
        "n_positive_patches": int(n_pos),
        "n_negative_patches": int(n_neg),
    }


def _manual_auroc(targets, probs):
    """Simple Mann-Whitney U approximation for AUROC."""
    pos_probs = probs[targets == 1]
    neg_probs = probs[targets == 0]
    if len(pos_probs) == 0 or len(neg_probs) == 0:
        return 0.0
    n_correct = 0
    n_total = len(pos_probs) * len(neg_probs)
    for pp in pos_probs:
        n_correct += (pp > neg_probs).sum() + 0.5 * (pp == neg_probs).sum()
    return float(n_correct / n_total)


def compute_region_metrics(
    pred_regions_list: list,
    annotations_list: list,
    iou_threshold: float = 0.3,
) -> dict:
    """Compute region-level detection metrics.

    For each GT bbox, check if any predicted region has IoU >= threshold.

    Args:
        pred_regions_list: List of per-slice region lists (from patches_to_regions).
        annotations_list: List of annotation dicts with bboxes_normalized.
        iou_threshold: IoU threshold for matching.

    Returns:
        Dict with region_recall, region_precision, region_f1,
        false_positive_rate, n_gt_bboxes, n_pred_regions.
    """
    n_gt_total = 0
    n_gt_matched = 0
    n_pred_total = 0
    n_pred_matched = 0
    n_neg_slices = 0
    n_neg_with_fp = 0

    for pred_regions, ann in zip(pred_regions_list, annotations_list):
        gt_bboxes = ann.get("bboxes_normalized", [])
        has_lesion = ann.get("has_lesion", 0)

        if not has_lesion:
            n_neg_slices += 1
            if len(pred_regions) > 0:
                n_neg_with_fp += 1
            continue

        n_gt_total += len(gt_bboxes)
        n_pred_total += len(pred_regions)

        # Match GT to predictions
        gt_matched = set()
        pred_matched = set()
        for gi, gt_box in enumerate(gt_bboxes):
            for pi, pred in enumerate(pred_regions):
                iou = _compute_iou(gt_box, pred["bbox_normalized"])
                if iou >= iou_threshold:
                    gt_matched.add(gi)
                    pred_matched.add(pi)

        n_gt_matched += len(gt_matched)
        n_pred_matched += len(pred_matched)

    recall = n_gt_matched / max(n_gt_total, 1)
    precision = n_pred_matched / max(n_pred_total, 1)
    f1 = 2 * recall * precision / max(recall + precision, 1e-8)
    fpr = n_neg_with_fp / max(n_neg_slices, 1)

    return {
        "region_recall": recall,
        "region_precision": precision,
        "region_f1": f1,
        "false_positive_rate": fpr,
        "n_gt_bboxes": n_gt_total,
        "n_pred_regions": n_pred_total,
        "n_gt_matched": n_gt_matched,
        "n_negative_slices": n_neg_slices,
    }


def _compute_iou(box_a, box_b):
    """Compute IoU between two [x1, y1, x2, y2] boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter

    return inter / max(union, 1e-8)


# ============================================================================
# Visualization
# ============================================================================


def visualize_patch_heatmap(
    image: torch.Tensor,
    patch_scores: torch.Tensor,
    patch_labels: torch.Tensor,
    gt_bboxes_normalized: list,
    pred_regions: list,
    save_path: str,
    img_size: int = 224,
):
    """Generate a three-panel visualization.

    Left: original image with GT bboxes.
    Center: predicted patch score heatmap (upsampled).
    Right: ground-truth patch labels (green=pos, yellow=ignore, red=neg).

    Args:
        image: (3, H, W) normalized tensor.
        patch_scores: (grid_h, grid_w) sigmoid scores.
        patch_labels: (N_patches,) with 1/0/-1 labels.
        gt_bboxes_normalized: List of [nx1,ny1,nx2,ny2] in [0,1].
        pred_regions: Output of patches_to_regions.
        save_path: Path to save the figure.
        img_size: Image size for denormalization.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    # Denormalize image
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_vis = (image.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()

    grid_h = patch_scores.shape[0]
    grid_w = patch_scores.shape[1]
    scores_np = patch_scores.detach().cpu().numpy()
    labels_np = patch_labels.cpu().numpy().reshape(grid_h, grid_w)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Left: original image with GT bboxes
    axes[0].imshow(img_vis)
    axes[0].set_title("Image + GT Bboxes")
    for bbox in gt_bboxes_normalized:
        nx1, ny1, nx2, ny2 = bbox
        rect = mpatches.Rectangle(
            (nx1 * img_size, ny1 * img_size),
            (nx2 - nx1) * img_size, (ny2 - ny1) * img_size,
            linewidth=2, edgecolor="lime", facecolor="none",
        )
        axes[0].add_patch(rect)
    # Overlay predicted region bboxes
    for region in pred_regions:
        rb = region["bbox_normalized"]
        rect = mpatches.Rectangle(
            (rb[0] * img_size, rb[1] * img_size),
            (rb[2] - rb[0]) * img_size, (rb[3] - rb[1]) * img_size,
            linewidth=2, edgecolor="cyan", facecolor="none", linestyle="--",
        )
        axes[0].add_patch(rect)
    axes[0].axis("off")

    # Center: predicted heatmap
    im = axes[1].imshow(scores_np, cmap="hot", vmin=0, vmax=1,
                        interpolation="bilinear")
    axes[1].set_title("Predicted Heatmap")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046)

    # Right: ground-truth labels
    label_vis = np.zeros((grid_h, grid_w, 3))
    label_vis[labels_np == 1.0] = [0, 1, 0]    # green = positive
    label_vis[labels_np == -1.0] = [1, 1, 0]   # yellow = ignore
    label_vis[labels_np == 0.0] = [0.8, 0.2, 0.2]  # red = negative
    axes[2].imshow(label_vis, interpolation="nearest")
    axes[2].set_title("GT Labels (G=pos, Y=ign, R=neg)")
    axes[2].axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
