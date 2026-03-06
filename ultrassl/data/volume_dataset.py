"""
Volume-level dataset utilities for MIL classification.

Provides two data loading modes:
  - Cached: Pre-extracted CLS token embeddings loaded from disk (fast training).
  - On-the-fly: Raw images loaded from WebDataset shards, forwarded through
    frozen backbone during training (no disk cache needed).

Also provides:
  - scan_volume_index(): Scan shards to build per-volume metadata index.
  - load_volume_split_extended(): Stratified train/val split with filtering.
  - extract_and_cache_embeddings(): One-time embedding extraction to disk.
  - collate_volumes(): Collation function for variable-length volumes.
"""

import glob
import io
import json
import logging
import os
import pickle
import random
import tarfile
from collections import defaultdict

import torch
import torch.distributed as dist
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

logger = logging.getLogger("ultrassl")

# Default class mapping: dataset folder name -> class index
CLASS_MAPPING = {"Class2": 0, "Class3": 1, "Class4": 2}


# ── Volume Index Scanner ─────────────────────────────────────────────

def scan_volume_index(shard_dir, datasets=None):
    """Scan all shards and build a per-volume metadata index.

    Reads JSON annotations from all shard .tar files, groups slices by
    sample_id (volume ID), and computes volume-level statistics.

    Args:
        shard_dir: Root directory containing <dataset_name>/shard-*.tar subdirs.
        datasets: Optional list of dataset names to include (None = all).

    Returns:
        dict: {sample_id: {
            "dataset": str,
            "has_lesion": int (1 if any slice positive),
            "n_slices": int,
            "n_positive_slices": int,
            "slice_keys": [(shard_path, key_prefix, slice_idx), ...],
        }}
    """
    volume_index = {}
    shard_files = sorted(glob.glob(os.path.join(shard_dir, "*/shard-*.tar")))

    if not shard_files:
        logger.warning(f"No shard files found under {shard_dir}")
        return volume_index

    for shard_path in shard_files:
        # Filter by dataset name if specified
        ds_name = os.path.basename(os.path.dirname(shard_path))
        if datasets is not None and ds_name not in datasets:
            continue

        try:
            with tarfile.open(shard_path, "r") as tar:
                json_members = [m for m in tar.getmembers() if m.name.endswith(".json")]
                for member in json_members:
                    f = tar.extractfile(member)
                    if f is None:
                        continue
                    ann = json.loads(f.read().decode())
                    sid = ann.get("sample_id", "")
                    has_lesion = ann.get("has_lesion", 0)
                    slice_idx = ann.get("slice_idx", 0)
                    key_prefix = member.name.rsplit(".", 1)[0]  # e.g. "00000123"

                    if sid not in volume_index:
                        volume_index[sid] = {
                            "dataset": ann.get("dataset", ds_name),
                            "has_lesion": 0,
                            "n_slices": 0,
                            "n_positive_slices": 0,
                            "slice_keys": [],
                        }

                    info = volume_index[sid]
                    info["n_slices"] += 1
                    info["n_positive_slices"] += has_lesion
                    if has_lesion:
                        info["has_lesion"] = 1
                    info["slice_keys"].append((shard_path, key_prefix, slice_idx))
        except Exception as e:
            logger.warning(f"Error reading {shard_path}: {e}")

    # Sort slice_keys by slice_idx for consistent ordering
    for info in volume_index.values():
        info["slice_keys"].sort(key=lambda x: x[2])

    logger.info(
        f"Scanned {len(shard_files)} shards: {len(volume_index)} volumes, "
        f"{sum(v['n_slices'] for v in volume_index.values())} total slices, "
        f"{sum(v['has_lesion'] for v in volume_index.values())} positive volumes"
    )

    return volume_index


# ── Train/Val Split ──────────────────────────────────────────────────

def load_volume_split_extended(
    volume_index,
    val_split=0.15,
    seed=42,
    rank=0,
    world_size=1,
    filter_positive_only=False,
    filter_datasets=None,
):
    """Split volumes into train/val sets, stratified by dataset.

    Extends the split logic from train_lesion_classifier.py with optional
    filtering for positive-only volumes and specific datasets.

    Args:
        volume_index: Output of scan_volume_index().
        val_split: Fraction of volumes to hold out for validation.
        seed: Random seed for reproducibility.
        rank: Current DDP rank.
        world_size: Total number of DDP processes.
        filter_positive_only: If True, only include volumes with has_lesion=1.
        filter_datasets: Only include volumes from these datasets (None = all).

    Returns:
        (train_volume_ids: set, val_volume_ids: set)
    """
    train_volumes = None
    val_volumes = None

    if rank == 0:
        # Filter volumes
        filtered = {}
        for sid, info in volume_index.items():
            if filter_datasets is not None and info["dataset"] not in filter_datasets:
                continue
            if filter_positive_only and info["has_lesion"] == 0:
                continue
            filtered[sid] = info

        if not filtered:
            logger.warning("No volumes after filtering. Using all data for training.")
            train_volumes = set()
            val_volumes = set()
        else:
            # Group by dataset for stratified split
            by_dataset = defaultdict(list)
            for sid, info in filtered.items():
                by_dataset[info["dataset"]].append(sid)

            rng = random.Random(seed)
            train_volumes = set()
            val_volumes = set()

            for ds_name in sorted(by_dataset.keys()):
                ds_vols = sorted(by_dataset[ds_name])
                rng.shuffle(ds_vols)
                n_val = max(1, int(len(ds_vols) * val_split))
                val_volumes.update(ds_vols[:n_val])
                train_volumes.update(ds_vols[n_val:])

            logger.info(f"Volume split: {len(train_volumes)} train, {len(val_volumes)} val")
            for ds_name in sorted(by_dataset.keys()):
                n_train = len([v for v in train_volumes if filtered[v]["dataset"] == ds_name])
                n_val = len([v for v in val_volumes if filtered[v]["dataset"] == ds_name])
                logger.info(f"  {ds_name}: {n_train} train, {n_val} val")

    # Broadcast to all ranks via gloo
    if world_size > 1:
        data = pickle.dumps((train_volumes, val_volumes))
        size_tensor = torch.tensor([len(data)], dtype=torch.long)
        dist.broadcast(size_tensor, src=0)
        buf_size = size_tensor.item()
        if rank == 0:
            buf = torch.ByteTensor(list(data))
        else:
            buf = torch.zeros(buf_size, dtype=torch.uint8)
        dist.broadcast(buf, src=0)
        if rank != 0:
            train_volumes, val_volumes = pickle.loads(buf.numpy().tobytes())

    return train_volumes, val_volumes


# ── Cached Volume Dataset ────────────────────────────────────────────

class CachedVolumeDataset(Dataset):
    """Load pre-extracted CLS embeddings grouped by volume.

    Each sample is one volume: K slices subsampled/padded to max_slices.

    Args:
        cache_dir: Path to embedding cache (contains embeddings/<sample_id>.pt).
        volume_ids: Set of volume IDs to include (from split).
        max_slices: Max slices per volume. Longer volumes subsampled, shorter padded.
        class_mapping: Dict mapping dataset name to class index.
    """

    def __init__(self, cache_dir, volume_ids, max_slices=32, class_mapping=None):
        self.cache_dir = cache_dir
        self.max_slices = max_slices
        self.class_mapping = class_mapping or CLASS_MAPPING
        self.embed_dir = os.path.join(cache_dir, "embeddings")

        # Filter to volumes that exist in cache
        self.volume_list = []
        for vid in sorted(volume_ids):
            pt_path = os.path.join(self.embed_dir, f"{vid}.pt")
            if os.path.isfile(pt_path):
                self.volume_list.append(vid)
            else:
                logger.debug(f"Skipping {vid}: not found in cache")

        logger.info(
            f"CachedVolumeDataset: {len(self.volume_list)} volumes "
            f"(max_slices={max_slices})"
        )

    def __len__(self):
        return len(self.volume_list)

    def __getitem__(self, idx):
        vid = self.volume_list[idx]
        data = torch.load(
            os.path.join(self.embed_dir, f"{vid}.pt"),
            map_location="cpu", weights_only=True,
        )
        cls_tokens = data["cls_tokens"]  # (N, embed_dim)
        N = cls_tokens.shape[0]

        # Subsample or pad to max_slices
        if N > self.max_slices:
            indices = torch.randperm(N)[: self.max_slices].sort().values
            cls_tokens = cls_tokens[indices]
            mask = torch.ones(self.max_slices, dtype=torch.float32)
        elif N < self.max_slices:
            pad = torch.zeros(self.max_slices - N, cls_tokens.shape[1])
            cls_tokens = torch.cat([cls_tokens, pad], dim=0)
            mask = torch.zeros(self.max_slices, dtype=torch.float32)
            mask[:N] = 1.0
        else:
            mask = torch.ones(self.max_slices, dtype=torch.float32)

        # Build labels
        has_lesion = int(data["has_lesion"])
        dataset_name = data["dataset"]
        class_idx = self.class_mapping.get(dataset_name, -1)

        labels = {
            "has_lesion": has_lesion,
            "class_idx": class_idx,
            "dataset": dataset_name,
            "sample_id": vid,
        }

        return cls_tokens, mask, labels


# ── On-the-fly Volume Dataset ────────────────────────────────────────

class OnTheFlyVolumeDataset(Dataset):
    """Load raw images from shards for on-the-fly feature extraction.

    Returns image tensors that the training script forwards through the
    frozen backbone. Slower than cached mode but needs no disk cache.

    Args:
        volume_index: Output of scan_volume_index().
        shard_dir: Root shard directory.
        volume_ids: Set of volume IDs to include.
        max_slices: Max slices per volume.
        class_mapping: Dict mapping dataset name to class index.
        img_size: Image resize target.
    """

    def __init__(
        self,
        volume_index,
        shard_dir,
        volume_ids,
        max_slices=32,
        class_mapping=None,
        img_size=224,
    ):
        self.volume_index = volume_index
        self.shard_dir = shard_dir
        self.max_slices = max_slices
        self.class_mapping = class_mapping or CLASS_MAPPING
        self.volume_list = sorted([v for v in volume_ids if v in volume_index])

        self.transform = transforms.Compose([
            transforms.Resize(
                (img_size, img_size),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        logger.info(
            f"OnTheFlyVolumeDataset: {len(self.volume_list)} volumes "
            f"(max_slices={max_slices}, img_size={img_size})"
        )

    def __len__(self):
        return len(self.volume_list)

    def __getitem__(self, idx):
        vid = self.volume_list[idx]
        info = self.volume_index[vid]
        slice_keys = info["slice_keys"]  # [(shard_path, key_prefix, slice_idx), ...]

        # Subsample if too many slices
        if len(slice_keys) > self.max_slices:
            indices = sorted(random.sample(range(len(slice_keys)), self.max_slices))
            slice_keys = [slice_keys[i] for i in indices]

        # Load images from tar files
        images = []
        # Group by shard for efficient tar access
        by_shard = defaultdict(list)
        for shard_path, key_prefix, slice_idx in slice_keys:
            by_shard[shard_path].append(key_prefix)

        loaded = {}
        for shard_path, keys in by_shard.items():
            try:
                with tarfile.open(shard_path, "r") as tar:
                    for member in tar.getmembers():
                        name_base = member.name.rsplit(".", 1)[0]
                        if name_base in keys and member.name.endswith(".png"):
                            f = tar.extractfile(member)
                            if f is not None:
                                img = Image.open(io.BytesIO(f.read()))
                                if img.mode != "RGB":
                                    img = img.convert("RGB")
                                loaded[name_base] = self.transform(img)
            except Exception as e:
                logger.warning(f"Error reading {shard_path}: {e}")

        # Collect in slice order
        for shard_path, key_prefix, slice_idx in slice_keys:
            if key_prefix in loaded:
                images.append(loaded[key_prefix])

        N = len(images)
        if N == 0:
            # Fallback: return zeros
            img_size = self.transform.transforms[0].size[0]
            images_t = torch.zeros(self.max_slices, 3, img_size, img_size)
            mask = torch.zeros(self.max_slices, dtype=torch.float32)
        elif N < self.max_slices:
            images_t = torch.stack(images)
            pad = torch.zeros(self.max_slices - N, *images_t.shape[1:])
            images_t = torch.cat([images_t, pad], dim=0)
            mask = torch.zeros(self.max_slices, dtype=torch.float32)
            mask[:N] = 1.0
        else:
            images_t = torch.stack(images[: self.max_slices])
            mask = torch.ones(self.max_slices, dtype=torch.float32)

        has_lesion = info["has_lesion"]
        dataset_name = info["dataset"]
        class_idx = self.class_mapping.get(dataset_name, -1)

        labels = {
            "has_lesion": has_lesion,
            "class_idx": class_idx,
            "dataset": dataset_name,
            "sample_id": vid,
        }

        return images_t, mask, labels


# ── Collation ────────────────────────────────────────────────────────

def collate_volumes(batch):
    """Collate variable-length volume batches.

    Args:
        batch: List of (data_tensor, mask, labels_dict) tuples.

    Returns:
        data: (B, K, ...) stacked tensor (embeddings or images).
        masks: (B, K) attention masks.
        labels: dict of batched label tensors/lists.
    """
    data = torch.stack([b[0] for b in batch])
    masks = torch.stack([b[1] for b in batch])

    label_keys = batch[0][2].keys()
    labels = {}
    for key in label_keys:
        vals = [b[2][key] for b in batch]
        if isinstance(vals[0], (int, float)):
            labels[key] = torch.tensor(vals)
        else:
            labels[key] = vals  # strings stay as list

    return data, masks, labels


# ── Embedding Extraction ─────────────────────────────────────────────

def extract_and_cache_embeddings(
    shard_dir,
    backbone_checkpoint,
    output_dir,
    arch="vit_base",
    patch_size=14,
    img_size=224,
    batch_size=64,
    num_workers=8,
    device="cuda",
    datasets=None,
):
    """Extract CLS tokens from frozen backbone for all slices, save per-volume.

    Output structure:
        output_dir/
            volume_index.json
            embeddings/
                <sample_id>.pt  (cls_tokens, slice_indices, has_lesion, dataset)

    Args:
        shard_dir: Root shard directory.
        backbone_checkpoint: Path to DINO teacher backbone checkpoint.
        output_dir: Directory to save cached embeddings.
        arch: Backbone architecture name.
        patch_size: ViT patch size.
        img_size: Input image size.
        batch_size: Batch size for extraction.
        num_workers: DataLoader workers.
        device: Device string.
        datasets: Optional dataset filter list.

    Returns:
        Path to output_dir.
    """
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(project_root / "dinov2"))

    from ultrassl.models.backbone import build_backbone

    # Build and freeze backbone
    backbone, embed_dim = build_backbone(
        model_name=arch,
        patch_size=patch_size,
        pretrained=backbone_checkpoint,
        img_size=img_size,
        drop_path_rate=0.0,
    )
    backbone.to(device)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    # Scan volume index
    volume_index = scan_volume_index(shard_dir, datasets)

    # Setup output
    embed_dir = os.path.join(output_dir, "embeddings")
    os.makedirs(embed_dir, exist_ok=True)

    # Detection transform (same as wds_labeled_dataset.py)
    det_transform = transforms.Compose([
        transforms.Resize(
            (img_size, img_size),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Process each volume
    n_volumes = len(volume_index)
    for vi, (sid, info) in enumerate(sorted(volume_index.items())):
        out_path = os.path.join(embed_dir, f"{sid}.pt")
        if os.path.isfile(out_path):
            continue  # Skip already extracted

        # Load all slices for this volume from tars
        slice_images = []
        slice_indices = []
        by_shard = defaultdict(list)
        for shard_path, key_prefix, slice_idx in info["slice_keys"]:
            by_shard[shard_path].append((key_prefix, slice_idx))

        for shard_path, items in by_shard.items():
            key_set = {kp for kp, _ in items}
            idx_map = {kp: si for kp, si in items}
            try:
                with tarfile.open(shard_path, "r") as tar:
                    for member in tar.getmembers():
                        name_base = member.name.rsplit(".", 1)[0]
                        if name_base in key_set and member.name.endswith(".png"):
                            f = tar.extractfile(member)
                            if f is not None:
                                img = Image.open(io.BytesIO(f.read()))
                                if img.mode != "RGB":
                                    img = img.convert("RGB")
                                slice_images.append(det_transform(img))
                                slice_indices.append(idx_map[name_base])
            except Exception as e:
                logger.warning(f"Error reading {shard_path} for {sid}: {e}")

        if not slice_images:
            logger.warning(f"No images loaded for volume {sid}")
            continue

        # Sort by slice index
        sorted_pairs = sorted(zip(slice_indices, slice_images), key=lambda x: x[0])
        slice_indices = [p[0] for p in sorted_pairs]
        slice_images = [p[1] for p in sorted_pairs]

        # Batch forward through backbone
        all_cls_tokens = []
        for i in range(0, len(slice_images), batch_size):
            batch = torch.stack(slice_images[i : i + batch_size]).to(device)
            with torch.no_grad():
                out = backbone(batch, is_training=True)
                cls_tokens = out["x_norm_clstoken"].cpu()  # (B, embed_dim)
            all_cls_tokens.append(cls_tokens)

        cls_tokens = torch.cat(all_cls_tokens, dim=0)  # (N, embed_dim)

        torch.save(
            {
                "cls_tokens": cls_tokens,
                "slice_indices": slice_indices,
                "has_lesion": info["has_lesion"],
                "dataset": info["dataset"],
                "n_positive_slices": info["n_positive_slices"],
            },
            out_path,
        )

        if (vi + 1) % 50 == 0 or (vi + 1) == n_volumes:
            logger.info(f"Extracted {vi + 1}/{n_volumes} volumes")

    # Save volume index
    index_for_json = {}
    for sid, info in volume_index.items():
        index_for_json[sid] = {
            "dataset": info["dataset"],
            "has_lesion": info["has_lesion"],
            "n_slices": info["n_slices"],
            "n_positive_slices": info["n_positive_slices"],
        }
    with open(os.path.join(output_dir, "volume_index.json"), "w") as f:
        json.dump(index_for_json, f, indent=2)

    logger.info(f"Embedding cache complete: {output_dir}")
    return output_dir


# ── Utilities ────────────────────────────────────────────────────────

def compute_class_weights(volume_index, volume_ids, class_mapping=None):
    """Compute inverse-frequency class weights for balanced training.

    Args:
        volume_index: Output of scan_volume_index().
        volume_ids: Set of volume IDs to compute weights over.
        class_mapping: Dict mapping dataset name to class index.

    Returns:
        torch.Tensor of shape (n_classes,) with per-class weights.
    """
    class_mapping = class_mapping or CLASS_MAPPING
    n_classes = len(class_mapping)
    counts = torch.zeros(n_classes)

    for vid in volume_ids:
        if vid not in volume_index:
            continue
        ds = volume_index[vid]["dataset"]
        if ds in class_mapping:
            counts[class_mapping[ds]] += 1

    # Inverse frequency: total / (n_classes * count_per_class)
    total = counts.sum()
    weights = total / (n_classes * counts.clamp(min=1))

    logger.info(f"Class weights: {dict(zip(class_mapping.keys(), weights.tolist()))}")
    return weights
