"""
WebDataset loader for labeled ultrasound shards.

Supports two modes:
  - "ssl": Returns (augmented_dict, ()) for DINOv2 self-supervised training.
           Annotations are ignored — all slices used as unlabeled data.
  - "detection": Returns (image_tensor, annotation_dict) for classifier training.

Features:
  - Dataset-balanced sampling via shard replication
  - Positive enrichment (subsample negatives) in both modes
  - Volume-aware filtering interface

Shards are expected at: <shard_dir>/<dataset_name>/shard-*.tar
Created by create_labeled_shards.py.
"""

import glob
import json
import logging
import math
import os
import random

import torch
from PIL import Image
from torchvision import transforms
import webdataset as wds

logger = logging.getLogger("ultrassl")


def _discover_shards(shard_dir: str, datasets: list = None) -> list:
    """Find all shard .tar files under shard_dir, optionally filtering by dataset name."""
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
    return shards


def _discover_shards_balanced(shard_dir: str, datasets: list = None) -> list:
    """Find shards with replication for dataset balance.

    Replicates smaller datasets' shard lists to match the largest dataset,
    ensuring approximately equal sampling across datasets.
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


def build_labeled_wds_dataset(
    shard_dir: str,
    mode: str = "ssl",
    transform=None,
    epoch_length: int = 1000,
    batch_size: int = 24,
    num_workers: int = 8,
    shuffle_buffer: int = 5000,
    world_size: int = 1,
    rank: int = 0,
    datasets: list = None,
    oversample_positive: float = 0.0,
    img_size: int = 224,
    balance_datasets: bool = True,
    pos_enrichment: float = 0.0,
):
    """Build a WebDataset pipeline from labeled shards.

    Args:
        shard_dir: Root directory containing <dataset_name>/shard-*.tar subdirs.
        mode: "ssl" for self-supervised (ignores labels), "detection" for classifier.
        transform: Augmentation transform (UltrasoundAugmentationDINO for SSL mode).
        epoch_length: Number of batches per epoch.
        batch_size: Batch size per GPU.
        num_workers: DataLoader workers.
        shuffle_buffer: Samples to buffer for shuffling.
        world_size: Number of DDP processes.
        rank: Current DDP rank.
        datasets: List of dataset names to include (None = all).
        oversample_positive: Factor for oversampling positive slices in detection mode.
            E.g., 3.0 means negatives are kept with probability 1/3.
        img_size: Image resize target for detection mode.
        balance_datasets: If True, replicate smaller datasets' shards to match largest.
        pos_enrichment: Positive enrichment factor for SSL mode.
            E.g., 10.0 means keep ~10% of negatives for ~1:1 Pos:Neg ratio.
            0.0 = no enrichment (keep all samples).

    Returns:
        (dataset, loader) tuple.
    """
    if balance_datasets:
        shards = _discover_shards_balanced(shard_dir, datasets)
    else:
        shards = _discover_shards(shard_dir, datasets)

    if not shards:
        raise FileNotFoundError(f"No shard files found under {shard_dir}")

    n_shards = len(shards)

    # Cap workers to shard count
    if num_workers > n_shards:
        logger.info(f"Labeled WDS: reducing num_workers {num_workers} -> {n_shards}")
        num_workers = n_shards

    logger.info(f"Labeled WDS [{mode}]: {n_shards} shards from {shard_dir}")

    if mode == "ssl":
        dataset, loader = _build_ssl_pipeline(
            shards, transform, shuffle_buffer, epoch_length,
            batch_size, num_workers, pos_enrichment)
    elif mode == "detection":
        dataset, loader = _build_detection_pipeline(
            shards, shuffle_buffer, epoch_length, batch_size,
            num_workers, oversample_positive, img_size)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'ssl' or 'detection'.")

    return dataset, loader


# ── SSL mode ──────────────────────────────────────────────────────────

def _build_ssl_pipeline(shards, transform, shuffle_buffer, epoch_length,
                        batch_size, num_workers, pos_enrichment=0.0):
    """SSL mode: ignore labels, return (augmented_image, ()) like build_wds_dataset."""

    def decode_ssl(sample):
        img = sample["png"]  # PIL image (decoded by webdataset)
        if img.mode != "RGB":
            img = img.convert("RGB")
        if transform is not None:
            img = transform(img)
        target = ()
        return img, target

    pipeline = (
        wds.WebDataset(shards, shardshuffle=True, nodesplitter=wds.split_by_node,
                        empty_check=False)
        .shuffle(shuffle_buffer)
        .decode("pil")
    )

    # Positive enrichment: subsample negatives to achieve ~1:1 ratio
    if pos_enrichment > 0:
        keep_prob = 1.0 / (1.0 + pos_enrichment)

        def pos_enrichment_filter(sample):
            ann = sample.get("json", b"{}")
            if isinstance(ann, bytes):
                ann = json.loads(ann.decode())
            elif isinstance(ann, str):
                ann = json.loads(ann)
            if ann.get("has_lesion", 0) == 1:
                return True
            return random.random() < keep_prob

        pipeline = pipeline.select(pos_enrichment_filter)

    dataset = pipeline.map(decode_ssl)

    loader = wds.WebLoader(
        dataset,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    n_batches = epoch_length
    loader = loader.repeat(2).slice(n_batches * batch_size)

    logger.info(f"Labeled WDS [ssl]: shuffle_buffer={shuffle_buffer}, "
                f"epoch={n_batches} batches, pos_enrichment={pos_enrichment}")
    return dataset, loader


# ── Detection mode ────────────────────────────────────────────────────

def _build_detection_pipeline(shards, shuffle_buffer, epoch_length,
                              batch_size, num_workers,
                              oversample_positive, img_size):
    """Detection mode: return (image_tensor, annotation_dict)."""

    detection_transform = transforms.Compose([
        transforms.Resize((img_size, img_size),
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    def decode_detection(sample):
        img = sample["png"]
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Parse annotation JSON
        ann = sample["json"]
        if isinstance(ann, bytes):
            ann = json.loads(ann.decode())
        elif isinstance(ann, str):
            ann = json.loads(ann)
        # else: already a dict (webdataset may auto-decode)

        tensor = detection_transform(img)
        return tensor, ann

    pipeline = (
        wds.WebDataset(shards, shardshuffle=True, nodesplitter=wds.split_by_node,
                        empty_check=False)
        .shuffle(shuffle_buffer)
        .decode("pil")
    )

    # Class-balanced sampling: keep all positives, subsample negatives
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

    dataset = pipeline.map(decode_detection)

    loader = wds.WebLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        collate_fn=collate_detection,
    )

    n_batches = epoch_length
    loader = loader.repeat(2).slice(n_batches)

    logger.info(f"Labeled WDS [detection]: batch_size={batch_size}, "
                f"oversample_positive={oversample_positive}, "
                f"epoch={n_batches} batches")
    return dataset, loader


def collate_detection(samples):
    """Collate for detection mode: stack image tensors, list annotations."""
    images = torch.stack([s[0] for s in samples])
    annotations = [s[1] for s in samples]
    return images, annotations
