#!/usr/bin/env python3
"""
Convert ultrasound image directories into WebDataset tar shards.

Packs many individual image files into a small number of .tar shard files.
Example: 110K PNGs → ~110 tar files (1000 images each).

Usage:
    python -m ultrassl.data.create_shards \
        --data-root config/data_root.json \
        --output-dir /path/to/shards \
        --images-per-shard 1000

The output directory will contain:
    shard-000000.tar
    shard-000001.tar
    ...

Each sample in a shard is stored as:
    <key>.png    (the image)
    <key>.json   (metadata: dataset name, original path)
"""

import argparse
import io
import json
import logging
import os
import re
import sys
from pathlib import Path

import webdataset as wds
from PIL import Image

# Add project root to path (and dinov2 for any transitive imports)
_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "dinov2"))
os.environ.setdefault("XFORMERS_DISABLED", "1")

# Import directly from dataset module to avoid __init__.py triggering dinov2 imports
from ultrassl.data.dataset import (
    _scan_directory,
    _detect_volume_slices,
    _subsample_volume,
    ALL_EXTENSIONS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("create_shards")


def _looks_like_volume_data(paths):
    if not paths:
        return False
    volumes = _detect_volume_slices(paths)
    if not volumes:
        return False
    n_groups = len(volumes)
    avg_slices = len(paths) / n_groups
    # 3D volume data: many groups (patient volumes) AND many slices per group
    # 2D datasets: few groups (categories) with many images — NOT volume data
    return n_groups >= 10 and avg_slices > 20


def load_image_bytes(path: str) -> bytes:
    """Load an image file and return PNG bytes (handles .npy/.npz too)."""
    import numpy as np

    ext = Path(path).suffix.lower()

    if ext in {".npy", ".npz"}:
        if ext == ".npz":
            data = np.load(path)
            arr = data[list(data.keys())[0]]
        else:
            arr = np.load(path)

        if arr.ndim == 3 and arr.shape[0] <= 4:
            arr = arr[0]
        elif arr.ndim == 3 and arr.shape[-1] <= 4:
            arr = arr[..., 0]
        elif arr.ndim > 3:
            arr = arr.reshape(-1, arr.shape[-2], arr.shape[-1])[0]

        if arr.dtype in (np.float32, np.float64):
            arr = ((arr / arr.max()) * 255).astype(np.uint8) if arr.max() > 0 else arr.astype(np.uint8)
        elif arr.dtype == np.uint16:
            arr = (arr / 256).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)

        img = Image.fromarray(arr, mode="L").convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    else:
        img = Image.open(path).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()


def main():
    parser = argparse.ArgumentParser(description="Convert images to WebDataset shards")
    parser.add_argument("--data-root", required=True, help="Path to data_root.json")
    parser.add_argument("--output-dir", required=True, help="Output directory for shards")
    parser.add_argument("--images-per-shard", type=int, default=1000, help="Images per tar shard")
    parser.add_argument("--volume-stride", type=int, default=3, help="Stride for volume subsampling")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset config
    with open(args.data_root) as f:
        config = json.load(f)

    # Collect all image paths
    all_paths = []
    all_labels = []

    for entry in config["data"]:
        name = entry["name"]
        root = entry["path"]

        if not os.path.isdir(root):
            logger.warning(f"Dataset '{name}' not found at {root}, skipping")
            continue

        raw_paths = _scan_directory(root, ALL_EXTENSIONS)
        if not raw_paths:
            logger.warning(f"Dataset '{name}': no images found")
            continue

        # Volume-aware subsampling
        if _looks_like_volume_data(raw_paths) and args.volume_stride > 1:
            volumes = _detect_volume_slices(raw_paths)
            subsampled = []
            for vol_id, slices in volumes:
                subsampled.extend(_subsample_volume(slices, stride=args.volume_stride))
            logger.info(f"Dataset '{name}': {len(raw_paths)} → {len(subsampled)} after stride-{args.volume_stride}")
            raw_paths = subsampled
        else:
            logger.info(f"Dataset '{name}': {len(raw_paths)} images")

        all_paths.extend(raw_paths)
        all_labels.extend([name] * len(raw_paths))

    logger.info(f"Total: {len(all_paths)} images")

    if not all_paths:
        logger.error("No images found. Check data_root.json paths.")
        return

    # Shuffle deterministically for balanced shards
    import random
    combined = list(zip(all_paths, all_labels))
    random.seed(42)
    random.shuffle(combined)
    all_paths, all_labels = zip(*combined)

    # Write shards
    n_shards = (len(all_paths) + args.images_per_shard - 1) // args.images_per_shard
    logger.info(f"Writing {n_shards} shards ({args.images_per_shard} images each)")

    shard_pattern = os.path.join(args.output_dir, "shard-%06d.tar")

    with wds.ShardWriter(shard_pattern, maxcount=args.images_per_shard) as sink:
        skipped = 0
        for i, (path, label) in enumerate(zip(all_paths, all_labels)):
            try:
                png_bytes = load_image_bytes(path)
                sample = {
                    "__key__": f"{i:08d}",
                    "png": png_bytes,
                    "json": json.dumps({"dataset": label, "path": path}).encode(),
                }
                sink.write(sample)
            except Exception as e:
                logger.warning(f"Skipping {path}: {e}")
                skipped += 1

            if (i + 1) % 5000 == 0:
                logger.info(f"  Processed {i + 1}/{len(all_paths)}")

    # Summary
    shard_files = sorted(Path(args.output_dir).glob("shard-*.tar"))
    total_size = sum(f.stat().st_size for f in shard_files)
    logger.info(f"Done! {len(shard_files)} shards, {total_size / 1e9:.1f} GB total")
    logger.info(f"  Written: {len(all_paths) - skipped}, Skipped: {skipped}")
    logger.info(f"  Output: {args.output_dir}")


if __name__ == "__main__":
    main()
