#!/usr/bin/env python3
"""
Convert labeled ultrasound slice directories into WebDataset tar shards.

Reads from processed 3D datasets with paired images/ and labels/ folders:
    images/<sample_id>/slice_XXXX.png
    labels/<sample_id>/slice_XXXX.txt

Each label .txt line: has_lesion x1 y1 x2 y2 (pixel coords).
Negative slices: 0 0 0 0 0. Multiple lines = multiple bboxes.

Output structure:
    wds/<dataset_name>/shard-000000.tar
    wds/<dataset_name>/shard-000001.tar
    ...
    wds/index.json   (per-dataset statistics)

Each sample in a shard:
    <key>.png    (RGB image)
    <key>.json   (annotation: dataset, sample_id, slice info, bboxes, area buckets)

Usage:
    python -m ultrassl.data.create_labeled_shards \
        --config config/data_label_root_3d.json \
        --output-dir wds/

    # With quality filtering and negative subsampling:
    python -m ultrassl.data.create_labeled_shards \
        --config config/data_label_root_3d.json \
        --output-dir wds/ \
        --skip-boundary 3 --min-variance 100 --neg-stride 2
"""

import argparse
import io
import json
import logging
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import webdataset as wds
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("create_labeled_shards")


# ── Frame quality filtering ──────────────────────────────────────────

def is_valid_frame(img_path, slice_idx, total_slices, skip_boundary=3, min_variance=100.0):
    """Check if a frame passes quality filters.

    Args:
        img_path: Path to the image file.
        slice_idx: 0-based slice index within the volume.
        total_slices: Total number of slices in this volume.
        skip_boundary: Skip first/last N slices (typically near-blank).
        min_variance: Minimum pixel variance threshold (reject blank frames).

    Returns:
        (is_valid, reason) tuple.
    """
    # Boundary check
    if skip_boundary > 0:
        if slice_idx < skip_boundary or slice_idx >= total_slices - skip_boundary:
            return False, "boundary"

    # Variance check (detect blank/near-uniform frames)
    if min_variance > 0:
        img = Image.open(img_path).convert("L")
        arr = np.array(img, dtype=np.float32)
        if arr.var() < min_variance:
            return False, "low_variance"

    return True, "ok"


# ── Label parsing with bbox sanitization ─────────────────────────────

def parse_label_file(label_path: str, img_width: int, img_height: int) -> dict:
    """Parse a label .txt file into an annotation dict with bbox sanitization.

    Each line: has_lesion x1 y1 x2 y2 (space-separated).
    Negative: 0 0 0 0 0.  Positive with bbox: 1 x1 y1 x2 y2.

    Sanitization:
    - Clamp coords to image bounds
    - Enforce x1 < x2, y1 < y2 (swap if inverted)
    - Drop tiny bboxes (area < 4 pixels)
    - Compute area ratios and size buckets

    Returns:
        {"has_lesion": 0|1, "bboxes": [...], "bboxes_normalized": [...],
         "bbox_area_ratios": [...], "bbox_area_bucket": "small"|"medium"|"large"|"none",
         "image_width": w, "image_height": h}
    """
    raw_bboxes = []
    has_lesion = 0

    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                logger.warning(f"Malformed line in {label_path}: '{line}'")
                continue
            cls = int(float(parts[0]))
            if cls == 1:
                has_lesion = 1
                x1, y1, x2, y2 = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                raw_bboxes.append([x1, y1, x2, y2])

    # Sanitize bboxes
    bboxes = []
    for x1, y1, x2, y2 in raw_bboxes:
        # Clamp to image bounds
        x1 = max(0.0, min(x1, float(img_width)))
        y1 = max(0.0, min(y1, float(img_height)))
        x2 = max(0.0, min(x2, float(img_width)))
        y2 = max(0.0, min(y2, float(img_height)))
        # Enforce x1 < x2, y1 < y2
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        # Drop tiny bboxes (area < 4 pixels)
        area = (x2 - x1) * (y2 - y1)
        if area < 4.0:
            continue
        bboxes.append([x1, y1, x2, y2])

    # If all bboxes were dropped, flip has_lesion
    if has_lesion == 1 and len(bboxes) == 0:
        has_lesion = 0

    # Compute normalized coordinates and area ratios
    img_area = float(img_width * img_height)
    bboxes_normalized = []
    bbox_area_ratios = []
    for x1, y1, x2, y2 in bboxes:
        bboxes_normalized.append([
            x1 / img_width, y1 / img_height,
            x2 / img_width, y2 / img_height,
        ])
        bbox_area_ratios.append((x2 - x1) * (y2 - y1) / img_area)

    # Area bucket: based on largest bbox
    if bbox_area_ratios:
        max_ratio = max(bbox_area_ratios)
        if max_ratio >= 0.05:
            bbox_area_bucket = "large"
        elif max_ratio >= 0.02:
            bbox_area_bucket = "medium"
        else:
            bbox_area_bucket = "small"
    else:
        bbox_area_bucket = "none"

    return {
        "has_lesion": has_lesion,
        "bboxes": bboxes,
        "bboxes_normalized": bboxes_normalized,
        "bbox_area_ratios": [round(r, 6) for r in bbox_area_ratios],
        "bbox_area_bucket": bbox_area_bucket,
        "image_width": img_width,
        "image_height": img_height,
    }


# ── Dataset scanning ──────────────────────────────────────────────────

def scan_labeled_dataset(dataset_path: str) -> list:
    """Scan images/ and labels/ directories, validate alignment.

    Returns list of dicts:
        {"image_path", "label_path", "sample_id", "slice_id", "slice_idx"}
    """
    images_dir = os.path.join(dataset_path, "images")
    labels_dir = os.path.join(dataset_path, "labels")

    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"images/ dir not found: {images_dir}")
    if not os.path.isdir(labels_dir):
        raise FileNotFoundError(f"labels/ dir not found: {labels_dir}")

    # Build lookup: (sample_id, slice_id) -> paths
    image_lookup = {}
    label_lookup = {}

    for sample_id in sorted(os.listdir(images_dir)):
        sample_dir = os.path.join(images_dir, sample_id)
        if not os.path.isdir(sample_dir):
            continue
        for fname in sorted(os.listdir(sample_dir)):
            if not fname.endswith(".png"):
                continue
            slice_id = Path(fname).stem  # e.g. "slice_0119"
            key = (sample_id, slice_id)
            image_lookup[key] = os.path.join(sample_dir, fname)

    for sample_id in sorted(os.listdir(labels_dir)):
        sample_dir = os.path.join(labels_dir, sample_id)
        if not os.path.isdir(sample_dir):
            continue
        for fname in sorted(os.listdir(sample_dir)):
            if not fname.endswith(".txt"):
                continue
            slice_id = Path(fname).stem
            key = (sample_id, slice_id)
            label_lookup[key] = os.path.join(sample_dir, fname)

    # Validate alignment
    image_keys = set(image_lookup.keys())
    label_keys = set(label_lookup.keys())

    orphan_images = image_keys - label_keys
    orphan_labels = label_keys - image_keys
    if orphan_images:
        logger.warning(f"  {len(orphan_images)} images without labels (skipping)")
    if orphan_labels:
        logger.warning(f"  {len(orphan_labels)} labels without images (skipping)")

    # Build aligned records
    matched_keys = sorted(image_keys & label_keys)
    records = []
    for sample_id, slice_id in matched_keys:
        # Extract numeric index from slice_id (e.g. "slice_0119" -> 119)
        m = re.search(r"(\d+)$", slice_id)
        slice_idx = int(m.group(1)) if m else 0

        records.append({
            "image_path": image_lookup[(sample_id, slice_id)],
            "label_path": label_lookup[(sample_id, slice_id)],
            "sample_id": sample_id,
            "slice_id": slice_id,
            "slice_idx": slice_idx,
        })

    return records


# ── Negative stride subsampling ──────────────────────────────────────

def get_volume_slice_counts(records):
    """Compute total slices per volume from records.

    Returns dict: sample_id -> total_slice_count
    """
    counts = defaultdict(int)
    for rec in records:
        counts[rec["sample_id"]] += 1
    return dict(counts)


def apply_negative_stride(records, neg_stride=1, label_cache=None):
    """Keep all positives, subsample negatives per-volume.

    Args:
        records: List of record dicts from scan_labeled_dataset().
        neg_stride: Keep every Nth negative slice per volume. 1 = keep all.
        label_cache: Optional dict mapping label_path -> has_lesion (0|1).
            If None, reads label files to determine positive/negative.

    Returns:
        Filtered list of records.
    """
    if neg_stride <= 1:
        return records

    # Group by sample_id
    by_volume = defaultdict(list)
    for rec in records:
        by_volume[rec["sample_id"]].append(rec)

    filtered = []
    n_dropped = 0
    for sample_id in sorted(by_volume.keys()):
        vol_records = sorted(by_volume[sample_id], key=lambda r: r["slice_idx"])
        neg_count = 0
        for rec in vol_records:
            # Check if positive
            is_positive = False
            if label_cache is not None and rec["label_path"] in label_cache:
                is_positive = label_cache[rec["label_path"]] == 1
            else:
                # Quick check: read first line of label file
                try:
                    with open(rec["label_path"], "r") as f:
                        first_line = f.readline().strip()
                    if first_line:
                        is_positive = int(float(first_line.split()[0])) == 1
                except Exception:
                    pass

            if is_positive:
                filtered.append(rec)
            else:
                if neg_count % neg_stride == 0:
                    filtered.append(rec)
                else:
                    n_dropped += 1
                neg_count += 1

    logger.info(f"  Negative stride={neg_stride}: dropped {n_dropped} negatives, "
                f"kept {len(filtered)}/{len(records)}")
    return filtered


# ── Image loading ─────────────────────────────────────────────────────

def load_image_as_rgb_bytes(path: str) -> tuple:
    """Load image, convert to RGB, return (PNG bytes, width, height)."""
    img = Image.open(path)
    w, h = img.size
    img_rgb = img.convert("RGB")
    buf = io.BytesIO()
    img_rgb.save(buf, format="PNG")
    return buf.getvalue(), w, h


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convert labeled ultrasound slices to WebDataset shards")
    parser.add_argument("--config", required=True,
                        help="Path to data_label_root_3d.json")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory (e.g. wds/)")
    parser.add_argument("--images-per-shard", type=int, default=5000,
                        help="Max images per tar shard")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for deterministic shuffling")
    parser.add_argument("--skip-boundary", type=int, default=3,
                        help="Skip first/last N slices per volume (0 to disable)")
    parser.add_argument("--min-variance", type=float, default=100.0,
                        help="Min pixel variance to keep frame (0 to disable)")
    parser.add_argument("--neg-stride", type=int, default=1,
                        help="Keep every Nth negative slice per volume (1 = keep all)")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = json.load(f)

    index = {}  # per-dataset stats for index.json

    for entry in config["data"]:
        ds_name = entry["name"]
        ds_path = entry["path"]

        logger.info(f"Processing dataset: {ds_name} ({ds_path})")

        if not os.path.isdir(ds_path):
            logger.warning(f"  Dataset path not found, skipping: {ds_path}")
            continue

        # Scan and validate
        records = scan_labeled_dataset(ds_path)
        logger.info(f"  Found {len(records)} matched image-label pairs")

        if not records:
            continue

        # Compute per-volume slice counts (needed for boundary filtering)
        vol_counts = get_volume_slice_counts(records)

        # Apply negative stride subsampling
        records = apply_negative_stride(records, neg_stride=args.neg_stride)

        # Deterministic shuffle (preserves reproducibility across runs)
        random.seed(args.seed)
        random.shuffle(records)

        # Per-dataset output directory
        ds_output_dir = os.path.join(args.output_dir, ds_name)
        os.makedirs(ds_output_dir, exist_ok=True)

        shard_pattern = os.path.join(ds_output_dir, "shard-%06d.tar")

        # Track statistics
        n_positive = 0
        n_negative = 0
        n_multi_bbox = 0
        total_bboxes = 0
        n_written = 0
        skipped = 0
        n_skipped_boundary = 0
        n_skipped_low_variance = 0
        sample_ids = set()
        bbox_area_dist = {"small": 0, "medium": 0, "large": 0}

        with wds.ShardWriter(shard_pattern, maxcount=args.images_per_shard) as sink:
            for i, rec in enumerate(records):
                try:
                    # Frame quality check
                    total_slices = vol_counts.get(rec["sample_id"], 0)
                    valid, reason = is_valid_frame(
                        rec["image_path"], rec["slice_idx"], total_slices,
                        skip_boundary=args.skip_boundary,
                        min_variance=args.min_variance,
                    )
                    if not valid:
                        if reason == "boundary":
                            n_skipped_boundary += 1
                        elif reason == "low_variance":
                            n_skipped_low_variance += 1
                        skipped += 1
                        continue

                    # Load image
                    png_bytes, img_w, img_h = load_image_as_rgb_bytes(rec["image_path"])

                    # Parse and sanitize label
                    annotation = parse_label_file(rec["label_path"], img_w, img_h)
                    annotation["dataset"] = ds_name
                    annotation["sample_id"] = rec["sample_id"]
                    annotation["slice_id"] = rec["slice_id"]
                    annotation["slice_idx"] = rec["slice_idx"]

                    # Write to shard
                    sample = {
                        "__key__": f"{n_written:08d}",
                        "png": png_bytes,
                        "json": json.dumps(annotation).encode(),
                    }
                    sink.write(sample)
                    n_written += 1

                    # Update stats
                    sample_ids.add(rec["sample_id"])
                    if annotation["has_lesion"]:
                        n_positive += 1
                        n_bboxes = len(annotation["bboxes"])
                        total_bboxes += n_bboxes
                        if n_bboxes > 1:
                            n_multi_bbox += 1
                        bucket = annotation["bbox_area_bucket"]
                        if bucket in bbox_area_dist:
                            bbox_area_dist[bucket] += 1
                    else:
                        n_negative += 1

                except Exception as e:
                    logger.warning(f"  Skipping {rec['image_path']}: {e}")
                    skipped += 1

                if (i + 1) % 5000 == 0:
                    logger.info(f"  Processed {i + 1}/{len(records)} "
                                f"(written {n_written}, skipped {skipped})")

        # Count shards
        shard_files = sorted(Path(ds_output_dir).glob("shard-*.tar"))
        total_size = sum(f.stat().st_size for f in shard_files)

        # Store stats
        ds_stats = {
            "n_volumes": len(sample_ids),
            "n_slices_total": n_positive + n_negative,
            "n_slices_positive": n_positive,
            "n_slices_negative": n_negative,
            "n_multi_bbox_slices": n_multi_bbox,
            "n_total_bboxes": total_bboxes,
            "n_shards": len(shard_files),
            "total_size_bytes": total_size,
            "skipped": skipped,
            "n_skipped_boundary": n_skipped_boundary,
            "n_skipped_low_variance": n_skipped_low_variance,
            "bbox_area_distribution": bbox_area_dist,
        }
        index[ds_name] = ds_stats

        logger.info(f"  {ds_name}: {len(shard_files)} shards, {total_size / 1e9:.2f} GB")
        logger.info(f"    Volumes: {len(sample_ids)}, "
                     f"Slices: {n_positive + n_negative} "
                     f"(+{n_positive} / -{n_negative}), "
                     f"Multi-bbox: {n_multi_bbox}")
        logger.info(f"    Skipped: {skipped} "
                     f"(boundary={n_skipped_boundary}, "
                     f"low_var={n_skipped_low_variance})")
        logger.info(f"    Bbox areas: {bbox_area_dist}")

    # Write index file
    index_path = os.path.join(args.output_dir, "index.json")
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)
    logger.info(f"Index written to {index_path}")


if __name__ == "__main__":
    main()
