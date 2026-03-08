#!/usr/bin/env python3
"""
Data preparation for patch-level lesion detection.

Scans WebDataset shards, performs volume-level 70/15/15 stratified split,
and writes metadata JSONs for training/validation/testing.

Usage:
    python prepare_patch_data.py \
        --shard-dir /path/to/Shards \
        --output-dir ./splits \
        --train-ratio 0.70 --val-ratio 0.15 --test-ratio 0.15

    # Use specific datasets only:
    python prepare_patch_data.py \
        --shard-dir /path/to/Shards \
        --output-dir ./splits \
        --datasets BIrads Class3 Class4 Abus
"""

import argparse
import json
import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ultrassl")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare data splits for patch-level lesion detection"
    )
    parser.add_argument("--shard-dir", required=True,
                        help="Root directory containing <dataset>/shard-*.tar")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory for split JSON files")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Dataset names to include (default: all)")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Validate ratios
    total = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total - 1.0) > 0.01:
        logger.error(f"Split ratios must sum to 1.0, got {total}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # Step 1: Scan shards
    logger.info("=== Scanning shards ===")
    from ultrassl.patch_detection import scan_volume_ids, volume_three_way_split

    volume_info = scan_volume_ids(args.shard_dir, args.datasets)

    if not volume_info:
        logger.error("No volumes found!")
        sys.exit(1)

    # Step 2: Three-way split
    logger.info("\n=== Splitting volumes ===")
    train_ids, val_ids, test_ids = volume_three_way_split(
        volume_info,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    # Step 3: Build split metadata
    splits = {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids,
    }

    summary = {}
    for split_name, split_ids in splits.items():
        volumes_dict = {}
        stats = {"n_volumes": 0, "n_slices": 0, "n_positive_slices": 0,
                 "n_positive_volumes": 0, "per_dataset": {}}

        for vid in sorted(split_ids):
            info = volume_info[vid]
            volumes_dict[vid] = info
            stats["n_volumes"] += 1
            stats["n_slices"] += info["n_slices"]
            stats["n_positive_slices"] += info.get("n_positive_slices", 0)
            if info["has_lesion"]:
                stats["n_positive_volumes"] += 1

            ds = info["dataset"]
            if ds not in stats["per_dataset"]:
                stats["per_dataset"][ds] = {
                    "n_volumes": 0, "n_positive_volumes": 0,
                    "n_slices": 0, "n_positive_slices": 0,
                }
            stats["per_dataset"][ds]["n_volumes"] += 1
            stats["per_dataset"][ds]["n_slices"] += info["n_slices"]
            stats["per_dataset"][ds]["n_positive_slices"] += info.get("n_positive_slices", 0)
            if info["has_lesion"]:
                stats["per_dataset"][ds]["n_positive_volumes"] += 1

        if stats["n_slices"] > 0:
            stats["positive_slice_ratio"] = round(
                stats["n_positive_slices"] / stats["n_slices"], 4
            )
        else:
            stats["positive_slice_ratio"] = 0.0

        split_data = {
            "volume_ids": sorted(split_ids),
            "volumes": volumes_dict,
            "stats": stats,
        }

        # Write split file
        split_path = os.path.join(args.output_dir, f"{split_name}_split.json")
        with open(split_path, "w") as f:
            json.dump(split_data, f, indent=2)
        logger.info(f"Wrote {split_path}")

        summary[split_name] = stats

    # Step 4: Write summary
    summary_path = os.path.join(args.output_dir, "split_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Wrote {summary_path}")

    # Print summary
    logger.info("\n=== Split Summary ===")
    for split_name, stats in summary.items():
        logger.info(
            f"  {split_name}: {stats['n_volumes']} volumes "
            f"({stats['n_positive_volumes']} pos), "
            f"{stats['n_slices']} slices "
            f"({stats['n_positive_slices']} pos, "
            f"ratio={stats['positive_slice_ratio']:.4f})"
        )
        for ds, ds_stats in sorted(stats["per_dataset"].items()):
            logger.info(
                f"    {ds}: {ds_stats['n_volumes']} volumes "
                f"({ds_stats['n_positive_volumes']} pos), "
                f"{ds_stats['n_slices']} slices "
                f"({ds_stats['n_positive_slices']} pos)"
            )


if __name__ == "__main__":
    main()
