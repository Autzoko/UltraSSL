#!/usr/bin/env python3
"""
Evaluate trained patch-level lesion detector on val/test sets.

Computes patch-level metrics (accuracy, precision, recall, F1, AUROC),
region-level metrics (recall, precision, F1 against GT bboxes), and
optionally generates heatmap visualizations.

Usage:
    # Evaluate on test set
    python eval_patch_detector.py \
        --config config/patch_detector.yaml \
        --checkpoint outputs/patch_detector/best_model.pth \
        --split test

    # Evaluate with visualizations
    python eval_patch_detector.py \
        --config config/patch_detector.yaml \
        --checkpoint outputs/patch_detector/best_model.pth \
        --split test --visualize 20

    # Evaluate on validation set
    python eval_patch_detector.py \
        --config config/patch_detector.yaml \
        --checkpoint outputs/patch_detector/best_model.pth \
        --split val
"""

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

# Ensure project root and dinov2 are importable
_project_root = Path(__file__).resolve().parent
_dinov2_root = _project_root / "dinov2"
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
if str(_dinov2_root) not in sys.path:
    sys.path.insert(0, str(_dinov2_root))

from ultrassl.patch_detection import (
    PatchDetector,
    assign_three_region_patch_labels,
    build_patch_detection_pipeline,
    compute_patch_metrics,
    compute_region_metrics,
    patches_to_regions,
    visualize_patch_heatmap,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("patch_detector")


@torch.no_grad()
def run_evaluation(model, loader, device, cfg, volume_ids):
    """Run inference on all slices and collect predictions."""
    model.eval()

    grid_h = model.grid_h
    grid_w = model.grid_w
    patch_size = model.patch_size
    img_size = model.img_size

    all_results = []

    for images, annotations in loader:
        # Volume filtering
        keep_mask = []
        for ann in annotations:
            sid = ann.get("sample_id", "")
            keep_mask.append(sid in volume_ids)

        if not any(keep_mask):
            continue

        keep_idx = [i for i, k in enumerate(keep_mask) if k]
        images_filtered = images[torch.tensor(keep_idx)].to(device, non_blocking=True)
        filtered_anns = [annotations[i] for i in keep_idx]

        with torch.amp.autocast("cuda", enabled=cfg.train.get("use_amp", True)):
            patch_logits, patch_map = model(images_filtered)

        patch_scores = torch.sigmoid(patch_logits).cpu()

        for i in range(images_filtered.shape[0]):
            ann = filtered_anns[i]

            # Patch labels for metric computation
            labels = assign_three_region_patch_labels(
                bboxes=ann.get("bboxes", []),
                img_width=ann["image_width"],
                img_height=ann["image_height"],
                grid_h=grid_h, grid_w=grid_w,
                patch_size=patch_size, img_size=img_size,
                shrink_ratio=cfg.labeling.shrink_ratio,
                expand_ratio=cfg.labeling.expand_ratio,
                pos_iou_thresh=cfg.labeling.positive_iou_thresh,
                ignore_iou_thresh=cfg.labeling.ignore_iou_thresh,
            )

            scores_2d = patch_scores[i].view(grid_h, grid_w)

            # Region extraction
            regions = patches_to_regions(
                scores_2d,
                threshold=cfg.eval.threshold,
                min_area_patches=cfg.eval.min_area_patches,
                max_regions=cfg.eval.max_regions,
                grid_h=grid_h,
                grid_w=grid_w,
            )

            all_results.append({
                "sample_id": ann.get("sample_id", ""),
                "slice_idx": ann.get("slice_idx", -1),
                "dataset": ann.get("dataset", ""),
                "has_lesion": ann.get("has_lesion", 0),
                "patch_logits": patch_logits[i].cpu(),
                "patch_labels": labels,
                "patch_scores_2d": scores_2d,
                "pred_regions": regions,
                "gt_bboxes_normalized": ann.get("bboxes_normalized", []),
                "image": images[keep_idx[i]],  # keep original for visualization
            })

    return all_results


def compute_all_metrics(results, cfg):
    """Compute patch-level and region-level metrics."""
    # Patch-level metrics
    all_logits = [r["patch_logits"] for r in results]
    all_labels = [r["patch_labels"] for r in results]
    patch_metrics = compute_patch_metrics(all_logits, all_labels)

    # Region-level metrics
    pred_regions_list = [r["pred_regions"] for r in results]
    annotations_list = [
        {"bboxes_normalized": r["gt_bboxes_normalized"],
         "has_lesion": r["has_lesion"]}
        for r in results
    ]
    region_metrics = compute_region_metrics(
        pred_regions_list, annotations_list,
        iou_threshold=cfg.eval.region_iou_thresh,
    )

    return {**patch_metrics, **region_metrics}


def compute_per_dataset_metrics(results, cfg):
    """Compute metrics broken down by dataset."""
    by_dataset = {}
    for r in results:
        ds = r["dataset"]
        if ds not in by_dataset:
            by_dataset[ds] = []
        by_dataset[ds].append(r)

    per_dataset = {}
    for ds_name, ds_results in sorted(by_dataset.items()):
        per_dataset[ds_name] = compute_all_metrics(ds_results, cfg)
        per_dataset[ds_name]["n_slices"] = len(ds_results)
        per_dataset[ds_name]["n_positive_slices"] = sum(
            1 for r in ds_results if r["has_lesion"]
        )

    return per_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate patch-level lesion detector"
    )
    parser.add_argument("--config", required=True, help="Path to patch_detector.yaml")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--split", default="test", choices=["val", "test"],
                        help="Which split to evaluate on (default: test)")
    parser.add_argument("--visualize", type=int, default=0,
                        help="Number of example heatmaps to visualize (default: 0)")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: from config)")
    args, overrides = parser.parse_known_args()

    # Config
    cfg = OmegaConf.load(args.config)
    if overrides:
        cli_conf = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(cfg, cli_conf)

    output_dir = Path(args.output_dir or cfg.train.output_dir) / f"eval_{args.split}"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("=" * 60)
    logger.info(f"Patch Detector Evaluation — {args.split} set")
    logger.info("=" * 60)
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Device: {device}")

    # Load split metadata
    split_path = os.path.join(cfg.data.split_dir, f"{args.split}_split.json")
    if not os.path.exists(split_path):
        logger.error(f"Split file not found: {split_path}")
        sys.exit(1)

    with open(split_path) as f:
        split_data = json.load(f)

    volume_ids = set(split_data["volume_ids"])
    logger.info(f"Loaded {args.split} split: {len(volume_ids)} volumes, "
                f"{split_data['stats']['n_slices']} slices")

    # Model
    model = PatchDetector(
        backbone_checkpoint=cfg.backbone.checkpoint,
        arch=cfg.backbone.arch,
        patch_size=cfg.backbone.patch_size,
        img_size=cfg.backbone.img_size,
        head_hidden_dim=cfg.model.head_hidden_dim,
    ).to(device)

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.head.load_state_dict(ckpt["head_state_dict"])

    # Load unfrozen backbone blocks if present
    if "unfrozen_blocks" in ckpt:
        for block_key, block_state in ckpt["unfrozen_blocks"].items():
            block_idx = int(block_key.split("_")[1])
            model.backbone.blocks[block_idx].load_state_dict(block_state)
        logger.info(f"Loaded {len(ckpt['unfrozen_blocks'])} unfrozen backbone blocks")

    logger.info(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")

    # Data loader (no oversampling, no shuffle for eval)
    datasets = cfg.data.get("datasets", None)
    if datasets is not None:
        datasets = list(datasets)

    logger.info("Building evaluation data pipeline ...")
    _, eval_loader = build_patch_detection_pipeline(
        shard_dir=cfg.data.shard_dir,
        datasets=datasets,
        img_size=cfg.backbone.img_size,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        shuffle_buffer=100,  # minimal shuffle for eval
        epoch_length=999999,  # process all data
        oversample_positive=0.0,  # no oversampling
        balance_datasets=False,
        volume_ids=volume_ids,
    )

    # Run evaluation
    logger.info("Running inference ...")
    results = run_evaluation(model, eval_loader, device, cfg, volume_ids)
    logger.info(f"Processed {len(results)} slices")

    if not results:
        logger.error("No results collected!")
        sys.exit(1)

    # Compute metrics
    logger.info("Computing metrics ...")
    overall_metrics = compute_all_metrics(results, cfg)
    per_dataset_metrics = compute_per_dataset_metrics(results, cfg)

    # Print results
    logger.info("\n=== Overall Metrics ===")
    logger.info(f"  Patch Accuracy:  {overall_metrics['accuracy']:.4f}")
    logger.info(f"  Patch Precision: {overall_metrics['precision']:.4f}")
    logger.info(f"  Patch Recall:    {overall_metrics['recall']:.4f}")
    logger.info(f"  Patch F1:        {overall_metrics['f1']:.4f}")
    logger.info(f"  Patch AUROC:     {overall_metrics['auroc']:.4f}")
    logger.info(f"  Region Recall:   {overall_metrics['region_recall']:.4f}")
    logger.info(f"  Region Precision:{overall_metrics['region_precision']:.4f}")
    logger.info(f"  Region F1:       {overall_metrics['region_f1']:.4f}")
    logger.info(f"  FP Rate (neg):   {overall_metrics['false_positive_rate']:.4f}")
    logger.info(f"  Patches: {overall_metrics.get('n_positive_patches', 0)} pos, "
                f"{overall_metrics.get('n_negative_patches', 0)} neg")

    logger.info("\n=== Per-Dataset Metrics ===")
    for ds_name, ds_metrics in per_dataset_metrics.items():
        n_sl = ds_metrics.get("n_slices", 0)
        n_pos = ds_metrics.get("n_positive_slices", 0)
        logger.info(
            f"  {ds_name} ({n_sl} slices, {n_pos} pos): "
            f"F1={ds_metrics['f1']:.4f} "
            f"AUROC={ds_metrics['auroc']:.4f} "
            f"RegRecall={ds_metrics['region_recall']:.4f}"
        )

    # Save results
    # Convert non-serializable items
    serializable_metrics = {
        "split": args.split,
        "checkpoint": args.checkpoint,
        "n_slices": len(results),
        "n_positive_slices": sum(1 for r in results if r["has_lesion"]),
        "overall": {k: v for k, v in overall_metrics.items()
                    if isinstance(v, (int, float))},
        "per_dataset": {
            ds: {k: v for k, v in m.items() if isinstance(v, (int, float))}
            for ds, m in per_dataset_metrics.items()
        },
    }

    results_path = output_dir / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump(serializable_metrics, f, indent=2)
    logger.info(f"\nResults saved to {results_path}")

    # Save per-slice predictions
    per_slice = []
    for r in results:
        per_slice.append({
            "sample_id": r["sample_id"],
            "slice_idx": r["slice_idx"],
            "dataset": r["dataset"],
            "has_lesion": r["has_lesion"],
            "pred_regions": r["pred_regions"],
            "gt_bboxes_normalized": r["gt_bboxes_normalized"],
        })

    predictions_path = output_dir / "per_slice_predictions.json"
    with open(predictions_path, "w") as f:
        json.dump(per_slice, f, indent=2)
    logger.info(f"Per-slice predictions saved to {predictions_path}")

    # Visualization
    if args.visualize > 0:
        logger.info(f"\nGenerating {args.visualize} visualizations ...")
        vis_dir = output_dir / "visualizations"
        vis_dir.mkdir(exist_ok=True)

        # Select positive slices for visualization (most informative)
        positive_results = [r for r in results if r["has_lesion"]]
        if len(positive_results) < args.visualize:
            # Add some negative slices too
            negative_results = [r for r in results if not r["has_lesion"]]
            rng = random.Random(42)
            rng.shuffle(negative_results)
            vis_results = positive_results + negative_results[
                :args.visualize - len(positive_results)
            ]
        else:
            rng = random.Random(42)
            vis_results = rng.sample(positive_results, args.visualize)

        for idx, r in enumerate(vis_results):
            save_path = vis_dir / f"{idx:03d}_{r['dataset']}_{r['sample_id']}_s{r['slice_idx']}.png"
            visualize_patch_heatmap(
                image=r["image"],
                patch_scores=r["patch_scores_2d"],
                patch_labels=r["patch_labels"],
                gt_bboxes_normalized=r["gt_bboxes_normalized"],
                pred_regions=r["pred_regions"],
                save_path=str(save_path),
                img_size=cfg.backbone.img_size,
            )

        logger.info(f"Visualizations saved to {vis_dir}")

    logger.info("\nEvaluation complete.")


if __name__ == "__main__":
    main()
