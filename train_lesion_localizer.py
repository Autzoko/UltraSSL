"""
Train lesion center heatmap localizer with multi-GPU DDP + AMP.

Predicts Gaussian center heatmaps from coronal ultrasound slices using
frozen DINO ViT-B/14 patch tokens + lightweight conv decoder.

Output center points serve as point prompts for AutoSAMUS segmentation.

Usage:
    # Multi-GPU training
    torchrun --nproc_per_node=4 train_lesion_localizer.py \
        --config config/lesion_localizer.yaml

    # Single GPU
    python train_lesion_localizer.py \
        --config config/lesion_localizer.yaml

    # Override config values
    torchrun --nproc_per_node=4 train_lesion_localizer.py \
        --config config/lesion_localizer.yaml \
        optim.lr=5e-4 optim.epochs=60

    # Inference mode: export center points as JSON
    python train_lesion_localizer.py \
        --config config/lesion_localizer.yaml \
        --inference --checkpoint outputs/lesion_localizer/best_model.pth \
        --datasets BIrads Class3 Class4 ABUS
"""

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP

# Ensure project root and dinov2 are importable
_project_root = Path(__file__).resolve().parent
_dinov2_root = _project_root / "dinov2"
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
if str(_dinov2_root) not in sys.path:
    sys.path.insert(0, str(_dinov2_root))

from ultrassl.lesion_localizer import (
    CenterNetFocalLoss,
    LesionLocalizer,
    build_heatmap_wds_pipeline,
    compute_localization_metrics,
    detect_peaks,
)

logger = logging.getLogger("lesion_localizer")


# ============================================================================
# Distributed helpers
# ============================================================================


def setup_distributed():
    """Initialize DDP if launched via torchrun, else single-GPU."""
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size
    return 0, 0, 1


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


# ============================================================================
# Learning rate schedule
# ============================================================================


def build_lr_scheduler(optimizer, warmup_epochs, total_epochs, steps_per_epoch):
    """Cosine decay with linear warmup."""
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = total_epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ============================================================================
# Training
# ============================================================================


def train_one_epoch(model, loader, optimizer, scheduler, scaler, criterion,
                    device, epoch, cfg):
    """Run one training epoch."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch_idx, (images, heatmaps, annotations) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        heatmaps = heatmaps.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=cfg.train.use_amp):
            pred_logits = model(images)  # (B, 1, 64, 64)
            loss = criterion(pred_logits, heatmaps)

        optimizer.zero_grad()
        scaler.scale(loss).backward()

        # Gradient clipping
        if cfg.optim.get("max_grad_norm", 0) > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.optim.max_grad_norm
            )

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()
        n_batches += 1

        if (
            is_main_process()
            and cfg.train.log_every > 0
            and (batch_idx + 1) % cfg.train.log_every == 0
        ):
            lr = optimizer.param_groups[0]["lr"]
            logger.info(
                f"  [Epoch {epoch}][{batch_idx + 1}] "
                f"loss={loss.item():.4f} lr={lr:.2e}"
            )

    return {"loss": total_loss / max(n_batches, 1)}


# ============================================================================
# Evaluation
# ============================================================================


@torch.no_grad()
def evaluate(model, loader, device, cfg):
    """Evaluate on validation data.

    Gathers predictions from all GPUs and computes localization metrics.
    """
    model.eval()
    threshold = cfg.eval.get("threshold", 0.3)
    nms_kernel = cfg.eval.get("nms_kernel", 3)
    max_det = cfg.eval.get("max_detections", 5)

    local_metrics = {
        "n_pos": 0,
        "n_hits": 0,
        "n_gt_bboxes": 0,
        "n_gt_recalled": 0,
        "total_distance": 0.0,
        "n_neg": 0,
        "n_false_pos": 0,
        "total_loss": 0.0,
        "n_batches": 0,
    }

    criterion = CenterNetFocalLoss(
        alpha=cfg.loss.get("alpha", 2.0),
        beta=cfg.loss.get("beta", 4.0),
    )

    for images, heatmaps, annotations in loader:
        images = images.to(device, non_blocking=True)
        heatmaps = heatmaps.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=cfg.train.use_amp):
            pred_logits = model(images)
            loss = criterion(pred_logits, heatmaps)

        local_metrics["total_loss"] += loss.item()
        local_metrics["n_batches"] += 1

        # Compute per-sample metrics
        pred_hm = torch.sigmoid(pred_logits).cpu()
        batch_metrics = compute_localization_metrics(
            pred_hm, annotations, threshold, nms_kernel, max_det
        )

        B = images.shape[0]
        local_metrics["n_pos"] += batch_metrics["n_positive_slices"]
        local_metrics["n_neg"] += batch_metrics["n_negative_slices"]
        local_metrics["n_hits"] += int(
            batch_metrics["center_hit_rate"] * batch_metrics["n_positive_slices"]
        )
        local_metrics["n_gt_recalled"] += int(
            batch_metrics["top_k_recall"] * max(
                sum(len(a.get("bboxes_normalized", []))
                    for a in annotations if a.get("has_lesion", 0)), 1
            )
        )
        local_metrics["n_gt_bboxes"] += sum(
            len(a.get("bboxes_normalized", []))
            for a in annotations if a.get("has_lesion", 0)
        )
        local_metrics["total_distance"] += (
            batch_metrics["mean_center_distance"] * batch_metrics["n_positive_slices"]
        )
        local_metrics["n_false_pos"] += int(
            batch_metrics["false_positive_rate"] * batch_metrics["n_negative_slices"]
        )

    # Gather across GPUs
    if dist.is_initialized():
        gathered = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, local_metrics)
        # Sum across GPUs
        combined = {k: 0 for k in local_metrics}
        for g in gathered:
            for k in combined:
                combined[k] += g[k]
    else:
        combined = local_metrics

    n_pos = max(combined["n_pos"], 1)
    n_neg = max(combined["n_neg"], 1)
    n_gt = max(combined["n_gt_bboxes"], 1)

    return {
        "val_loss": combined["total_loss"] / max(combined["n_batches"], 1),
        "center_hit_rate": combined["n_hits"] / n_pos,
        "top_k_recall": combined["n_gt_recalled"] / n_gt,
        "mean_center_distance": combined["total_distance"] / n_pos,
        "false_positive_rate": combined["n_false_pos"] / n_neg,
        "n_positive_slices": combined["n_pos"],
        "n_negative_slices": combined["n_neg"],
    }


# ============================================================================
# Checkpointing
# ============================================================================


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, path):
    """Save head checkpoint (backbone is frozen, not saved)."""
    state = model.module if hasattr(model, "module") else model
    torch.save(
        {
            "epoch": epoch,
            "head_state_dict": state.head.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "metrics": metrics,
        },
        path,
    )


# ============================================================================
# Inference
# ============================================================================


@torch.no_grad()
def run_inference(model, cfg, device, datasets, output_path):
    """Run inference on all slices and export center points as JSON.

    Output format per slice:
        {
            "sample_id": str,
            "slice_idx": int,
            "dataset": str,
            "centers": [{"x": float, "y": float, "confidence": float,
                         "x_pixel": int, "y_pixel": int}, ...],
            "gt_bboxes": [[nx1, ny1, nx2, ny2], ...]
        }
    """
    model.eval()
    threshold = cfg.eval.get("threshold", 0.3)
    nms_kernel = cfg.eval.get("nms_kernel", 3)
    max_det = cfg.eval.get("max_detections", 5)
    img_size = cfg.backbone.img_size

    # Build inference loader (no oversampling, no shuffling)
    _, loader = build_heatmap_wds_pipeline(
        shard_dir=cfg.data.shard_dir,
        datasets=datasets,
        heatmap_size=cfg.head.get("heatmap_size", 64),
        img_size=img_size,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        shuffle_buffer=100,  # minimal shuffle for inference
        epoch_length=999999,  # process all data
        oversample_positive=0.0,  # no oversampling
        balance_datasets=False,
    )

    results = []
    n_processed = 0

    for images, heatmaps, annotations in loader:
        images = images.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=cfg.train.use_amp):
            pred_logits = model(images)

        pred_hm = torch.sigmoid(pred_logits).cpu()

        for i in range(images.shape[0]):
            ann = annotations[i]
            peaks = detect_peaks(
                pred_hm[i], threshold, nms_kernel, max_det
            )

            # Add pixel coordinates (in original image space)
            orig_w = ann.get("original_width", img_size)
            orig_h = ann.get("original_height", img_size)
            for p in peaks:
                p["x_pixel"] = int(p["x"] * orig_w)
                p["y_pixel"] = int(p["y"] * orig_h)

            results.append({
                "sample_id": ann.get("sample_id", ""),
                "slice_idx": ann.get("slice_idx", -1),
                "dataset": ann.get("dataset", ""),
                "centers": peaks,
                "gt_bboxes": ann.get("bboxes_normalized", []),
                "has_lesion": ann.get("has_lesion", 0),
            })

            n_processed += 1

    # Write results
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Inference complete: {n_processed} slices → {output_path}")

    # Summary stats
    n_with_det = sum(1 for r in results if r["centers"])
    n_pos = sum(1 for r in results if r["has_lesion"])
    logger.info(
        f"  {n_with_det}/{n_processed} slices with detections, "
        f"{n_pos} positive slices in data"
    )

    return results


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Train lesion center heatmap localizer",
    )
    parser.add_argument(
        "--config", required=True, help="Path to lesion_localizer.yaml",
    )
    parser.add_argument(
        "--datasets", nargs="+", default=None,
        help="Dataset folders to use (default: from config)",
    )
    parser.add_argument(
        "--inference", action="store_true",
        help="Run inference mode (export center points as JSON)",
    )
    parser.add_argument(
        "--checkpoint", default="",
        help="Path to head checkpoint for inference",
    )
    args, overrides = parser.parse_known_args()

    # ── Config ────────────────────────────────────────────────────────────
    cfg = OmegaConf.load(args.config)
    if overrides:
        cli_conf = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(cfg, cli_conf)

    datasets = args.datasets or cfg.data.get("datasets", None)

    # ── Distributed ───────────────────────────────────────────────────────
    rank, local_rank, world_size = setup_distributed()
    distributed = world_size > 1
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # ── Logging ───────────────────────────────────────────────────────────
    logging.basicConfig(
        level=logging.INFO if is_main_process() else logging.WARNING,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if is_main_process():
        logger.info("=" * 60)
        logger.info("Lesion Center Heatmap Localizer")
        logger.info("=" * 60)
        logger.info(f"Datasets: {datasets}")
        logger.info(f"Mode: {'inference' if args.inference else 'training'}")
        logger.info(f"World size: {world_size}, Device: {device}")

    # ── Seed ──────────────────────────────────────────────────────────────
    seed = cfg.train.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ── Model ─────────────────────────────────────────────────────────────
    model = LesionLocalizer(
        backbone_checkpoint=cfg.backbone.checkpoint,
        arch=cfg.backbone.arch,
        patch_size=cfg.backbone.patch_size,
        img_size=cfg.backbone.img_size,
        unfreeze_last_n=cfg.head.get("unfreeze_last_n", 0),
    ).to(device)

    # Load head checkpoint if provided
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.head.load_state_dict(ckpt["head_state_dict"])
        if is_main_process():
            logger.info(f"Loaded head checkpoint from {args.checkpoint} "
                        f"(epoch {ckpt.get('epoch', '?')})")

    # ── Inference mode ────────────────────────────────────────────────────
    if args.inference:
        output_path = os.path.join(
            cfg.train.output_dir, "inference_results.json"
        )
        run_inference(model, cfg, device, datasets, output_path)
        cleanup_distributed()
        return

    # ── DDP ───────────────────────────────────────────────────────────────
    if distributed:
        model = DDP(model, device_ids=[local_rank],
                    find_unused_parameters=False)

    # ── Data ──────────────────────────────────────────────────────────────
    heatmap_size = cfg.head.get("heatmap_size", 64)
    min_sigma = cfg.head.get("min_sigma", 1.5)

    if is_main_process():
        logger.info("Building training data pipeline ...")

    _, train_loader = build_heatmap_wds_pipeline(
        shard_dir=cfg.data.shard_dir,
        datasets=datasets,
        heatmap_size=heatmap_size,
        img_size=cfg.backbone.img_size,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        shuffle_buffer=cfg.data.get("shuffle_buffer", 5000),
        epoch_length=cfg.data.get("epoch_length", 1000),
        oversample_positive=cfg.data.get("oversample_positive", 3.0),
        balance_datasets=True,
        min_sigma=min_sigma,
    )

    if is_main_process():
        logger.info("Building validation data pipeline ...")

    _, val_loader = build_heatmap_wds_pipeline(
        shard_dir=cfg.data.shard_dir,
        datasets=datasets,
        heatmap_size=heatmap_size,
        img_size=cfg.backbone.img_size,
        batch_size=cfg.data.batch_size,
        num_workers=max(cfg.data.num_workers // 2, 1),
        shuffle_buffer=1000,
        epoch_length=cfg.data.get("val_epoch_length", 200),
        oversample_positive=0.0,  # no oversampling for validation
        balance_datasets=False,
        min_sigma=min_sigma,
    )

    # ── Loss ──────────────────────────────────────────────────────────────
    criterion = CenterNetFocalLoss(
        alpha=cfg.loss.get("alpha", 2.0),
        beta=cfg.loss.get("beta", 4.0),
    )

    # ── Optimizer ─────────────────────────────────────────────────────────
    raw_model = model.module if hasattr(model, "module") else model
    trainable_params = raw_model.get_trainable_params()

    n_trainable = sum(p.numel() for p in trainable_params) / 1e6
    if is_main_process():
        logger.info(f"Trainable parameters: {n_trainable:.2f}M")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
    )

    steps_per_epoch = cfg.data.get("epoch_length", 1000)
    scheduler = build_lr_scheduler(
        optimizer, cfg.optim.warmup_epochs, cfg.optim.epochs, steps_per_epoch,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.train.use_amp)

    # ── Output dir ────────────────────────────────────────────────────────
    output_dir = Path(cfg.train.output_dir)
    if is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, output_dir / "config.yaml")
        logger.info(f"Output: {output_dir}")

    # ── Training loop ─────────────────────────────────────────────────────
    best_hit_rate = -1.0
    log_path = output_dir / "training_log.jsonl"

    for epoch in range(1, cfg.optim.epochs + 1):
        if is_main_process():
            logger.info(f"Epoch {epoch}/{cfg.optim.epochs}")

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            criterion, device, epoch, cfg,
        )

        val_metrics = evaluate(model, val_loader, device, cfg)

        if is_main_process():
            logger.info(
                f"  Train — loss={train_metrics['loss']:.4f}"
            )
            logger.info(
                f"  Val — loss={val_metrics['val_loss']:.4f} "
                f"hit_rate={val_metrics['center_hit_rate']:.4f} "
                f"recall={val_metrics['top_k_recall']:.4f} "
                f"dist={val_metrics['mean_center_distance']:.4f} "
                f"fpr={val_metrics['false_positive_rate']:.4f}"
            )
            logger.info(
                f"  Val slices — pos={val_metrics['n_positive_slices']} "
                f"neg={val_metrics['n_negative_slices']}"
            )

            # Log to file
            log_entry = {
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
                "lr": optimizer.param_groups[0]["lr"],
            }
            with open(log_path, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

            # Save best model (by center_hit_rate)
            hit_rate = val_metrics["center_hit_rate"]
            if hit_rate > best_hit_rate:
                best_hit_rate = hit_rate
                save_checkpoint(
                    model, optimizer, scheduler, epoch,
                    val_metrics, output_dir / "best_model.pth",
                )
                logger.info(f"  ** New best hit_rate: {best_hit_rate:.4f} — saved **")

            # Periodic checkpoint
            if epoch % 5 == 0:
                save_checkpoint(
                    model, optimizer, scheduler, epoch,
                    val_metrics, output_dir / f"checkpoint_epoch{epoch}.pth",
                )

    # ── Done ──────────────────────────────────────────────────────────────
    if is_main_process():
        logger.info("=" * 60)
        logger.info(f"Training complete. Best center_hit_rate: {best_hit_rate:.4f}")
        logger.info(f"Checkpoints: {output_dir}")
        logger.info("=" * 60)

    cleanup_distributed()


if __name__ == "__main__":
    main()
