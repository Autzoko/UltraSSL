#!/usr/bin/env python3
"""
Train patch-level lesion detector with three-region labeling.

No DDP — backbone is frozen and the MLP head is small (~197K params).
Each GPU trains independently on its own WebDataset shard subset.
Validation metrics are gathered across ranks via gloo.

Uses three-region patch labeling: shrunk bbox core (positive), expanded
bbox ring (ignore), outside (negative). Focal loss with hard negative
mining handles extreme class imbalance at the patch level.

Usage:
    # Multi-GPU training
    torchrun --nproc_per_node=4 train_patch_detector.py \
        --config config/patch_detector.yaml

    # Single GPU
    python train_patch_detector.py --config config/patch_detector.yaml

    # Override config values
    python train_patch_detector.py --config config/patch_detector.yaml \
        optim.lr=5e-4 optim.epochs=30
"""

import argparse
import json
import logging
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
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
    PatchFocalLoss,
    assign_three_region_patch_labels,
    build_patch_detection_pipeline,
    compute_patch_metrics,
)

logger = logging.getLogger("patch_detector")


# ============================================================================
# Distributed helpers
# ============================================================================


def setup_distributed():
    """Initialize gloo for lightweight communication (no DDP gradient sync)."""
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group(backend="gloo")
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
                    device, epoch, cfg, train_volume_ids):
    """Run one training epoch with volume filtering and three-region labeling."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    n_pos_slices = 0
    n_neg_slices = 0

    grid_h = model.grid_h
    grid_w = model.grid_w
    patch_size = model.patch_size
    img_size = model.img_size

    for batch_idx, (images, annotations) in enumerate(loader):
        # Volume filtering: keep only training volumes
        keep_mask = []
        for ann in annotations:
            sid = ann.get("sample_id", "")
            keep_mask.append(sid in train_volume_ids)

        if not any(keep_mask):
            continue

        keep_idx = [i for i, k in enumerate(keep_mask) if k]
        images = images[torch.tensor(keep_idx)].to(device, non_blocking=True)
        filtered_anns = [annotations[i] for i in keep_idx]

        # Compute three-region patch labels
        labels_list = []
        for ann in filtered_anns:
            labels = assign_three_region_patch_labels(
                bboxes=ann.get("bboxes", []),
                img_width=ann["image_width"],
                img_height=ann["image_height"],
                grid_h=grid_h,
                grid_w=grid_w,
                patch_size=patch_size,
                img_size=img_size,
                shrink_ratio=cfg.labeling.shrink_ratio,
                expand_ratio=cfg.labeling.expand_ratio,
                pos_iou_thresh=cfg.labeling.positive_iou_thresh,
                ignore_iou_thresh=cfg.labeling.ignore_iou_thresh,
            )
            labels_list.append(labels)

            if ann.get("has_lesion", 0):
                n_pos_slices += 1
            else:
                n_neg_slices += 1

        patch_labels = torch.stack(labels_list).to(device)  # (B, 256)

        with torch.amp.autocast("cuda", enabled=cfg.train.use_amp):
            patch_logits, _ = model(images)  # (B, 256)
            loss = criterion(patch_logits, patch_labels)

        optimizer.zero_grad()
        scaler.scale(loss).backward()

        if cfg.optim.get("max_grad_norm", 0) > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.get_trainable_params(), cfg.optim.max_grad_norm
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

    return {
        "loss": total_loss / max(n_batches, 1),
        "n_pos_slices": n_pos_slices,
        "n_neg_slices": n_neg_slices,
    }


# ============================================================================
# Evaluation
# ============================================================================


@torch.no_grad()
def evaluate(model, loader, device, cfg, val_volume_ids):
    """Evaluate patch-level metrics on validation data."""
    model.eval()

    grid_h = model.grid_h
    grid_w = model.grid_w
    patch_size = model.patch_size
    img_size = model.img_size

    all_logits = []
    all_labels = []
    total_loss = 0.0
    n_batches = 0

    criterion = PatchFocalLoss(
        alpha=cfg.loss.alpha,
        gamma=cfg.loss.gamma,
        neg_subsample_ratio=0.0,  # no subsampling for val
    )

    for images, annotations in loader:
        # Volume filtering
        keep_mask = []
        for ann in annotations:
            sid = ann.get("sample_id", "")
            keep_mask.append(sid in val_volume_ids)

        if not any(keep_mask):
            continue

        keep_idx = [i for i, k in enumerate(keep_mask) if k]
        images = images[torch.tensor(keep_idx)].to(device, non_blocking=True)
        filtered_anns = [annotations[i] for i in keep_idx]

        # Compute labels
        labels_list = []
        for ann in filtered_anns:
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
            labels_list.append(labels)

        patch_labels = torch.stack(labels_list).to(device)

        with torch.amp.autocast("cuda", enabled=cfg.train.use_amp):
            patch_logits, _ = model(images)
            loss = criterion(patch_logits, patch_labels)

        total_loss += loss.item()
        n_batches += 1

        all_logits.append(patch_logits.cpu())
        all_labels.append(patch_labels.cpu())

    # Flatten across batches
    if all_logits:
        flat_logits = [row for batch in all_logits for row in batch]
        flat_labels = [row for batch in all_labels for row in batch]
    else:
        flat_logits, flat_labels = [], []

    # Gather across GPUs
    if dist.is_initialized():
        gathered = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, (flat_logits, flat_labels))
        all_l, all_lb = [], []
        for g in gathered:
            all_l.extend(g[0])
            all_lb.extend(g[1])
        flat_logits, flat_labels = all_l, all_lb

    metrics = compute_patch_metrics(flat_logits, flat_labels)
    metrics["val_loss"] = total_loss / max(n_batches, 1)
    return metrics


# ============================================================================
# Checkpointing
# ============================================================================


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, path):
    """Save head checkpoint (backbone is frozen, not saved)."""
    state = {"epoch": epoch, "metrics": metrics}
    state["head_state_dict"] = model.head.state_dict()
    state["optimizer_state_dict"] = optimizer.state_dict()
    state["scheduler_state_dict"] = scheduler.state_dict()

    # If backbone blocks are unfrozen, save them too
    unfrozen_blocks = {}
    for i, block in enumerate(model.backbone.blocks):
        if any(p.requires_grad for p in block.parameters()):
            unfrozen_blocks[f"block_{i}"] = block.state_dict()
    if unfrozen_blocks:
        state["unfrozen_blocks"] = unfrozen_blocks

    torch.save(state, path)


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Train patch-level lesion detector"
    )
    parser.add_argument("--config", required=True, help="Path to patch_detector.yaml")
    args, overrides = parser.parse_known_args()

    # Config
    cfg = OmegaConf.load(args.config)
    if overrides:
        cli_conf = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(cfg, cli_conf)

    # Distributed
    rank, local_rank, world_size = setup_distributed()
    distributed = world_size > 1
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # Logging
    logging.basicConfig(
        level=logging.INFO if is_main_process() else logging.WARNING,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if is_main_process():
        logger.info("=" * 60)
        logger.info("Patch-Level Lesion Detector — Training")
        logger.info("=" * 60)
        logger.info(f"World size: {world_size}, Device: {device}")

    # Seed (per-rank for data diversity)
    seed = cfg.train.seed
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)

    # Load split metadata
    split_dir = cfg.data.split_dir
    train_split_path = os.path.join(split_dir, "train_split.json")
    val_split_path = os.path.join(split_dir, "val_split.json")

    if not os.path.exists(train_split_path):
        logger.error(
            f"Split file not found: {train_split_path}\n"
            f"Run prepare_patch_data.py first."
        )
        sys.exit(1)

    with open(train_split_path) as f:
        train_split = json.load(f)
    with open(val_split_path) as f:
        val_split = json.load(f)

    train_volume_ids = set(train_split["volume_ids"])
    val_volume_ids = set(val_split["volume_ids"])

    if is_main_process():
        logger.info(
            f"Loaded splits: {len(train_volume_ids)} train, "
            f"{len(val_volume_ids)} val volumes"
        )
        logger.info(f"  Train: {train_split['stats']['n_slices']} slices "
                     f"({train_split['stats']['n_positive_slices']} positive)")
        logger.info(f"  Val: {val_split['stats']['n_slices']} slices "
                     f"({val_split['stats']['n_positive_slices']} positive)")

    # Model
    model = PatchDetector(
        backbone_checkpoint=cfg.backbone.checkpoint,
        arch=cfg.backbone.arch,
        patch_size=cfg.backbone.patch_size,
        img_size=cfg.backbone.img_size,
        head_hidden_dim=cfg.model.head_hidden_dim,
        unfreeze_last_n=cfg.model.unfreeze_last_n,
    ).to(device)

    # Data loaders
    datasets = cfg.data.get("datasets", None)
    if datasets is not None:
        datasets = list(datasets)

    if is_main_process():
        logger.info("Building training data pipeline ...")

    _, train_loader = build_patch_detection_pipeline(
        shard_dir=cfg.data.shard_dir,
        datasets=datasets,
        img_size=cfg.backbone.img_size,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        shuffle_buffer=cfg.data.shuffle_buffer,
        epoch_length=cfg.data.epoch_length,
        oversample_positive=cfg.data.oversample_positive,
        balance_datasets=True,
        volume_ids=train_volume_ids,
    )

    if is_main_process():
        logger.info("Building validation data pipeline ...")

    _, val_loader = build_patch_detection_pipeline(
        shard_dir=cfg.data.shard_dir,
        datasets=datasets,
        img_size=cfg.backbone.img_size,
        batch_size=cfg.data.batch_size,
        num_workers=max(cfg.data.num_workers // 2, 1),
        shuffle_buffer=1000,
        epoch_length=cfg.data.get("val_epoch_length", 200),
        oversample_positive=0.0,  # no oversampling for validation
        balance_datasets=False,
        volume_ids=val_volume_ids,
    )

    # Loss
    criterion = PatchFocalLoss(
        alpha=cfg.loss.alpha,
        gamma=cfg.loss.gamma,
        neg_subsample_ratio=cfg.loss.neg_subsample_ratio,
    )

    # Optimizer
    trainable_params = model.get_trainable_params()
    n_trainable = sum(p.numel() for p in trainable_params) / 1e6
    if is_main_process():
        logger.info(f"Trainable parameters: {n_trainable:.2f}M")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
    )

    steps_per_epoch = cfg.data.epoch_length
    scheduler = build_lr_scheduler(
        optimizer, cfg.optim.warmup_epochs, cfg.optim.epochs, steps_per_epoch,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.train.use_amp)

    # Output dir
    output_dir = Path(cfg.train.output_dir)
    if is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, output_dir / "config.yaml")
        logger.info(f"Output: {output_dir}")

    # Training loop
    best_f1 = -1.0
    log_path = output_dir / "training_log.jsonl"

    for epoch in range(1, cfg.optim.epochs + 1):
        if is_main_process():
            logger.info(f"Epoch {epoch}/{cfg.optim.epochs}")

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            criterion, device, epoch, cfg, train_volume_ids,
        )

        val_metrics = evaluate(model, val_loader, device, cfg, val_volume_ids)

        if is_main_process():
            logger.info(
                f"  Train — loss={train_metrics['loss']:.4f} "
                f"pos={train_metrics['n_pos_slices']} neg={train_metrics['n_neg_slices']}"
            )
            logger.info(
                f"  Val — loss={val_metrics['val_loss']:.4f} "
                f"acc={val_metrics['accuracy']:.4f} "
                f"prec={val_metrics['precision']:.4f} "
                f"rec={val_metrics['recall']:.4f} "
                f"f1={val_metrics['f1']:.4f} "
                f"auroc={val_metrics['auroc']:.4f}"
            )
            logger.info(
                f"  Val patches — pos={val_metrics.get('n_positive_patches', 0)} "
                f"neg={val_metrics.get('n_negative_patches', 0)}"
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

            # Save best model (by F1)
            f1 = val_metrics["f1"]
            if f1 > best_f1:
                best_f1 = f1
                save_checkpoint(
                    model, optimizer, scheduler, epoch,
                    val_metrics, output_dir / "best_model.pth",
                )
                logger.info(f"  ** New best F1: {best_f1:.4f} — saved **")

            # Periodic checkpoint
            save_period = cfg.train.get("save_period", 5)
            if epoch % save_period == 0:
                save_checkpoint(
                    model, optimizer, scheduler, epoch,
                    val_metrics, output_dir / f"checkpoint_epoch{epoch}.pth",
                )

        # Sync between epochs
        if distributed:
            dist.barrier()

    # Done
    if is_main_process():
        logger.info("=" * 60)
        logger.info(f"Training complete. Best F1: {best_f1:.4f}")
        logger.info(f"Checkpoints: {output_dir}")
        logger.info("=" * 60)

    cleanup_distributed()


if __name__ == "__main__":
    main()
