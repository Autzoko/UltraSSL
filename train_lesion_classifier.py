#!/usr/bin/env python3
"""
Train a patch-level lesion classifier on frozen DINO features.

Uses bbox annotations to supervise patch-level predictions, then aggregates
via MIL (multiple instance learning) pooling for image-level has_lesion
classification. Designed to reduce false positives on negative slices.

Supports multi-GPU training via torchrun:
    torchrun --nproc_per_node=4 train_lesion_classifier.py \
        --config config/lesion_classifier.yaml

Single-GPU:
    python train_lesion_classifier.py --config config/lesion_classifier.yaml
"""

import argparse
import json
import logging
import math
import os
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP

# Ensure project root is on path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "dinov2"))
# Allow xFormers if available; set XFORMERS_DISABLED=1 in env to disable
# os.environ.setdefault("XFORMERS_DISABLED", "1")

from ultrassl.models.patch_classifier import PatchLesionClassifier, assign_patch_labels
from ultrassl.data.wds_labeled_dataset import build_labeled_wds_dataset

logger = logging.getLogger("ultrassl")


# ── Distributed helpers ──────────────────────────────────────────────

def setup_distributed():
    """Initialize DDP if launched via torchrun, otherwise single-GPU."""
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size
    return 0, 0, 1


def is_main_process(rank):
    return rank == 0


def cleanup_distributed(world_size):
    if world_size > 1:
        dist.destroy_process_group()


# ── Config ────────────────────────────────────────────────────────────

def load_config(config_path, cli_opts=None):
    cfg = OmegaConf.load(config_path)
    if cli_opts:
        cli_cfg = OmegaConf.from_dotlist(cli_opts)
        cfg = OmegaConf.merge(cfg, cli_cfg)
    return cfg


# ── Volume-aware train/val split ─────────────────────────────────────

def load_volume_split(shard_dir, val_split=0.15, seed=42, rank=0, world_size=1):
    """Split volumes into train/val sets, stratified by dataset.

    Scans shard annotations on rank 0 only, then broadcasts to all ranks.

    Returns:
        train_volumes: set of sample_id strings for training
        val_volumes: set of sample_id strings for validation
    """
    import glob
    import tarfile
    from collections import defaultdict

    train_volumes = None
    val_volumes = None

    if rank == 0:
        # Scan shards to collect volume info
        volume_info = {}

        shard_files = sorted(glob.glob(os.path.join(shard_dir, "*/shard-*.tar")))
        if not shard_files:
            logger.warning(f"No shard files found under {shard_dir}. "
                           "Using all data for training (no validation).")
        else:
            for shard_path in shard_files:
                try:
                    with tarfile.open(shard_path, "r") as tar:
                        for member in tar.getmembers():
                            if not member.name.endswith(".json"):
                                continue
                            f = tar.extractfile(member)
                            if f is None:
                                continue
                            ann = json.loads(f.read().decode())
                            sid = ann.get("sample_id", "")
                            ds_name = ann.get("dataset", "unknown")
                            has_lesion = ann.get("has_lesion", 0)
                            if sid not in volume_info:
                                volume_info[sid] = {"dataset": ds_name, "n_positive": 0}
                            volume_info[sid]["n_positive"] += has_lesion
                except Exception as e:
                    logger.warning(f"Error reading {shard_path}: {e}")

            if volume_info:
                by_dataset = defaultdict(list)
                for sid, info in volume_info.items():
                    by_dataset[info["dataset"]].append(sid)

                rng = random.Random(seed)
                train_volumes = set()
                val_volumes = set()

                for ds_name in sorted(by_dataset.keys()):
                    ds_volumes = sorted(by_dataset[ds_name])
                    rng.shuffle(ds_volumes)
                    n_val = max(1, int(len(ds_volumes) * val_split))
                    val_volumes.update(ds_volumes[:n_val])
                    train_volumes.update(ds_volumes[n_val:])

                n_pos_train = sum(volume_info[v]["n_positive"] for v in train_volumes)
                n_pos_val = sum(volume_info[v]["n_positive"] for v in val_volumes)

                logger.info(f"Volume split: {len(train_volumes)} train, {len(val_volumes)} val")
                for ds_name in sorted(by_dataset.keys()):
                    n_train_ds = len([v for v in train_volumes if volume_info[v]["dataset"] == ds_name])
                    n_val_ds = len([v for v in val_volumes if volume_info[v]["dataset"] == ds_name])
                    logger.info(f"  {ds_name}: {n_train_ds} train, {n_val_ds} val")
                logger.info(f"  Train positive slices: {n_pos_train}, Val positive slices: {n_pos_val}")
            else:
                logger.warning("No volume info found. Using all data for training.")

    # Broadcast split to all ranks
    if world_size > 1:
        data = pickle.dumps((train_volumes, val_volumes))
        size = torch.tensor([len(data)], dtype=torch.long, device="cuda")
        dist.broadcast(size, src=0)
        if rank != 0:
            data = bytes(size.item())
        buf = torch.ByteTensor(list(data)).cuda()
        if rank != 0:
            buf = torch.zeros(size.item(), dtype=torch.uint8, device="cuda")
        dist.broadcast(buf, src=0)
        if rank != 0:
            train_volumes, val_volumes = pickle.loads(buf.cpu().numpy().tobytes())

    return train_volumes, val_volumes


# ── Training ──────────────────────────────────────────────────────────

def train_classifier(cfg):
    """Main training loop for the patch-level lesion classifier."""
    # DDP setup
    rank, local_rank, world_size = setup_distributed()
    is_main = is_main_process(rank)

    if world_size > 1:
        device = torch.device(f"cuda:{local_rank}")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if is_main:
        logger.info(f"Using device: {device}, world_size: {world_size}")

    # Seed
    seed = cfg.train.get("seed", 42)
    torch.manual_seed(seed + rank)
    random.seed(seed + rank)
    np.random.seed(seed + rank)

    # Output dir
    output_dir = cfg.train.output_dir
    if is_main:
        os.makedirs(output_dir, exist_ok=True)

    # Resolve shard_dir
    shard_dir = cfg.data.shard_dir
    if not os.path.isabs(shard_dir):
        shard_dir = os.path.join(str(project_root), shard_dir)

    # Build model
    backbone_ckpt = cfg.backbone.checkpoint
    if not os.path.isabs(backbone_ckpt):
        backbone_ckpt = os.path.join(str(project_root), backbone_ckpt)

    model = PatchLesionClassifier(
        backbone_checkpoint=backbone_ckpt,
        arch=cfg.backbone.arch,
        patch_size=cfg.backbone.patch_size,
        img_size=cfg.backbone.img_size,
        head_type=cfg.classifier.head_type,
        mil_type=cfg.classifier.mil_type,
        topk_ratio=cfg.classifier.get("topk_ratio", 0.1),
        mlp_hidden_dim=cfg.classifier.get("mlp_hidden_dim", 256),
    )
    model.to(device)

    # Wrap in DDP — only head params have gradients, backbone is frozen
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
        raw_model = model.module
    else:
        raw_model = model

    # Optimizer — only head parameters
    optimizer = torch.optim.AdamW(
        raw_model.get_trainable_params(),
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.get("weight_decay", 1e-4),
    )

    # LR scheduler with warmup
    total_epochs = cfg.optim.epochs
    warmup_epochs = cfg.optim.get("warmup_epochs", 3)

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Loss
    pos_weight = torch.tensor([cfg.loss.pos_weight], device=device)
    patch_bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
    image_bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    patch_loss_weight = cfg.loss.patch_loss_weight
    image_loss_weight = cfg.loss.image_loss_weight
    ignore_margin = cfg.loss.get("ignore_margin", 0.0)

    # Grid info
    grid_h = cfg.backbone.img_size // cfg.backbone.patch_size
    grid_w = grid_h

    # Data — build train loader
    datasets_filter = cfg.data.get("datasets", None)
    if datasets_filter is not None:
        datasets_filter = list(datasets_filter)

    # Volume-aware split (rank 0 scans, broadcasts to all)
    train_volumes, val_volumes = load_volume_split(
        shard_dir, val_split=cfg.eval.get("val_split", 0.15),
        seed=seed, rank=rank, world_size=world_size)

    # Estimate epoch length from index.json
    index_path = os.path.join(shard_dir, "index.json")
    total_slices = 0
    if os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)
        for ds_stats in index.values():
            total_slices += ds_stats.get("n_slices_total", 0)
    epoch_length = max(100, total_slices // (cfg.data.batch_size * world_size))

    _, train_loader = build_labeled_wds_dataset(
        shard_dir=shard_dir,
        mode="detection",
        epoch_length=epoch_length,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        shuffle_buffer=cfg.data.get("shuffle_buffer", 5000),
        datasets=datasets_filter,
        oversample_positive=cfg.data.get("oversample_positive", 0.0),
        img_size=cfg.backbone.img_size,
    )

    log_every = cfg.train.get("log_every", 50)
    best_val_auroc = 0.0

    if is_main:
        logger.info(f"Training for {total_epochs} epochs, ~{epoch_length} batches/epoch")
        logger.info(f"Batch size: {cfg.data.batch_size}/gpu x {world_size} GPUs = "
                     f"{cfg.data.batch_size * world_size} effective")
        logger.info(f"Patch loss weight: {patch_loss_weight}, Image loss weight: {image_loss_weight}")
        logger.info(f"Pos weight: {cfg.loss.pos_weight}, Ignore margin: {ignore_margin}")

    # Training loop
    metrics_file = os.path.join(output_dir, "training_metrics.jsonl")

    for epoch in range(total_epochs):
        model.train()
        epoch_metrics = {
            "patch_loss": 0.0, "image_loss": 0.0, "total_loss": 0.0,
            "n_batches": 0, "n_positive_slices": 0, "n_negative_slices": 0,
        }

        for batch_idx, (images, annotations) in enumerate(train_loader):
            images = images.to(device)

            # Forward
            patch_logits, image_logit = model(images)

            # Assign patch-level labels from bboxes
            patch_labels_list = []
            image_labels_list = []
            keep_mask = []

            for ann in annotations:
                # Volume split filtering
                if train_volumes is not None:
                    sid = ann.get("sample_id", "")
                    if sid in val_volumes:
                        keep_mask.append(False)
                        continue
                keep_mask.append(True)

                pl = assign_patch_labels(
                    bboxes=ann.get("bboxes", []),
                    img_width=ann["image_width"],
                    img_height=ann["image_height"],
                    grid_h=grid_h,
                    grid_w=grid_w,
                    patch_size=cfg.backbone.patch_size,
                    img_size=cfg.backbone.img_size,
                    ignore_margin=ignore_margin,
                )
                patch_labels_list.append(pl)
                image_labels_list.append(float(ann.get("has_lesion", 0)))

            # Skip if all samples are from val set
            if not any(keep_mask):
                continue

            # Filter batch
            if not all(keep_mask):
                keep_idx = [i for i, k in enumerate(keep_mask) if k]
                keep_t = torch.tensor(keep_idx, device=device)
                patch_logits = patch_logits[keep_t]
                image_logit = image_logit[keep_t]

            patch_labels = torch.stack(patch_labels_list).to(device)
            image_labels = torch.tensor(image_labels_list, device=device).unsqueeze(1)

            # Patch loss (with ignore masking where label == -1)
            valid_mask = (patch_labels >= 0)
            patch_targets = patch_labels.clamp(min=0)
            patch_loss_raw = patch_bce(patch_logits.squeeze(-1), patch_targets)
            patch_loss = (patch_loss_raw * valid_mask).sum() / valid_mask.sum().clamp(min=1)

            # Image loss
            image_loss = image_bce(image_logit, image_labels)

            # Combined loss
            loss = patch_loss_weight * patch_loss + image_loss_weight * image_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track metrics
            epoch_metrics["patch_loss"] += patch_loss.item()
            epoch_metrics["image_loss"] += image_loss.item()
            epoch_metrics["total_loss"] += loss.item()
            epoch_metrics["n_batches"] += 1
            epoch_metrics["n_positive_slices"] += int(image_labels.sum().item())
            epoch_metrics["n_negative_slices"] += int((1 - image_labels).sum().item())

            if is_main and (batch_idx + 1) % log_every == 0:
                avg_loss = epoch_metrics["total_loss"] / epoch_metrics["n_batches"]
                avg_ploss = epoch_metrics["patch_loss"] / epoch_metrics["n_batches"]
                avg_iloss = epoch_metrics["image_loss"] / epoch_metrics["n_batches"]
                logger.info(
                    f"[epoch {epoch+1}/{total_epochs}] [batch {batch_idx+1}] "
                    f"loss={avg_loss:.4f} (patch={avg_ploss:.4f}, image={avg_iloss:.4f}) "
                    f"lr={optimizer.param_groups[0]['lr']:.6f}"
                )

        scheduler.step()

        # Epoch summary (rank 0 only)
        if is_main:
            n = max(1, epoch_metrics["n_batches"])
            logger.info(
                f"Epoch {epoch+1}/{total_epochs}: "
                f"loss={epoch_metrics['total_loss']/n:.4f} "
                f"(patch={epoch_metrics['patch_loss']/n:.4f}, "
                f"image={epoch_metrics['image_loss']/n:.4f}) "
                f"pos_slices={epoch_metrics['n_positive_slices']}, "
                f"neg_slices={epoch_metrics['n_negative_slices']}"
            )

            # Log to file
            with open(metrics_file, "a") as f:
                f.write(json.dumps({
                    "epoch": epoch + 1,
                    "patch_loss": epoch_metrics["patch_loss"] / n,
                    "image_loss": epoch_metrics["image_loss"] / n,
                    "total_loss": epoch_metrics["total_loss"] / n,
                    "lr": optimizer.param_groups[0]["lr"],
                }) + "\n")

        # Validation (rank 0 only)
        if is_main and val_volumes is not None:
            eval_model = raw_model
            val_metrics = evaluate(eval_model, train_loader, device, val_volumes,
                                   cfg, grid_h, grid_w, ignore_margin)
            logger.info(
                f"  Val: acc={val_metrics['accuracy']:.4f} "
                f"sens={val_metrics['sensitivity']:.4f} "
                f"spec={val_metrics['specificity']:.4f} "
                f"auroc={val_metrics['auroc']:.4f}"
            )

            if val_metrics["auroc"] > best_val_auroc:
                best_val_auroc = val_metrics["auroc"]
                save_path = os.path.join(output_dir, "best_model.pth")
                torch.save({
                    "head": raw_model.head.state_dict(),
                    "attn_pool": raw_model.attn_pool.state_dict() if hasattr(raw_model, "attn_pool") else None,
                    "epoch": epoch + 1,
                    "val_auroc": best_val_auroc,
                    "config": OmegaConf.to_container(cfg),
                }, save_path)
                logger.info(f"  New best model (AUROC={best_val_auroc:.4f}) saved to {save_path}")

        # Periodic checkpoint (rank 0 only)
        if is_main and ((epoch + 1) % 5 == 0 or (epoch + 1) == total_epochs):
            save_path = os.path.join(output_dir, f"checkpoint_epoch{epoch+1:03d}.pth")
            torch.save({
                "head": raw_model.head.state_dict(),
                "attn_pool": raw_model.attn_pool.state_dict() if hasattr(raw_model, "attn_pool") else None,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch + 1,
                "config": OmegaConf.to_container(cfg),
            }, save_path)

        # Sync all ranks before next epoch
        if world_size > 1:
            dist.barrier()

    if is_main:
        logger.info(f"Training complete. Best val AUROC: {best_val_auroc:.4f}")
        logger.info(f"Outputs saved to: {output_dir}")

    cleanup_distributed(world_size)


# ── Evaluation ────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device, val_volumes, cfg, grid_h, grid_w, ignore_margin):
    """Evaluate on validation volumes (filtered from the shared loader)."""
    model.eval()

    all_preds = []
    all_labels = []
    max_val_batches = 200

    n_batches = 0
    for images, annotations in loader:
        if n_batches >= max_val_batches:
            break

        # Filter to val volumes only
        val_indices = []
        val_image_labels = []
        for i, ann in enumerate(annotations):
            sid = ann.get("sample_id", "")
            if sid in val_volumes:
                val_indices.append(i)
                val_image_labels.append(float(ann.get("has_lesion", 0)))

        if not val_indices:
            n_batches += 1
            continue

        val_idx_t = torch.tensor(val_indices, device=device)
        val_images = images[val_idx_t].to(device)

        _, image_logit = model(val_images)
        probs = torch.sigmoid(image_logit).cpu().squeeze(-1)

        all_preds.extend(probs.tolist())
        all_labels.extend(val_image_labels)
        n_batches += 1

    model.train()

    if len(all_preds) < 2 or sum(all_labels) == 0 or sum(all_labels) == len(all_labels):
        return {"accuracy": 0.0, "sensitivity": 0.0, "specificity": 0.0, "auroc": 0.5}

    # Compute metrics
    preds = np.array(all_preds)
    labels = np.array(all_labels)
    binary_preds = (preds >= 0.5).astype(float)

    accuracy = (binary_preds == labels).mean()
    tp = ((binary_preds == 1) & (labels == 1)).sum()
    fn = ((binary_preds == 0) & (labels == 1)).sum()
    tn = ((binary_preds == 0) & (labels == 0)).sum()
    fp = ((binary_preds == 1) & (labels == 0)).sum()

    sensitivity = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)

    # AUROC
    try:
        from sklearn.metrics import roc_auc_score
        auroc = roc_auc_score(labels, preds)
    except Exception:
        auroc = _manual_auroc(labels, preds)

    return {
        "accuracy": float(accuracy),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "auroc": float(auroc),
        "n_samples": len(labels),
        "n_positive": int(labels.sum()),
    }


def _manual_auroc(labels, preds):
    """Simple AUROC computation without sklearn."""
    pos = preds[labels == 1]
    neg = preds[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    n_correct = 0
    n_total = len(pos) * len(neg)
    for p in pos:
        n_correct += (p > neg).sum() + 0.5 * (p == neg).sum()
    return float(n_correct / n_total)


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train patch-level lesion classifier on DINO features")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("opts", nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args()

    cli_opts = [o for o in args.opts if o and o != "--"]
    cfg = load_config(args.config, cli_opts)

    # Setup logging (only rank 0 writes to file)
    rank = int(os.environ.get("RANK", 0))
    os.makedirs(cfg.train.output_dir, exist_ok=True)
    handlers = [logging.StreamHandler()]
    if rank == 0:
        handlers.append(logging.FileHandler(
            os.path.join(cfg.train.output_dir, "train.log")))
    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=handlers,
    )

    train_classifier(cfg)


if __name__ == "__main__":
    main()
