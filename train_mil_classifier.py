"""
Train joint volume-level MIL classifier (has_lesion + subtype).

Multi-GPU DDP training with AMP on frozen DINO ViT-B/14 CLS embeddings
aggregated via gated-attention or top-k MIL pooling.

Supports two data modes:
    - On-the-fly: load slice images from WebDataset shards, extract CLS
      tokens through the frozen backbone during each forward pass.
    - Cached: load pre-extracted CLS token embeddings from disk (faster,
      requires running extract_embeddings.py first).

Usage:
    # Multi-GPU, on-the-fly (Class3 + Class4 only)
    torchrun --nproc_per_node=4 train_mil_classifier.py \\
        --config config/volume_classifier.yaml \\
        --datasets Class3 Class4

    # Multi-GPU, cached embeddings (faster)
    torchrun --nproc_per_node=4 train_mil_classifier.py \\
        --config config/volume_classifier.yaml \\
        --datasets Class3 Class4 \\
        --cache-dir /scratch/ll5582/Data/Ultrasound/embedding_cache

    # Single GPU
    python train_mil_classifier.py \\
        --config config/volume_classifier.yaml \\
        --datasets Class3 Class4

    # Override config values from CLI
    python train_mil_classifier.py \\
        --config config/volume_classifier.yaml \\
        --datasets Class3 Class4 \\
        optim.lr=1e-4 optim.epochs=30
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
import torch.nn as nn
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler

# Ensure project root and dinov2 are importable
_project_root = Path(__file__).resolve().parent
_dinov2_root = _project_root / "dinov2"
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
if str(_dinov2_root) not in sys.path:
    sys.path.insert(0, str(_dinov2_root))

from ultrassl.mil_classifier import (
    CachedVolumeDataset,
    FocalLoss,
    JointVolumeClassifier,
    VolumeShardDataset,
    collate_volumes,
    compute_binary_metrics,
    compute_subtype_metrics,
    extract_cls_tokens,
    scan_shard_volumes,
    volume_train_val_split,
)
from ultrassl.models.backbone import build_backbone

logger = logging.getLogger("mil_classifier")


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


def train_one_epoch(
    backbone,
    classifier,
    loader,
    optimizer,
    scheduler,
    scaler,
    focal_criterion,
    ce_criterion,
    device,
    epoch,
    cfg,
    cached_mode,
):
    """Run one training epoch.

    Returns:
        dict with avg total_loss, binary_loss, subtype_loss.
    """
    classifier.train()
    total_loss = 0.0
    total_bin = 0.0
    total_sub = 0.0
    n_batches = 0

    binary_w = cfg.loss.binary_loss_weight
    subtype_w = cfg.loss.multiclass_loss_weight

    for batch_idx, batch in enumerate(loader):
        mask = batch["mask"].to(device)
        has_lesion = batch["has_lesion"].to(device).unsqueeze(1)  # (B, 1)
        subtype = batch["subtype"].to(device)  # (B,)

        # Get CLS tokens
        if cached_mode:
            cls_tokens = batch["cls_tokens"].to(device)
        else:
            cls_tokens = extract_cls_tokens(
                backbone, batch["images"], device, chunk_size=64,
            )

        # Forward + loss
        with torch.amp.autocast("cuda", enabled=cfg.train.use_amp):
            out = classifier(cls_tokens, mask)

            bin_loss = focal_criterion(out["binary_logit"], has_lesion)
            sub_loss = ce_criterion(out["subtype_logits"], subtype)
            loss = binary_w * bin_loss + subtype_w * sub_loss

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()
        total_bin += bin_loss.item()
        total_sub += sub_loss.item()
        n_batches += 1

        if (
            is_main_process()
            and cfg.train.log_every > 0
            and (batch_idx + 1) % cfg.train.log_every == 0
        ):
            lr = optimizer.param_groups[0]["lr"]
            logger.info(
                f"  [Epoch {epoch}][{batch_idx + 1}/{len(loader)}] "
                f"loss={loss.item():.4f} "
                f"(bin={bin_loss.item():.4f}, sub={sub_loss.item():.4f}) "
                f"lr={lr:.2e}"
            )

    n = max(n_batches, 1)
    return {
        "total_loss": total_loss / n,
        "binary_loss": total_bin / n,
        "subtype_loss": total_sub / n,
    }


# ============================================================================
# Evaluation
# ============================================================================


@torch.no_grad()
def evaluate(
    backbone,
    classifier,
    loader,
    device,
    class_names,
    cached_mode,
    use_amp=True,
):
    """Evaluate on validation set.

    Gathers predictions from all GPUs via ``dist.all_gather_object``
    and computes metrics on rank 0.

    Returns:
        dict with ``binary`` and ``subtype`` metric sub-dicts.
    """
    classifier.eval()
    local_bin_probs = []
    local_bin_labels = []
    local_sub_probs = []
    local_sub_labels = []

    for batch in loader:
        mask = batch["mask"].to(device)

        if cached_mode:
            cls_tokens = batch["cls_tokens"].to(device)
        else:
            cls_tokens = extract_cls_tokens(
                backbone, batch["images"], device, chunk_size=64,
            )

        with torch.amp.autocast("cuda", enabled=use_amp):
            out = classifier(cls_tokens, mask)

        bin_prob = torch.sigmoid(out["binary_logit"]).squeeze(1).cpu()
        sub_prob = torch.softmax(out["subtype_logits"], dim=1).cpu()

        local_bin_probs.append(bin_prob)
        local_bin_labels.append(batch["has_lesion"])
        local_sub_probs.append(sub_prob)
        local_sub_labels.append(batch["subtype"])

    local_bin_probs = torch.cat(local_bin_probs).numpy()
    local_bin_labels = torch.cat(local_bin_labels).numpy()
    local_sub_probs = torch.cat(local_sub_probs).numpy()
    local_sub_labels = torch.cat(local_sub_labels).numpy()

    # Gather across GPUs
    if dist.is_initialized():
        gathered_bin_p = [None] * dist.get_world_size()
        gathered_bin_l = [None] * dist.get_world_size()
        gathered_sub_p = [None] * dist.get_world_size()
        gathered_sub_l = [None] * dist.get_world_size()
        dist.all_gather_object(gathered_bin_p, local_bin_probs)
        dist.all_gather_object(gathered_bin_l, local_bin_labels)
        dist.all_gather_object(gathered_sub_p, local_sub_probs)
        dist.all_gather_object(gathered_sub_l, local_sub_labels)
        all_bin_probs = np.concatenate(gathered_bin_p)
        all_bin_labels = np.concatenate(gathered_bin_l)
        all_sub_probs = np.concatenate(gathered_sub_p, axis=0)
        all_sub_labels = np.concatenate(gathered_sub_l)
    else:
        all_bin_probs = local_bin_probs
        all_bin_labels = local_bin_labels
        all_sub_probs = local_sub_probs
        all_sub_labels = local_sub_labels

    bin_metrics = compute_binary_metrics(all_bin_probs, all_bin_labels)
    sub_metrics = compute_subtype_metrics(all_sub_probs, all_sub_labels, class_names)

    return {"binary": bin_metrics, "subtype": sub_metrics}


# ============================================================================
# Checkpointing
# ============================================================================


def save_checkpoint(classifier, optimizer, scheduler, epoch, metrics, path):
    """Save classifier checkpoint (backbone is frozen, not saved)."""
    state = classifier.module if hasattr(classifier, "module") else classifier
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": state.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "metrics": metrics,
        },
        path,
    )


# ============================================================================
# Main
# ============================================================================


def main():
    # ── Args ──────────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="Train joint volume-level MIL classifier",
    )
    parser.add_argument(
        "--config", required=True, help="Path to volume_classifier.yaml",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["Class3", "Class4"],
        help="Dataset folders to use (default: Class3 Class4)",
    )
    parser.add_argument(
        "--cache-dir",
        default="",
        help="Pre-extracted embeddings dir (empty = on-the-fly extraction)",
    )
    args, overrides = parser.parse_known_args()

    # ── Config ────────────────────────────────────────────────────────────
    cfg = OmegaConf.load(args.config)
    if overrides:
        cli_conf = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(cfg, cli_conf)

    cache_dir = args.cache_dir or cfg.data.get("cache_dir", "")
    cached_mode = bool(cache_dir)
    datasets = tuple(args.datasets)

    # Build class map from dataset list
    class_map = {ds: i for i, ds in enumerate(datasets)}
    class_names = list(datasets)
    n_subtypes = len(datasets)

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
        logger.info("Volume-level MIL Classifier Training")
        logger.info("=" * 60)
        logger.info(f"Datasets: {datasets}")
        logger.info(f"Class map: {class_map}")
        logger.info(f"Mode: {'cached' if cached_mode else 'on-the-fly'}")
        logger.info(f"World size: {world_size}")
        logger.info(f"Device: {device}")

    # ── Seed ──────────────────────────────────────────────────────────────
    seed = cfg.train.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ── Data ──────────────────────────────────────────────────────────────
    shard_dir = cfg.data.shard_dir
    max_slices = cfg.classifier.max_slices
    val_ratio = cfg.eval.val_split

    if is_main_process():
        logger.info(f"Scanning shards in {shard_dir} ...")

    # All ranks scan independently (deterministic, avoids broadcast)
    volume_index = scan_shard_volumes(shard_dir, datasets)
    train_ids, val_ids = volume_train_val_split(volume_index, val_ratio, seed)

    if is_main_process():
        logger.info(f"Train: {len(train_ids)} volumes, Val: {len(val_ids)} volumes")

    # Datasets
    if cached_mode:
        train_ds = CachedVolumeDataset(
            cache_dir, train_ids, max_slices, class_map,
        )
        val_ds = CachedVolumeDataset(
            cache_dir, val_ids, max_slices, class_map,
        )
    else:
        train_ds = VolumeShardDataset(
            volume_index, train_ids, max_slices, class_map,
        )
        val_ds = VolumeShardDataset(
            volume_index, val_ids, max_slices, class_map,
        )

    # Samplers & loaders
    bs = cfg.data.batch_size
    nw = cfg.data.num_workers

    if distributed:
        train_sampler = DistributedSampler(train_ds, shuffle=True)
        val_sampler = DistributedSampler(val_ds, shuffle=False)
    else:
        train_sampler = RandomSampler(train_ds)
        val_sampler = None

    train_loader = DataLoader(
        train_ds,
        batch_size=bs,
        sampler=train_sampler,
        num_workers=nw,
        collate_fn=collate_volumes,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=bs,
        sampler=val_sampler,
        num_workers=nw,
        collate_fn=collate_volumes,
        pin_memory=True,
    )

    # ── Backbone (frozen, only needed for on-the-fly mode) ────────────────
    backbone = None
    embed_dim = 768  # default for ViT-B

    if not cached_mode:
        if is_main_process():
            logger.info("Loading frozen DINO backbone ...")
        backbone, embed_dim = build_backbone(
            model_name=cfg.backbone.arch,
            patch_size=cfg.backbone.patch_size,
            pretrained=cfg.backbone.checkpoint,
            img_size=cfg.backbone.img_size,
        )
        backbone.eval().to(device)
        for p in backbone.parameters():
            p.requires_grad_(False)
        if is_main_process():
            logger.info(f"Backbone loaded: embed_dim={embed_dim}")

    # ── Classifier ────────────────────────────────────────────────────────
    classifier = JointVolumeClassifier(
        embed_dim=embed_dim,
        hidden_dim=cfg.classifier.hidden_dim,
        n_subtypes=n_subtypes,
        mil_type=cfg.classifier.mil_type,
        topk=cfg.classifier.topk,
        dropout=cfg.classifier.dropout,
    ).to(device)

    if distributed:
        classifier = DDP(classifier, device_ids=[local_rank])

    n_params = sum(p.numel() for p in classifier.parameters()) / 1e3
    if is_main_process():
        logger.info(
            f"Classifier: mil={cfg.classifier.mil_type}, "
            f"hidden={cfg.classifier.hidden_dim}, subtypes={n_subtypes}, "
            f"params={n_params:.1f}K"
        )

    # ── Loss ──────────────────────────────────────────────────────────────
    focal_criterion = FocalLoss(
        alpha=cfg.loss.focal_alpha,
        gamma=cfg.loss.focal_gamma,
        pos_weight=cfg.loss.binary_pos_weight,
    )

    # Compute subtype class weights from training data
    subtype_counts = np.zeros(n_subtypes)
    for vid in train_ids:
        ds = volume_index[vid]["dataset"]
        subtype_counts[class_map[ds]] += 1

    class_weights = subtype_counts.sum() / (n_subtypes * np.maximum(subtype_counts, 1))
    class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)

    if is_main_process():
        logger.info(f"Subtype counts: {dict(zip(class_names, subtype_counts.astype(int).tolist()))}")
        logger.info(f"Subtype weights: {class_weights.cpu().tolist()}")

    ce_criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ── Optimizer + scheduler ─────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        classifier.parameters(),
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
    )
    scheduler = build_lr_scheduler(
        optimizer, cfg.optim.warmup_epochs, cfg.optim.epochs, len(train_loader),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.train.use_amp)

    # ── Output dir ────────────────────────────────────────────────────────
    output_dir = Path(cfg.train.output_dir)
    if is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)
        # Save config snapshot
        OmegaConf.save(cfg, output_dir / "config.yaml")
        logger.info(f"Output: {output_dir}")

    # ── Training loop ─────────────────────────────────────────────────────
    best_auroc = -1.0
    log_path = output_dir / "training_log.jsonl"

    for epoch in range(1, cfg.optim.epochs + 1):
        if distributed:
            train_sampler.set_epoch(epoch)

        if is_main_process():
            logger.info(f"Epoch {epoch}/{cfg.optim.epochs}")

        # Train
        train_metrics = train_one_epoch(
            backbone,
            classifier,
            train_loader,
            optimizer,
            scheduler,
            scaler,
            focal_criterion,
            ce_criterion,
            device,
            epoch,
            cfg,
            cached_mode,
        )

        # Validate
        val_metrics = evaluate(
            backbone,
            classifier,
            val_loader,
            device,
            class_names,
            cached_mode,
            use_amp=cfg.train.use_amp,
        )

        # Logging + checkpointing (rank 0 only)
        if is_main_process():
            bm = val_metrics["binary"]
            sm = val_metrics["subtype"]

            logger.info(
                f"  Train — loss={train_metrics['total_loss']:.4f} "
                f"(bin={train_metrics['binary_loss']:.4f}, "
                f"sub={train_metrics['subtype_loss']:.4f})"
            )
            logger.info(
                f"  Val binary — acc={bm['accuracy']:.4f} "
                f"sens={bm['sensitivity']:.4f} spec={bm['specificity']:.4f} "
                f"f1={bm['f1']:.4f} auroc={bm['auroc']}"
            )
            logger.info(
                f"  Val subtype — acc={sm['accuracy']:.4f} "
                f"f1={sm['per_class_f1']} macro_f1={sm['macro_f1']:.4f}"
            )
            logger.info(f"  Confusion matrix: {sm['confusion_matrix']}")

            # Log to file
            log_entry = {
                "epoch": epoch,
                "train": train_metrics,
                "val_binary": bm,
                "val_subtype": sm,
                "lr": optimizer.param_groups[0]["lr"],
            }
            with open(log_path, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

            # Save best model (by binary AUROC)
            auroc = bm["auroc"]
            if isinstance(auroc, float) and not np.isnan(auroc) and auroc > best_auroc:
                best_auroc = auroc
                save_checkpoint(
                    classifier,
                    optimizer,
                    scheduler,
                    epoch,
                    val_metrics,
                    output_dir / "best_model.pth",
                )
                logger.info(f"  ** New best AUROC: {best_auroc:.4f} — saved **")

            # Periodic checkpoint
            if epoch % 5 == 0:
                save_checkpoint(
                    classifier,
                    optimizer,
                    scheduler,
                    epoch,
                    val_metrics,
                    output_dir / f"checkpoint_epoch{epoch}.pth",
                )

    # ── Done ──────────────────────────────────────────────────────────────
    if is_main_process():
        logger.info("=" * 60)
        logger.info(f"Training complete. Best AUROC: {best_auroc:.4f}")
        logger.info(f"Checkpoints: {output_dir}")
        logger.info("=" * 60)

    cleanup_distributed()


if __name__ == "__main__":
    main()
