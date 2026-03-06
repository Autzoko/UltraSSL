#!/usr/bin/env python3
"""
Volume-level binary lesion presence classifier.

Predicts whether a breast ultrasound volume contains any lesion (has_lesion).
Uses frozen DINO ViT-B/14 CLS embeddings aggregated via MIL pooling.

Supports two modes:
  - Training: Train on all datasets with focal loss.
  - Inference: Predict has_lesion probability for each volume, output JSON.

Training (cached embeddings, recommended):
    torchrun --nproc_per_node=4 lesion_presence_classifier.py \
        --config config/volume_classifier.yaml \
        data.cache_dir=./embedding_cache

Training (on-the-fly extraction):
    torchrun --nproc_per_node=4 lesion_presence_classifier.py \
        --config config/volume_classifier.yaml

Inference:
    python lesion_presence_classifier.py \
        --config config/volume_classifier.yaml \
        --inference --checkpoint outputs/volume_classifier_binary/best_model.pth \
        --output-json predictions.json data.cache_dir=./embedding_cache
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
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "dinov2"))

from ultrassl.data.volume_dataset import (
    CachedVolumeDataset,
    OnTheFlyVolumeDataset,
    collate_volumes,
    compute_class_weights,
    load_volume_split_extended,
    scan_volume_index,
)
from ultrassl.models.volume_mil import FocalLoss, VolumeClassifier

logger = logging.getLogger("ultrassl")


# ── Distributed helpers ──────────────────────────────────────────────

def setup_distributed(use_nccl=False):
    """Initialize process group. NCCL for cached mode, gloo for on-the-fly."""
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        backend = "nccl" if use_nccl else "gloo"
        dist.init_process_group(backend=backend)
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size
    return 0, 0, 1


def cleanup_distributed(world_size):
    if world_size > 1:
        dist.destroy_process_group()


# ── Config ───────────────────────────────────────────────────────────

def load_config(config_path, cli_opts=None):
    cfg = OmegaConf.load(config_path)
    if cli_opts:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(cli_opts))
    return cfg


# ── Training ─────────────────────────────────────────────────────────

def train_presence(cfg):
    """Train binary lesion presence classifier."""
    use_cache = bool(cfg.data.get("cache_dir", ""))
    rank, local_rank, world_size = setup_distributed(use_nccl=use_cache)
    is_main = rank == 0

    if world_size > 1:
        device = torch.device(f"cuda:{local_rank}")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if is_main:
        logger.info(f"Device: {device}, world_size: {world_size}, cache: {use_cache}")

    # Seed
    seed = cfg.train.get("seed", 42)
    torch.manual_seed(seed + rank)
    random.seed(seed + rank)
    np.random.seed(seed + rank)

    output_dir = cfg.train.get("output_dir", "./outputs/volume_classifier_binary")
    if is_main:
        os.makedirs(output_dir, exist_ok=True)

    # Resolve paths
    shard_dir = cfg.data.shard_dir
    if not os.path.isabs(shard_dir):
        shard_dir = os.path.join(str(project_root), shard_dir)

    # Scan volume index
    if is_main:
        volume_index = scan_volume_index(shard_dir)
    else:
        volume_index = None

    # Broadcast volume index
    if world_size > 1:
        if is_main:
            data = pickle.dumps(volume_index)
            size_t = torch.tensor([len(data)], dtype=torch.long)
        else:
            size_t = torch.tensor([0], dtype=torch.long)
        dist.broadcast(size_t, src=0)
        if is_main:
            buf = torch.ByteTensor(list(data))
        else:
            buf = torch.zeros(size_t.item(), dtype=torch.uint8)
        dist.broadcast(buf, src=0)
        if not is_main:
            volume_index = pickle.loads(buf.numpy().tobytes())

    # Split
    train_vols, val_vols = load_volume_split_extended(
        volume_index,
        val_split=cfg.eval.get("val_split", 0.15),
        seed=seed,
        rank=rank,
        world_size=world_size,
    )

    # Dataset
    max_slices = cfg.classifier.get("max_slices", 32)
    batch_size = cfg.data.get("batch_size", 16)
    num_workers = cfg.data.get("num_workers", 4)

    if use_cache:
        cache_dir = cfg.data.cache_dir
        if not os.path.isabs(cache_dir):
            cache_dir = os.path.join(str(project_root), cache_dir)
        train_ds = CachedVolumeDataset(cache_dir, train_vols, max_slices=max_slices)
        val_ds = CachedVolumeDataset(cache_dir, val_vols, max_slices=max_slices)

        train_sampler = DistributedSampler(train_ds, shuffle=True) if world_size > 1 else RandomSampler(train_ds)
        val_sampler = DistributedSampler(val_ds, shuffle=False) if world_size > 1 else None

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, sampler=train_sampler,
            num_workers=num_workers, collate_fn=collate_volumes,
            pin_memory=True, drop_last=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, sampler=val_sampler,
            num_workers=num_workers, collate_fn=collate_volumes,
            pin_memory=True,
        )
    else:
        train_ds = OnTheFlyVolumeDataset(
            volume_index, shard_dir, train_vols,
            max_slices=max_slices, img_size=cfg.backbone.img_size,
        )
        val_ds = OnTheFlyVolumeDataset(
            volume_index, shard_dir, val_vols,
            max_slices=max_slices, img_size=cfg.backbone.img_size,
        )
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, collate_fn=collate_volumes,
            pin_memory=True, drop_last=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, collate_fn=collate_volumes,
            pin_memory=True,
        )

    # Model
    embed_dim = 768  # ViT-B
    model = VolumeClassifier(
        embed_dim=embed_dim,
        mil_type=cfg.classifier.get("mil_type", "gated_attention"),
        hidden_dim=cfg.classifier.get("hidden_dim", 256),
        enable_binary=True,
        enable_multiclass=False,
        topk=cfg.classifier.get("topk", 8),
        dropout=cfg.classifier.get("dropout", 0.25),
    )
    model.to(device)

    # On-the-fly mode: also need frozen backbone
    backbone = None
    if not use_cache:
        from ultrassl.models.backbone import build_backbone
        ckpt = cfg.backbone.checkpoint
        if not os.path.isabs(ckpt):
            ckpt = os.path.join(str(project_root), ckpt)
        backbone, _ = build_backbone(
            model_name=cfg.backbone.arch, patch_size=cfg.backbone.patch_size,
            pretrained=ckpt, img_size=cfg.backbone.img_size, drop_path_rate=0.0,
        )
        backbone.to(device)
        backbone.eval()
        for p in backbone.parameters():
            p.requires_grad = False

    # DDP wrapper (cached mode only)
    if use_cache and world_size > 1:
        model = DDP(model, device_ids=[local_rank])
    raw_model = model.module if isinstance(model, DDP) else model

    # Optimizer
    optimizer = torch.optim.AdamW(
        raw_model.get_trainable_params(),
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.get("weight_decay", 1e-3),
    )

    # LR scheduler
    total_epochs = cfg.optim.epochs
    warmup_epochs = cfg.optim.get("warmup_epochs", 5)

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Loss
    focal_loss = FocalLoss(
        alpha=cfg.loss.get("focal_alpha", 0.25),
        gamma=cfg.loss.get("focal_gamma", 2.0),
        pos_weight=cfg.loss.get("binary_pos_weight", None),
    ).to(device)

    # AMP
    use_amp = cfg.train.get("use_amp", True) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    log_every = cfg.train.get("log_every", 10)
    best_val_auroc = 0.0
    metrics_file = os.path.join(output_dir, "training_metrics.jsonl")

    if is_main:
        logger.info(f"Training binary presence classifier for {total_epochs} epochs")
        logger.info(f"Train: {len(train_ds)} volumes, Val: {len(val_ds)} volumes")
        logger.info(f"AMP: {use_amp}, batch_size: {batch_size}")

    for epoch in range(total_epochs):
        model.train()
        if use_cache and world_size > 1:
            train_sampler.set_epoch(epoch)

        epoch_loss = 0.0
        n_batches = 0

        for batch_idx, (data, masks, labels) in enumerate(train_loader):
            data = data.to(device)
            masks = masks.to(device)
            has_lesion = labels["has_lesion"].float().unsqueeze(1).to(device)

            # On-the-fly: extract embeddings first
            if not use_cache and backbone is not None:
                B, K, C, H, W = data.shape
                flat = data.reshape(B * K, C, H, W)
                with torch.no_grad():
                    out = backbone(flat, is_training=True)
                    embeddings = out["x_norm_clstoken"].reshape(B, K, -1)
            else:
                embeddings = data

            optimizer.zero_grad()

            if use_amp:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    out = model(embeddings, masks)
                    loss = focal_loss(out["binary_logit"], has_lesion)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                out = model(embeddings, masks)
                loss = focal_loss(out["binary_logit"], has_lesion)
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

            if is_main and (batch_idx + 1) % log_every == 0:
                logger.info(
                    f"[epoch {epoch+1}/{total_epochs}] [batch {batch_idx+1}] "
                    f"loss={epoch_loss/n_batches:.4f} lr={optimizer.param_groups[0]['lr']:.6f}"
                )

        scheduler.step()

        # Epoch summary
        avg_loss = epoch_loss / max(1, n_batches)
        if is_main:
            logger.info(f"Epoch {epoch+1}/{total_epochs}: loss={avg_loss:.4f}")

        # Validation
        val_metrics = evaluate_binary(
            model, val_loader, device, backbone=backbone, use_cache=use_cache,
            use_amp=use_amp, rank=rank, world_size=world_size,
        )

        if is_main and val_metrics is not None:
            logger.info(
                f"  Val: acc={val_metrics['accuracy']:.4f} "
                f"sens={val_metrics['sensitivity']:.4f} "
                f"spec={val_metrics['specificity']:.4f} "
                f"auroc={val_metrics['auroc']:.4f} "
                f"(n={val_metrics['n_samples']}, pos={val_metrics['n_positive']})"
            )

            with open(metrics_file, "a") as f:
                f.write(json.dumps({
                    "epoch": epoch + 1, "loss": avg_loss,
                    "lr": optimizer.param_groups[0]["lr"],
                    **val_metrics,
                }) + "\n")

            if val_metrics["auroc"] > best_val_auroc:
                best_val_auroc = val_metrics["auroc"]
                torch.save({
                    "model": raw_model.state_dict(),
                    "epoch": epoch + 1,
                    "val_auroc": best_val_auroc,
                    "config": OmegaConf.to_container(cfg),
                }, os.path.join(output_dir, "best_model.pth"))
                logger.info(f"  New best (AUROC={best_val_auroc:.4f})")

        # Periodic checkpoint
        if is_main and ((epoch + 1) % 5 == 0 or (epoch + 1) == total_epochs):
            torch.save({
                "model": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch + 1,
                "config": OmegaConf.to_container(cfg),
            }, os.path.join(output_dir, f"checkpoint_epoch{epoch+1:03d}.pth"))

        if world_size > 1:
            dist.barrier()

    if is_main:
        logger.info(f"Training complete. Best val AUROC: {best_val_auroc:.4f}")

    cleanup_distributed(world_size)


# ── Evaluation ───────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_binary(model, loader, device, backbone=None, use_cache=True,
                    use_amp=False, rank=0, world_size=1):
    """Evaluate binary classifier on validation set."""
    model.eval()
    all_preds = []
    all_labels = []

    for data, masks, labels in loader:
        data = data.to(device)
        masks = masks.to(device)

        if not use_cache and backbone is not None:
            B, K, C, H, W = data.shape
            flat = data.reshape(B * K, C, H, W)
            out = backbone(flat, is_training=True)
            embeddings = out["x_norm_clstoken"].reshape(B, K, -1)
        else:
            embeddings = data

        if use_amp:
            with torch.amp.autocast("cuda", dtype=torch.float16):
                out = model(embeddings, masks)
        else:
            out = model(embeddings, masks)

        probs = torch.sigmoid(out["binary_logit"]).cpu().squeeze(-1)
        all_preds.extend(probs.tolist())
        all_labels.extend(labels["has_lesion"].tolist())

    # Gather across ranks
    if world_size > 1:
        all_preds_gathered = [None] * world_size
        all_labels_gathered = [None] * world_size
        dist.all_gather_object(all_preds_gathered, all_preds)
        dist.all_gather_object(all_labels_gathered, all_labels)
        if rank == 0:
            all_preds = [p for sublist in all_preds_gathered for p in sublist]
            all_labels = [l for sublist in all_labels_gathered for l in sublist]
        else:
            model.train()
            return None

    metrics = _compute_binary_metrics(all_preds, all_labels)
    model.train()
    return metrics


def _compute_binary_metrics(preds, labels):
    """Compute binary classification metrics."""
    if len(preds) < 2:
        return {"accuracy": 0, "sensitivity": 0, "specificity": 0,
                "auroc": 0.5, "n_samples": len(preds), "n_positive": 0}

    preds = np.array(preds)
    labels = np.array(labels)
    binary = (preds >= 0.5).astype(float)

    tp = ((binary == 1) & (labels == 1)).sum()
    fn = ((binary == 0) & (labels == 1)).sum()
    tn = ((binary == 0) & (labels == 0)).sum()
    fp = ((binary == 1) & (labels == 0)).sum()

    accuracy = (binary == labels).mean()
    sensitivity = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)

    try:
        from sklearn.metrics import roc_auc_score
        auroc = roc_auc_score(labels, preds) if len(set(labels)) > 1 else 0.5
    except ImportError:
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
    pos = preds[labels == 1]
    neg = preds[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    n_correct = sum((p > neg).sum() + 0.5 * (p == neg).sum() for p in pos)
    return float(n_correct / (len(pos) * len(neg)))


# ── Inference ────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(cfg, checkpoint_path, output_path):
    """Run inference on all volumes, output per-volume predictions."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = VolumeClassifier(
        embed_dim=768,
        mil_type=cfg.classifier.get("mil_type", "gated_attention"),
        hidden_dim=cfg.classifier.get("hidden_dim", 256),
        enable_binary=True,
        enable_multiclass=False,
        topk=cfg.classifier.get("topk", 8),
        dropout=0.0,  # No dropout at inference
    )
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    # Data
    shard_dir = cfg.data.shard_dir
    if not os.path.isabs(shard_dir):
        shard_dir = os.path.join(str(project_root), shard_dir)

    use_cache = bool(cfg.data.get("cache_dir", ""))
    max_slices = cfg.classifier.get("max_slices", 32)

    if use_cache:
        cache_dir = cfg.data.cache_dir
        if not os.path.isabs(cache_dir):
            cache_dir = os.path.join(str(project_root), cache_dir)
        volume_index = scan_volume_index(shard_dir)
        all_vols = set(volume_index.keys())
        ds = CachedVolumeDataset(cache_dir, all_vols, max_slices=max_slices)
    else:
        volume_index = scan_volume_index(shard_dir)
        all_vols = set(volume_index.keys())
        ds = OnTheFlyVolumeDataset(
            volume_index, shard_dir, all_vols,
            max_slices=max_slices, img_size=cfg.backbone.img_size,
        )

    loader = DataLoader(
        ds, batch_size=cfg.data.get("batch_size", 16), shuffle=False,
        num_workers=cfg.data.get("num_workers", 4), collate_fn=collate_volumes,
    )

    # Backbone for on-the-fly mode
    backbone = None
    if not use_cache:
        from ultrassl.models.backbone import build_backbone
        ckpt_path = cfg.backbone.checkpoint
        if not os.path.isabs(ckpt_path):
            ckpt_path = os.path.join(str(project_root), ckpt_path)
        backbone, _ = build_backbone(
            model_name=cfg.backbone.arch, patch_size=cfg.backbone.patch_size,
            pretrained=ckpt_path, img_size=cfg.backbone.img_size, drop_path_rate=0.0,
        )
        backbone.to(device)
        backbone.eval()

    predictions = {}
    for data, masks, labels in loader:
        data = data.to(device)
        masks = masks.to(device)

        if not use_cache and backbone is not None:
            B, K, C, H, W = data.shape
            flat = data.reshape(B * K, C, H, W)
            out_bb = backbone(flat, is_training=True)
            embeddings = out_bb["x_norm_clstoken"].reshape(B, K, -1)
        else:
            embeddings = data

        out = model(embeddings, masks)
        probs = torch.sigmoid(out["binary_logit"]).cpu().squeeze(-1)
        attn = out["attn_weights"].cpu()

        for i, sid in enumerate(labels["sample_id"]):
            n_valid = int(masks[i].sum().item())
            top_attn_indices = attn[i, :n_valid].topk(min(5, n_valid)).indices.tolist()

            predictions[sid] = {
                "has_lesion_prob": float(probs[i]),
                "has_lesion_pred": int(probs[i] >= 0.5),
                "dataset": labels["dataset"][i],
                "n_slices": n_valid,
                "top_slice_indices": top_attn_indices,
            }

    # Add ground truth if available
    for sid, pred in predictions.items():
        if sid in volume_index:
            pred["gt_has_lesion"] = volume_index[sid]["has_lesion"]

    with open(output_path, "w") as f:
        json.dump(predictions, f, indent=2)

    # Summary
    n_total = len(predictions)
    n_pred_pos = sum(1 for p in predictions.values() if p["has_lesion_pred"] == 1)
    logger.info(f"Inference complete: {n_total} volumes, {n_pred_pos} predicted positive")
    logger.info(f"Predictions saved to {output_path}")

    # If ground truth available, compute metrics
    gt_labels = [p.get("gt_has_lesion", -1) for p in predictions.values()]
    if all(g >= 0 for g in gt_labels):
        pred_probs = [p["has_lesion_prob"] for p in predictions.values()]
        metrics = _compute_binary_metrics(pred_probs, gt_labels)
        logger.info(
            f"  Metrics: acc={metrics['accuracy']:.4f} "
            f"sens={metrics['sensitivity']:.4f} "
            f"spec={metrics['specificity']:.4f} "
            f"auroc={metrics['auroc']:.4f}"
        )


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Volume-level binary lesion presence classifier")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--inference", action="store_true",
                        help="Run inference mode instead of training")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Model checkpoint path (for inference)")
    parser.add_argument("--output-json", type=str, default="predictions.json",
                        help="Output JSON path (for inference)")
    parser.add_argument("opts", nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args()

    cli_opts = [o for o in args.opts if o and o != "--"]
    cfg = load_config(args.config, cli_opts)

    rank = int(os.environ.get("RANK", 0))
    output_dir = cfg.train.get("output_dir", "./outputs/volume_classifier_binary")
    os.makedirs(output_dir, exist_ok=True)

    handlers = [logging.StreamHandler()]
    if rank == 0:
        handlers.append(logging.FileHandler(os.path.join(output_dir, "train.log")))
    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=handlers,
    )

    if args.inference:
        if args.checkpoint is None:
            args.checkpoint = os.path.join(output_dir, "best_model.pth")
        run_inference(cfg, args.checkpoint, args.output_json)
    else:
        train_presence(cfg)


if __name__ == "__main__":
    main()
