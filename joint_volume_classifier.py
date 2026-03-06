#!/usr/bin/env python3
"""
Joint volume-level classifier: binary lesion detection + multi-class subtype.

Predicts both has_lesion (binary) and lesion class (Class2/Class3/Class4)
from frozen DINO ViT-B/14 CLS embeddings aggregated via MIL pooling.

- Binary head: Focal loss on all volumes (all datasets have has_lesion labels).
- Multi-class head: Weighted CrossEntropy on Class2/3/4 volumes only.
  BIrads/Abus/Duying volumes have class_idx=-1 (ignored in class loss).

Training (cached embeddings, recommended):
    torchrun --nproc_per_node=4 joint_volume_classifier.py \
        --config config/volume_classifier.yaml \
        data.cache_dir=./embedding_cache

Training (on-the-fly extraction):
    torchrun --nproc_per_node=4 joint_volume_classifier.py \
        --config config/volume_classifier.yaml
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
    CLASS_MAPPING,
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


def load_config(config_path, cli_opts=None):
    cfg = OmegaConf.load(config_path)
    if cli_opts:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(cli_opts))
    return cfg


# ── Training ─────────────────────────────────────────────────────────

def train_joint(cfg):
    """Train joint binary + multi-class volume classifier."""
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

    seed = cfg.train.get("seed", 42)
    torch.manual_seed(seed + rank)
    random.seed(seed + rank)
    np.random.seed(seed + rank)

    output_dir = cfg.train.get("output_dir", "./outputs/volume_classifier_joint")
    if is_main:
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Device: {device}, world_size: {world_size}, cache: {use_cache}")

    # Resolve paths
    shard_dir = cfg.data.shard_dir
    if not os.path.isabs(shard_dir):
        shard_dir = os.path.join(str(project_root), shard_dir)

    # Scan volume index
    if is_main:
        volume_index = scan_volume_index(shard_dir)
    else:
        volume_index = None

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

    # Class mapping
    class_mapping = dict(cfg.get("class_mapping", CLASS_MAPPING))
    n_classes = len(class_mapping)

    # Split
    train_vols, val_vols = load_volume_split_extended(
        volume_index,
        val_split=cfg.eval.get("val_split", 0.15),
        seed=seed, rank=rank, world_size=world_size,
    )

    # Compute class weights for multi-class loss
    cfg_class_weights = cfg.loss.get("class_weights", None)
    if cfg_class_weights is not None:
        class_weights = torch.tensor(list(cfg_class_weights), dtype=torch.float32)
    else:
        class_weights = compute_class_weights(volume_index, train_vols, class_mapping)
    class_weights = class_weights.to(device)

    # Dataset
    max_slices = cfg.classifier.get("max_slices", 32)
    batch_size = cfg.data.get("batch_size", 16)
    num_workers = cfg.data.get("num_workers", 4)

    if use_cache:
        cache_dir = cfg.data.cache_dir
        if not os.path.isabs(cache_dir):
            cache_dir = os.path.join(str(project_root), cache_dir)
        train_ds = CachedVolumeDataset(cache_dir, train_vols, max_slices=max_slices,
                                       class_mapping=class_mapping)
        val_ds = CachedVolumeDataset(cache_dir, val_vols, max_slices=max_slices,
                                     class_mapping=class_mapping)
        train_sampler = DistributedSampler(train_ds, shuffle=True) if world_size > 1 else RandomSampler(train_ds)
        val_sampler = DistributedSampler(val_ds, shuffle=False) if world_size > 1 else None
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, sampler=train_sampler,
            num_workers=num_workers, collate_fn=collate_volumes,
            pin_memory=True, drop_last=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, sampler=val_sampler,
            num_workers=num_workers, collate_fn=collate_volumes, pin_memory=True,
        )
    else:
        train_ds = OnTheFlyVolumeDataset(
            volume_index, shard_dir, train_vols, max_slices=max_slices,
            class_mapping=class_mapping, img_size=cfg.backbone.img_size,
        )
        val_ds = OnTheFlyVolumeDataset(
            volume_index, shard_dir, val_vols, max_slices=max_slices,
            class_mapping=class_mapping, img_size=cfg.backbone.img_size,
        )
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, collate_fn=collate_volumes,
            pin_memory=True, drop_last=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, collate_fn=collate_volumes, pin_memory=True,
        )

    # Model
    model = VolumeClassifier(
        embed_dim=768,
        mil_type=cfg.classifier.get("mil_type", "gated_attention"),
        hidden_dim=cfg.classifier.get("hidden_dim", 256),
        n_classes=n_classes,
        enable_binary=True,
        enable_multiclass=True,
        topk=cfg.classifier.get("topk", 8),
        dropout=cfg.classifier.get("dropout", 0.25),
    )
    model.to(device)

    # On-the-fly backbone
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

    if use_cache and world_size > 1:
        model = DDP(model, device_ids=[local_rank])
    raw_model = model.module if isinstance(model, DDP) else model

    # Optimizer
    optimizer = torch.optim.AdamW(
        raw_model.get_trainable_params(),
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.get("weight_decay", 1e-3),
    )

    total_epochs = cfg.optim.epochs
    warmup_epochs = cfg.optim.get("warmup_epochs", 5)

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Losses
    focal_loss = FocalLoss(
        alpha=cfg.loss.get("focal_alpha", 0.25),
        gamma=cfg.loss.get("focal_gamma", 2.0),
        pos_weight=cfg.loss.get("binary_pos_weight", None),
    ).to(device)
    ce_loss = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1)

    binary_weight = cfg.loss.get("binary_loss_weight", 1.0)
    mc_weight = cfg.loss.get("multiclass_loss_weight", 0.5)

    use_amp = cfg.train.get("use_amp", True) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    log_every = cfg.train.get("log_every", 10)
    best_val_auroc = 0.0
    metrics_file = os.path.join(output_dir, "training_metrics.jsonl")

    if is_main:
        logger.info(f"Training joint classifier for {total_epochs} epochs")
        logger.info(f"Train: {len(train_ds)} volumes, Val: {len(val_ds)} volumes")
        logger.info(f"Class weights: {class_weights.tolist()}")
        logger.info(f"Loss weights: binary={binary_weight}, multiclass={mc_weight}")

    for epoch in range(total_epochs):
        model.train()
        if use_cache and world_size > 1:
            train_sampler.set_epoch(epoch)

        epoch_binary_loss = 0.0
        epoch_mc_loss = 0.0
        epoch_total_loss = 0.0
        n_batches = 0
        n_mc_batches = 0

        for batch_idx, (data, masks, labels) in enumerate(train_loader):
            data = data.to(device)
            masks = masks.to(device)
            has_lesion = labels["has_lesion"].float().unsqueeze(1).to(device)
            class_idx = labels["class_idx"].long().to(device)

            if not use_cache and backbone is not None:
                B, K, C, H, W = data.shape
                flat = data.reshape(B * K, C, H, W)
                with torch.no_grad():
                    out_bb = backbone(flat, is_training=True)
                    embeddings = out_bb["x_norm_clstoken"].reshape(B, K, -1)
            else:
                embeddings = data

            optimizer.zero_grad()

            if use_amp:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    out = model(embeddings, masks)
                    b_loss = focal_loss(out["binary_logit"], has_lesion)
                    class_mask = class_idx >= 0
                    if class_mask.any():
                        m_loss = ce_loss(out["class_logits"][class_mask], class_idx[class_mask])
                    else:
                        m_loss = torch.tensor(0.0, device=device)
                    loss = binary_weight * b_loss + mc_weight * m_loss
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                out = model(embeddings, masks)
                b_loss = focal_loss(out["binary_logit"], has_lesion)
                class_mask = class_idx >= 0
                if class_mask.any():
                    m_loss = ce_loss(out["class_logits"][class_mask], class_idx[class_mask])
                else:
                    m_loss = torch.tensor(0.0, device=device)
                loss = binary_weight * b_loss + mc_weight * m_loss
                loss.backward()
                optimizer.step()

            epoch_binary_loss += b_loss.item()
            epoch_total_loss += loss.item()
            n_batches += 1
            if class_mask.any():
                epoch_mc_loss += m_loss.item()
                n_mc_batches += 1

            if is_main and (batch_idx + 1) % log_every == 0:
                avg = epoch_total_loss / n_batches
                avg_b = epoch_binary_loss / n_batches
                avg_m = epoch_mc_loss / max(1, n_mc_batches)
                logger.info(
                    f"[epoch {epoch+1}/{total_epochs}] [batch {batch_idx+1}] "
                    f"loss={avg:.4f} (binary={avg_b:.4f}, mc={avg_m:.4f}) "
                    f"lr={optimizer.param_groups[0]['lr']:.6f}"
                )

        scheduler.step()

        if is_main:
            n = max(1, n_batches)
            logger.info(
                f"Epoch {epoch+1}/{total_epochs}: "
                f"loss={epoch_total_loss/n:.4f} "
                f"(binary={epoch_binary_loss/n:.4f}, "
                f"mc={epoch_mc_loss/max(1,n_mc_batches):.4f})"
            )

        # Validation
        val_metrics = evaluate_joint(
            model, val_loader, device, backbone=backbone, use_cache=use_cache,
            use_amp=use_amp, rank=rank, world_size=world_size,
            n_classes=n_classes, class_mapping=class_mapping,
        )

        if is_main and val_metrics is not None:
            bm = val_metrics["binary"]
            logger.info(
                f"  Val binary: acc={bm['accuracy']:.4f} "
                f"sens={bm['sensitivity']:.4f} spec={bm['specificity']:.4f} "
                f"auroc={bm['auroc']:.4f} (n={bm['n_samples']}, pos={bm['n_positive']})"
            )
            if val_metrics["multiclass"]["n_samples"] > 0:
                mm = val_metrics["multiclass"]
                logger.info(
                    f"  Val multi-class: acc={mm['accuracy']:.4f} "
                    f"per_class_acc={mm['per_class_accuracy']} "
                    f"(n={mm['n_samples']})"
                )

            with open(metrics_file, "a") as f:
                f.write(json.dumps({
                    "epoch": epoch + 1,
                    "loss": epoch_total_loss / max(1, n_batches),
                    "binary_auroc": bm["auroc"],
                    "binary_accuracy": bm["accuracy"],
                    "mc_accuracy": val_metrics["multiclass"]["accuracy"],
                    "lr": optimizer.param_groups[0]["lr"],
                }) + "\n")

            if bm["auroc"] > best_val_auroc:
                best_val_auroc = bm["auroc"]
                torch.save({
                    "model": raw_model.state_dict(),
                    "epoch": epoch + 1,
                    "val_auroc": best_val_auroc,
                    "config": OmegaConf.to_container(cfg),
                }, os.path.join(output_dir, "best_model.pth"))
                logger.info(f"  New best (AUROC={best_val_auroc:.4f})")

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
def evaluate_joint(model, loader, device, backbone=None, use_cache=True,
                   use_amp=False, rank=0, world_size=1, n_classes=3,
                   class_mapping=None):
    """Evaluate joint classifier: binary + multi-class metrics."""
    model.eval()
    binary_preds, binary_labels = [], []
    mc_preds, mc_labels = [], []

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

        if use_amp:
            with torch.amp.autocast("cuda", dtype=torch.float16):
                out = model(embeddings, masks)
        else:
            out = model(embeddings, masks)

        # Binary
        probs = torch.sigmoid(out["binary_logit"]).cpu().squeeze(-1)
        binary_preds.extend(probs.tolist())
        binary_labels.extend(labels["has_lesion"].tolist())

        # Multi-class (only for Class2/3/4 volumes)
        class_idx = labels["class_idx"]
        valid = class_idx >= 0
        if valid.any():
            mc_pred = out["class_logits"][valid].argmax(dim=1).cpu()
            mc_preds.extend(mc_pred.tolist())
            mc_labels.extend(class_idx[valid].tolist())

    # Gather
    if world_size > 1:
        gathered = [None] * world_size
        dist.all_gather_object(gathered, (binary_preds, binary_labels, mc_preds, mc_labels))
        if rank == 0:
            binary_preds = [p for g in gathered for p in g[0]]
            binary_labels = [l for g in gathered for l in g[1]]
            mc_preds = [p for g in gathered for p in g[2]]
            mc_labels = [l for g in gathered for l in g[3]]
        else:
            model.train()
            return None

    # Binary metrics
    binary_metrics = _compute_binary_metrics(binary_preds, binary_labels)

    # Multi-class metrics
    mc_metrics = _compute_multiclass_metrics(mc_preds, mc_labels, n_classes, class_mapping)

    model.train()
    return {"binary": binary_metrics, "multiclass": mc_metrics}


def _compute_binary_metrics(preds, labels):
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

    try:
        from sklearn.metrics import roc_auc_score
        auroc = roc_auc_score(labels, preds) if len(set(labels)) > 1 else 0.5
    except ImportError:
        pos, neg = preds[labels == 1], preds[labels == 0]
        auroc = 0.5 if len(pos) == 0 or len(neg) == 0 else float(
            sum((p > neg).sum() + 0.5 * (p == neg).sum() for p in pos) / (len(pos) * len(neg)))

    return {
        "accuracy": float((binary == labels).mean()),
        "sensitivity": float(tp / max(1, tp + fn)),
        "specificity": float(tn / max(1, tn + fp)),
        "auroc": float(auroc),
        "n_samples": len(labels),
        "n_positive": int(labels.sum()),
    }


def _compute_multiclass_metrics(preds, labels, n_classes, class_mapping=None):
    if not preds:
        return {"accuracy": 0, "per_class_accuracy": {}, "n_samples": 0}

    preds = np.array(preds)
    labels = np.array(labels)
    accuracy = (preds == labels).mean()

    # Per-class accuracy
    inv_mapping = {v: k for k, v in (class_mapping or CLASS_MAPPING).items()}
    per_class = {}
    for c in range(n_classes):
        mask = labels == c
        if mask.sum() > 0:
            name = inv_mapping.get(c, str(c))
            per_class[name] = float((preds[mask] == c).mean())

    return {
        "accuracy": float(accuracy),
        "per_class_accuracy": per_class,
        "n_samples": len(labels),
    }


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Joint volume-level classifier (binary + multi-class)")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("opts", nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args()

    cli_opts = [o for o in args.opts if o and o != "--"]
    cfg = load_config(args.config, cli_opts)

    rank = int(os.environ.get("RANK", 0))
    output_dir = cfg.train.get("output_dir", "./outputs/volume_classifier_joint")
    os.makedirs(output_dir, exist_ok=True)

    handlers = [logging.StreamHandler()]
    if rank == 0:
        handlers.append(logging.FileHandler(os.path.join(output_dir, "train.log")))
    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=handlers,
    )

    train_joint(cfg)


if __name__ == "__main__":
    main()
