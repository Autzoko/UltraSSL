#!/usr/bin/env python3
"""
Volume-level lesion subtype classifier (Class2 / Class3 / Class4).

Second-stage classifier that predicts the lesion subtype only for volumes
that are known or predicted to be lesion-positive. Uses frozen DINO ViT-B/14
CLS embeddings aggregated via MIL pooling.

Training (on known positive volumes from Class2/3/4):
    torchrun --nproc_per_node=4 lesion_subtype_classifier.py \
        --config config/volume_classifier.yaml \
        data.cache_dir=./embedding_cache

Inference (on all volumes or filtered by lesion_presence_classifier):
    python lesion_subtype_classifier.py \
        --config config/volume_classifier.yaml \
        --inference --checkpoint outputs/volume_classifier_subtype/best_model.pth \
        --output-json subtype_predictions.json data.cache_dir=./embedding_cache

    # With filtering by presence predictions:
    python lesion_subtype_classifier.py \
        --config config/volume_classifier.yaml \
        --inference --checkpoint outputs/volume_classifier_subtype/best_model.pth \
        --filter-json predictions.json \
        --output-json subtype_predictions.json data.cache_dir=./embedding_cache
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
from ultrassl.models.volume_mil import MultiClassFocalLoss, VolumeClassifier

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

def train_subtype(cfg):
    """Train multi-class subtype classifier on positive volumes only."""
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

    output_dir = cfg.train.get("output_dir", "./outputs/volume_classifier_subtype")
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

    class_mapping = dict(cfg.get("class_mapping", CLASS_MAPPING))
    n_classes = len(class_mapping)
    subtype_datasets = list(class_mapping.keys())

    # Split: positive volumes from Class2/3/4 only
    train_vols, val_vols = load_volume_split_extended(
        volume_index,
        val_split=cfg.eval.get("val_split", 0.15),
        seed=seed, rank=rank, world_size=world_size,
        filter_positive_only=True,
        filter_datasets=subtype_datasets,
    )

    if is_main:
        logger.info(f"Subtype training: {len(train_vols)} train, {len(val_vols)} val "
                     f"(positive volumes from {subtype_datasets})")

    # Class weights
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

    # Model (multi-class only)
    model = VolumeClassifier(
        embed_dim=768,
        mil_type=cfg.classifier.get("mil_type", "gated_attention"),
        hidden_dim=cfg.classifier.get("hidden_dim", 256),
        n_classes=n_classes,
        enable_binary=False,
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

    # Loss
    mc_loss_type = cfg.loss.get("multiclass_loss", "ce")
    if mc_loss_type == "focal":
        loss_fn = MultiClassFocalLoss(
            gamma=cfg.loss.get("focal_gamma", 2.0),
            weight=class_weights,
        )
    else:
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    use_amp = cfg.train.get("use_amp", True) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    log_every = cfg.train.get("log_every", 10)
    best_val_acc = 0.0
    metrics_file = os.path.join(output_dir, "training_metrics.jsonl")

    if is_main:
        logger.info(f"Training subtype classifier for {total_epochs} epochs")
        logger.info(f"Train: {len(train_ds)} volumes, Val: {len(val_ds)} volumes")
        logger.info(f"Class weights: {class_weights.tolist()}")
        logger.info(f"Loss: {mc_loss_type}")

    for epoch in range(total_epochs):
        model.train()
        if use_cache and world_size > 1:
            train_sampler.set_epoch(epoch)

        epoch_loss = 0.0
        n_batches = 0
        n_correct = 0
        n_total = 0

        for batch_idx, (data, masks, labels) in enumerate(train_loader):
            data = data.to(device)
            masks = masks.to(device)
            class_idx = labels["class_idx"].long().to(device)

            if not use_cache and backbone is not None:
                B, K, C, H, W = data.shape
                flat = data.reshape(B * K, C, H, W)
                with torch.no_grad():
                    out_bb = backbone(flat, is_training=True)
                    embeddings = out_bb["x_norm_clstoken"].reshape(B, K, -1)
            else:
                embeddings = data

            # All samples should have valid class labels (filter_positive_only=True)
            valid = class_idx >= 0
            if not valid.any():
                continue

            optimizer.zero_grad()

            if use_amp:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    out = model(embeddings, masks)
                    loss = loss_fn(out["class_logits"][valid], class_idx[valid])
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                out = model(embeddings, masks)
                loss = loss_fn(out["class_logits"][valid], class_idx[valid])
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
            preds = out["class_logits"][valid].argmax(dim=1)
            n_correct += (preds == class_idx[valid]).sum().item()
            n_total += valid.sum().item()

            if is_main and (batch_idx + 1) % log_every == 0:
                train_acc = n_correct / max(1, n_total)
                logger.info(
                    f"[epoch {epoch+1}/{total_epochs}] [batch {batch_idx+1}] "
                    f"loss={epoch_loss/n_batches:.4f} acc={train_acc:.4f} "
                    f"lr={optimizer.param_groups[0]['lr']:.6f}"
                )

        scheduler.step()

        if is_main:
            n = max(1, n_batches)
            train_acc = n_correct / max(1, n_total)
            logger.info(
                f"Epoch {epoch+1}/{total_epochs}: "
                f"loss={epoch_loss/n:.4f} train_acc={train_acc:.4f}"
            )

        # Validation
        val_metrics = evaluate_multiclass(
            model, val_loader, device, backbone=backbone, use_cache=use_cache,
            use_amp=use_amp, rank=rank, world_size=world_size,
            n_classes=n_classes, class_mapping=class_mapping,
        )

        if is_main and val_metrics is not None:
            logger.info(
                f"  Val: acc={val_metrics['accuracy']:.4f} "
                f"per_class={val_metrics['per_class_accuracy']} "
                f"(n={val_metrics['n_samples']})"
            )

            with open(metrics_file, "a") as f:
                f.write(json.dumps({
                    "epoch": epoch + 1,
                    "loss": epoch_loss / max(1, n_batches),
                    "train_acc": n_correct / max(1, n_total),
                    **val_metrics,
                    "lr": optimizer.param_groups[0]["lr"],
                }) + "\n")

            if val_metrics["accuracy"] > best_val_acc:
                best_val_acc = val_metrics["accuracy"]
                torch.save({
                    "model": raw_model.state_dict(),
                    "epoch": epoch + 1,
                    "val_accuracy": best_val_acc,
                    "config": OmegaConf.to_container(cfg),
                }, os.path.join(output_dir, "best_model.pth"))
                logger.info(f"  New best (acc={best_val_acc:.4f})")

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
        logger.info(f"Training complete. Best val accuracy: {best_val_acc:.4f}")
    cleanup_distributed(world_size)


# ── Evaluation ───────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_multiclass(model, loader, device, backbone=None, use_cache=True,
                        use_amp=False, rank=0, world_size=1, n_classes=3,
                        class_mapping=None):
    """Evaluate multi-class classifier."""
    model.eval()
    all_preds, all_labels = [], []

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

        class_idx = labels["class_idx"]
        valid = class_idx >= 0
        if valid.any():
            preds = out["class_logits"][valid].argmax(dim=1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(class_idx[valid].tolist())

    if world_size > 1:
        gathered = [None] * world_size
        dist.all_gather_object(gathered, (all_preds, all_labels))
        if rank == 0:
            all_preds = [p for g in gathered for p in g[0]]
            all_labels = [l for g in gathered for l in g[1]]
        else:
            model.train()
            return None

    if not all_preds:
        model.train()
        return {"accuracy": 0, "per_class_accuracy": {}, "n_samples": 0}

    preds = np.array(all_preds)
    labels = np.array(all_labels)
    accuracy = (preds == labels).mean()

    inv_mapping = {v: k for k, v in (class_mapping or CLASS_MAPPING).items()}
    per_class = {}
    for c in range(n_classes):
        mask = labels == c
        if mask.sum() > 0:
            name = inv_mapping.get(c, str(c))
            per_class[name] = float((preds[mask] == c).mean())

    model.train()
    return {
        "accuracy": float(accuracy),
        "per_class_accuracy": per_class,
        "n_samples": len(labels),
    }


# ── Inference ────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(cfg, checkpoint_path, output_path, filter_json=None):
    """Run subtype inference on positive volumes."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class_mapping = dict(cfg.get("class_mapping", CLASS_MAPPING))
    n_classes = len(class_mapping)
    inv_mapping = {v: k for k, v in class_mapping.items()}

    # Load model
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = VolumeClassifier(
        embed_dim=768,
        mil_type=cfg.classifier.get("mil_type", "gated_attention"),
        hidden_dim=cfg.classifier.get("hidden_dim", 256),
        n_classes=n_classes,
        enable_binary=False,
        enable_multiclass=True,
        topk=cfg.classifier.get("topk", 8),
        dropout=0.0,
    )
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    # Determine which volumes to process
    shard_dir = cfg.data.shard_dir
    if not os.path.isabs(shard_dir):
        shard_dir = os.path.join(str(project_root), shard_dir)

    volume_index = scan_volume_index(shard_dir)

    # Filter volumes
    if filter_json is not None:
        with open(filter_json) as f:
            presence_preds = json.load(f)
        target_vols = set(
            sid for sid, pred in presence_preds.items()
            if pred.get("has_lesion_pred", 0) == 1
        )
        logger.info(f"Filtered to {len(target_vols)} positive volumes from {filter_json}")
    else:
        # Use all volumes with has_lesion=1
        target_vols = set(
            sid for sid, info in volume_index.items() if info["has_lesion"] == 1
        )
        logger.info(f"Using {len(target_vols)} positive volumes (ground truth)")

    use_cache = bool(cfg.data.get("cache_dir", ""))
    max_slices = cfg.classifier.get("max_slices", 32)

    if use_cache:
        cache_dir = cfg.data.cache_dir
        if not os.path.isabs(cache_dir):
            cache_dir = os.path.join(str(project_root), cache_dir)
        ds = CachedVolumeDataset(cache_dir, target_vols, max_slices=max_slices,
                                 class_mapping=class_mapping)
    else:
        ds = OnTheFlyVolumeDataset(
            volume_index, shard_dir, target_vols, max_slices=max_slices,
            class_mapping=class_mapping, img_size=cfg.backbone.img_size,
        )

    loader = DataLoader(
        ds, batch_size=cfg.data.get("batch_size", 16), shuffle=False,
        num_workers=cfg.data.get("num_workers", 4), collate_fn=collate_volumes,
    )

    # Backbone for on-the-fly
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
        probs = torch.softmax(out["class_logits"], dim=1).cpu()
        pred_classes = probs.argmax(dim=1)

        for i, sid in enumerate(labels["sample_id"]):
            class_probs = {inv_mapping[c]: float(probs[i, c]) for c in range(n_classes)}
            pred_class = inv_mapping[pred_classes[i].item()]
            gt_class_idx = labels["class_idx"][i].item()
            gt_class = inv_mapping.get(gt_class_idx, "unknown") if gt_class_idx >= 0 else "unknown"

            predictions[sid] = {
                "predicted_class": pred_class,
                "class_probabilities": class_probs,
                "gt_class": gt_class,
                "dataset": labels["dataset"][i],
            }

    with open(output_path, "w") as f:
        json.dump(predictions, f, indent=2)

    # Summary
    n_total = len(predictions)
    class_counts = {}
    for pred in predictions.values():
        c = pred["predicted_class"]
        class_counts[c] = class_counts.get(c, 0) + 1

    logger.info(f"Inference complete: {n_total} volumes")
    logger.info(f"Predicted class distribution: {class_counts}")
    logger.info(f"Predictions saved to {output_path}")

    # If ground truth available
    gt_known = [p for p in predictions.values() if p["gt_class"] != "unknown"]
    if gt_known:
        correct = sum(1 for p in gt_known if p["predicted_class"] == p["gt_class"])
        logger.info(f"Accuracy (known GT): {correct}/{len(gt_known)} = {correct/len(gt_known):.4f}")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Volume-level lesion subtype classifier (Class2/3/4)")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--inference", action="store_true",
                        help="Run inference mode")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Model checkpoint path (for inference)")
    parser.add_argument("--filter-json", type=str, default=None,
                        help="Presence predictions JSON to filter positive volumes")
    parser.add_argument("--output-json", type=str, default="subtype_predictions.json")
    parser.add_argument("opts", nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args()

    cli_opts = [o for o in args.opts if o and o != "--"]
    cfg = load_config(args.config, cli_opts)

    rank = int(os.environ.get("RANK", 0))
    output_dir = cfg.train.get("output_dir", "./outputs/volume_classifier_subtype")
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
        run_inference(cfg, args.checkpoint, args.output_json, args.filter_json)
    else:
        train_subtype(cfg)


if __name__ == "__main__":
    main()
