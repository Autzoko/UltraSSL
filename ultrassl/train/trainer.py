"""
Main training loop for UltraSSL DINOv2 domain-adaptive pretraining.

Config-driven, supports single-GPU and DDP multi-GPU.
Uses standard PyTorch AMP (no FSDP/xformers required).
"""

import argparse
import logging
import math
import os
import sys
from functools import partial
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler
from omegaconf import OmegaConf

# Ensure dinov2 is importable
_project_root = Path(__file__).resolve().parent.parent.parent
_dinov2_root = _project_root / "dinov2"
if str(_dinov2_root) not in sys.path:
    sys.path.insert(0, str(_dinov2_root))

os.environ.setdefault("XFORMERS_DISABLED", "1")

from dinov2.data.collate import collate_data_and_cast
from dinov2.data.masking import MaskingGenerator
from dinov2.utils.utils import CosineScheduler

from ultrassl.data.dataset import UltrasoundDataset
from ultrassl.data.augmentations import UltrasoundAugmentationDINO
from ultrassl.models.ssl_meta_arch import UltraSSLMetaArch
from ultrassl.utils.diagnostics import DiagnosticsLogger

# WebDataset is optional — only needed when using shard-based loading
try:
    from ultrassl.data.wds_dataset import build_wds_dataset
    from ultrassl.data.wds_labeled_dataset import build_labeled_wds_dataset
    HAS_WEBDATASET = True
except ImportError:
    HAS_WEBDATASET = False

logger = logging.getLogger("ultrassl")


def setup_logging(output_dir, rank=0):
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, f"train_rank{rank}.log")

    handlers = [logging.StreamHandler()]
    if rank == 0:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=handlers,
    )


def setup_distributed():
    """Initialize DDP if launched via torchrun."""
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    return 0, 1, 0


def load_config(config_path, cli_opts=None):
    """Load and merge YAML config."""
    cfg = OmegaConf.load(config_path)

    # Apply CLI overrides
    if cli_opts:
        cli_cfg = OmegaConf.from_dotlist(cli_opts)
        cfg = OmegaConf.merge(cfg, cli_cfg)

    # Compute derived values
    if cfg.optim.get("lr", 0) == 0:
        # Use base_lr directly — it's already tuned for domain adaptation / fine-tuning.
        # Sqrt batch-size scaling is only appropriate for from-scratch training at bs=1024+.
        cfg.optim.lr = cfg.optim.base_lr
        logger.info(f"Using LR: {cfg.optim.lr:.6f} (base_lr, no batch-size scaling)")

    return cfg


def build_schedulers(cfg):
    """Build cosine schedulers for LR, WD, momentum, and teacher temp."""
    epoch_len = cfg.train.OFFICIAL_EPOCH_LENGTH
    total_iters = cfg.optim.epochs * epoch_len

    lr_schedule = CosineScheduler(
        base_value=cfg.optim.lr,
        final_value=cfg.optim.min_lr,
        total_iters=total_iters,
        warmup_iters=cfg.optim.warmup_epochs * epoch_len,
        start_warmup_value=0,
    )

    wd_schedule = CosineScheduler(
        base_value=cfg.optim.weight_decay,
        final_value=cfg.optim.weight_decay_end,
        total_iters=total_iters,
    )

    momentum_schedule = CosineScheduler(
        base_value=cfg.teacher.momentum_teacher,
        final_value=cfg.teacher.final_momentum_teacher,
        total_iters=total_iters,
    )

    teacher_temp_schedule = CosineScheduler(
        base_value=cfg.teacher.teacher_temp,
        final_value=cfg.teacher.teacher_temp,
        total_iters=cfg.teacher.warmup_teacher_temp_epochs * epoch_len,
        warmup_iters=cfg.teacher.warmup_teacher_temp_epochs * epoch_len,
        start_warmup_value=cfg.teacher.warmup_teacher_temp,
    )

    last_layer_lr_schedule = CosineScheduler(
        base_value=cfg.optim.lr,
        final_value=cfg.optim.min_lr,
        total_iters=total_iters,
        warmup_iters=cfg.optim.warmup_epochs * epoch_len,
        start_warmup_value=0,
    )
    # Freeze last layer during early training
    freeze_iters = cfg.optim.freeze_last_layer_epochs * epoch_len
    last_layer_lr_schedule.schedule[:freeze_iters] = 0

    return lr_schedule, wd_schedule, momentum_schedule, teacher_temp_schedule, last_layer_lr_schedule


def apply_optim_scheduler(optimizer, lr, wd, last_layer_lr):
    """Apply scheduled LR and WD to optimizer param groups."""
    for param_group in optimizer.param_groups:
        is_last = param_group.get("is_last_layer", False)
        lr_mult = param_group.get("lr_multiplier", 1.0)
        wd_mult = param_group.get("wd_multiplier", 1.0)
        param_group["weight_decay"] = wd * wd_mult
        param_group["lr"] = (last_layer_lr if is_last else lr) * lr_mult


def save_checkpoint(model, optimizer, iteration, cfg, is_best=False):
    """Save training checkpoint and teacher backbone."""
    output_dir = cfg.train.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Full training state
    ckpt = {
        "iteration": iteration,
        "student": model.student.state_dict(),
        "teacher": model.teacher.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    ckpt_path = os.path.join(output_dir, f"checkpoint_{iteration:07d}.pth")
    torch.save(ckpt, ckpt_path)

    # Always save latest
    latest_path = os.path.join(output_dir, "checkpoint_latest.pth")
    torch.save(ckpt, latest_path)

    # Export teacher backbone only (the main output for downstream use)
    teacher_backbone_state = {}
    for k, v in model.teacher.backbone.state_dict().items():
        teacher_backbone_state[k] = v
    teacher_path = os.path.join(output_dir, f"teacher_backbone_{iteration:07d}.pth")
    torch.save({"model": teacher_backbone_state, "iteration": iteration}, teacher_path)

    # Keep symlink to latest teacher
    latest_teacher = os.path.join(output_dir, "teacher_backbone_latest.pth")
    if os.path.islink(latest_teacher) or os.path.exists(latest_teacher):
        os.remove(latest_teacher)
    os.symlink(os.path.basename(teacher_path), latest_teacher)

    logger.info(f"Saved checkpoint at iteration {iteration}: {ckpt_path}")

    # Cleanup old checkpoints
    max_to_keep = cfg.checkpoint.get("max_to_keep", 3)
    ckpts = sorted(
        [f for f in os.listdir(output_dir) if f.startswith("checkpoint_") and f.endswith(".pth") and f != "checkpoint_latest.pth"],
    )
    while len(ckpts) > max_to_keep:
        old = ckpts.pop(0)
        old_path = os.path.join(output_dir, old)
        os.remove(old_path)
        # Also remove corresponding teacher
        old_teacher = old.replace("checkpoint_", "teacher_backbone_")
        old_teacher_path = os.path.join(output_dir, old_teacher)
        if os.path.exists(old_teacher_path):
            os.remove(old_teacher_path)


def resume_from_checkpoint(model, optimizer, cfg):
    """Resume training from latest checkpoint if available."""
    latest_path = os.path.join(cfg.train.output_dir, "checkpoint_latest.pth")
    if not os.path.isfile(latest_path):
        return 0

    logger.info(f"Resuming from checkpoint: {latest_path}")
    ckpt = torch.load(latest_path, map_location="cpu", weights_only=False)

    # Load model weights (handle DDP wrapper)
    model_to_load = model.module if isinstance(model, DDP) else model
    model_to_load.student.load_state_dict(ckpt["student"])
    model_to_load.teacher.load_state_dict(ckpt["teacher"])
    optimizer.load_state_dict(ckpt["optimizer"])

    start_iter = ckpt["iteration"] + 1
    logger.info(f"Resumed from iteration {start_iter}")
    return start_iter


def get_device(local_rank=0):
    """Select best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device(f"cuda:{local_rank}")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def train(cfg):
    """Main training function."""
    rank, world_size, local_rank = setup_distributed()
    device = get_device(local_rank)
    is_main = rank == 0

    setup_logging(cfg.train.output_dir, rank)
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    logger.info(f"Device: {device}, World size: {world_size}")

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    # ================================================================
    # BUILD MODEL
    # ================================================================
    model = UltraSSLMetaArch(cfg).to(device)

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    raw_model = model.module if isinstance(model, DDP) else model

    # ================================================================
    # BUILD OPTIMIZER
    # ================================================================
    param_groups = raw_model.get_params_groups()
    optimizer = torch.optim.AdamW(
        param_groups,
        betas=(cfg.optim.get("adamw_beta1", 0.9), cfg.optim.get("adamw_beta2", 0.999)),
    )

    # ================================================================
    # BUILD SCHEDULERS
    # ================================================================
    lr_sched, wd_sched, mom_sched, temp_sched, last_layer_lr_sched = build_schedulers(cfg)

    # ================================================================
    # BUILD DATA PIPELINE
    # ================================================================
    aug_cfg = cfg.get("augmentation", None)
    data_transform = UltrasoundAugmentationDINO(
        global_crops_scale=list(cfg.crops.global_crops_scale),
        local_crops_scale=list(cfg.crops.local_crops_scale),
        local_crops_number=cfg.crops.local_crops_number,
        global_crops_size=cfg.crops.global_crops_size,
        local_crops_size=cfg.crops.local_crops_size,
        aug_cfg=aug_cfg,
    )

    data_cfg = cfg.get("data", {})

    # Mask generator for iBOT
    img_size = cfg.crops.global_crops_size
    patch_size = cfg.model.patch_size
    n_tokens = (img_size // patch_size) ** 2
    mask_generator = MaskingGenerator(
        input_size=(img_size // patch_size, img_size // patch_size),
        max_num_patches=0.5 * img_size // patch_size * img_size // patch_size,
    )

    # Use float16 on CUDA, float32 on MPS/CPU (MPS doesn't support float16 well)
    collate_dtype = torch.half if device.type == "cuda" else torch.float32

    collate_fn = partial(
        collate_data_and_cast,
        mask_ratio_tuple=list(cfg.ibot.mask_ratio_min_max),
        mask_probability=cfg.ibot.mask_sample_probability,
        n_tokens=n_tokens,
        mask_generator=mask_generator,
        dtype=collate_dtype,
    )

    # Choose data loading mode: WebDataset shards or raw image files
    shard_dir = data_cfg.get("shard_dir", "")
    use_webdataset = bool(shard_dir) and HAS_WEBDATASET

    if use_webdataset:
        # --- WebDataset mode: read from .tar shards ---
        if not os.path.isabs(shard_dir):
            shard_dir = os.path.join(str(_project_root), shard_dir)
        logger.info(f"Using WebDataset shards from: {shard_dir}")

        data_mode = data_cfg.get("mode", "unlabeled")
        if data_mode == "ssl":
            # Labeled shards in SSL mode (ignore annotations, use all slices)
            logger.info("Using labeled shards in SSL mode (ignoring annotations)")
            wds_dataset, wds_loader = build_labeled_wds_dataset(
                shard_dir=shard_dir,
                mode="ssl",
                transform=data_transform,
                epoch_length=cfg.train.OFFICIAL_EPOCH_LENGTH,
                batch_size=cfg.train.batch_size_per_gpu,
                num_workers=cfg.train.num_workers,
                world_size=world_size,
                rank=rank,
                balance_datasets=data_cfg.get("balance_datasets", True),
                pos_enrichment=data_cfg.get("pos_enrichment", 0.0),
            )
        else:
            # Standard unlabeled shards
            wds_dataset, wds_loader = build_wds_dataset(
                shard_dir=shard_dir,
                transform=data_transform,
                epoch_length=cfg.train.OFFICIAL_EPOCH_LENGTH,
                batch_size=cfg.train.batch_size_per_gpu,
                num_workers=cfg.train.num_workers,
                world_size=world_size,
                rank=rank,
            )

        # Wrap with collation — WebDataset yields individual samples,
        # we batch + collate them the same way as the standard pipeline
        data_loader = DataLoader(
            wds_dataset,
            batch_size=cfg.train.batch_size_per_gpu,
            num_workers=cfg.train.num_workers,
            collate_fn=collate_fn,
            drop_last=True,
            pin_memory=device.type == "cuda",
            persistent_workers=cfg.train.num_workers > 0,
        )
        dataset_size_str = f"WebDataset shards in {shard_dir}"
    else:
        # --- Standard mode: read raw image files ---
        if shard_dir and not HAS_WEBDATASET:
            logger.warning("shard_dir is set but webdataset is not installed. "
                           "Falling back to raw file loading. Install: pip install webdataset")

        data_root_json = data_cfg.get("data_root_json", "config/data_root.json")
        if not os.path.isabs(data_root_json):
            data_root_json = os.path.join(str(_project_root), data_root_json)

        dataset = UltrasoundDataset(
            data_root_json=data_root_json,
            transform=data_transform,
            target_transform=lambda _: (),
            volume_slice_stride=data_cfg.get("volume_slice_stride", 3),
            min_slice_entropy=data_cfg.get("min_slice_entropy", 0.0),
        )

        if len(dataset) == 0:
            logger.warning("Dataset is empty! Check data paths. Running in dry-run mode.")

        if world_size > 1:
            sampler = DistributedSampler(dataset, shuffle=True)
        else:
            sampler = RandomSampler(dataset)

        data_loader = DataLoader(
            dataset,
            batch_size=cfg.train.batch_size_per_gpu,
            sampler=sampler,
            num_workers=cfg.train.num_workers,
            collate_fn=collate_fn,
            drop_last=True,
            pin_memory=device.type == "cuda",
            persistent_workers=cfg.train.num_workers > 0,
        )
        dataset_size_str = f"{len(dataset)} images"

    # ================================================================
    # RESUME
    # ================================================================
    start_iter = resume_from_checkpoint(raw_model, optimizer, cfg)

    # ================================================================
    # DIAGNOSTICS
    # ================================================================
    diagnostics = DiagnosticsLogger(
        output_dir=cfg.train.output_dir,
        embed_check_period=cfg.get("diagnostics", {}).get("embed_check_period", 500),
        num_probe_images=cfg.get("diagnostics", {}).get("num_probe_images", 32),
    )

    # ================================================================
    # TRAINING LOOP
    # ================================================================
    epoch_len = cfg.train.OFFICIAL_EPOCH_LENGTH
    max_iter = cfg.optim.epochs * epoch_len
    save_period = cfg.checkpoint.get("save_period_epochs", 5) * epoch_len

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    raw_model.train()
    iteration = start_iter

    logger.info(f"Starting training: {start_iter} -> {max_iter} iterations")
    logger.info(f"  Dataset: {dataset_size_str}, Batch size: {cfg.train.batch_size_per_gpu}")
    logger.info(f"  Epoch length: {epoch_len}, Total epochs: {cfg.optim.epochs}")
    logger.info(f"  AMP: {'enabled' if use_amp else 'disabled (non-CUDA device)'}")

    data_iter = iter(data_loader)

    while iteration < max_iter:
        # Get next batch (infinite loop over data)
        try:
            data = next(data_iter)
        except StopIteration:
            if world_size > 1:
                epoch = iteration // epoch_len
                sampler.set_epoch(epoch)
            data_iter = iter(data_loader)
            data = next(data_iter)

        current_batch_size = data["collated_global_crops"].shape[0] / 2

        # Apply schedules
        lr = lr_sched[iteration]
        wd = wd_sched[iteration]
        mom = mom_sched[iteration]
        teacher_temp = temp_sched[iteration]
        last_layer_lr = last_layer_lr_sched[iteration]
        apply_optim_scheduler(optimizer, lr, wd, last_layer_lr)

        # Forward-backward (AMP on CUDA only)
        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.amp.autocast("cuda", dtype=torch.float16):
                loss, loss_dict = raw_model.forward_backward(data, teacher_temp=teacher_temp)
            scaler.scale(loss).backward()
            if cfg.optim.clip_grad:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(raw_model.student.parameters(), cfg.optim.clip_grad)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss, loss_dict = raw_model.forward_backward(data, teacher_temp=teacher_temp)
            loss.backward()
            if cfg.optim.clip_grad:
                torch.nn.utils.clip_grad_norm_(raw_model.student.parameters(), cfg.optim.clip_grad)
            optimizer.step()

        # EMA teacher update
        raw_model.update_teacher(mom)

        # --- Logging ---
        if world_size > 1:
            for v in loss_dict.values():
                if torch.is_tensor(v):
                    dist.all_reduce(v)
        loss_dict_reduced = {
            k: (v.item() / world_size if torch.is_tensor(v) else v)
            for k, v in loss_dict.items()
        }

        if math.isnan(sum(loss_dict_reduced.values())):
            logger.error(f"NaN detected at iteration {iteration}!")
            if is_main:
                save_checkpoint(raw_model, optimizer, iteration, cfg)
            raise RuntimeError("NaN loss detected")

        if is_main:
            diagnostics.log_iteration(
                iteration, loss_dict_reduced, lr, wd, mom, current_batch_size
            )

            if iteration % 10 == 0:
                losses_str = ", ".join(f"{k}={v:.4f}" for k, v in loss_dict_reduced.items())
                total = sum(loss_dict_reduced.values())
                logger.info(
                    f"[iter {iteration}/{max_iter}] loss={total:.4f} ({losses_str}) "
                    f"lr={lr:.6f} mom={mom:.4f}"
                )

            # Embedding diagnostics
            diagnostics.check_embeddings(iteration, raw_model.teacher.backbone, device)

        # --- Checkpointing ---
        if is_main and iteration > 0 and iteration % save_period == 0:
            save_checkpoint(raw_model, optimizer, iteration, cfg)

        iteration += 1

    # Final save
    if is_main:
        save_checkpoint(raw_model, optimizer, iteration, cfg)
        logger.info("Training complete!")

    if world_size > 1:
        dist.destroy_process_group()
