#!/usr/bin/env python3
"""
Pre-extract DINO CLS token embeddings from all volumes in WebDataset shards.

Creates a per-volume cache of CLS token embeddings for fast training of
volume-level classifiers. Each volume is saved as a .pt file with shape
(N_slices, embed_dim).

Usage:
    # Single GPU:
    python extract_embeddings.py \
        --config config/volume_classifier.yaml \
        --output-dir ./embedding_cache

    # Multi-GPU (each GPU processes its shard subset):
    torchrun --nproc_per_node=4 extract_embeddings.py \
        --config config/volume_classifier.yaml \
        --output-dir ./embedding_cache
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch

# Ensure project root is on path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "dinov2"))

from omegaconf import OmegaConf

from ultrassl.data.volume_dataset import extract_and_cache_embeddings

logger = logging.getLogger("ultrassl")


def main():
    parser = argparse.ArgumentParser(
        description="Extract DINO CLS embeddings from WebDataset shards")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--output-dir", required=True,
                        help="Directory to save embedding cache")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for backbone forward pass")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("opts", nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args()

    # Load config
    cfg = OmegaConf.load(args.config)
    cli_opts = [o for o in args.opts if o and o != "--"]
    if cli_opts:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(cli_opts))

    # Setup logging
    rank = int(os.environ.get("RANK", 0))
    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    # Device
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    elif torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # Resolve paths
    shard_dir = cfg.data.shard_dir
    if not os.path.isabs(shard_dir):
        shard_dir = os.path.join(str(project_root), shard_dir)

    backbone_ckpt = cfg.backbone.checkpoint
    if not os.path.isabs(backbone_ckpt):
        backbone_ckpt = os.path.join(str(project_root), backbone_ckpt)

    datasets_filter = cfg.data.get("datasets", None)
    if datasets_filter is not None:
        datasets_filter = list(datasets_filter)

    logger.info(f"Extracting embeddings: {shard_dir} -> {args.output_dir}")
    logger.info(f"Backbone: {backbone_ckpt}")
    logger.info(f"Device: {device}, batch_size: {args.batch_size}")

    extract_and_cache_embeddings(
        shard_dir=shard_dir,
        backbone_checkpoint=backbone_ckpt,
        output_dir=args.output_dir,
        arch=cfg.backbone.arch,
        patch_size=cfg.backbone.patch_size,
        img_size=cfg.backbone.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        datasets=datasets_filter,
    )


if __name__ == "__main__":
    main()
