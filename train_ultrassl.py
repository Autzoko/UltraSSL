#!/usr/bin/env python3
"""
UltraSSL: DINOv2 domain-adaptive self-supervised pretraining for breast ultrasound.

Entry point for training. Config-driven via YAML.

Usage:
    # Single GPU
    python train_ultrassl.py --config config/ultrassl_vitb14.yaml

    # Multi-GPU (2 GPUs)
    torchrun --nproc_per_node=2 train_ultrassl.py --config config/ultrassl_vitb14.yaml

    # Override config values
    python train_ultrassl.py --config config/ultrassl_vitb14.yaml \
        optim.epochs=20 train.batch_size_per_gpu=16 model.pretrained=/path/to/weights.pth
"""

import argparse
import os
import sys
from pathlib import Path

# Ensure project root is on path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "dinov2"))

# Disable xformers — use PyTorch SDPA fallback
os.environ.setdefault("XFORMERS_DISABLED", "1")

from ultrassl.train.trainer import train, load_config, setup_distributed, setup_logging


def main():
    parser = argparse.ArgumentParser(
        description="UltraSSL: DINOv2 ultrasound pretraining",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config", required=True, type=str,
        help="Path to YAML config file (e.g., config/ultrassl_vitb14.yaml)",
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Start fresh training, ignore existing checkpoints",
    )
    parser.add_argument(
        "opts", nargs=argparse.REMAINDER, default=[],
        help="Override config values: key=value pairs (e.g., optim.epochs=20)",
    )

    args = parser.parse_args()

    # Parse CLI overrides (filter out empty strings and '--')
    cli_opts = [o for o in args.opts if o and o != "--"]

    # Setup distributed if available
    rank, _, _ = setup_distributed()

    # Load config
    cfg = load_config(args.config, cli_opts)

    # Setup logging
    setup_logging(cfg.train.output_dir, rank)

    # Optionally clear resume checkpoint
    if args.no_resume:
        latest = os.path.join(cfg.train.output_dir, "checkpoint_latest.pth")
        if os.path.exists(latest):
            os.remove(latest)

    # Train
    train(cfg)


if __name__ == "__main__":
    main()
