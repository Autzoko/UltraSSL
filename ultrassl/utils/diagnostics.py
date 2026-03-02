"""
Diagnostics and logging utilities for UltraSSL training.

- Loss curve logging to JSON
- Periodic embedding sanity checks (mean/std, cosine similarity, PCA variance)
"""

import json
import logging
import os
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger("ultrassl")


class DiagnosticsLogger:
    """Logs training metrics and performs periodic embedding sanity checks.

    Args:
        output_dir: Directory to write log files.
        embed_check_period: Run embedding checks every N iterations (0 to disable).
        num_probe_images: Number of fixed images for embedding checks.
    """

    def __init__(self, output_dir, embed_check_period=500, num_probe_images=32):
        self.output_dir = output_dir
        self.embed_check_period = embed_check_period
        self.num_probe_images = num_probe_images
        os.makedirs(output_dir, exist_ok=True)

        self.metrics_path = os.path.join(output_dir, "training_metrics.jsonl")
        self.embed_log_path = os.path.join(output_dir, "embedding_diagnostics.jsonl")

        self._probe_images = None
        self._running_stats = defaultdict(list)
        self._start_time = time.time()

    def log_iteration(self, iteration, loss_dict, lr, wd, mom, batch_size):
        """Log metrics for a single training iteration."""
        record = {
            "iteration": iteration,
            "timestamp": time.time() - self._start_time,
            "lr": lr,
            "wd": wd,
            "momentum": mom,
            "batch_size": batch_size,
        }

        total_loss = 0.0
        for k, v in loss_dict.items():
            val = v.item() if torch.is_tensor(v) else float(v)
            record[k] = val
            total_loss += val
        record["total_loss"] = total_loss

        with open(self.metrics_path, "a") as f:
            f.write(json.dumps(record) + "\n")

        # Track running stats for periodic summary
        for k, v in record.items():
            if isinstance(v, (int, float)) and k != "iteration":
                self._running_stats[k].append(v)

    def log_summary(self, iteration, window=100):
        """Log a summary of recent metrics."""
        summary = {"iteration": iteration}
        for k, vals in self._running_stats.items():
            recent = vals[-window:]
            if recent:
                summary[f"{k}_avg"] = np.mean(recent)
        logger.info(f"[iter {iteration}] " + ", ".join(
            f"{k}={v:.4f}" for k, v in summary.items() if k != "iteration"
        ))
        return summary

    def set_probe_images(self, dataset, device="cuda"):
        """Select fixed probe images for embedding diagnostics."""
        n = min(self.num_probe_images, len(dataset))
        indices = torch.randperm(len(dataset))[:n]
        images = []
        for idx in indices:
            img, _ = dataset[idx.item()]
            # img is the augmented dict; we need raw tensor
            if isinstance(img, dict):
                # Use the first global crop
                images.append(img["global_crops"][0])
            elif torch.is_tensor(img):
                images.append(img)
        if images:
            self._probe_images = torch.stack(images).to(device)
            logger.info(f"Set {len(images)} probe images for embedding diagnostics")

    @torch.no_grad()
    def check_embeddings(self, iteration, teacher_backbone, device="cuda"):
        """Run embedding sanity checks on fixed probe images."""
        if self.embed_check_period <= 0:
            return
        if iteration % self.embed_check_period != 0:
            return
        if self._probe_images is None:
            return

        teacher_backbone.eval()
        out = teacher_backbone(self._probe_images, is_training=True)
        cls_tokens = out["x_norm_clstoken"]  # (N, D)
        patch_tokens = out["x_norm_patchtokens"]  # (N, num_patches, D)

        record = {"iteration": iteration}

        # CLS token stats
        record["cls_mean"] = cls_tokens.mean().item()
        record["cls_std"] = cls_tokens.std().item()
        record["cls_norm_mean"] = cls_tokens.norm(dim=-1).mean().item()

        # Patch token stats
        record["patch_mean"] = patch_tokens.mean().item()
        record["patch_std"] = patch_tokens.std().item()
        record["patch_norm_mean"] = patch_tokens.norm(dim=-1).mean().item()

        # Cosine similarity between random CLS token pairs
        if cls_tokens.shape[0] >= 2:
            cls_normed = F.normalize(cls_tokens, dim=-1)
            cos_sim = torch.mm(cls_normed, cls_normed.t())
            # Off-diagonal elements
            mask = ~torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
            cos_vals = cos_sim[mask]
            record["cls_cos_sim_mean"] = cos_vals.mean().item()
            record["cls_cos_sim_std"] = cos_vals.std().item()

        # PCA variance ratio of patch embeddings (collapse check)
        flat_patches = patch_tokens.reshape(-1, patch_tokens.shape[-1])
        if flat_patches.shape[0] > 10:
            # Sample if too many patches
            if flat_patches.shape[0] > 1000:
                idx = torch.randperm(flat_patches.shape[0])[:1000]
                flat_patches = flat_patches[idx]
            centered = flat_patches - flat_patches.mean(0, keepdim=True)
            try:
                _, S, _ = torch.svd_lowrank(centered.float(), q=min(10, centered.shape[-1]))
                var_ratios = (S ** 2) / (S ** 2).sum()
                record["pca_top1_var_ratio"] = var_ratios[0].item()
                record["pca_top5_var_ratio"] = var_ratios[:5].sum().item()
            except Exception:
                pass  # SVD can fail with bad numerics early in training

        with open(self.embed_log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

        logger.info(
            f"[embed check iter {iteration}] "
            f"cls_std={record.get('cls_std', 0):.4f}, "
            f"patch_std={record.get('patch_std', 0):.4f}, "
            f"cos_sim={record.get('cls_cos_sim_mean', 0):.4f}, "
            f"pca_top1={record.get('pca_top1_var_ratio', 0):.4f}"
        )
