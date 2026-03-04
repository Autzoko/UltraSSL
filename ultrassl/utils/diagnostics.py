"""
Diagnostics and logging utilities for UltraSSL training.

- Loss curve logging to JSON
- Periodic embedding sanity checks (mean/std, cosine similarity, PCA variance)
- NN retrieval diversity check (collapse detection)
- Patch token diversity check (within-image representation diversity)
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

        # NN retrieval diversity check
        nn_record = self._check_nn_retrieval(cls_tokens)
        record.update(nn_record)

        # Patch diversity check
        patch_div_record = self._check_patch_diversity(patch_tokens)
        record.update(patch_div_record)

        with open(self.embed_log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

        logger.info(
            f"[embed check iter {iteration}] "
            f"cls_std={record.get('cls_std', 0):.4f}, "
            f"patch_std={record.get('patch_std', 0):.4f}, "
            f"cos_sim={record.get('cls_cos_sim_mean', 0):.4f}, "
            f"pca_top1={record.get('pca_top1_var_ratio', 0):.4f}, "
            f"nn_sim={record.get('nn_avg_sim', 0):.4f}, "
            f"patch_div={record.get('patch_intra_sim_mean', 0):.4f}"
        )

        # Alert on potential collapse
        if record.get("cls_cos_sim_mean", 0) > 0.95:
            logger.warning(f"[COLLAPSE WARNING iter {iteration}] "
                           f"CLS cosine similarity very high ({record['cls_cos_sim_mean']:.4f}). "
                           f"Features may be collapsing.")
        if record.get("patch_intra_sim_mean", 0) > 0.90:
            logger.warning(f"[COLLAPSE WARNING iter {iteration}] "
                           f"Patch intra-image similarity very high ({record['patch_intra_sim_mean']:.4f}). "
                           f"Patch tokens may be collapsing to uniform representation.")

    def _check_nn_retrieval(self, cls_tokens):
        """Check nearest-neighbor retrieval diversity from CLS tokens.

        For each probe image, find its top-5 nearest neighbors by cosine similarity.
        High avg similarity with low uniqueness indicates collapse.

        Returns dict of metrics to merge into the embedding record.
        """
        record = {}
        n = cls_tokens.shape[0]
        if n < 3:
            return record

        cls_normed = F.normalize(cls_tokens, dim=-1)
        cos_sim = torch.mm(cls_normed, cls_normed.t())

        # Zero out diagonal (self-similarity)
        cos_sim.fill_diagonal_(0.0)

        k = min(5, n - 1)
        topk_vals, topk_idx = cos_sim.topk(k, dim=1)

        # Average NN similarity (high = collapse)
        record["nn_avg_sim"] = topk_vals.mean().item()
        record["nn_min_sim"] = topk_vals.min().item()
        record["nn_max_sim"] = topk_vals.max().item()

        # Fraction of unique NNs (low = collapse, all pointing to same neighbor)
        unique_nns = len(set(topk_idx.flatten().tolist()))
        record["nn_unique_fraction"] = unique_nns / max(1, n * k)

        return record

    def _check_patch_diversity(self, patch_tokens):
        """Check within-image patch token diversity.

        For each probe image, compute pairwise cosine similarity of its patch tokens.
        High intra-image similarity means all patches produce the same representation.

        Returns dict of metrics to merge into the embedding record.
        """
        record = {}
        n_images = patch_tokens.shape[0]
        if n_images == 0:
            return record

        intra_sims = []
        for i in range(min(n_images, 8)):  # Cap to avoid slow computation
            patches = patch_tokens[i]  # (num_patches, D)
            patches_normed = F.normalize(patches, dim=-1)
            sim = torch.mm(patches_normed, patches_normed.t())
            # Off-diagonal mean
            mask = ~torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)
            intra_sims.append(sim[mask].mean().item())

        record["patch_intra_sim_mean"] = float(np.mean(intra_sims))
        record["patch_intra_sim_std"] = float(np.std(intra_sims))

        return record
