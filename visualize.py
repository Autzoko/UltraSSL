#!/usr/bin/env python3
"""
Visualize learned features from a trained UltraSSL teacher backbone.

Produces DINO/DINOv2-style visualizations:
1. Self-attention maps — CLS token attention to all patches (per head + mean)
2. PCA of patch tokens — 3 principal components mapped to RGB
3. Cosine similarity map — query patch similarity to all other patches

Usage:
    python visualize.py \
        --checkpoint outputs/ultrassl_vitb14/teacher_backbone_latest.pth \
        --images /path/to/img1.png /path/to/img2.png \
        --output-dir vis_output/

    # Or point to a directory of images:
    python visualize.py \
        --checkpoint outputs/ultrassl_vitb14/teacher_backbone_latest.pth \
        --image-dir /path/to/ultrasound/images/ \
        --num-images 8 \
        --output-dir vis_output/
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# Setup paths
_project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(_project_root / "dinov2"))
os.environ.setdefault("XFORMERS_DISABLED", "1")

from ultrassl.models.backbone import build_backbone

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("visualize")


# ── Image loading ──────────────────────────────────────────────────────

def load_image(path, img_size=224):
    """Load an image and return (original_pil, preprocessed_tensor)."""
    img = Image.open(path).convert("RGB")
    preprocess = transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tensor = preprocess(img).unsqueeze(0)  # (1, 3, H, W)
    # Also keep a resized original for overlay
    original = img.resize((img_size, img_size), Image.BICUBIC)
    return original, tensor


def find_images(image_dir, extensions=(".png", ".jpg", ".jpeg", ".bmp", ".tif"), max_images=8):
    """Find image files in a directory (non-recursive, sorted)."""
    paths = []
    for f in sorted(os.listdir(image_dir)):
        if Path(f).suffix.lower() in extensions:
            paths.append(os.path.join(image_dir, f))
            if len(paths) >= max_images:
                break
    return paths


# ── Attention extraction ───────────────────────────────────────────────

class AttentionExtractor:
    """Hook into the last attention block to capture Q, K and compute attention weights."""

    def __init__(self, backbone):
        self.attn_weights = None
        # Register hook on the last block's attention module
        last_block = backbone.blocks[-1]
        last_block.attn.qkv.register_forward_hook(self._hook_qkv)
        self._num_heads = last_block.attn.num_heads

    def _hook_qkv(self, module, input, output):
        """Capture QKV output and compute attention weights manually."""
        # output shape: (B, N, 3 * dim)
        B, N, _ = output.shape
        dim = _ // 3
        head_dim = dim // self._num_heads

        qkv = output.reshape(B, N, 3, self._num_heads, head_dim)
        q, k, _ = qkv.unbind(2)  # each: (B, N, num_heads, head_dim)
        q = q.transpose(1, 2)  # (B, num_heads, N, head_dim)
        k = k.transpose(1, 2)

        scale = head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale  # (B, num_heads, N, N)
        attn = attn.softmax(dim=-1)
        self.attn_weights = attn.detach().cpu()

    def get_cls_attention(self):
        """Return CLS token's attention to all patch tokens.

        Returns: (num_heads, num_patches) array.
        """
        if self.attn_weights is None:
            return None
        # attn shape: (1, num_heads, N, N) where N = 1 (CLS) + num_patches
        # CLS token is index 0, patch tokens start at index 1
        cls_attn = self.attn_weights[0, :, 0, 1:]  # (num_heads, num_patches)
        return cls_attn.numpy()


# ── Visualization functions ────────────────────────────────────────────

def visualize_attention_maps(cls_attn, grid_size, original_img, output_path):
    """Draw self-attention heatmaps (per head + mean), overlaid on the original image.

    Args:
        cls_attn: (num_heads, num_patches) attention weights.
        grid_size: (H, W) patch grid dimensions.
        original_img: PIL Image (resized to model input size).
        output_path: Where to save the figure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    num_heads = cls_attn.shape[0]
    # Reshape to spatial grid
    attn_maps = cls_attn.reshape(num_heads, *grid_size)  # (num_heads, gh, gw)
    mean_attn = attn_maps.mean(axis=0)  # (gh, gw)

    # Layout: original + mean + each head
    n_cols = min(num_heads + 2, 7)
    n_rows = (num_heads + 2 + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    for ax in axes.flat:
        ax.axis("off")

    # Original image
    axes[0, 0].imshow(original_img)
    axes[0, 0].set_title("Original", fontsize=10)

    # Mean attention
    ax = axes[0, 1]
    ax.imshow(original_img, alpha=0.4)
    im = ax.imshow(
        _upsample_attn(mean_attn, original_img.size[0]),
        cmap="inferno", alpha=0.6, interpolation="bilinear",
    )
    ax.set_title("Mean attn", fontsize=10)

    # Per-head attention
    for i in range(num_heads):
        row = (i + 2) // n_cols
        col = (i + 2) % n_cols
        if row >= n_rows:
            break
        ax = axes[row, col]
        ax.imshow(original_img, alpha=0.4)
        ax.imshow(
            _upsample_attn(attn_maps[i], original_img.size[0]),
            cmap="inferno", alpha=0.6, interpolation="bilinear",
        )
        ax.set_title(f"Head {i}", fontsize=9)

    plt.suptitle("Self-Attention Maps (CLS → patches)", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Attention maps → {output_path}")


def _upsample_attn(attn_map, target_size):
    """Upsample a small attention map to the target image size."""
    t = torch.from_numpy(attn_map).float().unsqueeze(0).unsqueeze(0)
    t = F.interpolate(t, size=(target_size, target_size), mode="bilinear", align_corners=False)
    return t.squeeze().numpy()


def visualize_pca(patch_tokens_list, grid_size, original_imgs, output_path):
    """PCA visualization of patch tokens mapped to RGB.

    Computes PCA across all images jointly so colors are consistent.

    Args:
        patch_tokens_list: List of (num_patches, dim) arrays, one per image.
        grid_size: (H, W) patch grid dimensions.
        original_imgs: List of PIL Images.
        output_path: Where to save the figure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Stack all patch tokens for joint PCA
    all_tokens = np.concatenate(patch_tokens_list, axis=0)  # (total_patches, dim)
    all_tokens = all_tokens - all_tokens.mean(axis=0, keepdims=True)

    # PCA via SVD
    U, S, Vt = np.linalg.svd(all_tokens, full_matrices=False)
    components = Vt[:3]  # top 3 principal components

    n_images = len(patch_tokens_list)
    fig, axes = plt.subplots(2, n_images, figsize=(3 * n_images, 6))
    if n_images == 1:
        axes = axes[:, np.newaxis]

    offset = 0
    for i in range(n_images):
        n_patches = patch_tokens_list[i].shape[0]
        tokens = patch_tokens_list[i]
        tokens_centered = tokens - all_tokens.mean(axis=0, keepdims=True)

        # Project to 3 PCA components
        proj = tokens_centered @ components.T  # (num_patches, 3)
        # Normalize each component to [0, 1] for RGB
        for c in range(3):
            mn, mx = proj[:, c].min(), proj[:, c].max()
            if mx > mn:
                proj[:, c] = (proj[:, c] - mn) / (mx - mn)
            else:
                proj[:, c] = 0.5

        pca_img = proj.reshape(*grid_size, 3)
        # Upsample to image size
        pca_img_up = np.array(Image.fromarray(
            (pca_img * 255).astype(np.uint8)
        ).resize(original_imgs[i].size, Image.BILINEAR)) / 255.0

        axes[0, i].imshow(original_imgs[i])
        axes[0, i].set_title("Original", fontsize=10)
        axes[0, i].axis("off")

        axes[1, i].imshow(pca_img_up)
        axes[1, i].set_title("PCA features", fontsize=10)
        axes[1, i].axis("off")

        offset += n_patches

    plt.suptitle("Patch Token PCA (top 3 components → RGB)", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  PCA visualization → {output_path}")


def visualize_similarity(patch_tokens, grid_size, original_img, output_path,
                         query_points=None):
    """Cosine similarity map from query patches to all other patches.

    Args:
        patch_tokens: (num_patches, dim) array.
        grid_size: (H, W) patch grid dimensions.
        original_img: PIL Image.
        output_path: Where to save.
        query_points: List of (row, col) in patch grid. Defaults to center + corners.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gh, gw = grid_size
    if query_points is None:
        query_points = [
            (gh // 2, gw // 2),          # center
            (gh // 4, gw // 4),          # upper-left region
            (3 * gh // 4, 3 * gw // 4),  # lower-right region
        ]

    # Normalize for cosine similarity
    tokens_normed = patch_tokens / (np.linalg.norm(patch_tokens, axis=1, keepdims=True) + 1e-8)

    n_queries = len(query_points)
    fig, axes = plt.subplots(1, n_queries + 1, figsize=(3.5 * (n_queries + 1), 3.5))

    axes[0].imshow(original_img)
    # Mark query points on the original
    for qr, qc in query_points:
        pixel_r = int((qr + 0.5) / gh * original_img.size[1])
        pixel_c = int((qc + 0.5) / gw * original_img.size[0])
        axes[0].plot(pixel_c, pixel_r, "x", markersize=10, markeredgewidth=2)
    axes[0].set_title("Query points", fontsize=10)
    axes[0].axis("off")

    for i, (qr, qc) in enumerate(query_points):
        query_idx = qr * gw + qc
        sim = tokens_normed @ tokens_normed[query_idx]  # (num_patches,)
        sim_map = sim.reshape(gh, gw)

        ax = axes[i + 1]
        ax.imshow(original_img, alpha=0.3)
        ax.imshow(
            _upsample_attn(sim_map, original_img.size[0]),
            cmap="coolwarm", alpha=0.7, vmin=-0.5, vmax=1.0,
            interpolation="bilinear",
        )
        ax.set_title(f"Sim from ({qr},{qc})", fontsize=9)
        ax.axis("off")

    plt.suptitle("Cosine Similarity Maps", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Similarity maps → {output_path}")


# ── Main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Visualize UltraSSL learned features")
    parser.add_argument("--checkpoint", required=True, help="Path to teacher_backbone_*.pth")
    parser.add_argument("--images", nargs="*", help="Paths to individual images")
    parser.add_argument("--image-dir", help="Directory containing images")
    parser.add_argument("--num-images", type=int, default=8, help="Max images from --image-dir")
    parser.add_argument("--output-dir", default="vis_output", help="Output directory")
    parser.add_argument("--arch", default="vit_base", help="Model architecture")
    parser.add_argument("--patch-size", type=int, default=14, help="Patch size")
    parser.add_argument("--img-size", type=int, default=224, help="Input image size")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Collect image paths
    image_paths = []
    if args.images:
        image_paths.extend(args.images)
    if args.image_dir:
        image_paths.extend(find_images(args.image_dir, max_images=args.num_images))
    if not image_paths:
        parser.error("Provide --images or --image-dir")

    logger.info(f"Visualizing {len(image_paths)} images")

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone, embed_dim = build_backbone(
        model_name=args.arch,
        patch_size=args.patch_size,
        pretrained=args.checkpoint,
        img_size=args.img_size,
        drop_path_rate=0.0,
    )
    backbone.eval().to(device)

    # Setup attention extraction
    attn_extractor = AttentionExtractor(backbone)

    grid_h = grid_w = args.img_size // args.patch_size  # e.g., 16 for 224/14
    grid_size = (grid_h, grid_w)
    logger.info(f"Patch grid: {grid_h}x{grid_w} = {grid_h * grid_w} patches")

    # Process each image
    all_patch_tokens = []
    all_originals = []

    for img_path in image_paths:
        name = Path(img_path).stem
        logger.info(f"Processing: {name}")

        original, tensor = load_image(img_path, args.img_size)
        tensor = tensor.to(device)

        with torch.no_grad():
            out = backbone(tensor, is_training=True)

        cls_token = out["x_norm_clstoken"].cpu().numpy()    # (1, dim)
        patch_tokens = out["x_norm_patchtokens"].cpu().numpy()[0]  # (num_patches, dim)

        all_patch_tokens.append(patch_tokens)
        all_originals.append(original)

        # 1. Attention maps
        cls_attn = attn_extractor.get_cls_attention()
        if cls_attn is not None:
            visualize_attention_maps(
                cls_attn, grid_size, original,
                os.path.join(args.output_dir, f"{name}_attention.png"),
            )

        # 2. Cosine similarity
        visualize_similarity(
            patch_tokens, grid_size, original,
            os.path.join(args.output_dir, f"{name}_similarity.png"),
        )

    # 3. Joint PCA across all images
    if all_patch_tokens:
        visualize_pca(
            all_patch_tokens, grid_size, all_originals,
            os.path.join(args.output_dir, "pca_features.png"),
        )

    logger.info(f"All visualizations saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
