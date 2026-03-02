"""
Swappable ViT backbone loader with DINOv2 pretrained weight support.

Uses the DinoVisionTransformer from the dinov2 package directly.
DINOv2 official weights use patch_size=14 (not 16).
"""

import logging
import os
import sys
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn

logger = logging.getLogger("ultrassl")

# Ensure dinov2 is importable
_project_root = Path(__file__).resolve().parent.parent.parent
_dinov2_root = _project_root / "dinov2"
if str(_dinov2_root) not in sys.path:
    sys.path.insert(0, str(_dinov2_root))

# Disable xformers to avoid hard dependency — PyTorch SDPA is used as fallback
os.environ.setdefault("XFORMERS_DISABLED", "1")

from dinov2.models.vision_transformer import DinoVisionTransformer
from dinov2.layers import NestedTensorBlock as Block, MemEffAttention


# Architecture specs: (embed_dim, depth, num_heads, mlp_ratio)
VIT_SPECS = {
    "vit_small":  (384,  12, 6,  4),
    "vit_base":   (768,  12, 12, 4),
    "vit_large":  (1024, 24, 16, 4),
    "vit_giant2": (1536, 40, 24, 4),
}

# Map friendly names to torch hub model IDs
DINOV2_HUB_MODELS = {
    "dinov2_vits14": "facebookresearch/dinov2:dinov2_vits14",
    "dinov2_vitb14": "facebookresearch/dinov2:dinov2_vitb14",
    "dinov2_vitl14": "facebookresearch/dinov2:dinov2_vitl14",
    "dinov2_vitg14": "facebookresearch/dinov2:dinov2_vitg14",
}


def build_backbone(
    model_name: str = "vit_base",
    patch_size: int = 14,
    pretrained: str = "dinov2_vitb14",
    img_size: int = 224,
    drop_path_rate: float = 0.1,
    init_values: float = 1e-5,
    **kwargs,
):
    """Build a ViT backbone, optionally loading DINOv2 pretrained weights.

    Args:
        model_name: One of 'vit_small', 'vit_base', 'vit_large', 'vit_giant2'.
        patch_size: Patch size (14 for DINOv2 official weights).
        pretrained: DINOv2 model name (e.g., 'dinov2_vitb14'), local path, or '' for no pretrain.
        img_size: Input image size.
        drop_path_rate: Stochastic depth rate for the student.
        init_values: LayerScale init value.

    Returns:
        (backbone, embed_dim) tuple.
    """
    if model_name not in VIT_SPECS:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(VIT_SPECS.keys())}")

    embed_dim, depth, num_heads, mlp_ratio = VIT_SPECS[model_name]

    backbone = DinoVisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=3,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        block_fn=partial(Block, attn_class=MemEffAttention),
        init_values=init_values,
        drop_path_rate=drop_path_rate,
        num_register_tokens=0,
        interpolate_antialias=False,
        interpolate_offset=0.1,
        ffn_layer="mlp",
        block_chunks=0,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
    )

    if pretrained:
        _load_pretrained_weights(backbone, pretrained, model_name, patch_size)

    n_params = sum(p.numel() for p in backbone.parameters()) / 1e6
    logger.info(f"Built {model_name} (patch={patch_size}, embed={embed_dim}, "
                f"depth={depth}, params={n_params:.1f}M)")

    return backbone, embed_dim


def _load_pretrained_weights(backbone, pretrained, model_name, patch_size):
    """Load pretrained weights from torch hub or local file."""
    state_dict = None

    # Try local file first
    if os.path.isfile(pretrained):
        logger.info(f"Loading pretrained weights from local file: {pretrained}")
        ckpt = torch.load(pretrained, map_location="cpu", weights_only=False)
        if "model" in ckpt:
            state_dict = ckpt["model"]
        elif "teacher" in ckpt:
            # Extract backbone from teacher checkpoint
            state_dict = {}
            for k, v in ckpt["teacher"].items():
                if k.startswith("backbone."):
                    state_dict[k[len("backbone."):]] = v
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt

    # Try torch hub
    elif pretrained in DINOV2_HUB_MODELS:
        logger.info(f"Loading pretrained weights from torch hub: {pretrained}")
        repo, model_id = DINOV2_HUB_MODELS[pretrained].rsplit(":", 1)
        hub_model = torch.hub.load(repo, model_id, pretrained=True)
        state_dict = hub_model.state_dict()
        del hub_model

    # Try direct torch hub string
    elif ":" in pretrained:
        logger.info(f"Loading from torch hub: {pretrained}")
        repo, model_id = pretrained.rsplit(":", 1)
        hub_model = torch.hub.load(repo, model_id, pretrained=True)
        state_dict = hub_model.state_dict()
        del hub_model

    if state_dict is None:
        logger.warning(f"Could not load pretrained weights: {pretrained}")
        return

    # Strip common prefixes
    cleaned = {}
    for k, v in state_dict.items():
        for prefix in ["module.", "backbone.", "encoder."]:
            if k.startswith(prefix):
                k = k[len(prefix):]
        cleaned[k] = v

    # Filter out incompatible keys (head weights, different sizes)
    model_dict = backbone.state_dict()
    compatible = {}
    skipped = []
    for k, v in cleaned.items():
        if k in model_dict:
            if v.shape == model_dict[k].shape:
                compatible[k] = v
            else:
                skipped.append(f"{k}: {v.shape} vs {model_dict[k].shape}")
        else:
            skipped.append(f"{k}: not in model")

    missing = [k for k in model_dict if k not in compatible]

    backbone.load_state_dict(compatible, strict=False)
    logger.info(f"Loaded {len(compatible)}/{len(model_dict)} parameters from pretrained weights")
    if skipped:
        logger.info(f"Skipped {len(skipped)} keys: {skipped[:5]}{'...' if len(skipped) > 5 else ''}")
    if missing:
        logger.info(f"Missing {len(missing)} keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
