# UltraSSL

Domain-adaptive DINOv2 self-supervised pretraining for breast ultrasound images.

UltraSSL continues self-supervised training from official DINOv2 pretrained weights on unlabeled ultrasound data, producing a ViT encoder with strong dense patch-level features for downstream segmentation and prompting tasks.

## Method

The pipeline implements the full DINOv2 teacher-student self-distillation framework:

- **DINO loss** — CLS token cross-entropy between student and EMA teacher across multi-crop views
- **iBOT loss** — masked patch prediction for dense feature learning (critical for segmentation)
- **KoLeo regularizer** — encourages diverse, uniformly distributed representations
- **EMA teacher** — exponential moving average of the student, exported as the final encoder

Built as a clean adaptation layer on top of the [official DINOv2 codebase](https://github.com/facebookresearch/dinov2), with no modifications to the original repo. Removes the FSDP/xformers dependency so training runs on any single GPU or multi-GPU setup via standard PyTorch DDP.

### Ultrasound-specific design

| Component | Adaptation |
|---|---|
| Augmentations | Grayscale-aware (no hue/saturation jitter), added speckle noise and gamma correction |
| Frequency augmentations | FDA amplitude mixing, spectral band randomization, spectral dropout |
| Crop scales | Conservative local crops ([0.15, 0.4]) to preserve small lesions |
| Data loading | Multi-source with volume-aware stride subsampling for 3D ABUS data |
| WebDataset | Pack images into tar shards for HPC file-count quota compliance |
| Backbone | Swappable ViT-B/14 (default) with DINOv2 pretrained weight initialization |
| Hyperparameters | Fine-tuning regime: low LR (5e-5), high EMA momentum (0.996) |

## Project structure

```
UltraSSL/
├── config/
│   ├── data_root.json              # Dataset path definitions
│   └── ultrassl_vitb14.yaml        # Training config (ViT-B/14)
├── dinov2/                         # Official DINOv2 repo (git submodule, unmodified)
├── ultrassl/
│   ├── data/
│   │   ├── create_shards.py        # Convert raw images → WebDataset tar shards
│   │   ├── wds_dataset.py          # WebDataset-based data loading
│   │   ├── freq_augment.py         # FDA, spectral band randomization, spectral dropout
│   │   ├── augmentations.py        # Multi-crop pipeline with ultrasound transforms
│   │   └── dataset.py              # Multi-source dataset with volume-aware loading
│   ├── models/
│   │   ├── backbone.py             # Swappable ViT backbone + weight loading
│   │   └── ssl_meta_arch.py        # Teacher-student architecture (no FSDP)
│   ├── train/
│   │   └── trainer.py              # Training loop (single-GPU / DDP)
│   └── utils/
│       └── diagnostics.py          # Loss logging + embedding sanity checks
├── train_ultrassl.py               # Entry point
└── requirements_ultrassl.txt       # Dependencies
```

## Quick start

### 1. Clone

```bash
git clone --recurse-submodules https://github.com/Autzoko/UltraSSL.git
cd UltraSSL
```

### 2. Create conda environment

```bash
# Create environment with Python 3.10 (3.9 also works)
conda create -n ultrassl python=3.10 -y
conda activate ultrassl

# Install PyTorch (pick ONE line matching your hardware)
# CUDA 11.8
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia -y
# CUDA 12.1
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y
# CPU only (for data preprocessing / testing without GPU)
conda install pytorch torchvision cpuonly -c pytorch -y

# Install remaining dependencies
pip install -r requirements_ultrassl.txt
```

The `requirements_ultrassl.txt` installs: `omegaconf`, `fvcore`, `iopath`, `webdataset`, `numpy`, `Pillow`. No xformers needed — the pipeline uses PyTorch's native SDPA attention.

Verify the installation:

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "from ultrassl.data.dataset import UltrasoundDataset; print('UltraSSL OK')"
```

### 3. Prepare data

#### Option A: Raw image files (local training)

Edit `config/data_root.json` to point to your ultrasound image directories:

```json
{
  "data": [
    { "name": "BIrads", "path": "/path/to/BIrads_all" },
    { "name": "ABUS",   "path": "/path/to/ABUS_all" }
  ]
}
```

Each path should be a directory containing images (`.png`, `.jpg`, `.bmp`, `.tif`, `.npy`) in any subfolder structure. 3D volumes (many numbered slices per folder) are automatically detected and subsampled.

#### Option B: WebDataset shards (recommended for HPC)

Pack raw images into a small number of `.tar` shard files. This converts 100K+ individual files into ~100 tar files, avoiding inode/file-count quota limits on HPC clusters.

```bash
python -m ultrassl.data.create_shards \
    --data-root config/data_root.json \
    --output-dir shards/ \
    --images-per-shard 1000
```

Then upload the `shards/` directory to your HPC (only ~100 files instead of 100K+).

### 4. Train

#### From raw image files (default)

```bash
# Single GPU
python train_ultrassl.py --config config/ultrassl_vitb14.yaml

# Multi-GPU (e.g., 2 GPUs)
torchrun --nproc_per_node=2 train_ultrassl.py --config config/ultrassl_vitb14.yaml
```

#### From WebDataset shards

```bash
python train_ultrassl.py --config config/ultrassl_vitb14.yaml \
    data.shard_dir=shards/
```

Override any config value from the command line:

```bash
python train_ultrassl.py --config config/ultrassl_vitb14.yaml \
    optim.epochs=20 \
    train.batch_size_per_gpu=16 \
    model.pretrained=/path/to/local/weights.pth
```

### 5. Outputs

Training writes to `outputs/ultrassl_vitb14/` (configurable via `train.output_dir`):

| File | Description |
|---|---|
| `teacher_backbone_latest.pth` | EMA teacher encoder — the main output for downstream use |
| `checkpoint_latest.pth` | Full training state (student + teacher + optimizer) for resuming |
| `training_metrics.jsonl` | Per-iteration losses, LR, momentum (one JSON object per line) |
| `embedding_diagnostics.jsonl` | Periodic embedding stats: PCA variance, cosine similarity, norms |

Resume training automatically by re-running the same command (reads `checkpoint_latest.pth`). Use `--no-resume` to start fresh.

## Using the trained encoder

```python
import torch
from ultrassl.models.backbone import build_backbone

# Load the exported teacher backbone
backbone, embed_dim = build_backbone(
    model_name="vit_base",
    patch_size=14,
    pretrained="outputs/ultrassl_vitb14/teacher_backbone_latest.pth",
)
backbone.eval().cuda()

# Extract dense features from an image
img = torch.randn(1, 3, 224, 224).cuda()  # your preprocessed ultrasound image
with torch.no_grad():
    out = backbone(img, is_training=True)
    cls_token = out["x_norm_clstoken"]       # (1, 768) — global representation
    patch_tokens = out["x_norm_patchtokens"] # (1, 256, 768) — dense patch features
```

Patch tokens can be reshaped to a spatial feature map: `patch_tokens.reshape(1, 16, 16, 768).permute(0, 3, 1, 2)` for segmentation heads.

## Switching backbones

The backbone is fully configurable in the YAML config:

```yaml
# ViT-B/14 (default, ~86M params)
model:
  arch: vit_base
  patch_size: 14
  pretrained: "dinov2_vitb14"

# ViT-L/14 (~307M params)
model:
  arch: vit_large
  patch_size: 14
  pretrained: "dinov2_vitl14"

# ViT-S/14 (~22M params, lighter)
model:
  arch: vit_small
  patch_size: 14
  pretrained: "dinov2_vits14"
```

Or override from CLI: `model.arch=vit_large model.pretrained=dinov2_vitl14`.

## Frequency-domain augmentations

Three augmentation types, all preserving phase (anatomical structure) and only modifying amplitude:

| Augmentation | Description | Key params |
|---|---|---|
| **FDA amplitude mix** | Swaps low-frequency amplitude with a self-referencing perturbed copy | `fda_beta: [0.01, 0.15]` |
| **Spectral band randomization** | Scales amplitude in concentric frequency bands by random factors | `spectral_band_magnitude: 0.3` |
| **Spectral dropout** | Zeros out random annular frequency bands | `spectral_dropout_ratio: 0.15` |

Applied with overall probability `freq_augment_p: 0.3` per view. One augmentation is randomly chosen each time. All parameters are configurable in the `augmentation:` section of the YAML config.

## Key config reference

| Parameter | Default | Notes |
|---|---|---|
| `model.arch` | `vit_base` | `vit_small`, `vit_base`, `vit_large`, `vit_giant2` |
| `model.patch_size` | `14` | DINOv2 uses 14; match pretrained weights |
| `optim.base_lr` | `5e-5` | Fine-tuning regime; DINOv2 from-scratch uses 4e-3 |
| `optim.epochs` | `50` | Domain adaptation, not from-scratch training |
| `teacher.momentum_teacher` | `0.996` | Higher than DINOv2's 0.992 for stable adaptation |
| `crops.local_crops_scale` | `[0.15, 0.4]` | Larger minimum than DINOv2's 0.05 to keep small lesions |
| `crops.local_crops_number` | `6` | Reduced from 8 for single-GPU memory |
| `ibot.loss_weight` | `1.0` | Kept strong for dense patch feature learning |
| `train.batch_size_per_gpu` | `24` | Fits ViT-B/14 + 6 local crops on one GPU |
| `data.shard_dir` | `""` | Set to shard directory to use WebDataset loading |
| `augmentation.freq_augment_p` | `0.3` | Probability of applying frequency augmentation |
