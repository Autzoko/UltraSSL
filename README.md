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
| Backbone | Swappable ViT-B/14 (default) with DINOv2 pretrained weight initialization |
| Hyperparameters | Fine-tuning regime: low LR (5e-5), high EMA momentum (0.996) |

## Project structure

```
UltraSSL/
├── config/
│   ├── data_root.json              # Dataset path definitions
│   └── ultrassl_vitb14.yaml        # Training config (ViT-B/14)
├── dinov2/                         # Official DINOv2 repo (unmodified)
├── ultrassl/
│   ├── data/
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

### 1. Install dependencies

```bash
pip install -r requirements_ultrassl.txt
```

Core requirements: `torch>=2.0`, `torchvision>=0.15`, `omegaconf`, `fvcore`, `iopath`. No xformers needed.

### 2. Configure dataset paths

Edit `config/data_root.json` to point to your ultrasound image directories:

```json
{
  "data": [
    { "name": "BIrads", "path": "/path/to/BIrads_all" },
    { "name": "ABUS",   "path": "/path/to/ABUS_all" }
  ]
}
```

Each path should be a directory containing images (`.png`, `.jpg`, `.bmp`, `.tif`, `.npy`) in any subfolder structure. ABUS volumes (3D data stored as numbered slices in subdirectories) are automatically detected and subsampled.

### 3. Train

```bash
# Single GPU
python train_ultrassl.py --config config/ultrassl_vitb14.yaml

# Multi-GPU (e.g., 2 GPUs)
torchrun --nproc_per_node=2 train_ultrassl.py --config config/ultrassl_vitb14.yaml
```

Override any config value from the command line:

```bash
python train_ultrassl.py --config config/ultrassl_vitb14.yaml \
    optim.epochs=20 \
    train.batch_size_per_gpu=16 \
    model.pretrained=/path/to/local/weights.pth
```

### 4. Outputs

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
| `augmentation.freq_augment_p` | `0.3` | Probability of applying frequency augmentation |

## HPC deployment (NYU Abu Dhabi Jubail)

The project is packaged for the Jubail HPC cluster using Singularity containers.

### File layout

```
singularity/
├── ultrassl.def                 # Singularity definition file
├── build.sh                     # Build the .sif container
├── transfer_to_hpc.sh           # Transfer container + data to HPC
├── setup_hpc.sh                 # One-time HPC directory setup
├── submit_train.slurm           # Single-GPU SLURM job
└── submit_train_multigpu.slurm  # Multi-GPU (DDP) SLURM job
config/
└── data_root_hpc.json           # Dataset paths for HPC (/data/ bind mount)
```

### Step-by-step

**1. Build the container** (on a Linux machine with sudo, or remotely via Sylabs):

```bash
# Local build (requires sudo)
sudo bash singularity/build.sh

# OR remote build (no root needed)
singularity remote login
bash singularity/build.sh --remote
```

**2. Transfer to HPC**:

```bash
bash singularity/transfer_to_hpc.sh <your-netid>
```

This transfers the `.sif` container, job scripts, and optionally the dataset to `/scratch/<netid>/`.

**3. On the HPC**, run the setup script and submit:

```bash
ssh <netid>@jubail.abudhabi.nyu.edu
cd /scratch/$USER/ultrassl
bash setup_hpc.sh        # verify directories and datasets
sbatch submit_train.slurm  # single GPU
# or
sbatch submit_train_multigpu.slurm  # 2 GPUs with DDP
```

**4. Monitor**:

```bash
squeue -u $USER                                      # job status
tail -f /scratch/$USER/ultrassl/logs/ultrassl_<id>.out  # live output
```

### HPC storage layout

| Path | Purpose | Quota |
|------|---------|-------|
| `/scratch/$USER/ultrassl/ultrassl.sif` | Container | 5TB total |
| `/scratch/$USER/data/BIrads_all/` | Dataset | (shared) |
| `/scratch/$USER/ultrassl/outputs/` | Checkpoints, metrics | (shared) |
| `$HOME` | Do NOT run jobs from here | 50GB |

Data on `/scratch` is deleted after 90 days of inactivity. Move completed outputs to `/archive/$USER/` for long-term storage.
