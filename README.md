# UltraSSL

Domain-adaptive DINOv2 self-supervised pretraining for breast ultrasound images.

UltraSSL continues self-supervised training from official DINOv2 pretrained weights on unlabeled breast ultrasound data, producing a ViT encoder with strong dense patch-level features for a downstream lesion-presence classifier that reduces false positives.

---

## Pipeline overview

```
                           UltraSSL Pipeline
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ Stage 1: Data Preprocessing                                  │   │
│  │                                                              │   │
│  │  3D Ultrasound Volumes                                       │   │
│  │  (BIrads, Class2, Class3, Class4, Abus, Duying)               │   │
│  │         │                                                    │   │
│  │         ▼                                                    │   │
│  │  External preprocessing script                               │   │
│  │  (volume → coronal 2D slices + bbox labels)                  │   │
│  │         │                                                    │   │
│  │         ▼                                                    │   │
│  │  create_labeled_shards.py                                    │   │
│  │  • Quality filtering (boundary skip, variance threshold)     │   │
│  │  • Bbox sanitization (clamp, dedup, area classification)     │   │
│  │  • Negative stride subsampling                               │   │
│  │  • Pack into WebDataset .tar shards + index.json             │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                          │                                          │
│                          ▼                                          │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ Stage 2: Self-Supervised Pretraining (DINOv2)                │   │
│  │                                                              │   │
│  │  WebDataset shards (labels ignored)                          │   │
│  │         │                                                    │   │
│  │         ▼                                                    │   │
│  │  Dataset-balanced + pos-enriched sampling                    │   │
│  │         │                                                    │   │
│  │         ▼                                                    │   │
│  │  Multi-crop augmentation (2 global + 2 local crops)          │   │
│  │  + frequency-domain augmentations (FDA, spectral)            │   │
│  │         │                                                    │   │
│  │         ▼                                                    │   │
│  │  Teacher-Student Self-Distillation                           │   │
│  │  ┌─────────────┐    EMA     ┌─────────────┐                 │   │
│  │  │   Student    │──────────▶│   Teacher    │                 │   │
│  │  │  ViT-B/14   │           │  ViT-B/14   │                 │   │
│  │  │  + heads    │           │  + heads    │                 │   │
│  │  └──────┬──────┘           └──────┬──────┘                 │   │
│  │         │                         │                          │   │
│  │         ▼                         ▼                          │   │
│  │  DINO loss (CLS) + iBOT loss (patches) + KoLeo              │   │
│  │                                                              │   │
│  │  Output: teacher_backbone_latest.pth                         │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                          │                                          │
│                          ▼                                          │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ Stage 3: Downstream Lesion Classifier                        │   │
│  │                                                              │   │
│  │  WebDataset shards (with labels)                             │   │
│  │         │                                                    │   │
│  │         ▼                                                    │   │
│  │  Frozen DINO backbone ──▶ patch tokens (B, 256, 768)         │   │
│  │         │                                                    │   │
│  │         ▼                                                    │   │
│  │  MLP head (768→256→1) ──▶ patch logits (B, 256, 1)          │   │
│  │         │                                                    │   │
│  │         ▼                                                    │   │
│  │  MIL top-k pooling ──▶ image logit (B, 1)                   │   │
│  │         │                                                    │   │
│  │         ▼                                                    │   │
│  │  Patch BCE + Image BCE ──▶ combined loss                     │   │
│  │                                                              │   │
│  │  Output: best_model.pth (per-slice lesion predictions)       │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Data

### Source datasets

UltraSSL trains on 6 breast ultrasound datasets totaling **1,756 volumes / ~632K coronal 2D slices**:

| Dataset | Volumes | Slices | Positive | Rate | Resolution | Notes |
|---------|---------|--------|----------|------|------------|-------|
| BIrads  | 57      | 22,300 | 3,218    | 14.4%| 1017×500   | BI-RADS classified |
| Class2  | 147     | 51,450 | 4,634    | 9.0% | 1017×500   | |
| Class3  | 207     | 72,450 | 4,611    | 6.4% | 1017×500   | |
| Class4  | 216     | 75,250 | 6,718    | 8.9% | 1017×500   | |
| Abus    | 200     | 68,444 | 6,656    | 9.7% | 682×865    | 3D ABUS volumes |
| Duying  | 929     | 341,830| 0        | 0.0% | 1017×500   | Negative controls |

Overall: **4.0% positive** (25,837 slices with lesions). Duying provides a large negative control set (54% of all slices).

### Data format

Each dataset is preprocessed externally into paired `images/` and `labels/` directories:

```
processed/
├── BIrads/
│   ├── images/
│   │   ├── <volume_id>/
│   │   │   ├── slice_0000.png    # Grayscale coronal slice
│   │   │   ├── slice_0001.png
│   │   │   └── ...
│   │   └── ...
│   ├── labels/
│   │   ├── <volume_id>/
│   │   │   ├── slice_0000.txt    # One line per bbox: has_lesion x1 y1 x2 y2
│   │   │   ├── slice_0001.txt    # Negative: "0 0 0 0 0"
│   │   │   └── ...
│   │   └── ...
│   └── summary.csv
├── Class2/
├── Class3/
├── Class4/
├── Abus/
└── Duying/
```

### Data preprocessing (shard creation)

`create_labeled_shards.py` converts raw directories into WebDataset `.tar` shards. During conversion it applies:

1. **Image-label alignment validation** — bidirectional check that every image has a label and vice versa. Orphans are warned and skipped.

2. **Frame quality filtering**:
   - **Boundary skip** (`--skip-boundary 3`): first/last N slices per volume are removed (typically near-blank ultrasound frames at volume edges)
   - **Variance threshold** (`--min-variance 100`): frames with pixel variance below threshold are dropped (blank or near-uniform frames)

3. **Bbox sanitization**:
   - Coordinates clamped to image bounds
   - Inverted coordinates (x1 > x2) are swapped
   - Tiny bboxes (area < 4 pixels) are dropped as annotation noise
   - If all bboxes for a positive slice are dropped, the label flips to negative

4. **Bbox area classification**: each bbox is categorized as `small` (<2% of image area), `medium` (2-5%), or `large` (>=5%)

5. **Negative stride subsampling** (`--neg-stride N`): within each volume, keep every Nth negative slice. All positive slices are kept unconditionally. Reduces redundancy from adjacent similar negative slices.

6. **Deterministic shuffling** (seed=42) before writing to shards for reproducibility.

Each shard sample contains:
- `<key>.png` — RGB image (grayscale replicated to 3 channels for pretrained weight compatibility)
- `<key>.json` — annotation payload:
```json
{
  "dataset": "BIrads",
  "sample_id": "106_1612",
  "slice_id": "slice_0119",
  "slice_idx": 119,
  "has_lesion": 1,
  "bboxes": [[495.0, 355.0, 560.0, 398.0]],
  "bboxes_normalized": [[0.487, 0.71, 0.551, 0.796]],
  "bbox_area_ratios": [0.0086],
  "bbox_area_bucket": "small",
  "image_width": 1017,
  "image_height": 500
}
```

An `index.json` file is written with per-dataset statistics: total/positive/negative counts, skip reasons, bbox area distribution, shard count, and total size.

---

## Model architecture

### Backbone: Vision Transformer (ViT-B/14)

The backbone is a standard Vision Transformer from the DINOv2 codebase:

| Property | Value |
|----------|-------|
| Architecture | ViT-Base |
| Patch size | 14×14 pixels |
| Input size | 224×224 pixels |
| Patch grid | 16×16 = 256 tokens |
| Embedding dim | 768 |
| Layers | 12 transformer blocks |
| Heads | 12 attention heads |
| Parameters | ~86M |
| Pretrained init | `dinov2_vitb14` (official ImageNet weights from torch hub) |

The input image (any resolution) is resized to 224×224 via `RandomResizedCrop` during training. The ViT splits this into a 16×16 grid of 14×14 patches, producing 256 patch tokens plus 1 CLS token, each of dimension 768.

Grayscale ultrasound images are replicated to 3 channels (RGB) for compatibility with ImageNet-pretrained weights.

### SSL meta-architecture: Teacher-Student Self-Distillation

`UltraSSLMetaArch` implements the DINOv2 self-distillation framework without FSDP or xformers:

```
Input image
    │
    ├──▶ Multi-crop augmentation
    │       ├── 2 global crops (224×224, scale [0.4, 1.0])
    │       └── 2 local crops  (98×98,   scale [0.25, 0.6])
    │
    ├──▶ Student (ViT-B/14 + DINOHead)
    │       • Processes all crops (global + local)
    │       • iBOT: receives random patch masks on global crops
    │       • Gradients flow through student only
    │
    └──▶ Teacher (ViT-B/14 + DINOHead)
            • Processes global crops only (no masks)
            • No gradients — updated via EMA of student
            • Exported as final encoder for downstream use
```

**Projection head** (`DINOHead`): MLP that projects CLS/patch tokens to a 65536-dim prototype space.
- Architecture: `768 → 2048 → 2048 → 256 → 65536` (3 hidden layers + bottleneck + prototypes)
- Shared between DINO CLS loss and iBOT patch loss (unless `ibot.separate_head: true`)

### SSL training objectives

Three loss functions are computed jointly each iteration:

| Loss | Weight | What it does |
|------|--------|--------------|
| **DINO** | 1.0 | Cross-entropy between student and teacher CLS token distributions across views. Student sees local+global crops; teacher sees global only (cross-view consistency). Teacher outputs are centered and sharpened with a temperature schedule. |
| **iBOT** | 1.0 | Masked patch prediction — random rectangular patches of the student's global crop input are masked, and the student must reconstruct the teacher's patch-level representations. Critical for learning dense, spatially-aware features. |
| **KoLeo** | 0.1 | Kozachenko-Leonenko entropy estimator on CLS token embeddings. Encourages uniform distribution of representations in the embedding space, preventing collapse to a small number of clusters. |

**Teacher update**: exponential moving average (EMA) of all student parameters.
- Momentum starts at 0.996 and cosine-anneals to 1.0 over training
- Higher initial momentum than DINOv2's default (0.992) for stable domain adaptation

### Training schedule

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Learning rate | 5e-5 (cosine decay to 1e-6) | Fine-tuning regime, not from-scratch |
| Warmup | 5 epochs | |
| Weight decay | 0.04 → 0.3 (cosine) | |
| Grad clipping | 3.0 | |
| Layerwise LR decay | 0.9 | Earlier layers get lower LR |
| Patch embed LR mult | 0.1 | Pretrained patch embedding changes slowly |
| Freeze last layer | 1 epoch | Stabilizes early training |
| Teacher temperature | 0.04 → 0.07 (10-epoch warmup) | Sharpens teacher output gradually |
| Epochs | 50 | |
| Batch size | 24 per GPU | |
| Epoch length | 5000 iterations | ~600K slices / (24 × 4 GPUs) ≈ 6250, rounded |

### Multi-GPU training

UltraSSL uses PyTorch DistributedDataParallel (DDP) via `torchrun`. Each GPU gets a unique subset of WebDataset shards (via `wds.split_by_node`) and its own batch. Gradients are synchronized across GPUs automatically.

```bash
# 4 GPUs
torchrun --standalone --nproc_per_node=4 train_ultrassl.py \
    --config config/ultrassl_vitb14_3d_labeled.yaml

# SLURM
sbatch scripts/train_dino.sbatch
```

### Data sampling strategy

With 632K slices across 6 datasets of varying sizes (BIrads: 22K vs Duying: 342K), UltraSSL uses dataset-balanced shard replication: smaller datasets' shard lists are replicated to match the largest dataset's shard count, ensuring approximately equal sampling across all datasets.

For SSL pretraining, all slices (positive and negative) are used as unlabeled data (`pos_enrichment: 0.0`). Labels are ignored — the model learns representations from all ultrasound image structure.

### Augmentation pipeline

Each image passes through the following pipeline per crop:

```
Original image (variable resolution, grayscale → 3ch RGB)
    │
    ▼
RandomResizedCrop (224×224 global or 98×98 local)
    │
    ▼
RandomHorizontalFlip (p=0.5) + RandomVerticalFlip (p=0.3)
    │
    ▼
Frequency augmentation (p=0.15) ─── one of:
    ├── FDA amplitude mix (beta=[0.01, 0.15])
    ├── Spectral band randomization (magnitude=0.3)
    └── Spectral dropout (ratio=0.15)
    │
    ▼
ColorJitter (brightness=0.3, contrast=0.3, no hue/saturation)
    │
    ▼
Gaussian noise (speckle simulation, std=[0.01, 0.05], p=0.4)
    │
    ▼
Random gamma correction (gamma=[0.7, 1.5], p=0.3)
    │
    ▼
GaussianBlur (varies by view: p=1.0 global1, p=0.1 global2, p=0.5 local)
    │
    ▼
ToTensor + Normalize (ImageNet mean/std)
```

Key adaptations for ultrasound:
- No hue/saturation jitter or solarize (ultrasound is grayscale)
- Frequency augmentations modify only amplitude (preserve phase = anatomical structure)
- Conservative crop scales to avoid cropping out small lesions entirely
- Gaussian noise simulates ultrasound speckle

### Collapse detection diagnostics

UltraSSL monitors for representation collapse every 500 iterations using a set of fixed probe images:

| Metric | Healthy range | Collapse indicator |
|--------|--------------|-------------------|
| CLS cosine similarity | < 0.7 | > 0.95 (all images produce same embedding) |
| Patch intra-image similarity | < 0.5 | > 0.90 (all patches identical within image) |
| NN unique fraction | > 0.5 | < 0.2 (all queries return same neighbors) |
| PCA top-1 variance ratio | < 0.5 | > 0.8 (one component dominates) |

Automatic warnings are logged when collapse thresholds are exceeded.

---

## Downstream: Patch-level lesion classifier

### Architecture

The classifier trains a lightweight head on frozen DINO features:

```
Input (B, 3, 224, 224)
    │
    ▼
Frozen DINO teacher backbone
    │
    ▼
Patch tokens (B, 256, 768)     ─── 256 = 16×16 patch grid
    │
    ▼
MLP head: 768 → 256 (GELU, Dropout) → 1
    │
    ▼
Patch logits (B, 256, 1)       ─── per-patch lesion score
    │
    ▼
MIL top-k pooling (top 10%)   ─── average of top ~26 patch scores
    │
    ▼
Image logit (B, 1)             ─── slice-level lesion score
```

### Patch label assignment

Bounding boxes from annotations are mapped to the 16×16 patch grid:

1. Scale bbox from original image space to 224×224: `sx = 224/img_width`
2. For each patch (r, c), compute center: `cx = (c+0.5)*14`, `cy = (r+0.5)*14`
3. Patch is **positive** (label=1) if its center falls inside any scaled bbox
4. Patches within `ignore_margin` pixels of a bbox border get label=-1 (excluded from loss)
5. All other patches are **negative** (label=0)

### Loss function

Combined patch-level and image-level supervision:

```
loss = patch_loss_weight * BCE(patch_logits, patch_labels, pos_weight=10.0, ignore=-1)
     + image_loss_weight * BCE(image_logit, has_lesion, pos_weight=10.0)
```

- `pos_weight=10.0` compensates for 8.9% positive rate (~1/0.089 ≈ 11.2)
- Patches with label=-1 (ignore margin) are masked out of the patch loss

### Train/val split

Volumes are split 85/15 for training/validation, **stratified by dataset** to ensure proportional representation:

```
BIrads: 49 train, 8 val
Class3: 176 train, 31 val
Class4: 183 train, 32 val
ABUS:   170 train, 30 val
```

Split is by volume (not by slice) to prevent data leakage from adjacent slices of the same scan.

---

## Project structure

```
UltraSSL/
├── config/
│   ├── data_root.json                # 2D dataset path definitions
│   ├── data_label_root_3d.json       # 3D labeled dataset paths (local Mac)
│   ├── data_label_root_3d_hpc.json  # 3D labeled dataset paths (HPC /scratch)
│   ├── ultrassl_vitb14.yaml          # SSL config for 2D datasets
│   ├── ultrassl_vitb14_3d_labeled.yaml  # SSL config for 632K 3D slices (multi-GPU)
│   └── lesion_classifier.yaml        # Downstream classifier config
├── dinov2/                           # Official DINOv2 repo (git submodule, unmodified)
├── ultrassl/
│   ├── data/
│   │   ├── create_shards.py          # Convert raw images → WebDataset tar shards
│   │   ├── create_labeled_shards.py  # Convert labeled 3D slices → shards with annotations
│   │   ├── wds_dataset.py            # WebDataset loader (unlabeled)
│   │   ├── wds_labeled_dataset.py    # WebDataset loader (labeled, SSL + detection modes)
│   │   ├── freq_augment.py           # FDA, spectral band randomization, spectral dropout
│   │   ├── augmentations.py          # Multi-crop pipeline with ultrasound transforms
│   │   └── dataset.py               # Multi-source dataset with volume-aware loading
│   ├── models/
│   │   ├── backbone.py               # Swappable ViT backbone + weight loading
│   │   ├── ssl_meta_arch.py          # Teacher-student architecture (no FSDP)
│   │   └── patch_classifier.py       # Patch-level lesion classifier with MIL pooling
│   ├── train/
│   │   └── trainer.py                # Training loop (single-GPU / DDP)
│   └── utils/
│       └── diagnostics.py            # Loss logging + embedding/collapse diagnostics
├── scripts/
│   ├── train_dino.sbatch             # SLURM: multi-GPU DINOv2 training
│   └── create_shards.sbatch          # SLURM: shard creation (no GPU)
├── train_ultrassl.py                 # SSL training entry point
├── train_lesion_classifier.py        # Downstream classifier entry point
├── visualize.py                      # Attention maps, PCA, similarity visualization
└── requirements_ultrassl.txt         # Dependencies
```

---

## Quick start

### 1. Clone

```bash
git clone --recurse-submodules https://github.com/Autzoko/UltraSSL.git
cd UltraSSL
```

### 2. Create conda environment

```bash
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

Verify:

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "from ultrassl.data.dataset import UltrasoundDataset; print('UltraSSL OK')"
```

---

## Running the full pipeline

### Step 1: Configure dataset paths

Edit `config/data_label_root_3d.json` (local) or `config/data_label_root_3d_hpc.json` (HPC):

```json
{
  "data": [
    { "name": "BIrads", "path": "/path/to/processed/BIrads" },
    { "name": "Class2", "path": "/path/to/processed/Class2" },
    { "name": "Class3", "path": "/path/to/processed/Class3" },
    { "name": "Class4", "path": "/path/to/processed/Class4" },
    { "name": "Abus",   "path": "/path/to/processed/Abus" },
    { "name": "Duying", "path": "/path/to/processed/Duying" }
  ]
}
```

### Step 2: Create WebDataset shards

```bash
python -m ultrassl.data.create_labeled_shards \
    --config config/data_label_root_3d.json \
    --output-dir wds/ \
    --skip-boundary 3 \
    --min-variance 100 \
    --neg-stride 1
```

| Flag | Default | Description |
|------|---------|-------------|
| `--skip-boundary N` | `3` | Skip first/last N slices per volume (near-blank) |
| `--min-variance V` | `100.0` | Drop frames with pixel variance below V |
| `--neg-stride N` | `1` | Keep every Nth negative slice per volume (1 = all) |
| `--images-per-shard` | `5000` | Max samples per tar shard |

Output:
```
wds/
├── BIrads/shard-000000.tar ... shard-NNNNNN.tar
├── Class2/shard-000000.tar ...
├── Class3/shard-000000.tar ...
├── Class4/shard-000000.tar ...
├── Abus/shard-000000.tar ...
├── Duying/shard-000000.tar ...
└── index.json
```

### Step 3: Train DINOv2 (self-supervised)

```bash
# Single GPU (debug/testing)
python train_ultrassl.py --config config/ultrassl_vitb14_3d_labeled.yaml \
    data.shard_dir=/path/to/shards

# Multi-GPU (recommended, 4 GPUs)
torchrun --standalone --nproc_per_node=4 train_ultrassl.py \
    --config config/ultrassl_vitb14_3d_labeled.yaml \
    data.shard_dir=/path/to/shards

# SLURM (HPC)
sbatch scripts/train_dino.sbatch
```

Override config values from CLI:

```bash
python train_ultrassl.py --config config/ultrassl_vitb14_3d_labeled.yaml \
    optim.epochs=30 \
    data.pos_enrichment=5.0 \
    crops.local_crops_number=4
```

Output in `outputs/ultrassl_vitb14_3d_labeled/`:

| File | Description |
|---|---|
| `teacher_backbone_latest.pth` | EMA teacher encoder for downstream use |
| `checkpoint_latest.pth` | Full training state for resuming |
| `training_metrics.jsonl` | Per-iteration losses, LR, momentum |
| `embedding_diagnostics.jsonl` | Cosine sim, PCA, NN retrieval, patch diversity |

Resume automatically by re-running the same command. Use `--no-resume` to start fresh.

### Step 4: Visualize features (optional)

```bash
python visualize.py \
    --backbone outputs/ultrassl_vitb14_3d_labeled/teacher_backbone_latest.pth \
    --shard wds/BIrads/shard-000000.tar \
    --n-images 16 \
    --output-dir vis_output/
```

What to look for:
- Attention maps should highlight anatomically meaningful regions (not point-like hotspots)
- Similarity maps should show spatial variation (not uniform red = collapse)
- PCA features should reveal tissue boundaries and structures

### Step 5: Train lesion classifier

```bash
python train_lesion_classifier.py --config config/lesion_classifier.yaml
```

Override config values:

```bash
python train_lesion_classifier.py --config config/lesion_classifier.yaml \
    optim.epochs=50 \
    classifier.mil_type=attention \
    loss.pos_weight=8.0
```

Output in `outputs/lesion_classifier/`:

| File | Description |
|---|---|
| `best_model.pth` | Best model by validation AUROC |
| `checkpoint_epoch*.pth` | Periodic checkpoints |
| `training_metrics.jsonl` | Per-epoch loss and evaluation metrics |

---

## Alternative: 2D unlabeled data

For training on raw 2D image directories without labels:

```bash
# Edit config/data_root.json with dataset paths, then:

# From raw files
python train_ultrassl.py --config config/ultrassl_vitb14.yaml

# Or pack into shards first
python -m ultrassl.data.create_shards \
    --data-root config/data_root.json \
    --output-dir shards/ \
    --images-per-shard 1000

python train_ultrassl.py --config config/ultrassl_vitb14.yaml \
    data.shard_dir=shards/
```

---

## Using the trained encoder

```python
import torch
from ultrassl.models.backbone import build_backbone

backbone, embed_dim = build_backbone(
    model_name="vit_base",
    patch_size=14,
    pretrained="outputs/ultrassl_vitb14_3d_labeled/teacher_backbone_latest.pth",
)
backbone.eval().cuda()

img = torch.randn(1, 3, 224, 224).cuda()
with torch.no_grad():
    out = backbone(img, is_training=True)
    cls_token = out["x_norm_clstoken"]       # (1, 768) — global representation
    patch_tokens = out["x_norm_patchtokens"] # (1, 256, 768) — dense patch features
```

Reshape to spatial feature map for segmentation: `patch_tokens.reshape(1, 16, 16, 768).permute(0, 3, 1, 2)`.

## Switching backbones

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

# ViT-S/14 (~22M params)
model:
  arch: vit_small
  patch_size: 14
  pretrained: "dinov2_vits14"
```

Or override from CLI: `model.arch=vit_large model.pretrained=dinov2_vitl14`.

## Key config reference

| Parameter | Default | Notes |
|---|---|---|
| `model.arch` | `vit_base` | `vit_small`, `vit_base`, `vit_large`, `vit_giant2` |
| `model.patch_size` | `14` | DINOv2 uses 14; match pretrained weights |
| `optim.base_lr` | `5e-5` | Fine-tuning regime; DINOv2 from-scratch uses 4e-3 |
| `optim.epochs` | `50` | Domain adaptation, not from-scratch training |
| `teacher.momentum_teacher` | `0.996` | Higher than DINOv2's 0.992 for stable adaptation |
| `crops.local_crops_scale` | `[0.25, 0.6]` | Larger minimum than DINOv2's 0.05 to keep small lesions |
| `crops.local_crops_number` | `2` | 2 local crops for multi-GPU memory efficiency |
| `ibot.loss_weight` | `1.0` | Kept strong for dense patch feature learning |
| `train.batch_size_per_gpu` | `24` | Fits ViT-B/14 + 2 local crops on one GPU |
| `data.shard_dir` | `""` | Set to shard directory to use WebDataset loading |
| `data.balance_datasets` | `true` | Replicate smaller datasets' shards for balanced sampling |
| `data.pos_enrichment` | `0.0` | 0 = use all slices for SSL (no negative subsampling) |
| `augmentation.freq_augment_p` | `0.15` | Probability of applying frequency augmentation |
