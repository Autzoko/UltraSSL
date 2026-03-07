# UltraSSL

Domain-adaptive DINOv2 self-supervised pretraining for breast ultrasound images.

UltraSSL continues self-supervised training from official DINOv2 pretrained weights on unlabeled breast ultrasound data, producing a ViT encoder with strong dense patch-level features. The frozen encoder feeds three downstream systems:
1. **Patch-level lesion classifier** — per-slice has_lesion detection using MIL on patch tokens
2. **Volume-level MIL classifiers** — whole-volume binary (has_lesion) and multi-class (Class2/3/4 subtyping) using CLS token embeddings with gated attention pooling
3. **Lesion center heatmap localizer** — predicts Gaussian center heatmaps from patch tokens for AutoSAMUS point prompt generation

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
│  │  (BIrads, Class2, Class3, Class4, Abus, Duying)              │   │
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
│  │ Stage 3: Patch-Level Lesion Classifier                       │   │
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
│                          │                                          │
│                          ▼                                          │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ Stage 4: Volume-Level MIL Classifiers                        │   │
│  │                                                              │   │
│  │  Pre-extract DINO CLS tokens (extract_embeddings.py)         │   │
│  │  ──▶ Cache: embedding_cache/<sample_id>.pt                   │   │
│  │         │                                                    │   │
│  │         ▼                                                    │   │
│  │  Volume = stack of K CLS tokens (K, 768)                     │   │
│  │         │                                                    │   │
│  │         ▼                                                    │   │
│  │  Gated Attention MIL Pool ──▶ volume embed (768)             │   │
│  │         │                                                    │   │
│  │         ▼                                                    │   │
│  │  Projector (768→256) + Heads:                                │   │
│  │  ┌────────────────┬──────────────────┬───────────────────┐   │   │
│  │  │ Joint          │ Presence         │ Subtype           │   │   │
│  │  │ binary (B,1)   │ binary (B,1)     │ class (B,3)       │   │   │
│  │  │ + class (B,3)  │                  │ (positive only)   │   │   │
│  │  │ Focal + CE     │ Focal            │ Weighted CE       │   │   │
│  │  └────────────────┴──────────────────┴───────────────────┘   │   │
│  │                                                              │   │
│  │  Output: volume-level lesion detection + subtype prediction  │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                          │                                          │
│                          ▼                                          │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ Stage 5: Lesion Center Heatmap Localizer                    │   │
│  │                                                              │   │
│  │  WebDataset shards (with bbox annotations)                   │   │
│  │         │                                                    │   │
│  │         ▼                                                    │   │
│  │  Frozen DINO backbone ──▶ patch tokens (B, 256, 768)         │   │
│  │         │                                                    │   │
│  │         ▼                                                    │   │
│  │  Reshape to spatial grid (B, 768, 16, 16)                    │   │
│  │         │                                                    │   │
│  │         ▼                                                    │   │
│  │  LocalizationHead (~1.4M params)                             │   │
│  │  Conv decoder: 16×16 → 32×32 → 64×64                        │   │
│  │         │                                                    │   │
│  │         ▼                                                    │   │
│  │  CenterNet Focal Loss vs Gaussian GT heatmap                 │   │
│  │         │                                                    │   │
│  │         ▼                                                    │   │
│  │  NMS peak detection ──▶ center points (x, y, confidence)     │   │
│  │         │                                                    │   │
│  │         ▼                                                    │   │
│  │  Output: JSON center points → AutoSAMUS point prompts        │   │
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
| BIrads  | 57      | 22,300 | 3,218    | 14.4%| 1017x500   | BI-RADS classified |
| Class2  | 147     | 51,450 | 4,634    | 9.0% | 1017x500   | |
| Class3  | 207     | 72,450 | 4,611    | 6.4% | 1017x500   | |
| Class4  | 216     | 75,250 | 6,718    | 8.9% | 1017x500   | |
| Abus    | 200     | 68,444 | 6,656    | 9.7% | 682x865    | 3D ABUS volumes |
| Duying  | 929     | 341,830| 0        | 0.0% | 1017x500   | Negative controls |

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
| Patch size | 14x14 pixels |
| Input size | 224x224 pixels |
| Patch grid | 16x16 = 256 tokens |
| Embedding dim | 768 |
| Layers | 12 transformer blocks |
| Heads | 12 attention heads |
| Parameters | ~86M |
| Pretrained init | `dinov2_vitb14` (official ImageNet weights from torch hub) |

The input image (any resolution) is resized to 224x224 via `RandomResizedCrop` during training. The ViT splits this into a 16x16 grid of 14x14 patches, producing 256 patch tokens plus 1 CLS token, each of dimension 768.

Grayscale ultrasound images are replicated to 3 channels (RGB) for compatibility with ImageNet-pretrained weights.

### SSL meta-architecture: Teacher-Student Self-Distillation

`UltraSSLMetaArch` implements the DINOv2 self-distillation framework without FSDP or xformers:

```
Input image
    │
    ├──▶ Multi-crop augmentation
    │       ├── 2 global crops (224x224, scale [0.4, 1.0])
    │       └── 2 local crops  (98x98,   scale [0.25, 0.6])
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
| Epoch length | 5000 iterations | ~600K slices / (24 x 4 GPUs) ~ 6250, rounded |

### Data sampling strategy

With 632K slices across 6 datasets of varying sizes (BIrads: 22K vs Duying: 342K), UltraSSL uses dataset-balanced shard replication: smaller datasets' shard lists are replicated to match the largest dataset's shard count, ensuring approximately equal sampling across all datasets.

For SSL pretraining, all slices (positive and negative) are used as unlabeled data (`pos_enrichment: 0.0`). Labels are ignored — the model learns representations from all ultrasound image structure.

### Augmentation pipeline

Each image passes through the following pipeline per crop:

```
Original image (variable resolution, grayscale → 3ch RGB)
    │
    ▼
RandomResizedCrop (224x224 global or 98x98 local)
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
Patch tokens (B, 256, 768)     ─── 256 = 16x16 patch grid
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

Bounding boxes from annotations are mapped to the 16x16 patch grid:

1. Scale bbox from original image space to 224x224: `sx = 224/img_width`
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

- `pos_weight=10.0` compensates for 8.9% positive rate (~1/0.089 ~ 11.2)
- Patches with label=-1 (ignore margin) are masked out of the patch loss

### Train/val split

Volumes are split 85/15 for training/validation, **stratified by dataset** to ensure proportional representation. Split is by volume (not by slice) to prevent data leakage from adjacent slices of the same scan.

---

## Downstream: Volume-level MIL classifiers

Stage 4 introduces three volume-level classifiers that operate on stacks of frozen DINO CLS token embeddings using Multiple Instance Learning (MIL) aggregation. Each volume is treated as a "bag" of slice embeddings.

### Embedding extraction

Before training volume classifiers, pre-extract DINO CLS tokens from all slices:

```bash
python extract_embeddings.py \
    --config config/volume_classifier.yaml \
    --output-dir ./embedding_cache \
    --batch-size 128
```

This produces per-volume cache files:
```
embedding_cache/
├── embeddings/
│   ├── <sample_id_1>.pt    # {"cls_tokens": (N,768), "slice_indices": [...], "has_lesion": 0/1, "dataset": "..."}
│   ├── <sample_id_2>.pt
│   └── ...
└── volume_index.json        # {sample_id: {dataset, has_lesion, n_slices, ...}}
```

### MIL architecture

All three classifiers share the same `VolumeClassifier` module:

```
Volume slices: K CLS tokens (K, 768)     ─── K = max_slices (32)
    │
    ├── Pad (if fewer slices) or subsample (if more)
    │
    ▼
Gated Attention MIL Pool                   ─── or Top-K Pool
    │   • Tanh attention path x Sigmoid gate path
    │   • Softmax-weighted aggregation
    │   • Attention mask for padded positions
    │
    ▼
Volume embedding (768)
    │
    ▼
Projector: Linear(768→256) + GELU + Dropout(0.25)
    │
    ▼
Heads (configurable per classifier):
    ├── Binary head: Linear(256→1)   ──▶ has_lesion logit
    └── Multi-class head: Linear(256→3) ──▶ Class2/Class3/Class4 logits
```

### Three classifier variants

| Classifier | Script | Binary head | Multi-class head | Training data |
|------------|--------|-------------|------------------|---------------|
| **Joint** | `joint_volume_classifier.py` | Yes | Yes | All 6 datasets |
| **Presence** | `lesion_presence_classifier.py` | Yes | No | All 6 datasets |
| **Subtype** | `lesion_subtype_classifier.py` | No | Yes | Class2/3/4 only (positive volumes) |

### Loss functions

**Binary (Focal Loss)**:
```
FocalLoss(alpha=0.25, gamma=2.0, pos_weight=3.0)
```
Down-weights easy examples, focuses on hard positives/negatives. All 6 datasets contribute to binary loss since all volumes have `has_lesion` labels.

**Multi-class (Cross-Entropy)**:
```
CrossEntropyLoss(weight=auto_computed_class_weights)
```
Only volumes from Class2/Class3/Class4 contribute (BIrads/Abus/Duying get `class_idx=-1`, excluded via masking). Class weights are auto-computed from inverse class frequencies.

**Joint loss**: `binary_weight * focal_loss + multiclass_weight * ce_loss` (default 1.0 + 0.5).

### Inference modes

**Lesion presence inference** — filter volumes for downstream processing:
```bash
python lesion_presence_classifier.py \
    --config config/volume_classifier.yaml \
    --inference --checkpoint outputs/volume_classifier/best_model.pth \
    --output-json predictions/presence.json \
    data.cache_dir=./embedding_cache
```
Output: `{sample_id: {has_lesion_prob, has_lesion_pred, top_slice_indices, dataset, n_slices}}`

**Lesion subtype inference** — classify positive volumes:
```bash
python lesion_subtype_classifier.py \
    --config config/volume_classifier.yaml \
    --inference --checkpoint outputs/volume_classifier/best_subtype.pth \
    --filter-json predictions/presence.json \
    --output-json predictions/subtypes.json \
    data.cache_dir=./embedding_cache
```
The `--filter-json` flag accepts presence classifier output, running subtype prediction only on volumes predicted as positive.

### Train/val split

Volume-level classifiers use a 85/15 train/val split, **stratified by dataset**. Split is volume-aware (no data leakage). The split index is broadcast across GPUs via gloo for consistency.

### Multi-GPU support

- **Cached embeddings mode**: standard NCCL DDP with `DistributedSampler`
- **On-the-fly mode**: gloo backend, no DDP wrapper (each GPU trains independently — backbone is frozen, only ~200K MIL head parameters train)

---

## Downstream: Lesion center heatmap localizer

Stage 5 trains a lightweight conv decoder on frozen DINO patch tokens to predict Gaussian center heatmaps. Detected center points serve as point prompts for AutoSAMUS segmentation.

### Architecture

```
Input (B, 3, 224, 224)
    │
    ▼
Frozen DINO teacher backbone
    │
    ▼
Patch tokens (B, 256, 768)     ─── 256 = 16x16 patch grid
    │
    ▼
Reshape to spatial (B, 768, 16, 16)
    │
    ▼
LocalizationHead (~1.4M trainable params):
    Conv2d(768→256, 1×1) + BN + ReLU        16×16
    Conv2d(256→256, 3×3, pad=1) + BN + ReLU 16×16
    ConvTranspose2d(256→128, 4, s=2, p=1)   32×32 + BN + ReLU
    ConvTranspose2d(128→64, 4, s=2, p=1)    64×64 + BN + ReLU
    Conv2d(64→1, 1×1)                       64×64 logits
    │
    ▼
Heatmap (B, 1, 64, 64)         ─── sigmoid → [0,1]
```

Final conv bias is initialized to `log(1/99) ≈ -2.19` so initial predictions are near zero (CenterNet convention for sparse targets).

### Gaussian heatmap targets

Ground-truth heatmaps are generated from `bboxes_normalized` annotations:

1. For each bbox: center = `((nx1+nx2)/2 * H, (ny1+ny2)/2 * H)` where H=64
2. Sigma = `max(min(bbox_w, bbox_h) / 6, 1.5)` in heatmap pixels
3. Gaussian rendered within 3-sigma radius
4. Overlapping bboxes use element-wise max (no additive blending)
5. Negative slices (no bboxes) → all-zero heatmap

### Loss: CenterNet focal loss

Modified focal loss for continuous Gaussian targets (alpha=2.0, beta=4.0):

| Pixel type | Loss term |
|---|---|
| **Center** (target=1) | `-(1-p)^α · log(p)` |
| **Near-center** (0<target<1) | `-(1-target)^β · p^α · log(1-p)` |
| **Background** (target=0) | `-p^α · log(1-p)` |

Loss is normalized by N = number of center points in the batch (not positive pixels).

### Data pipeline

WebDataset streaming with balanced sampling:

- **Positive enrichment**: `oversample_positive=3.0` → keep 1/3 of negatives, achieving ~1:3 pos:neg ratio
- **Dataset balance**: smaller datasets' shard lists are replicated to match largest
- **Excludes Duying** (0% positive rate — no lesion annotations)
- Resize to 224×224 + ImageNet normalization

### Evaluation metrics

| Metric | Description |
|---|---|
| `center_hit_rate` | Fraction of positive slices where top-1 peak falls inside any GT bbox |
| `top_k_recall` | Fraction of GT bboxes with at least one peak inside them |
| `mean_center_distance` | Mean L2 from predicted peak to nearest GT center (normalized coords) |
| `false_positive_rate` | Fraction of negative slices with any detection |

Best model saved by `center_hit_rate`.

### Peak detection

NMS via `F.max_pool2d` comparison (CenterNet standard):

1. Apply sigmoid to logits
2. Run max-pool with kernel=3 on heatmap
3. Keep pixels that are local maxima and above threshold (default 0.3)
4. Return top-k peaks sorted by confidence, with normalized (x, y) coords

### Inference output

Inference mode exports JSON for direct AutoSAMUS prompt generation:

```json
[
  {
    "sample_id": "106_1612",
    "slice_idx": 119,
    "dataset": "BIrads",
    "has_lesion": 1,
    "centers": [
      {"x": 0.52, "y": 0.73, "confidence": 0.91, "x_pixel": 529, "y_pixel": 365},
      {"x": 0.31, "y": 0.45, "confidence": 0.67, "x_pixel": 315, "y_pixel": 225}
    ],
    "gt_bboxes": [[0.487, 0.71, 0.551, 0.796]]
  }
]
```

---

## Project structure

```
UltraSSL/
├── config/
│   ├── data_root_2d.json                # 2D dataset path definitions
│   ├── data_root_3d.json                # 3D dataset path definitions
│   ├── data_label_root_3d.json          # 3D labeled dataset paths (local Mac)
│   ├── data_label_root_3d_hpc.json      # 3D labeled dataset paths (HPC /scratch)
│   ├── ultrassl_vitb14.yaml             # SSL config for 2D datasets
│   ├── ultrassl_vitb14_2d.yaml          # SSL config (2D variant)
│   ├── ultrassl_vitb14_3d_labeled.yaml  # SSL config for 632K 3D slices (multi-GPU)
│   ├── lesion_classifier.yaml           # Patch-level classifier config
│   ├── volume_classifier.yaml           # Volume-level MIL classifier config
│   └── lesion_localizer.yaml            # Lesion center heatmap localizer config
├── dinov2/                              # Official DINOv2 repo (git submodule, unmodified)
├── ultrassl/
│   ├── data/
│   │   ├── create_shards.py             # Convert raw images → WebDataset tar shards
│   │   ├── create_labeled_shards.py     # Convert labeled 3D slices → shards with annotations
│   │   ├── wds_dataset.py               # WebDataset loader (unlabeled)
│   │   ├── wds_labeled_dataset.py       # WebDataset loader (labeled, SSL + detection modes)
│   │   ├── volume_dataset.py            # Volume-level dataset (cached + on-the-fly)
│   │   ├── freq_augment.py              # FDA, spectral band randomization, spectral dropout
│   │   ├── augmentations.py             # Multi-crop pipeline with ultrasound transforms
│   │   └── dataset.py                   # Multi-source dataset with volume-aware loading
│   ├── models/
│   │   ├── backbone.py                  # Swappable ViT backbone + weight loading
│   │   ├── ssl_meta_arch.py             # Teacher-student architecture (no FSDP)
│   │   ├── patch_classifier.py          # Patch-level lesion classifier with MIL pooling
│   │   └── volume_mil.py               # Volume-level MIL classifier (gated attention / top-k)
│   ├── train/
│   │   └── trainer.py                   # Training loop (single-GPU / DDP)
│   ├── lesion_localizer.py              # Lesion center heatmap localizer + data pipeline
│   ├── mil_classifier.py               # Volume-level MIL classifier components
│   └── utils/
│       └── diagnostics.py               # Loss logging + embedding/collapse diagnostics
├── scripts/
│   ├── create_shards.sbatch             # SLURM: shard creation (no GPU)
│   ├── train_dino.sbatch                # SLURM: multi-GPU DINOv2 training
│   ├── train_classifier.sbatch          # SLURM: patch-level classifier training
│   ├── extract_embeddings.sbatch        # SLURM: DINO CLS token extraction
│   └── train_volume_classifiers.sbatch  # SLURM: volume-level classifier training
├── train_ultrassl.py                    # SSL pretraining entry point
├── train_lesion_classifier.py           # Patch-level classifier entry point
├── train_mil_classifier.py              # Volume-level MIL classifier entry point
├── train_lesion_localizer.py            # Lesion center heatmap localizer entry point
├── extract_embeddings.py                # Embedding extraction entry point
├── joint_volume_classifier.py           # Joint binary + multi-class classifier
├── lesion_presence_classifier.py        # Binary has_lesion classifier + inference
├── lesion_subtype_classifier.py         # Multi-class subtype classifier + inference
├── visualize.py                         # Attention maps, PCA, similarity visualization
└── requirements_ultrassl.txt            # Dependencies
```

---

## HPC environment setup (from scratch)

These instructions target a SLURM-based HPC cluster with NVIDIA GPUs (e.g., NYU Greene).

### 1. Clone the repository

```bash
cd /scratch/$USER/Projects
git clone --recurse-submodules https://github.com/Autzoko/UltraSSL.git
cd UltraSSL
```

### 2. Create conda environment

```bash
# Create env in /scratch (faster I/O than $HOME)
conda create --prefix /scratch/$USER/envs/ultrassl python=3.10 -y
conda activate /scratch/$USER/envs/ultrassl

# Load CUDA module (check available versions with: module avail cuda)
module load cuda/11.8.0

# Install PyTorch matching CUDA version
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install remaining dependencies
pip install -r requirements_ultrassl.txt
```

### 3. Verify installation

```bash
python -c "
import torch
print(f'PyTorch {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"

python -c "from ultrassl.models.backbone import build_backbone; print('Backbone OK')"
python -c "from ultrassl.models.volume_mil import VolumeClassifier; print('Volume MIL OK')"
python -c "from ultrassl.data.volume_dataset import scan_volume_index; print('Volume dataset OK')"
```

### 4. Prepare data

Ensure WebDataset shards are available at a `/scratch` path:

```bash
# If shards need to be created from raw data:
python -m ultrassl.data.create_labeled_shards \
    --config config/data_label_root_3d_hpc.json \
    --output-dir /scratch/$USER/Data/Ultrasound/Shards \
    --skip-boundary 3 --min-variance 100 --neg-stride 1

# Or via SLURM:
sbatch scripts/create_shards.sbatch
```

### 5. Update config paths

Edit the YAML config files to point to your `/scratch` paths:

```bash
# In config/ultrassl_vitb14_3d_labeled.yaml:
#   data.shard_dir: "/scratch/$USER/Data/Ultrasound/Shards"

# In config/volume_classifier.yaml:
#   backbone.checkpoint: "/scratch/$USER/Projects/UltraSSL/outputs/ultrassl_vitb14_3d_labeled/teacher_backbone_latest.pth"
#   data.shard_dir: "/scratch/$USER/Data/Ultrasound/Shards"
```

---

## Running the full pipeline

### Step 1: Create WebDataset shards

```bash
python -m ultrassl.data.create_labeled_shards \
    --config config/data_label_root_3d.json \
    --output-dir /scratch/$USER/Data/Ultrasound/Shards \
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
Shards/
├── BIrads/shard-000000.tar ... shard-NNNNNN.tar
├── Class2/shard-000000.tar ...
├── Class3/shard-000000.tar ...
├── Class4/shard-000000.tar ...
├── Abus/shard-000000.tar ...
├── Duying/shard-000000.tar ...
└── index.json
```

### Step 2: Train DINOv2 (self-supervised pretraining)

```bash
# Single GPU (debug/testing)
python train_ultrassl.py --config config/ultrassl_vitb14_3d_labeled.yaml \
    data.shard_dir=/scratch/$USER/Data/Ultrasound/Shards

# Multi-GPU (recommended, 4 GPUs)
torchrun --standalone --nproc_per_node=4 train_ultrassl.py \
    --config config/ultrassl_vitb14_3d_labeled.yaml \
    data.shard_dir=/scratch/$USER/Data/Ultrasound/Shards

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

### Step 3: Visualize features (optional)

```bash
python visualize.py \
    --backbone outputs/ultrassl_vitb14_3d_labeled/teacher_backbone_latest.pth \
    --shard Shards/BIrads/shard-000000.tar \
    --n-images 16 \
    --output-dir vis_output/
```

What to look for:
- Attention maps should highlight anatomically meaningful regions (not point-like hotspots)
- Similarity maps should show spatial variation (not uniform red = collapse)
- PCA features should reveal tissue boundaries and structures

### Step 4: Train patch-level lesion classifier

```bash
# Single GPU
python train_lesion_classifier.py --config config/lesion_classifier.yaml

# Multi-GPU
torchrun --standalone --nproc_per_node=4 train_lesion_classifier.py \
    --config config/lesion_classifier.yaml

# SLURM
sbatch scripts/train_classifier.sbatch
```

Output in `outputs/lesion_classifier/`:

| File | Description |
|---|---|
| `best_model.pth` | Best model by validation AUROC |
| `checkpoint_epoch*.pth` | Periodic checkpoints |
| `training_metrics.jsonl` | Per-epoch loss and evaluation metrics |

### Step 5: Extract DINO CLS embeddings (one-time)

Pre-extract frozen backbone CLS tokens for fast volume classifier training:

```bash
# Single GPU (recommended — extraction is I/O bound)
python extract_embeddings.py \
    --config config/volume_classifier.yaml \
    --output-dir /scratch/$USER/Data/Ultrasound/embedding_cache \
    --batch-size 128

# SLURM
sbatch scripts/extract_embeddings.sbatch
```

### Step 6: Train volume-level classifiers

All three classifiers share the same config and cache directory. Choose which to train:

```bash
CACHE=/scratch/$USER/Data/Ultrasound/embedding_cache

# Joint classifier (binary + multi-class)
torchrun --standalone --nproc_per_node=4 joint_volume_classifier.py \
    --config config/volume_classifier.yaml \
    data.cache_dir=$CACHE

# Lesion presence classifier (binary only)
torchrun --standalone --nproc_per_node=4 lesion_presence_classifier.py \
    --config config/volume_classifier.yaml \
    data.cache_dir=$CACHE

# Lesion subtype classifier (multi-class, positive volumes only)
torchrun --standalone --nproc_per_node=4 lesion_subtype_classifier.py \
    --config config/volume_classifier.yaml \
    data.cache_dir=$CACHE

# SLURM (default: joint)
sbatch scripts/train_volume_classifiers.sbatch

# SLURM (specific script)
sbatch --export=SCRIPT=lesion_presence_classifier.py scripts/train_volume_classifiers.sbatch
sbatch --export=SCRIPT=lesion_subtype_classifier.py scripts/train_volume_classifiers.sbatch
```

Output in `outputs/volume_classifier/`:

| File | Description |
|---|---|
| `best_model.pth` | Best model by validation AUROC (binary) or macro F1 (multi-class) |
| `checkpoint_epoch*.pth` | Periodic checkpoints every 5 epochs |
| `training_log.jsonl` | Per-epoch loss and metrics |

### Step 7: Run inference pipeline

The inference pipeline chains presence detection → subtype classification:

```bash
# 1. Predict which volumes contain lesions
python lesion_presence_classifier.py \
    --config config/volume_classifier.yaml \
    --inference \
    --checkpoint outputs/volume_classifier/best_presence.pth \
    --output-json predictions/presence.json \
    data.cache_dir=$CACHE

# 2. Classify positive volumes into subtypes
python lesion_subtype_classifier.py \
    --config config/volume_classifier.yaml \
    --inference \
    --checkpoint outputs/volume_classifier/best_subtype.pth \
    --filter-json predictions/presence.json \
    --output-json predictions/subtypes.json \
    data.cache_dir=$CACHE
```

### Step 8: Train lesion center heatmap localizer

```bash
# Multi-GPU (recommended)
torchrun --standalone --nproc_per_node=4 train_lesion_localizer.py \
    --config config/lesion_localizer.yaml

# Single GPU
python train_lesion_localizer.py --config config/lesion_localizer.yaml

# Override config values
torchrun --standalone --nproc_per_node=4 train_lesion_localizer.py \
    --config config/lesion_localizer.yaml \
    optim.lr=5e-4 optim.epochs=60 data.oversample_positive=5.0

# Specify datasets explicitly
torchrun --standalone --nproc_per_node=4 train_lesion_localizer.py \
    --config config/lesion_localizer.yaml \
    --datasets BIrads Class3 Class4 ABUS
```

Output in `outputs/lesion_localizer/`:

| File | Description |
|---|---|
| `best_model.pth` | Best model by validation center_hit_rate |
| `checkpoint_epoch*.pth` | Periodic checkpoints every 5 epochs |
| `training_log.jsonl` | Per-epoch loss and localization metrics |

### Step 9: Run localizer inference (export center points for AutoSAMUS)

```bash
python train_lesion_localizer.py \
    --config config/lesion_localizer.yaml \
    --inference \
    --checkpoint outputs/lesion_localizer/best_model.pth \
    --datasets BIrads Class3 Class4 ABUS
```

Output: `outputs/lesion_localizer/inference_results.json` — list of per-slice center point candidates with normalized and pixel coordinates, ready for AutoSAMUS point prompt generation.

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
