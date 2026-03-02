# HPC Deployment Guide

Run UltraSSL on the NYU Abu Dhabi Jubail HPC cluster.

This guide covers the full workflow from your Mac: pack data, push code, build the Singularity container on the HPC, and submit training jobs.

## Prerequisites

- NYU AD HPC account (NetID + SSH access to `jubail.abudhabi.nyu.edu`)
- SSH key set up for the HPC (`ssh-copy-id <netid>@jubail.abudhabi.nyu.edu`)
- GitHub account (to transfer code to HPC)
- Preprocessed ultrasound dataset on your Mac (e.g., under `/Volumes/Lang/Research/Data/3D Ultrasound/processed/`)
- Sylabs account for remote container builds (free, see Step 4)

## Architecture overview

The container only holds the **runtime environment** (PyTorch + CUDA + Python packages). Code and data are mounted at runtime, not baked in.

```
┌──────────────────────────────────────────────────────────────────────┐
│  Singularity Container (ultrassl.sif)                                │
│  Base image: nvcr.io/nvidia/pytorch:23.10-py3                       │
│  + omegaconf, fvcore, iopath, Pillow                                │
│                                                                      │
│  Mount points:                                                       │
│   /opt/ultrassl ← source code          (bind mount from /scratch)   │
│   /data         ← dataset images       (SquashFS overlay, 1 file)   │
│   /outputs      ← checkpoints + logs   (bind mount from /scratch)   │
└──────────────────────────────────────────────────────────────────────┘
```

Why SquashFS for data:
- HPC `$SCRATCH` has a **500K file limit**. A single dataset can have 100K+ image files.
- SquashFS packs all images into **one compressed file**. Singularity mounts it as a read-only overlay — the training code sees a normal `/data/` directory.
- No code changes needed. The dataset loader reads files from `/data/BIrads_all/...` as usual.

What lives on `/scratch/$USER/ultrassl/`:

| File | Approx. size | Description |
|------|-------------|-------------|
| `ultrassl.sif` | ~8 GB | Singularity container (PyTorch + CUDA + deps) |
| `ultrassl_data.sqsh` | ~5-15 GB | All dataset images in one file |
| `source/` | ~50 MB | Git clone of the project code |
| `outputs/` | varies | Training checkpoints, metrics |
| `logs/` | small | SLURM stdout/stderr |
| `config/data_root_hpc.json` | tiny | Dataset path config for HPC |

Total files on `$SCRATCH`: **~50** (instead of 100K+).

---

## Step 1: Install tools on your Mac

```bash
# SquashFS tools (to pack dataset into a single file)
brew install squashfs
```

Verify:

```bash
mksquashfs -version
```

---

## Step 2: Pack dataset into SquashFS

This packs all your preprocessed image directories into a single `.sqsh` file.

```bash
cd /path/to/UltraSSL

bash singularity/pack_data.sh "/Volumes/Lang/Research/Data/3D Ultrasound/processed"
```

This reads all subdirectories (e.g., `BIrads_all/`, `Duying_all/`, ...) and creates `ultrassl_data.sqsh` in the current directory.

If you only have BIrads preprocessed so far, that's fine — only the available datasets will be packed. You can re-run this script later when more datasets are ready.

To pack a specific subset or use a custom output path:

```bash
bash singularity/pack_data.sh "/path/to/data" "my_data.sqsh"
```

Expected output:

```
=============================================
Packing ultrasound data into SquashFS
  Source:  /Volumes/Lang/Research/Data/3D Ultrasound/processed
  Output:  ultrassl_data.sqsh
=============================================
  Adding: BIrads_all
Counting files...
  Total files: 109805
Building SquashFS image (this may take a while)...
...
=============================================
Done!
  Output:     ultrassl_data.sqsh
  Size:       5.2G
  Files:      109805 packed into 1 file
=============================================
```

---

## Step 3: Push code to GitHub

```bash
cd /path/to/UltraSSL

# Initialize git repo (first time only)
git init
git add .
git commit -m "Initial commit: UltraSSL pipeline"

# Create a GitHub repo (private recommended), then connect and push
git remote add origin git@github.com:<your-username>/UltraSSL.git
git branch -M main
git push -u origin main
```

> **Note on dinov2/**: If the `dinov2/` directory is already its own git repo (cloned separately), add it as a submodule instead:
> ```bash
> git submodule add https://github.com/facebookresearch/dinov2.git dinov2
> ```
> Otherwise, if it's just a plain directory, `git add .` will include it normally.

---

## Step 4: Set up Sylabs account (one-time)

The HPC login node does not have root access, so you cannot build Singularity containers locally. Instead, use the **Sylabs remote builder** (free).

1. Go to [https://cloud.sylabs.io](https://cloud.sylabs.io) and create an account
2. Generate an access token at [https://cloud.sylabs.io/tokens](https://cloud.sylabs.io/tokens)
3. Save the token — you will paste it on the HPC in Step 6

---

## Step 5: Upload data to HPC

Upload the single `.sqsh` file to `/scratch`. This is the only large transfer.

```bash
rsync -avP ultrassl_data.sqsh <netid>@jubail.abudhabi.nyu.edu:/scratch/<netid>/ultrassl/
```

Replace `<netid>` with your NYU NetID.

For a ~5-10 GB file, this takes roughly 10-30 minutes depending on your connection. You can also use `scp`:

```bash
scp ultrassl_data.sqsh <netid>@jubail.abudhabi.nyu.edu:/scratch/<netid>/ultrassl/
```

---

## Step 6: Clone code and build container on HPC

SSH into the HPC:

```bash
ssh <netid>@jubail.abudhabi.nyu.edu
```

### 6a. Create project directories

```bash
mkdir -p /scratch/$USER/ultrassl/{outputs,logs,config}
```

### 6b. Clone the project

```bash
cd /scratch/$USER/ultrassl
git clone https://github.com/<your-username>/UltraSSL.git source
```

If you used a git submodule for dinov2:

```bash
cd source
git submodule update --init --recursive
cd ..
```

### 6c. Copy the HPC config and SLURM scripts

```bash
# Data path config for HPC (maps /data/BIrads_all etc.)
cp source/config/data_root_hpc.json config/data_root_hpc.json

# SLURM job scripts
cp source/singularity/submit_train.slurm .
cp source/singularity/submit_train_multigpu.slurm .
```

### 6d. Build the Singularity container

```bash
module load singularity

# Redirect cache to avoid filling $HOME (50GB quota)
export SINGULARITY_CACHEDIR=$TMPDIR
export SINGULARITY_TMPDIR=$TMPDIR

# Authenticate with Sylabs (first time only — paste your token from Step 4)
singularity remote login

# Build the container remotely (~10 minutes)
cd /scratch/$USER/ultrassl/source
singularity build --remote /scratch/$USER/ultrassl/ultrassl.sif singularity/ultrassl.def
```

If the remote build fails (network issues, quota), you can also build with `--fakeroot` if available:

```bash
singularity build --fakeroot /scratch/$USER/ultrassl/ultrassl.sif singularity/ultrassl.def
```

### 6e. Verify everything is in place

```bash
ls -lh /scratch/$USER/ultrassl/
```

You should see:

```
ultrassl.sif            # ~8 GB  — container
ultrassl_data.sqsh      # ~5 GB  — dataset
source/                 #        — git clone
config/
  data_root_hpc.json    #        — dataset paths
outputs/                #        — (empty, will fill during training)
logs/                   #        — (empty)
submit_train.slurm
submit_train_multigpu.slurm
```

---

## Step 7: Submit a training job

### Single GPU (A100)

```bash
sbatch /scratch/$USER/ultrassl/submit_train.slurm
```

This requests:
- 1 A100 GPU (40 or 80 GB VRAM)
- 16 CPU cores, 64 GB RAM
- 2-day time limit
- `nvidia` partition

### Multi-GPU (2x A100, DDP)

```bash
sbatch /scratch/$USER/ultrassl/submit_train_multigpu.slurm
```

This uses `torchrun` for distributed data parallel with 2 GPUs.

### Custom overrides

Edit the SLURM script, or override config values by modifying the `singularity exec` line. For example, to change batch size and epochs, change the last lines of `submit_train.slurm`:

```bash
    python /opt/ultrassl/train_ultrassl.py \
        --config "$CONFIG" \
        train.output_dir=/outputs/ultrassl_vitb14 \
        train.num_workers=16 \
        train.batch_size_per_gpu=32 \
        optim.epochs=20 \
        data.data_root_json=/opt/ultrassl/config/data_root.json
```

---

## Step 8: Monitor training

```bash
# Check job status
squeue -u $USER

# Watch live output
tail -f /scratch/$USER/ultrassl/logs/ultrassl_<jobid>.out

# Check GPU utilization (while job is running)
srun --jobid=<jobid> nvidia-smi
```

### Training outputs

All outputs are written to `/scratch/$USER/ultrassl/outputs/ultrassl_vitb14/`:

| File | Description |
|------|-------------|
| `teacher_backbone_latest.pth` | EMA teacher encoder — main output for downstream use |
| `checkpoint_latest.pth` | Full state for resuming training |
| `training_metrics.jsonl` | Per-iteration losses (JSON lines) |
| `embedding_diagnostics.jsonl` | Periodic embedding stats |

### Resume training

If a job is interrupted or hits the time limit, simply re-submit the same script. It automatically resumes from `checkpoint_latest.pth`.

```bash
sbatch /scratch/$USER/ultrassl/submit_train.slurm
```

---

## Updating code

When you change code locally, push to GitHub and pull on the HPC. **No container rebuild needed** — code is bind-mounted, not baked in.

```bash
# On your Mac
cd /path/to/UltraSSL
git add . && git commit -m "Update augmentations" && git push

# On the HPC
cd /scratch/$USER/ultrassl/source
git pull
```

Then re-submit the job.

You only need to rebuild `ultrassl.sif` if you change Python dependencies (e.g., add a new `pip install` package). In that case, edit `singularity/ultrassl.def` and re-run the build command from Step 6d.

---

## Updating data

When more datasets are preprocessed, re-run the pack script on your Mac and re-upload:

```bash
# On your Mac — re-pack with all datasets
bash singularity/pack_data.sh "/Volumes/Lang/Research/Data/3D Ultrasound/processed"

# Upload (overwrites the old .sqsh)
rsync -avP ultrassl_data.sqsh <netid>@jubail.abudhabi.nyu.edu:/scratch/<netid>/ultrassl/
```

---

## Saving results

`$SCRATCH` auto-deletes files untouched for 90 days. When training is done, copy important outputs to `$ARCHIVE`:

```bash
mkdir -p /archive/$USER/ultrassl
cp /scratch/$USER/ultrassl/outputs/ultrassl_vitb14/teacher_backbone_latest.pth /archive/$USER/ultrassl/
cp /scratch/$USER/ultrassl/outputs/ultrassl_vitb14/training_metrics.jsonl /archive/$USER/ultrassl/
```

Or download to your Mac:

```bash
scp <netid>@jubail.abudhabi.nyu.edu:/scratch/<netid>/ultrassl/outputs/ultrassl_vitb14/teacher_backbone_latest.pth .
```

---

## Troubleshooting

### "singularity: command not found"

```bash
module load singularity
```

### Container build fails with cache errors

```bash
export SINGULARITY_CACHEDIR=$TMPDIR
export SINGULARITY_TMPDIR=$TMPDIR
```

### "ERROR: Not found: .../ultrassl_data.sqsh"

Make sure the `.sqsh` file was uploaded to `/scratch/$USER/ultrassl/` (not a subdirectory).

### Job stuck in queue

Check partition availability:

```bash
sinfo -p nvidia
```

Try requesting fewer resources (e.g., 1 GPU instead of 2, less memory).

### Out of GPU memory during training

Reduce batch size via config override:

```bash
# Edit submit_train.slurm, add to the python command:
train.batch_size_per_gpu=16
```

### Training produces NaN losses

This usually means the learning rate is too high or data is corrupted. Try:

```bash
optim.base_lr=1.0e-5
```

### "No images found" for a dataset

Check that `config/data_root_hpc.json` paths match the directory names inside your `.sqsh`. You can inspect the contents:

```bash
unsquashfs -l /scratch/$USER/ultrassl/ultrassl_data.sqsh | head -20
```

### Email notifications

Uncomment and set your email in the SLURM script:

```bash
#SBATCH --mail-user=YOUR_NETID@nyu.edu
```

---

## Quick reference

```bash
# === From your Mac ===
brew install squashfs                        # one-time
bash singularity/pack_data.sh "/path/to/data"  # pack dataset
git push                                     # push code
rsync -avP ultrassl_data.sqsh <netid>@jubail.abudhabi.nyu.edu:/scratch/<netid>/ultrassl/

# === On the HPC ===
ssh <netid>@jubail.abudhabi.nyu.edu
cd /scratch/$USER/ultrassl/source
git pull                                     # get latest code
sbatch /scratch/$USER/ultrassl/submit_train.slurm   # train
squeue -u $USER                              # check status
tail -f /scratch/$USER/ultrassl/logs/ultrassl_*.out  # watch output
```
