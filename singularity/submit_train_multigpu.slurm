#!/bin/bash
# =============================================================================
# SLURM job script for UltraSSL multi-GPU training on Jubail HPC
#
# Usage:
#   sbatch /scratch/$USER/ultrassl/submit_train_multigpu.slurm
# =============================================================================

#SBATCH -J ultrassl-mg                     # Job name
#SBATCH -o /scratch/%u/ultrassl/logs/ultrassl_mg_%j.out
#SBATCH -e /scratch/%u/ultrassl/logs/ultrassl_mg_%j.err
#SBATCH -p nvidia                          # GPU partition
#SBATCH -G 2                               # Number of GPUs
#SBATCH -n 2                               # Number of tasks (1 per GPU)
#SBATCH -c 16                              # CPU cores per task
#SBATCH --mem=128G                         # Memory
#SBATCH -t 2-00:00:00                      # Time limit: 2 days
#SBATCH --mail-type=BEGIN,END,FAIL

# =============================================================================
# CONFIGURATION
# =============================================================================
SIF_FILE="/scratch/${USER}/ultrassl/ultrassl.sif"
SOURCE_DIR="/scratch/${USER}/ultrassl/source"
DATA_SQSH="/scratch/${USER}/ultrassl/ultrassl_data.sqsh"
OUTPUT_DIR="/scratch/${USER}/ultrassl/outputs"
CONFIG="/opt/ultrassl/config/ultrassl_vitb14.yaml"
DATA_ROOT_JSON="/scratch/${USER}/ultrassl/config/data_root_hpc.json"
NUM_GPUS=2

# =============================================================================
# SETUP
# =============================================================================
mkdir -p "$OUTPUT_DIR"
mkdir -p "/scratch/${USER}/ultrassl/logs"

module purge
module load singularity

export SINGULARITY_CACHEDIR=$TMPDIR
export SINGULARITY_TMPDIR=$TMPDIR

echo "============================================="
echo "UltraSSL Multi-GPU Training"
echo "  Job ID:     $SLURM_JOB_ID"
echo "  Node:       $(hostname)"
echo "  GPUs:       $NUM_GPUS"
echo "  Start:      $(date)"
echo "============================================="

# =============================================================================
# RUN TRAINING (torchrun for DDP)
# =============================================================================

singularity exec --nv \
    --overlay "$DATA_SQSH:ro" \
    --bind "$SOURCE_DIR:/opt/ultrassl" \
    --bind "$OUTPUT_DIR:/outputs" \
    --bind "$DATA_ROOT_JSON:/opt/ultrassl/config/data_root.json" \
    "$SIF_FILE" \
    torchrun --nproc_per_node=$NUM_GPUS \
        /opt/ultrassl/train_ultrassl.py \
        --config "$CONFIG" \
        train.output_dir=/outputs/ultrassl_vitb14 \
        train.num_workers=16 \
        data.data_root_json=/opt/ultrassl/config/data_root.json

echo "Training finished at $(date)"
