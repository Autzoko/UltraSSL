#!/bin/bash
# =============================================================================
# SLURM job script for UltraSSL training on NYU Abu Dhabi Jubail HPC
#
# Usage:
#   sbatch /scratch/$USER/ultrassl/submit_train.slurm
#
# Prerequisites:
#   1. ultrassl.sif built at /scratch/$USER/ultrassl/ultrassl.sif
#   2. Source code at /scratch/$USER/ultrassl/source/
#   3. Data packed as /scratch/$USER/ultrassl/ultrassl_data.sqsh  (1 file!)
# =============================================================================

#SBATCH -J ultrassl                        # Job name
#SBATCH -o /scratch/%u/ultrassl/logs/ultrassl_%j.out   # stdout
#SBATCH -e /scratch/%u/ultrassl/logs/ultrassl_%j.err   # stderr
#SBATCH -p nvidia                          # GPU partition
#SBATCH -G 1                               # Number of GPUs
#SBATCH -n 1                               # Number of tasks
#SBATCH -c 16                              # CPU cores per task
#SBATCH --mem=64G                          # Memory
#SBATCH -t 2-00:00:00                      # Time limit: 2 days
#SBATCH --mail-type=BEGIN,END,FAIL         # Email notifications
# #SBATCH --mail-user=YOUR_EMAIL@nyu.edu   # Uncomment and set your email

# =============================================================================
# CONFIGURATION
# =============================================================================

SIF_FILE="/scratch/${USER}/ultrassl/ultrassl.sif"
SOURCE_DIR="/scratch/${USER}/ultrassl/source"
DATA_SQSH="/scratch/${USER}/ultrassl/ultrassl_data.sqsh"
OUTPUT_DIR="/scratch/${USER}/ultrassl/outputs"
CONFIG="/opt/ultrassl/config/ultrassl_vitb14.yaml"
DATA_ROOT_JSON="/scratch/${USER}/ultrassl/config/data_root_hpc.json"

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
echo "UltraSSL Training Job"
echo "  Job ID:     $SLURM_JOB_ID"
echo "  Node:       $(hostname)"
echo "  GPUs:       ${SLURM_GPUS:-1}"
echo "  Container:  $SIF_FILE"
echo "  Data:       $DATA_SQSH"
echo "  Output:     $OUTPUT_DIR"
echo "  Start:      $(date)"
echo "============================================="

# Verify files
for f in "$SIF_FILE" "$DATA_SQSH" "$DATA_ROOT_JSON"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: Not found: $f"
        exit 1
    fi
done
if [ ! -d "$SOURCE_DIR/ultrassl" ]; then
    echo "ERROR: Source code not found at $SOURCE_DIR"
    exit 1
fi

# =============================================================================
# RUN TRAINING
#
# Mounts:
#   --overlay ultrassl_data.sqsh  → data appears at /data/ inside container
#   --bind source:/opt/ultrassl   → project code
#   --bind outputs:/outputs       → checkpoint output
# =============================================================================

singularity exec --nv \
    --overlay "$DATA_SQSH:ro" \
    --bind "$SOURCE_DIR:/opt/ultrassl" \
    --bind "$OUTPUT_DIR:/outputs" \
    --bind "$DATA_ROOT_JSON:/opt/ultrassl/config/data_root.json" \
    "$SIF_FILE" \
    python /opt/ultrassl/train_ultrassl.py \
        --config "$CONFIG" \
        train.output_dir=/outputs/ultrassl_vitb14 \
        train.num_workers=16 \
        data.data_root_json=/opt/ultrassl/config/data_root.json

echo ""
echo "Training finished at $(date)"
echo "Outputs saved to: $OUTPUT_DIR"
