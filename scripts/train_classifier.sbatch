#!/bin/bash
#SBATCH --job-name=lesion-cls
#SBATCH --output=logs/classifier_%j.out
#SBATCH --error=logs/classifier_%j.err
#SBATCH --partition=nvidia
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=12:00:00

# =============================================================================
# Train patch-level lesion classifier on frozen DINO features
#
# Multi-GPU training via torchrun (DDP).
# Backbone is frozen — only the MLP head trains.
# Reads from WebDataset shards with bbox annotations.
#
# Adjust --gres=gpu:N and --ntasks-per-node=N together for different GPU counts.
# =============================================================================

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_NTASKS_PER_NODE"
echo "Start Time: $(date)"
echo ""

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /scratch/ll5582/envs/ultrassl
module load cuda/11.8.0

if [ $? -ne 0 ]; then
    echo "Error: Failed to activate conda environment"
    exit 1
fi

export XFORMERS_DISABLED=1

PROJECT_DIR=/scratch/ll5582/Projects/UltraSSL
CONFIG=${PROJECT_DIR}/config/lesion_classifier.yaml

cd ${PROJECT_DIR}

mkdir -p logs outputs/lesion_classifier

NGPUS=${SLURM_NTASKS_PER_NODE:-4}

echo "=== Lesion Classifier Training ==="
echo "Config: ${CONFIG}"
echo "GPUs: ${NGPUS}"
echo ""
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

torchrun \
    --standalone \
    --nproc_per_node=${NGPUS} \
    train_lesion_classifier.py \
    --config ${CONFIG}

if [ $? -eq 0 ]; then
    echo ""
    echo "Training completed successfully!"
    echo "Outputs: outputs/lesion_classifier/"
else
    echo ""
    echo "Training failed with exit code $?"
    exit 1
fi

echo "End Time: $(date)"
