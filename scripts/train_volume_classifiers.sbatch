#!/bin/bash
#SBATCH --job-name=vol-cls
#SBATCH --output=logs/vol_cls_%j.out
#SBATCH --error=logs/vol_cls_%j.err
#SBATCH --partition=nvidia
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00

# =============================================================================
# Train volume-level classifiers using cached DINO embeddings
#
# Set SCRIPT to choose which classifier to train:
#   - joint_volume_classifier.py       (binary + multi-class, ~6h)
#   - lesion_presence_classifier.py    (binary only, ~4h)
#   - lesion_subtype_classifier.py     (multi-class, positive-only, ~2h)
#
# Prerequisites:
#   Run extract_embeddings.sbatch first to create the embedding cache.
#
# Usage:
#   sbatch scripts/train_volume_classifiers.sbatch
#   sbatch --export=SCRIPT=lesion_presence_classifier.py scripts/train_volume_classifiers.sbatch
# =============================================================================

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_NTASKS_PER_NODE"
echo "Start Time: $(date)"
echo ""

source $(conda info --base)/etc/profile.d/conda.sh
conda activate /scratch/ll5582/envs/ultrassl
module load cuda/11.8.0

if [ $? -ne 0 ]; then
    echo "Error: Failed to activate conda environment"
    exit 1
fi

PROJECT_DIR=/scratch/ll5582/Projects/UltraSSL
CONFIG=${PROJECT_DIR}/config/volume_classifier.yaml
CACHE_DIR=/scratch/ll5582/Data/Ultrasound/embedding_cache

# Default to joint classifier if not specified
SCRIPT=${SCRIPT:-joint_volume_classifier.py}

cd ${PROJECT_DIR}
mkdir -p logs outputs

NGPUS=${SLURM_NTASKS_PER_NODE:-4}

echo "=== Volume Classifier Training ==="
echo "Script: ${SCRIPT}"
echo "Config: ${CONFIG}"
echo "Cache: ${CACHE_DIR}"
echo "GPUs: ${NGPUS}"
echo ""
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

torchrun \
    --standalone \
    --nproc_per_node=${NGPUS} \
    ${SCRIPT} \
    --config ${CONFIG} \
    data.cache_dir=${CACHE_DIR}

if [ $? -eq 0 ]; then
    echo ""
    echo "Training completed successfully!"
else
    echo ""
    echo "Training failed with exit code $?"
    exit 1
fi

echo "End Time: $(date)"
