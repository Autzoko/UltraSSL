#!/bin/bash
#SBATCH --job-name=lesion-cls
#SBATCH --output=logs/classifier_%j.out
#SBATCH --error=logs/classifier_%j.err
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00

# =============================================================================
# Train patch-level lesion classifier on frozen DINO features
#
# Single GPU — backbone is frozen, only the MLP head trains.
# Reads from WebDataset shards with bbox annotations.
# =============================================================================

echo "Job ID: $SLURM_JOB_ID"
echo "Start Time: $(date)"
echo ""

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /scratch/ll5582/envs/ultrassl

if [ $? -ne 0 ]; then
    echo "Error: Failed to activate conda environment"
    exit 1
fi

module load cuda/11.8.0

PROJECT_DIR=/scratch/ll5582/Projects/UltraSSL
cd ${PROJECT_DIR}

mkdir -p logs outputs/lesion_classifier

echo "=== Lesion Classifier Training ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

python train_lesion_classifier.py \
    --config config/lesion_classifier.yaml

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
