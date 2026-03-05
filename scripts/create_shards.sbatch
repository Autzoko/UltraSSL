#!/bin/bash
#SBATCH --job-name=create-shards
#SBATCH --output=logs/create_shards_%j.out
#SBATCH --error=logs/create_shards_%j.err
#SBATCH --partition=nvidia
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00

# =============================================================================
# Create WebDataset shards from processed 3D ultrasound data
#
# Input:  /scratch/ll5582/Data/Ultrasound/processed/{BIrads,Class2,...}/images+labels
# Output: /scratch/ll5582/Data/Ultrasound/Shards/{BIrads,Class2,...}/shard-*.tar
#
# Run this BEFORE training. No GPU required.
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

PROJECT_DIR=/scratch/ll5582/Projects/UltraSSL
cd ${PROJECT_DIR}

mkdir -p logs

echo "=== Creating WebDataset Shards ==="
echo ""

python -m ultrassl.data.create_labeled_shards \
    --config config/data_label_root_3d_hpc.json \
    --output-dir /scratch/ll5582/Data/Ultrasound/Shards \
    --images-per-shard 5000 \
    --skip-boundary 3 \
    --min-variance 100 \
    --neg-stride 1

if [ $? -eq 0 ]; then
    echo ""
    echo "Shard creation completed successfully!"
    echo "Output: /scratch/ll5582/Data/Ultrasound/Shards/"
    ls -la /scratch/ll5582/Data/Ultrasound/Shards/
else
    echo ""
    echo "Shard creation failed with exit code $?"
    exit 1
fi

echo "End Time: $(date)"
