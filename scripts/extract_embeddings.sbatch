#!/bin/bash
#SBATCH --job-name=extract-emb
#SBATCH --output=logs/extract_emb_%j.out
#SBATCH --error=logs/extract_emb_%j.err
#SBATCH --partition=nvidia
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00

# =============================================================================
# Pre-extract DINO CLS token embeddings from all volumes
#
# One-time step before training volume classifiers.
# Uses single GPU — extraction is I/O bound, not compute bound.
# =============================================================================

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
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

cd ${PROJECT_DIR}
mkdir -p logs ${CACHE_DIR}

echo "=== Extracting DINO CLS Embeddings ==="
echo "Config: ${CONFIG}"
echo "Output: ${CACHE_DIR}"
echo ""
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

python extract_embeddings.py \
    --config ${CONFIG} \
    --output-dir ${CACHE_DIR} \
    --batch-size 128

if [ $? -eq 0 ]; then
    echo ""
    echo "Extraction completed successfully!"
    echo "Cache saved to: ${CACHE_DIR}"
else
    echo ""
    echo "Extraction failed with exit code $?"
    exit 1
fi

echo "End Time: $(date)"
