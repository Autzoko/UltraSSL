#!/bin/bash
#SBATCH --job-name=ultrassl-dino
#SBATCH --output=logs/dino_%j.out
#SBATCH --error=logs/dino_%j.err
#SBATCH --partition=nvidia
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --mem=512G
#SBATCH --time=72:00:00

# =============================================================================
# UltraSSL: DINOv2 domain-adaptive pretraining on breast ultrasound
#
# Multi-GPU training using torchrun (DistributedDataParallel).
# Reads from WebDataset shards created by create_labeled_shards.py.
#
# Prerequisites:
#   1. Create shards: python -m ultrassl.data.create_labeled_shards \
#          --config config/data_label_root_3d.json \
#          --output-dir /scratch/ll5582/Data/Ultrasound/Shards \
#          --skip-boundary 3 --min-variance 100 --neg-stride 1
#   2. DINOv2 ViT-B/14 checkpoint (auto-downloaded from torch hub on first run)
#
# Adjust --gres=gpu:N and --ntasks-per-node=N together for different GPU counts.
#
# Test command (single GPU, sanity check only):
#   python train_ultrassl.py --sanity-check --config config/ultrassl_vitb14_3d_labeled.yaml
#
# Test command (multi-GPU, no sbatch):
#   torchrun --standalone --nproc_per_node=8 train_ultrassl.py \
#       --config config/ultrassl_vitb14_3d_labeled.yaml
# =============================================================================

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_NTASKS_PER_NODE"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo ""

# Activate conda environment
echo "Activating conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /scratch/ll5582/envs/ultrassl
module load cuda/11.8.0

if [ $? -ne 0 ]; then
    echo "Error: Failed to activate conda environment"
    exit 1
fi

echo "Conda environment activated successfully"
echo "Python path: $(which python)"
echo ""

# export XFORMERS_DISABLED=1  # uncomment to force PyTorch SDPA fallback

# Change to project directory
PROJECT_DIR=/scratch/ll5582/Projects/UltraSSL
CONFIG=${PROJECT_DIR}/config/ultrassl_vitb14_3d_labeled.yaml
SHARD_DIR=/scratch/ll5582/Data/Ultrasound/Shards

cd ${PROJECT_DIR}
echo "Changed to directory: $(pwd)"
echo ""

# Create directories
mkdir -p logs outputs

# Number of GPUs
NGPUS=${SLURM_NTASKS_PER_NODE:-4}

echo "=== UltraSSL DINOv2 Training ==="
echo "Config: ${CONFIG}"
echo "Shard dir: ${SHARD_DIR}"
echo "GPUs: ${NGPUS}"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo ""

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Launch multi-GPU training via torchrun
torchrun \
    --standalone \
    --nproc_per_node=${NGPUS} \
    train_ultrassl.py \
    --config ${CONFIG} \
    data.shard_dir=${SHARD_DIR}

if [ $? -eq 0 ]; then
    echo ""
    echo "Training completed successfully!"
    echo "Checkpoints saved to: outputs/ultrassl_vitb14_3d_labeled/"
else
    echo ""
    echo "Training failed with exit code $?"
    exit 1
fi

echo "End Time: $(date)"
