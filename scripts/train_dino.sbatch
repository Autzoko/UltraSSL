#!/bin/bash
#SBATCH --job-name=ultrassl-dino
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=90:00:00
#SBATCH --output=logs/dino_%j.out
#SBATCH --error=logs/dino_%j.err

set -e

# --- Paths ---
PROJECT_DIR=/scratch/ll5582/Projects/UltraSSL
CONFIG=${PROJECT_DIR}/config/ultrassl_vitb14_3d_labeled.yaml

# --- Environment ---
source ~/.bashrc
conda activate ../../envs/ultrassl
export XFORMERS_DISABLED=1

cd ${PROJECT_DIR}
mkdir -p logs

echo "=== Job ${SLURM_JOB_ID} started at $(date) ==="
echo "Node: $(hostname), GPU: ${CUDA_VISIBLE_DEVICES}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python train_ultrassl.py --config ${CONFIG}

echo "=== Job ${SLURM_JOB_ID} finished at $(date) ==="
