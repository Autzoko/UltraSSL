#!/bin/bash
#SBATCH --job-name=ultrassl-dino
#SBATCH --output=logs/dino_%j.out
#SBATCH --error=logs/dino_%j.err
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=90:00:00

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
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

export XFORMERS_DISABLED=1

# Change to project directory
PROJECT_DIR=/scratch/ll5582/Projects/UltraSSL
CONFIG=${PROJECT_DIR}/config/ultrassl_vitb14_3d_labeled.yaml

cd ${PROJECT_DIR}
echo "Changed to directory: $(pwd)"
echo ""

# Create logs directory
mkdir -p logs

echo "Starting UltraSSL DINOv2 training"
echo "Config: ${CONFIG}"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo ""

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python train_ultrassl.py --config ${CONFIG}

if [ $? -eq 0 ]; then
    echo ""
    echo "Training completed successfully!"
else
    echo ""
    echo "Training failed with exit code $?"
    exit 1
fi

echo "End Time: $(date)"
