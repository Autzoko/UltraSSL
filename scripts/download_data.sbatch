#!/bin/bash
#SBATCH --job-name=gdown
#SBATCH --output=logs/gdown_%j.out
#SBATCH --error=logs/gdown_%j.err
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=24:00:00

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo ""

source $(conda info --base)/etc/profile.d/conda.sh
conda activate /scratch/ll5582/envs/ultrassl

if [ $? -ne 0 ]; then
    echo "Error: Failed to activate conda environment"
    exit 1
fi

DOWNLOAD_DIR=/scratch/ll5582/Data/Ultrasound
mkdir -p ${DOWNLOAD_DIR} logs

cd ${DOWNLOAD_DIR}

echo "Downloading to: ${DOWNLOAD_DIR}"
echo ""

gdown --id 1kWAMMxV-Zg3LeB6bvhwfIjOxKM_8vT6d

if [ $? -eq 0 ]; then
    echo ""
    echo "Download completed successfully!"
    ls -lh ${DOWNLOAD_DIR}
else
    echo ""
    echo "Download failed with exit code $?"
    exit 1
fi

echo "End Time: $(date)"
