#!/bin/bash
# =============================================================================
# Transfer UltraSSL source code, scripts, and data to NYU Abu Dhabi Jubail HPC
#
# Usage:
#   bash singularity/transfer_to_hpc.sh <netid>
#
# Example:
#   bash singularity/transfer_to_hpc.sh lt1234
#
# This script transfers:
#   1. Project source code (ultrassl/, dinov2/, config/, etc.)
#   2. Singularity build files and SLURM job scripts
#   3. Dataset (optional, prompted)
#
# After transfer, build the container ON the HPC (Singularity is Linux-only).
# =============================================================================

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <nyu_netid>"
    echo "Example: $0 lt1234"
    exit 1
fi

NETID="$1"
HPC_HOST="${NETID}@jubail.abudhabi.nyu.edu"
HPC_PROJECT="/scratch/${NETID}/ultrassl"
HPC_SOURCE="$HPC_PROJECT/source"
HPC_DATA="/scratch/${NETID}/data"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "============================================="
echo "Transferring UltraSSL to Jubail HPC"
echo "  NetID:      $NETID"
echo "  HPC host:   $HPC_HOST"
echo "  Source:      $HPC_SOURCE"
echo "============================================="

# Create remote directories
echo ""
echo "Creating remote directories..."
ssh "$HPC_HOST" "mkdir -p $HPC_SOURCE $HPC_PROJECT/config $HPC_PROJECT/outputs $HPC_PROJECT/logs $HPC_DATA"

# Transfer project source code (needed for Singularity build on HPC)
echo ""
echo "Transferring project source code..."
rsync -avP --exclude='__pycache__' --exclude='*.pyc' --exclude='.git' --exclude='outputs' \
    "$PROJECT_ROOT/ultrassl" \
    "$PROJECT_ROOT/dinov2" \
    "$PROJECT_ROOT/config" \
    "$PROJECT_ROOT/train_ultrassl.py" \
    "$PROJECT_ROOT/requirements_ultrassl.txt" \
    "$HPC_HOST:$HPC_SOURCE/"

# Transfer Singularity build files and SLURM scripts
echo ""
echo "Transferring Singularity files and SLURM scripts..."
rsync -avP \
    "$SCRIPT_DIR/ultrassl.def" \
    "$SCRIPT_DIR/build.sh" \
    "$SCRIPT_DIR/setup_hpc.sh" \
    "$SCRIPT_DIR/submit_train.slurm" \
    "$SCRIPT_DIR/submit_train_multigpu.slurm" \
    "$HPC_HOST:$HPC_PROJECT/"

rsync -avP \
    "$PROJECT_ROOT/config/data_root_hpc.json" \
    "$HPC_HOST:$HPC_PROJECT/config/"

# Transfer data (optional)
echo ""
read -p "Transfer dataset to HPC? (y/n): " transfer_data
if [ "$transfer_data" = "y" ]; then
    LOCAL_DATA="/Volumes/Lang/Research/Data/3D Ultrasound/processed"
    read -p "Local data path [$LOCAL_DATA]: " custom_data
    LOCAL_DATA="${custom_data:-$LOCAL_DATA}"

    if [ -d "$LOCAL_DATA" ]; then
        echo "Transferring dataset from $LOCAL_DATA..."
        for ds_dir in "$LOCAL_DATA"/*/; do
            ds_name=$(basename "$ds_dir")
            echo "  Syncing $ds_name..."
            rsync -avP "$ds_dir" "$HPC_HOST:$HPC_DATA/$ds_name/"
        done
    else
        echo "WARNING: Data path not found: $LOCAL_DATA"
    fi
fi

echo ""
echo "============================================="
echo "Transfer complete!"
echo ""
echo "Next steps on the HPC:"
echo "  ssh $HPC_HOST"
echo "  cd $HPC_PROJECT"
echo ""
echo "  # 1. Build the Singularity container (one-time)"
echo "  module load singularity"
echo "  cd $HPC_SOURCE"
echo "  singularity build --remote $HPC_PROJECT/ultrassl.sif singularity/ultrassl.def"
echo "  #   (requires: singularity remote login  — get token from cloud.sylabs.io)"
echo ""
echo "  # 2. Verify setup"
echo "  cd $HPC_PROJECT"
echo "  bash setup_hpc.sh"
echo ""
echo "  # 3. Submit training job"
echo "  sbatch submit_train.slurm"
echo "============================================="
