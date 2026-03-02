#!/bin/bash
# =============================================================================
# Set up UltraSSL on NYU Abu Dhabi Jubail HPC
#
# Run this script on the HPC login node after transferring files:
#   bash setup_hpc.sh
#
# Prerequisites:
#   1. ultrassl.sif transferred to /scratch/$USER/ultrassl/
#   2. Dataset transferred to /scratch/$USER/data/
# =============================================================================

set -euo pipefail

SCRATCH="/scratch/${USER}"
PROJECT_DIR="$SCRATCH/ultrassl"

echo "============================================="
echo "Setting up UltraSSL on Jubail HPC"
echo "  User:       $USER"
echo "  Project:    $PROJECT_DIR"
echo "============================================="

# Create directory structure
mkdir -p "$PROJECT_DIR/config"
mkdir -p "$PROJECT_DIR/outputs"
mkdir -p "$PROJECT_DIR/logs"
mkdir -p "$SCRATCH/data"

echo ""
echo "Directory structure created:"
echo "  $PROJECT_DIR/"
echo "  ├── ultrassl.sif           # Singularity container"
echo "  ├── config/"
echo "  │   └── data_root_hpc.json # HPC data paths"
echo "  ├── outputs/               # Training outputs"
echo "  └── logs/                  # SLURM logs"
echo "  $SCRATCH/data/             # Dataset directory"
echo "      ├── BIrads_all/"
echo "      ├── Duying_all/"
echo "      └── ..."

# Check if container exists
if [ -f "$PROJECT_DIR/ultrassl.sif" ]; then
    echo ""
    echo "Container found: $PROJECT_DIR/ultrassl.sif"
    echo "  Size: $(du -h "$PROJECT_DIR/ultrassl.sif" | cut -f1)"
else
    echo ""
    echo "WARNING: Container not found at $PROJECT_DIR/ultrassl.sif"
    echo "  Transfer it with: scp ultrassl.sif ${USER}@jubail.abudhabi.nyu.edu:$PROJECT_DIR/"
fi

# Create HPC data config if not present
if [ ! -f "$PROJECT_DIR/config/data_root_hpc.json" ]; then
    cat > "$PROJECT_DIR/config/data_root_hpc.json" << 'DATAJSON'
{
    "data":
    [
        {
            "name": "BIrads",
            "path": "/data/BIrads_all"
        },
        {
            "name": "Duying",
            "path": "/data/Duying_all"
        },
        {
            "name": "Class3",
            "path": "/data/Class3_all"
        },
        {
            "name": "Class4",
            "path": "/data/Class4_all"
        },
        {
            "name": "ABUS",
            "path": "/data/ABUS_all"
        }
    ]
}
DATAJSON
    echo "Created: $PROJECT_DIR/config/data_root_hpc.json"
fi

# Check dataset
echo ""
echo "Checking datasets in $SCRATCH/data/:"
for ds in BIrads_all Duying_all Class3_all Class4_all ABUS_all; do
    if [ -d "$SCRATCH/data/$ds" ]; then
        count=$(find "$SCRATCH/data/$ds" -name "*.png" -o -name "*.jpg" -o -name "*.npy" 2>/dev/null | wc -l)
        echo "  $ds: $count images"
    else
        echo "  $ds: NOT FOUND"
    fi
done

# Check quota
echo ""
echo "Storage usage:"
myquota 2>/dev/null || echo "  (myquota not available on login node)"

echo ""
echo "============================================="
echo "Setup complete. To start training:"
echo ""
echo "  # Single GPU:"
echo "  sbatch $PROJECT_DIR/submit_train.slurm"
echo ""
echo "  # Multi-GPU (2 GPUs):"
echo "  sbatch $PROJECT_DIR/submit_train_multigpu.slurm"
echo ""
echo "  # Monitor job:"
echo "  squeue -u $USER"
echo "  tail -f $PROJECT_DIR/logs/ultrassl_<jobid>.out"
echo "============================================="
