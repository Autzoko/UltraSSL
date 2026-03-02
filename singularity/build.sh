#!/bin/bash
# =============================================================================
# Build UltraSSL Singularity container
#
# NOTE: Singularity is Linux-only. Cannot run on macOS.
#
# This script must be run on a Linux machine where Singularity is installed:
#   - On the HPC login node (use --remote for Sylabs cloud build)
#   - On a Linux workstation with sudo
#
# Workflow from macOS:
#   1. Transfer code to HPC:  bash singularity/transfer_to_hpc.sh <netid>
#   2. SSH to HPC:            ssh <netid>@jubail.abudhabi.nyu.edu
#   3. Build there:           cd /scratch/$USER/ultrassl && bash build.sh --remote
#
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DEF_FILE="$SCRIPT_DIR/ultrassl.def"
SIF_FILE="$SCRIPT_DIR/ultrassl.sif"

# Check platform
if [[ "$(uname)" == "Darwin" ]]; then
    echo "ERROR: Singularity cannot run on macOS."
    echo ""
    echo "To build the container, transfer the project to the HPC first:"
    echo "  bash singularity/transfer_to_hpc.sh <your-netid>"
    echo ""
    echo "Then SSH to the HPC and build there:"
    echo "  ssh <netid>@jubail.abudhabi.nyu.edu"
    echo "  cd /scratch/\$USER/ultrassl/source"
    echo "  bash singularity/build.sh --remote"
    exit 1
fi

# Parse args
REMOTE_FLAG=""
if [[ "${1:-}" == "--remote" ]]; then
    REMOTE_FLAG="--remote"
    echo "Building remotely via Sylabs cloud..."
    echo "Make sure you have run: singularity remote login"
else
    echo "Building locally (requires sudo/root)..."
fi

cd "$PROJECT_ROOT"

echo "============================================="
echo "Building UltraSSL Singularity container"
echo "  Definition: $DEF_FILE"
echo "  Output:     $SIF_FILE"
echo "  Context:    $PROJECT_ROOT"
echo "============================================="

# Set cache directories to avoid filling $HOME (50GB quota on HPC)
export SINGULARITY_CACHEDIR="${SINGULARITY_CACHEDIR:-${TMPDIR:-/tmp}/singularity-cache}"
export SINGULARITY_TMPDIR="${SINGULARITY_TMPDIR:-${TMPDIR:-/tmp}/singularity-tmp}"
mkdir -p "$SINGULARITY_CACHEDIR" "$SINGULARITY_TMPDIR"

# Build
singularity build $REMOTE_FLAG "$SIF_FILE" "$DEF_FILE"

echo ""
echo "Build complete: $SIF_FILE"
echo "Container size: $(du -h "$SIF_FILE" | cut -f1)"
echo ""
echo "Next steps:"
echo "  1. Move container:  mv $SIF_FILE /scratch/\$USER/ultrassl/"
echo "  2. Submit job:      sbatch /scratch/\$USER/ultrassl/submit_train.slurm"
