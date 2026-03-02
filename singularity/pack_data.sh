#!/bin/bash
# =============================================================================
# Pack ultrasound datasets into a single SquashFS image for HPC
#
# SquashFS packs all files into ONE compressed file. Singularity mounts it
# as a normal directory at runtime — no file count quota issues on HPC.
#
# Install (macOS):   brew install squashfs
# Install (Linux):   sudo apt install squashfs-tools
#
# Usage:
#   bash singularity/pack_data.sh [data_source_dir] [output_file]
#
# Example:
#   bash singularity/pack_data.sh "/Volumes/Lang/Research/Data/3D Ultrasound/processed"
#
# This creates ultrassl_data.sqsh containing:
#   /data/BIrads_all/...
#   /data/Duying_all/...
#   etc.
#
# Then upload the single file to HPC:
#   rsync -avP ultrassl_data.sqsh <netid>@jubail.abudhabi.nyu.edu:/scratch/<netid>/ultrassl/
# =============================================================================

set -euo pipefail

# Defaults
DATA_SOURCE="${1:-/Volumes/Lang/Research/Data/3D Ultrasound/processed}"
OUTPUT="${2:-ultrassl_data.sqsh}"

# Check mksquashfs
if ! command -v mksquashfs &>/dev/null; then
    echo "ERROR: mksquashfs not found."
    echo "  macOS:  brew install squashfs"
    echo "  Linux:  sudo apt install squashfs-tools"
    exit 1
fi

# Resolve to absolute path
DATA_SOURCE="$(cd "$DATA_SOURCE" && pwd)"

echo "============================================="
echo "Packing ultrasound data into SquashFS"
echo "  Source:  $DATA_SOURCE"
echo "  Output:  $OUTPUT"
echo "============================================="

# Find all dataset subdirectories
DATASETS=()
for ds_dir in "$DATA_SOURCE"/*/; do
    [ -d "$ds_dir" ] || continue
    ds_name=$(basename "$ds_dir")
    echo "  Found dataset: $ds_name"
    DATASETS+=("$ds_dir")
done

if [ ${#DATASETS[@]} -eq 0 ]; then
    echo "ERROR: No dataset directories found in $DATA_SOURCE"
    exit 1
fi

# Count total files
echo ""
echo "Counting files..."
TOTAL_FILES=$(find "${DATASETS[@]}" -type f | wc -l | tr -d ' ')
echo "  Total files: $TOTAL_FILES"
echo "  Datasets:    ${#DATASETS[@]}"

# Build SquashFS image directly from the source directory.
# Use -keep-as-directory so the source dir becomes the root.
# The files will appear at /BIrads_all/..., /Duying_all/..., etc.
# inside the squashfs. We wrap them under /data/ by creating a
# staging directory with hard structure.
echo ""
echo "Creating staging directory..."
STAGING_ROOT="$(mktemp -d)"
STAGING_DATA="$STAGING_ROOT/data"
mkdir -p "$STAGING_DATA"

# Use rsync to copy actual files (not symlinks) into staging
for ds_dir in "${DATASETS[@]}"; do
    ds_name=$(basename "$ds_dir")
    echo "  Staging: $ds_name ..."
    rsync -a "$ds_dir" "$STAGING_DATA/$ds_name/"
done

echo ""
echo "Building SquashFS image (this may take a while)..."
NPROC="$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)"

mksquashfs "$STAGING_ROOT" "$OUTPUT" \
    -noappend \
    -processors "$NPROC"

# Cleanup staging
rm -rf "$STAGING_ROOT"

# Report
SQS_SIZE=$(du -h "$OUTPUT" | cut -f1)
echo ""
echo "============================================="
echo "Done!"
echo "  Output:     $OUTPUT"
echo "  Size:       $SQS_SIZE"
echo "  Files:      $TOTAL_FILES packed into 1 file"
echo ""
echo "Upload to HPC:"
echo "  rsync -avP $OUTPUT <netid>@jubail.abudhabi.nyu.edu:/scratch/<netid>/ultrassl/"
echo ""
echo "The data will appear at /data/ inside the container."
echo "============================================="
