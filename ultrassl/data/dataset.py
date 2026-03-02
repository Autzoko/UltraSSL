"""
Robust ultrasound dataset loader for DINOv2 pretraining.

Handles:
- Multiple dataset roots from config/data_root.json
- Both 2D ultrasound images and 2D slices from 3D ABUS volumes
- Volume-aware stride subsampling to reduce near-identical adjacent slices
- Graceful handling of evolving folder structures (re-scan on stale cache)
- Transparent loading of .png/.jpg/.bmp/.tif/.npy/.npz files
- Grayscale → 3-channel RGB conversion for pretrained weight compatibility
"""

import json
import logging
import os
import re
import hashlib
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

logger = logging.getLogger("ultrassl")

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
ARRAY_EXTENSIONS = {".npy", ".npz"}
ALL_EXTENSIONS = IMAGE_EXTENSIONS | ARRAY_EXTENSIONS


def _is_image_file(path: str, extensions: set) -> bool:
    return Path(path).suffix.lower() in extensions


def _load_image_as_rgb(path: str) -> Image.Image:
    """Load any supported image file as a 3-channel RGB PIL Image."""
    ext = Path(path).suffix.lower()

    if ext in ARRAY_EXTENSIONS:
        if ext == ".npz":
            data = np.load(path)
            arr = data[list(data.keys())[0]]
        else:
            arr = np.load(path)

        # Handle various array shapes
        if arr.ndim == 3 and arr.shape[0] <= 4:  # C, H, W
            arr = arr[0]  # Take first channel
        elif arr.ndim == 3 and arr.shape[-1] <= 4:  # H, W, C
            arr = arr[..., 0]
        elif arr.ndim > 3:
            arr = arr.reshape(-1, arr.shape[-2], arr.shape[-1])[0]

        # Normalize to uint8
        if arr.dtype == np.float64 or arr.dtype == np.float32:
            if arr.max() <= 1.0:
                arr = (arr * 255).astype(np.uint8)
            else:
                arr = arr.astype(np.uint8)
        elif arr.dtype == np.uint16:
            arr = (arr / 256).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)

        img = Image.fromarray(arr, mode="L")
    else:
        img = Image.open(path)

    # Convert to RGB (grayscale → 3 identical channels)
    return img.convert("RGB")


def _scan_directory(
    root: str,
    extensions: set,
    subdir_filter: str = "img",
) -> List[str]:
    """Recursively find all image files under root.

    Args:
        root: Root directory to scan.
        extensions: Allowed file extensions.
        subdir_filter: If set, only include files whose parent directory name
            matches this string. This filters out overlay/label directories
            in typical ultrasound volume layouts (patient/study/img/*.png).
            Set to "" to disable filtering.
    """
    paths = []
    if not os.path.isdir(root):
        logger.warning(f"Dataset root does not exist: {root}")
        return paths
    for dirpath, _, filenames in os.walk(root):
        # Skip directories that don't match the filter (e.g., overlay/, label/)
        if subdir_filter and os.path.basename(dirpath) != subdir_filter:
            # Also accept the root itself and intermediate directories that
            # aren't leaf directories containing images
            if any(_is_image_file(f, extensions) for f in filenames[:3]):
                # This directory has images but doesn't match the filter — skip
                continue
        for fname in sorted(filenames):
            if _is_image_file(fname, extensions):
                paths.append(os.path.join(dirpath, fname))
    return paths


def _detect_volume_slices(paths: List[str]) -> List[Tuple[str, List[str]]]:
    """Group paths by parent directory to detect volume sequences.

    Returns list of (volume_id, [slice_paths]) where volume_id is the
    parent directory path and slices are sorted numerically.
    """
    from collections import defaultdict
    volume_groups = defaultdict(list)
    for p in paths:
        parent = os.path.dirname(p)
        volume_groups[parent].append(p)

    volumes = []
    for vol_id, vol_paths in volume_groups.items():
        # Sort by numeric component in filename for proper slice ordering
        def _sort_key(fp):
            nums = re.findall(r"\d+", os.path.basename(fp))
            return int(nums[-1]) if nums else 0
        vol_paths.sort(key=_sort_key)
        volumes.append((vol_id, vol_paths))
    return volumes


def _subsample_volume(
    slice_paths: List[str],
    stride: int = 3,
    min_slices_for_subsampling: int = 10,
) -> List[str]:
    """Subsample volume slices to reduce near-duplicate adjacent slices.

    Only applies subsampling if the volume has enough slices to warrant it.
    """
    if len(slice_paths) < min_slices_for_subsampling:
        return slice_paths
    return slice_paths[::stride]


class UltrasoundDataset(Dataset):
    """Multi-source ultrasound dataset with volume-aware loading.

    Args:
        data_root_json: Path to data_root.json with dataset definitions.
        transform: Augmentation transform (e.g., UltrasoundAugmentationDINO).
        target_transform: Transform for targets (not used in SSL, defaults to lambda: ()).
        extensions: Set of allowed file extensions.
        volume_slice_stride: Stride for subsampling 3D volume slices.
        min_slice_entropy: Minimum entropy threshold to keep a slice (0 to disable).
        rescan_interval: Seconds between directory re-scans (0 = no re-scan).
    """

    def __init__(
        self,
        data_root_json: str,
        transform=None,
        target_transform=None,
        extensions: Optional[set] = None,
        volume_slice_stride: int = 3,
        min_slice_entropy: float = 0.0,
        rescan_interval: int = 0,
    ):
        self.transform = transform
        self.target_transform = target_transform or (lambda _: ())
        self.extensions = extensions or ALL_EXTENSIONS
        self.volume_slice_stride = volume_slice_stride
        self.min_slice_entropy = min_slice_entropy
        self.rescan_interval = rescan_interval
        self._last_scan_time = 0

        # Load dataset roots
        with open(data_root_json, "r") as f:
            config = json.load(f)

        self.dataset_entries = config["data"]
        self.image_paths: List[str] = []
        self.dataset_labels: List[str] = []  # source dataset name per image

        self._scan_all()

    def _resolve_path(self, path: str) -> str:
        """Try the configured path, and if missing, try the 'processd' variant."""
        if os.path.isdir(path):
            return path
        # Try without final 'e' in 'processed' → 'processd'
        alt = path.replace("/processed/", "/processd/")
        if os.path.isdir(alt):
            logger.info(f"Using alternate path: {alt} (original {path} not found)")
            return alt
        # Try with 'e' added: 'processd' → 'processed'
        alt2 = path.replace("/processd/", "/processed/")
        if os.path.isdir(alt2):
            logger.info(f"Using alternate path: {alt2} (original {path} not found)")
            return alt2
        return path  # Return original, will warn during scan

    def _looks_like_volume_data(self, paths: List[str]) -> bool:
        """Auto-detect volume data: many images grouped in folders with >50 slices each."""
        if not paths:
            return False
        volumes = _detect_volume_slices(paths)
        if not volumes:
            return False
        avg_slices = len(paths) / len(volumes)
        return avg_slices > 20  # Volumes typically have hundreds of slices

    def _scan_all(self):
        """Scan all dataset roots and build the image path list."""
        self.image_paths = []
        self.dataset_labels = []

        for entry in self.dataset_entries:
            name = entry["name"]
            root = self._resolve_path(entry["path"])
            raw_paths = _scan_directory(root, self.extensions)

            if not raw_paths:
                logger.warning(f"Dataset '{name}' at {root}: no images found")
                continue

            # Auto-detect volume data and apply subsampling
            is_volume = self._looks_like_volume_data(raw_paths)
            if is_volume and self.volume_slice_stride > 1:
                volumes = _detect_volume_slices(raw_paths)
                subsampled = []
                for vol_id, slices in volumes:
                    subsampled.extend(
                        _subsample_volume(slices, stride=self.volume_slice_stride)
                    )
                logger.info(
                    f"Dataset '{name}': {len(raw_paths)} slices → "
                    f"{len(subsampled)} after stride-{self.volume_slice_stride} subsampling "
                    f"({len(volumes)} volumes, detected as 3D volume data)"
                )
                raw_paths = subsampled
            else:
                logger.info(f"Dataset '{name}': {len(raw_paths)} images")

            self.image_paths.extend(raw_paths)
            self.dataset_labels.extend([name] * len(raw_paths))

        self._last_scan_time = time.time()
        logger.info(f"Total dataset size: {len(self.image_paths)} images")

    def _maybe_rescan(self):
        """Re-scan directories if enough time has passed (for evolving datasets)."""
        if self.rescan_interval > 0:
            if time.time() - self._last_scan_time > self.rescan_interval:
                logger.info("Re-scanning dataset directories...")
                self._scan_all()

    def __len__(self):
        self._maybe_rescan()
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]

        try:
            img = _load_image_as_rgb(path)
        except Exception as e:
            logger.warning(f"Failed to load {path}: {e}. Using random fallback.")
            fallback_idx = torch.randint(0, len(self.image_paths), (1,)).item()
            if fallback_idx == idx:
                fallback_idx = (idx + 1) % len(self.image_paths)
            img = _load_image_as_rgb(self.image_paths[fallback_idx])

        if self.transform is not None:
            img = self.transform(img)

        target = self.target_transform(idx)

        return img, target
