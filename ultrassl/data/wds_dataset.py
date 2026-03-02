"""
WebDataset-based data loading for UltraSSL.

Reads images from .tar shard files created by create_shards.py.
Produces the same (image, target) output format as UltrasoundDataset,
so it plugs directly into the existing augmentation + collation pipeline.
"""

import glob
import logging
import os

from PIL import Image
import webdataset as wds

logger = logging.getLogger("ultrassl")


def build_wds_dataset(
    shard_dir: str,
    transform=None,
    epoch_length: int = 1000,
    batch_size: int = 24,
    num_workers: int = 8,
    shuffle_buffer: int = 5000,
    world_size: int = 1,
    rank: int = 0,
):
    """Build a WebDataset pipeline that returns (augmented_image, target) tuples.

    Args:
        shard_dir: Directory containing shard-XXXXXX.tar files.
        transform: Augmentation transform (UltrasoundAugmentationDINO).
        epoch_length: Number of batches per epoch.
        batch_size: Batch size per GPU.
        num_workers: DataLoader workers.
        shuffle_buffer: Number of samples to buffer for shuffling.
        world_size: Number of DDP processes.
        rank: Current DDP rank.

    Returns:
        WebLoader (drop-in replacement for DataLoader).
    """
    # Find all shards
    shard_pattern = os.path.join(shard_dir, "shard-*.tar")
    shards = sorted(glob.glob(shard_pattern))
    if not shards:
        raise FileNotFoundError(f"No shard files found in {shard_dir}")

    n_shards = len(shards)
    logger.info(f"WebDataset: {n_shards} shards from {shard_dir}")

    # Use brace expansion pattern for webdataset
    shard_list = os.path.join(shard_dir, "shard-{000000..%06d}.tar" % (n_shards - 1))

    def decode_sample(sample):
        """Decode a webdataset sample to (PIL Image, target)."""
        img = sample["png"]  # Already decoded to PIL by webdataset
        if img.mode != "RGB":
            img = img.convert("RGB")

        if transform is not None:
            img = transform(img)

        target = ()
        return img, target

    # Build pipeline
    dataset = (
        wds.WebDataset(shards, shardshuffle=True, nodesplitter=wds.split_by_node)
        .shuffle(shuffle_buffer)
        .decode("pil")
        .map(decode_sample)
    )

    # WebLoader wraps DataLoader with epoch length control
    loader = wds.WebLoader(
        dataset,
        batch_size=None,  # batching is handled by collate_fn in trainer
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    # Set epoch length (WebDataset streams infinitely, so we define epoch boundaries)
    n_batches = epoch_length
    loader = loader.repeat(2).slice(n_batches * batch_size)

    logger.info(f"WebDataset: shuffle_buffer={shuffle_buffer}, epoch={n_batches} batches")

    return dataset, loader
