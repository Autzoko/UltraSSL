from .dataset import UltrasoundDataset

# Lazy imports — these require dinov2 on PYTHONPATH
def __getattr__(name):
    if name == "UltrasoundAugmentationDINO":
        from .augmentations import UltrasoundAugmentationDINO
        return UltrasoundAugmentationDINO
    if name in ("FDAAmplitudeMix", "SpectralBandRandomization", "SpectralDropout"):
        from . import freq_augment
        return getattr(freq_augment, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
