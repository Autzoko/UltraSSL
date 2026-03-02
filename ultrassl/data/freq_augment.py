"""
Frequency-domain augmentations for ultrasound self-supervised pretraining.

Three augmentation types, all preserving phase (which encodes anatomical structures)
and only modifying amplitude in the Fourier domain:

1. FDAAmplitudeMix  - Fourier Domain Adaptation: swap low-freq amplitude with a
                      self-referencing perturbed copy of the same image.
2. SpectralBandRandomization - Randomly scale amplitude in concentric frequency bands.
3. SpectralDropout  - Zero out random annular frequency bands.
"""

import random
import numpy as np
import torch
from PIL import Image, ImageEnhance


def _to_numpy_gray(img):
    """Convert PIL Image or tensor to float32 numpy [H, W] in [0, 1]."""
    if isinstance(img, Image.Image):
        arr = np.array(img.convert("L"), dtype=np.float32) / 255.0
    elif isinstance(img, torch.Tensor):
        arr = img.float().numpy()
        if arr.ndim == 3:
            arr = arr.mean(axis=0)
        if arr.max() > 1.0:
            arr = arr / 255.0
    elif isinstance(img, np.ndarray):
        arr = img.astype(np.float32)
        if arr.ndim == 3:
            arr = arr.mean(axis=-1) if arr.shape[-1] <= 4 else arr.mean(axis=0)
        if arr.max() > 1.0:
            arr = arr / 255.0
    else:
        raise TypeError(f"Unsupported image type: {type(img)}")
    return arr


def _numpy_to_pil_rgb(arr):
    """Convert float32 [H, W] array in [0, 1] back to PIL RGB Image."""
    arr = np.clip(arr, 0.0, 1.0)
    arr_uint8 = (arr * 255.0).astype(np.uint8)
    return Image.fromarray(arr_uint8, mode="L").convert("RGB")


def _make_circular_mask(h, w, radius_ratio):
    """Create a circular low-frequency mask centered at (0, 0) for fftshift'd spectrum."""
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
    radius = radius_ratio * min(h, w) / 2.0
    return (dist <= radius).astype(np.float32)


def _make_annular_bands(h, w, n_bands):
    """Partition frequency space into n_bands concentric annular bands.
    Returns list of boolean masks, from low-freq (center) to high-freq (edge)."""
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
    max_r = np.sqrt(cy ** 2 + cx ** 2)
    edges = np.linspace(0, max_r, n_bands + 1)
    bands = []
    for i in range(n_bands):
        mask = ((dist >= edges[i]) & (dist < edges[i + 1])).astype(np.float32)
        bands.append(mask)
    return bands


class FDAAmplitudeMix:
    """Fourier Domain Adaptation–style amplitude mixing (self-referencing).

    Creates a perturbed copy of the input image (via random gamma + noise),
    then swaps low-frequency amplitude components between source and reference.
    Phase is always preserved from the source, maintaining structural content.

    Args:
        beta_range: Tuple (min, max) for the low-freq radius ratio.
        p: Probability of applying this augmentation.
    """

    def __init__(self, beta_range=(0.01, 0.15), p=0.3):
        self.beta_range = beta_range
        self.p = p

    def _make_reference(self, img_np):
        """Create a perturbed version of the image as frequency reference."""
        ref = img_np.copy()
        # Random gamma
        gamma = random.uniform(0.6, 1.6)
        ref = np.power(np.clip(ref, 1e-8, 1.0), gamma)
        # Random additive noise
        noise_std = random.uniform(0.02, 0.08)
        ref = ref + np.random.randn(*ref.shape).astype(np.float32) * noise_std
        return np.clip(ref, 0.0, 1.0)

    def __call__(self, img):
        """Apply FDA augmentation. Input: PIL Image (RGB). Output: PIL Image (RGB)."""
        if random.random() > self.p:
            return img

        src = _to_numpy_gray(img)
        ref = self._make_reference(src)
        h, w = src.shape

        beta = random.uniform(*self.beta_range)
        mask = _make_circular_mask(h, w, beta)

        # FFT of source and reference
        src_fft = np.fft.fftshift(np.fft.fft2(src))
        ref_fft = np.fft.fftshift(np.fft.fft2(ref))

        src_amp = np.abs(src_fft)
        src_phase = np.angle(src_fft)
        ref_amp = np.abs(ref_fft)

        # Blend amplitude: replace low-freq of source with reference's low-freq
        mixed_amp = src_amp * (1.0 - mask) + ref_amp * mask

        # Reconstruct with source phase
        mixed_fft = mixed_amp * np.exp(1j * src_phase)
        result = np.real(np.fft.ifft2(np.fft.ifftshift(mixed_fft)))
        result = np.clip(result, 0.0, 1.0).astype(np.float32)

        return _numpy_to_pil_rgb(result)


class SpectralBandRandomization:
    """Randomly perturb amplitude in concentric frequency bands.

    Partitions the Fourier spectrum into annular bands and multiplies each
    band's amplitude by a random factor, adding controlled spectral variation
    while preserving phase (structure).

    Args:
        n_bands: Number of concentric frequency bands.
        magnitude: Maximum perturbation factor (amplitude *= uniform(1-mag, 1+mag)).
        p: Probability of applying this augmentation.
    """

    def __init__(self, n_bands=4, magnitude=0.3, p=0.3):
        self.n_bands = n_bands
        self.magnitude = magnitude
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img

        src = _to_numpy_gray(img)
        h, w = src.shape
        bands = _make_annular_bands(h, w, self.n_bands)

        fft = np.fft.fftshift(np.fft.fft2(src))
        amp = np.abs(fft)
        phase = np.angle(fft)

        for band_mask in bands:
            factor = random.uniform(1.0 - self.magnitude, 1.0 + self.magnitude)
            amp = amp * (1.0 + band_mask * (factor - 1.0))

        result_fft = amp * np.exp(1j * phase)
        result = np.real(np.fft.ifft2(np.fft.ifftshift(result_fft)))
        result = np.clip(result, 0.0, 1.0).astype(np.float32)

        return _numpy_to_pil_rgb(result)


class SpectralDropout:
    """Zero out random annular frequency bands.

    Randomly selects a subset of frequency bands and zeros their amplitude,
    creating a form of spectral regularization. Phase is preserved.

    Args:
        n_bands: Number of concentric frequency bands to partition into.
        drop_ratio: Fraction of bands to drop (0 to 1).
        p: Probability of applying this augmentation.
    """

    def __init__(self, n_bands=6, drop_ratio=0.15, p=0.2):
        self.n_bands = n_bands
        self.drop_ratio = drop_ratio
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img

        src = _to_numpy_gray(img)
        h, w = src.shape
        bands = _make_annular_bands(h, w, self.n_bands)

        fft = np.fft.fftshift(np.fft.fft2(src))
        amp = np.abs(fft)
        phase = np.angle(fft)

        n_drop = max(1, int(self.n_bands * self.drop_ratio))
        drop_indices = random.sample(range(self.n_bands), n_drop)

        for idx in drop_indices:
            amp = amp * (1.0 - bands[idx])

        result_fft = amp * np.exp(1j * phase)
        result = np.real(np.fft.ifft2(np.fft.ifftshift(result_fft)))
        result = np.clip(result, 0.0, 1.0).astype(np.float32)

        return _numpy_to_pil_rgb(result)


class RandomFrequencyAugment:
    """Meta-augmentation: randomly applies one of the three frequency augmentations.

    Args:
        fda_beta: Beta range for FDA.
        spectral_band_magnitude: Magnitude for SpectralBandRandomization.
        spectral_dropout_ratio: Drop ratio for SpectralDropout.
        p: Overall probability of applying any frequency augmentation.
    """

    def __init__(
        self,
        fda_beta=(0.01, 0.15),
        spectral_band_magnitude=0.3,
        spectral_dropout_ratio=0.15,
        p=0.3,
    ):
        self.p = p
        self.augments = [
            FDAAmplitudeMix(beta_range=fda_beta, p=1.0),
            SpectralBandRandomization(magnitude=spectral_band_magnitude, p=1.0),
            SpectralDropout(drop_ratio=spectral_dropout_ratio, p=1.0),
        ]

    def __call__(self, img):
        if random.random() > self.p:
            return img
        aug = random.choice(self.augments)
        return aug(img)
