"""
Ultrasound-adapted multi-crop augmentation for DINOv2 pretraining.

Key differences from standard DINOv2 augmentations:
- Grayscale-aware: no hue/saturation jitter, no random grayscale conversion
- No solarize (not meaningful for ultrasound)
- Conservative crop scales to preserve small lesions
- Additional: Gaussian noise (speckle), random gamma, vertical flip
- Frequency-domain augmentations (FDA, spectral band randomization, spectral dropout)
"""

import logging
import random

import numpy as np
from PIL import Image, ImageFilter
from torchvision import transforms

from dinov2.data.transforms import GaussianBlur, make_normalize_transform
from .freq_augment import RandomFrequencyAugment

logger = logging.getLogger("ultrassl")


class RandomGaussianNoise:
    """Add Gaussian noise to a PIL Image (simulates ultrasound speckle).

    Args:
        std_range: (min_std, max_std) for noise standard deviation (in [0, 1] scale).
        p: Probability of applying noise.
    """

    def __init__(self, std_range=(0.01, 0.05), p=0.4):
        self.std_range = std_range
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img
        arr = np.array(img, dtype=np.float32) / 255.0
        std = random.uniform(*self.std_range)
        noise = np.random.randn(*arr.shape).astype(np.float32) * std
        arr = np.clip(arr + noise, 0.0, 1.0)
        return Image.fromarray((arr * 255).astype(np.uint8), mode=img.mode)


class RandomGamma:
    """Random gamma correction for contrast variation.

    Args:
        gamma_range: (min_gamma, max_gamma).
        p: Probability of applying.
    """

    def __init__(self, gamma_range=(0.7, 1.5), p=0.3):
        self.gamma_range = gamma_range
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img
        gamma = random.uniform(*self.gamma_range)
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = np.power(np.clip(arr, 1e-8, 1.0), gamma)
        return Image.fromarray((arr * 255).astype(np.uint8), mode=img.mode)


class UltrasoundAugmentationDINO:
    """Multi-crop augmentation pipeline adapted for ultrasound images.

    Produces the same output dict as DINOv2's DataAugmentationDINO:
        {
            "global_crops": [tensor, tensor],       # 2 global crops
            "global_crops_teacher": [tensor, tensor],
            "local_crops": [tensor, ...],           # N local crops
            "offsets": ()
        }

    Args:
        global_crops_scale: Scale range for global crop RandomResizedCrop.
        local_crops_scale: Scale range for local crop RandomResizedCrop.
        local_crops_number: Number of local crops per image.
        global_crops_size: Pixel size of global crops.
        local_crops_size: Pixel size of local crops.
        aug_cfg: Optional OmegaConf node with augmentation hyperparameters.
    """

    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        global_crops_size=224,
        local_crops_size=98,
        aug_cfg=None,
    ):
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size

        # Unpack augmentation config with defaults
        brightness = 0.3 if aug_cfg is None else aug_cfg.get("brightness_jitter", 0.3)
        contrast = 0.3 if aug_cfg is None else aug_cfg.get("contrast_jitter", 0.3)
        noise_std = (0.01, 0.05) if aug_cfg is None else tuple(aug_cfg.get("gaussian_noise_std", [0.01, 0.05]))
        noise_p = 0.4 if aug_cfg is None else aug_cfg.get("gaussian_noise_p", 0.4)
        gamma_range = (0.7, 1.5) if aug_cfg is None else tuple(aug_cfg.get("gamma_range", [0.7, 1.5]))
        gamma_p = 0.3 if aug_cfg is None else aug_cfg.get("gamma_p", 0.3)
        vflip_p = 0.3 if aug_cfg is None else aug_cfg.get("vertical_flip_p", 0.3)
        freq_p = 0.3 if aug_cfg is None else aug_cfg.get("freq_augment_p", 0.3)
        fda_beta = (0.01, 0.15) if aug_cfg is None else tuple(aug_cfg.get("fda_beta", [0.01, 0.15]))
        spectral_mag = 0.3 if aug_cfg is None else aug_cfg.get("spectral_band_magnitude", 0.3)
        spectral_drop = 0.15 if aug_cfg is None else aug_cfg.get("spectral_dropout_ratio", 0.15)

        logger.info("###################################")
        logger.info("UltraSSL augmentation parameters:")
        logger.info(f"  global_crops_scale: {global_crops_scale}")
        logger.info(f"  local_crops_scale: {local_crops_scale}")
        logger.info(f"  local_crops_number: {local_crops_number}")
        logger.info(f"  global_crops_size: {global_crops_size}")
        logger.info(f"  local_crops_size: {local_crops_size}")
        logger.info(f"  brightness/contrast jitter: {brightness}/{contrast}")
        logger.info(f"  gaussian noise: std={noise_std}, p={noise_p}")
        logger.info(f"  gamma: range={gamma_range}, p={gamma_p}")
        logger.info(f"  freq augment p: {freq_p}")
        logger.info("###################################")

        # --- Geometric augmentations (crop + flip) ---
        self.geometric_augmentation_global = transforms.Compose([
            transforms.RandomResizedCrop(
                global_crops_size,
                scale=global_crops_scale,
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=vflip_p),
        ])

        self.geometric_augmentation_local = transforms.Compose([
            transforms.RandomResizedCrop(
                local_crops_size,
                scale=local_crops_scale,
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=vflip_p),
        ])

        # --- Intensity augmentations (grayscale-aware) ---
        # No hue/saturation (ultrasound is grayscale); no solarize
        color_jittering = transforms.RandomApply(
            [transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=0, hue=0)],
            p=0.5,
        )

        gaussian_noise = RandomGaussianNoise(std_range=noise_std, p=noise_p)
        random_gamma = RandomGamma(gamma_range=gamma_range, p=gamma_p)

        # Frequency-domain augmentations
        freq_augment = RandomFrequencyAugment(
            fda_beta=fda_beta,
            spectral_band_magnitude=spectral_mag,
            spectral_dropout_ratio=spectral_drop,
            p=freq_p,
        )

        # Global view 1: strong blur + noise
        global_transfo1_extra = GaussianBlur(p=1.0)
        # Global view 2: mild blur + gamma
        global_transfo2_extra = GaussianBlur(p=0.1)
        # Local views: medium blur
        local_transfo_extra = GaussianBlur(p=0.5)

        # --- Normalization (ImageNet stats for pretrained weight compat) ---
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            make_normalize_transform(),
        ])

        # --- Compose per-view pipelines ---
        # Frequency augment is applied on PIL before intensity transforms
        self.global_transfo1 = transforms.Compose([
            freq_augment,
            color_jittering,
            gaussian_noise,
            global_transfo1_extra,
            self.normalize,
        ])
        self.global_transfo2 = transforms.Compose([
            freq_augment,
            color_jittering,
            random_gamma,
            global_transfo2_extra,
            self.normalize,
        ])
        self.local_transfo = transforms.Compose([
            color_jittering,
            gaussian_noise,
            local_transfo_extra,
            self.normalize,
        ])

    def __call__(self, image):
        output = {}

        # Global crops (2)
        im1_base = self.geometric_augmentation_global(image)
        global_crop_1 = self.global_transfo1(im1_base)

        im2_base = self.geometric_augmentation_global(image)
        global_crop_2 = self.global_transfo2(im2_base)

        output["global_crops"] = [global_crop_1, global_crop_2]
        output["global_crops_teacher"] = [global_crop_1, global_crop_2]

        # Local crops (N)
        local_crops = [
            self.local_transfo(self.geometric_augmentation_local(image))
            for _ in range(self.local_crops_number)
        ]
        output["local_crops"] = local_crops
        output["offsets"] = ()

        return output
