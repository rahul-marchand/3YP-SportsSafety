"""Data augmentation for semantic segmentation.

Provides synchronized transforms for image-mask pairs where spatial
transforms are applied identically to both image and mask.
"""

import random

import cv2
import numpy as np

from ..config import COLLECTED_CROP_SIZE


class SegmentationTransform:
    """Synchronized transforms for image and mask pairs.

    All spatial transforms (flip, rotation) are applied identically to both
    image and mask. Intensity transforms (brightness, contrast, noise) only
    apply to the image.

    Masks use INTER_NEAREST interpolation to preserve integer class labels.
    """

    def __init__(
        self,
        horizontal_flip_prob: float = 0.5,
        vertical_flip_prob: float = 0.0,
        rotation_degrees: float = 10.0,
        brightness_range: tuple[float, float] = (0.8, 1.2),
        contrast_range: tuple[float, float] = (0.8, 1.2),
        gaussian_noise_std: float = 0.02,
        random_crop_size: tuple[int, int] | None = None,
        sclera_brighten_prob: float = 0.0,
        sclera_boost_range: tuple[float, float] = (1.8, 2.4),
        sclera_feather_radius: int = 15,
        resolution_downsample_size: tuple[int, int] | None = None,
    ):
        """Initialize transform.

        Args:
            horizontal_flip_prob: Probability of horizontal flip
            vertical_flip_prob: Probability of vertical flip
            rotation_degrees: Max rotation angle (applies randomly in [-deg, +deg])
            brightness_range: (min, max) brightness factor (1.0 = no change)
            contrast_range: (min, max) contrast factor (1.0 = no change)
            gaussian_noise_std: Standard deviation of Gaussian noise (0 to disable)
            random_crop_size: Optional (H, W) for random cropping. None to disable.
            sclera_brighten_prob: Probability of applying sclera brightening.
                Simulates the bright sclera appearance in our collected headset
                images, which differs from the darker sclera in OpenEDS (caused
                by different IR illumination setups). Uses the ground-truth mask
                to identify sclera pixels (class 1) so only the correct region
                is affected. Set to 0.0 to disable.
            sclera_boost_range: (min, max) multiplicative brightness boost applied
                to sclera pixels when sclera_brighten_prob fires.
            sclera_feather_radius: Radius (px) of the Gaussian blur used to
                feather the sclera mask before applying the boost. Creates a
                smooth brightness falloff at the iris/sclera boundary.
            resolution_downsample_size: If set, the image is downsampled to this
                (H, W) and then upsampled back to its original size. Simulates
                the lower effective resolution of our collected images. Set to
                None to disable.
        """
        self.horizontal_flip_prob = horizontal_flip_prob
        self.vertical_flip_prob = vertical_flip_prob
        self.rotation_degrees = rotation_degrees
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.gaussian_noise_std = gaussian_noise_std
        self.random_crop_size = random_crop_size
        self.sclera_brighten_prob = sclera_brighten_prob
        self.sclera_boost_range = sclera_boost_range
        self.sclera_feather_radius = sclera_feather_radius
        self.resolution_downsample_size = resolution_downsample_size

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Apply synchronized transforms to image and mask.

        Args:
            image: Grayscale image (H, W) as uint8 or float
            mask: Segmentation mask (H, W) with integer class labels

        Returns:
            Tuple of (transformed_image, transformed_mask)
        """
        # Ensure image is float for transformations
        was_uint8 = image.dtype == np.uint8
        if was_uint8:
            image = image.astype(np.float32)

        # Random horizontal flip
        if random.random() < self.horizontal_flip_prob:
            image = np.fliplr(image).copy()
            mask = np.fliplr(mask).copy()

        # Random vertical flip
        if random.random() < self.vertical_flip_prob:
            image = np.flipud(image).copy()
            mask = np.flipud(mask).copy()

        # Random rotation
        if self.rotation_degrees > 0:
            angle = random.uniform(-self.rotation_degrees, self.rotation_degrees)
            image, mask = self._rotate(image, mask, angle)

        # Random crop
        if self.random_crop_size is not None:
            image, mask = self._random_crop(image, mask, self.random_crop_size)

        # Intensity transforms (image only)
        # Brightness
        if self.brightness_range != (1.0, 1.0):
            brightness = random.uniform(*self.brightness_range)
            image = image * brightness

        # Contrast
        if self.contrast_range != (1.0, 1.0):
            contrast = random.uniform(*self.contrast_range)
            mean = image.mean()
            image = (image - mean) * contrast + mean

        # Gaussian noise
        if self.gaussian_noise_std > 0:
            noise = np.random.randn(*image.shape).astype(np.float32) * self.gaussian_noise_std
            if was_uint8:
                noise *= 255.0
            image = image + noise

        # Sclera brightening: selectively boosts sclera pixels to simulate the
        # bright sclera appearance seen in our collected headset images.
        if self.sclera_brighten_prob > 0 and random.random() < self.sclera_brighten_prob:
            boost = random.uniform(*self.sclera_boost_range)
            image = self._brighten_sclera(image, mask, boost, self.sclera_feather_radius)

        # Resolution downsampling: shrink then restore to simulate the lower
        # effective resolution of our collected images (112x112 after cropping).
        if self.resolution_downsample_size is not None:
            image = self._simulate_low_resolution(image, self.resolution_downsample_size)

        # Clip values
        if was_uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        else:
            image = np.clip(image, 0, 1)

        return image, mask

    def _rotate(
        self, image: np.ndarray, mask: np.ndarray, angle: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Rotate image and mask by the same angle."""
        h, w = image.shape[:2]
        center = (w / 2, h / 2)
        rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        rotated_image = cv2.warpAffine(
            image,
            rot_matrix,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        rotated_mask = cv2.warpAffine(
            mask.astype(np.float32),
            rot_matrix,
            (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_REFLECT_101,
        ).astype(mask.dtype)

        return rotated_image, rotated_mask

    def _brighten_sclera(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        boost: float,
        feather_radius: int,
    ) -> np.ndarray:
        """Selectively brighten sclera pixels using the ground-truth mask.

        The sclera mask is Gaussian-blurred before applying the boost so that
        the brightness increase fades out gradually at the iris/sclera boundary.
        """
        sclera_binary = (mask == 1).astype(np.float32)
        kernel_size = feather_radius * 2 + 1
        weight = cv2.GaussianBlur(sclera_binary, (kernel_size, kernel_size), feather_radius / 2)
        image = image + weight * image * (boost - 1)
        return image

    def _simulate_low_resolution(
        self,
        image: np.ndarray,
        downsample_size: tuple[int, int],
    ) -> np.ndarray:
        """Downsample and restore to simulate lower effective resolution."""
        original_h, original_w = image.shape[:2]
        down_h, down_w = downsample_size
        small = cv2.resize(image, (down_w, down_h), interpolation=cv2.INTER_AREA)
        restored = cv2.resize(small, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
        return restored

    def _random_crop(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        crop_size: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Random crop from image and mask."""
        h, w = image.shape[:2]
        crop_h, crop_w = crop_size

        if crop_h >= h or crop_w >= w:
            return image, mask

        top = random.randint(0, h - crop_h)
        left = random.randint(0, w - crop_w)

        cropped_image = image[top : top + crop_h, left : left + crop_w]
        cropped_mask = mask[top : top + crop_h, left : left + crop_w]
        return cropped_image, cropped_mask


def get_standard_transforms() -> SegmentationTransform:
    """Standard augmentation: flip, rotate, brightness, contrast, noise.

    Returns:
        SegmentationTransform configured for training without domain adaptation
    """
    return SegmentationTransform(
        horizontal_flip_prob=0.5,
        vertical_flip_prob=0.0,
        rotation_degrees=10.0,
        brightness_range=(0.6, 1.0),
        contrast_range=(0.8, 1.2),
        gaussian_noise_std=0.02,
        random_crop_size=None,
        sclera_brighten_prob=0.0,
        resolution_downsample_size=None,
    )


def get_domain_transforms() -> SegmentationTransform:
    """Standard augmentation + domain adaptation for collected headset images.

    Includes sclera brightening (always applied, random boost amount) and
    resolution downsampling to simulate the lower effective resolution of
    our collected images (112x112 after vignette cropping vs 640x400 OpenEDS).

    Returns:
        SegmentationTransform configured for training with domain adaptation
    """
    return SegmentationTransform(
        horizontal_flip_prob=0.5,
        vertical_flip_prob=0.0,
        rotation_degrees=10.0,
        brightness_range=(0.6, 1.0),
        contrast_range=(0.8, 1.2),
        gaussian_noise_std=0.02,
        random_crop_size=None,
        sclera_brighten_prob=1.0,
        sclera_boost_range=(1.8, 2.4),
        sclera_feather_radius=15,
        resolution_downsample_size=COLLECTED_CROP_SIZE,
    )


def get_val_transforms() -> None:
    """No augmentation for validation/test.

    Returns:
        None (no transforms applied)
    """
    return None
