"""OpenEDS Dataset loader for pupil/iris segmentation."""

from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from ..config import (
    CLAHE_CLIP_LIMIT,
    CLAHE_TILE_SIZE,
    GAMMA,
    NORMALIZE_MEAN,
    NORMALIZE_STD,
    OPENEDS_IMAGE_SIZE,
)


class OpenEDSDataset(Dataset):
    """Dataset for OpenEDS semantic segmentation (pupil, iris, sclera, background)."""

    NUM_CLASSES = 4

    def __init__(
        self,
        root_dir: str | Path,
        split: Literal["train", "validation", "test"] = "train",
        apply_preprocessing: bool = True,
        target_size: tuple[int, int] | None = None,
        transform=None,
    ):
        """
        Initialize OpenEDS dataset.

        Args:
            root_dir: Root directory containing train/validation/test folders
            split: Dataset split to use
            apply_preprocessing: Whether to apply gamma correction and CLAHE
            target_size: Optional (H, W) to resize images. None keeps original size.
            transform: Optional SegmentationTransform for data augmentation.
                Called as transform(image, mask) -> (image, mask) on numpy arrays.
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.apply_preprocessing = apply_preprocessing
        self.target_size = target_size or OPENEDS_IMAGE_SIZE
        self.transform = transform

        # OpenEDS structure: root/split/images/*.png and root/split/labels/*.npy
        self.split_dir = self.root_dir / split
        self.images_dir = self.split_dir / "images"
        self.labels_dir = self.split_dir / "labels"

        if not self.images_dir.exists():
            raise ValueError(f"Images directory not found: {self.images_dir}")

        # Get all image files
        self.image_files = sorted(self.images_dir.glob("*.png"))
        if len(self.image_files) == 0:
            raise ValueError(f"No PNG images found in {self.images_dir}")

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Apply gamma correction and CLAHE preprocessing."""
        # Gamma correction
        image = np.power(image / 255.0, GAMMA) * 255.0
        image = image.astype(np.uint8)

        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_SIZE)
        image = clahe.apply(image)

        return image

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset.

        Returns:
            Tuple of (image, mask) where:
            - image: (1, H, W) float tensor, normalized
            - mask: (H, W) long tensor with class labels 0-3
        """
        img_path = self.image_files[idx]

        # Load grayscale image
        image = np.array(Image.open(img_path).convert("L"))

        # Load label (try .npy first, then .png)
        label_path_npy = self.labels_dir / f"{img_path.stem}.npy"
        label_path_png = self.labels_dir / f"{img_path.stem}.png"

        if label_path_npy.exists():
            mask = np.load(label_path_npy)
        elif label_path_png.exists():
            mask = np.array(Image.open(label_path_png))
        else:
            raise FileNotFoundError(f"No label found for {img_path.name}")

        # Apply preprocessing
        if self.apply_preprocessing:
            image = self._preprocess(image)

        # Apply augmentation (training only — val/test should pass transform=None)
        if self.transform is not None:
            image, mask = self.transform(image, mask)

        # Resize if needed
        if (image.shape[0], image.shape[1]) != self.target_size:
            image = cv2.resize(image, (self.target_size[1], self.target_size[0]))
            mask = cv2.resize(
                mask, (self.target_size[1], self.target_size[0]), interpolation=cv2.INTER_NEAREST
            )

        # Normalize image
        image = image.astype(np.float32) / 255.0
        image = (image - NORMALIZE_MEAN) / NORMALIZE_STD

        # Convert to tensors
        image_tensor = torch.from_numpy(image).unsqueeze(0)  # (1, H, W)
        mask_tensor = torch.from_numpy(mask).long()  # (H, W)

        return image_tensor, mask_tensor


def get_dataloaders(
    root_dir: str | Path,
    batch_size: int = 8,
    num_workers: int = 4,
    apply_preprocessing: bool = True,
    target_size: tuple[int, int] | None = None,
    augmentation: str = "none",
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.

    Args:
        root_dir: Root directory containing the dataset
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for data loading
        apply_preprocessing: Whether to apply gamma correction and CLAHE
        target_size: Optional (H, W) to resize images
        augmentation: Augmentation mode for training data. One of:
            "none" — no augmentation (default)
            "standard" — flip, rotation, brightness, contrast, noise
            "domain" — standard + sclera brightening + resolution downsample

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    from .augmentation import get_domain_transforms, get_standard_transforms

    valid_augmentations = ("none", "standard", "domain")
    if augmentation not in valid_augmentations:
        raise ValueError(
            f"Unknown augmentation '{augmentation}'. Choose from: {valid_augmentations}"
        )

    if augmentation == "standard":
        train_transform = get_standard_transforms()
    elif augmentation == "domain":
        train_transform = get_domain_transforms()
    else:
        train_transform = None

    use_pin_memory = torch.cuda.is_available()

    train_dataset = OpenEDSDataset(
        root_dir,
        split="train",
        apply_preprocessing=apply_preprocessing,
        target_size=target_size,
        transform=train_transform,
    )
    val_dataset = OpenEDSDataset(
        root_dir,
        split="validation",
        apply_preprocessing=apply_preprocessing,
        target_size=target_size,
    )
    test_dataset = OpenEDSDataset(
        root_dir, split="test", apply_preprocessing=apply_preprocessing, target_size=target_size
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )

    return train_loader, val_loader, test_loader
