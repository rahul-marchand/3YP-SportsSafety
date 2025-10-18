"""
DataLoader for Columbia Gaze Dataset.

Parses image filenames to extract gaze angles and provides PyTorch Dataset interface.
Filename format: {subject}_{distance}_{pitch}_{vertical}_{horizontal}.jpg
Example: 0052_2m_-15P_0V_10H.jpg
"""

import re
from pathlib import Path
from typing import Literal

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class ColumbiaGazeDataset(Dataset):
    """Dataset for Columbia Gaze Data Set with gaze angle regression."""

    def __init__(
        self,
        root_dir: str | Path,
        transform: transforms.Compose | None = None,
        split: Literal["train", "val", "test"] | None = None,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
    ):
        """
        Initialize the Columbia Gaze Dataset.

        Args:
            root_dir: Root directory containing the dataset
            transform: Optional image transforms
            split: Which split to use ('train', 'val', 'test', or None for all data)
            train_ratio: Proportion of data for training
            val_ratio: Proportion of data for validation
            test_ratio: Proportion of data for testing
            seed: Random seed for reproducible splits
        """
        self.root_dir = Path(root_dir)
        self.transform = transform or self._default_transform()

        # Find all images in the dataset
        self.image_paths = sorted(
            self.root_dir.glob("columbia_gaze_data_set/Columbia Gaze Data Set/*/*.jpg")
        )

        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {self.root_dir}")

        # Parse all labels from filenames
        self.labels = [self._parse_filename(path.name) for path in self.image_paths]

        # Handle train/val/test split
        if split is not None:
            total_size = len(self.image_paths)
            train_size = int(total_size * train_ratio)
            val_size = int(total_size * val_ratio)

            # Create splits with fixed seed for reproducibility
            generator = torch.Generator().manual_seed(seed)
            indices = torch.randperm(total_size, generator=generator).tolist()

            if split == "train":
                indices = indices[:train_size]
            elif split == "val":
                indices = indices[train_size : train_size + val_size]
            elif split == "test":
                indices = indices[train_size + val_size :]
            else:
                raise ValueError(f"Invalid split: {split}")

            self.image_paths = [self.image_paths[i] for i in indices]
            self.labels = [self.labels[i] for i in indices]

    @staticmethod
    def _default_transform() -> transforms.Compose:
        """Default transform for MobileNet preprocessing."""
        return transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    @staticmethod
    def _parse_filename(filename: str) -> tuple[float, float]:
        """
        Parse gaze angles from filename.

        Filename format: {subject}_{distance}_{pitch}_{vertical}_{horizontal}.jpg
        Example: 0052_2m_-15P_0V_10H.jpg -> (0.0, 10.0)

        Args:
            filename: Image filename

        Returns:
            Tuple of (vertical_angle, horizontal_angle) in degrees
        """
        # Pattern: {subject}_{distance}_{pitch}_{vertical}_{horizontal}.jpg
        pattern = r"\d+_\d+m_-?\d+P_(-?\d+)V_(-?\d+)H\.jpg"
        match = re.match(pattern, filename)

        if not match:
            raise ValueError(f"Filename {filename} does not match expected pattern")

        vertical = float(match.group(1))
        horizontal = float(match.group(2))

        return (vertical, horizontal)

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset.

        Args:
            idx: Index of the sample

        Returns:
            Tuple of (image, label) where label is [vertical, horizontal] angles
        """
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        # Get label (vertical, horizontal angles)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        return image, label


def get_dataloaders(
    root_dir: str | Path,
    batch_size: int = 32,
    num_workers: int = 4,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.

    Args:
        root_dir: Root directory containing the dataset
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for data loading
        train_ratio: Proportion of data for training
        val_ratio: Proportion of data for validation
        test_ratio: Proportion of data for testing
        seed: Random seed for reproducible splits

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets for each split
    train_dataset = ColumbiaGazeDataset(
        root_dir,
        split="train",
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    val_dataset = ColumbiaGazeDataset(
        root_dir,
        split="val",
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    test_dataset = ColumbiaGazeDataset(
        root_dir,
        split="test",
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    # Only use pin_memory if CUDA is available
    use_pin_memory = torch.cuda.is_available()

    # Create dataloaders
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
