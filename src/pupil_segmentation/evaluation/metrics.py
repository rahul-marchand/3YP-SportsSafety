"""Segmentation and benchmarking metrics."""

import time
from dataclasses import dataclass, field

import numpy as np
import torch

from ..config import CLASS_NAMES, NUM_CLASSES


@dataclass
class SegmentationMetrics:
    """Per-class and mean segmentation metrics."""

    iou_per_class: dict[str, float] = field(default_factory=dict)
    dice_per_class: dict[str, float] = field(default_factory=dict)
    mean_iou: float = 0.0
    mean_dice: float = 0.0


@dataclass
class EllipseMetrics:
    """Ellipse fitting accuracy metrics."""

    center_error: float = 0.0  # Mean L2 distance (pixels)
    axis_error: float = 0.0  # Mean axis length error (pixels)
    angle_error: float = 0.0  # Mean angle error (degrees)


@dataclass
class BenchmarkResults:
    """Complete benchmark results."""

    segmentation: SegmentationMetrics = field(default_factory=SegmentationMetrics)
    ellipse_pupil: EllipseMetrics = field(default_factory=EllipseMetrics)
    ellipse_iris: EllipseMetrics = field(default_factory=EllipseMetrics)
    pupil_iris_ratio_mae: float = 0.0
    inference_time_ms: float = 0.0
    model_size_mb: float = 0.0
    fps: float = 0.0

    def to_dict(self) -> dict:
        """Convert to flat dictionary for logging."""
        return {
            "mean_iou": self.segmentation.mean_iou,
            "mean_dice": self.segmentation.mean_dice,
            "iou_pupil": self.segmentation.iou_per_class.get("pupil", 0),
            "iou_iris": self.segmentation.iou_per_class.get("iris", 0),
            "dice_pupil": self.segmentation.dice_per_class.get("pupil", 0),
            "dice_iris": self.segmentation.dice_per_class.get("iris", 0),
            "pupil_center_error": self.ellipse_pupil.center_error,
            "pupil_axis_error": self.ellipse_pupil.axis_error,
            "iris_center_error": self.ellipse_iris.center_error,
            "iris_axis_error": self.ellipse_iris.axis_error,
            "pupil_iris_ratio_mae": self.pupil_iris_ratio_mae,
            "inference_time_ms": self.inference_time_ms,
            "fps": self.fps,
            "model_size_mb": self.model_size_mb,
        }


def compute_iou(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int = NUM_CLASSES,
    class_names: list[str] = CLASS_NAMES,
    eps: float = 1e-6,
) -> dict[str, float]:
    """
    Compute per-class IoU (Intersection over Union).

    Args:
        pred: Predicted class indices (B, H, W) or (H, W)
        target: Ground truth class indices (B, H, W) or (H, W)
        num_classes: Number of classes
        class_names: Names for each class
        eps: Small epsilon to avoid division by zero

    Returns:
        Dictionary mapping class names to IoU values
    """
    pred = pred.flatten()
    target = target.flatten()

    iou_dict = {}
    for cls_id in range(num_classes):
        pred_mask = pred == cls_id
        target_mask = target == cls_id

        intersection = (pred_mask & target_mask).sum().float()
        union = (pred_mask | target_mask).sum().float()

        iou = (intersection + eps) / (union + eps)
        iou_dict[class_names[cls_id]] = iou.item()

    return iou_dict


def compute_dice(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int = NUM_CLASSES,
    class_names: list[str] = CLASS_NAMES,
    eps: float = 1e-6,
) -> dict[str, float]:
    """
    Compute per-class Dice coefficient.

    Args:
        pred: Predicted class indices (B, H, W) or (H, W)
        target: Ground truth class indices (B, H, W) or (H, W)
        num_classes: Number of classes
        class_names: Names for each class
        eps: Small epsilon to avoid division by zero

    Returns:
        Dictionary mapping class names to Dice values
    """
    pred = pred.flatten()
    target = target.flatten()

    dice_dict = {}
    for cls_id in range(num_classes):
        pred_mask = pred == cls_id
        target_mask = target == cls_id

        intersection = (pred_mask & target_mask).sum().float()
        total = pred_mask.sum().float() + target_mask.sum().float()

        dice = (2 * intersection + eps) / (total + eps)
        dice_dict[class_names[cls_id]] = dice.item()

    return dice_dict


def measure_inference_time(
    model: torch.nn.Module,
    input_size: tuple[int, int, int, int],
    device: str,
    num_warmup: int = 10,
    num_runs: int = 100,
) -> float:
    """
    Measure average inference time in milliseconds.

    Args:
        model: Model to benchmark
        input_size: Input tensor shape (B, C, H, W)
        device: Device to run on
        num_warmup: Number of warmup iterations
        num_runs: Number of timed iterations

    Returns:
        Average inference time in milliseconds
    """
    model.eval()
    dummy_input = torch.randn(input_size, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy_input)

    # Synchronize if using CUDA
    if device == "cuda":
        torch.cuda.synchronize()

    # Timed runs
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = model(dummy_input)
            if device == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms

    return float(np.mean(times))


def get_model_size_mb(model: torch.nn.Module) -> float:
    """
    Get model size in megabytes.

    Args:
        model: PyTorch model

    Returns:
        Model size in MB
    """
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 * 1024)


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    """
    Count total and trainable parameters.

    Returns:
        Tuple of (total_params, trainable_params)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
