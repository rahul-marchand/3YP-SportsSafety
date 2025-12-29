"""Evaluation metrics and utilities for pupil segmentation."""

from .ellipse_fitting import EllipseParams, compute_pupil_iris_ratio, fit_ellipse_to_mask
from .metrics import BenchmarkResults, compute_dice, compute_iou, get_model_size_mb

__all__ = [
    "EllipseParams",
    "fit_ellipse_to_mask",
    "compute_pupil_iris_ratio",
    "compute_iou",
    "compute_dice",
    "get_model_size_mb",
    "BenchmarkResults",
]
