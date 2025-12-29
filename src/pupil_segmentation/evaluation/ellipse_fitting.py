"""Ellipse fitting from segmentation masks using OpenCV."""

from dataclasses import dataclass

import cv2
import numpy as np

from ..config import CLASS_IDS


@dataclass
class EllipseParams:
    """Ellipse parameters from cv2.fitEllipse."""

    center: tuple[float, float]  # (cx, cy)
    axes: tuple[float, float]  # (major_axis, minor_axis) as diameters
    angle: float  # rotation angle in degrees
    valid: bool = True  # False if fitting failed

    @property
    def major_axis(self) -> float:
        """Major axis diameter."""
        return self.axes[0]

    @property
    def minor_axis(self) -> float:
        """Minor axis diameter."""
        return self.axes[1]

    @property
    def mean_diameter(self) -> float:
        """Mean of major and minor axis."""
        return (self.axes[0] + self.axes[1]) / 2.0


def fit_ellipse_to_mask(
    mask: np.ndarray,
    class_id: int,
    min_contour_points: int = 5,
) -> EllipseParams:
    """
    Fit ellipse to a binary mask region.

    Args:
        mask: Segmentation mask (H, W) with class labels
        class_id: Class to fit ellipse to (2=iris, 3=pupil)
        min_contour_points: Minimum points required for ellipse fitting

    Returns:
        EllipseParams with fitted ellipse, or valid=False if fitting failed
    """
    # Create binary mask for target class
    binary = (mask == class_id).astype(np.uint8)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return EllipseParams((0.0, 0.0), (0.0, 0.0), 0.0, valid=False)

    # Get largest contour
    largest = max(contours, key=cv2.contourArea)

    if len(largest) < min_contour_points:
        return EllipseParams((0.0, 0.0), (0.0, 0.0), 0.0, valid=False)

    try:
        (cx, cy), (w, h), angle = cv2.fitEllipse(largest)
        # Ensure major >= minor
        major, minor = max(w, h), min(w, h)
        return EllipseParams((cx, cy), (major, minor), angle)
    except cv2.error:
        return EllipseParams((0.0, 0.0), (0.0, 0.0), 0.0, valid=False)


def compute_pupil_iris_ratio(
    mask: np.ndarray,
    pupil_class: int = CLASS_IDS["pupil"],
    iris_class: int = CLASS_IDS["iris"],
) -> float:
    """
    Compute pupil/iris diameter ratio from fitted ellipses.

    Uses mean diameter (average of major and minor axes) for robustness.

    Args:
        mask: Segmentation mask (H, W) with class labels
        pupil_class: Class ID for pupil (default: 3)
        iris_class: Class ID for iris (default: 2)

    Returns:
        Ratio of pupil diameter to iris diameter (0.0 if either is invalid)
    """
    pupil = fit_ellipse_to_mask(mask, pupil_class)
    iris = fit_ellipse_to_mask(mask, iris_class)

    if not pupil.valid or not iris.valid:
        return 0.0

    if iris.mean_diameter == 0:
        return 0.0

    return pupil.mean_diameter / iris.mean_diameter


def compute_ellipse_error(
    pred_params: EllipseParams,
    gt_params: EllipseParams,
) -> dict[str, float]:
    """
    Compute error between predicted and ground truth ellipse parameters.

    Args:
        pred_params: Predicted ellipse parameters
        gt_params: Ground truth ellipse parameters

    Returns:
        Dictionary with center_error, axis_error, angle_error (in pixels/degrees)
    """
    if not pred_params.valid or not gt_params.valid:
        return {
            "center_error": float("inf"),
            "axis_error": float("inf"),
            "angle_error": float("inf"),
        }

    # Center error (L2 distance in pixels)
    center_error = np.sqrt(
        (pred_params.center[0] - gt_params.center[0]) ** 2
        + (pred_params.center[1] - gt_params.center[1]) ** 2
    )

    # Axis error (mean absolute error of major and minor axes)
    axis_error = (
        abs(pred_params.major_axis - gt_params.major_axis)
        + abs(pred_params.minor_axis - gt_params.minor_axis)
    ) / 2.0

    # Angle error (handle wraparound at 180 degrees)
    angle_diff = abs(pred_params.angle - gt_params.angle) % 180
    angle_error = min(angle_diff, 180 - angle_diff)

    return {
        "center_error": center_error,
        "axis_error": axis_error,
        "angle_error": angle_error,
    }
