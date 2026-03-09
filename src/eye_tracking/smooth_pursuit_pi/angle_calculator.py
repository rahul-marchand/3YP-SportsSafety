"""
Gaze angle calculation from dot position and headset geometry.
For a beam splitter headset, angles are computed from screen-center displacement
and total optical distance (display-to-splitter + splitter-to-eye).
"""

import numpy as np
from config import DisplayConfig, GeometryConfig


class AngleCalculator:
    """Calculate gaze angles from stimulus dot position."""

    def __init__(self, display: DisplayConfig, geometry: GeometryConfig):
        self.display = display
        self.geometry = geometry
        self.px_to_cm_x = display.width_cm / display.width_px
        self.px_to_cm_y = display.height_cm / display.height_px

    def calculate(self, dot_x_px: float, dot_y_px: float) -> tuple[float, float]:
        """
        Calculate gaze angles from dot position on stimulus display.

        Args:
            dot_x_px: Dot x position in pixels
            dot_y_px: Dot y position in pixels

        Returns:
            (theta_h, theta_v) in degrees. Positive = right/down.
        """
        dx_cm = (dot_x_px - self.geometry.screen_center_x_px) * self.px_to_cm_x
        dy_cm = (dot_y_px - self.geometry.screen_center_y_px) * self.px_to_cm_y

        theta_h = np.degrees(np.arctan(dx_cm / self.geometry.total_distance_cm))
        theta_v = np.degrees(np.arctan(dy_cm / self.geometry.total_distance_cm))

        return float(theta_h), float(theta_v)

    def max_angle(self) -> tuple[float, float]:
        """Return maximum achievable angle in each direction."""
        half_w_cm = (self.display.width_cm / 2)
        half_h_cm = (self.display.height_cm / 2)
        max_h = np.degrees(np.arctan(half_w_cm / self.geometry.total_distance_cm))
        max_v = np.degrees(np.arctan(half_h_cm / self.geometry.total_distance_cm))
        return float(max_h), float(max_v)
