"""
Stimulus display manager using Pygame.
Renders a moving dot for smooth pursuit testing on the headset's internal display.
"""

import time
import math

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

from config import DisplayConfig, StimulusConfig


class DotPosition:
    """Current dot position with timestamp."""
    __slots__ = ("x", "y", "timestamp")

    def __init__(self, x: float, y: float, timestamp: float):
        self.x = x
        self.y = y
        self.timestamp = timestamp


class DisplayManager:
    """
    Manages the stimulus display for smooth pursuit testing.
    Renders a moving dot on a fullscreen pygame surface.
    """

    def __init__(self, display_cfg: DisplayConfig, stimulus_cfg: StimulusConfig):
        self.display_cfg = display_cfg
        self.stimulus_cfg = stimulus_cfg
        self._screen = None
        self._clock = None
        self._running = False
        self._start_time = 0.0

    def initialize(self):
        """Initialize the pygame display."""
        if not PYGAME_AVAILABLE:
            print("WARNING: pygame not available. Using mock display.")
            return

        pygame.init()
        if self.display_cfg.fullscreen:
            self._screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
            info = pygame.display.Info()
            self.display_cfg.width_px = info.current_w
            self.display_cfg.height_px = info.current_h
        else:
            self._screen = pygame.display.set_mode(
                (self.display_cfg.width_px, self.display_cfg.height_px)
            )
        self._clock = pygame.time.Clock()
        pygame.mouse.set_visible(False)
        print(f"Display initialized: {self.display_cfg.width_px}x{self.display_cfg.height_px}")

    def get_dot_position(self, elapsed_time: float) -> DotPosition:
        """
        Calculate dot position at a given time into the stimulus.

        Args:
            elapsed_time: Seconds since stimulus started.

        Returns:
            DotPosition with pixel coordinates and timestamp.
        """
        cx = self.display_cfg.width_px / 2
        cy = self.display_cfg.height_px / 2

        px_per_cm_x = self.display_cfg.width_px / self.display_cfg.width_cm
        amplitude_cm = self.stimulus_cfg.amplitude_deg  # Approximate: 1 deg ~ 1 cm at typical distances

        if self.stimulus_cfg.pattern == "sinusoidal":
            period = (2 * self.stimulus_cfg.amplitude_deg) / self.stimulus_cfg.speed_deg_per_sec
            phase = (2 * math.pi * elapsed_time) / period

            if self.stimulus_cfg.direction == "horizontal":
                offset_cm = amplitude_cm * math.sin(phase)
                x = cx + offset_cm * px_per_cm_x
                y = cy
            elif self.stimulus_cfg.direction == "vertical":
                x = cx
                offset_cm = amplitude_cm * math.sin(phase)
                y = cy + offset_cm * (self.display_cfg.height_px / self.display_cfg.height_cm)
            else:
                x, y = cx, cy

        elif self.stimulus_cfg.pattern == "linear":
            half_traverse_time = self.stimulus_cfg.amplitude_deg / self.stimulus_cfg.speed_deg_per_sec
            full_cycle = 2 * half_traverse_time
            t_in_cycle = elapsed_time % full_cycle

            if t_in_cycle < half_traverse_time:
                frac = t_in_cycle / half_traverse_time
                offset_cm = -amplitude_cm + 2 * amplitude_cm * frac
            else:
                frac = (t_in_cycle - half_traverse_time) / half_traverse_time
                offset_cm = amplitude_cm - 2 * amplitude_cm * frac

            if self.stimulus_cfg.direction == "horizontal":
                x = cx + offset_cm * px_per_cm_x
                y = cy
            else:
                x = cx
                y = cy + offset_cm * (self.display_cfg.height_px / self.display_cfg.height_cm)
        else:
            x, y = cx, cy

        return DotPosition(x=x, y=y, timestamp=time.time())

    def render_dot(self, dot_pos: DotPosition):
        """Render the dot at the given position."""
        if not PYGAME_AVAILABLE or self._screen is None:
            return

        self._screen.fill(self.display_cfg.bg_color)
        pygame.draw.circle(
            self._screen,
            self.display_cfg.dot_color,
            (int(dot_pos.x), int(dot_pos.y)),
            self.display_cfg.dot_radius_px,
        )
        pygame.display.flip()

    def render_text(self, text: str, font_size: int = 36, color: tuple = (255, 255, 255)):
        """Render centered text on screen."""
        if not PYGAME_AVAILABLE or self._screen is None:
            print(f"Display: {text}")
            return

        self._screen.fill(self.display_cfg.bg_color)
        font = pygame.font.Font(None, font_size)
        lines = text.split("\n")
        total_height = len(lines) * (font_size + 5)
        y_start = (self.display_cfg.height_px - total_height) // 2

        for i, line in enumerate(lines):
            surface = font.render(line, True, color)
            rect = surface.get_rect(center=(self.display_cfg.width_px // 2,
                                            y_start + i * (font_size + 5)))
            self._screen.blit(surface, rect)
        pygame.display.flip()

    def tick(self):
        """Advance one frame and maintain target FPS."""
        if self._clock:
            self._clock.tick(self.display_cfg.fps)

    def check_quit(self) -> bool:
        """Check for quit events (ESC key or window close)."""
        if not PYGAME_AVAILABLE:
            return False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return True
        return False

    def close(self):
        """Shut down the display."""
        if PYGAME_AVAILABLE:
            pygame.quit()

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, *args):
        self.close()
