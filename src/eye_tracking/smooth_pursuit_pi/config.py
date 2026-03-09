"""
Shared configuration for the Raspberry Pi smooth pursuit system.
All hardware-specific parameters and defaults are defined here.
"""

from dataclasses import dataclass, field
from pathlib import Path
import json


@dataclass
class DisplayConfig:
    """Stimulus display parameters."""
    width_px: int = 640
    height_px: int = 480
    width_cm: float = 12.0
    height_cm: float = 9.0
    fps: int = 60
    dot_radius_px: int = 15
    dot_color: tuple = (255, 255, 255)
    bg_color: tuple = (0, 0, 0)
    fullscreen: bool = True


@dataclass
class CameraConfig:
    """Binocular camera parameters."""
    width: int = 640
    height: int = 480
    fps: int = 120
    format: str = "YUV420"
    encoding: str = "h264"
    encoding_bitrate: int = 5_000_000
    exposure_time_us: int = 4000
    analogue_gain: float = 4.0


@dataclass
class GeometryConfig:
    """Beam splitter / headset geometry."""
    total_distance_cm: float = 40.0
    screen_center_x_px: int = 320
    screen_center_y_px: int = 240


@dataclass
class StimulusConfig:
    """Smooth pursuit stimulus pattern."""
    speed_deg_per_sec: float = 15.0
    amplitude_deg: float = 15.0
    duration_sec: float = 30.0
    direction: str = "horizontal"
    pattern: str = "sinusoidal"


@dataclass
class StorageConfig:
    """Local storage paths on Pi."""
    base_dir: Path = field(default_factory=lambda: Path.home() / "smooth_pursuit_data")
    sessions_dir: Path = field(default_factory=lambda: Path.home() / "smooth_pursuit_data" / "sessions")

    def __post_init__(self):
        self.base_dir = Path(self.base_dir)
        self.sessions_dir = Path(self.sessions_dir)


@dataclass
class MCUConfig:
    """MCU UART communication (optional)."""
    port: str = "/dev/ttyAMA0"
    baudrate: int = 115200
    timeout: float = 1.0
    enabled: bool = False


@dataclass
class InferenceConfig:
    """Inference engine parameters."""
    model_path: str = ""
    input_size: tuple = (224, 224)
    confidence_threshold: float = 0.5
    gain_normal_range: tuple = (0.8, 1.2)


@dataclass
class SystemConfig:
    """Top-level system configuration."""
    display: DisplayConfig = field(default_factory=DisplayConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    stimulus: StimulusConfig = field(default_factory=StimulusConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    mcu: MCUConfig = field(default_factory=MCUConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    def save(self, path: str):
        """Save configuration to JSON."""
        data = {
            "display": self.display.__dict__,
            "camera": self.camera.__dict__,
            "geometry": self.geometry.__dict__,
            "stimulus": self.stimulus.__dict__,
            "storage": {
                "base_dir": str(self.storage.base_dir),
                "sessions_dir": str(self.storage.sessions_dir),
            },
            "mcu": self.mcu.__dict__,
            "inference": {
                **self.inference.__dict__,
                "input_size": list(self.inference.input_size),
                "gain_normal_range": list(self.inference.gain_normal_range),
            },
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "SystemConfig":
        """Load configuration from JSON."""
        with open(path) as f:
            data = json.load(f)

        cfg = cls()
        if "display" in data:
            d = data["display"]
            if "dot_color" in d:
                d["dot_color"] = tuple(d["dot_color"])
            if "bg_color" in d:
                d["bg_color"] = tuple(d["bg_color"])
            cfg.display = DisplayConfig(**d)
        if "camera" in data:
            cfg.camera = CameraConfig(**data["camera"])
        if "geometry" in data:
            cfg.geometry = GeometryConfig(**data["geometry"])
        if "stimulus" in data:
            cfg.stimulus = StimulusConfig(**data["stimulus"])
        if "storage" in data:
            cfg.storage = StorageConfig(
                base_dir=Path(data["storage"]["base_dir"]),
                sessions_dir=Path(data["storage"]["sessions_dir"]),
            )
        if "mcu" in data:
            cfg.mcu = MCUConfig(**data["mcu"])
        if "inference" in data:
            inf = data["inference"]
            if "input_size" in inf:
                inf["input_size"] = tuple(inf["input_size"])
            if "gain_normal_range" in inf:
                inf["gain_normal_range"] = tuple(inf["gain_normal_range"])
            cfg.inference = InferenceConfig(**inf)
        return cfg
