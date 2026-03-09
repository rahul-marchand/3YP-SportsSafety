"""
Binocular camera manager using picamera2 for dual MIPI CSI-2 cameras.
Handles simultaneous capture from left and right eye cameras on Raspberry Pi 5.
"""

import time
import threading
from pathlib import Path
from dataclasses import dataclass

try:
    from picamera2 import Picamera2
    from picamera2.encoders import H264Encoder, Quality
    from picamera2.outputs import FfmpegOutput
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False

from config import CameraConfig


@dataclass
class FrameResult:
    """A captured frame with metadata."""
    image: "np.ndarray"
    timestamp: float
    camera_id: str


class CameraManager:
    """
    Manages dual CSI cameras on Raspberry Pi 5.
    Pi 5 has two native CSI-2 ports (cam0, cam1).
    """

    def __init__(self, config: CameraConfig):
        self.config = config
        self._cameras: dict[str, "Picamera2"] = {}
        self._recording = False
        self._lock = threading.Lock()

        if not PICAMERA2_AVAILABLE:
            print("WARNING: picamera2 not available. Using mock camera for development.")

    def initialize(self, camera_ids: list[str] = None):
        """
        Initialize cameras.

        Args:
            camera_ids: List of camera identifiers. Defaults to ["left", "right"].
                        Pass ["left"] for monocular smooth pursuit.
        """
        if camera_ids is None:
            camera_ids = ["left", "right"]

        if not PICAMERA2_AVAILABLE:
            print(f"Mock: Would initialize cameras: {camera_ids}")
            return

        available = Picamera2.global_camera_info()
        print(f"Available cameras: {len(available)}")
        for i, cam_info in enumerate(available):
            print(f"  Camera {i}: {cam_info}")

        for i, cam_id in enumerate(camera_ids):
            if i >= len(available):
                print(f"WARNING: Camera index {i} not available, skipping '{cam_id}'")
                continue

            cam = Picamera2(i)
            video_config = cam.create_video_configuration(
                main={"size": (self.config.width, self.config.height),
                      "format": "RGB888"},
                controls={
                    "FrameRate": self.config.fps,
                    "ExposureTime": self.config.exposure_time_us,
                    "AnalogueGain": self.config.analogue_gain,
                    "AeEnable": False,
                    "AwbEnable": False,
                },
            )
            cam.configure(video_config)
            self._cameras[cam_id] = cam
            print(f"Initialized camera '{cam_id}' (index {i}): "
                  f"{self.config.width}x{self.config.height} @ {self.config.fps}fps")

    def start(self):
        """Start all cameras."""
        for cam_id, cam in self._cameras.items():
            cam.start()
            print(f"Camera '{cam_id}' started")

    def stop(self):
        """Stop all cameras."""
        for cam_id, cam in self._cameras.items():
            cam.stop()
            print(f"Camera '{cam_id}' stopped")

    def capture_frame(self, camera_id: str = "left") -> FrameResult | None:
        """
        Capture a single frame from the specified camera.

        Args:
            camera_id: Which camera to capture from ("left" or "right")

        Returns:
            FrameResult with image array and timestamp, or None if camera unavailable.
        """
        if not PICAMERA2_AVAILABLE:
            import numpy as np
            mock_frame = np.random.randint(0, 255,
                (self.config.height, self.config.width, 3), dtype="uint8")
            return FrameResult(
                image=mock_frame,
                timestamp=time.time(),
                camera_id=camera_id,
            )

        cam = self._cameras.get(camera_id)
        if cam is None:
            return None

        frame = cam.capture_array()
        return FrameResult(
            image=frame,
            timestamp=time.time(),
            camera_id=camera_id,
        )

    def capture_binocular(self) -> tuple[FrameResult | None, FrameResult | None]:
        """Capture from both cameras as close in time as possible."""
        left = self.capture_frame("left")
        right = self.capture_frame("right")
        return left, right

    def start_recording(self, session_dir: Path):
        """
        Start recording encoded video to files.

        Args:
            session_dir: Directory to save video files.
        """
        session_dir.mkdir(parents=True, exist_ok=True)
        self._recording = True

        if not PICAMERA2_AVAILABLE:
            print(f"Mock: Would start recording to {session_dir}")
            return

        for cam_id, cam in self._cameras.items():
            output_path = str(session_dir / f"{cam_id}_eye.mp4")
            encoder = H264Encoder(bitrate=self.config.encoding_bitrate)
            output = FfmpegOutput(output_path)
            cam.start_recording(encoder, output)
            print(f"Recording '{cam_id}' to {output_path}")

    def stop_recording(self):
        """Stop recording on all cameras."""
        if not self._recording:
            return

        self._recording = False

        if not PICAMERA2_AVAILABLE:
            print("Mock: Would stop recording")
            return

        for cam_id, cam in self._cameras.items():
            cam.stop_recording()
            print(f"Stopped recording '{cam_id}'")

    def close(self):
        """Release all camera resources."""
        self.stop_recording()
        for cam_id, cam in self._cameras.items():
            cam.close()
        self._cameras.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
