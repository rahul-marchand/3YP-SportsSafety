#!/usr/bin/env python3
"""
Smooth pursuit inference engine for Raspberry Pi 5.

Runs a trained gaze model in real-time during dot stimulus presentation.
Calculates smooth pursuit metrics (gain, velocity errors, lag) and
displays clinical diagnosis on the headset screen.

Usage:
    python inference_engine.py --model model_40cm.onnx --distance 40
"""

import argparse
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

from config import SystemConfig
from angle_calculator import AngleCalculator
from camera_manager import CameraManager
from display_manager import DisplayManager
from mcu_interface import MCUInterface

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass
class FrameMetrics:
    """Metrics for a single frame during inference."""
    timestamp: float
    elapsed: float
    dot_x: float
    dot_y: float
    theta_h_target: float
    theta_v_target: float
    theta_h_pred: float
    theta_v_pred: float
    error_h: float
    error_v: float
    expected_vel_h: float | None = None
    actual_vel_h: float | None = None
    vel_error_h: float | None = None


@dataclass
class SessionDiagnosis:
    """Final diagnosis for a smooth pursuit session."""
    num_frames: int = 0
    duration_sec: float = 0.0
    mean_error_h: float = 0.0
    mean_error_v: float = 0.0
    rms_error: float = 0.0
    smooth_pursuit_gain_h: float = 0.0
    mean_vel_error_h: float = 0.0
    diagnosis: str = ""
    detail: str = ""


class GazeModel:
    """ONNX Runtime gaze estimation model."""

    def __init__(self, model_path: str, input_size: tuple = (224, 224)):
        self.input_size = input_size

        if not ONNX_AVAILABLE:
            print("WARNING: onnxruntime not installed. Using mock predictions.")
            self._session = None
            return

        self._session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self._input_name = self._session.get_inputs()[0].name
        print(f"Model loaded: {model_path}")
        print(f"  Input: {self._session.get_inputs()[0].shape}")
        print(f"  Output: {self._session.get_outputs()[0].shape}")

    def predict(self, frame: np.ndarray) -> tuple[float, float]:
        """
        Predict gaze angles from a camera frame.

        Args:
            frame: RGB image array (H, W, 3)

        Returns:
            (theta_h, theta_v) predicted angles in degrees
        """
        preprocessed = self._preprocess(frame)

        if self._session is None:
            return (np.random.randn() * 5, np.random.randn() * 2)

        outputs = self._session.run(None, {self._input_name: preprocessed})
        prediction = outputs[0][0]
        return float(prediction[0]), float(prediction[1])

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Resize, normalize, and reshape frame for model input."""
        resized = cv2.resize(frame, self.input_size)
        normalized = (resized.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        # HWC -> NCHW
        transposed = normalized.transpose(2, 0, 1)
        return transposed[np.newaxis, ...]


def calculate_diagnosis(metrics: list[FrameMetrics], config: SystemConfig) -> SessionDiagnosis:
    """Calculate smooth pursuit diagnosis from collected frame metrics."""
    if len(metrics) < 10:
        return SessionDiagnosis(diagnosis="Insufficient data", detail="Need at least 10 frames")

    diag = SessionDiagnosis()
    diag.num_frames = len(metrics)
    diag.duration_sec = metrics[-1].elapsed - metrics[0].elapsed

    errors_h = [m.error_h for m in metrics]
    errors_v = [m.error_v for m in metrics]
    diag.mean_error_h = float(np.mean(errors_h))
    diag.mean_error_v = float(np.mean(errors_v))
    diag.rms_error = float(np.sqrt(diag.mean_error_h**2 + diag.mean_error_v**2))

    vel_frames = [m for m in metrics if m.expected_vel_h is not None]
    if len(vel_frames) > 5:
        expected_vels = [m.expected_vel_h for m in vel_frames]
        actual_vels = [m.actual_vel_h for m in vel_frames]
        vel_errors = [m.vel_error_h for m in vel_frames]

        diag.mean_vel_error_h = float(np.mean(vel_errors))

        mean_expected = np.mean(expected_vels)
        mean_actual = np.mean(actual_vels)
        if abs(mean_expected) > 0.1:
            diag.smooth_pursuit_gain_h = float(mean_actual / mean_expected)

    gain = diag.smooth_pursuit_gain_h
    gain_low, gain_high = config.inference.gain_normal_range

    if gain_low <= gain <= gain_high:
        diag.diagnosis = "NORMAL"
        diag.detail = f"Smooth pursuit gain ({gain:.2f}) is within normal range ({gain_low}-{gain_high})"
    elif gain < gain_low:
        diag.diagnosis = "REDUCED PURSUIT"
        diag.detail = f"Gain ({gain:.2f}) below normal: eyes not keeping up with target"
    else:
        diag.diagnosis = "OVERSHOOT"
        diag.detail = f"Gain ({gain:.2f}) above normal: eyes overshooting target"

    return diag


def run_inference(config: SystemConfig):
    """Main inference loop."""
    print("\n" + "=" * 50)
    print("  SMOOTH PURSUIT ASSESSMENT")
    print("=" * 50)
    print(f"  Model:     {config.inference.model_path}")
    print(f"  Distance:  {config.geometry.total_distance_cm} cm")
    print(f"  Duration:  {config.stimulus.duration_sec} sec")
    print("=" * 50)

    model = GazeModel(config.inference.model_path, config.inference.input_size)
    angle_calc = AngleCalculator(config.display, config.geometry)
    camera = CameraManager(config.camera)
    display = DisplayManager(config.display, config.stimulus)
    mcu = MCUInterface(config.mcu)

    all_metrics: list[FrameMetrics] = []
    prev_metric: FrameMetrics | None = None

    try:
        camera.initialize(["left"])
        display.initialize()
        mcu.connect()

        display.render_text(
            "SMOOTH PURSUIT ASSESSMENT\n\n"
            "Follow the moving dot with your eyes\n\n"
            "Press any key to start\nESC to cancel"
        )

        waiting = True
        while waiting:
            if display.check_quit():
                print("Cancelled.")
                return
            try:
                import pygame
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            return
                        waiting = False
            except ImportError:
                waiting = False

        display.render_text("Starting in 3...")
        time.sleep(1)
        display.render_text("Starting in 2...")
        time.sleep(1)
        display.render_text("Starting in 1...")
        time.sleep(1)

        camera.start()
        mcu.start_test()

        print("\nInference running...")
        start_time = time.time()
        frame_count = 0

        while True:
            elapsed = time.time() - start_time

            if elapsed >= config.stimulus.duration_sec:
                break
            if display.check_quit():
                print("\nStopped by user")
                break

            dot_pos = display.get_dot_position(elapsed)
            display.render_dot(dot_pos)

            frame_result = camera.capture_frame("left")
            if frame_result is None:
                display.tick()
                continue

            theta_h_pred, theta_v_pred = model.predict(frame_result.image)
            theta_h_target, theta_v_target = angle_calc.calculate(dot_pos.x, dot_pos.y)

            metric = FrameMetrics(
                timestamp=frame_result.timestamp,
                elapsed=elapsed,
                dot_x=dot_pos.x,
                dot_y=dot_pos.y,
                theta_h_target=theta_h_target,
                theta_v_target=theta_v_target,
                theta_h_pred=theta_h_pred,
                theta_v_pred=theta_v_pred,
                error_h=abs(theta_h_pred - theta_h_target),
                error_v=abs(theta_v_pred - theta_v_target),
            )

            if prev_metric is not None:
                dt = metric.timestamp - prev_metric.timestamp
                if dt > 0:
                    metric.expected_vel_h = (theta_h_target - prev_metric.theta_h_target) / dt
                    metric.actual_vel_h = (theta_h_pred - prev_metric.theta_h_pred) / dt
                    metric.vel_error_h = abs(metric.expected_vel_h - metric.actual_vel_h)

            all_metrics.append(metric)
            prev_metric = metric
            frame_count += 1

            if frame_count % 120 == 0:
                print(f"  Frame {frame_count}: error_h={metric.error_h:.2f}°, "
                      f"elapsed={elapsed:.1f}s")

            display.tick()

        mcu.stop_test()
        camera.stop()

        diagnosis = calculate_diagnosis(all_metrics, config)

        print("\n" + "=" * 50)
        print("  ASSESSMENT RESULTS")
        print("=" * 50)
        print(f"  Frames:     {diagnosis.num_frames}")
        print(f"  Duration:   {diagnosis.duration_sec:.1f}s")
        print(f"\n  ANGLE ERRORS:")
        print(f"    Horizontal: {diagnosis.mean_error_h:.2f}°")
        print(f"    Vertical:   {diagnosis.mean_error_v:.2f}°")
        print(f"    RMS:        {diagnosis.rms_error:.2f}°")
        print(f"\n  VELOCITY ANALYSIS:")
        print(f"    Mean vel error: {diagnosis.mean_vel_error_h:.2f}°/s")
        print(f"    Pursuit gain:   {diagnosis.smooth_pursuit_gain_h:.3f}")
        print(f"\n  DIAGNOSIS: {diagnosis.diagnosis}")
        print(f"    {diagnosis.detail}")
        print("=" * 50)

        result_color = (0, 255, 0) if diagnosis.diagnosis == "NORMAL" else (255, 100, 100)
        display.render_text(
            f"ASSESSMENT COMPLETE\n\n"
            f"Pursuit Gain: {diagnosis.smooth_pursuit_gain_h:.2f}\n"
            f"RMS Error: {diagnosis.rms_error:.2f} deg\n\n"
            f"Diagnosis: {diagnosis.diagnosis}\n"
            f"{diagnosis.detail}",
            color=result_color,
        )
        time.sleep(8)

    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        mcu.close()
        camera.close()
        display.close()


def main():
    parser = argparse.ArgumentParser(description="Smooth pursuit inference on Raspberry Pi")
    parser.add_argument("--model", type=str, required=True, help="Path to ONNX model file")
    parser.add_argument("--distance", type=float, required=True,
                        help="Total distance eye-to-virtual-image in cm")
    parser.add_argument("--duration", type=float, default=30.0,
                        help="Test duration in seconds")
    parser.add_argument("--speed", type=float, default=15.0,
                        help="Dot speed in degrees/second")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to JSON config file")
    args = parser.parse_args()

    if args.config:
        config = SystemConfig.load(args.config)
    else:
        config = SystemConfig()

    config.inference.model_path = args.model
    config.geometry.total_distance_cm = args.distance
    config.stimulus.duration_sec = args.duration
    config.stimulus.speed_deg_per_sec = args.speed
    config.geometry.screen_center_x_px = config.display.width_px // 2
    config.geometry.screen_center_y_px = config.display.height_px // 2

    run_inference(config)


if __name__ == "__main__":
    main()
