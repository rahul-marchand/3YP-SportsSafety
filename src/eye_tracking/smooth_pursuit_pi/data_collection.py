#!/usr/bin/env python3
"""
Data collection application for smooth pursuit eye tracking on Raspberry Pi 5.

Displays a moving dot stimulus while recording binocular eye video.
Pairs dot positions with frame timestamps and saves everything locally.
Data is later uploaded for model training.

Usage:
    python data_collection.py                        # Interactive mode
    python data_collection.py --distance 40          # Set distance directly
    python data_collection.py --duration 60          # 60 second session
"""

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from config import SystemConfig
from angle_calculator import AngleCalculator
from camera_manager import CameraManager
from display_manager import DisplayManager
from mcu_interface import MCUInterface


def create_session_dir(config: SystemConfig) -> Path:
    """Create a timestamped session directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dist = int(config.geometry.total_distance_cm)
    session_name = f"session_{timestamp}_{dist}cm"
    session_dir = config.storage.sessions_dir / session_name
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


def save_session_metadata(session_dir: Path, config: SystemConfig, frame_count: int,
                          duration: float):
    """Save session metadata as JSON."""
    metadata = {
        "session_id": session_dir.name,
        "created_at": datetime.now().isoformat(),
        "total_distance_cm": config.geometry.total_distance_cm,
        "display_width_px": config.display.width_px,
        "display_height_px": config.display.height_px,
        "display_width_cm": config.display.width_cm,
        "display_height_cm": config.display.height_cm,
        "camera_fps": config.camera.fps,
        "camera_resolution": f"{config.camera.width}x{config.camera.height}",
        "stimulus_pattern": config.stimulus.pattern,
        "stimulus_direction": config.stimulus.direction,
        "stimulus_speed_deg_s": config.stimulus.speed_deg_per_sec,
        "stimulus_amplitude_deg": config.stimulus.amplitude_deg,
        "stimulus_duration_sec": config.stimulus.duration_sec,
        "actual_duration_sec": duration,
        "frame_count": frame_count,
    }
    with open(session_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


def run_data_collection(config: SystemConfig):
    """Main data collection loop."""
    print("\n" + "=" * 50)
    print("  SMOOTH PURSUIT DATA COLLECTION")
    print("=" * 50)
    print(f"  Distance:  {config.geometry.total_distance_cm} cm")
    print(f"  Duration:  {config.stimulus.duration_sec} sec")
    print(f"  Pattern:   {config.stimulus.pattern} ({config.stimulus.direction})")
    print(f"  Camera:    {config.camera.width}x{config.camera.height} @ {config.camera.fps}fps")
    print("=" * 50)

    session_dir = create_session_dir(config)
    print(f"\nSession directory: {session_dir}")

    angle_calc = AngleCalculator(config.display, config.geometry)
    max_h, max_v = angle_calc.max_angle()
    print(f"Max angles: ±{max_h:.1f}° horizontal, ±{max_v:.1f}° vertical")

    camera = CameraManager(config.camera)
    display = DisplayManager(config.display, config.stimulus)
    mcu = MCUInterface(config.mcu)

    dot_log_path = session_dir / "dot_positions.csv"

    try:
        camera.initialize(["left", "right"])
        display.initialize()
        mcu.connect()

        display.render_text("SMOOTH PURSUIT DATA COLLECTION\n\nPress any key to start\nESC to cancel")

        waiting = True
        while waiting:
            if display.check_quit():
                print("Cancelled by user.")
                return
            try:
                import pygame
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            print("Cancelled by user.")
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
        camera.start_recording(session_dir)
        mcu.start_test()

        print("\nRecording started...")

        frame_count = 0
        start_time = time.time()

        with open(dot_log_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                "frame_number", "timestamp", "elapsed_sec",
                "dot_x_px", "dot_y_px", "theta_h_deg", "theta_v_deg",
            ])

            while True:
                elapsed = time.time() - start_time

                if elapsed >= config.stimulus.duration_sec:
                    break
                if display.check_quit():
                    print("\nStopped by user (ESC)")
                    break

                dot_pos = display.get_dot_position(elapsed)
                display.render_dot(dot_pos)

                theta_h, theta_v = angle_calc.calculate(dot_pos.x, dot_pos.y)

                writer.writerow([
                    frame_count, dot_pos.timestamp, f"{elapsed:.6f}",
                    f"{dot_pos.x:.1f}", f"{dot_pos.y:.1f}",
                    f"{theta_h:.4f}", f"{theta_v:.4f}",
                ])

                frame_count += 1

                if frame_count % 300 == 0:
                    print(f"  Frames: {frame_count}, Elapsed: {elapsed:.1f}s")

                display.tick()

        actual_duration = time.time() - start_time

        mcu.stop_test()
        camera.stop_recording()
        camera.stop()

        save_session_metadata(session_dir, config, frame_count, actual_duration)

        avg_fps = frame_count / actual_duration if actual_duration > 0 else 0

        print("\n" + "=" * 50)
        print("  DATA COLLECTION COMPLETE")
        print("=" * 50)
        print(f"  Session:    {session_dir.name}")
        print(f"  Frames:     {frame_count}")
        print(f"  Duration:   {actual_duration:.1f}s")
        print(f"  Avg FPS:    {avg_fps:.1f}")
        print(f"  Dot log:    {dot_log_path.name}")
        print(f"  Video:      left_eye.mp4, right_eye.mp4")
        print("=" * 50)

        display.render_text(
            f"COLLECTION COMPLETE\n\n"
            f"Frames: {frame_count}\n"
            f"Duration: {actual_duration:.1f}s\n"
            f"Avg FPS: {avg_fps:.1f}\n\n"
            f"Session saved to:\n{session_dir.name}"
        )
        time.sleep(3)

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        mcu.close()
        camera.close()
        display.close()


def main():
    parser = argparse.ArgumentParser(description="Smooth pursuit data collection on Raspberry Pi")
    parser.add_argument("--distance", type=float, default=None,
                        help="Total distance eye-to-virtual-image in cm")
    parser.add_argument("--duration", type=float, default=None,
                        help="Stimulus duration in seconds")
    parser.add_argument("--speed", type=float, default=None,
                        help="Dot speed in degrees/second")
    parser.add_argument("--pattern", choices=["sinusoidal", "linear"], default=None,
                        help="Stimulus movement pattern")
    parser.add_argument("--direction", choices=["horizontal", "vertical"], default=None,
                        help="Stimulus movement direction")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to JSON config file")
    parser.add_argument("--monocular", action="store_true",
                        help="Use single camera only (left eye)")
    args = parser.parse_args()

    if args.config:
        config = SystemConfig.load(args.config)
    else:
        config = SystemConfig()

    if args.distance is not None:
        config.geometry.total_distance_cm = args.distance
    if args.duration is not None:
        config.stimulus.duration_sec = args.duration
    if args.speed is not None:
        config.stimulus.speed_deg_per_sec = args.speed
    if args.pattern is not None:
        config.stimulus.pattern = args.pattern
    if args.direction is not None:
        config.stimulus.direction = args.direction

    if args.distance is None:
        try:
            dist = float(input("Enter total distance (eye to virtual image) in cm [40]: ").strip() or "40")
            config.geometry.total_distance_cm = dist
        except (ValueError, EOFError):
            config.geometry.total_distance_cm = 40.0

    config.geometry.screen_center_x_px = config.display.width_px // 2
    config.geometry.screen_center_y_px = config.display.height_px // 2

    run_data_collection(config)


if __name__ == "__main__":
    main()
