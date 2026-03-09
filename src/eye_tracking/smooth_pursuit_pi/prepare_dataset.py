#!/usr/bin/env python3
"""
Prepare training dataset from collected Pi sessions.

Decodes video frames from recorded sessions and pairs them with
ground-truth angles from the dot position log. Outputs a dataset
directory of individual frames + a labels CSV ready for training.

Usage:
    python prepare_dataset.py --sessions_dir ~/smooth_pursuit_data/sessions
    python prepare_dataset.py --sessions_dir ./sessions --output_dir ./dataset --distance 40
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def load_dot_log(session_dir: Path) -> list[dict]:
    """Load the dot position CSV log from a session."""
    log_path = session_dir / "dot_positions.csv"
    if not log_path.exists():
        print(f"  WARNING: No dot_positions.csv in {session_dir.name}")
        return []

    rows = []
    with open(log_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "frame_number": int(row["frame_number"]),
                "timestamp": float(row["timestamp"]),
                "elapsed_sec": float(row["elapsed_sec"]),
                "dot_x_px": float(row["dot_x_px"]),
                "dot_y_px": float(row["dot_y_px"]),
                "theta_h_deg": float(row["theta_h_deg"]),
                "theta_v_deg": float(row["theta_v_deg"]),
            })
    return rows


def extract_frames_from_video(video_path: Path, target_timestamps: list[float],
                              tolerance_sec: float = 0.01) -> dict[int, np.ndarray]:
    """
    Extract frames from an encoded video file at specific timestamps.

    Args:
        video_path: Path to .mp4 video
        target_timestamps: List of elapsed-time timestamps to extract
        tolerance_sec: Matching tolerance in seconds

    Returns:
        Dict mapping dot-log frame index -> image array
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  ERROR: Cannot open video {video_path}")
        return {}

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / video_fps if video_fps > 0 else 0
    print(f"  Video: {video_path.name} ({total_frames} frames, {video_fps:.1f}fps, {video_duration:.1f}s)")

    sorted_targets = sorted(enumerate(target_timestamps), key=lambda x: x[1])

    extracted = {}
    video_frame_idx = 0
    target_idx = 0

    while cap.isOpened() and target_idx < len(sorted_targets):
        ret, frame = cap.read()
        if not ret:
            break

        video_time = video_frame_idx / video_fps

        while target_idx < len(sorted_targets):
            dot_log_idx, target_time = sorted_targets[target_idx]
            if abs(video_time - target_time) <= tolerance_sec:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                extracted[dot_log_idx] = frame_rgb
                target_idx += 1
            elif video_time > target_time + tolerance_sec:
                target_idx += 1
            else:
                break

        video_frame_idx += 1

    cap.release()
    return extracted


def process_session(session_dir: Path, output_dir: Path, resize: tuple = (224, 224),
                    subsample: int = 1) -> list[dict]:
    """
    Process a single session: decode video and pair with ground-truth angles.

    Args:
        session_dir: Path to session directory
        output_dir: Path to output dataset directory
        resize: Target frame size for training
        subsample: Take every Nth frame (1 = all frames)

    Returns:
        List of label records for this session
    """
    print(f"\nProcessing: {session_dir.name}")

    metadata_path = session_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
        print(f"  Distance: {metadata.get('total_distance_cm')}cm, "
              f"Frames: {metadata.get('frame_count')}")

    dot_log = load_dot_log(session_dir)
    if not dot_log:
        return []

    if subsample > 1:
        dot_log = dot_log[::subsample]

    video_path = session_dir / "left_eye.mp4"
    if not video_path.exists():
        video_candidates = list(session_dir.glob("*.mp4"))
        if video_candidates:
            video_path = video_candidates[0]
        else:
            print(f"  ERROR: No video file found in {session_dir.name}")
            return []

    elapsed_times = [row["elapsed_sec"] for row in dot_log]
    extracted = extract_frames_from_video(video_path, elapsed_times)

    print(f"  Matched {len(extracted)}/{len(dot_log)} frames from video")

    session_output = output_dir / "frames" / session_dir.name
    session_output.mkdir(parents=True, exist_ok=True)

    labels = []
    for idx, row in enumerate(dot_log):
        if idx not in extracted:
            continue

        frame = extracted[idx]
        resized = cv2.resize(frame, resize)

        frame_filename = f"frame_{row['frame_number']:06d}.jpg"
        frame_path = session_output / frame_filename
        cv2.imwrite(str(frame_path), cv2.cvtColor(resized, cv2.COLOR_RGB2BGR))

        labels.append({
            "session": session_dir.name,
            "frame_file": str(frame_path.relative_to(output_dir)),
            "theta_h": row["theta_h_deg"],
            "theta_v": row["theta_v_deg"],
            "dot_x": row["dot_x_px"],
            "dot_y": row["dot_y_px"],
            "elapsed_sec": row["elapsed_sec"],
        })

    print(f"  Saved {len(labels)} training samples")
    return labels


def main():
    parser = argparse.ArgumentParser(description="Prepare training dataset from Pi sessions")
    parser.add_argument("--sessions_dir", type=str, required=True,
                        help="Directory containing session folders")
    parser.add_argument("--output_dir", type=str, default="dataset",
                        help="Output dataset directory")
    parser.add_argument("--distance", type=float, default=None,
                        help="Only include sessions at this distance (cm)")
    parser.add_argument("--resize", type=int, default=224,
                        help="Resize frames to this square size")
    parser.add_argument("--subsample", type=int, default=1,
                        help="Take every Nth frame (1 = all, 4 = every 4th)")
    args = parser.parse_args()

    sessions_dir = Path(args.sessions_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    session_dirs = sorted([d for d in sessions_dir.iterdir() if d.is_dir()])
    print(f"Found {len(session_dirs)} session(s) in {sessions_dir}")

    if args.distance is not None:
        filtered = []
        for sd in session_dirs:
            meta_path = sd / "metadata.json"
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                if meta.get("total_distance_cm") == args.distance:
                    filtered.append(sd)
        session_dirs = filtered
        print(f"Filtered to {len(session_dirs)} session(s) at {args.distance}cm")

    all_labels = []
    for session_dir in session_dirs:
        labels = process_session(session_dir, output_dir,
                                 resize=(args.resize, args.resize),
                                 subsample=args.subsample)
        all_labels.extend(labels)

    labels_path = output_dir / "labels.csv"
    with open(labels_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "session", "frame_file", "theta_h", "theta_v",
            "dot_x", "dot_y", "elapsed_sec",
        ])
        writer.writeheader()
        writer.writerows(all_labels)

    print(f"\n{'=' * 50}")
    print(f"  DATASET PREPARATION COMPLETE")
    print(f"{'=' * 50}")
    print(f"  Total samples: {len(all_labels)}")
    print(f"  Sessions:      {len(session_dirs)}")
    print(f"  Labels file:   {labels_path}")
    print(f"  Frames dir:    {output_dir / 'frames'}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
