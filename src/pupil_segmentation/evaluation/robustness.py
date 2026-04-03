"""Robustness experiments for PLR parameter extraction.

Tests parameter recovery under increasing noise and blink-induced data dropout.
All experiments use synthetic PLR waveforms with known ground truth.

Usage:
    uv run python -m src.pupil_segmentation.evaluation.robustness --output_dir results/robustness
"""

import argparse
import json
from pathlib import Path

import numpy as np

from .plr import PLRParameters, extract_plr_parameters, generate_synthetic_plr

# Ground truth parameters (matching generate_synthetic_plr defaults)
GT = {
    "baseline": 6.0,
    "amplitude": 2.0,
    "latency": 0.2,
    "tau_c": 0.15,
    "tau_d": 1.5,
}


def _parameter_errors(params: PLRParameters) -> dict[str, float]:
    """Compute absolute errors against ground truth."""
    return {
        "baseline_diameter": abs(params.baseline_diameter - GT["baseline"]),
        "constriction_amplitude": abs(params.constriction_amplitude - GT["amplitude"]),
        "constriction_latency": abs(params.constriction_latency - GT["latency"]),
        "recovery_time_constant": abs(params.recovery_time_constant - GT["tau_d"]),
        "fit_r_squared": params.fit_r_squared,
    }


def _inject_blinks(
    diameters: np.ndarray,
    n_frames: int,
    rng: np.random.Generator,
    post_stimulus_start: int,
) -> np.ndarray:
    """Zero out n_frames consecutive frames at a random post-stimulus position."""
    result = diameters.copy()
    max_start = len(result) - n_frames
    if post_stimulus_start >= max_start:
        return result
    start = rng.integers(post_stimulus_start, max_start)
    result[start : start + n_frames] = 0.0
    return result


def run_noise_experiment(
    noise_levels: list[float],
    n_trials: int = 100,
    seed: int = 42,
) -> dict:
    """Test PLR recovery under increasing measurement noise."""
    rng = np.random.default_rng(seed)
    results = []

    for sigma in noise_levels:
        errors_list = []
        for _trial in range(n_trials):
            ts, diameters, onset = generate_synthetic_plr(
                noise_std=sigma,
                seed=int(rng.integers(0, 2**31)),
            )
            params = extract_plr_parameters(diameters, ts, onset, fps=30.0)
            errors_list.append(_parameter_errors(params))

        # Aggregate
        agg = {}
        for key in errors_list[0]:
            vals = [e[key] for e in errors_list]
            agg[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

        results.append({"noise_std": sigma, "n_trials": n_trials, "metrics": agg})

    return {"ground_truth": GT, "results": results}


def run_blink_experiment(
    blink_durations: list[int],
    n_trials: int = 100,
    seed: int = 42,
    fps: float = 30.0,
) -> dict:
    """Test PLR recovery with blink-induced data dropout."""
    rng = np.random.default_rng(seed)
    results = []

    for n_frames in blink_durations:
        errors_list = []
        for _trial in range(n_trials):
            ts, diameters, onset = generate_synthetic_plr(
                noise_std=0.1,
                seed=int(rng.integers(0, 2**31)),
            )
            post_start = int(onset * fps) + 1
            diameters = _inject_blinks(diameters, n_frames, rng, post_start)
            params = extract_plr_parameters(diameters, ts, onset, fps=fps)
            errors_list.append(_parameter_errors(params))

        agg = {}
        for key in errors_list[0]:
            vals = [e[key] for e in errors_list]
            agg[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

        results.append(
            {
                "blink_frames": n_frames,
                "blink_duration_ms": n_frames / fps * 1000,
                "n_trials": n_trials,
                "metrics": agg,
            }
        )

    return {"ground_truth": GT, "baseline_noise_std": 0.1, "results": results}


def run_combined_experiment(
    noise_levels: list[float],
    blink_durations: list[int],
    n_trials: int = 50,
    seed: int = 42,
    fps: float = 30.0,
) -> dict:
    """Test PLR recovery under combined noise and blinks."""
    rng = np.random.default_rng(seed)
    results = []

    for sigma in noise_levels:
        for n_frames in blink_durations:
            errors_list = []
            for _trial in range(n_trials):
                ts, diameters, onset = generate_synthetic_plr(
                    noise_std=sigma,
                    seed=int(rng.integers(0, 2**31)),
                )
                post_start = int(onset * fps) + 1
                diameters = _inject_blinks(diameters, n_frames, rng, post_start)
                params = extract_plr_parameters(diameters, ts, onset, fps=fps)
                errors_list.append(_parameter_errors(params))

            agg = {}
            for key in errors_list[0]:
                vals = [e[key] for e in errors_list]
                agg[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

            results.append(
                {
                    "noise_std": sigma,
                    "blink_frames": n_frames,
                    "n_trials": n_trials,
                    "metrics": agg,
                }
            )

    return {"ground_truth": GT, "results": results}


def main():
    parser = argparse.ArgumentParser(description="PLR extraction robustness experiments")
    parser.add_argument("--output_dir", type=Path, default=Path("results/robustness"))
    parser.add_argument("--n_trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--experiment",
        choices=["noise", "blinks", "combined", "all"],
        default="all",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    noise_levels = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
    blink_durations = [3, 5, 8, 10]

    if args.experiment in ("noise", "all"):
        print("Running noise experiment...")
        result = run_noise_experiment(noise_levels, args.n_trials, args.seed)
        path = args.output_dir / "noise.json"
        with open(path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Saved to {path}")

    if args.experiment in ("blinks", "all"):
        print("Running blink experiment...")
        result = run_blink_experiment(blink_durations, args.n_trials, args.seed)
        path = args.output_dir / "blinks.json"
        with open(path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Saved to {path}")

    if args.experiment in ("combined", "all"):
        print("Running combined experiment...")
        result = run_combined_experiment(
            noise_levels, blink_durations, args.n_trials // 2, args.seed
        )
        path = args.output_dir / "combined.json"
        with open(path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Saved to {path}")

    print("Done.")


if __name__ == "__main__":
    main()
