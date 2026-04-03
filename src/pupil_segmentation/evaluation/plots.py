"""Generate report figures for the pupil segmentation project.

Outputs PDF files suitable for LaTeX inclusion.

Usage:
    uv run python -m src.pupil_segmentation.evaluation.plots --help
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .plr import _plr_model, extract_plr_parameters, generate_synthetic_plr, smooth_plr

# Consistent style
plt.rcParams.update(
    {
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
    }
)


def plot_training_curves(csv_dir: Path, output_path: Path) -> None:
    """Plot training curves from metrics CSVs.

    Expects csv_dir to contain subdirectories (one per experiment),
    each with a metrics.csv file.
    """
    experiments = {}
    for csv_path in sorted(csv_dir.rglob("metrics.csv")):
        name = csv_path.parent.name
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        experiments[name] = {
            "epoch": [int(r["epoch"]) for r in rows],
            "train_loss": [float(r["train_loss"]) for r in rows],
            "val_loss": [float(r["val_loss"]) for r in rows],
            "val_iou": [float(r["val_iou"]) for r in rows],
            "val_dice": [float(r["val_dice"]) for r in rows],
        }

    if not experiments:
        print(f"No metrics.csv files found in {csv_dir}")
        return

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    metrics = [
        ("train_loss", "Train Loss"),
        ("val_loss", "Val Loss"),
        ("val_iou", "Val IoU"),
        ("val_dice", "Val Dice"),
    ]

    for ax, (key, title) in zip(axes.flat, metrics, strict=False):
        for name, data in experiments.items():
            ax.plot(data["epoch"], data[key], label=name, linewidth=1.2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Training curves saved to {output_path}")


def plot_plr_fitted_curve(output_path: Path, seed: int = 42) -> None:
    """Generate the PLR explainer figure: raw data + fitted curve + annotations."""
    ts, diameters, onset = generate_synthetic_plr(
        noise_std=0.2,
        seed=seed,
        fps=60,
        duration=4.0,
    )
    params = extract_plr_parameters(diameters, ts, onset, fps=60.0)
    smoothed = smooth_plr(diameters)

    # Fitted curve
    t_fit = ts[ts >= onset]
    d_fit = _plr_model(
        t_fit,
        params.baseline_diameter,
        params.constriction_amplitude,
        onset + params.constriction_latency,
        0.15,  # tau_c from defaults
        params.recovery_time_constant,
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(ts, diameters, s=3, alpha=0.3, color="grey", label="Raw data", zorder=1)
    ax.plot(ts, smoothed, color="steelblue", linewidth=1, label="Smoothed", alpha=0.7, zorder=2)
    ax.plot(t_fit, d_fit, color="crimson", linewidth=2, label="Fitted model", zorder=3)

    # Stimulus onset
    ax.axvline(onset, color="orange", linestyle="--", linewidth=1, label="Stimulus onset")

    # Annotations
    d0 = params.baseline_diameter
    A = params.constriction_amplitude
    t_min = t_fit[np.argmin(d_fit)]

    # Baseline
    ax.annotate(
        "",
        xy=(onset - 0.3, d0),
        xytext=(onset + 0.5, d0),
        arrowprops={"arrowstyle": "<->", "color": "black", "lw": 1},
    )
    ax.text(onset + 0.6, d0 + 0.05, f"Baseline = {d0:.1f}", fontsize=8)

    # Amplitude
    ax.annotate(
        "",
        xy=(t_min, d0),
        xytext=(t_min, d0 - A),
        arrowprops={"arrowstyle": "<->", "color": "darkgreen", "lw": 1.5},
    )
    ax.text(
        t_min + 0.1,
        d0 - A / 2,
        f"A = {A:.2f}\n({params.constriction_amplitude_pct:.0f}%)",
        fontsize=8,
        color="darkgreen",
    )

    # Latency
    ax.annotate(
        "",
        xy=(onset, d0 - 0.15),
        xytext=(onset + params.constriction_latency, d0 - 0.15),
        arrowprops={"arrowstyle": "<->", "color": "purple", "lw": 1},
    )
    ax.text(
        onset + 0.02,
        d0 - 0.3,
        f"Latency = {params.constriction_latency * 1000:.0f} ms",
        fontsize=8,
        color="purple",
    )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Pupil diameter (mm)")
    ax.set_title(f"PLR Parametric Curve Fitting (R² = {params.fit_r_squared:.3f})")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"PLR fitted curve saved to {output_path}")


def plot_noise_robustness(results_json: Path, output_path: Path) -> None:
    """Plot PLR parameter recovery vs noise level."""
    with open(results_json) as f:
        data = json.load(f)

    results = data["results"]
    sigmas = [r["noise_std"] for r in results]

    params_to_plot = [
        ("constriction_amplitude", "Amplitude error (mm)"),
        ("constriction_latency", "Latency error (s)"),
        ("recovery_time_constant", "Recovery τ error (s)"),
        ("fit_r_squared", "R²"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))

    for ax, (key, ylabel) in zip(axes, params_to_plot, strict=False):
        means = [r["metrics"][key]["mean"] for r in results]
        stds = [r["metrics"][key]["std"] for r in results]

        if key == "fit_r_squared":
            ax.errorbar(sigmas, means, yerr=stds, marker="o", capsize=3, color="steelblue")
            ax.set_ylabel("R²")
            ax.set_ylim(0, 1.05)
        else:
            ax.errorbar(sigmas, means, yerr=stds, marker="o", capsize=3, color="crimson")
            ax.set_ylabel(ylabel)

        ax.set_xlabel("Noise σ (mm)")
        ax.grid(True, alpha=0.3)

    fig.suptitle("PLR Parameter Recovery Under Increasing Noise", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Noise robustness plot saved to {output_path}")


def plot_blink_robustness(results_json: Path, output_path: Path) -> None:
    """Plot PLR parameter recovery vs blink duration."""
    with open(results_json) as f:
        data = json.load(f)

    results = data["results"]
    durations_ms = [r["blink_duration_ms"] for r in results]

    params_to_plot = [
        ("constriction_amplitude", "Amplitude error (mm)"),
        ("constriction_latency", "Latency error (s)"),
        ("recovery_time_constant", "Recovery τ error (s)"),
        ("fit_r_squared", "R²"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))

    for ax, (key, ylabel) in zip(axes, params_to_plot, strict=False):
        means = [r["metrics"][key]["mean"] for r in results]
        stds = [r["metrics"][key]["std"] for r in results]

        if key == "fit_r_squared":
            ax.errorbar(durations_ms, means, yerr=stds, marker="s", capsize=3, color="steelblue")
            ax.set_ylabel("R²")
            ax.set_ylim(0, 1.05)
        else:
            ax.errorbar(durations_ms, means, yerr=stds, marker="s", capsize=3, color="crimson")
            ax.set_ylabel(ylabel)

        ax.set_xlabel("Blink duration (ms)")
        ax.grid(True, alpha=0.3)

    fig.suptitle("PLR Parameter Recovery Under Blink-Induced Data Dropout", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Blink robustness plot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate report figures")
    parser.add_argument("--figures_dir", type=Path, default=Path("figures"))
    parser.add_argument("--csv_dir", type=Path, default=None, help="Dir with training CSVs")
    parser.add_argument(
        "--robustness_dir", type=Path, default=None, help="Dir with robustness JSONs"
    )
    parser.add_argument(
        "--plot",
        choices=["training", "plr", "noise", "blinks", "all"],
        default="all",
    )
    args = parser.parse_args()
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    if args.plot in ("plr", "all"):
        plot_plr_fitted_curve(args.figures_dir / "plr_fitted_curve.pdf")

    if args.plot in ("training", "all") and args.csv_dir:
        plot_training_curves(args.csv_dir, args.figures_dir / "training_curves.pdf")

    if args.plot in ("noise", "all") and args.robustness_dir:
        plot_noise_robustness(
            args.robustness_dir / "noise.json",
            args.figures_dir / "robustness_noise.pdf",
        )

    if args.plot in ("blinks", "all") and args.robustness_dir:
        plot_blink_robustness(
            args.robustness_dir / "blinks.json",
            args.figures_dir / "robustness_blinks.pdf",
        )

    print("Done.")


if __name__ == "__main__":
    main()
