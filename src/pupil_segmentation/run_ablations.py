"""Experiment runner for segmentation ablation studies.

Usage:
    uv run python src/pupil_segmentation/run_ablations.py --data_dir <path> --dry_run
    uv run python src/pupil_segmentation/run_ablations.py --data_dir <path> --experiment ritnet_ce
"""

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Experiment:
    name: str
    model: str
    loss: str
    preprocessing: bool
    augmentation: str = "none"
    note: str = ""


EXPERIMENTS = [
    # Batch 1: All independent, can run in parallel
    # -- Loss ablation (RITnet, default preprocessing)
    Experiment("ritnet_ce", "ritnet", "ce", True),
    Experiment("ritnet_dice", "ritnet", "dice", True),
    Experiment("ritnet_ce_dice", "ritnet", "ce_dice", True),
    Experiment("ritnet_compound", "ritnet", "compound", True),
    # -- Architecture baseline (U-Net with CE for fair initial comparison)
    Experiment("unet_ce", "unet", "ce", True),
    # -- Preprocessing ablation (CE baseline, no preprocessing)
    Experiment("ritnet_no_preproc", "ritnet", "ce", False),
    # Batch 2: Run after Batch 1 results (update loss to best from Batch 1)
    # Experiment("unet_{best}", "unet", "{best_loss}", True),
    # Experiment("ritnet_{best}_no_preproc", "ritnet", "{best_loss}", False),
    # -- Augmentation ablation (update loss to best from Batch 1)
    Experiment(
        "ritnet_compound_aug_std",
        "ritnet",
        "compound",
        True,
        augmentation="standard",
        note="Best baseline + standard augmentation",
    ),
    Experiment(
        "ritnet_compound_aug_domain",
        "ritnet",
        "compound",
        True,
        augmentation="domain",
        note="Best baseline + standard + domain adaptation",
    ),
]


def build_command(
    exp: Experiment,
    data_dir: Path,
    num_epochs: int,
    batch_size: int,
) -> list[str]:
    save_dir = Path("checkpoints") / exp.name
    cmd = [
        sys.executable,
        "src/pupil_segmentation/train.py",
        "--data_dir",
        str(data_dir),
        "--model",
        exp.model,
        "--loss",
        exp.loss,
        "--save_dir",
        str(save_dir),
        "--num_epochs",
        str(num_epochs),
        "--batch_size",
        str(batch_size),
        "--no_wandb",
    ]
    if not exp.preprocessing:
        cmd.append("--no_preprocessing")
    if exp.augmentation != "none":
        cmd.extend(["--augmentation", exp.augmentation])
    return cmd


def main():
    parser = argparse.ArgumentParser(description="Run segmentation ablation experiments")
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--dry_run", action="store_true", help="Print commands without executing")
    parser.add_argument("--experiment", type=str, default=None, help="Run only this experiment")
    args = parser.parse_args()

    experiments = EXPERIMENTS
    if args.experiment:
        experiments = [e for e in experiments if e.name == args.experiment]
        if not experiments:
            names = [e.name for e in EXPERIMENTS]
            print(f"Unknown experiment '{args.experiment}'. Available: {names}")
            sys.exit(1)

    print(f"{'[DRY RUN] ' if args.dry_run else ''}Running {len(experiments)} experiment(s)\n")

    for i, exp in enumerate(experiments, 1):
        cmd = build_command(exp, args.data_dir, args.num_epochs, args.batch_size)
        cmd_str = " ".join(cmd)

        print(f"[{i}/{len(experiments)}] {exp.name}")
        if exp.note:
            print(f"  Note: {exp.note}")
        print(f"  {cmd_str}")

        if args.dry_run:
            print()
            continue

        start = time.time()
        result = subprocess.run(cmd)
        elapsed = time.time() - start
        status = "OK" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
        print(f"  {status} in {elapsed:.0f}s\n")


if __name__ == "__main__":
    main()
