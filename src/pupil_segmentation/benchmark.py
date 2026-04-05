"""Benchmarking script for pupil segmentation models."""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import wandb
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.pupil_segmentation.config import CLASS_IDS, OPENEDS_IMAGE_SIZE  # noqa: E402
from src.pupil_segmentation.Dataset.dataloader import get_dataloaders  # noqa: E402
from src.pupil_segmentation.evaluation.ellipse_fitting import (  # noqa: E402
    compute_ellipse_error,
    compute_pupil_iris_ratio,
    fit_ellipse_to_mask,
)
from src.pupil_segmentation.evaluation.metrics import (  # noqa: E402
    BenchmarkResults,
    EllipseMetrics,
    SegmentationMetrics,
    compute_dice,
    compute_iou,
    get_model_size_mb,
    measure_inference_time,
)
from src.pupil_segmentation.models import create_model  # noqa: E402


@torch.no_grad()
def benchmark_model(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: str,
    target_size: tuple[int, int] | None = None,
) -> BenchmarkResults:
    """Run full benchmark on a model."""
    model.eval()

    # Collect metrics
    all_iou = {k: [] for k in ["background", "sclera", "iris", "pupil"]}
    all_dice = {k: [] for k in ["background", "sclera", "iris", "pupil"]}
    pupil_ellipse_errors = {"center": [], "axis": [], "angle": []}
    iris_ellipse_errors = {"center": [], "axis": [], "angle": []}
    ratio_errors = []

    print("Evaluating on test set...")
    for images, masks in tqdm(test_loader, desc="Benchmarking"):
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)
        preds = outputs.argmax(dim=1)

        # Per-sample metrics
        for pred, mask in zip(preds, masks, strict=False):
            pred_np = pred.cpu().numpy()
            mask_np = mask.cpu().numpy()

            # Segmentation metrics
            iou = compute_iou(pred, mask)
            dice = compute_dice(pred, mask)
            for cls_name in all_iou:
                all_iou[cls_name].append(iou[cls_name])
                all_dice[cls_name].append(dice[cls_name])

            # Ellipse fitting metrics
            pred_pupil = fit_ellipse_to_mask(pred_np, CLASS_IDS["pupil"])
            gt_pupil = fit_ellipse_to_mask(mask_np, CLASS_IDS["pupil"])
            pred_iris = fit_ellipse_to_mask(pred_np, CLASS_IDS["iris"])
            gt_iris = fit_ellipse_to_mask(mask_np, CLASS_IDS["iris"])

            pupil_err = compute_ellipse_error(pred_pupil, gt_pupil)
            iris_err = compute_ellipse_error(pred_iris, gt_iris)

            if all(np.isfinite(v) for v in pupil_err.values()):
                pupil_ellipse_errors["center"].append(pupil_err["center_error"])
                pupil_ellipse_errors["axis"].append(pupil_err["axis_error"])
                pupil_ellipse_errors["angle"].append(pupil_err["angle_error"])

            if all(np.isfinite(v) for v in iris_err.values()):
                iris_ellipse_errors["center"].append(iris_err["center_error"])
                iris_ellipse_errors["axis"].append(iris_err["axis_error"])
                iris_ellipse_errors["angle"].append(iris_err["angle_error"])

            # Pupil/iris ratio
            pred_ratio = compute_pupil_iris_ratio(pred_np)
            gt_ratio = compute_pupil_iris_ratio(mask_np)
            if pred_ratio > 0 and gt_ratio > 0:
                ratio_errors.append(abs(pred_ratio - gt_ratio))

    # Aggregate results
    seg_metrics = SegmentationMetrics(
        iou_per_class={k: np.mean(v) for k, v in all_iou.items()},
        dice_per_class={k: np.mean(v) for k, v in all_dice.items()},
        mean_iou=np.mean([np.mean(v) for v in all_iou.values()]),
        mean_dice=np.mean([np.mean(v) for v in all_dice.values()]),
    )

    pupil_metrics = EllipseMetrics(
        center_error=np.median(pupil_ellipse_errors["center"])
        if pupil_ellipse_errors["center"]
        else 0,
        axis_error=np.median(pupil_ellipse_errors["axis"])
        if pupil_ellipse_errors["axis"]
        else 0,
        angle_error=np.median(pupil_ellipse_errors["angle"])
        if pupil_ellipse_errors["angle"]
        else 0,
    )

    iris_metrics = EllipseMetrics(
        center_error=np.median(iris_ellipse_errors["center"])
        if iris_ellipse_errors["center"]
        else 0,
        axis_error=np.median(iris_ellipse_errors["axis"])
        if iris_ellipse_errors["axis"]
        else 0,
        angle_error=np.median(iris_ellipse_errors["angle"])
        if iris_ellipse_errors["angle"]
        else 0,
    )

    # Timing
    print("Measuring inference time...")
    img_size = target_size or OPENEDS_IMAGE_SIZE
    input_size = (1, 1, img_size[0], img_size[1])
    inference_time = measure_inference_time(model, input_size, device)

    return BenchmarkResults(
        segmentation=seg_metrics,
        ellipse_pupil=pupil_metrics,
        ellipse_iris=iris_metrics,
        pupil_iris_ratio_mae=np.median(ratio_errors) if ratio_errors else 0,
        inference_time_ms=inference_time,
        model_size_mb=get_model_size_mb(model),
        fps=1000.0 / inference_time if inference_time > 0 else 0,
    )


def print_results(results: BenchmarkResults) -> None:
    """Print benchmark results in a formatted table."""
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)

    print("\nSegmentation Metrics:")
    print("-" * 40)
    print(f"  Mean IoU:  {results.segmentation.mean_iou:.4f}")
    print(f"  Mean Dice: {results.segmentation.mean_dice:.4f}")
    for cls_name in ["pupil", "iris", "sclera", "background"]:
        iou = results.segmentation.iou_per_class.get(cls_name, 0)
        dice = results.segmentation.dice_per_class.get(cls_name, 0)
        print(f"  {cls_name:12s} IoU: {iou:.4f}, Dice: {dice:.4f}")

    print("\nEllipse Fitting (Pupil):")
    print("-" * 40)
    print(f"  Center error: {results.ellipse_pupil.center_error:.2f} px")
    print(f"  Axis error:   {results.ellipse_pupil.axis_error:.2f} px")
    print(f"  Angle error:  {results.ellipse_pupil.angle_error:.2f} deg")

    print("\nEllipse Fitting (Iris):")
    print("-" * 40)
    print(f"  Center error: {results.ellipse_iris.center_error:.2f} px")
    print(f"  Axis error:   {results.ellipse_iris.axis_error:.2f} px")
    print(f"  Angle error:  {results.ellipse_iris.angle_error:.2f} deg")

    print("\nPupil/Iris Ratio:")
    print("-" * 40)
    print(f"  MAE: {results.pupil_iris_ratio_mae:.4f}")

    print("\nPerformance:")
    print("-" * 40)
    print(f"  Inference time: {results.inference_time_ms:.2f} ms")
    print(f"  FPS:            {results.fps:.1f}")
    print(f"  Model size:     {results.model_size_mb:.2f} MB")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Benchmark pupil segmentation models")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Model checkpoint path")
    parser.add_argument("--data_dir", type=Path, required=True, help="OpenEDS dataset directory")
    parser.add_argument(
        "--model", type=str, default="ritnet", help="Model architecture (ritnet, unet)"
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output_json", type=Path, default=None, help="Save results to JSON")
    parser.add_argument(
        "--target_size",
        type=int,
        nargs=2,
        default=None,
        metavar=("H", "W"),
        help="Resize images to (H, W). Must match training resolution.",
    )
    parser.add_argument("--wandb", action="store_true", help="Log to wandb")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model
    print(f"Loading {args.model} from {args.checkpoint}...")
    model = create_model(args.model, pretrained_path=args.checkpoint, device=device)

    # Load test data
    target_size = tuple(args.target_size) if args.target_size else None
    print(f"Loading test data...{f' (resized to {target_size})' if target_size else ''}")
    _, _, test_loader = get_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target_size=target_size,
    )

    # Run benchmark
    results = benchmark_model(model, test_loader, device, target_size=target_size)
    print_results(results)

    # Save JSON
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        output = {
            "model": args.model,
            "checkpoint": str(args.checkpoint),
            "timestamp": datetime.now().isoformat(),
            **results.to_dict(),
        }
        with open(args.output_json, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Results saved to {args.output_json}")

    # Log to wandb
    if args.wandb:
        wandb.init(entity="3YP", project="pupil-segmentation", job_type="benchmark")
        wandb.log(results.to_dict())
        wandb.finish()


if __name__ == "__main__":
    main()
