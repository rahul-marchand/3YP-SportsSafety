"""Inference script for pupil segmentation on collected headset images.

Provides:
- Single image inference with vignette cropping for headset camera
- Batch directory processing
- Visualization with ellipse fitting
- Export of segmentation masks and ellipse parameters

Usage:
    uv run python src/pupil_segmentation/predict.py \
        --checkpoint checkpoints/best_model.pth \
        --input path/to/image_or_dir \
        --output results/headset_validation
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.amp import autocast
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.pupil_segmentation.config import (  # noqa: E402
    CLASS_IDS,
    NORMALIZE_MEAN,
    NORMALIZE_STD,
    OPENEDS_IMAGE_SIZE,
)
from src.pupil_segmentation.evaluation.ellipse_fitting import (  # noqa: E402
    EllipseParams,
    compute_pupil_iris_ratio,
    fit_ellipse_to_mask,
)
from src.pupil_segmentation.models import create_model  # noqa: E402

# Color palette for visualization
# background=black, sclera=white, iris=blue, pupil=green
PALETTE = np.array(
    [
        [0, 0, 0],  # background
        [255, 255, 255],  # sclera
        [0, 100, 255],  # iris (blue)
        [0, 255, 0],  # pupil (green)
    ],
    dtype=np.uint8,
)


def crop_headset_vignette(image: np.ndarray) -> np.ndarray:
    """Crop fixed dead space from headset camera vignette.

    Removes 25% top/bottom and 25% left/right.
    """
    h, w = image.shape[:2]
    return image[int(0.25 * h) : int(0.75 * h), int(0.25 * w) : int(0.75 * w)]


def preprocess_image(image: np.ndarray) -> tuple[torch.Tensor, tuple[int, int]]:
    """Preprocess a grayscale image for inference.

    Crops the headset vignette, resizes, and normalizes. No gamma correction
    or CLAHE is applied — our collected images have a different appearance
    to OpenEDS so those transforms are not beneficial.

    Args:
        image: Grayscale image (H, W) as uint8

    Returns:
        Tuple of (tensor, original_size) where tensor is (1, 1, H, W)
    """
    # Crop dead space from headset vignette
    image = crop_headset_vignette(image)

    # Resize to expected size
    original_size = image.shape[:2]
    if original_size != OPENEDS_IMAGE_SIZE:
        image = cv2.resize(image, (OPENEDS_IMAGE_SIZE[1], OPENEDS_IMAGE_SIZE[0]))

    # Normalize
    image = image.astype(np.float32) / 255.0
    image = (image - NORMALIZE_MEAN) / NORMALIZE_STD

    # Convert to tensor
    tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

    return tensor, original_size


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """Convert class indices to RGB visualization.

    Args:
        mask: Segmentation mask (H, W) with class indices 0-3

    Returns:
        RGB image (H, W, 3)
    """
    return PALETTE[mask]


def draw_ellipses(
    image: np.ndarray,
    pupil_params: EllipseParams,
    iris_params: EllipseParams,
    pupil_color: tuple[int, int, int] = (0, 255, 0),
    iris_color: tuple[int, int, int] = (0, 100, 255),
    thickness: int = 2,
) -> np.ndarray:
    """Draw fitted ellipses on image.

    Args:
        image: Input image (H, W) or (H, W, 3) in RGB
        pupil_params: Fitted pupil ellipse parameters
        iris_params: Fitted iris ellipse parameters
        pupil_color: RGB color for pupil ellipse
        iris_color: RGB color for iris ellipse
        thickness: Line thickness

    Returns:
        Image with ellipses drawn (H, W, 3) in RGB
    """
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image = image.copy()

    # Draw iris ellipse (larger, draw first)
    if iris_params.valid:
        center = (int(iris_params.center[0]), int(iris_params.center[1]))
        axes = (int(iris_params.axes[0] / 2), int(iris_params.axes[1] / 2))
        cv2.ellipse(image, center, axes, iris_params.angle, 0, 360, iris_color, thickness)
        cv2.circle(image, center, 3, iris_color, -1)

    # Draw pupil ellipse
    if pupil_params.valid:
        center = (int(pupil_params.center[0]), int(pupil_params.center[1]))
        axes = (int(pupil_params.axes[0] / 2), int(pupil_params.axes[1] / 2))
        cv2.ellipse(image, center, axes, pupil_params.angle, 0, 360, pupil_color, thickness)
        cv2.circle(image, center, 3, pupil_color, -1)

    return image


class PupilSegmentor:
    """Inference wrapper for pupil segmentation.

    Provides easy-to-use interface for running segmentation on images
    and extracting pupil/iris parameters.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        model_name: str = "ritnet",
        device: str = "cuda",
    ):
        """Initialize segmentor.

        Args:
            checkpoint_path: Path to trained model checkpoint
            model_name: Model architecture name ('ritnet' or 'unet')
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.device = device
        self.model = create_model(model_name, pretrained_path=checkpoint_path, device=device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, image: np.ndarray) -> dict:
        """Run segmentation on a single image.

        Args:
            image: Grayscale image (H, W) as uint8

        Returns:
            Dictionary with:
            - 'mask': Segmentation mask (H, W) with class indices
            - 'pupil_ellipse': EllipseParams for pupil
            - 'iris_ellipse': EllipseParams for iris
            - 'pupil_iris_ratio': float ratio of pupil/iris diameter
        """
        tensor, original_size = preprocess_image(image)
        tensor = tensor.to(self.device)

        with autocast(device_type=self.device, enabled=self.device != "cpu"):
            output = self.model(tensor)

        mask = output.argmax(dim=1).squeeze(0).cpu().numpy()

        # Resize mask back to original size if needed
        if original_size != OPENEDS_IMAGE_SIZE:
            mask = cv2.resize(
                mask.astype(np.uint8),
                (original_size[1], original_size[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        pupil_ellipse = fit_ellipse_to_mask(mask, CLASS_IDS["pupil"])
        iris_ellipse = fit_ellipse_to_mask(mask, CLASS_IDS["iris"])
        ratio = compute_pupil_iris_ratio(mask)

        return {
            "mask": mask,
            "pupil_ellipse": pupil_ellipse,
            "iris_ellipse": iris_ellipse,
            "pupil_iris_ratio": ratio,
        }

    def predict_and_visualize(
        self,
        image: np.ndarray,
        result: dict | None = None,
        show_ellipses: bool = True,
        alpha: float = 0.5,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run segmentation and create visualization.

        Args:
            image: Grayscale image (H, W) as uint8
            result: Pre-computed result from predict(). If None, runs inference.
            show_ellipses: Whether to draw fitted ellipses
            alpha: Blend factor for mask overlay (0=image only, 1=mask only)

        Returns:
            Tuple of (mask, visualization as RGB)
        """
        if result is None:
            result = self.predict(image)
        mask = result["mask"]

        # Create visualization (crop to match mask dimensions)
        # Use RGB throughout since output is saved with PIL
        image_rgb = cv2.cvtColor(crop_headset_vignette(image), cv2.COLOR_GRAY2RGB)
        mask_colored = colorize_mask(mask)

        visualization = cv2.addWeighted(image_rgb, 1 - alpha, mask_colored, alpha, 0)

        if show_ellipses:
            visualization = draw_ellipses(
                visualization,
                result["pupil_ellipse"],
                result["iris_ellipse"],
            )

        return mask, visualization


def ellipse_to_dict(ellipse: EllipseParams) -> dict:
    """Convert EllipseParams to JSON-serializable dict."""
    return {
        "center": list(ellipse.center),
        "axes": list(ellipse.axes),
        "angle": ellipse.angle,
        "valid": ellipse.valid,
    }


def process_single_image(
    segmentor: PupilSegmentor,
    input_path: Path,
    output_dir: Path,
    save_mask: bool = True,
    save_viz: bool = True,
    save_ellipse: bool = True,
    show_ellipses: bool = True,
) -> dict:
    """Process a single image and save outputs."""
    image = np.array(Image.open(input_path).convert("L"))

    result = segmentor.predict(image)
    mask = result["mask"]
    stem = input_path.stem

    if save_mask:
        mask_path = output_dir / f"{stem}_mask.png"
        Image.fromarray(mask.astype(np.uint8)).save(mask_path)

    if save_viz:
        _, viz = segmentor.predict_and_visualize(image, result=result, show_ellipses=show_ellipses)
        viz_path = output_dir / f"{stem}_viz.png"
        Image.fromarray(viz).save(viz_path)

    ellipse_data = {
        "pupil": ellipse_to_dict(result["pupil_ellipse"]),
        "iris": ellipse_to_dict(result["iris_ellipse"]),
        "pupil_iris_ratio": result["pupil_iris_ratio"],
    }

    if save_ellipse:
        json_path = output_dir / f"{stem}_ellipse.json"
        with open(json_path, "w") as f:
            json.dump(ellipse_data, f, indent=2)

    return ellipse_data


def process_directory(
    segmentor: PupilSegmentor,
    input_dir: Path,
    output_dir: Path,
    save_masks: bool = True,
    save_viz: bool = True,
    save_ellipse: bool = True,
    show_ellipses: bool = True,
) -> None:
    """Process all images in a directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    image_files = [f for f in input_dir.iterdir() if f.suffix.lower() in image_extensions]

    if not image_files:
        print(f"No images found in {input_dir}")
        return

    print(f"Processing {len(image_files)} images...")

    all_results = {}
    for img_path in tqdm(image_files, desc="Processing"):
        try:
            result = process_single_image(
                segmentor,
                img_path,
                output_dir,
                save_mask=save_masks,
                save_viz=save_viz,
                save_ellipse=save_ellipse,
                show_ellipses=show_ellipses,
            )
            all_results[img_path.name] = result
        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")

    summary_path = output_dir / "results_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {output_dir}")
    print(f"Summary: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Run pupil segmentation inference")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to model checkpoint")
    parser.add_argument("--input", type=Path, required=True, help="Input image or directory")
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    parser.add_argument(
        "--model", type=str, default="ritnet", help="Model architecture (ritnet, unet)"
    )
    parser.add_argument("--no_mask", action="store_true", help="Don't save segmentation masks")
    parser.add_argument("--no_viz", action="store_true", help="Don't save visualizations")
    parser.add_argument("--no_ellipse", action="store_true", help="Don't save ellipse parameters")
    parser.add_argument(
        "--no_draw_ellipse",
        action="store_true",
        help="Don't draw ellipses on visualization",
    )

    args = parser.parse_args()

    if not args.checkpoint.exists():
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    if not args.input.exists():
        print(f"Error: Input not found: {args.input}")
        sys.exit(1)

    args.output.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Loading {args.model} from {args.checkpoint}...")

    segmentor = PupilSegmentor(args.checkpoint, model_name=args.model, device=device)

    if args.input.is_file():
        print(f"Processing single image: {args.input}")
        result = process_single_image(
            segmentor,
            args.input,
            args.output,
            save_mask=not args.no_mask,
            save_viz=not args.no_viz,
            save_ellipse=not args.no_ellipse,
            show_ellipses=not args.no_draw_ellipse,
        )
        print("\nResults:")
        print(f"  Pupil center: {result['pupil']['center']}")
        print(f"  Iris center:  {result['iris']['center']}")
        print(f"  Pupil/Iris ratio: {result['pupil_iris_ratio']:.4f}")
    else:
        process_directory(
            segmentor,
            args.input,
            args.output,
            save_masks=not args.no_mask,
            save_viz=not args.no_viz,
            save_ellipse=not args.no_ellipse,
            show_ellipses=not args.no_draw_ellipse,
        )


if __name__ == "__main__":
    main()
