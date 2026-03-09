#!/usr/bin/env python3
"""
Export a trained PyTorch gaze model to ONNX format for Pi inference.

Usage:
    python export_onnx.py --checkpoint model_40cm.pth --output model_40cm.onnx
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mobilenet.model import GazeMobileNet


def export_to_onnx(checkpoint_path: str, output_path: str, input_size: int = 224):
    """
    Export a PyTorch gaze model checkpoint to ONNX format.

    Args:
        checkpoint_path: Path to PyTorch .pth checkpoint
        output_path: Path to save .onnx file
        input_size: Model input resolution (default 224)
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    model = GazeMobileNet(pretrained=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        val_loss = checkpoint.get("val_loss", "unknown")
        epoch = checkpoint.get("epoch", "unknown")
        print(f"  Epoch: {epoch}, Val Loss: {val_loss}")
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    dummy_input = torch.randn(1, 3, input_size, input_size)

    print(f"Exporting to ONNX: {output_path}")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["input"],
        output_names=["angles"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "angles": {0: "batch_size"},
        },
        opset_version=17,
    )

    import os
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Exported successfully: {output_path} ({size_mb:.1f} MB)")

    try:
        import onnxruntime as ort
        session = ort.InferenceSession(output_path)
        test_input = dummy_input.numpy()
        result = session.run(None, {"input": test_input})
        print(f"Verification: input {test_input.shape} -> output {result[0].shape}")
        print(f"  Sample prediction: theta_h={result[0][0][0]:.4f}, theta_v={result[0][0][1]:.4f}")
    except ImportError:
        print("onnxruntime not installed, skipping verification")


def main():
    parser = argparse.ArgumentParser(description="Export gaze model to ONNX")
    parser.add_argument("--checkpoint", type=str, required=True, help="PyTorch checkpoint path")
    parser.add_argument("--output", type=str, default=None, help="Output ONNX path")
    parser.add_argument("--input_size", type=int, default=224, help="Input image size")
    args = parser.parse_args()

    if args.output is None:
        args.output = Path(args.checkpoint).stem + ".onnx"

    export_to_onnx(args.checkpoint, args.output, args.input_size)


if __name__ == "__main__":
    main()
