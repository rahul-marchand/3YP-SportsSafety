"""Segmentation models for pupil/iris detection."""

from pathlib import Path

import torch
import torch.nn as nn

from .ritnet import RITnet, create_ritnet
from .unet import UNet

__all__ = ["RITnet", "UNet", "create_ritnet", "MODEL_REGISTRY", "create_model"]


MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "ritnet": RITnet,
    "unet": UNet,
}


def create_model(
    name: str,
    pretrained_path: str | Path | None = None,
    device: str = "cuda",
    **kwargs,
) -> nn.Module:
    """Create a segmentation model by name.

    Available: 'ritnet', 'unet'.
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Choose from: {list(MODEL_REGISTRY.keys())}")

    model = MODEL_REGISTRY[name](**kwargs)

    if pretrained_path is not None:
        checkpoint = torch.load(pretrained_path, map_location=device, weights_only=True)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

    return model.to(device)
