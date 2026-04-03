"""Loss functions for pupil/iris segmentation.

Implements the compound loss from the RITnet paper:
  L = CE + Dice + Boundary

References:
  - RITnet: https://arxiv.org/abs/1910.00694
"""

from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeneralisedDiceLoss(nn.Module):
    """Generalised Dice loss for multi-class segmentation.

    Weights each class inversely by its volume, preventing the loss
    from being dominated by large classes (background/sclera).
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, C, H, W) raw model output
            targets: (B, H, W) integer class labels
        """
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()

        # Per-class weights: inverse of squared volume
        weights = 1.0 / (targets_one_hot.sum(dim=(0, 2, 3)) ** 2 + self.eps)

        intersection = (probs * targets_one_hot).sum(dim=(0, 2, 3))
        union = probs.sum(dim=(0, 2, 3)) + targets_one_hot.sum(dim=(0, 2, 3))

        dice = (2.0 * weights * intersection + self.eps) / (weights * union + self.eps)
        return 1.0 - dice.mean()


class BoundaryLoss(nn.Module):
    """Boundary-aware loss that upweights pixels near class boundaries.

    Computes a distance-based weight map from the ground truth boundaries
    and applies it to the cross-entropy loss, focusing learning on the
    pupil-iris boundary where diameter accuracy is most sensitive.
    """

    def __init__(self, dilation_radius: int = 3):
        super().__init__()
        self.dilation_radius = dilation_radius
        kernel_size = 2 * dilation_radius + 1
        self.register_buffer(
            "_kernel",
            torch.ones(1, 1, kernel_size, kernel_size),
        )

    def _boundary_mask(self, targets: torch.Tensor) -> torch.Tensor:
        """Extract boundary pixels via dilation - erosion."""
        masks = []
        for cls_id in range(targets.max().item() + 1):
            binary = (targets == cls_id).float().unsqueeze(1)
            dilated = F.conv2d(binary, self._kernel, padding=self.dilation_radius)
            dilated = (dilated > 0).float()
            eroded = F.conv2d(binary, self._kernel, padding=self.dilation_radius)
            eroded = (eroded >= self._kernel.numel()).float()
            masks.append(dilated - eroded)
        boundary = torch.clamp(sum(masks), 0, 1).squeeze(1)
        return boundary

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        boundary = self._boundary_mask(targets)
        # Upweight boundary pixels: 1 + boundary_weight * boundary_mask
        weight_map = 1.0 + 5.0 * boundary
        ce = F.cross_entropy(logits, targets, reduction="none")
        return (ce * weight_map).mean()


class CompoundLoss(nn.Module):
    """CE + Dice + Boundary compound loss from RITnet paper."""

    def __init__(
        self,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        boundary_weight: float = 1.0,
    ):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight

        self.ce = nn.CrossEntropyLoss()
        self.dice = GeneralisedDiceLoss()
        self.boundary = BoundaryLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = self.ce_weight * self.ce(logits, targets)
        loss += self.dice_weight * self.dice(logits, targets)
        loss += self.boundary_weight * self.boundary(logits, targets)
        return loss


LOSS_REGISTRY: dict[str, Callable[[], nn.Module]] = {
    "ce": nn.CrossEntropyLoss,
    "dice": GeneralisedDiceLoss,
    "ce_dice": lambda: CompoundLoss(ce_weight=1.0, dice_weight=1.0, boundary_weight=0.0),
    "compound": CompoundLoss,
}


def get_loss(name: str) -> nn.Module:
    """Create a loss function by name.

    Available: 'ce', 'dice', 'ce_dice', 'compound'
    """
    if name not in LOSS_REGISTRY:
        raise ValueError(f"Unknown loss '{name}'. Choose from: {list(LOSS_REGISTRY.keys())}")
    factory = LOSS_REGISTRY[name]
    return factory()
