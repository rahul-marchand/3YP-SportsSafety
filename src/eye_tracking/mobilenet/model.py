"""
MobileNetV2 model for gaze direction regression.

Uses pretrained MobileNetV2 with a custom regression head for predicting
vertical and horizontal gaze angles.
"""

import torch
import torch.nn as nn
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2


class GazeMobileNet(nn.Module):
    """MobileNetV2 with regression head for gaze angle prediction."""

    def __init__(self, pretrained: bool = True, freeze_backbone: bool = False):
        """
        Initialize the gaze estimation model.

        Args:
            pretrained: Whether to use pretrained ImageNet weights
            freeze_backbone: Whether to freeze the backbone during training
        """
        super().__init__()

        # Load pretrained MobileNetV2
        if pretrained:
            weights = MobileNet_V2_Weights.IMAGENET1K_V1
            self.backbone = mobilenet_v2(weights=weights)
        else:
            self.backbone = mobilenet_v2(weights=None)

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Get the number of features from the last convolutional layer
        # MobileNetV2 has 1280 features before the classifier
        num_features = self.backbone.classifier[1].in_features

        # Replace classifier with regression head
        # Output: 2 values (horizontal angle, vertical angle) - matches training data format [theta_h, theta_v]
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(512, 2),  # 2 outputs: horizontal (theta_h), vertical (theta_v)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)

        Returns:
            Predicted gaze angles of shape (batch_size, 2) [horizontal (theta_h), vertical (theta_v)]
        """
        return self.backbone(x)

    def unfreeze_backbone(self):
        """Unfreeze the backbone for fine-tuning."""
        for param in self.backbone.features.parameters():
            param.requires_grad = True

    def freeze_backbone(self):
        """Freeze the backbone."""
        for param in self.backbone.features.parameters():
            param.requires_grad = False


def create_model(
    pretrained: bool = True, freeze_backbone: bool = False, device: str = "cuda"
) -> GazeMobileNet:
    """
    Create and initialize the gaze estimation model.

    Args:
        pretrained: Whether to use pretrained ImageNet weights
        freeze_backbone: Whether to freeze the backbone during training
        device: Device to move the model to

    Returns:
        Initialized model
    """
    model = GazeMobileNet(pretrained=pretrained, freeze_backbone=freeze_backbone)
    model = model.to(device)
    return model


if __name__ == "__main__":
    # Test the model
    model = create_model(pretrained=True, freeze_backbone=False, device="cpu")

    # Create dummy input
    dummy_input = torch.randn(4, 3, 224, 224)

    # Forward pass
    output = model(dummy_input)

    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output (sample): {output[0]}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
