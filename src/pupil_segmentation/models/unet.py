"""Standard U-Net baseline for pupil/iris segmentation."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import NUM_CLASSES, RITNET_IN_CHANNELS


class _ConvBlock(nn.Module):
    """Two 3x3 convolutions with BatchNorm and ReLU."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    """Standard U-Net with 4 encoder/decoder levels.

    Provided as a baseline for comparison against RITnet.
    """

    def __init__(
        self,
        in_channels: int = RITNET_IN_CHANNELS,
        num_classes: int = NUM_CLASSES,
        base_channels: int = 32,
    ):
        super().__init__()
        c = base_channels

        # Encoder
        self.enc1 = _ConvBlock(in_channels, c)
        self.enc2 = _ConvBlock(c, c * 2)
        self.enc3 = _ConvBlock(c * 2, c * 4)
        self.enc4 = _ConvBlock(c * 4, c * 8)

        # Bottleneck
        self.bottleneck = _ConvBlock(c * 8, c * 16)

        # Decoder
        self.up4 = nn.ConvTranspose2d(c * 16, c * 8, 2, stride=2)
        self.dec4 = _ConvBlock(c * 16, c * 8)
        self.up3 = nn.ConvTranspose2d(c * 8, c * 4, 2, stride=2)
        self.dec3 = _ConvBlock(c * 8, c * 4)
        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2)
        self.dec2 = _ConvBlock(c * 4, c * 2)
        self.up1 = nn.ConvTranspose2d(c * 2, c, 2, stride=2)
        self.dec1 = _ConvBlock(c * 2, c)

        self.pool = nn.MaxPool2d(2)
        self.out_conv = nn.Conv2d(c, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder with skip connections
        d4 = self.dec4(self._match_and_cat(self.up4(b), e4))
        d3 = self.dec3(self._match_and_cat(self.up3(d4), e3))
        d2 = self.dec2(self._match_and_cat(self.up2(d3), e2))
        d1 = self.dec1(self._match_and_cat(self.up1(d2), e1))

        return self.out_conv(d1)

    @staticmethod
    def _match_and_cat(upsampled: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Pad upsampled tensor to match skip connection spatial dims, then concatenate."""
        dh = skip.shape[2] - upsampled.shape[2]
        dw = skip.shape[3] - upsampled.shape[3]
        if dh != 0 or dw != 0:
            upsampled = F.pad(upsampled, [0, dw, 0, dh])
        return torch.cat([upsampled, skip], dim=1)
