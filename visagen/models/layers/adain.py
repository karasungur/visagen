"""Adaptive Instance Normalization (AdaIN) Layer.

Implements style-based normalization for neural face synthesis,
allowing dynamic modulation of feature statistics based on style vectors.

References:
    - Huang & Belongie, "Arbitrary Style Transfer in Real-time with
      Adaptive Instance Normalization", ICCV 2017
    - Legacy DFL: core/leras/layers/AdaIN.py
"""

from __future__ import annotations

import torch
import torch.nn as nn


class AdaptiveInstanceNorm2d(nn.Module):
    """Adaptive Instance Normalization for style transfer.

    Normalizes content features using instance statistics and applies
    learned affine transformations derived from a style vector.

    This enables the network to modulate feature statistics based on
    identity-specific information, crucial for preserving facial
    characteristics during face swapping.

    Args:
        in_channels: Number of input feature channels
        style_dim: Dimension of the style/MLP vector

    Example:
        >>> adain = AdaptiveInstanceNorm2d(256, 512)
        >>> content = torch.randn(1, 256, 64, 64)
        >>> style = torch.randn(1, 512)
        >>> output = adain(content, style)
        >>> output.shape
        torch.Size([1, 256, 64, 64])
    """

    def __init__(self, in_channels: int, style_dim: int) -> None:
        """Initialize AdaIN layer.

        Args:
            in_channels: Number of input feature channels
            style_dim: Dimension of the style vector
        """
        super().__init__()
        self.in_channels = in_channels
        self.style_dim = style_dim

        # Style vector to affine parameters projection
        self.gamma_fc = nn.Linear(style_dim, in_channels)
        self.beta_fc = nn.Linear(style_dim, in_channels)

        # Initialize gamma to 1 and beta to 0 for identity transform
        nn.init.ones_(self.gamma_fc.weight.data[:, 0])
        nn.init.zeros_(self.gamma_fc.weight.data[:, 1:])
        nn.init.zeros_(self.gamma_fc.bias.data)
        nn.init.zeros_(self.beta_fc.weight.data)
        nn.init.zeros_(self.beta_fc.bias.data)

    def forward(
        self,
        content: torch.Tensor,
        style: torch.Tensor,
    ) -> torch.Tensor:
        """Apply adaptive instance normalization.

        Args:
            content: Content features (B, C, H, W)
            style: Style vector (B, style_dim)

        Returns:
            Normalized and modulated features (B, C, H, W)
        """
        batch_size, channels, height, width = content.shape

        # Instance normalization: normalize per-instance, per-channel
        mean = content.mean(dim=[2, 3], keepdim=True)
        std = content.std(dim=[2, 3], keepdim=True) + 1e-5
        normalized = (content - mean) / std

        # Compute affine parameters from style vector
        gamma = self.gamma_fc(style).view(batch_size, channels, 1, 1)
        beta = self.beta_fc(style).view(batch_size, channels, 1, 1)

        # Apply style modulation
        return normalized * gamma + beta

    def extra_repr(self) -> str:
        """Return extra representation string."""
        return f"in_channels={self.in_channels}, style_dim={self.style_dim}"


class AdaINResBlock(nn.Module):
    """Residual block with AdaIN for style-conditioned processing.

    Combines convolutional processing with AdaIN normalization for
    style-aware feature transformation.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        style_dim: Dimension of style vector
        upsample: Whether to upsample spatially by 2x

    Example:
        >>> block = AdaINResBlock(256, 256, 512)
        >>> x = torch.randn(1, 256, 32, 32)
        >>> style = torch.randn(1, 512)
        >>> out = block(x, style)
        >>> out.shape
        torch.Size([1, 256, 32, 32])
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        style_dim: int,
        upsample: bool = False,
    ) -> None:
        """Initialize AdaIN residual block.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            style_dim: Dimension of style vector
            upsample: Whether to upsample spatially
        """
        super().__init__()
        self.upsample = upsample

        # First conv + AdaIN
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.adain1 = AdaptiveInstanceNorm2d(out_channels, style_dim)

        # Second conv + AdaIN
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.adain2 = AdaptiveInstanceNorm2d(out_channels, style_dim)

        # Activation
        self.act = nn.SiLU(inplace=True)

        # Skip connection
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        else:
            self.skip = nn.Identity()

        # Upsampling
        if upsample:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        else:
            self.up = nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        style: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through residual block.

        Args:
            x: Input features (B, C_in, H, W)
            style: Style vector (B, style_dim)

        Returns:
            Output features (B, C_out, H', W')
        """
        # Upsample if needed
        x = self.up(x)

        # Skip connection
        skip = self.skip(x)

        # Main path
        out = self.conv1(x)
        out = self.adain1(out, style)
        out = self.act(out)

        out = self.conv2(out)
        out = self.adain2(out, style)

        # Residual connection
        out = out + skip
        out = self.act(out)

        return out
