"""
CBAM: Convolutional Block Attention Module

Implementation based on:
"CBAM: Convolutional Block Attention Module" (Woo et al., ECCV 2018)

This module applies channel attention followed by spatial attention
to help the model focus on important features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """
    Channel Attention Module.

    Applies attention across channel dimension using both max-pool and avg-pool
    features passed through a shared MLP.

    Args:
        in_channels: Number of input channels.
        reduction_ratio: Channel reduction ratio for the MLP. Default: 16.
    """

    def __init__(self, in_channels: int, reduction_ratio: int = 16) -> None:
        super().__init__()
        self.in_channels = in_channels
        reduced_channels = max(in_channels // reduction_ratio, 1)

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, reduced_channels, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(reduced_channels, in_channels, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply channel attention.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Attention-weighted tensor of shape (B, C, H, W).
        """
        batch_size, channels, _, _ = x.shape

        # Global average pooling: (B, C, H, W) -> (B, C)
        avg_pool = F.adaptive_avg_pool2d(x, 1).view(batch_size, channels)

        # Global max pooling: (B, C, H, W) -> (B, C)
        max_pool = F.adaptive_max_pool2d(x, 1).view(batch_size, channels)

        # Shared MLP
        avg_out = self.mlp(avg_pool)
        max_out = self.mlp(max_pool)

        # Combine and apply sigmoid
        attention = torch.sigmoid(avg_out + max_out)

        # Reshape and apply attention: (B, C) -> (B, C, 1, 1)
        return x * attention.view(batch_size, channels, 1, 1)


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module.

    Applies attention across spatial dimensions using channel-wise max and avg
    features passed through a convolution.

    Args:
        kernel_size: Convolution kernel size. Default: 7.
    """

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels=2,
            out_channels=1,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply spatial attention.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Attention-weighted tensor of shape (B, C, H, W).
        """
        # Channel-wise average: (B, C, H, W) -> (B, 1, H, W)
        avg_pool = torch.mean(x, dim=1, keepdim=True)

        # Channel-wise max: (B, C, H, W) -> (B, 1, H, W)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)

        # Concatenate: (B, 2, H, W)
        combined = torch.cat([avg_pool, max_pool], dim=1)

        # Convolution + sigmoid: (B, 2, H, W) -> (B, 1, H, W)
        attention = torch.sigmoid(self.conv(combined))

        return x * attention


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.

    Sequentially applies channel attention and spatial attention.

    Args:
        in_channels: Number of input channels.
        reduction_ratio: Channel reduction ratio for channel attention. Default: 16.
        spatial_kernel_size: Kernel size for spatial attention conv. Default: 7.

    Example:
        >>> cbam = CBAM(in_channels=256)
        >>> x = torch.randn(2, 256, 32, 32)
        >>> out = cbam(x)
        >>> out.shape
        torch.Size([2, 256, 32, 32])
    """

    def __init__(
        self,
        in_channels: int,
        reduction_ratio: int = 16,
        spatial_kernel_size: int = 7,
    ) -> None:
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(spatial_kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply CBAM attention.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Attention-weighted tensor of shape (B, C, H, W).
        """
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x
