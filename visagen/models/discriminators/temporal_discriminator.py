"""
Temporal Discriminator for Video Consistency.

Implements 3D Conv based discriminators for temporal consistency in video
face swapping. Reduces flicker artifacts by learning temporal patterns.

Reference:
    "Recycle-GAN: Unsupervised Video Retargeting" (Bansal et al., 2018)
    "vid2vid: Video-to-Video Synthesis" (Wang et al., 2018)
"""

from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F


def _identity_module(module: nn.Module) -> nn.Module:
    """Typed identity wrapper for optional spectral norm."""
    return module


class ResidualBlock3D(nn.Module):
    """
    3D Residual block for temporal processing.

    Args:
        channels: Number of input/output channels.
        kernel_size: Convolution kernel size. Default: 3.
    """

    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv3d(channels, channels, kernel_size, padding=padding)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        residual = x
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = self.conv2(x)
        return F.leaky_relu(x + residual, 0.2)


class TemporalDiscriminator(nn.Module):
    """
    3D Conv Discriminator for temporal consistency.

    Uses 3D convolutions to analyze temporal patterns across frames.
    Outputs a single score indicating temporal consistency of the sequence.

    Args:
        in_channels: Number of input channels. Default: 3.
        base_ch: Base channel count. Default: 32.
        sequence_length: Expected frame count. Default: 5.
        use_spectral_norm: Apply spectral normalization. Default: False.

    Example:
        >>> D = TemporalDiscriminator(sequence_length=5)
        >>> x = torch.randn(2, 3, 5, 256, 256)  # (B, C, T, H, W)
        >>> score = D(x)
        >>> score.shape
        torch.Size([2, 1])
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_ch: int = 32,
        sequence_length: int = 5,
        use_spectral_norm: bool = False,
    ) -> None:
        super().__init__()

        self.sequence_length = sequence_length
        norm_fn = nn.utils.spectral_norm if use_spectral_norm else _identity_module

        # Encoder path with spatial downsampling
        # Input: (B, C, T, H, W) -> progressively reduce spatial dims
        self.conv1 = norm_fn(
            nn.Conv3d(in_channels, base_ch, kernel_size=3, stride=(1, 2, 2), padding=1)
        )  # H/2, W/2

        self.conv2 = norm_fn(
            nn.Conv3d(base_ch, base_ch * 2, kernel_size=3, stride=(1, 2, 2), padding=1)
        )  # H/4, W/4

        self.conv3 = norm_fn(
            nn.Conv3d(
                base_ch * 2, base_ch * 4, kernel_size=3, stride=(1, 2, 2), padding=1
            )
        )  # H/8, W/8

        self.conv4 = norm_fn(
            nn.Conv3d(
                base_ch * 4, base_ch * 8, kernel_size=3, stride=(2, 2, 2), padding=1
            )
        )  # T/2, H/16, W/16

        # Residual block for temporal processing
        self.res_block = ResidualBlock3D(base_ch * 8)

        # Final temporal aggregation
        self.conv5 = norm_fn(
            nn.Conv3d(
                base_ch * 8, base_ch * 8, kernel_size=3, stride=(2, 2, 2), padding=1
            )
        )

        # Global average pooling + FC
        self.fc = norm_fn(nn.Linear(base_ch * 8, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input sequence (B, C, T, H, W).

        Returns:
            Temporal discrimination score (B, 1).
        """
        # Encoder
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = F.leaky_relu(self.conv4(x), 0.2)

        # Residual processing
        x = self.res_block(x)

        # Final conv
        x = F.leaky_relu(self.conv5(x), 0.2)

        # Global average pooling
        x = F.adaptive_avg_pool3d(x, 1)
        x = x.view(x.size(0), -1)

        # Output score
        return cast(torch.Tensor, self.fc(x))


class TemporalPatchDiscriminator(nn.Module):
    """
    Combined temporal + spatial discriminator.

    Returns both temporal consistency score and per-frame spatial
    discrimination scores. Uses 2D PatchDiscriminator for spatial
    and 3D Conv for temporal patterns.

    Args:
        in_channels: Number of input channels. Default: 3.
        base_ch: Base channel count for temporal path. Default: 32.
        spatial_base_ch: Base channel count for spatial path. Default: 64.
        sequence_length: Expected frame count. Default: 5.
        use_spectral_norm: Apply spectral normalization. Default: False.

    Example:
        >>> D = TemporalPatchDiscriminator(sequence_length=5)
        >>> x = torch.randn(2, 3, 5, 256, 256)  # (B, C, T, H, W)
        >>> temporal, spatial = D(x)
        >>> temporal.shape
        torch.Size([2, 1])
        >>> spatial.shape
        torch.Size([2, 5, 1, 30, 30])
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_ch: int = 32,
        spatial_base_ch: int = 64,
        sequence_length: int = 5,
        use_spectral_norm: bool = False,
    ) -> None:
        super().__init__()

        self.sequence_length = sequence_length

        # Temporal discriminator (3D)
        self.temporal_disc = TemporalDiscriminator(
            in_channels=in_channels,
            base_ch=base_ch,
            sequence_length=sequence_length,
            use_spectral_norm=use_spectral_norm,
        )

        # Spatial discriminator (2D) - applied per frame
        norm_fn = nn.utils.spectral_norm if use_spectral_norm else _identity_module

        self.spatial_layers = nn.Sequential(
            norm_fn(nn.Conv2d(in_channels, spatial_base_ch, 4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            norm_fn(
                nn.Conv2d(spatial_base_ch, spatial_base_ch * 2, 4, stride=2, padding=1)
            ),
            nn.LeakyReLU(0.2, inplace=True),
            norm_fn(
                nn.Conv2d(
                    spatial_base_ch * 2, spatial_base_ch * 4, 4, stride=2, padding=1
                )
            ),
            nn.LeakyReLU(0.2, inplace=True),
            norm_fn(
                nn.Conv2d(
                    spatial_base_ch * 4, spatial_base_ch * 4, 4, stride=1, padding=1
                )
            ),
            nn.LeakyReLU(0.2, inplace=True),
            norm_fn(nn.Conv2d(spatial_base_ch * 4, 1, 4, stride=1, padding=1)),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input sequence (B, C, T, H, W).

        Returns:
            Tuple of:
            - temporal_score: (B, 1) temporal discrimination
            - spatial_scores: (B, T, 1, H', W') per-frame spatial discrimination
        """
        B, C, T, H, W = x.shape

        # Temporal discrimination
        temporal_score = self.temporal_disc(x)

        # Spatial discrimination per frame
        # Reshape: (B, C, T, H, W) -> (B*T, C, H, W)
        x_spatial = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)

        # Apply spatial discriminator
        spatial_out = self.spatial_layers(x_spatial)  # (B*T, 1, H', W')

        # Reshape back: (B*T, 1, H', W') -> (B, T, 1, H', W')
        _, _, Hp, Wp = spatial_out.shape
        spatial_scores = spatial_out.view(B, T, 1, Hp, Wp)

        return temporal_score, spatial_scores


class LightweightTemporalDiscriminator(nn.Module):
    """
    Lightweight temporal discriminator for memory-constrained training.

    Uses fewer layers and smaller channels for reduced memory footprint.
    Suitable for high-resolution inputs or limited GPU memory.

    Args:
        in_channels: Number of input channels. Default: 3.
        base_ch: Base channel count. Default: 16.
        sequence_length: Expected frame count. Default: 5.
        use_spectral_norm: Apply spectral normalization. Default: False.

    Example:
        >>> D = LightweightTemporalDiscriminator()
        >>> x = torch.randn(2, 3, 5, 256, 256)
        >>> score = D(x)
        >>> score.shape
        torch.Size([2, 1])
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_ch: int = 16,
        sequence_length: int = 5,
        use_spectral_norm: bool = False,
    ) -> None:
        super().__init__()

        self.sequence_length = sequence_length
        norm_fn = nn.utils.spectral_norm if use_spectral_norm else _identity_module

        # Simplified encoder
        self.encoder = nn.Sequential(
            norm_fn(
                nn.Conv3d(
                    in_channels, base_ch, kernel_size=3, stride=(1, 2, 2), padding=1
                )
            ),
            nn.LeakyReLU(0.2, inplace=True),
            norm_fn(
                nn.Conv3d(
                    base_ch, base_ch * 2, kernel_size=3, stride=(1, 2, 2), padding=1
                )
            ),
            nn.LeakyReLU(0.2, inplace=True),
            norm_fn(
                nn.Conv3d(
                    base_ch * 2, base_ch * 4, kernel_size=3, stride=(2, 2, 2), padding=1
                )
            ),
            nn.LeakyReLU(0.2, inplace=True),
            norm_fn(
                nn.Conv3d(
                    base_ch * 4, base_ch * 4, kernel_size=3, stride=(2, 2, 2), padding=1
                )
            ),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.fc = norm_fn(nn.Linear(base_ch * 4, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input sequence (B, C, T, H, W).

        Returns:
            Temporal discrimination score (B, 1).
        """
        x = self.encoder(x)
        x = F.adaptive_avg_pool3d(x, 1)
        x = x.view(x.size(0), -1)
        return cast(torch.Tensor, self.fc(x))
