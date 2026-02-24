"""
PatchGAN Discriminator for Visagen GAN training.

Implements UNet-based PatchGAN discriminator with optional Spectral Normalization.
UNet-based PatchGAN discriminator for Visagen.

Reference:
    "A U-Net Based Discriminator for Generative Adversarial Networks"
    (Schonfeld et al., 2020) - https://arxiv.org/abs/2002.12655
"""

from typing import cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _identity_module(module: nn.Module) -> nn.Module:
    """Typed identity wrapper for optional spectral norm."""
    return module


class ResidualBlock(nn.Module):
    """
    Residual block with LeakyReLU activation.

    Args:
        channels: Number of input/output channels.
        kernel_size: Convolution kernel size. Default: 3.
    """

    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        residual = x
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = self.conv2(x)
        return F.leaky_relu(x + residual, 0.2)


class PatchDiscriminator(nn.Module):
    """
    Simple PatchGAN Discriminator.

    Basic discriminator that outputs a grid of real/fake predictions.

    Args:
        in_channels: Number of input channels. Default: 3.
        base_ch: Base channel count. Default: 64.
        n_layers: Number of downsampling layers. Default: 3.
        use_spectral_norm: Apply spectral normalization. Default: False.

    Example:
        >>> d = PatchDiscriminator()
        >>> x = torch.randn(2, 3, 256, 256)
        >>> out = d(x)
        >>> out.shape
        torch.Size([2, 1, 30, 30])
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_ch: int = 64,
        n_layers: int = 3,
        use_spectral_norm: bool = False,
    ) -> None:
        super().__init__()

        norm_fn = nn.utils.spectral_norm if use_spectral_norm else _identity_module

        layers = []

        # First layer without normalization
        layers.append(norm_fn(nn.Conv2d(in_channels, base_ch, 4, stride=2, padding=1)))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        # Intermediate layers
        ch_mult = 1
        ch_mult_prev = 1
        for n in range(1, n_layers):
            ch_mult_prev = ch_mult
            ch_mult = min(2**n, 8)
            layers.append(
                norm_fn(
                    nn.Conv2d(
                        base_ch * ch_mult_prev,
                        base_ch * ch_mult,
                        4,
                        stride=2,
                        padding=1,
                    )
                )
            )
            layers.append(nn.LeakyReLU(0.2, inplace=True))

        # Last intermediate layer
        ch_mult_prev = ch_mult
        ch_mult = min(2**n_layers, 8)
        layers.append(
            norm_fn(
                nn.Conv2d(
                    base_ch * ch_mult_prev, base_ch * ch_mult, 4, stride=1, padding=1
                )
            )
        )
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        # Output layer
        layers.append(norm_fn(nn.Conv2d(base_ch * ch_mult, 1, 4, stride=1, padding=1)))

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input image (B, C, H, W).

        Returns:
            Patch discrimination scores (B, 1, H', W').
        """
        return cast(torch.Tensor, self.model(x))


class UNetPatchDiscriminator(nn.Module):
    """
    U-Net based PatchGAN Discriminator.

    Combines encoder-decoder architecture with skip connections
    for multi-scale discrimination. Returns both global (center)
    and local (patch-level) discrimination scores.

    Args:
        in_channels: Number of input channels. Default: 3.
        patch_size: Target receptive field size. Default: 70.
        base_ch: Base channel count. Default: 16.
        use_spectral_norm: Apply spectral normalization. Default: False.
        max_channels: Maximum channel count. Default: 512.

    Returns:
        Tuple of (center_out, final_out):
        - center_out: Global discrimination score from bottleneck
        - final_out: Per-patch discrimination scores at full resolution

    Example:
        >>> d = UNetPatchDiscriminator(patch_size=70)
        >>> x = torch.randn(2, 3, 256, 256)
        >>> center, final = d(x)
        >>> center.shape, final.shape
        (torch.Size([2, 1, 16, 16]), torch.Size([2, 1, 256, 256]))
    """

    def __init__(
        self,
        in_channels: int = 3,
        patch_size: int = 70,
        base_ch: int = 16,
        use_spectral_norm: bool = False,
        max_channels: int = 512,
    ) -> None:
        super().__init__()

        self.patch_size = patch_size
        self.use_spectral_norm = use_spectral_norm

        # Find optimal architecture for target patch size
        layers = self._find_archi(patch_size)
        self.n_layers = len(layers)

        # Channel progression: level -1, 0, 1, 2, ...
        level_chs = {
            i - 1: min(base_ch * (2**i), max_channels) for i in range(len(layers) + 1)
        }
        self.level_chs = level_chs

        # Normalization wrapper
        norm_fn = nn.utils.spectral_norm if use_spectral_norm else _identity_module

        # Input conv (1x1 to adjust channels)
        self.in_conv = norm_fn(nn.Conv2d(in_channels, level_chs[-1], 1))

        # Encoder (downsampling path)
        self.down_convs = nn.ModuleList()
        for i, (kernel_size, stride) in enumerate(layers):
            padding = kernel_size // 2
            conv = norm_fn(
                nn.Conv2d(
                    level_chs[i - 1], level_chs[i], kernel_size, stride, padding=padding
                )
            )
            self.down_convs.append(conv)

        # Decoder (upsampling path)
        self.up_convs = nn.ModuleList()
        for i in reversed(range(len(layers))):
            kernel_size, stride = layers[i]
            padding = kernel_size // 2
            output_padding = stride - 1

            # First upconv doesn't concat, others do (2x channels)
            in_ch = level_chs[i] * (2 if i != len(layers) - 1 else 1)

            conv = norm_fn(
                nn.ConvTranspose2d(
                    in_ch,
                    level_chs[i - 1],
                    kernel_size,
                    stride,
                    padding=padding,
                    output_padding=output_padding,
                )
            )
            self.up_convs.append(conv)

        # Center outputs (from bottleneck)
        self.center_out = norm_fn(nn.Conv2d(level_chs[len(layers) - 1], 1, 1))
        self.center_conv = norm_fn(
            nn.Conv2d(level_chs[len(layers) - 1], level_chs[len(layers) - 1], 1)
        )

        # Final output (after decoder, with skip connection)
        self.final_out = norm_fn(nn.Conv2d(level_chs[-1] * 2, 1, 1))

    def _calc_receptive_field(self, layers: list[tuple[int, int]]) -> int:
        """
        Calculate receptive field size for given layer configuration.

        Same result as https://fomoro.com/research/article/receptive-field-calculator

        Args:
            layers: List of (kernel_size, stride) tuples.

        Returns:
            Receptive field size in pixels.
        """
        rf = 0
        ts = 1
        for i, (k, s) in enumerate(layers):
            if i == 0:
                rf = k
            else:
                rf += (k - 1) * ts
            ts *= s
        return rf

    def _find_archi(
        self, target_patch_size: int, max_layers: int = 9
    ) -> list[tuple[int, int]]:
        """
        Find optimal 3x3 conv architecture for target patch size.

        Uses exhaustive search to find layer configuration that achieves
        receptive field closest to target with minimal layers.

        Args:
            target_patch_size: Desired receptive field size.
            max_layers: Maximum number of layers to consider.

        Returns:
            List of (kernel_size, stride) tuples.
        """
        candidates: dict[int, tuple[int, int, list[tuple[int, int]]]] = {}

        for layers_count in range(1, max_layers + 1):
            # Try all stride combinations
            val = 1 << (layers_count - 1)
            while True:
                val -= 1

                layers = []
                sum_st = 0

                # First layer always stride 2
                layers.append((3, 2))
                sum_st += 2

                # Remaining layers: stride 1 or 2
                for i in range(layers_count - 1):
                    st = 1 + (1 if val & (1 << i) != 0 else 0)
                    layers.append((3, st))
                    sum_st += st

                rf = self._calc_receptive_field(layers)

                # Keep best config for each receptive field
                existing = candidates.get(rf)
                if existing is None:
                    candidates[rf] = (layers_count, sum_st, layers)
                elif layers_count < existing[0] or (
                    layers_count == existing[0] and sum_st > existing[1]
                ):
                    candidates[rf] = (layers_count, sum_st, layers)

                if val == 0:
                    break

        # Find closest to target
        rf_values = sorted(candidates.keys())
        closest_rf = rf_values[np.abs(np.array(rf_values) - target_patch_size).argmin()]

        return candidates[closest_rf][2]

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input image (B, C, H, W).

        Returns:
            Tuple of:
            - center_out: Global discrimination from bottleneck (B, 1, H', W')
            - final_out: Per-patch discrimination at input resolution (B, 1, H, W)
        """
        # Input projection
        x = F.leaky_relu(self.in_conv(x), 0.2)

        # Encoder with skip connections
        encodings: list[torch.Tensor] = []
        for conv in self.down_convs:
            encodings.insert(0, x)  # Save before downsampling
            x = F.leaky_relu(conv(x), 0.2)

        # Center outputs
        center_out = self.center_out(x)
        x = F.leaky_relu(self.center_conv(x), 0.2)

        # Decoder with skip connections
        for up_conv, enc in zip(self.up_convs, encodings, strict=False):
            x = F.leaky_relu(up_conv(x), 0.2)
            # Concatenate with skip connection
            x = torch.cat([enc, x], dim=1)

        # Final output
        final_out = self.final_out(x)

        return center_out, final_out


class MultiScaleDiscriminator(nn.Module):
    """
    Multi-scale discriminator for improved training.

    Applies multiple discriminators at different image scales
    for multi-level adversarial supervision.

    Args:
        in_channels: Number of input channels. Default: 3.
        n_scales: Number of scales. Default: 3.
        base_ch: Base channel count. Default: 64.
        use_spectral_norm: Apply spectral normalization. Default: False.

    Example:
        >>> d = MultiScaleDiscriminator(n_scales=3)
        >>> x = torch.randn(2, 3, 256, 256)
        >>> outputs = d(x)  # List of 3 outputs
    """

    def __init__(
        self,
        in_channels: int = 3,
        n_scales: int = 3,
        base_ch: int = 64,
        use_spectral_norm: bool = False,
    ) -> None:
        super().__init__()

        self.n_scales = n_scales

        self.discriminators = nn.ModuleList(
            [
                PatchDiscriminator(
                    in_channels=in_channels,
                    base_ch=base_ch,
                    use_spectral_norm=use_spectral_norm,
                )
                for _ in range(n_scales)
            ]
        )

        self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Forward pass at multiple scales.

        Args:
            x: Input image (B, C, H, W).

        Returns:
            List of discrimination outputs at each scale.
        """
        outputs = []

        for i, disc in enumerate(self.discriminators):
            outputs.append(disc(x))
            if i < self.n_scales - 1:
                x = self.downsample(x)

        return outputs
