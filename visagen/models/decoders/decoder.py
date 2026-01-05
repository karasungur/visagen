"""
Decoder for Visagen.

Reconstructs images from encoded features using upsampling blocks
with skip connections and optional CBAM attention.

Uses SiLU (Swish) activation throughout as per AGENTS.md specification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from visagen.models.layers.attention import CBAM


class DecoderBlock(nn.Module):
    """
    Decoder upsampling block.

    Architecture:
    - Upsample 2x (bilinear or transposed conv)
    - Concatenate skip connection
    - Conv -> Norm -> SiLU
    - Conv -> Norm -> SiLU
    - Optional CBAM attention

    Args:
        in_channels: Input channels (from previous decoder layer).
        skip_channels: Channels from skip connection.
        out_channels: Output channels.
        use_attention: Whether to apply CBAM attention. Default: True.
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        use_attention: bool = True,
    ) -> None:
        super().__init__()

        # Upsampling
        self.upsample = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=False
        )

        # After concat: in_channels + skip_channels
        concat_channels = in_channels + skip_channels

        # First conv block
        self.conv1 = nn.Conv2d(concat_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.InstanceNorm2d(out_channels, affine=True)

        # Second conv block
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.InstanceNorm2d(out_channels, affine=True)

        # Activation (Swish/SiLU as per spec)
        self.act = nn.SiLU(inplace=True)

        # Optional attention
        self.attention = CBAM(out_channels) if use_attention else nn.Identity()

    def forward(
        self, x: torch.Tensor, skip: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Upsample and combine with skip connection.

        Args:
            x: Input tensor from previous decoder layer.
            skip: Skip connection tensor from encoder.

        Returns:
            Upsampled and processed tensor.
        """
        # Upsample
        x = self.upsample(x)

        # Concatenate skip connection if provided
        if skip is not None:
            # Handle size mismatch if any
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(
                    x, size=skip.shape[2:], mode="bilinear", align_corners=False
                )
            x = torch.cat([x, skip], dim=1)

        # Conv blocks
        x = self.act(self.norm1(self.conv1(x)))
        x = self.act(self.norm2(self.conv2(x)))

        # Attention
        x = self.attention(x)

        return x


class Decoder(nn.Module):
    """
    Decoder network for image reconstruction.

    Takes encoded features and skip connections from encoder,
    progressively upsamples to reconstruct the output image.

    Args:
        latent_channels: Channels in the latent/encoded features. Default: 512.
        dims: Channel dimensions for each decoder stage. Default: [512, 256, 128, 64].
        skip_dims: Channel dimensions of skip connections. Default: [256, 128, 64, 64].
        out_channels: Number of output channels. Default: 3.
        use_attention: Whether to use CBAM attention. Default: True.

    Example:
        >>> decoder = Decoder()
        >>> latent = torch.randn(2, 512, 16, 16)
        >>> skips = [torch.randn(2, d, s, s) for d, s in zip([256, 128, 64, 64], [16, 32, 64, 64])]
        >>> out = decoder(latent, skips[::-1])  # Reverse order for decoder
        >>> out.shape
        torch.Size([2, 3, 256, 256])
    """

    def __init__(
        self,
        latent_channels: int = 512,
        dims: list[int] = None,
        skip_dims: list[int] = None,
        out_channels: int = 3,
        use_attention: bool = True,
    ) -> None:
        super().__init__()

        if dims is None:
            dims = [512, 256, 128, 64]
        if skip_dims is None:
            skip_dims = [256, 128, 64, 64]

        self.num_stages = len(dims)

        # Build decoder blocks
        self.blocks = nn.ModuleList()

        in_ch = latent_channels
        for i in range(self.num_stages):
            skip_ch = skip_dims[i] if i < len(skip_dims) else 0
            out_ch = dims[i]

            block = DecoderBlock(
                in_channels=in_ch,
                skip_channels=skip_ch,
                out_channels=out_ch,
                use_attention=use_attention,
            )
            self.blocks.append(block)
            in_ch = out_ch

        # Final output layer
        self.final_upsample = nn.Upsample(
            scale_factor=4, mode="bilinear", align_corners=False
        )
        self.final_conv = nn.Sequential(
            nn.Conv2d(dims[-1], dims[-1], kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(dims[-1], out_channels, kernel_size=1),
            nn.Tanh(),  # Output in [-1, 1] range
        )

    def forward(
        self,
        x: torch.Tensor,
        skip_features: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """
        Decode latent features to image.

        Args:
            x: Latent features from encoder.
            skip_features: List of skip connection features (in reverse order,
                          i.e., from deepest to shallowest).

        Returns:
            Reconstructed image tensor of shape (B, out_channels, H, W).
        """
        if skip_features is None:
            skip_features = [None] * self.num_stages

        # Decoder blocks
        for i, block in enumerate(self.blocks):
            skip = skip_features[i] if i < len(skip_features) else None
            x = block(x, skip)

        # Final upsampling and output
        x = self.final_upsample(x)
        x = self.final_conv(x)

        return x
