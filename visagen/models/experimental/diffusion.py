"""
Diffusion AutoEncoder for ultra-realistic face texture generation.

Architecture:
- Structure Branch: ConvNeXtEncoder (existing) -> Structure preservation
- Texture Branch: SD VAE Encoder -> Texture details
- Fusion: Cross-attention for feature combination
- Decoder: Progressive upsampling with texture injection

This module provides a hybrid encoder that combines the structural
preservation capabilities of ConvNeXt with the ultra-realistic
texture generation of Stable Diffusion's VAE.

Example:
    >>> from visagen.models.experimental.diffusion import DiffusionAutoEncoder
    >>> model = DiffusionAutoEncoder(image_size=256)
    >>> output = model(input_image)

Reference:
    Preechakul et al. "Diffusion Autoencoders" (CVPR 2022)
    https://diff-ae.github.io/
"""

from __future__ import annotations

from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from visagen.models.encoders.convnext import ConvNeXtEncoder
from visagen.models.layers.adain import AdaptiveInstanceNorm2d
from visagen.models.layers.attention import CBAM


class TextureEncoder(nn.Module):
    """
    SD VAE-based texture encoder.

    Extracts high-frequency skin texture details using the
    Stable Diffusion VAE encoder. The VAE is frozen during
    training to preserve its learned texture representations.

    Args:
        in_channels: Number of input channels (default: 3).
        latent_dim: Dimension of output features (default: 512).
        pretrained: Whether to load pretrained VAE weights (default: True).

    Note:
        Requires `diffusers` package: pip install diffusers
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = 512,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.pretrained = pretrained

        # Lazy load diffusers to avoid import errors
        self._vae: Any | None = None

        # Projection to match structure encoder dimensions
        # VAE latent is 4 channels, project to latent_dim
        self.projection = nn.Sequential(
            nn.Conv2d(4, latent_dim, 1),
            nn.SiLU(),
            nn.Conv2d(latent_dim, latent_dim, 3, padding=1),
        )

    def _load_vae(self) -> nn.Module:
        """Lazy load the VAE model."""
        if self._vae is None:
            try:
                from diffusers import AutoencoderKL

                self._vae = AutoencoderKL.from_pretrained(
                    "stabilityai/sd-vae-ft-mse",
                    torch_dtype=torch.float32,
                )
                assert self._vae is not None
                self._vae.requires_grad_(False)  # Freeze VAE
            except ImportError:
                raise ImportError(
                    "diffusers package is required for TextureEncoder. "
                    "Install with: pip install 'visagen[experimental]'"
                )
        assert self._vae is not None
        return cast(nn.Module, self._vae)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract texture features from input image.

        Args:
            x: Input image (B, 3, H, W) normalized to [-1, 1].

        Returns:
            Texture latent (B, latent_dim, H/8, W/8).
        """
        vae = cast(Any, self._load_vae().to(x.device))

        with torch.no_grad():
            # VAE encode: (B, 3, H, W) -> (B, 4, H/8, W/8)
            latent = vae.encode(x).latent_dist.sample()
            latent = latent * vae.config.scaling_factor

        # Project to match structure encoder
        return cast(torch.Tensor, self.projection(latent))


class TextureEncoderLite(nn.Module):
    """
    Lightweight texture encoder without pretrained VAE.

    This is a fallback encoder that doesn't require the diffusers
    package. It uses a simple convolutional encoder to extract
    texture features, but won't achieve the same quality as the
    full TextureEncoder.

    Args:
        in_channels: Number of input channels (default: 3).
        latent_dim: Dimension of output features (default: 512).
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = 512,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim

        # Simple encoder: 8x downsample
        self.encoder = nn.Sequential(
            # 256 -> 128
            nn.Conv2d(in_channels, 64, 4, stride=2, padding=1),
            nn.SiLU(),
            # 128 -> 64
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            # 64 -> 32
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.GroupNorm(8, 256),
            nn.SiLU(),
            # Project to latent_dim
            nn.Conv2d(256, latent_dim, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract texture features from input image.

        Args:
            x: Input image (B, 3, H, W).

        Returns:
            Texture latent (B, latent_dim, H/8, W/8).
        """
        return cast(torch.Tensor, self.encoder(x))


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention fusion between structure and texture features.

    Structure features query texture features for detail injection.
    This allows the model to selectively incorporate texture details
    while preserving structural information.

    Args:
        dim: Feature dimension.
        num_heads: Number of attention heads (default: 8).
        dropout: Dropout rate (default: 0.1).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(
        self,
        structure: torch.Tensor,
        texture: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fuse structure and texture features via cross-attention.

        Args:
            structure: (B, C, H, W) structure features (query).
            texture: (B, C, H, W) texture features (key/value).

        Returns:
            Fused features (B, C, H, W).
        """
        B, C, H, W = structure.shape

        # Reshape to sequence: (B, H*W, C)
        s = structure.flatten(2).transpose(1, 2)
        t = texture.flatten(2).transpose(1, 2)

        # Layer norm
        s = self.norm1(s)
        t = self.norm2(t)

        # Compute attention
        q = self.q_proj(s).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(t).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(t).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, -1, C)
        out = self.out_proj(out)

        # Reshape back to spatial
        out = out.transpose(1, 2).reshape(B, C, H, W)

        # Residual connection
        return cast(torch.Tensor, structure + out)


class DiffusionDecoderBlock(nn.Module):
    """
    Single decoder block with texture injection.

    Each block upsamples by 2x, applies convolutions with CBAM
    attention, and injects texture via AdaIN.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        use_attention: Whether to use CBAM attention (default: True).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_attention: bool = True,
    ) -> None:
        super().__init__()

        self.upsample = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=False
        )

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.attention = CBAM(out_channels) if use_attention else nn.Identity()

        # Texture injection via AdaIN
        self.adain = AdaptiveInstanceNorm2d(out_channels, out_channels)

    def forward(
        self,
        x: torch.Tensor,
        style: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass with optional texture injection.

        Args:
            x: Input features (B, C_in, H, W).
            style: Style vector for AdaIN (B, C_out).

        Returns:
            Output features (B, C_out, H*2, W*2).
        """
        x = self.upsample(x)
        x = F.silu(self.norm1(self.conv1(x)))
        x = self.conv2(x)
        x = self.attention(x)

        # Apply AdaIN if style is provided
        if style is not None:
            x = self.adain(x, style)

        return F.silu(x)


class DiffusionDecoder(nn.Module):
    """
    Progressive decoder with texture injection at each scale.

    Uses AdaIN for style modulation, allowing texture details
    to be injected at multiple resolutions.

    Args:
        latent_channels: Number of input latent channels (default: 512).
        dims: Channel dimensions for each block (default: [512, 256, 128, 64]).
        out_channels: Number of output channels (default: 3).
        use_attention: Whether to use CBAM attention (default: True).
    """

    def __init__(
        self,
        latent_channels: int = 512,
        dims: list[int] | None = None,
        out_channels: int = 3,
        use_attention: bool = True,
    ) -> None:
        super().__init__()
        dims = dims or [512, 256, 128, 64]

        self.blocks = nn.ModuleList()

        in_ch = latent_channels
        for out_ch in dims:
            self.blocks.append(
                DiffusionDecoderBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    use_attention=use_attention,
                )
            )
            in_ch = out_ch

        # Final output: 4x upsample to match input resolution
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False),
            nn.Conv2d(dims[-1], dims[-1], 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(dims[-1], out_channels, 1),
            nn.Tanh(),
        )

    def forward(
        self,
        latent: torch.Tensor,
        texture_styles: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """
        Decode latent with texture injection.

        Args:
            latent: Fused latent (B, latent_channels, H, W).
            texture_styles: List of style vectors for each block.

        Returns:
            Generated image (B, out_channels, H_out, W_out).
        """
        x = latent

        for i, block in enumerate(self.blocks):
            style = None
            if texture_styles is not None and i < len(texture_styles):
                style = texture_styles[i]
            x = block(x, style)

        return cast(torch.Tensor, self.final(x))


class DiffusionAutoEncoder(nn.Module):
    """
    Hybrid encoder combining ConvNeXt structure preservation
    with Stable Diffusion texture generation.

    Key Innovation:
    - Structure branch preserves facial geometry (identity)
    - Texture branch captures skin pores, wrinkles, hair detail
    - Cross-attention fusion allows controlled detail injection
    - Progressive decoder with AdaIN texture modulation

    Compatible with existing DFLModule training pipeline.

    Args:
        image_size: Input image size (default: 256).
        structure_dims: ConvNeXt encoder dimensions.
        structure_depths: ConvNeXt encoder depths.
        texture_dim: Texture encoder output dimension (default: 512).
        decoder_dims: Decoder block dimensions.
        use_pretrained_vae: Whether to use pretrained SD VAE (default: True).
        use_attention: Whether to use CBAM in decoder (default: True).

    Example:
        >>> model = DiffusionAutoEncoder(image_size=256)
        >>> output = model(torch.randn(1, 3, 256, 256))
        >>> print(output.shape)  # torch.Size([1, 3, 256, 256])
    """

    def __init__(
        self,
        image_size: int = 256,
        structure_dims: list[int] | None = None,
        structure_depths: list[int] | None = None,
        texture_dim: int = 512,
        decoder_dims: list[int] | None = None,
        use_pretrained_vae: bool = True,
        use_attention: bool = True,
    ) -> None:
        super().__init__()
        structure_dims = structure_dims or [64, 128, 256, 512]
        structure_depths = structure_depths or [2, 2, 4, 2]
        decoder_dims = decoder_dims or [512, 256, 128, 64]

        self.image_size = image_size

        # Structure encoder (existing ConvNeXt)
        self.structure_encoder = ConvNeXtEncoder(
            in_channels=3,
            dims=structure_dims,
            depths=structure_depths,
        )

        # Texture encoder (SD VAE-based or lite)
        self.texture_encoder: nn.Module
        if use_pretrained_vae:
            self.texture_encoder = TextureEncoder(
                latent_dim=texture_dim,
                pretrained=True,
            )
        else:
            self.texture_encoder = TextureEncoderLite(
                latent_dim=texture_dim,
            )

        # Cross-attention fusion
        self.fusion = CrossAttentionFusion(dim=structure_dims[-1])

        # Progressive decoder with texture injection
        self.decoder = DiffusionDecoder(
            latent_channels=structure_dims[-1],
            dims=decoder_dims,
            use_attention=use_attention,
        )

        # Multi-scale texture style extractors
        self.texture_projectors = nn.ModuleList(
            [nn.Conv2d(texture_dim, dim, 1) for dim in decoder_dims]
        )

    def forward(
        self,
        x: torch.Tensor,
        landmarks: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass through the diffusion autoencoder.

        Args:
            x: Input face image (B, 3, H, W) normalized to [-1, 1].
            landmarks: Optional face landmarks (B, 68, 2) for guidance.

        Returns:
            Reconstructed face image (B, 3, H, W).
        """
        # Structure encoding
        structure_features, structure_latent = self.structure_encoder(x)

        # Texture encoding
        texture_latent = self.texture_encoder(x)

        # Resize texture to match structure latent
        if texture_latent.shape[2:] != structure_latent.shape[2:]:
            texture_latent = F.interpolate(
                texture_latent,
                size=structure_latent.shape[2:],
                mode="bilinear",
                align_corners=False,
            )

        # Cross-attention fusion
        fused_latent = self.fusion(structure_latent, texture_latent)

        # Multi-scale texture styles for decoder
        texture_styles = []
        for i, proj in enumerate(self.texture_projectors):
            # Project texture and compute global style
            tex_proj = proj(
                F.interpolate(
                    texture_latent,
                    scale_factor=2**i,
                    mode="bilinear",
                    align_corners=False,
                )
            )
            style = tex_proj.mean(dim=[2, 3])  # Global average pool
            texture_styles.append(style)

        # Progressive decoding with texture injection
        output = self.decoder(fused_latent, texture_styles)

        return cast(torch.Tensor, output)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode image to latent space for manipulation.

        Args:
            x: Input image (B, 3, H, W).

        Returns:
            Tuple of (structure_latent, texture_latent).
        """
        _, structure_latent = self.structure_encoder(x)
        texture_latent = self.texture_encoder(x)
        return structure_latent, texture_latent

    def decode(
        self,
        structure_latent: torch.Tensor,
        texture_latent: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decode from latent space.

        Args:
            structure_latent: Structure features from encoder.
            texture_latent: Texture features from encoder.

        Returns:
            Reconstructed image (B, 3, H, W).
        """
        # Resize texture to match structure
        if texture_latent.shape[2:] != structure_latent.shape[2:]:
            texture_latent = F.interpolate(
                texture_latent,
                size=structure_latent.shape[2:],
                mode="bilinear",
                align_corners=False,
            )

        # Fuse and decode
        fused = self.fusion(structure_latent, texture_latent)

        texture_styles = []
        for i, proj in enumerate(self.texture_projectors):
            tex_proj = proj(
                F.interpolate(
                    texture_latent,
                    scale_factor=2**i,
                    mode="bilinear",
                    align_corners=False,
                )
            )
            style = tex_proj.mean(dim=[2, 3])
            texture_styles.append(style)

        return cast(torch.Tensor, self.decoder(fused, texture_styles))
