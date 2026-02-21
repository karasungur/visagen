"""
ConvNeXt Encoder for Visagen.

Implementation based on:
"A ConvNet for the 2020s" (Liu et al., CVPR 2022)

ConvNeXt modernizes ResNet with techniques from Vision Transformers:
- Depthwise separable convolutions
- Inverted bottleneck (expand -> process -> contract)
- LayerNorm instead of BatchNorm
- GELU activation
- Stochastic depth (DropPath)
"""

import torch
import torch.nn as nn


class DropPath(nn.Module):
    """
    Stochastic Depth (Drop Path) regularization.

    Randomly drops entire residual branches during training.

    Args:
        drop_prob: Probability of dropping the path. Default: 0.0.
    """

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1 - self.drop_prob
        # Shape: (batch_size, 1, 1, 1) for broadcasting
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # Binarize
        return x.div(keep_prob) * random_tensor


class LayerNorm2d(nn.Module):
    """
    LayerNorm for 2D feature maps (channels-first format).

    Args:
        num_channels: Number of channels.
        eps: Small constant for numerical stability. Default: 1e-6.
    """

    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt Block.

    Architecture:
    - 7x7 depthwise conv
    - LayerNorm
    - 1x1 conv (expand to 4x channels)
    - GELU activation
    - 1x1 conv (contract back)
    - DropPath (stochastic depth)

    Args:
        dim: Number of input/output channels.
        drop_path: Drop path rate. Default: 0.0.
        layer_scale_init: Initial value for layer scale. Default: 1e-6.
    """

    def __init__(
        self,
        dim: int,
        drop_path: float = 0.0,
        layer_scale_init: float = 1e-6,
    ) -> None:
        super().__init__()

        # Depthwise convolution
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)

        # Normalization
        self.norm = LayerNorm2d(dim)

        # Pointwise convolutions (inverted bottleneck)
        self.pwconv1 = nn.Conv2d(dim, dim * 4, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(dim * 4, dim, kernel_size=1)

        # Layer scale (learnable per-channel scaling)
        self.gamma = nn.Parameter(
            layer_scale_init * torch.ones(dim), requires_grad=True
        )

        # Stochastic depth
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        # Depthwise conv
        x = self.dwconv(x)
        x = self.norm(x)

        # Inverted bottleneck
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        # Layer scale: (C,) -> (1, C, 1, 1)
        x = self.gamma[:, None, None] * x

        # Residual + drop path
        x = residual + self.drop_path(x)

        return x


class DownsampleLayer(nn.Module):
    """
    Downsample layer between ConvNeXt stages.

    Uses LayerNorm + 2x2 stride-2 convolution.

    Args:
        in_dim: Input channels.
        out_dim: Output channels.
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.norm = LayerNorm2d(in_dim)
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.conv(x)
        return x


class ConvNeXtEncoder(nn.Module):
    """
    ConvNeXt Encoder for feature extraction.

    Extracts multi-scale features from input images using ConvNeXt blocks.
    Returns features at each stage for skip connections.

    Args:
        in_channels: Number of input channels. Default: 3.
        dims: Channel dimensions for each stage. Default: [64, 128, 256, 512].
        depths: Number of blocks per stage. Default: [2, 2, 4, 2].
        drop_path_rate: Maximum drop path rate. Default: 0.1.

    Example:
        >>> encoder = ConvNeXtEncoder()
        >>> x = torch.randn(2, 3, 256, 256)
        >>> features, latent = encoder(x)
        >>> len(features)
        4
        >>> latent.shape
        torch.Size([2, 512, 16, 16])
    """

    def __init__(
        self,
        in_channels: int = 3,
        dims: list[int] | None = None,
        depths: list[int] | None = None,
        drop_path_rate: float = 0.1,
    ) -> None:
        super().__init__()

        if dims is None:
            dims = [64, 128, 256, 512]
        if depths is None:
            depths = [2, 2, 4, 2]

        assert len(dims) == len(depths), "dims and depths must have same length"

        self.num_stages = len(dims)
        self.dims = dims

        # Stem: 4x4 conv with stride 4 (like patchify)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], kernel_size=4, stride=4),
            LayerNorm2d(dims[0]),
        )

        # Stochastic depth decay rule
        total_blocks = sum(depths)
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]

        # Build stages
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        cur_block = 0
        for i in range(self.num_stages):
            # Stage blocks
            stage = nn.Sequential(
                *[
                    ConvNeXtBlock(dims[i], drop_path=dp_rates[cur_block + j])
                    for j in range(depths[i])
                ]
            )
            self.stages.append(stage)
            cur_block += depths[i]

            # Downsample (except for last stage)
            if i < self.num_stages - 1:
                downsample = DownsampleLayer(dims[i], dims[i + 1])
                self.downsamples.append(downsample)

    def forward(self, x: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
        """
        Extract multi-scale features.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            features: List of feature maps at each stage (for skip connections).
            latent: Final encoded features.
        """
        features = []

        # Stem
        x = self.stem(x)

        # Stages with downsampling
        for i in range(self.num_stages):
            x = self.stages[i](x)
            features.append(x)

            if i < self.num_stages - 1:
                x = self.downsamples[i](x)

        return features, x
