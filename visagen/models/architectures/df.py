"""
DF (Direct Face) Architecture for Visagen.

Port of DeepFaceLab's DF architecture to PyTorch.
This is the standard autoencoder architecture used in SAEHD model.

Architecture Options:
    -t: Transformer mode (deeper encoder/decoder with residual blocks)
    -d: Double resolution output (2x upscale at the end)
    -u: Use pixel normalization in encoder
    -c: Use cosine activation instead of LeakyReLU

Example:
    >>> archi = DFArchi(resolution=256, e_ch=64, d_ch=64, ae_dims=256)
    >>> encoder = archi.Encoder()
    >>> inter = archi.Inter(ae_out_ch=256)
    >>> decoder = archi.Decoder()
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PixelNorm(nn.Module):
    """Pixel-wise feature normalization."""

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x / torch.sqrt(torch.mean(x**2, dim=1, keepdim=True) + self.eps)


class Downscale(nn.Module):
    """Downscaling block with conv + activation."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 5,
        use_cos_act: bool = False,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_ch, out_ch, kernel_size=kernel_size, stride=2, padding=kernel_size // 2
        )
        self.use_cos_act = use_cos_act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.use_cos_act:
            x = x * torch.cos(x)
        else:
            x = F.leaky_relu(x, 0.1)
        return x


class Upscale(nn.Module):
    """Upscaling block using pixel shuffle."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        use_cos_act: bool = False,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_ch, out_ch * 4, kernel_size=kernel_size, padding=kernel_size // 2
        )
        self.use_cos_act = use_cos_act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.use_cos_act:
            x = x * torch.cos(x)
        else:
            x = F.leaky_relu(x, 0.1)
        x = F.pixel_shuffle(x, 2)
        return x


class ResidualBlock(nn.Module):
    """Residual block with two convolutions."""

    def __init__(
        self,
        ch: int,
        kernel_size: int = 3,
        use_cos_act: bool = False,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            ch, ch, kernel_size=kernel_size, padding=kernel_size // 2
        )
        self.conv2 = nn.Conv2d(
            ch, ch, kernel_size=kernel_size, padding=kernel_size // 2
        )
        self.use_cos_act = use_cos_act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        if self.use_cos_act:
            x = self.conv1(x)
            x = x * torch.cos(x)
            x = self.conv2(x)
            x = (residual + x) * torch.cos(residual + x)
        else:
            x = self.conv1(x)
            x = F.leaky_relu(x, 0.2)
            x = self.conv2(x)
            x = F.leaky_relu(residual + x, 0.2)
        return x


class DFArchi(nn.Module):
    """
    DF (Direct Face) Architecture.

    This is the standard SAEHD architecture from DeepFaceLab.
    It consists of:
    - Shared Encoder
    - Inter (bottleneck) - one per identity (src/dst)
    - Decoder - one per identity (src/dst)

    Args:
        resolution: Input/output resolution. Default: 256.
        in_ch: Input channels. Default: 3.
        e_ch: Encoder base channels. Default: 64.
        d_ch: Decoder base channels. Default: 64.
        d_mask_ch: Decoder mask channels. Default: 22.
        ae_dims: Autoencoder bottleneck dimensions. Default: 256.
        opts: Architecture options string. Default: "".
            't' - Transformer mode (deeper network)
            'd' - Double resolution output
            'u' - Use pixel normalization
            'c' - Use cosine activation
    """

    def __init__(
        self,
        resolution: int = 256,
        in_ch: int = 3,
        e_ch: int = 64,
        d_ch: int = 64,
        d_mask_ch: int = 22,
        ae_dims: int = 256,
        opts: str = "",
    ) -> None:
        super().__init__()

        self.resolution = resolution
        self.in_ch = in_ch
        self.e_ch = e_ch
        self.d_ch = d_ch
        self.d_mask_ch = d_mask_ch
        self.ae_dims = ae_dims
        self.opts = opts

        self.use_transformer = "t" in opts
        self.use_double = "d" in opts
        self.use_pixel_norm = "u" in opts
        self.use_cos_act = "c" in opts

        # Calculate lowest dense resolution
        if self.use_double:
            self.lowest_dense_res = resolution // 32
        else:
            self.lowest_dense_res = resolution // 16

        # Calculate encoder output channels
        self.encoder_out_ch = e_ch * 8

        # Calculate encoder output resolution
        if self.use_transformer:
            self.encoder_out_res = resolution // 32
        else:
            self.encoder_out_res = resolution // 16

        # Flatten size for Inter input
        self.flatten_size = (
            self.encoder_out_ch * self.encoder_out_res * self.encoder_out_res
        )

    def Encoder(self) -> nn.Module:
        """Create encoder module."""
        return _DFEncoder(
            in_ch=self.in_ch,
            e_ch=self.e_ch,
            use_transformer=self.use_transformer,
            use_pixel_norm=self.use_pixel_norm,
            use_cos_act=self.use_cos_act,
        )

    def Inter(self, ae_out_ch: int = 256) -> nn.Module:
        """
        Create inter (bottleneck) module.

        Args:
            ae_out_ch: Output channels from inter. Default: 256.
        """
        return _DFInter(
            in_features=self.flatten_size,
            ae_dims=self.ae_dims,
            ae_out_ch=ae_out_ch,
            lowest_dense_res=self.lowest_dense_res,
            use_transformer=self.use_transformer,
            use_cos_act=self.use_cos_act,
        )

    def Decoder(self, ae_out_ch: int = 256) -> nn.Module:
        """
        Create decoder module.

        Args:
            ae_out_ch: Input channels from inter. Default: 256.
        """
        return _DFDecoder(
            in_ch=ae_out_ch,
            d_ch=self.d_ch,
            d_mask_ch=self.d_mask_ch,
            use_transformer=self.use_transformer,
            use_double=self.use_double,
            use_cos_act=self.use_cos_act,
        )


class _DFEncoder(nn.Module):
    """DF Encoder implementation."""

    def __init__(
        self,
        in_ch: int,
        e_ch: int,
        use_transformer: bool = False,
        use_pixel_norm: bool = False,
        use_cos_act: bool = False,
    ) -> None:
        super().__init__()

        self.use_transformer = use_transformer
        self.use_pixel_norm = use_pixel_norm
        self.use_cos_act = use_cos_act

        if use_transformer:
            # Deeper encoder with residual blocks
            self.down1 = Downscale(in_ch, e_ch, kernel_size=5, use_cos_act=use_cos_act)
            self.res1 = ResidualBlock(e_ch, use_cos_act=use_cos_act)
            self.down2 = Downscale(
                e_ch, e_ch * 2, kernel_size=5, use_cos_act=use_cos_act
            )
            self.down3 = Downscale(
                e_ch * 2, e_ch * 4, kernel_size=5, use_cos_act=use_cos_act
            )
            self.down4 = Downscale(
                e_ch * 4, e_ch * 8, kernel_size=5, use_cos_act=use_cos_act
            )
            self.down5 = Downscale(
                e_ch * 8, e_ch * 8, kernel_size=5, use_cos_act=use_cos_act
            )
            self.res5 = ResidualBlock(e_ch * 8, use_cos_act=use_cos_act)
        else:
            # Standard 4-stage downscaling
            self.downs = nn.ModuleList()
            last_ch = in_ch
            for i in range(4):
                cur_ch = e_ch * min(2**i, 8)
                self.downs.append(
                    Downscale(last_ch, cur_ch, kernel_size=5, use_cos_act=use_cos_act)
                )
                last_ch = cur_ch

        if use_pixel_norm:
            self.pixel_norm = PixelNorm()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_transformer:
            x = self.down1(x)
            x = self.res1(x)
            x = self.down2(x)
            x = self.down3(x)
            x = self.down4(x)
            x = self.down5(x)
            x = self.res5(x)
        else:
            for down in self.downs:
                x = down(x)

        # Flatten
        x = x.view(x.size(0), -1)

        if self.use_pixel_norm:
            x = self.pixel_norm(x)

        return x


class _DFInter(nn.Module):
    """DF Inter (bottleneck) implementation."""

    def __init__(
        self,
        in_features: int,
        ae_dims: int,
        ae_out_ch: int,
        lowest_dense_res: int,
        use_transformer: bool = False,
        use_cos_act: bool = False,
    ) -> None:
        super().__init__()

        self.ae_out_ch = ae_out_ch
        self.lowest_dense_res = lowest_dense_res
        self.use_transformer = use_transformer
        self.use_cos_act = use_cos_act

        self.dense1 = nn.Linear(in_features, ae_dims)
        self.dense2 = nn.Linear(
            ae_dims, lowest_dense_res * lowest_dense_res * ae_out_ch
        )

        if not use_transformer:
            self.upscale1 = Upscale(ae_out_ch, ae_out_ch, use_cos_act=use_cos_act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dense1(x)
        x = self.dense2(x)

        # Reshape to 4D
        x = x.view(
            x.size(0), self.ae_out_ch, self.lowest_dense_res, self.lowest_dense_res
        )

        if not self.use_transformer:
            x = self.upscale1(x)

        return x


class _DFDecoder(nn.Module):
    """DF Decoder implementation."""

    def __init__(
        self,
        in_ch: int,
        d_ch: int,
        d_mask_ch: int,
        use_transformer: bool = False,
        use_double: bool = False,
        use_cos_act: bool = False,
    ) -> None:
        super().__init__()

        self.use_transformer = use_transformer
        self.use_double = use_double
        self.use_cos_act = use_cos_act

        if use_transformer:
            # Deeper decoder with 4 upscale stages
            self.upscale0 = Upscale(
                in_ch, d_ch * 8, kernel_size=3, use_cos_act=use_cos_act
            )
            self.upscale1 = Upscale(
                d_ch * 8, d_ch * 8, kernel_size=3, use_cos_act=use_cos_act
            )
            self.upscale2 = Upscale(
                d_ch * 8, d_ch * 4, kernel_size=3, use_cos_act=use_cos_act
            )
            self.upscale3 = Upscale(
                d_ch * 4, d_ch * 2, kernel_size=3, use_cos_act=use_cos_act
            )
            self.res0 = ResidualBlock(d_ch * 8, kernel_size=3, use_cos_act=use_cos_act)
            self.res1 = ResidualBlock(d_ch * 8, kernel_size=3, use_cos_act=use_cos_act)
            self.res2 = ResidualBlock(d_ch * 4, kernel_size=3, use_cos_act=use_cos_act)
            self.res3 = ResidualBlock(d_ch * 2, kernel_size=3, use_cos_act=use_cos_act)

            # Mask decoder
            self.upscalem0 = Upscale(
                in_ch, d_mask_ch * 8, kernel_size=3, use_cos_act=use_cos_act
            )
            self.upscalem1 = Upscale(
                d_mask_ch * 8, d_mask_ch * 8, kernel_size=3, use_cos_act=use_cos_act
            )
            self.upscalem2 = Upscale(
                d_mask_ch * 8, d_mask_ch * 4, kernel_size=3, use_cos_act=use_cos_act
            )
            self.upscalem3 = Upscale(
                d_mask_ch * 4, d_mask_ch * 2, kernel_size=3, use_cos_act=use_cos_act
            )
        else:
            # Standard 3-stage decoder
            self.upscale0 = Upscale(
                in_ch, d_ch * 8, kernel_size=3, use_cos_act=use_cos_act
            )
            self.upscale1 = Upscale(
                d_ch * 8, d_ch * 4, kernel_size=3, use_cos_act=use_cos_act
            )
            self.upscale2 = Upscale(
                d_ch * 4, d_ch * 2, kernel_size=3, use_cos_act=use_cos_act
            )
            self.res0 = ResidualBlock(d_ch * 8, kernel_size=3, use_cos_act=use_cos_act)
            self.res1 = ResidualBlock(d_ch * 4, kernel_size=3, use_cos_act=use_cos_act)
            self.res2 = ResidualBlock(d_ch * 2, kernel_size=3, use_cos_act=use_cos_act)

            # Mask decoder
            self.upscalem0 = Upscale(
                in_ch, d_mask_ch * 8, kernel_size=3, use_cos_act=use_cos_act
            )
            self.upscalem1 = Upscale(
                d_mask_ch * 8, d_mask_ch * 4, kernel_size=3, use_cos_act=use_cos_act
            )
            self.upscalem2 = Upscale(
                d_mask_ch * 4, d_mask_ch * 2, kernel_size=3, use_cos_act=use_cos_act
            )

        # Output convolutions
        self.out_conv = nn.Conv2d(d_ch * 2, 3, kernel_size=1)

        if use_double:
            # Additional convs for 2x upscale via pixel shuffle
            self.out_conv1 = nn.Conv2d(d_ch * 2, 3, kernel_size=3, padding=1)
            self.out_conv2 = nn.Conv2d(d_ch * 2, 3, kernel_size=3, padding=1)
            self.out_conv3 = nn.Conv2d(d_ch * 2, 3, kernel_size=3, padding=1)

            if use_transformer:
                self.upscalem4 = Upscale(
                    d_mask_ch * 2, d_mask_ch, kernel_size=3, use_cos_act=use_cos_act
                )
            else:
                self.upscalem3 = Upscale(
                    d_mask_ch * 2, d_mask_ch, kernel_size=3, use_cos_act=use_cos_act
                )

            self.out_convm = nn.Conv2d(d_mask_ch, 1, kernel_size=1)
        else:
            self.out_convm = nn.Conv2d(d_mask_ch * 2, 1, kernel_size=1)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Decode latent to image and mask.

        Args:
            z: Latent tensor from Inter module.

        Returns:
            Tuple of (image, mask) tensors.
        """
        # Image decoder
        x = self.upscale0(z)
        x = self.res0(x)
        x = self.upscale1(x)
        x = self.res1(x)
        x = self.upscale2(x)
        x = self.res2(x)

        if self.use_transformer:
            x = self.upscale3(x)
            x = self.res3(x)

        if self.use_double:
            # 2x upscale via pixel shuffle of 4 conv outputs
            out = torch.cat(
                [
                    self.out_conv(x),
                    self.out_conv1(x),
                    self.out_conv2(x),
                    self.out_conv3(x),
                ],
                dim=1,
            )
            x = torch.sigmoid(F.pixel_shuffle(out, 2))
        else:
            x = torch.sigmoid(self.out_conv(x))

        # Mask decoder
        m = self.upscalem0(z)
        m = self.upscalem1(m)
        m = self.upscalem2(m)

        if self.use_transformer:
            m = self.upscalem3(m)
            if self.use_double:
                m = self.upscalem4(m)
        else:
            if self.use_double:
                m = self.upscalem3(m)

        m = torch.sigmoid(self.out_convm(m))

        return x, m
