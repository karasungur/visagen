"""
LIAE (Lightweight Inter-AB-B Encoder) Architecture for Visagen.

LIAE architecture for Visagen.
LIAE differs from DF by using a shared decoder with InterAB/InterB code concatenation.

Key Differences from DF:
    - DF: Separate decoders for src/dst
    - LIAE: Single shared decoder, InterAB + InterB concatenation

Architecture Options:
    -t: Transformer mode (deeper encoder/decoder with residual blocks)
    -d: Double resolution output (2x upscale at the end)
    -u: Use pixel normalization in encoder
    -c: Use cosine activation instead of LeakyReLU

Example:
    >>> archi = LIAEArchi(resolution=256, e_ch=64, d_ch=64, ae_dims=256)
    >>> encoder = archi.Encoder()
    >>> inter_AB = archi.InterAB()
    >>> inter_B = archi.InterB()
    >>> decoder = archi.Decoder()
    >>>
    >>> # Forward pass for source
    >>> enc_src = encoder(src_img)
    >>> code_src = inter_AB(enc_src)
    >>>
    >>> # Forward pass for destination
    >>> enc_dst = encoder(dst_img)
    >>> code_dst_AB = inter_AB(enc_dst)
    >>> code_dst_B = inter_B(enc_dst)
    >>> code_dst = torch.cat([code_dst_AB, code_dst_B], dim=1)
    >>>
    >>> # Swap: src's AB + dst's B
    >>> code_swap = torch.cat([code_src, code_dst_B], dim=1)
    >>> swapped_face, mask = decoder(code_swap)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from visagen.models.architectures.df import (
    Downscale,
    PixelNorm,
    ResidualBlock,
    Upscale,
)


class LIAEArchi(nn.Module):
    """
    LIAE (Lightweight Inter-AB-B Encoder) Architecture.

    This is an alternative SAEHD architecture that uses
    less memory by sharing the decoder between src and dst.

    Architecture:
    - Shared Encoder
    - InterAB - maps encoder output to ae_dims*2 (used for both src and dst)
    - InterB - maps encoder output to ae_dims*2 (used only for dst)
    - Shared Decoder - takes concatenated codes (ae_dims*4 total)

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

        # Inter output channels (ae_dims * 2 for each Inter)
        self.inter_out_ch = ae_dims * 2

    def Encoder(self) -> nn.Module:
        """Create encoder module (shared between src and dst)."""
        return _LIAEEncoder(
            in_ch=self.in_ch,
            e_ch=self.e_ch,
            use_transformer=self.use_transformer,
            use_pixel_norm=self.use_pixel_norm,
            use_cos_act=self.use_cos_act,
        )

    def InterAB(self) -> nn.Module:
        """
        Create InterAB module.

        Used for both src and dst to extract identity-invariant features.
        """
        return _LIAEInter(
            in_features=self.flatten_size,
            ae_dims=self.ae_dims,
            ae_out_ch=self.inter_out_ch,
            lowest_dense_res=self.lowest_dense_res,
            use_transformer=self.use_transformer,
            use_cos_act=self.use_cos_act,
        )

    def InterB(self) -> nn.Module:
        """
        Create InterB module.

        Used only for dst to extract identity-specific features.
        """
        return _LIAEInter(
            in_features=self.flatten_size,
            ae_dims=self.ae_dims,
            ae_out_ch=self.inter_out_ch,
            lowest_dense_res=self.lowest_dense_res,
            use_transformer=self.use_transformer,
            use_cos_act=self.use_cos_act,
        )

    def Decoder(self) -> nn.Module:
        """
        Create decoder module (shared between src and dst).

        Takes concatenated InterAB + InterB codes (ae_dims * 4 channels).
        """
        # Decoder input is concatenation of InterAB and InterB outputs
        decoder_in_ch = self.inter_out_ch * 2  # ae_dims * 4
        return _LIAEDecoder(
            in_ch=decoder_in_ch,
            d_ch=self.d_ch,
            d_mask_ch=self.d_mask_ch,
            use_transformer=self.use_transformer,
            use_double=self.use_double,
            use_cos_act=self.use_cos_act,
        )


class _LIAEEncoder(nn.Module):
    """LIAE Encoder implementation (same as DF Encoder)."""

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


class _LIAEInter(nn.Module):
    """LIAE Inter (InterAB or InterB) implementation."""

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


class _LIAEDecoder(nn.Module):
    """LIAE Decoder implementation (shared decoder for src and dst)."""

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
            z: Concatenated latent tensor (InterAB + InterB codes).

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
