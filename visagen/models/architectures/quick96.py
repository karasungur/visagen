"""
Quick96 Architecture for fast inference.

Lightweight 96x96 resolution model optimized for:
- Mobile deployment
- Real-time inference
- Low memory usage

This is a wrapper around DFArchi with fixed 96x96 resolution
and optimized default dimensions from DeepFaceLab's Quick96 model.

Example:
    >>> archi = Quick96Archi()
    >>> encoder = archi.Encoder()
    >>> inter = archi.Inter()
    >>> decoder_src = archi.Decoder(name='src')
    >>> decoder_dst = archi.Decoder(name='dst')
"""

import torch
import torch.nn as nn

from visagen.models.architectures.df import DFArchi


class Quick96Archi:
    """
    Quick96 architecture factory.

    Fixed 96x96 resolution with optimized dimensions for speed.
    Uses DF architecture with 'ud' options (upscale + double).

    The Quick96 model from DeepFaceLab uses:
    - resolution: 96 (fixed)
    - ae_dims: 128 (smaller than standard 256)
    - e_dims: 64
    - d_dims: 64
    - d_mask_dims: 16 (smaller than standard 22)

    Args:
        ae_dims: Autoencoder dimensions. Default: 128.
        e_dims: Encoder dimensions. Default: 64.
        d_dims: Decoder dimensions. Default: 64.
        d_mask_dims: Decoder mask dimensions. Default: 16.

    Example:
        >>> archi = Quick96Archi()
        >>> encoder = archi.Encoder()
        >>> inter = archi.Inter()
        >>> decoder_src = archi.Decoder(name='src')
        >>> decoder_dst = archi.Decoder(name='dst')
    """

    RESOLUTION = 96  # Fixed resolution

    def __init__(
        self,
        ae_dims: int = 128,
        e_ch: int = 64,
        d_ch: int = 64,
        d_mask_ch: int = 16,
    ) -> None:
        self.ae_dims = ae_dims
        self.e_ch = e_ch
        self.d_ch = d_ch
        self.d_mask_ch = d_mask_ch

        # Use DF architecture with 'ud' options (upscale + double)
        self._df = DFArchi(
            resolution=self.RESOLUTION,
            ae_dims=ae_dims,
            e_ch=e_ch,
            d_ch=d_ch,
            d_mask_ch=d_mask_ch,
            opts="ud",  # upscale + double
        )

    def Encoder(self) -> nn.Module:
        """Create encoder module."""
        return self._df.Encoder()

    def Inter(self, ae_out_ch: int | None = None) -> nn.Module:
        """
        Create inter (bottleneck) module.

        Args:
            ae_out_ch: Output channels. Defaults to ae_dims.
        """
        if ae_out_ch is None:
            ae_out_ch = self.ae_dims
        return self._df.Inter(ae_out_ch=ae_out_ch)

    def Decoder(self, name: str = "src", ae_out_ch: int | None = None) -> nn.Module:
        """
        Create decoder module.

        Args:
            name: Decoder name (for identification only).
            ae_out_ch: Input channels from inter. Defaults to ae_dims.

        Returns:
            Decoder module.
        """
        if ae_out_ch is None:
            ae_out_ch = self.ae_dims
        return self._df.Decoder(ae_out_ch=ae_out_ch)

    def get_encoder_out_ch(self) -> int:
        """Get encoder output channels (flattened)."""
        return self._df.encoder_out_ch

    def get_inter_out_ch(self) -> int:
        """Get inter output channels."""
        return self._df.ae_dims


class Quick96Model(nn.Module):
    """
    Complete Quick96 model for training and inference.

    Combines encoder, inter, and dual decoders into a single module.
    This is a ready-to-use model that handles the full forward pass.

    Architecture:
        - Shared encoder for both src and dst
        - Shared inter (bottleneck) layer
        - Separate src and dst decoders

    Args:
        ae_dims: Autoencoder dimensions. Default: 128.
        e_dims: Encoder dimensions. Default: 64.
        d_dims: Decoder dimensions. Default: 64.
        d_mask_dims: Decoder mask dimensions. Default: 16.

    Example:
        >>> model = Quick96Model()
        >>> src = torch.randn(1, 3, 96, 96)
        >>> dst = torch.randn(1, 3, 96, 96)
        >>> pred_src, mask_src = model(src, mode='src')
        >>> pred_dst, mask_dst = model(dst, mode='dst')
        >>> pred_swap, mask_swap = model.swap(dst)  # src face on dst
    """

    RESOLUTION = 96

    def __init__(
        self,
        ae_dims: int = 128,
        e_ch: int = 64,
        d_ch: int = 64,
        d_mask_ch: int = 16,
    ) -> None:
        super().__init__()

        self.ae_dims = ae_dims
        self.e_ch = e_ch
        self.d_ch = d_ch
        self.d_mask_ch = d_mask_ch

        archi = Quick96Archi(
            ae_dims=ae_dims,
            e_ch=e_ch,
            d_ch=d_ch,
            d_mask_ch=d_mask_ch,
        )

        self.encoder = archi.Encoder()
        self.inter = archi.Inter()
        self.decoder_src = archi.Decoder(name="src")
        self.decoder_dst = archi.Decoder(name="dst")

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to latent code.

        Args:
            x: Input image (B, 3, 96, 96).

        Returns:
            Latent code tensor.
        """
        features = self.encoder(x)
        code = self.inter(features)
        return code

    def decode_src(self, code: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Decode using source decoder.

        Args:
            code: Latent code from inter module.

        Returns:
            Tuple of (reconstructed_image, mask).
        """
        return self.decoder_src(code)

    def decode_dst(self, code: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Decode using destination decoder.

        Args:
            code: Latent code from inter module.

        Returns:
            Tuple of (reconstructed_image, mask).
        """
        return self.decoder_dst(code)

    def forward(
        self,
        x: torch.Tensor,
        mode: str = "src",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input image (B, 3, 96, 96).
            mode: 'src' or 'dst' decoder to use.

        Returns:
            Tuple of (reconstructed_image, mask).
        """
        code = self.encode(x)

        if mode == "src":
            return self.decode_src(code)
        else:
            return self.decode_dst(code)

    def swap(self, dst: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Swap face: apply src decoder to dst encoding.

        This is the core face swap operation - encode the destination
        face and decode it using the source decoder to get the source
        identity on the destination face.

        Args:
            dst: Destination face (B, 3, 96, 96).

        Returns:
            Tuple of (swapped_face, mask).
        """
        code = self.encode(dst)
        return self.decode_src(code)

    def get_training_outputs(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """
        Get all outputs needed for training.

        Args:
            src: Source face (B, 3, 96, 96).
            dst: Destination face (B, 3, 96, 96).

        Returns:
            Dictionary with keys:
                - 'src_src': Source reconstructed with src decoder
                - 'dst_dst': Destination reconstructed with dst decoder
                - 'src_dst': Source face swapped onto destination (swap result)
        """
        # Encode both
        src_code = self.encode(src)
        dst_code = self.encode(dst)

        # Reconstruct source with source decoder
        src_src = self.decode_src(src_code)

        # Reconstruct destination with destination decoder
        dst_dst = self.decode_dst(dst_code)

        # Swap: destination encoded, source decoder
        src_dst = self.decode_src(dst_code)

        return {
            "src_src": src_src,
            "dst_dst": dst_dst,
            "src_dst": src_dst,
        }
