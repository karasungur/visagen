"""Tests for DF (Direct Face) architecture."""

import pytest
import torch

from visagen.models.architectures.df import (
    DFArchi,
    Downscale,
    PixelNorm,
    ResidualBlock,
    Upscale,
)


class TestBasicBlocks:
    """Test basic building blocks."""

    def test_pixel_norm(self):
        """Test PixelNorm layer."""
        norm = PixelNorm()
        x = torch.randn(2, 64, 16, 16)
        out = norm(x)

        assert out.shape == x.shape
        # Check normalization (mean of squared values should be ~1)
        mean_sq = (out**2).mean(dim=1)
        assert mean_sq.mean().item() == pytest.approx(1.0, rel=0.1)

    def test_downscale(self):
        """Test Downscale block."""
        down = Downscale(3, 64, kernel_size=5)
        x = torch.randn(2, 3, 64, 64)
        out = down(x)

        assert out.shape == (2, 64, 32, 32)

    def test_downscale_cos_activation(self):
        """Test Downscale with cosine activation."""
        down = Downscale(3, 64, kernel_size=5, use_cos_act=True)
        x = torch.randn(2, 3, 64, 64)
        out = down(x)

        assert out.shape == (2, 64, 32, 32)

    def test_upscale(self):
        """Test Upscale block."""
        up = Upscale(64, 32, kernel_size=3)
        x = torch.randn(2, 64, 16, 16)
        out = up(x)

        assert out.shape == (2, 32, 32, 32)

    def test_residual_block(self):
        """Test ResidualBlock."""
        res = ResidualBlock(64, kernel_size=3)
        x = torch.randn(2, 64, 16, 16)
        out = res(x)

        assert out.shape == x.shape


class TestDFArchi:
    """Test DFArchi factory class."""

    def test_default_init(self):
        """Test default initialization."""
        archi = DFArchi(resolution=256)

        assert archi.resolution == 256
        assert archi.e_ch == 64
        assert archi.d_ch == 64
        assert archi.ae_dims == 256
        assert not archi.use_transformer
        assert not archi.use_double

    def test_transformer_mode(self):
        """Test transformer mode initialization."""
        archi = DFArchi(resolution=256, opts="t")

        assert archi.use_transformer
        assert archi.encoder_out_res == 8  # 256 // 32

    def test_double_mode(self):
        """Test double resolution mode."""
        archi = DFArchi(resolution=256, opts="d")

        assert archi.use_double
        assert archi.lowest_dense_res == 8  # 256 // 32

    def test_pixel_norm_mode(self):
        """Test pixel norm mode."""
        archi = DFArchi(resolution=256, opts="u")

        assert archi.use_pixel_norm

    def test_cosine_activation_mode(self):
        """Test cosine activation mode."""
        archi = DFArchi(resolution=256, opts="c")

        assert archi.use_cos_act


class TestDFEncoder:
    """Test DF Encoder."""

    def test_encoder_standard(self):
        """Test standard encoder forward pass."""
        archi = DFArchi(resolution=256, e_ch=64)
        encoder = archi.Encoder()

        x = torch.randn(2, 3, 256, 256)
        out = encoder(x)

        # Output should be flattened
        expected_size = 64 * 8 * 16 * 16  # e_ch * 8 * (res/16)^2
        assert out.shape == (2, expected_size)

    def test_encoder_transformer_mode(self):
        """Test transformer mode encoder."""
        archi = DFArchi(resolution=256, e_ch=64, opts="t")
        encoder = archi.Encoder()

        x = torch.randn(2, 3, 256, 256)
        out = encoder(x)

        # Transformer mode has 5 downscales (res/32)
        expected_size = 64 * 8 * 8 * 8  # e_ch * 8 * (res/32)^2
        assert out.shape == (2, expected_size)

    def test_encoder_with_pixel_norm(self):
        """Test encoder with pixel normalization."""
        archi = DFArchi(resolution=256, e_ch=64, opts="u")
        encoder = archi.Encoder()

        x = torch.randn(2, 3, 256, 256)
        out = encoder(x)

        assert out.shape[0] == 2
        # Check normalized (mean of squared values should be ~1)
        mean_sq = (out**2).mean(dim=1)
        assert mean_sq.mean().item() == pytest.approx(1.0, rel=0.1)


class TestDFInter:
    """Test DF Inter (bottleneck)."""

    def test_inter_standard(self):
        """Test standard inter forward pass."""
        archi = DFArchi(resolution=256, e_ch=64, ae_dims=256)
        encoder = archi.Encoder()
        inter = archi.Inter(ae_out_ch=256)

        x = torch.randn(2, 3, 256, 256)
        encoded = encoder(x)
        out = inter(encoded)

        # Inter outputs 4D tensor with upscale
        # lowest_dense_res = 16, then upscale 2x = 32
        assert out.shape == (2, 256, 32, 32)

    def test_inter_transformer_mode(self):
        """Test transformer mode inter."""
        archi = DFArchi(resolution=256, e_ch=64, ae_dims=256, opts="t")
        encoder = archi.Encoder()
        inter = archi.Inter(ae_out_ch=256)

        x = torch.randn(2, 3, 256, 256)
        encoded = encoder(x)
        out = inter(encoded)

        # Transformer mode: no upscale in inter, output is lowest_dense_res
        assert out.shape == (2, 256, 16, 16)


class TestDFDecoder:
    """Test DF Decoder."""

    def test_decoder_standard(self):
        """Test standard decoder forward pass."""
        archi = DFArchi(resolution=256, e_ch=64, d_ch=64, d_mask_ch=22, ae_dims=256)
        encoder = archi.Encoder()
        inter = archi.Inter(ae_out_ch=256)
        decoder = archi.Decoder(ae_out_ch=256)

        x = torch.randn(2, 3, 256, 256)
        encoded = encoder(x)
        latent = inter(encoded)
        img, mask = decoder(latent)

        # Standard mode: 3 upscales from 32 -> 256
        assert img.shape == (2, 3, 256, 256)
        assert mask.shape == (2, 1, 256, 256)

        # Check output range
        assert img.min() >= 0.0 and img.max() <= 1.0
        assert mask.min() >= 0.0 and mask.max() <= 1.0

    def test_decoder_transformer_mode(self):
        """Test transformer mode decoder."""
        archi = DFArchi(
            resolution=256, e_ch=64, d_ch=64, d_mask_ch=22, ae_dims=256, opts="t"
        )
        encoder = archi.Encoder()
        inter = archi.Inter(ae_out_ch=256)
        decoder = archi.Decoder(ae_out_ch=256)

        x = torch.randn(2, 3, 256, 256)
        encoded = encoder(x)
        latent = inter(encoded)
        img, mask = decoder(latent)

        # Transformer mode: 4 upscales from 16 -> 256
        assert img.shape == (2, 3, 256, 256)
        assert mask.shape == (2, 1, 256, 256)

    def test_decoder_double_mode(self):
        """Test double resolution decoder."""
        archi = DFArchi(
            resolution=256, e_ch=64, d_ch=64, d_mask_ch=22, ae_dims=256, opts="d"
        )
        encoder = archi.Encoder()
        inter = archi.Inter(ae_out_ch=256)
        decoder = archi.Decoder(ae_out_ch=256)

        x = torch.randn(2, 3, 256, 256)
        encoded = encoder(x)
        latent = inter(encoded)
        img, mask = decoder(latent)

        # Double mode uses pixel shuffle for higher quality output at same resolution
        # The extra upscale compensates for smaller lowest_dense_res (res/32 vs res/16)
        assert img.shape == (2, 3, 256, 256)
        assert mask.shape == (2, 1, 256, 256)


class TestFullPipeline:
    """Test full encoder-inter-decoder pipeline."""

    def test_full_pipeline_standard(self):
        """Test complete standard pipeline."""
        archi = DFArchi(resolution=256, e_ch=64, d_ch=64, d_mask_ch=22, ae_dims=256)

        encoder = archi.Encoder()
        inter_src = archi.Inter(ae_out_ch=256)
        inter_dst = archi.Inter(ae_out_ch=256)
        decoder_src = archi.Decoder(ae_out_ch=256)
        decoder_dst = archi.Decoder(ae_out_ch=256)

        # Source and destination inputs
        src = torch.randn(2, 3, 256, 256)
        dst = torch.randn(2, 3, 256, 256)

        # Encode
        src_encoded = encoder(src)
        dst_encoded = encoder(dst)

        # Inter
        src_latent = inter_src(src_encoded)
        dst_latent = inter_dst(dst_encoded)

        # Decode
        src_img, src_mask = decoder_src(src_latent)
        dst_img, dst_mask = decoder_dst(dst_latent)

        # Face swap: use src encoder with dst inter
        swap_latent = inter_dst(src_encoded)
        swap_img, swap_mask = decoder_dst(swap_latent)

        assert src_img.shape == (2, 3, 256, 256)
        assert dst_img.shape == (2, 3, 256, 256)
        assert swap_img.shape == (2, 3, 256, 256)

    def test_gradient_flow(self):
        """Test gradient flow through the network."""
        archi = DFArchi(resolution=128, e_ch=32, d_ch=32, d_mask_ch=16, ae_dims=128)

        encoder = archi.Encoder()
        inter = archi.Inter(ae_out_ch=128)
        decoder = archi.Decoder(ae_out_ch=128)

        x = torch.randn(1, 3, 128, 128, requires_grad=True)

        encoded = encoder(x)
        latent = inter(encoded)
        img, mask = decoder(latent)

        loss = img.mean() + mask.mean()
        loss.backward()

        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_different_resolutions(self):
        """Test with different input resolutions."""
        for resolution in [128, 256, 512]:
            archi = DFArchi(
                resolution=resolution, e_ch=32, d_ch=32, d_mask_ch=16, ae_dims=128
            )

            encoder = archi.Encoder()
            inter = archi.Inter(ae_out_ch=128)
            decoder = archi.Decoder(ae_out_ch=128)

            x = torch.randn(1, 3, resolution, resolution)
            encoded = encoder(x)
            latent = inter(encoded)
            img, mask = decoder(latent)

            assert img.shape == (1, 3, resolution, resolution)
            assert mask.shape == (1, 1, resolution, resolution)
