"""Tests for LIAE (Lightweight Inter-AB-B Encoder) architecture."""

import pytest
import torch

from visagen.models.architectures.liae import LIAEArchi


class TestLIAEArchi:
    """Test LIAEArchi factory class."""

    def test_default_initialization(self) -> None:
        """Test default parameters."""
        archi = LIAEArchi()
        assert archi.resolution == 256
        assert archi.e_ch == 64
        assert archi.d_ch == 64
        assert archi.ae_dims == 256
        assert archi.inter_out_ch == 512  # ae_dims * 2

    def test_custom_initialization(self) -> None:
        """Test custom parameters."""
        archi = LIAEArchi(
            resolution=128,
            e_ch=32,
            d_ch=48,
            ae_dims=128,
            opts="tc",
        )
        assert archi.resolution == 128
        assert archi.e_ch == 32
        assert archi.d_ch == 48
        assert archi.ae_dims == 128
        assert archi.use_transformer is True
        assert archi.use_cos_act is True

    def test_opts_parsing(self) -> None:
        """Test architecture options parsing."""
        archi = LIAEArchi(opts="tduc")
        assert archi.use_transformer is True
        assert archi.use_double is True
        assert archi.use_pixel_norm is True
        assert archi.use_cos_act is True

    def test_lowest_dense_res_standard(self) -> None:
        """Test lowest dense resolution for standard mode."""
        archi = LIAEArchi(resolution=256, opts="")
        assert archi.lowest_dense_res == 16  # 256 // 16

    def test_lowest_dense_res_double(self) -> None:
        """Test lowest dense resolution for double mode."""
        archi = LIAEArchi(resolution=256, opts="d")
        assert archi.lowest_dense_res == 8  # 256 // 32

    def test_encoder_out_res_standard(self) -> None:
        """Test encoder output resolution for standard mode."""
        archi = LIAEArchi(resolution=256, opts="")
        assert archi.encoder_out_res == 16  # 256 // 16

    def test_encoder_out_res_transformer(self) -> None:
        """Test encoder output resolution for transformer mode."""
        archi = LIAEArchi(resolution=256, opts="t")
        assert archi.encoder_out_res == 8  # 256 // 32


class TestLIAEEncoder:
    """Test LIAE Encoder module."""

    @pytest.fixture
    def archi(self) -> LIAEArchi:
        return LIAEArchi(resolution=256, e_ch=64)

    def test_encoder_output_shape(self, archi: LIAEArchi) -> None:
        """Test encoder output shape."""
        encoder = archi.Encoder()
        x = torch.randn(2, 3, 256, 256)
        out = encoder(x)
        # Flattened: e_ch*8 * (256/16)^2 = 512 * 16 * 16 = 131072
        assert out.shape == (2, archi.flatten_size)

    def test_encoder_transformer_mode(self) -> None:
        """Test encoder in transformer mode."""
        archi = LIAEArchi(resolution=256, e_ch=64, opts="t")
        encoder = archi.Encoder()
        x = torch.randn(2, 3, 256, 256)
        out = encoder(x)
        # Transformer: 256/32 = 8, so 512 * 8 * 8 = 32768
        assert out.shape == (2, archi.flatten_size)

    def test_encoder_pixel_norm(self) -> None:
        """Test encoder with pixel normalization."""
        archi = LIAEArchi(resolution=256, opts="u")
        encoder = archi.Encoder()
        x = torch.randn(2, 3, 256, 256)
        out = encoder(x)
        assert out.shape == (2, archi.flatten_size)


class TestLIAEInter:
    """Test LIAE Inter modules (InterAB and InterB)."""

    @pytest.fixture
    def archi(self) -> LIAEArchi:
        return LIAEArchi(resolution=256, ae_dims=256)

    def test_inter_ab_output_shape(self, archi: LIAEArchi) -> None:
        """Test InterAB output shape."""
        inter_ab = archi.InterAB()
        x = torch.randn(2, archi.flatten_size)
        out = inter_ab(x)
        # Output: ae_dims*2 channels, lowest_dense_res*2 spatial (after upscale)
        expected_ch = archi.inter_out_ch  # 512
        expected_res = archi.lowest_dense_res * 2  # 32
        assert out.shape == (2, expected_ch, expected_res, expected_res)

    def test_inter_b_output_shape(self, archi: LIAEArchi) -> None:
        """Test InterB output shape."""
        inter_b = archi.InterB()
        x = torch.randn(2, archi.flatten_size)
        out = inter_b(x)
        expected_ch = archi.inter_out_ch
        expected_res = archi.lowest_dense_res * 2
        assert out.shape == (2, expected_ch, expected_res, expected_res)

    def test_inter_transformer_mode(self) -> None:
        """Test Inter in transformer mode (no upscale)."""
        archi = LIAEArchi(resolution=256, ae_dims=256, opts="t")
        inter_ab = archi.InterAB()
        x = torch.randn(2, archi.flatten_size)
        out = inter_ab(x)
        # No upscale in transformer mode
        expected_res = archi.lowest_dense_res  # 8
        assert out.shape == (2, archi.inter_out_ch, expected_res, expected_res)


class TestLIAEDecoder:
    """Test LIAE Decoder module."""

    @pytest.fixture
    def archi(self) -> LIAEArchi:
        return LIAEArchi(resolution=256, d_ch=64, d_mask_ch=22, ae_dims=256)

    def test_decoder_output_shape(self, archi: LIAEArchi) -> None:
        """Test decoder output shape."""
        decoder = archi.Decoder()
        # Decoder input is concatenated InterAB + InterB
        in_ch = archi.inter_out_ch * 2  # 1024
        spatial = archi.lowest_dense_res * 2  # 32
        z = torch.randn(2, in_ch, spatial, spatial)
        img, mask = decoder(z)
        assert img.shape == (2, 3, 256, 256)
        assert mask.shape == (2, 1, 256, 256)

    def test_decoder_double_mode(self) -> None:
        """Test decoder in double resolution mode."""
        archi = LIAEArchi(resolution=256, opts="d")
        decoder = archi.Decoder()
        in_ch = archi.inter_out_ch * 2
        spatial = archi.lowest_dense_res * 2  # 16 for double mode
        z = torch.randn(2, in_ch, spatial, spatial)
        img, mask = decoder(z)
        # Double mode uses pixel shuffle for higher quality at same resolution
        assert img.shape == (2, 3, 256, 256)
        assert mask.shape == (2, 1, 256, 256)

    def test_decoder_transformer_mode(self) -> None:
        """Test decoder in transformer mode."""
        archi = LIAEArchi(resolution=256, opts="t")
        decoder = archi.Decoder()
        in_ch = archi.inter_out_ch * 2
        spatial = archi.lowest_dense_res  # No upscale in Inter
        z = torch.randn(2, in_ch, spatial, spatial)
        img, mask = decoder(z)
        assert img.shape == (2, 3, 256, 256)
        assert mask.shape == (2, 1, 256, 256)

    def test_decoder_output_range(self, archi: LIAEArchi) -> None:
        """Test decoder outputs are in [0, 1] range."""
        decoder = archi.Decoder()
        in_ch = archi.inter_out_ch * 2
        spatial = archi.lowest_dense_res * 2
        z = torch.randn(2, in_ch, spatial, spatial)
        img, mask = decoder(z)
        assert img.min() >= 0.0 and img.max() <= 1.0
        assert mask.min() >= 0.0 and mask.max() <= 1.0


class TestLIAEEndToEnd:
    """End-to-end tests for LIAE architecture."""

    def test_full_forward_pass(self) -> None:
        """Test complete forward pass."""
        archi = LIAEArchi(resolution=256, e_ch=64, d_ch=64, ae_dims=256)
        encoder = archi.Encoder()
        inter_ab = archi.InterAB()
        inter_b = archi.InterB()
        decoder = archi.Decoder()

        src_img = torch.randn(2, 3, 256, 256)
        dst_img = torch.randn(2, 3, 256, 256)

        # Encode
        enc_src = encoder(src_img)
        enc_dst = encoder(dst_img)

        # Get codes
        code_src_ab = inter_ab(enc_src)
        code_dst_ab = inter_ab(enc_dst)
        code_dst_b = inter_b(enc_dst)

        # Destination reconstruction: AB + B
        code_dst = torch.cat([code_dst_ab, code_dst_b], dim=1)
        dst_recon, dst_mask = decoder(code_dst)

        # Swap: src's AB + dst's B
        code_swap = torch.cat([code_src_ab, code_dst_b], dim=1)
        swapped, swap_mask = decoder(code_swap)

        assert dst_recon.shape == (2, 3, 256, 256)
        assert swapped.shape == (2, 3, 256, 256)

    def test_gradients_flow(self) -> None:
        """Test that gradients flow through the network."""
        archi = LIAEArchi(resolution=128, e_ch=32, d_ch=32, ae_dims=128)
        encoder = archi.Encoder()
        inter_ab = archi.InterAB()
        inter_b = archi.InterB()
        decoder = archi.Decoder()

        x = torch.randn(1, 3, 128, 128, requires_grad=True)

        enc = encoder(x)
        code_ab = inter_ab(enc)
        code_b = inter_b(enc)
        code = torch.cat([code_ab, code_b], dim=1)
        img, mask = decoder(code)

        loss = img.mean() + mask.mean()
        loss.backward()

        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_different_resolutions(self) -> None:
        """Test architecture with different resolutions."""
        for res in [128, 192, 256]:
            archi = LIAEArchi(resolution=res, e_ch=32, d_ch=32, ae_dims=128)
            encoder = archi.Encoder()
            inter_ab = archi.InterAB()
            inter_b = archi.InterB()
            decoder = archi.Decoder()

            x = torch.randn(1, 3, res, res)
            enc = encoder(x)
            code_ab = inter_ab(enc)
            code_b = inter_b(enc)
            code = torch.cat([code_ab, code_b], dim=1)
            img, mask = decoder(code)

            assert img.shape == (1, 3, res, res)
            assert mask.shape == (1, 1, res, res)

    def test_all_opts_combinations(self) -> None:
        """Test various option combinations."""
        opts_list = ["", "t", "d", "u", "c", "td", "tuc", "tduc"]
        for opts in opts_list:
            archi = LIAEArchi(resolution=128, e_ch=32, d_ch=32, ae_dims=64, opts=opts)
            encoder = archi.Encoder()
            inter_ab = archi.InterAB()
            inter_b = archi.InterB()
            decoder = archi.Decoder()

            x = torch.randn(1, 3, 128, 128)
            enc = encoder(x)
            code_ab = inter_ab(enc)
            code_b = inter_b(enc)
            code = torch.cat([code_ab, code_b], dim=1)
            img, mask = decoder(code)

            # All modes output same resolution (double uses pixel shuffle for quality)
            assert img.shape == (1, 3, 128, 128)
