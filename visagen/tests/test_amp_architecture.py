"""Tests for AMP (Amplified Morphable Portrait) architecture."""

import pytest
import torch

from visagen.models.architectures.amp import AMPArchi, morph_code


class TestMorphCode:
    """Test morph_code function."""

    def test_morph_training_mode(self) -> None:
        """Test morphing in training mode with binomial sampling."""
        code_src = torch.ones(4, 256, 32, 32)
        code_dst = torch.zeros(4, 256, 32, 32)

        # With morph_factor=1.0, should always select src
        result = morph_code(code_src, code_dst, morph_factor=1.0, training=True)
        assert torch.allclose(result, code_src)

        # With morph_factor=0.0, should always select dst
        result = morph_code(code_src, code_dst, morph_factor=0.0, training=True)
        assert torch.allclose(result, code_dst)

    def test_morph_inference_mode(self) -> None:
        """Test morphing in inference mode with fixed value."""
        code_src = torch.ones(2, 256, 32, 32)
        code_dst = torch.zeros(2, 256, 32, 32)

        # morph_value=1.0 -> full src
        result = morph_code(code_src, code_dst, morph_value=1.0, training=False)
        assert torch.allclose(result, code_src)

        # morph_value=0.0 -> full dst
        result = morph_code(code_src, code_dst, morph_value=0.0, training=False)
        assert torch.allclose(result, code_dst)

        # morph_value=0.5 -> blend
        result = morph_code(code_src, code_dst, morph_value=0.5, training=False)
        expected = 0.5 * code_src + 0.5 * code_dst
        assert torch.allclose(result, expected)

    def test_morph_default_inference(self) -> None:
        """Test default morph_value in inference mode."""
        code_src = torch.ones(2, 256, 32, 32)
        code_dst = torch.zeros(2, 256, 32, 32)

        # Default morph_value should be 1.0 (full src)
        result = morph_code(code_src, code_dst, training=False)
        assert torch.allclose(result, code_src)

    def test_morph_gradient_flow(self) -> None:
        """Test gradients flow through morph."""
        code_src = torch.randn(2, 256, 32, 32, requires_grad=True)
        code_dst = torch.randn(2, 256, 32, 32, requires_grad=True)

        result = morph_code(code_src, code_dst, morph_value=0.7, training=False)
        loss = result.sum()
        loss.backward()

        assert code_src.grad is not None
        assert code_dst.grad is not None


class TestAMPArchi:
    """Test AMPArchi factory class."""

    def test_default_initialization(self) -> None:
        """Test default parameters."""
        archi = AMPArchi()
        assert archi.resolution == 256
        assert archi.e_ch == 64
        assert archi.d_ch == 64
        assert archi.ae_dims == 256

    def test_custom_initialization(self) -> None:
        """Test custom parameters."""
        archi = AMPArchi(
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
        archi = AMPArchi(opts="tduc")
        assert archi.use_transformer is True
        assert archi.use_double is True
        assert archi.use_pixel_norm is True
        assert archi.use_cos_act is True

    def test_lowest_dense_res_standard(self) -> None:
        """Test lowest dense resolution for standard mode."""
        archi = AMPArchi(resolution=256, opts="")
        assert archi.lowest_dense_res == 16

    def test_lowest_dense_res_double(self) -> None:
        """Test lowest dense resolution for double mode."""
        archi = AMPArchi(resolution=256, opts="d")
        assert archi.lowest_dense_res == 8


class TestAMPEncoder:
    """Test AMP Encoder module."""

    @pytest.fixture
    def archi(self) -> AMPArchi:
        return AMPArchi(resolution=256, e_ch=64)

    def test_encoder_output_shape(self, archi: AMPArchi) -> None:
        """Test encoder output shape."""
        encoder = archi.Encoder()
        x = torch.randn(2, 3, 256, 256)
        out = encoder(x)
        assert out.shape == (2, archi.flatten_size)

    def test_encoder_transformer_mode(self) -> None:
        """Test encoder in transformer mode."""
        archi = AMPArchi(resolution=256, e_ch=64, opts="t")
        encoder = archi.Encoder()
        x = torch.randn(2, 3, 256, 256)
        out = encoder(x)
        assert out.shape == (2, archi.flatten_size)

    def test_encoder_pixel_norm(self) -> None:
        """Test encoder with pixel normalization."""
        archi = AMPArchi(resolution=256, opts="u")
        encoder = archi.Encoder()
        x = torch.randn(2, 3, 256, 256)
        out = encoder(x)
        assert out.shape == (2, archi.flatten_size)


class TestAMPInter:
    """Test AMP Inter modules."""

    @pytest.fixture
    def archi(self) -> AMPArchi:
        return AMPArchi(resolution=256, ae_dims=256)

    def test_inter_src_output_shape(self, archi: AMPArchi) -> None:
        """Test InterSrc output shape."""
        inter_src = archi.InterSrc()
        x = torch.randn(2, archi.flatten_size)
        out = inter_src(x)
        expected_res = archi.lowest_dense_res * 2  # 32
        assert out.shape == (2, archi.ae_dims, expected_res, expected_res)

    def test_inter_dst_output_shape(self, archi: AMPArchi) -> None:
        """Test InterDst output shape."""
        inter_dst = archi.InterDst()
        x = torch.randn(2, archi.flatten_size)
        out = inter_dst(x)
        expected_res = archi.lowest_dense_res * 2
        assert out.shape == (2, archi.ae_dims, expected_res, expected_res)

    def test_inter_transformer_mode(self) -> None:
        """Test Inter in transformer mode (no upscale)."""
        archi = AMPArchi(resolution=256, ae_dims=256, opts="t")
        inter_src = archi.InterSrc()
        x = torch.randn(2, archi.flatten_size)
        out = inter_src(x)
        expected_res = archi.lowest_dense_res  # 8
        assert out.shape == (2, archi.ae_dims, expected_res, expected_res)


class TestAMPDecoder:
    """Test AMP Decoder module."""

    @pytest.fixture
    def archi(self) -> AMPArchi:
        return AMPArchi(resolution=256, d_ch=64, d_mask_ch=22, ae_dims=256)

    def test_decoder_output_shape(self, archi: AMPArchi) -> None:
        """Test decoder output shape."""
        decoder = archi.Decoder()
        spatial = archi.lowest_dense_res * 2  # 32
        z = torch.randn(2, archi.ae_dims, spatial, spatial)
        img, mask = decoder(z)
        assert img.shape == (2, 3, 256, 256)
        assert mask.shape == (2, 1, 256, 256)

    def test_decoder_double_mode(self) -> None:
        """Test decoder in double resolution mode."""
        archi = AMPArchi(resolution=256, opts="d")
        decoder = archi.Decoder()
        spatial = archi.lowest_dense_res * 2  # 16 for double mode
        z = torch.randn(2, archi.ae_dims, spatial, spatial)
        img, mask = decoder(z)
        # Double mode uses pixel shuffle for higher quality at same resolution
        assert img.shape == (2, 3, 256, 256)
        assert mask.shape == (2, 1, 256, 256)

    def test_decoder_transformer_mode(self) -> None:
        """Test decoder in transformer mode."""
        archi = AMPArchi(resolution=256, opts="t")
        decoder = archi.Decoder()
        spatial = archi.lowest_dense_res  # No upscale in Inter
        z = torch.randn(2, archi.ae_dims, spatial, spatial)
        img, mask = decoder(z)
        assert img.shape == (2, 3, 256, 256)
        assert mask.shape == (2, 1, 256, 256)

    def test_decoder_output_range(self, archi: AMPArchi) -> None:
        """Test decoder outputs are in [0, 1] range."""
        decoder = archi.Decoder()
        spatial = archi.lowest_dense_res * 2
        z = torch.randn(2, archi.ae_dims, spatial, spatial)
        img, mask = decoder(z)
        assert img.min() >= 0.0 and img.max() <= 1.0
        assert mask.min() >= 0.0 and mask.max() <= 1.0


class TestAMPEndToEnd:
    """End-to-end tests for AMP architecture."""

    def test_full_forward_pass_training(self) -> None:
        """Test complete forward pass in training mode."""
        archi = AMPArchi(resolution=256, e_ch=64, d_ch=64, ae_dims=256)
        encoder = archi.Encoder()
        inter_src = archi.InterSrc()
        inter_dst = archi.InterDst()
        decoder = archi.Decoder()

        src_img = torch.randn(2, 3, 256, 256)

        # Encode
        enc = encoder(src_img)

        # Get codes from both Inter modules
        code_src = inter_src(enc)
        code_dst = inter_dst(enc)

        # Morph (training mode)
        code = morph_code(code_src, code_dst, morph_factor=0.5, training=True)

        # Decode
        img, mask = decoder(code)

        assert img.shape == (2, 3, 256, 256)
        assert mask.shape == (2, 1, 256, 256)

    def test_full_forward_pass_inference(self) -> None:
        """Test complete forward pass in inference mode."""
        archi = AMPArchi(resolution=256, e_ch=64, d_ch=64, ae_dims=256)
        encoder = archi.Encoder()
        inter_src = archi.InterSrc()
        inter_dst = archi.InterDst()
        decoder = archi.Decoder()

        src_img = torch.randn(2, 3, 256, 256)

        enc = encoder(src_img)
        code_src = inter_src(enc)
        code_dst = inter_dst(enc)

        # Test different morph values
        for morph_val in [0.0, 0.3, 0.5, 0.7, 1.0]:
            code = morph_code(code_src, code_dst, morph_value=morph_val, training=False)
            img, mask = decoder(code)
            assert img.shape == (2, 3, 256, 256)

    def test_gradients_flow(self) -> None:
        """Test that gradients flow through the network."""
        archi = AMPArchi(resolution=128, e_ch=32, d_ch=32, ae_dims=128)
        encoder = archi.Encoder()
        inter_src = archi.InterSrc()
        inter_dst = archi.InterDst()
        decoder = archi.Decoder()

        x = torch.randn(1, 3, 128, 128, requires_grad=True)

        enc = encoder(x)
        code_src = inter_src(enc)
        code_dst = inter_dst(enc)
        code = morph_code(code_src, code_dst, morph_value=0.5, training=False)
        img, mask = decoder(code)

        loss = img.mean() + mask.mean()
        loss.backward()

        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_different_resolutions(self) -> None:
        """Test architecture with different resolutions."""
        for res in [128, 192, 256]:
            archi = AMPArchi(resolution=res, e_ch=32, d_ch=32, ae_dims=128)
            encoder = archi.Encoder()
            inter_src = archi.InterSrc()
            inter_dst = archi.InterDst()
            decoder = archi.Decoder()

            x = torch.randn(1, 3, res, res)
            enc = encoder(x)
            code_src = inter_src(enc)
            code_dst = inter_dst(enc)
            code = morph_code(code_src, code_dst, morph_value=0.5, training=False)
            img, mask = decoder(code)

            assert img.shape == (1, 3, res, res)
            assert mask.shape == (1, 1, res, res)

    def test_all_opts_combinations(self) -> None:
        """Test various option combinations."""
        opts_list = ["", "t", "d", "u", "c", "td", "tuc", "tduc"]
        for opts in opts_list:
            archi = AMPArchi(resolution=128, e_ch=32, d_ch=32, ae_dims=64, opts=opts)
            encoder = archi.Encoder()
            inter_src = archi.InterSrc()
            inter_dst = archi.InterDst()
            decoder = archi.Decoder()

            x = torch.randn(1, 3, 128, 128)
            enc = encoder(x)
            code_src = inter_src(enc)
            code_dst = inter_dst(enc)
            code = morph_code(code_src, code_dst, morph_value=0.5, training=False)
            img, mask = decoder(code)

            # All modes output same resolution (double uses pixel shuffle for quality)
            assert img.shape == (1, 3, 128, 128)

    def test_morph_produces_different_outputs(self) -> None:
        """Test that different morph values produce different outputs."""
        archi = AMPArchi(resolution=128, e_ch=32, d_ch=32, ae_dims=64)
        encoder = archi.Encoder()
        inter_src = archi.InterSrc()
        inter_dst = archi.InterDst()
        decoder = archi.Decoder()

        x = torch.randn(1, 3, 128, 128)
        enc = encoder(x)
        code_src = inter_src(enc)
        code_dst = inter_dst(enc)

        code_0 = morph_code(code_src, code_dst, morph_value=0.0, training=False)
        code_1 = morph_code(code_src, code_dst, morph_value=1.0, training=False)

        img_0, _ = decoder(code_0)
        img_1, _ = decoder(code_1)

        # Outputs should be different
        assert not torch.allclose(img_0, img_1)
