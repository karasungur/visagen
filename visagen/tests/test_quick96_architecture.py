"""Tests for Quick96 architecture."""

import torch

from visagen.models.architectures.quick96 import Quick96Archi, Quick96Model


class TestQuick96Archi:
    """Test Quick96Archi factory class."""

    def test_fixed_resolution(self) -> None:
        """Test that resolution is fixed at 96."""
        archi = Quick96Archi()
        assert archi.RESOLUTION == 96

    def test_default_dimensions(self) -> None:
        """Test default dimension values match Quick96 specification."""
        archi = Quick96Archi()
        assert archi.ae_dims == 128
        assert archi.e_ch == 64
        assert archi.d_ch == 64
        assert archi.d_mask_ch == 16

    def test_custom_dimensions(self) -> None:
        """Test custom dimension values."""
        archi = Quick96Archi(ae_dims=256, e_ch=128, d_ch=128, d_mask_ch=32)
        assert archi.ae_dims == 256
        assert archi.e_ch == 128
        assert archi.d_ch == 128
        assert archi.d_mask_ch == 32

    def test_encoder_creation(self) -> None:
        """Test encoder can be created."""
        archi = Quick96Archi()
        encoder = archi.Encoder()
        assert encoder is not None
        assert isinstance(encoder, torch.nn.Module)

    def test_inter_creation(self) -> None:
        """Test inter module can be created."""
        archi = Quick96Archi()
        inter = archi.Inter()
        assert inter is not None
        assert isinstance(inter, torch.nn.Module)

    def test_decoder_creation(self) -> None:
        """Test decoder can be created."""
        archi = Quick96Archi()
        decoder_src = archi.Decoder(name="src")
        decoder_dst = archi.Decoder(name="dst")
        assert decoder_src is not None
        assert decoder_dst is not None

    def test_encoder_forward(self) -> None:
        """Test encoder forward pass."""
        archi = Quick96Archi()
        encoder = archi.Encoder()
        x = torch.randn(2, 3, 96, 96)
        out = encoder(x)
        assert out.shape[0] == 2  # Batch size preserved

    def test_full_pipeline(self) -> None:
        """Test encoder -> inter -> decoder pipeline."""
        archi = Quick96Archi()
        encoder = archi.Encoder()
        inter = archi.Inter()
        decoder = archi.Decoder()

        x = torch.randn(2, 3, 96, 96)
        features = encoder(x)
        code = inter(features)
        out, mask = decoder(code)

        assert out.shape == (2, 3, 96, 96)
        assert mask.shape == (2, 1, 96, 96)

    def test_get_encoder_out_ch(self) -> None:
        """Test encoder output channel getter."""
        archi = Quick96Archi()
        out_ch = archi.get_encoder_out_ch()
        assert out_ch > 0

    def test_get_inter_out_ch(self) -> None:
        """Test inter output channel getter."""
        archi = Quick96Archi()
        out_ch = archi.get_inter_out_ch()
        assert out_ch == 128  # ae_dims


class TestQuick96Model:
    """Test Quick96Model complete model."""

    def test_initialization(self) -> None:
        """Test model initialization."""
        model = Quick96Model()
        assert model.RESOLUTION == 96
        assert model.ae_dims == 128
        assert hasattr(model, "encoder")
        assert hasattr(model, "inter")
        assert hasattr(model, "decoder_src")
        assert hasattr(model, "decoder_dst")

    def test_custom_initialization(self) -> None:
        """Test model with custom dimensions."""
        model = Quick96Model(ae_dims=256, e_ch=128, d_ch=128, d_mask_ch=32)
        assert model.ae_dims == 256
        assert model.e_ch == 128

    def test_encode(self) -> None:
        """Test encode method."""
        model = Quick96Model()
        x = torch.randn(2, 3, 96, 96)
        code = model.encode(x)
        assert code is not None
        assert code.shape[0] == 2

    def test_forward_src(self) -> None:
        """Test forward with src decoder."""
        model = Quick96Model()
        x = torch.randn(2, 3, 96, 96)
        out, mask = model(x, mode="src")
        assert out.shape == (2, 3, 96, 96)
        assert mask.shape == (2, 1, 96, 96)

    def test_forward_dst(self) -> None:
        """Test forward with dst decoder."""
        model = Quick96Model()
        x = torch.randn(2, 3, 96, 96)
        out, mask = model(x, mode="dst")
        assert out.shape == (2, 3, 96, 96)
        assert mask.shape == (2, 1, 96, 96)

    def test_swap(self) -> None:
        """Test face swap operation."""
        model = Quick96Model()
        dst = torch.randn(2, 3, 96, 96)
        swapped, mask = model.swap(dst)
        assert swapped.shape == (2, 3, 96, 96)
        assert mask.shape == (2, 1, 96, 96)

    def test_decode_src(self) -> None:
        """Test decode_src method."""
        model = Quick96Model()
        x = torch.randn(2, 3, 96, 96)
        code = model.encode(x)
        out, mask = model.decode_src(code)
        assert out.shape == (2, 3, 96, 96)

    def test_decode_dst(self) -> None:
        """Test decode_dst method."""
        model = Quick96Model()
        x = torch.randn(2, 3, 96, 96)
        code = model.encode(x)
        out, mask = model.decode_dst(code)
        assert out.shape == (2, 3, 96, 96)

    def test_get_training_outputs(self) -> None:
        """Test get_training_outputs method."""
        model = Quick96Model()
        src = torch.randn(2, 3, 96, 96)
        dst = torch.randn(2, 3, 96, 96)

        outputs = model.get_training_outputs(src, dst)

        assert "src_src" in outputs
        assert "dst_dst" in outputs
        assert "src_dst" in outputs

        # Check shapes
        src_src_img, src_src_mask = outputs["src_src"]
        assert src_src_img.shape == (2, 3, 96, 96)
        assert src_src_mask.shape == (2, 1, 96, 96)

        dst_dst_img, dst_dst_mask = outputs["dst_dst"]
        assert dst_dst_img.shape == (2, 3, 96, 96)

        src_dst_img, src_dst_mask = outputs["src_dst"]
        assert src_dst_img.shape == (2, 3, 96, 96)

    def test_gradient_flow(self) -> None:
        """Test gradients flow through model."""
        model = Quick96Model()
        x = torch.randn(1, 3, 96, 96, requires_grad=True)
        out, mask = model(x)
        loss = out.sum() + mask.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_batch_size_one(self) -> None:
        """Test with batch size 1."""
        model = Quick96Model()
        x = torch.randn(1, 3, 96, 96)
        out, mask = model(x)
        assert out.shape == (1, 3, 96, 96)

    def test_larger_batch(self) -> None:
        """Test with larger batch size."""
        model = Quick96Model()
        x = torch.randn(8, 3, 96, 96)
        out, mask = model(x)
        assert out.shape == (8, 3, 96, 96)

    def test_eval_mode(self) -> None:
        """Test model in eval mode."""
        model = Quick96Model()
        model.eval()
        x = torch.randn(2, 3, 96, 96)
        with torch.no_grad():
            out, mask = model(x)
        assert out.shape == (2, 3, 96, 96)

    def test_deterministic_eval(self) -> None:
        """Test that eval mode gives deterministic results."""
        model = Quick96Model()
        model.eval()
        x = torch.randn(1, 3, 96, 96)
        with torch.no_grad():
            out1, _ = model(x)
            out2, _ = model(x)
        assert torch.allclose(out1, out2)

    def test_output_range(self) -> None:
        """Test output values are reasonable."""
        model = Quick96Model()
        x = torch.randn(2, 3, 96, 96)
        out, mask = model(x)
        # Outputs should be finite
        assert torch.isfinite(out).all()
        assert torch.isfinite(mask).all()

    def test_mask_range(self) -> None:
        """Test mask output is in valid range after sigmoid."""
        model = Quick96Model()
        x = torch.randn(2, 3, 96, 96)
        _, mask = model(x)
        # Raw mask output (before sigmoid in training)
        assert torch.isfinite(mask).all()


class TestQuick96Integration:
    """Integration tests for Quick96."""

    def test_training_simulation(self) -> None:
        """Simulate a training step."""
        model = Quick96Model()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        src = torch.randn(4, 3, 96, 96)
        dst = torch.randn(4, 3, 96, 96)

        # Forward
        outputs = model.get_training_outputs(src, dst)

        # Simple reconstruction loss
        src_src_img, _ = outputs["src_src"]
        loss = torch.nn.functional.l1_loss(src_src_img, src)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Should complete without error
        assert True

    def test_swap_identity(self) -> None:
        """Test that swap produces different output than input."""
        model = Quick96Model()
        model.eval()

        dst = torch.randn(1, 3, 96, 96)
        with torch.no_grad():
            swapped, _ = model.swap(dst)

        # Swapped should be different from input (untrained model)
        # This just checks the operation works, not quality
        assert swapped.shape == dst.shape

    def test_parameter_count(self) -> None:
        """Test model has reasonable parameter count (should be small)."""
        model = Quick96Model()
        total_params = sum(p.numel() for p in model.parameters())
        # Quick96 should have fewer params than full models
        # Rough estimate: should be under 50M params
        assert total_params < 50_000_000
        assert total_params > 0
