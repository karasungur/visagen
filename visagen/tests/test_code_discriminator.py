"""Tests for CodeDiscriminator."""

import torch

from visagen.models.discriminators.code_discriminator import (
    CodeDiscriminator,
    code_discriminator_loss,
)


class TestCodeDiscriminator:
    """Test CodeDiscriminator class."""

    def test_default_initialization(self) -> None:
        """Test default parameters."""
        disc = CodeDiscriminator()
        assert disc.in_ch == 256
        assert disc.hidden_ch == 256
        assert disc.num_layers == 3

    def test_custom_initialization(self) -> None:
        """Test custom parameters."""
        disc = CodeDiscriminator(in_ch=512, hidden_ch=128, num_layers=4)
        assert disc.in_ch == 512
        assert disc.hidden_ch == 128
        assert disc.num_layers == 4

    def test_output_shape(self) -> None:
        """Test output shape is (B, 1)."""
        disc = CodeDiscriminator(in_ch=256)
        code = torch.randn(4, 256, 8, 8)
        output = disc(code)
        assert output.shape == (4, 1)

    def test_different_spatial_sizes(self) -> None:
        """Test with different spatial sizes."""
        disc = CodeDiscriminator(in_ch=256)
        for size in [4, 8, 16, 32]:
            code = torch.randn(2, 256, size, size)
            output = disc(code)
            assert output.shape == (2, 1)

    def test_different_channel_sizes(self) -> None:
        """Test with different channel sizes."""
        for ch in [64, 128, 256, 512]:
            disc = CodeDiscriminator(in_ch=ch)
            code = torch.randn(2, ch, 8, 8)
            output = disc(code)
            assert output.shape == (2, 1)

    def test_gradient_flow(self) -> None:
        """Test gradients flow through discriminator."""
        disc = CodeDiscriminator(in_ch=256)
        code = torch.randn(2, 256, 8, 8, requires_grad=True)
        output = disc(code)
        loss = output.sum()
        loss.backward()
        assert code.grad is not None
        assert code.grad.abs().sum() > 0

    def test_spectral_norm(self) -> None:
        """Test with spectral normalization."""
        disc = CodeDiscriminator(in_ch=256, use_spectral_norm=True)
        code = torch.randn(2, 256, 8, 8)
        output = disc(code)
        assert output.shape == (2, 1)

    def test_single_layer(self) -> None:
        """Test with single layer."""
        disc = CodeDiscriminator(in_ch=256, num_layers=1)
        code = torch.randn(2, 256, 8, 8)
        output = disc(code)
        assert output.shape == (2, 1)

    def test_many_layers(self) -> None:
        """Test with many layers."""
        disc = CodeDiscriminator(in_ch=256, num_layers=6)
        code = torch.randn(2, 256, 8, 8)
        output = disc(code)
        assert output.shape == (2, 1)

    def test_batch_size_one(self) -> None:
        """Test with batch size 1."""
        disc = CodeDiscriminator(in_ch=256)
        code = torch.randn(1, 256, 8, 8)
        output = disc(code)
        assert output.shape == (1, 1)

    def test_deterministic_output(self) -> None:
        """Test that same input gives same output."""
        disc = CodeDiscriminator(in_ch=256)
        disc.eval()
        code = torch.randn(2, 256, 8, 8)
        output1 = disc(code)
        output2 = disc(code)
        assert torch.allclose(output1, output2)


class TestCodeDiscriminatorLoss:
    """Test code_discriminator_loss function."""

    def test_bce_loss(self) -> None:
        """Test BCE loss computation."""
        disc = CodeDiscriminator(in_ch=256)
        src_code = torch.randn(4, 256, 8, 8)
        dst_code = torch.randn(4, 256, 8, 8)

        g_loss, d_loss = code_discriminator_loss(
            disc, src_code, dst_code, loss_type="bce"
        )

        assert g_loss.shape == ()
        assert d_loss.shape == ()
        assert g_loss.item() >= 0
        assert d_loss.item() >= 0

    def test_hinge_loss(self) -> None:
        """Test hinge loss computation."""
        disc = CodeDiscriminator(in_ch=256)
        src_code = torch.randn(4, 256, 8, 8)
        dst_code = torch.randn(4, 256, 8, 8)

        g_loss, d_loss = code_discriminator_loss(
            disc, src_code, dst_code, loss_type="hinge"
        )

        assert g_loss.shape == ()
        assert d_loss.shape == ()

    def test_gradient_flow_generator(self) -> None:
        """Test gradients flow to generator (src_code)."""
        disc = CodeDiscriminator(in_ch=256)
        src_code = torch.randn(4, 256, 8, 8, requires_grad=True)
        dst_code = torch.randn(4, 256, 8, 8)

        g_loss, _ = code_discriminator_loss(disc, src_code, dst_code)
        g_loss.backward()

        assert src_code.grad is not None
        assert src_code.grad.abs().sum() > 0

    def test_no_gradient_to_dst_in_g_loss(self) -> None:
        """Test dst_code doesn't receive gradients from g_loss."""
        disc = CodeDiscriminator(in_ch=256)
        src_code = torch.randn(4, 256, 8, 8, requires_grad=True)
        dst_code = torch.randn(4, 256, 8, 8, requires_grad=True)

        g_loss, _ = code_discriminator_loss(disc, src_code, dst_code)
        g_loss.backward()

        # dst_code should not have gradients from g_loss
        # (it's detached in the function)
        assert dst_code.grad is None

    def test_invalid_loss_type(self) -> None:
        """Test invalid loss type raises error."""
        disc = CodeDiscriminator(in_ch=256)
        src_code = torch.randn(4, 256, 8, 8)
        dst_code = torch.randn(4, 256, 8, 8)

        raised = False
        try:
            code_discriminator_loss(disc, src_code, dst_code, loss_type="invalid")
        except ValueError as e:
            raised = True
            assert "invalid" in str(e)
        assert raised, "Should raise ValueError"

    def test_different_batch_sizes(self) -> None:
        """Test with different batch sizes."""
        disc = CodeDiscriminator(in_ch=256)
        for batch_size in [1, 2, 4, 8]:
            src_code = torch.randn(batch_size, 256, 8, 8)
            dst_code = torch.randn(batch_size, 256, 8, 8)
            g_loss, d_loss = code_discriminator_loss(disc, src_code, dst_code)
            assert g_loss.shape == ()
            assert d_loss.shape == ()


class TestCodeDiscriminatorTraining:
    """Test CodeDiscriminator in training scenario."""

    def test_training_loop_simulation(self) -> None:
        """Simulate a training loop."""
        disc = CodeDiscriminator(in_ch=128)
        optimizer = torch.optim.Adam(disc.parameters(), lr=1e-4)

        # Simulate training
        for _ in range(3):
            src_code = torch.randn(4, 128, 8, 8)
            dst_code = torch.randn(4, 128, 8, 8)

            _, d_loss = code_discriminator_loss(disc, src_code, dst_code)

            optimizer.zero_grad()
            d_loss.backward()
            optimizer.step()

        # Should complete without error
        assert True

    def test_discriminator_can_distinguish(self) -> None:
        """Test discriminator can learn to distinguish codes."""
        disc = CodeDiscriminator(in_ch=64, hidden_ch=32)
        optimizer = torch.optim.Adam(disc.parameters(), lr=1e-3)

        # Create distinct src and dst distributions
        src_mean = torch.zeros(64, 4, 4)
        dst_mean = torch.ones(64, 4, 4)

        # Train for a few steps
        for _ in range(50):
            src_code = src_mean + torch.randn(8, 64, 4, 4) * 0.1
            dst_code = dst_mean + torch.randn(8, 64, 4, 4) * 0.1

            _, d_loss = code_discriminator_loss(disc, src_code, dst_code)

            optimizer.zero_grad()
            d_loss.backward()
            optimizer.step()

        # After training, discriminator should give different scores
        disc.eval()
        with torch.no_grad():
            src_score = disc(src_mean.unsqueeze(0)).item()
            dst_score = disc(dst_mean.unsqueeze(0)).item()

        # dst should score higher (more real)
        assert dst_score > src_score
