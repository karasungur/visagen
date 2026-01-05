"""Tests for Adaptive Instance Normalization (AdaIN) layer."""

import torch

from visagen.models.layers.adain import AdaINResBlock, AdaptiveInstanceNorm2d


class TestAdaptiveInstanceNorm2d:
    """Tests for AdaptiveInstanceNorm2d layer."""

    def test_shape_preservation(self) -> None:
        """Test that AdaIN preserves spatial dimensions."""
        batch, channels, height, width = 2, 64, 32, 32
        style_dim = 128

        content = torch.randn(batch, channels, height, width)
        style = torch.randn(batch, style_dim)

        adain = AdaptiveInstanceNorm2d(channels, style_dim)
        output = adain(content, style)

        assert output.shape == content.shape

    def test_different_batch_sizes(self) -> None:
        """Test AdaIN with different batch sizes."""
        channels, height, width = 64, 32, 32
        style_dim = 128

        adain = AdaptiveInstanceNorm2d(channels, style_dim)

        for batch_size in [1, 2, 4, 8]:
            content = torch.randn(batch_size, channels, height, width)
            style = torch.randn(batch_size, style_dim)
            output = adain(content, style)
            assert output.shape == (batch_size, channels, height, width)

    def test_different_spatial_sizes(self) -> None:
        """Test AdaIN with different spatial dimensions."""
        batch, channels = 2, 64
        style_dim = 128

        adain = AdaptiveInstanceNorm2d(channels, style_dim)

        for size in [16, 32, 64, 128]:
            content = torch.randn(batch, channels, size, size)
            style = torch.randn(batch, style_dim)
            output = adain(content, style)
            assert output.shape == (batch, channels, size, size)

    def test_gradient_flow(self) -> None:
        """Test that gradients flow through AdaIN."""
        batch, channels, height, width = 2, 64, 32, 32
        style_dim = 128

        content = torch.randn(batch, channels, height, width, requires_grad=True)
        style = torch.randn(batch, style_dim, requires_grad=True)

        adain = AdaptiveInstanceNorm2d(channels, style_dim)
        output = adain(content, style)
        loss = output.sum()
        loss.backward()

        assert content.grad is not None
        assert style.grad is not None
        assert not torch.isnan(content.grad).any()
        assert not torch.isnan(style.grad).any()

    def test_identity_initialization(self) -> None:
        """Test that initial gamma is ~1 and beta is ~0."""
        channels = 64
        style_dim = 128

        adain = AdaptiveInstanceNorm2d(channels, style_dim)

        # Check bias initialization
        assert torch.allclose(adain.gamma_fc.bias, torch.zeros(channels), atol=1e-5)
        assert torch.allclose(adain.beta_fc.bias, torch.zeros(channels), atol=1e-5)

    def test_normalization_effect(self) -> None:
        """Test that content is normalized."""
        batch, channels, height, width = 2, 64, 32, 32
        style_dim = 128

        # Create content with specific mean/std
        content = torch.randn(batch, channels, height, width) * 10 + 5

        # Non-zero style that produces gamma=1, beta=0
        style = torch.ones(batch, style_dim)

        adain = AdaptiveInstanceNorm2d(channels, style_dim)
        # Reset weights so that: gamma = 1/style_dim * sum(style) = 1.0
        # and beta = 0
        with torch.no_grad():
            # gamma_fc: all 1/style_dim weights, zero bias
            adain.gamma_fc.weight.fill_(1.0 / style_dim)
            adain.gamma_fc.bias.zero_()
            # beta_fc: zero weights and bias
            adain.beta_fc.weight.zero_()
            adain.beta_fc.bias.zero_()

        output = adain(content, style)

        # Check approximate normalization per instance
        for b in range(batch):
            for c in range(channels):
                instance = output[b, c]
                mean = instance.mean()
                std = instance.std()
                # Should be approximately normalized
                assert abs(mean.item()) < 0.15
                assert abs(std.item() - 1.0) < 0.15


class TestAdaINResBlock:
    """Tests for AdaINResBlock."""

    def test_shape_preservation(self) -> None:
        """Test that AdaINResBlock preserves shape when channels match."""
        batch, channels, height, width = 2, 64, 32, 32
        style_dim = 128

        x = torch.randn(batch, channels, height, width)
        style = torch.randn(batch, style_dim)

        block = AdaINResBlock(channels, channels, style_dim)
        output = block(x, style)

        assert output.shape == x.shape

    def test_channel_change(self) -> None:
        """Test AdaINResBlock with different input/output channels."""
        batch, height, width = 2, 32, 32
        in_channels, out_channels = 64, 128
        style_dim = 256

        x = torch.randn(batch, in_channels, height, width)
        style = torch.randn(batch, style_dim)

        block = AdaINResBlock(in_channels, out_channels, style_dim)
        output = block(x, style)

        assert output.shape == (batch, out_channels, height, width)

    def test_upsample(self) -> None:
        """Test AdaINResBlock with upsampling."""
        batch, channels, height, width = 2, 64, 16, 16
        style_dim = 128

        x = torch.randn(batch, channels, height, width)
        style = torch.randn(batch, style_dim)

        block = AdaINResBlock(channels, channels, style_dim, upsample=True)
        output = block(x, style)

        assert output.shape == (batch, channels, height * 2, width * 2)

    def test_gradient_flow(self) -> None:
        """Test gradient flow through AdaINResBlock."""
        batch, channels, height, width = 2, 64, 32, 32
        style_dim = 128

        x = torch.randn(batch, channels, height, width, requires_grad=True)
        style = torch.randn(batch, style_dim, requires_grad=True)

        block = AdaINResBlock(channels, channels, style_dim)
        output = block(x, style)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert style.grad is not None

    def test_multiple_blocks(self) -> None:
        """Test stacking multiple AdaINResBlocks."""
        batch, height, width = 2, 32, 32
        style_dim = 128

        x = torch.randn(batch, 64, height, width)
        style = torch.randn(batch, style_dim)

        blocks = [
            AdaINResBlock(64, 128, style_dim),
            AdaINResBlock(128, 256, style_dim),
            AdaINResBlock(256, 256, style_dim),
        ]

        output = x
        for block in blocks:
            output = block(output, style)

        assert output.shape == (batch, 256, height, width)
