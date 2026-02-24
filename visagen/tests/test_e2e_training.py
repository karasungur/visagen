"""
End-to-end training integration tests.

These tests verify that experimental models can be instantiated
through TrainingModule and perform forward passes correctly.
"""

import pytest
import torch

from visagen.training.training_module import TrainingModule

# =============================================================================
# Model Type Integration Tests
# =============================================================================


class TestModelTypeIntegration:
    """Test different model types can be instantiated and forward."""

    @pytest.fixture
    def batch_size(self):
        """Standard batch size for tests."""
        return 2

    @pytest.fixture
    def input_64(self, batch_size):
        """Create 64x64 input tensor."""
        return torch.randn(batch_size, 3, 64, 64)

    @pytest.fixture
    def input_32(self, batch_size):
        """Create 32x32 input tensor."""
        return torch.randn(batch_size, 3, 32, 32)

    def test_standard_model_forward(self, input_64, batch_size):
        """Test standard model forward pass."""
        model = TrainingModule(
            image_size=64,
            encoder_dims=[32, 64],
            encoder_depths=[1, 1],
            decoder_dims=[64, 32],
            model_type="standard",
        )

        with torch.no_grad():
            result = model(input_64)
        output = result[0] if isinstance(result, tuple) else result

        assert output.shape == (batch_size, 3, 64, 64)

    def test_standard_model_has_encoder_decoder(self):
        """Test standard model has encoder and decoder."""
        model = TrainingModule(
            image_size=64,
            encoder_dims=[32, 64],
            encoder_depths=[1, 1],
            decoder_dims=[64, 32],
            model_type="standard",
        )

        assert model.encoder is not None
        assert model.decoder is not None
        assert model.model is None
        assert model.model_type == "standard"

    def test_diffusion_model_forward(self, input_64, batch_size):
        """Test diffusion model forward pass."""
        model = TrainingModule(
            image_size=64,
            encoder_dims=[32, 64, 128, 256],
            encoder_depths=[1, 1, 2, 1],
            decoder_dims=[256, 128, 64, 32],
            model_type="diffusion",
            use_pretrained_vae=False,  # Use lite encoder for testing
        )

        with torch.no_grad():
            output = model(input_64)

        # Output may have different size due to decoder architecture
        assert output.shape[0] == batch_size
        assert output.shape[1] == 3  # RGB channels

    def test_diffusion_model_has_unified_model(self):
        """Test diffusion model uses unified DiffusionAutoEncoder."""
        model = TrainingModule(
            image_size=64,
            encoder_dims=[32, 64, 128, 256],
            encoder_depths=[1, 1, 2, 1],
            decoder_dims=[256, 128, 64, 32],
            model_type="diffusion",
            use_pretrained_vae=False,
        )

        assert model.model is not None
        assert model.encoder is None
        assert model.decoder is None
        assert model.model_type == "diffusion"

    def test_eg3d_model_forward(self, input_32, batch_size):
        """Test EG3D model forward pass."""
        model = TrainingModule(
            image_size=32,
            encoder_dims=[32, 64, 128, 256],
            encoder_depths=[1, 1, 2, 1],
            model_type="eg3d",
            eg3d_latent_dim=128,
            eg3d_plane_channels=8,
            eg3d_render_resolution=16,
        )

        with torch.no_grad():
            output = model(input_32)

        assert output.shape == (batch_size, 3, 32, 32)

    def test_eg3d_model_structure(self):
        """Test EG3D model has encoder and generator."""
        model = TrainingModule(
            image_size=32,
            encoder_dims=[32, 64, 128, 256],
            encoder_depths=[1, 1, 2, 1],
            model_type="eg3d",
            eg3d_latent_dim=128,
            eg3d_plane_channels=8,
            eg3d_render_resolution=16,
        )

        assert model.encoder is not None  # EG3DEncoder
        assert model.model is not None  # EG3DGenerator
        assert model.decoder is None
        assert model.model_type == "eg3d"


# =============================================================================
# Texture Loss Integration Tests
# =============================================================================


class TestTextureLossIntegration:
    """Test texture loss integration in TrainingModule."""

    def test_texture_loss_property_lazy_loading(self):
        """Test texture loss is lazy loaded."""
        model = TrainingModule(
            image_size=64,
            encoder_dims=[32, 64],
            encoder_depths=[1, 1],
            decoder_dims=[64, 32],
            model_type="standard",
            diffusion_texture_weight=0.0,
        )

        # Should be None when weight is 0
        assert model._texture_loss is None
        assert model.texture_loss is None

    def test_texture_loss_enabled_when_weight_positive(self):
        """Test texture loss is loaded when weight > 0."""
        pytest.importorskip(
            "torchvision", reason="torchvision required for texture loss"
        )

        model = TrainingModule(
            image_size=64,
            encoder_dims=[32, 64, 128, 256],
            encoder_depths=[1, 1, 2, 1],
            decoder_dims=[256, 128, 64, 32],
            model_type="diffusion",
            use_pretrained_vae=False,
            diffusion_texture_weight=5.0,
        )

        assert model.texture_weight == 5.0
        # Access the property to trigger lazy loading
        texture_loss = model.texture_loss
        assert texture_loss is not None

    def test_compute_loss_includes_texture(self):
        """Test compute_loss includes texture loss when enabled."""
        pytest.importorskip(
            "torchvision", reason="torchvision required for texture loss"
        )

        model = TrainingModule(
            image_size=64,
            encoder_dims=[32, 64, 128, 256],
            encoder_depths=[1, 1, 2, 1],
            decoder_dims=[256, 128, 64, 32],
            model_type="standard",
            diffusion_texture_weight=5.0,
        )

        pred = torch.randn(2, 3, 64, 64)
        target = torch.randn(2, 3, 64, 64)

        total, losses = model.compute_loss(pred, target)

        assert "texture" in losses
        assert losses["texture"] > 0


# =============================================================================
# Hyperparameter Configuration Tests
# =============================================================================


class TestHyperparameterConfiguration:
    """Test hyperparameter saving and access."""

    def test_model_type_in_hparams(self):
        """Test model_type is saved in hyperparameters."""
        model = TrainingModule(
            image_size=64,
            encoder_dims=[32, 64],
            encoder_depths=[1, 1],
            decoder_dims=[64, 32],
            model_type="diffusion",
            use_pretrained_vae=False,
        )

        assert model.hparams.model_type == "diffusion"
        assert model.hparams.use_pretrained_vae is False

    def test_eg3d_params_in_hparams(self):
        """Test EG3D parameters are saved in hyperparameters."""
        model = TrainingModule(
            image_size=32,
            encoder_dims=[32, 64, 128, 256],
            encoder_depths=[1, 1, 2, 1],
            model_type="eg3d",
            eg3d_latent_dim=256,
            eg3d_plane_channels=16,
            eg3d_render_resolution=32,
        )

        assert model.hparams.eg3d_latent_dim == 256
        assert model.hparams.eg3d_plane_channels == 16
        assert model.hparams.eg3d_render_resolution == 32


# =============================================================================
# Optimizer Configuration Tests
# =============================================================================


class TestOptimizerConfiguration:
    """Test optimizer configuration for different model types."""

    def test_standard_model_optimizer_params(self):
        """Test standard model optimizer uses encoder+decoder params."""
        model = TrainingModule(
            image_size=64,
            encoder_dims=[32, 64],
            encoder_depths=[1, 1],
            decoder_dims=[64, 32],
            model_type="standard",
        )

        # Count parameters
        encoder_params = sum(p.numel() for p in model.encoder.parameters())
        decoder_params = sum(p.numel() for p in model.decoder.parameters())
        total_expected = encoder_params + decoder_params

        # Model should have these trainable params
        total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert total_trainable == total_expected

    def test_diffusion_model_optimizer_params(self):
        """Test diffusion model optimizer uses unified model params."""
        model = TrainingModule(
            image_size=64,
            encoder_dims=[32, 64, 128, 256],
            encoder_depths=[1, 1, 2, 1],
            decoder_dims=[256, 128, 64, 32],
            model_type="diffusion",
            use_pretrained_vae=False,
        )

        # Model should have trainable params only in self.model
        model_params = sum(p.numel() for p in model.model.parameters())
        total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert total_trainable == model_params
