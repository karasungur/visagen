"""Integration tests for DFLModule with new features.

Tests blur_out_mask, true_face_power (CodeDiscriminator), and style losses.
"""

import torch

from visagen.training.dfl_module import DFLModule


class TestDFLModuleBlurOutMask:
    """Test DFLModule with blur_out_mask feature."""

    def test_blur_out_mask_initialization(self) -> None:
        """Test that blur_out_mask flag is stored correctly."""
        module = DFLModule(blur_out_mask=True)
        assert module.blur_out_mask_enabled is True

        module_disabled = DFLModule(blur_out_mask=False)
        assert module_disabled.blur_out_mask_enabled is False

    def test_blur_out_mask_default_disabled(self) -> None:
        """Test that blur_out_mask is disabled by default."""
        module = DFLModule()
        assert module.blur_out_mask_enabled is False


class TestDFLModuleTrueFacePower:
    """Test DFLModule with true_face_power (CodeDiscriminator)."""

    def test_code_discriminator_initialization(self) -> None:
        """Test that CodeDiscriminator is created when true_face_power > 0."""
        module = DFLModule(true_face_power=0.1)
        assert module.code_discriminator is not None
        assert module.true_face_power == 0.1

    def test_code_discriminator_disabled_by_default(self) -> None:
        """Test that CodeDiscriminator is None when true_face_power = 0."""
        module = DFLModule(true_face_power=0.0)
        assert module.code_discriminator is None

    def test_manual_optimization_enabled_with_true_face(self) -> None:
        """Test that manual optimization is enabled with true_face_power."""
        module = DFLModule(true_face_power=0.1)
        assert module.automatic_optimization is False

    def test_code_discriminator_forward(self) -> None:
        """Test CodeDiscriminator forward pass."""
        module = DFLModule(true_face_power=0.1, encoder_dims=[64, 128, 256, 512])
        assert module.code_discriminator is not None

        # Create fake latent code matching encoder output
        latent = torch.randn(2, 512, 4, 4)  # Typical latent shape
        output = module.code_discriminator(latent)

        # Output should be (B, 1) for binary classification
        assert output.shape == (2, 1)


class TestDFLModuleStyleLosses:
    """Test DFLModule with style losses."""

    def test_face_style_weight_stored(self) -> None:
        """Test that face_style_weight is stored correctly."""
        module = DFLModule(face_style_weight=0.1)
        assert module.face_style_weight == 0.1

    def test_bg_style_weight_stored(self) -> None:
        """Test that bg_style_weight is stored correctly."""
        module = DFLModule(bg_style_weight=0.1)
        assert module.bg_style_weight == 0.1

    def test_style_weights_default_zero(self) -> None:
        """Test that style weights are zero by default."""
        module = DFLModule()
        assert module.face_style_weight == 0.0
        assert module.bg_style_weight == 0.0

    def test_compute_loss_with_mask(self) -> None:
        """Test compute_loss accepts mask parameter."""
        module = DFLModule(face_style_weight=0.1, bg_style_weight=0.1)

        pred = torch.randn(2, 3, 256, 256)
        target = torch.randn(2, 3, 256, 256)
        mask = torch.ones(2, 1, 256, 256)

        # Should not raise
        total_loss, loss_dict = module.compute_loss(pred, target, mask=mask)

        assert "total" in loss_dict
        assert total_loss.ndim == 0  # Scalar


class TestDFLModuleConfigureOptimizers:
    """Test configure_optimizers with various feature combinations."""

    def test_ae_only_single_optimizer(self) -> None:
        """Test AE mode returns single optimizer dict."""
        module = DFLModule()
        result = module.configure_optimizers()

        assert isinstance(result, dict)
        assert "optimizer" in result

    def test_gan_two_optimizers(self) -> None:
        """Test GAN mode returns two optimizers."""
        module = DFLModule(gan_power=0.1)
        result = module.configure_optimizers()

        assert isinstance(result, tuple)
        optimizers, schedulers = result
        assert len(optimizers) == 2
        assert len(schedulers) == 2

    def test_true_face_only_two_optimizers(self) -> None:
        """Test true_face_power only mode returns two optimizers."""
        module = DFLModule(true_face_power=0.1)
        result = module.configure_optimizers()

        assert isinstance(result, tuple)
        optimizers, schedulers = result
        assert len(optimizers) == 2  # generator + code_discriminator
        assert len(schedulers) == 2

    def test_gan_plus_true_face_three_optimizers(self) -> None:
        """Test GAN + true_face mode returns three optimizers."""
        module = DFLModule(gan_power=0.1, true_face_power=0.1)
        result = module.configure_optimizers()

        assert isinstance(result, tuple)
        optimizers, schedulers = result
        assert len(optimizers) == 3  # generator + discriminator + code_discriminator
        assert len(schedulers) == 3


class TestDFLModuleForwardPass:
    """Test DFLModule forward pass with new features."""

    def test_forward_unchanged(self) -> None:
        """Test that forward pass works with new parameters."""
        module = DFLModule(
            blur_out_mask=True,
            true_face_power=0.1,
            face_style_weight=0.1,
            bg_style_weight=0.1,
        )

        x = torch.randn(2, 3, 256, 256)
        output = module(x)

        assert output.shape == (2, 3, 256, 256)

    def test_gradient_flow_with_features(self) -> None:
        """Test gradients flow through model with new features."""
        module = DFLModule(
            face_style_weight=0.1,
            bg_style_weight=0.1,
        )

        x = torch.randn(1, 3, 256, 256, requires_grad=True)
        mask = torch.ones(1, 1, 256, 256)

        output = module(x)
        loss, _ = module.compute_loss(output, x, mask=mask)
        loss.backward()

        assert x.grad is not None
        assert x.grad.abs().sum() > 0


class TestDFLModuleCombinedFeatures:
    """Test DFLModule with multiple features enabled."""

    def test_all_features_enabled(self) -> None:
        """Test initialization with all new features enabled."""
        module = DFLModule(
            blur_out_mask=True,
            true_face_power=0.1,
            face_style_weight=0.1,
            bg_style_weight=0.1,
            gan_power=0.1,
        )

        assert module.blur_out_mask_enabled is True
        assert module.true_face_power == 0.1
        assert module.face_style_weight == 0.1
        assert module.bg_style_weight == 0.1
        assert module.code_discriminator is not None
        assert module.discriminator is not None

    def test_compute_loss_all_losses(self) -> None:
        """Test compute_loss with all loss types."""
        module = DFLModule(
            dssim_weight=10.0,
            l1_weight=10.0,
            face_style_weight=0.1,
            bg_style_weight=0.1,
        )

        pred = torch.randn(2, 3, 256, 256)
        target = torch.randn(2, 3, 256, 256)
        mask = torch.ones(2, 1, 256, 256)

        total_loss, loss_dict = module.compute_loss(pred, target, mask=mask)

        assert "dssim" in loss_dict
        assert "l1" in loss_dict
        assert "face_style" in loss_dict
        assert "bg_style" in loss_dict
        assert "total" in loss_dict


class TestDFLModuleHyperparameters:
    """Test that hyperparameters are saved correctly."""

    def test_new_params_in_hparams(self) -> None:
        """Test new parameters are saved in hparams."""
        module = DFLModule(
            blur_out_mask=True,
            true_face_power=0.5,
            face_style_weight=0.2,
            bg_style_weight=0.3,
        )

        # Check hparams contains new values
        assert module.hparams.get("blur_out_mask") is True
        assert module.hparams.get("true_face_power") == 0.5
        assert module.hparams.get("face_style_weight") == 0.2
        assert module.hparams.get("bg_style_weight") == 0.3
