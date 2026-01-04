"""Tests for GAN training integration."""

import pytest
import torch

from visagen.training.losses import GANLoss, DiscriminatorLoss, TotalVariationLoss
from visagen.training.dfl_module import DFLModule


class TestGANLoss:
    """Tests for GANLoss."""

    @pytest.fixture
    def vanilla_loss(self):
        return GANLoss(mode="vanilla")

    @pytest.fixture
    def lsgan_loss(self):
        return GANLoss(mode="lsgan")

    @pytest.fixture
    def hinge_loss(self):
        return GANLoss(mode="hinge")

    def test_vanilla_real(self, vanilla_loss):
        """Vanilla loss for real should be low when predictions are high."""
        pred_high = torch.ones(2, 1, 16, 16) * 10  # High logits
        loss = vanilla_loss(pred_high, target_is_real=True)

        assert loss.item() >= 0
        assert loss.dim() == 0  # Scalar

    def test_vanilla_fake(self, vanilla_loss):
        """Vanilla loss for fake should be low when predictions are low."""
        pred_low = torch.ones(2, 1, 16, 16) * -10  # Low logits
        loss = vanilla_loss(pred_low, target_is_real=False)

        assert loss.item() >= 0

    def test_lsgan_loss(self, lsgan_loss):
        """LSGAN should use MSE loss."""
        pred = torch.randn(2, 1, 16, 16)
        loss_real = lsgan_loss(pred, target_is_real=True)
        loss_fake = lsgan_loss(pred, target_is_real=False)

        assert loss_real.item() >= 0
        assert loss_fake.item() >= 0

    def test_hinge_loss_real(self, hinge_loss):
        """Hinge loss for real: -mean(pred)."""
        pred = torch.ones(2, 1, 8, 8) * 2
        loss = hinge_loss(pred, target_is_real=True)

        # -mean(2) = -2
        assert abs(loss.item() + 2.0) < 0.01

    def test_hinge_loss_fake(self, hinge_loss):
        """Hinge loss for fake: mean(pred)."""
        pred = torch.ones(2, 1, 8, 8) * 3
        loss = hinge_loss(pred, target_is_real=False)

        # mean(3) = 3
        assert abs(loss.item() - 3.0) < 0.01

    def test_gradient_flow(self, vanilla_loss):
        """Gradients should flow through GAN loss."""
        pred = torch.randn(2, 1, 16, 16, requires_grad=True)
        loss = vanilla_loss(pred, target_is_real=True)
        loss.backward()

        assert pred.grad is not None

    def test_invalid_mode(self):
        """Invalid mode should raise ValueError."""
        with pytest.raises(ValueError):
            GANLoss(mode="invalid")


class TestDiscriminatorLoss:
    """Tests for DiscriminatorLoss."""

    @pytest.fixture
    def d_loss(self):
        return DiscriminatorLoss(mode="vanilla")

    @pytest.fixture
    def d_loss_hinge(self):
        return DiscriminatorLoss(mode="hinge")

    def test_combined_loss(self, d_loss):
        """Discriminator loss combines real and fake losses."""
        d_real = torch.randn(2, 1, 16, 16)
        d_fake = torch.randn(2, 1, 16, 16)

        loss = d_loss(d_real, d_fake)

        assert loss.item() >= 0
        assert loss.dim() == 0

    def test_perfect_discrimination(self, d_loss):
        """Perfect discrimination should have low loss."""
        d_real = torch.ones(2, 1, 8, 8) * 10  # High = real
        d_fake = torch.ones(2, 1, 8, 8) * -10  # Low = fake

        loss = d_loss(d_real, d_fake)

        # Loss should be very low for perfect discrimination
        assert loss.item() < 0.1

    def test_hinge_discriminator(self, d_loss_hinge):
        """Hinge discriminator loss should work correctly."""
        d_real = torch.ones(2, 1, 8, 8) * 2  # > 1
        d_fake = torch.ones(2, 1, 8, 8) * -2  # < -1

        loss = d_loss_hinge(d_real, d_fake)

        # max(0, 1-2) + max(0, 1+(-2)) = 0 + 0 = 0
        assert loss.item() < 0.1

    def test_gradient_flow(self, d_loss):
        """Gradients should flow through discriminator loss."""
        d_real = torch.randn(2, 1, 16, 16, requires_grad=True)
        d_fake = torch.randn(2, 1, 16, 16, requires_grad=True)

        loss = d_loss(d_real, d_fake)
        loss.backward()

        assert d_real.grad is not None
        assert d_fake.grad is not None


class TestTotalVariationLoss:
    """Tests for TotalVariationLoss."""

    @pytest.fixture
    def tv_loss(self):
        return TotalVariationLoss(weight=1.0)

    def test_smooth_image_low_loss(self, tv_loss):
        """Smooth image should have low TV loss."""
        smooth = torch.ones(2, 3, 64, 64)
        loss = tv_loss(smooth)

        assert loss.item() == 0.0

    def test_noisy_image_high_loss(self, tv_loss):
        """Noisy image should have higher TV loss."""
        noisy = torch.randn(2, 3, 64, 64)
        loss = tv_loss(noisy)

        assert loss.item() > 0

    def test_weight_scaling(self):
        """Weight should scale the loss."""
        x = torch.randn(2, 3, 32, 32)

        loss1 = TotalVariationLoss(weight=1.0)(x)
        loss2 = TotalVariationLoss(weight=0.5)(x)

        assert abs(loss1.item() / 2 - loss2.item()) < 0.01

    def test_gradient_flow(self, tv_loss):
        """Gradients should flow through TV loss."""
        x = torch.randn(2, 3, 32, 32, requires_grad=True)
        loss = tv_loss(x)
        loss.backward()

        assert x.grad is not None


class TestDFLModuleGAN:
    """Tests for DFLModule with GAN training."""

    @pytest.fixture
    def module_ae(self):
        """Module without GAN (autoencoder only)."""
        return DFLModule(
            image_size=64,
            encoder_dims=[16, 32, 64, 128],
            encoder_depths=[1, 1, 1, 1],
            decoder_dims=[128, 64, 32, 16],
            gan_power=0.0,
        )

    @pytest.fixture
    def module_gan(self):
        """Module with GAN training."""
        return DFLModule(
            image_size=64,
            encoder_dims=[16, 32, 64, 128],
            encoder_depths=[1, 1, 1, 1],
            decoder_dims=[128, 64, 32, 16],
            gan_power=0.1,
            gan_patch_size=16,
            gan_base_ch=8,
        )

    def test_ae_no_discriminator(self, module_ae):
        """AE mode should not have discriminator."""
        assert module_ae.discriminator is None
        assert module_ae.gan_power == 0.0

    def test_gan_has_discriminator(self, module_gan):
        """GAN mode should have discriminator."""
        assert module_gan.discriminator is not None
        assert module_gan.gan_power > 0

    def test_gan_has_loss_functions(self, module_gan):
        """GAN mode should have GAN loss functions."""
        assert module_gan.gan_loss is not None
        assert module_gan.d_loss_fn is not None
        assert module_gan.tv_loss is not None

    def test_ae_automatic_optimization(self, module_ae):
        """AE mode should use automatic optimization."""
        # automatic_optimization is True by default
        assert getattr(module_ae, "automatic_optimization", True)

    def test_gan_manual_optimization(self, module_gan):
        """GAN mode should use manual optimization."""
        assert module_gan.automatic_optimization is False

    def test_forward_pass_ae(self, module_ae):
        """Forward pass should work in AE mode."""
        x = torch.randn(2, 3, 64, 64)
        out = module_ae(x)

        assert out.shape == x.shape

    def test_forward_pass_gan(self, module_gan):
        """Forward pass should work in GAN mode."""
        x = torch.randn(2, 3, 64, 64)
        out = module_gan(x)

        assert out.shape == x.shape

    def test_discriminator_forward(self, module_gan):
        """Discriminator should produce outputs."""
        x = torch.randn(2, 3, 64, 64)
        center, final = module_gan.discriminator(x)

        assert center.shape[0] == 2
        assert final.shape == x.shape[:1] + (1,) + x.shape[2:]

    def test_compute_loss(self, module_gan):
        """compute_loss should return loss dict."""
        pred = torch.randn(2, 3, 64, 64)
        target = torch.randn(2, 3, 64, 64)

        total, losses = module_gan.compute_loss(pred, target)

        assert "total" in losses
        assert "dssim" in losses
        assert "l1" in losses

    def test_configure_optimizers_ae(self, module_ae):
        """AE mode should return single optimizer."""
        # Mock trainer
        module_ae.trainer = type("Trainer", (), {"max_epochs": 10})()

        result = module_ae.configure_optimizers()

        assert "optimizer" in result
        assert "lr_scheduler" in result

    def test_configure_optimizers_gan(self, module_gan):
        """GAN mode should return two optimizers."""
        # Mock trainer
        module_gan.trainer = type("Trainer", (), {"max_epochs": 10})()

        result = module_gan.configure_optimizers()

        assert isinstance(result, tuple)
        assert len(result) == 2
        optimizers, schedulers = result
        assert len(optimizers) == 2
        assert len(schedulers) == 2


class TestGANModes:
    """Tests for different GAN loss modes."""

    @pytest.mark.parametrize("mode", ["vanilla", "lsgan", "hinge"])
    def test_gan_modes(self, mode):
        """All GAN modes should work."""
        module = DFLModule(
            image_size=64,
            encoder_dims=[16, 32, 64, 128],
            encoder_depths=[1, 1, 1, 1],
            decoder_dims=[128, 64, 32, 16],
            gan_power=0.1,
            gan_mode=mode,
            gan_patch_size=16,
            gan_base_ch=8,
        )

        assert module.gan_mode == mode
        assert module.gan_loss is not None

        # Test forward pass
        x = torch.randn(2, 3, 64, 64)
        out = module(x)
        assert out.shape == x.shape
