"""
Tests for training loss functions.

These tests verify the loss functions compute correctly
and produce expected gradients.
"""

import pytest
import torch
import torch.nn.functional as F

from visagen.training.losses import (
    CombinedLoss,
    DSSIMLoss,
    EyesMouthLoss,
    GazeLoss,
    MultiScaleDSSIMLoss,
    StyleLoss,
)


class TestDSSIMLoss:
    """Tests for DSSIM loss."""

    @pytest.fixture
    def dssim_loss(self):
        """Create DSSIM loss instance."""
        return DSSIMLoss(filter_size=11)

    def test_identical_images_low_loss(self, dssim_loss):
        """Identical images should have near-zero DSSIM loss."""
        x = torch.rand(2, 3, 64, 64)
        loss = dssim_loss(x, x)

        assert loss.item() < 0.01  # Should be very close to 0

    def test_different_images_high_loss(self, dssim_loss):
        """Very different images should have high DSSIM loss."""
        x = torch.rand(2, 3, 64, 64)
        y = 1 - x  # Inverted image

        loss = dssim_loss(x, y)

        assert loss.item() > 0.3  # Should be significantly higher

    def test_output_shape(self, dssim_loss):
        """Output should be a scalar."""
        x = torch.rand(2, 3, 64, 64)
        y = torch.rand(2, 3, 64, 64)

        loss = dssim_loss(x, y)

        assert loss.dim() == 0  # Scalar

    def test_gradient_flow(self, dssim_loss):
        """Gradients should flow through the loss."""
        x = torch.rand(2, 3, 64, 64, requires_grad=True)
        y = torch.rand(2, 3, 64, 64)

        loss = dssim_loss(x, y)
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_range_0_to_1(self, dssim_loss):
        """DSSIM should be in range [0, 1]."""
        for _ in range(5):
            x = torch.rand(2, 3, 64, 64)
            y = torch.rand(2, 3, 64, 64)

            loss = dssim_loss(x, y)

            assert 0 <= loss.item() <= 1

    def test_symmetry(self, dssim_loss):
        """DSSIM(x, y) should equal DSSIM(y, x)."""
        x = torch.rand(2, 3, 64, 64)
        y = torch.rand(2, 3, 64, 64)

        loss_xy = dssim_loss(x, y)
        loss_yx = dssim_loss(y, x)

        torch.testing.assert_close(loss_xy, loss_yx, atol=1e-6, rtol=1e-6)


class TestMultiScaleDSSIMLoss:
    """Tests for multi-scale DSSIM loss."""

    @pytest.fixture
    def ms_dssim_loss(self):
        """Create multi-scale DSSIM loss instance."""
        return MultiScaleDSSIMLoss(filter_sizes=(7, 11))

    def test_identical_images_low_loss(self, ms_dssim_loss):
        """Identical images should have near-zero loss."""
        x = torch.rand(2, 3, 64, 64)
        loss = ms_dssim_loss(x, x)

        assert loss.item() < 0.01

    def test_output_shape(self, ms_dssim_loss):
        """Output should be a scalar."""
        x = torch.rand(2, 3, 64, 64)
        y = torch.rand(2, 3, 64, 64)

        loss = ms_dssim_loss(x, y)

        assert loss.dim() == 0

    def test_gradient_flow(self, ms_dssim_loss):
        """Gradients should flow through the loss."""
        x = torch.rand(2, 3, 64, 64, requires_grad=True)
        y = torch.rand(2, 3, 64, 64)

        loss = ms_dssim_loss(x, y)
        loss.backward()

        assert x.grad is not None


class TestEyesMouthLoss:
    """Tests for eyes/mouth priority loss."""

    @pytest.fixture
    def eyes_mouth_loss(self):
        """Create eyes/mouth loss instance."""
        return EyesMouthLoss(weight_multiplier=30.0)

    def test_without_landmarks(self, eyes_mouth_loss):
        """Without landmarks, should compute uniform L1 loss."""
        x = torch.rand(2, 3, 64, 64)
        y = torch.rand(2, 3, 64, 64)

        loss = eyes_mouth_loss(x, y, landmarks=None)
        expected = F.l1_loss(x, y)

        torch.testing.assert_close(loss, expected)

    def test_with_landmarks(self, eyes_mouth_loss):
        """With landmarks, loss should be different from uniform."""
        x = torch.rand(2, 3, 64, 64)
        y = torch.rand(2, 3, 64, 64)

        # Create dummy landmarks (68 points)
        landmarks = torch.rand(2, 68, 2) * 64

        loss = eyes_mouth_loss(x, y, landmarks=landmarks)

        assert loss.item() >= 0

    def test_output_shape(self, eyes_mouth_loss):
        """Output should be a scalar."""
        x = torch.rand(2, 3, 64, 64)
        y = torch.rand(2, 3, 64, 64)

        loss = eyes_mouth_loss(x, y)

        assert loss.dim() == 0

    def test_gradient_flow(self, eyes_mouth_loss):
        """Gradients should flow through the loss."""
        x = torch.rand(2, 3, 64, 64, requires_grad=True)
        y = torch.rand(2, 3, 64, 64)

        loss = eyes_mouth_loss(x, y)
        loss.backward()

        assert x.grad is not None


class TestGazeLoss:
    """Tests for GazeLoss."""

    @pytest.fixture
    def gaze_loss(self):
        """Create GazeLoss instance."""
        return GazeLoss(eye_size=32)

    def test_without_landmarks_fallback(self, gaze_loss):
        """Without landmarks, should fall back to L1 loss."""
        x = torch.rand(2, 3, 64, 64)
        y = torch.rand(2, 3, 64, 64)

        loss = gaze_loss(x, y, landmarks=None)
        expected = F.l1_loss(x, y)

        torch.testing.assert_close(loss, expected)

    def test_with_landmarks(self, gaze_loss):
        """With landmarks, should compute eye-specific loss."""
        x = torch.rand(2, 3, 64, 64)
        y = torch.rand(2, 3, 64, 64)

        # Create landmarks with eyes roughly in upper middle region
        landmarks = torch.zeros(2, 68, 2)
        # Left eye (indices 36-41): around (20, 20)
        landmarks[:, 36:42, 0] = torch.tensor([16, 18, 20, 22, 24, 20])
        landmarks[:, 36:42, 1] = torch.tensor([18, 16, 16, 18, 20, 20])
        # Right eye (indices 42-47): around (44, 20)
        landmarks[:, 42:48, 0] = torch.tensor([40, 42, 44, 46, 48, 44])
        landmarks[:, 42:48, 1] = torch.tensor([18, 16, 16, 18, 20, 20])

        loss = gaze_loss(x, y, landmarks=landmarks)

        assert loss.item() >= 0
        assert loss.dim() == 0  # Scalar

    def test_identical_images_low_loss(self, gaze_loss):
        """Identical images should have near-zero loss with landmarks."""
        x = torch.rand(2, 3, 64, 64)

        # Landmarks with varying y values for valid bounding boxes
        landmarks = torch.zeros(2, 68, 2)
        landmarks[:, 36:42, 0] = torch.tensor([16, 18, 20, 22, 24, 20])
        landmarks[:, 36:42, 1] = torch.tensor([18, 16, 16, 18, 22, 22])
        landmarks[:, 42:48, 0] = torch.tensor([40, 42, 44, 46, 48, 44])
        landmarks[:, 42:48, 1] = torch.tensor([18, 16, 16, 18, 22, 22])

        loss = gaze_loss(x, x, landmarks=landmarks)

        assert loss.item() < 1e-6

    def test_output_shape(self, gaze_loss):
        """Output should be a scalar."""
        x = torch.rand(2, 3, 64, 64)
        y = torch.rand(2, 3, 64, 64)

        # Landmarks with valid eye regions (varying y coords)
        landmarks = torch.zeros(2, 68, 2)
        landmarks[:, 36:42, 0] = torch.tensor([16, 18, 20, 22, 24, 20])
        landmarks[:, 36:42, 1] = torch.tensor([18, 16, 16, 18, 22, 22])
        landmarks[:, 42:48, 0] = torch.tensor([40, 42, 44, 46, 48, 44])
        landmarks[:, 42:48, 1] = torch.tensor([18, 16, 16, 18, 22, 22])

        loss = gaze_loss(x, y, landmarks=landmarks)

        assert loss.dim() == 0  # Scalar

    def test_gradient_flow(self, gaze_loss):
        """Gradients should flow through the loss."""
        x = torch.rand(2, 3, 64, 64, requires_grad=True)
        y = torch.rand(2, 3, 64, 64)

        # Landmarks need valid eye bounding boxes (y coords must vary for height > 0)
        landmarks = torch.zeros(2, 68, 2)
        # Left eye with varying y values
        landmarks[:, 36:42, 0] = torch.tensor([16, 18, 20, 22, 24, 20])
        landmarks[:, 36:42, 1] = torch.tensor([18, 16, 16, 18, 22, 22])
        # Right eye with varying y values
        landmarks[:, 42:48, 0] = torch.tensor([40, 42, 44, 46, 48, 44])
        landmarks[:, 42:48, 1] = torch.tensor([18, 16, 16, 18, 22, 22])

        loss = gaze_loss(x, y, landmarks=landmarks)

        # Check that loss is differentiable (gradients should flow through the images)
        assert loss.requires_grad, f"Loss should require grad but got {loss}"
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape


class TestStyleLoss:
    """Tests for style loss."""

    @pytest.fixture
    def style_loss(self):
        """Create style loss instance."""
        return StyleLoss()

    def test_identical_images_low_loss(self, style_loss):
        """Identical images should have near-zero style loss."""
        x = torch.rand(2, 3, 64, 64)
        loss = style_loss(x, x)

        assert loss.item() < 1e-6

    def test_different_images_nonzero_loss(self, style_loss):
        """Different images should have non-zero style loss."""
        x = torch.rand(2, 3, 64, 64)
        y = torch.rand(2, 3, 64, 64)

        loss = style_loss(x, y)

        assert loss.item() > 0

    def test_with_mask(self, style_loss):
        """Should work with mask."""
        x = torch.rand(2, 3, 64, 64)
        y = torch.rand(2, 3, 64, 64)
        mask = torch.ones(2, 1, 64, 64)
        mask[:, :, 32:, :] = 0  # Mask bottom half

        loss = style_loss(x, y, mask=mask)

        assert loss.item() >= 0

    def test_gradient_flow(self, style_loss):
        """Gradients should flow through the loss."""
        x = torch.rand(2, 3, 64, 64, requires_grad=True)
        y = torch.rand(2, 3, 64, 64)

        loss = style_loss(x, y)
        loss.backward()

        assert x.grad is not None


class TestCombinedLoss:
    """Tests for combined loss."""

    @pytest.fixture
    def combined_loss(self):
        """Create combined loss instance (without optional losses)."""
        return CombinedLoss(
            dssim_weight=10.0,
            l1_weight=10.0,
            lpips_weight=0.0,
            id_weight=0.0,
            eyes_mouth_weight=0.0,
        )

    def test_returns_tuple(self, combined_loss):
        """Should return (total_loss, loss_dict)."""
        x = torch.rand(2, 3, 64, 64)
        y = torch.rand(2, 3, 64, 64)

        result = combined_loss(x, y)

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_loss_dict_keys(self, combined_loss):
        """Loss dict should contain expected keys."""
        x = torch.rand(2, 3, 64, 64)
        y = torch.rand(2, 3, 64, 64)

        total, losses = combined_loss(x, y)

        assert "dssim" in losses
        assert "l1" in losses
        assert "total" in losses

    def test_total_is_weighted_sum(self, combined_loss):
        """Total should be weighted sum of components."""
        x = torch.rand(2, 3, 64, 64)
        y = torch.rand(2, 3, 64, 64)

        total, losses = combined_loss(x, y)

        expected_total = 10.0 * losses["dssim"] + 10.0 * losses["l1"]

        torch.testing.assert_close(total, expected_total, atol=1e-5, rtol=1e-5)

    def test_gradient_flow(self, combined_loss):
        """Gradients should flow through combined loss."""
        x = torch.rand(2, 3, 64, 64, requires_grad=True)
        y = torch.rand(2, 3, 64, 64)

        total, _ = combined_loss(x, y)
        total.backward()

        assert x.grad is not None


class TestDFLModuleLossIntegration:
    """Integration tests for DFLModule with losses."""

    def test_training_step_with_losses(self):
        """DFLModule training step should compute all losses."""
        from visagen.training.dfl_module import DFLModule

        module = DFLModule(
            image_size=64,
            encoder_dims=[32, 64, 128, 256],
            encoder_depths=[1, 1, 1, 1],
            decoder_dims=[256, 128, 64, 32],
            dssim_weight=10.0,
            l1_weight=10.0,
        )

        src = torch.rand(2, 3, 64, 64)
        dst = torch.rand(2, 3, 64, 64)

        # training_step expects dict-based batch (src_dict, dst_dict)
        src_dict = {"image": src, "landmarks": torch.rand(2, 68, 2)}
        dst_dict = {"image": dst, "landmarks": torch.rand(2, 68, 2)}

        loss = module.training_step((src_dict, dst_dict), 0)

        assert loss.item() > 0
        assert loss.requires_grad

    def test_compute_loss_returns_dict(self):
        """compute_loss should return loss dictionary."""
        from visagen.training.dfl_module import DFLModule

        module = DFLModule(
            image_size=64,
            encoder_dims=[32, 64, 128, 256],
            encoder_depths=[1, 1, 1, 1],
            decoder_dims=[256, 128, 64, 32],
        )

        x = torch.rand(2, 3, 64, 64)
        pred = module(x)

        total, losses = module.compute_loss(pred, x)

        assert "dssim" in losses
        assert "l1" in losses
        assert "total" in losses

    def test_identical_input_low_reconstruction_loss(self):
        """Identical pred and target should have low loss."""
        from visagen.training.dfl_module import DFLModule

        module = DFLModule(
            image_size=64,
            encoder_dims=[32, 64, 128, 256],
            encoder_depths=[1, 1, 1, 1],
            decoder_dims=[256, 128, 64, 32],
        )

        x = torch.rand(2, 3, 64, 64)

        total, losses = module.compute_loss(x, x)

        # L1 should be 0, DSSIM should be near 0
        assert losses["l1"].item() < 1e-6
        assert losses["dssim"].item() < 0.01


class TestDSSIMResolution:
    """Tests for DSSIM resolution-based filter size."""

    def test_filter_size_calculation_96(self):
        """Test filter size for 96px resolution."""
        ms_dssim = MultiScaleDSSIMLoss(resolution=96)
        assert len(ms_dssim.losses) == 2

    def test_filter_size_calculation_256(self):
        """Test filter size for 256px resolution."""
        ms_dssim = MultiScaleDSSIMLoss(resolution=256)
        # 256/11.6 = 22, |1 = 23
        assert ms_dssim.losses[0].filter_size == 23

    def test_filter_size_calculation_512(self):
        """Test filter size for 512px resolution."""
        ms_dssim = MultiScaleDSSIMLoss(resolution=512)
        # 512/11.6 = 44, |1 = 45
        assert ms_dssim.losses[0].filter_size == 45

    def test_single_scale_dssim_filter_size(self):
        """Test single-scale DSSIM with explicit filter size."""
        dssim = DSSIMLoss(filter_size=15)
        assert dssim.filter_size == 15

    def test_multiscale_creates_multiple_scales(self):
        """Test that multi-scale DSSIM creates multiple loss modules."""
        ms_dssim = MultiScaleDSSIMLoss(resolution=256)
        assert len(ms_dssim.losses) >= 2
        assert len(ms_dssim.weights) == len(ms_dssim.losses)
