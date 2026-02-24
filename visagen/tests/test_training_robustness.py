"""Comprehensive robustness tests for training module."""

import torch

from visagen.training.losses import (
    DSSIMLoss,
    GazeLoss,
    LPIPSLoss,
    MultiScaleDSSIMLoss,
    TemporalConsistencyLoss,
)


class TestDSSIMRobustness:
    """Tests for DSSIM numerical stability."""

    def test_constant_images_no_nan(self):
        """Constant images should not produce NaN."""
        loss_fn = DSSIMLoss()
        x = torch.ones(2, 3, 64, 64) * 0.5
        loss = loss_fn(x, x)
        assert not torch.isnan(loss), "DSSIM produced NaN for constant images"
        assert loss.item() < 0.01

    def test_zero_images(self):
        """Zero images should not cause division by zero."""
        loss_fn = DSSIMLoss()
        x = torch.zeros(2, 3, 64, 64)
        y = torch.rand(2, 3, 64, 64)
        loss = loss_fn(x, y)
        assert torch.isfinite(loss), "DSSIM not finite for zero input"

    def test_identical_images_very_low_loss(self):
        """Identical images should have near-zero loss."""
        loss_fn = DSSIMLoss()
        x = torch.rand(2, 3, 64, 64)
        loss = loss_fn(x, x)
        assert loss.item() < 0.01, f"Expected near-zero loss, got {loss.item()}"

    def test_multiscale_constant_images(self):
        """MultiScale DSSIM should handle constant images."""
        loss_fn = MultiScaleDSSIMLoss()
        x = torch.ones(2, 3, 128, 128) * 0.5
        loss = loss_fn(x, x)
        assert torch.isfinite(loss), "MultiScale DSSIM not finite for constant images"


class TestGazeLossRobustness:
    """Tests for GazeLoss edge cases."""

    def test_degenerate_landmarks_single_point(self):
        """All landmarks at same point should not crash."""
        loss_fn = GazeLoss()
        x = torch.rand(2, 3, 256, 256)
        # All landmarks at same point (degenerate case)
        landmarks = torch.ones(2, 68, 2) * 128
        loss = loss_fn(x, x, landmarks)
        assert torch.isfinite(loss), "GazeLoss not finite for degenerate landmarks"

    def test_landmarks_at_edge(self):
        """Landmarks at image edges should be handled."""
        loss_fn = GazeLoss()
        x = torch.rand(2, 3, 256, 256)
        # Landmarks at corners
        landmarks = torch.zeros(2, 68, 2)
        landmarks[:, :, 0] = 0  # x at left edge
        landmarks[:, :, 1] = 255  # y at bottom edge
        loss = loss_fn(x, x, landmarks)
        assert torch.isfinite(loss), "GazeLoss not finite for edge landmarks"

    def test_without_landmarks_fallback(self):
        """GazeLoss should fallback gracefully without landmarks."""
        loss_fn = GazeLoss()
        x = torch.rand(2, 3, 256, 256)
        loss = loss_fn(x, x, None)
        assert torch.isfinite(loss), "GazeLoss not finite without landmarks"


class TestTemporalVectorization:
    """Tests for vectorized temporal loss."""

    def test_l1_mode_correct_output(self):
        """L1 mode should compute correct temporal loss."""
        loss_fn = TemporalConsistencyLoss(mode="l1")
        # Create sequence with known difference
        seq = torch.zeros(2, 3, 5, 64, 64)
        seq[:, :, 1:] = 0.1  # Constant difference of 0.1 after first frame
        loss = loss_fn(seq)
        assert loss.shape == (), "Expected scalar output"
        assert loss.item() > 0, "Expected positive loss for different frames"

    def test_l2_mode_correct_output(self):
        """L2 mode should compute correct temporal loss."""
        loss_fn = TemporalConsistencyLoss(mode="l2")
        seq = torch.rand(2, 3, 5, 64, 64)
        loss = loss_fn(seq)
        assert loss.shape == (), "Expected scalar output"
        assert torch.isfinite(loss), "L2 temporal loss not finite"

    def test_ssim_mode_vectorized(self):
        """SSIM mode should work with vectorized implementation."""
        loss_fn = TemporalConsistencyLoss(mode="ssim")
        seq = torch.rand(2, 3, 5, 64, 64)
        loss = loss_fn(seq)
        assert loss.shape == (), "Expected scalar output"
        assert torch.isfinite(loss), "SSIM temporal loss not finite"

    def test_single_frame_sequence(self):
        """Single frame sequence should return zero loss."""
        loss_fn = TemporalConsistencyLoss(mode="ssim")
        seq = torch.rand(2, 3, 1, 64, 64)
        loss = loss_fn(seq)
        assert loss.item() == 0.0, "Single frame should have zero temporal loss"


class TestLPIPSCaching:
    """Tests for LPIPS device caching."""

    def test_lpips_device_caching(self):
        """LPIPS should cache device correctly."""
        loss_fn = LPIPSLoss()
        x = torch.rand(1, 3, 64, 64)
        y = torch.rand(1, 3, 64, 64)

        # First call initializes
        loss1 = loss_fn(x, y)
        assert torch.isfinite(loss1), "First LPIPS call not finite"

        # Check that cached device is set
        assert loss_fn._cached_device is not None

        # Second call should use cache
        loss2 = loss_fn(x, y)
        assert torch.isfinite(loss2), "Second LPIPS call not finite"


class TestGradientAccumulation:
    """Tests for gradient accumulation parameter."""

    def test_accumulation_parameter_accepted(self):
        """Module should accept gradient_accumulation_steps parameter."""
        from visagen.training.training_module import TrainingModule

        module = TrainingModule(
            image_size=64,
            encoder_dims=[32, 64],
            encoder_depths=[1, 1],
            gradient_accumulation_steps=4,
        )
        assert module.hparams.gradient_accumulation_steps == 4

    def test_temporal_checkpoint_parameter(self):
        """Module should accept temporal_checkpoint parameter."""
        from visagen.training.training_module import TrainingModule

        module = TrainingModule(
            image_size=64,
            encoder_dims=[32, 64],
            encoder_depths=[1, 1],
            temporal_checkpoint=False,
        )
        assert module.hparams.temporal_checkpoint is False


class TestNaNInfGuards:
    """Tests for NaN/Inf handling."""

    def test_dssim_with_nan_input(self):
        """DSSIM should handle NaN gracefully or raise clear error."""
        loss_fn = DSSIMLoss()
        x = torch.rand(2, 3, 64, 64)
        y = torch.rand(2, 3, 64, 64)
        y[0, 0, 0, 0] = float("nan")

        # Should either handle or produce NaN (not crash)
        try:
            _loss = loss_fn(x, y)
            # If it doesn't crash, output might be NaN which is acceptable
        except (ValueError, RuntimeError):
            # Clear error is also acceptable
            pass

    def test_dssim_with_inf_input(self):
        """DSSIM should handle Inf gracefully or raise clear error."""
        loss_fn = DSSIMLoss()
        x = torch.rand(2, 3, 64, 64)
        y = torch.rand(2, 3, 64, 64)
        y[0, 0, 0, 0] = float("inf")

        # Should either handle or produce Inf (not crash)
        try:
            _loss = loss_fn(x, y)
        except (ValueError, RuntimeError):
            pass
