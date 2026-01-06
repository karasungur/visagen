"""
Tests for Landmark Data Flow.

Tests verify that landmarks are properly passed through:
1. FaceDataset -> PairedFaceDataset -> DFLModule.training_step
2. EyesMouthLoss and GazeLoss receive landmarks when weights > 0
3. TransformWrapper preserves landmarks during augmentation

These tests address the critical bug fixed in FAZ A where landmarks
were being discarded in PairedFaceDataset.__getitem__().
"""


import pytest
import torch

from visagen.data.datamodule import PairedFaceDataset, TransformWrapper

# =============================================================================
# Mock Dataset for Testing
# =============================================================================


class MockFaceDataset(torch.utils.data.Dataset):
    """
    Mock dataset that returns dict with image and landmarks.

    Used for testing the data pipeline without requiring real DFL images.
    """

    def __init__(self, num_samples: int = 5, image_size: int = 256):
        self.num_samples = num_samples
        self.image_size = image_size

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "image": torch.randn(3, self.image_size, self.image_size),
            "landmarks": torch.rand(68, 2),
            "mask": torch.ones(1, self.image_size, self.image_size),
        }


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_dataset():
    """Create a mock dataset for testing."""
    return MockFaceDataset(num_samples=5, image_size=64)


@pytest.fixture
def mock_paired_datasets():
    """Create mock src and dst datasets."""
    return MockFaceDataset(num_samples=5, image_size=64), MockFaceDataset(
        num_samples=5, image_size=64
    )


# =============================================================================
# MockFaceDataset Tests (verifies dict structure)
# =============================================================================


class TestMockFaceDataset:
    """Test that MockFaceDataset returns correct structure."""

    def test_returns_dict_with_landmarks(self, mock_dataset):
        """MockFaceDataset should return dict with landmarks key."""
        sample = mock_dataset[0]

        assert isinstance(sample, dict)
        assert "image" in sample
        assert "landmarks" in sample
        assert sample["image"].shape == (3, 64, 64)
        assert sample["landmarks"].shape == (68, 2)

    def test_landmarks_are_tensors(self, mock_dataset):
        """Landmarks should be torch tensors."""
        sample = mock_dataset[0]

        assert isinstance(sample["landmarks"], torch.Tensor)
        assert sample["landmarks"].dtype == torch.float32

    def test_landmarks_value_range(self, mock_dataset):
        """Landmarks should be in [0, 1] range (normalized)."""
        sample = mock_dataset[0]

        landmarks = sample["landmarks"]
        assert landmarks.min() >= 0
        assert landmarks.max() <= 1


# =============================================================================
# PairedFaceDataset Tests
# =============================================================================


class TestPairedFaceDatasetLandmarks:
    """Test that PairedFaceDataset preserves landmarks."""

    def test_returns_dict_by_default(self, mock_paired_datasets):
        """PairedFaceDataset should return dicts by default (return_dict=True)."""
        src_dataset, dst_dataset = mock_paired_datasets
        paired = PairedFaceDataset(src_dataset, dst_dataset)

        src_sample, dst_sample = paired[0]

        assert isinstance(src_sample, dict)
        assert isinstance(dst_sample, dict)

    def test_preserves_landmarks(self, mock_paired_datasets):
        """PairedFaceDataset should preserve landmarks in returned dicts."""
        src_dataset, dst_dataset = mock_paired_datasets
        paired = PairedFaceDataset(src_dataset, dst_dataset)

        src_sample, dst_sample = paired[0]

        assert "landmarks" in src_sample
        assert "landmarks" in dst_sample
        assert src_sample["landmarks"].shape == (68, 2)
        assert dst_sample["landmarks"].shape == (68, 2)

    def test_legacy_mode_returns_tensors_only(self, mock_paired_datasets):
        """PairedFaceDataset with return_dict=False returns only image tensors."""
        src_dataset, dst_dataset = mock_paired_datasets
        paired = PairedFaceDataset(src_dataset, dst_dataset, return_dict=False)

        src_img, dst_img = paired[0]

        assert isinstance(src_img, torch.Tensor)
        assert isinstance(dst_img, torch.Tensor)
        assert src_img.shape == (3, 64, 64)
        assert dst_img.shape == (3, 64, 64)

    def test_preserves_image_data(self, mock_paired_datasets):
        """PairedFaceDataset should preserve image data."""
        src_dataset, dst_dataset = mock_paired_datasets
        paired = PairedFaceDataset(src_dataset, dst_dataset)

        src_sample, dst_sample = paired[0]

        assert "image" in src_sample
        assert "image" in dst_sample
        assert src_sample["image"].shape == (3, 64, 64)
        assert dst_sample["image"].shape == (3, 64, 64)


# =============================================================================
# TransformWrapper Tests
# =============================================================================


class TestTransformWrapperLandmarks:
    """Test that TransformWrapper preserves landmarks during transformation."""

    def test_preserves_landmarks_with_transform(self, mock_paired_datasets):
        """TransformWrapper should preserve landmarks when applying transforms."""
        from visagen.data.augmentations import FaceAugmentationPipeline

        src_dataset, dst_dataset = mock_paired_datasets
        paired = PairedFaceDataset(src_dataset, dst_dataset)

        transform = FaceAugmentationPipeline(target_size=64)
        wrapped = TransformWrapper(paired, transform)

        src_sample, dst_sample = wrapped[0]

        # Landmarks should still be present
        assert "landmarks" in src_sample
        assert "landmarks" in dst_sample
        assert src_sample["landmarks"].shape == (68, 2)
        assert dst_sample["landmarks"].shape == (68, 2)

    def test_applies_transform_to_images(self, mock_paired_datasets):
        """TransformWrapper should apply transforms to images."""
        from visagen.data.augmentations import FaceAugmentationPipeline

        src_dataset, dst_dataset = mock_paired_datasets
        paired = PairedFaceDataset(src_dataset, dst_dataset)

        transform = FaceAugmentationPipeline(target_size=64)
        wrapped = TransformWrapper(paired, transform)

        trans_src, trans_dst = wrapped[0]

        # Images should be valid
        assert trans_src["image"].shape == (3, 64, 64)
        assert trans_dst["image"].shape == (3, 64, 64)


# =============================================================================
# DFLModule Integration Tests
# =============================================================================


class TestDFLModuleLandmarkIntegration:
    """Test that DFLModule properly receives and uses landmarks."""

    def test_training_step_accepts_dict_batch(self):
        """DFLModule.training_step should accept dict-based batch."""
        from visagen.training.dfl_module import DFLModule

        module = DFLModule(image_size=64)

        # Create mock batch with dicts
        src_dict = {
            "image": torch.randn(2, 3, 64, 64),
            "landmarks": torch.rand(2, 68, 2),
        }
        dst_dict = {
            "image": torch.randn(2, 3, 64, 64),
            "landmarks": torch.rand(2, 68, 2),
        }
        batch = (src_dict, dst_dict)

        # Should not raise
        loss = module.training_step(batch, 0)
        assert loss is not None
        assert torch.isfinite(loss)

    def test_compute_loss_with_landmarks(self):
        """DFLModule.compute_loss should accept landmarks parameter."""
        from visagen.training.dfl_module import DFLModule

        module = DFLModule(image_size=64)

        pred = torch.randn(2, 3, 64, 64)
        target = torch.randn(2, 3, 64, 64)
        landmarks = torch.rand(2, 68, 2)

        # Should not raise
        total_loss, loss_dict = module.compute_loss(pred, target, landmarks)
        assert torch.isfinite(total_loss)
        assert "total" in loss_dict

    def test_eyes_mouth_loss_computed_when_enabled(self):
        """Eyes/Mouth loss should be computed when weight > 0."""
        from visagen.training.dfl_module import DFLModule

        module = DFLModule(image_size=64, eyes_mouth_weight=1.0)

        pred = torch.randn(2, 3, 64, 64)
        target = torch.randn(2, 3, 64, 64)
        landmarks = torch.rand(2, 68, 2)

        total_loss, loss_dict = module.compute_loss(pred, target, landmarks)

        assert "eyes_mouth" in loss_dict
        assert torch.isfinite(loss_dict["eyes_mouth"])

    def test_gaze_loss_computed_when_enabled(self):
        """Gaze loss should be computed when weight > 0."""
        from visagen.training.dfl_module import DFLModule

        module = DFLModule(image_size=64, gaze_weight=1.0)

        pred = torch.randn(2, 3, 64, 64)
        target = torch.randn(2, 3, 64, 64)
        landmarks = torch.rand(2, 68, 2)

        total_loss, loss_dict = module.compute_loss(pred, target, landmarks)

        assert "gaze" in loss_dict
        assert torch.isfinite(loss_dict["gaze"])

    def test_no_landmark_losses_when_disabled(self):
        """Landmark-based losses should not appear when weights are 0."""
        from visagen.training.dfl_module import DFLModule

        module = DFLModule(image_size=64, eyes_mouth_weight=0.0, gaze_weight=0.0)

        pred = torch.randn(2, 3, 64, 64)
        target = torch.randn(2, 3, 64, 64)
        landmarks = torch.rand(2, 68, 2)

        total_loss, loss_dict = module.compute_loss(pred, target, landmarks)

        assert "eyes_mouth" not in loss_dict
        assert "gaze" not in loss_dict

    def test_no_landmark_losses_when_landmarks_none(self):
        """Landmark-based losses should not appear when landmarks are None."""
        from visagen.training.dfl_module import DFLModule

        module = DFLModule(image_size=64, eyes_mouth_weight=1.0, gaze_weight=1.0)

        pred = torch.randn(2, 3, 64, 64)
        target = torch.randn(2, 3, 64, 64)

        # Pass None for landmarks
        total_loss, loss_dict = module.compute_loss(pred, target, None)

        # Should not error, but landmark losses should not be computed
        assert "eyes_mouth" not in loss_dict
        assert "gaze" not in loss_dict

    def test_validation_step_accepts_dict_batch(self):
        """DFLModule.validation_step should accept dict-based batch."""
        from visagen.training.dfl_module import DFLModule

        module = DFLModule(image_size=64)

        # Create mock batch with dicts
        src_dict = {
            "image": torch.randn(2, 3, 64, 64),
            "landmarks": torch.rand(2, 68, 2),
        }
        dst_dict = {
            "image": torch.randn(2, 3, 64, 64),
            "landmarks": torch.rand(2, 68, 2),
        }
        batch = (src_dict, dst_dict)

        # Should not raise
        loss = module.validation_step(batch, 0)
        assert loss is not None
        assert torch.isfinite(loss)


# =============================================================================
# End-to-End Integration Test
# =============================================================================


class TestEndToEndLandmarkFlow:
    """End-to-end test for landmark data flow through the pipeline."""

    def test_full_pipeline_with_landmarks(self, mock_paired_datasets):
        """Test complete data flow from dataset to loss computation."""
        from visagen.data.augmentations import FaceAugmentationPipeline
        from visagen.training.dfl_module import DFLModule

        src_dataset, dst_dataset = mock_paired_datasets

        # Create paired dataset (should preserve landmarks)
        paired = PairedFaceDataset(src_dataset, dst_dataset)

        # Apply transforms (should preserve landmarks)
        transform = FaceAugmentationPipeline(target_size=64)
        dataset = TransformWrapper(paired, transform)

        # Get a sample
        src_sample, dst_sample = dataset[0]

        # Verify landmarks are present
        assert "landmarks" in src_sample
        assert "landmarks" in dst_sample

        # Create batch (simulating DataLoader)
        batch = (
            {k: v.unsqueeze(0) for k, v in src_sample.items()},
            {k: v.unsqueeze(0) for k, v in dst_sample.items()},
        )

        # Create model with landmark losses enabled
        module = DFLModule(image_size=64, eyes_mouth_weight=1.0, gaze_weight=1.0)

        # Run training step
        loss = module.training_step(batch, 0)

        # Verify training completed
        assert loss is not None
        assert torch.isfinite(loss)
