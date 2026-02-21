"""
Tests for dataset and augmentation components.

Covers warp functions, augmentations, dataset loading, and datamodule.
"""

import pickle
import struct
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from visagen.data.augmentations import FaceAugmentationPipeline, SimpleAugmentation
from visagen.data.datamodule import (
    PairedFaceDataset,
    TransformWrapper,
    UniformYawPairSampler,
    paired_face_collate_fn,
)
from visagen.data.face_dataset import FaceDataset, PackedFacesetReader
from visagen.data.face_sample import FaceSample
from visagen.data.warp import (
    gen_affine_params,
    gen_legacy_warp_params,
    gen_warp_params,
    warp_by_params,
    warp_legacy_by_params,
)


class TestFaceSample:
    """Tests for FaceSample dataclass."""

    def test_sample_creation(self):
        """Test FaceSample can be created with required fields."""
        sample = FaceSample(
            filepath=Path("/test/face.jpg"),
            face_type="whole_face",
            shape=(512, 512, 3),
            landmarks=np.zeros((68, 2), dtype=np.float32),
        )
        assert sample.filepath == Path("/test/face.jpg")
        assert sample.face_type == "whole_face"
        assert sample.shape == (512, 512, 3)

    def test_sample_optional_fields(self):
        """Test FaceSample optional fields have correct defaults."""
        sample = FaceSample(
            filepath=Path("/test/face.jpg"),
            face_type="whole_face",
            shape=(512, 512, 3),
            landmarks=np.zeros((68, 2), dtype=np.float32),
        )
        assert sample.xseg_mask is None
        assert sample.seg_ie_polys is None
        assert sample.eyebrows_expand_mod == 1.0
        assert sample.source_filename is None
        assert sample.image_to_face_mat is None

    def test_sample_loads_from_packed_faceset_offset(self, tmp_path):
        """FaceSample should decode images from packed faceset offsets."""
        image = np.full((16, 16, 3), 127, dtype=np.uint8)
        ok, encoded = cv2.imencode(".jpg", image)
        assert ok
        image_bytes = encoded.tobytes()

        packed_path = tmp_path / "faceset.pak"
        with open(packed_path, "wb") as f:
            f.write(b"HEAD")
            offset = f.tell()
            f.write(image_bytes)
            size = len(image_bytes)
            f.write(b"TAIL")

        sample = FaceSample(
            filepath=tmp_path / "face.jpg",
            face_type="whole_face",
            shape=(16, 16, 3),
            landmarks=np.zeros((68, 2), dtype=np.float32),
            packed_faceset_path=packed_path,
            packed_offset=offset,
            packed_size=size,
        )

        loaded = sample.load_image()
        assert loaded.shape == (16, 16, 3)
        assert loaded.dtype == np.float32
        assert 0 <= loaded.min() <= loaded.max() <= 1.0

    def test_sample_reads_raw_bytes_from_packed_faceset_offset(self, tmp_path):
        """FaceSample should expose encoded bytes from packed offsets."""
        image = np.full((16, 16, 3), 90, dtype=np.uint8)
        ok, encoded = cv2.imencode(".jpg", image)
        assert ok
        image_bytes = encoded.tobytes()

        packed_path = tmp_path / "faceset.pak"
        with open(packed_path, "wb") as f:
            f.write(b"HEAD")
            offset = f.tell()
            f.write(image_bytes)
            size = len(image_bytes)
            f.write(b"TAIL")

        sample = FaceSample(
            filepath=tmp_path / "face.jpg",
            face_type="whole_face",
            shape=(16, 16, 3),
            landmarks=np.zeros((68, 2), dtype=np.float32),
            packed_faceset_path=packed_path,
            packed_offset=offset,
            packed_size=size,
        )

        raw = sample.read_raw_bytes()
        assert raw == image_bytes

    def test_sample_reads_raw_bytes_from_regular_path(self, tmp_path):
        """FaceSample should expose encoded bytes from filesystem path."""
        image = np.full((12, 12, 3), 150, dtype=np.uint8)
        image_path = tmp_path / "face.jpg"
        ok = cv2.imwrite(str(image_path), image)
        assert ok

        sample = FaceSample(
            filepath=image_path,
            face_type="whole_face",
            shape=(12, 12, 3),
            landmarks=np.zeros((68, 2), dtype=np.float32),
        )

        raw = sample.read_raw_bytes()
        assert raw == image_path.read_bytes()

    def test_sample_landmarks_shape(self):
        """Test landmarks have correct shape."""
        landmarks = np.random.randn(68, 2).astype(np.float32)
        sample = FaceSample(
            filepath=Path("/test/face.jpg"),
            face_type="whole_face",
            shape=(512, 512, 3),
            landmarks=landmarks,
        )
        assert sample.landmarks.shape == (68, 2)


class TestPackedFacesetSupport:
    """Tests for legacy `faceset.pak` compatibility in FaceDataset."""

    @staticmethod
    def _write_faceset_archive(tmp_path: Path, configs: list[dict], image_bytes: bytes):
        configs_raw = pickle.dumps(configs, protocol=4)
        packed_path = tmp_path / "faceset.pak"
        with open(packed_path, "wb") as f:
            f.write(struct.pack("Q", 1))  # version
            f.write(struct.pack("Q", len(configs_raw)))
            f.write(configs_raw)
            f.write(struct.pack("Q", 0))
            f.write(struct.pack("Q", len(image_bytes)))
            f.write(image_bytes)
        return packed_path

    def test_face_dataset_loads_samples_from_faceset_pak(self, tmp_path):
        image = np.full((32, 32, 3), 200, dtype=np.uint8)
        ok, encoded = cv2.imencode(".jpg", image)
        assert ok
        image_bytes = encoded.tobytes()

        sample_config = {
            "sample_type": 1,
            "filename": "face_00001.jpg",
            "face_type": 4,
            "shape": (32, 32, 3),
            "landmarks": np.zeros((68, 2), dtype=np.float32).tolist(),
            "seg_ie_polys": {"polys": []},
            "xseg_mask": None,
            "xseg_mask_compressed": None,
            "eyebrows_expand_mod": 1.0,
            "source_filename": "src_00001.jpg",
            "person_name": None,
        }
        self._write_faceset_archive(tmp_path, [sample_config], image_bytes)

        dataset = FaceDataset(tmp_path, target_size=32)
        assert len(dataset) == 1

        item = dataset[0]
        assert isinstance(item, dict)
        assert item["image"].shape == (3, 32, 32)
        assert item["landmarks"].shape == (68, 2)
        assert item["seg_ie_polys"] == {"polys": []}

    def test_face_dataset_can_disable_packed_faceset_loading(self, tmp_path):
        image = np.full((16, 16, 3), 180, dtype=np.uint8)
        ok, encoded = cv2.imencode(".jpg", image)
        assert ok
        image_bytes = encoded.tobytes()

        sample_config = {
            "sample_type": 1,
            "filename": "face_00001.jpg",
            "face_type": 4,
            "shape": (16, 16, 3),
            "landmarks": np.zeros((68, 2), dtype=np.float32).tolist(),
        }
        self._write_faceset_archive(tmp_path, [sample_config], image_bytes)

        with pytest.raises(ValueError, match="allow_packed_faceset=True"):
            FaceDataset(tmp_path, allow_packed_faceset=False)

    def test_packed_faceset_reader_raises_on_corrupt_archive(self, tmp_path):
        packed_path = tmp_path / "faceset.pak"
        packed_path.write_bytes(b"not-a-valid-packed-archive")

        reader = PackedFacesetReader(packed_path)
        with pytest.raises(ValueError, match="Failed to parse faceset archive"):
            reader.read_samples(tmp_path)


class TestDatasetSamplingAndTransforms:
    """Tests for pairing behavior and transform contract wiring."""

    def test_paired_dataset_supports_sampler_generated_tuple_indices(self):
        class _MockDataset:
            def __len__(self):
                return 5

            def __getitem__(self, idx):
                return {
                    "image": torch.full((1,), float(idx)),
                    "landmarks": torch.zeros(68, 2),
                    "face_type": 0,
                }

        paired = PairedFaceDataset(_MockDataset(), _MockDataset())
        src, dst = paired[(1, 3)]
        assert src["image"].item() == 1.0
        assert dst["image"].item() == 3.0

    def test_uniform_yaw_pair_sampler_samples_src_and_dst_independently(self):
        class _MockDataset:
            def __init__(self, fixed_idx: int):
                self.fixed_idx = fixed_idx

            def __len__(self):
                return 8

            def sample_uniform_yaw(self):
                return self.fixed_idx

        sampler = UniformYawPairSampler(
            _MockDataset(fixed_idx=2),
            _MockDataset(fixed_idx=5),
            num_samples=4,
            seed=42,
        )
        indices = list(iter(sampler))
        assert indices == [(2, 5), (2, 5), (2, 5), (2, 5)]

    def test_paired_collate_preserves_seg_ie_polys_as_metadata_list(self):
        batch = [
            (
                {
                    "image": torch.zeros(3, 8, 8),
                    "landmarks": torch.zeros(68, 2),
                    "face_type": 0,
                    "seg_ie_polys": {"polys": [1]},
                },
                {
                    "image": torch.ones(3, 8, 8),
                    "landmarks": torch.zeros(68, 2),
                    "face_type": 0,
                    "seg_ie_polys": {"polys": [2]},
                },
            ),
            (
                {
                    "image": torch.zeros(3, 8, 8),
                    "landmarks": torch.zeros(68, 2),
                    "face_type": 0,
                    "seg_ie_polys": {"polys": [3]},
                },
                {
                    "image": torch.ones(3, 8, 8),
                    "landmarks": torch.zeros(68, 2),
                    "face_type": 0,
                    "seg_ie_polys": {"polys": [4]},
                },
            ),
        ]

        src_batch, dst_batch = paired_face_collate_fn(batch)
        assert isinstance(src_batch["image"], torch.Tensor)
        assert src_batch["image"].shape == (2, 3, 8, 8)
        assert src_batch["seg_ie_polys"] == [{"polys": [1]}, {"polys": [3]}]
        assert dst_batch["seg_ie_polys"] == [{"polys": [2]}, {"polys": [4]}]

    def test_transform_wrapper_passes_cross_target_image_for_color_transfer(self):
        class _PairDataset:
            def __len__(self):
                return 1

            def __getitem__(self, idx):
                src = {
                    "image": torch.full((3, 8, 8), 1.0),
                    "landmarks": torch.zeros(68, 2),
                    "face_type": 0,
                }
                dst = {
                    "image": torch.full((3, 8, 8), 2.0),
                    "landmarks": torch.zeros(68, 2),
                    "face_type": 0,
                }
                return src, dst

        class _CaptureTransform:
            def __init__(self):
                self.calls = []

            def __call__(self, image, mask=None, target_image=None):
                if target_image is not None:
                    self.calls.append(target_image.clone())
                else:
                    self.calls.append(None)
                return image, mask

        transform = _CaptureTransform()
        wrapped = TransformWrapper(_PairDataset(), transform=transform)
        wrapped[0]

        assert len(transform.calls) == 2
        assert transform.calls[0] is not None
        assert transform.calls[1] is not None
        assert torch.allclose(transform.calls[0], torch.full((3, 8, 8), 2.0))
        assert torch.allclose(transform.calls[1], torch.full((3, 8, 8), 1.0))


class TestWarpFunctions:
    """Tests for grid warping functions."""

    def test_gen_warp_params_returns_dict(self):
        """Test gen_warp_params returns expected structure."""
        params = gen_warp_params(256)

        assert isinstance(params, dict)
        assert "grid" in params
        assert "size" in params

    def test_gen_warp_params_grid_shape(self):
        """Test warp grid has correct shape."""
        params = gen_warp_params(256)

        assert params["grid"].shape == (1, 256, 256, 2)

    def test_gen_warp_params_different_sizes(self):
        """Test warp params work for different sizes."""
        for size in [64, 128, 256, 512]:
            params = gen_warp_params(size)
            assert params["grid"].shape == (1, size, size, 2)

    def test_gen_warp_params_reproducible(self):
        """Test warp params are reproducible with same seed."""
        gen1 = torch.Generator().manual_seed(42)
        gen2 = torch.Generator().manual_seed(42)

        params1 = gen_warp_params(256, rng=gen1)
        params2 = gen_warp_params(256, rng=gen2)

        torch.testing.assert_close(params1["grid"], params2["grid"])

    def test_warp_by_params_shape_preserved(self):
        """Test warp preserves image shape."""
        image = torch.randn(1, 3, 256, 256)
        params = gen_warp_params(256)

        warped = warp_by_params(image, params)

        assert warped.shape == image.shape

    def test_warp_by_params_unbatched(self):
        """Test warp works with unbatched input."""
        image = torch.randn(3, 256, 256)
        params = gen_warp_params(256)

        warped = warp_by_params(image, params)

        assert warped.shape == image.shape

    def test_warp_by_params_range(self):
        """Test warp keeps values in reasonable range."""
        image = torch.rand(1, 3, 256, 256)
        params = gen_warp_params(256)

        warped = warp_by_params(image, params)

        # Allow some slight overshooting due to interpolation
        assert warped.min() >= -0.1
        assert warped.max() <= 1.1

    def test_gen_affine_params_returns_matrix(self):
        """Test gen_affine_params returns affine matrix."""
        params = gen_affine_params(256)

        assert "matrix" in params
        assert params["matrix"].shape == (2, 3)

    def test_gen_legacy_warp_params_returns_legacy_keys(self):
        params = gen_legacy_warp_params(256, rng=np.random.default_rng(42))
        assert {"mapx", "mapy", "rmat", "flip", "w"} <= set(params.keys())
        assert params["mapx"].shape == (256, 256)
        assert params["mapy"].shape == (256, 256)

    def test_warp_legacy_by_params_preserves_shape(self):
        image = torch.rand(1, 3, 256, 256)
        params = gen_legacy_warp_params(256, rng=np.random.default_rng(7))
        warped = warp_legacy_by_params(image, params)
        assert warped.shape == image.shape


class TestFaceAugmentationPipeline:
    """Tests for augmentation pipeline."""

    @pytest.fixture
    def pipeline(self):
        """Create default augmentation pipeline."""
        return FaceAugmentationPipeline(target_size=256)

    def test_pipeline_output_shape(self, pipeline):
        """Test pipeline preserves image shape."""
        image = torch.randn(3, 256, 256)

        augmented, _ = pipeline(image)

        assert augmented.shape == image.shape

    def test_pipeline_batched_input(self, pipeline):
        """Test pipeline handles batched input."""
        image = torch.randn(4, 3, 256, 256)

        augmented, _ = pipeline(image)

        assert augmented.shape == image.shape

    def test_pipeline_with_mask(self, pipeline):
        """Test pipeline augments mask consistently."""
        image = torch.randn(3, 256, 256)
        mask = torch.ones(1, 256, 256)

        aug_image, aug_mask = pipeline(image, mask)

        assert aug_mask is not None
        assert aug_mask.shape == mask.shape

    def test_pipeline_normalized_input(self, pipeline):
        """Test pipeline handles [-1, 1] normalized input."""
        image = torch.randn(3, 256, 256).clamp(-1, 1)

        augmented, _ = pipeline(image)

        # Output should still be in normalized range (with some tolerance)
        assert augmented.min() >= -1.5
        assert augmented.max() <= 1.5

    def test_pipeline_0_1_input(self, pipeline):
        """Test pipeline handles [0, 1] input."""
        image = torch.rand(3, 256, 256)

        augmented, _ = pipeline(image)

        # Should clamp to [0, 1]
        assert augmented.min() >= 0
        assert augmented.max() <= 1

    def test_pipeline_no_augmentation(self):
        """Test pipeline with all augmentations disabled."""
        pipeline = FaceAugmentationPipeline(
            target_size=256,
            apply_geometric=False,
            apply_color=False,
        )
        image = torch.rand(3, 256, 256)

        augmented, _ = pipeline(image)

        # Without augmentation, should be very similar to input
        # (only clamping applied)
        torch.testing.assert_close(augmented, image, atol=0.01, rtol=0.01)

    def test_pipeline_flip_probability(self):
        """Test flip is applied approximately at expected rate."""
        pipeline = FaceAugmentationPipeline(
            target_size=256,
            random_flip_prob=0.5,
            random_warp=False,
            rotation_range=(0, 0),
            scale_range=(0, 0),
            translation_range=(0, 0),
            apply_color=False,
        )

        # Create asymmetric image
        image = torch.zeros(3, 256, 256)
        image[:, :, :128] = 1.0  # Left half white

        flip_count = 0
        trials = 100
        for _ in range(trials):
            aug, _ = pipeline(image.clone())
            # Check if right side is now white (was flipped)
            if aug[:, :, 128:].mean() > 0.5:
                flip_count += 1

        # Should be around 50% with some variance
        assert 30 < flip_count < 70, f"Flip count {flip_count} outside expected range"

    def test_pipeline_custom_params(self):
        """Test pipeline with custom augmentation parameters."""
        pipeline = FaceAugmentationPipeline(
            target_size=128,
            random_flip_prob=0.0,
            random_warp=False,
            rotation_range=(-5, 5),
            scale_range=(-0.1, 0.1),
            translation_range=(-0.1, 0.1),
            hsv_shift_amount=0.2,
        )
        image = torch.rand(3, 128, 128)

        augmented, _ = pipeline(image)

        assert augmented.shape == (3, 128, 128)

    def test_pipeline_supports_strict_warp_mode(self):
        pipeline = FaceAugmentationPipeline(
            target_size=64,
            warp_mode="strict",
            apply_color=False,
        )
        image = torch.rand(3, 64, 64)
        mask = torch.ones(1, 64, 64)

        aug_image, aug_mask = pipeline(image, mask)
        assert aug_image.shape == image.shape
        assert aug_mask is not None
        assert aug_mask.shape == mask.shape


class TestSimpleAugmentation:
    """Tests for SimpleAugmentation."""

    def test_simple_augmentation_resize(self):
        """Test SimpleAugmentation resizes image."""
        aug = SimpleAugmentation(target_size=256)
        image = torch.randn(3, 512, 512)

        result, _ = aug(image)

        assert result.shape == (3, 256, 256)

    def test_simple_augmentation_no_change(self):
        """Test SimpleAugmentation doesn't change correct size."""
        aug = SimpleAugmentation(target_size=256)
        image = torch.randn(3, 256, 256)

        result, _ = aug(image)

        torch.testing.assert_close(result, image)


class TestHSVConversion:
    """Tests for HSV color conversion."""

    def test_rgb_hsv_roundtrip(self):
        """Test RGB -> HSV -> RGB roundtrip."""
        pipeline = FaceAugmentationPipeline(target_size=256)

        # Create test image
        image = torch.rand(1, 3, 64, 64)

        # Convert to HSV and back
        hsv = pipeline._rgb_to_hsv(image)
        rgb = pipeline._hsv_to_rgb(hsv)

        torch.testing.assert_close(rgb, image, atol=1e-4, rtol=1e-4)

    def test_hsv_shape(self):
        """Test HSV output has correct shape."""
        pipeline = FaceAugmentationPipeline(target_size=256)
        image = torch.rand(2, 3, 64, 64)

        hsv = pipeline._rgb_to_hsv(image)

        assert hsv.shape == image.shape

    def test_hsv_value_range(self):
        """Test HSV values are in [0, 1]."""
        pipeline = FaceAugmentationPipeline(target_size=256)
        image = torch.rand(1, 3, 64, 64)

        hsv = pipeline._rgb_to_hsv(image)

        assert hsv.min() >= 0
        assert hsv.max() <= 1


class TestIntegration:
    """Integration tests for the full data pipeline."""

    def test_augmented_batch_shape(self):
        """Test full pipeline produces correct batch shape."""
        pipeline = FaceAugmentationPipeline(target_size=256)

        batch_size = 4
        images = torch.randn(batch_size, 3, 256, 256)

        augmented = []
        for i in range(batch_size):
            aug, _ = pipeline(images[i])
            augmented.append(aug)

        batch = torch.stack(augmented)
        assert batch.shape == (batch_size, 3, 256, 256)

    def test_mask_and_image_consistent(self):
        """Test that geometric transforms are consistent between image and mask."""
        pipeline = FaceAugmentationPipeline(
            target_size=256,
            random_flip_prob=1.0,  # Always flip
            random_warp=False,
            apply_color=False,
        )

        # Create image with known pattern
        image = torch.zeros(3, 256, 256)
        image[:, :, :128] = 1.0  # Left half white

        # Create matching mask
        mask = torch.zeros(1, 256, 256)
        mask[:, :, :128] = 1.0

        aug_image, aug_mask = pipeline(image, mask)

        # After flip, right side should be white for both
        assert aug_image[:, :, 128:].mean() > 0.9
        assert aug_mask[:, :, 128:].mean() > 0.9

    def test_gradient_flow(self):
        """Test gradients flow through augmentation pipeline."""
        pipeline = FaceAugmentationPipeline(
            target_size=256,
            random_warp=False,  # Warp can break gradients
        )

        image = torch.rand(3, 256, 256, requires_grad=True)
        augmented, _ = pipeline(image)

        # Compute loss and backprop
        loss = augmented.sum()
        loss.backward()

        assert image.grad is not None
        assert image.grad.shape == image.shape
