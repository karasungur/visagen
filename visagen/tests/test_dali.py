"""
Tests for NVIDIA DALI data loading pipeline.

Tests cover:
    - DALI availability detection
    - Warp grid generation
    - DataModule fallback behavior
    - Integration with PyTorch Lightning
"""

import numpy as np
import pytest

from visagen.data.dali_pipeline import DALI_AVAILABLE, check_dali_available
from visagen.data.dali_warp import (
    DALIAffineGenerator,
    DALIWarpGridGenerator,
    apply_warp_grid_numpy,
    gen_dali_affine_matrix,
    gen_dali_warp_grid,
)


class TestDALIAvailability:
    """Tests for DALI availability detection."""

    def test_check_dali_available_returns_bool(self):
        """check_dali_available should return boolean."""
        result = check_dali_available()
        assert isinstance(result, bool)

    def test_dali_available_constant_matches_function(self):
        """DALI_AVAILABLE constant should match function result."""
        assert DALI_AVAILABLE == check_dali_available()


class TestDALIWarpGrid:
    """Tests for DFL-style warp grid generation."""

    def test_gen_dali_warp_grid_shape(self):
        """Generated grids should have correct shape."""
        size = 256
        batch_size = 4
        grids = gen_dali_warp_grid(size, batch_size)

        assert grids.shape == (batch_size, size, size, 2)
        assert grids.dtype == np.float32

    def test_gen_dali_warp_grid_range(self):
        """Generated grids should be in [-1, 1] range."""
        grids = gen_dali_warp_grid(256, batch_size=2)

        # Most values should be close to the [-1, 1] range
        # Slight overflow is possible due to displacement
        assert grids.min() >= -1.5
        assert grids.max() <= 1.5

    def test_gen_dali_warp_grid_reproducibility(self):
        """Same seed should produce identical grids."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)

        grid1 = gen_dali_warp_grid(128, batch_size=1, rng=rng1)
        grid2 = gen_dali_warp_grid(128, batch_size=1, rng=rng2)

        np.testing.assert_array_equal(grid1, grid2)

    def test_gen_dali_warp_grid_variability(self):
        """Different batches should have different grids."""
        grids = gen_dali_warp_grid(128, batch_size=4)

        # Each grid in batch should be different
        for i in range(3):
            assert not np.allclose(grids[i], grids[i + 1])

    def test_gen_dali_warp_grid_different_sizes(self):
        """Should work with various image sizes."""
        for size in [64, 128, 256, 512]:
            grids = gen_dali_warp_grid(size, batch_size=1)
            assert grids.shape == (1, size, size, 2)


class TestDALIWarpGridGenerator:
    """Tests for DALI external source warp grid generator."""

    def test_generator_initialization(self):
        """Generator should initialize without errors."""
        generator = DALIWarpGridGenerator(size=256, seed=42)
        assert generator.size == 256

    def test_generator_call(self):
        """Generator call should return single grid."""

        class MockSampleInfo:
            idx = 0
            idx_in_epoch = 0
            idx_in_batch = 0
            iteration = 0

        generator = DALIWarpGridGenerator(size=256, seed=42)
        grid = generator(MockSampleInfo())

        assert grid.shape == (256, 256, 2)
        assert grid.dtype == np.float32

    def test_generator_produces_different_grids(self):
        """Successive calls should produce different grids."""

        class MockSampleInfo:
            idx = 0
            idx_in_epoch = 0
            idx_in_batch = 0
            iteration = 0

        generator = DALIWarpGridGenerator(size=128, seed=42)
        grid1 = generator(MockSampleInfo())
        grid2 = generator(MockSampleInfo())

        # Successive calls should be different
        assert not np.allclose(grid1, grid2)

    def test_generator_reset_reseeds_epoch(self):
        """Reset should reseed generator for a new epoch stream."""

        class MockSampleInfo:
            idx = 0
            idx_in_epoch = 0
            idx_in_batch = 0
            iteration = 0

        generator = DALIWarpGridGenerator(size=128, seed=42)
        grid1 = generator(MockSampleInfo())
        generator.reset()
        grid2 = generator(MockSampleInfo())

        assert not np.allclose(grid1, grid2)


class TestDALIAffineMatrix:
    """Tests for affine transformation matrix generation."""

    def test_gen_dali_affine_matrix_shape(self):
        """Generated matrices should have correct shape."""
        matrices = gen_dali_affine_matrix(256, batch_size=4)

        assert matrices.shape == (4, 2, 3)
        assert matrices.dtype == np.float32

    def test_gen_dali_affine_matrix_identity_approximation(self):
        """With zero rotation/scale/translation, should approximate identity."""
        matrices = gen_dali_affine_matrix(
            size=256,
            batch_size=1,
            rotation_range=(0, 0),
            scale_range=(1.0, 1.0),
            translation_range=(0, 0),
        )

        # Should be close to [[1, 0, 0], [0, 1, 0]]
        expected = np.array([[[1, 0, 0], [0, 1, 0]]], dtype=np.float32)
        np.testing.assert_array_almost_equal(matrices, expected, decimal=5)

    def test_gen_dali_affine_matrix_reproducibility(self):
        """Same seed should produce identical matrices."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)

        m1 = gen_dali_affine_matrix(256, batch_size=2, rng=rng1)
        m2 = gen_dali_affine_matrix(256, batch_size=2, rng=rng2)

        np.testing.assert_array_equal(m1, m2)


class TestDALIAffineGenerator:
    """Tests for DALI external source affine generator."""

    def test_generator_initialization(self):
        """Generator should initialize with parameters."""
        generator = DALIAffineGenerator(
            size=256,
            rotation_range=(-15, 15),
            scale_range=(0.9, 1.1),
        )
        assert generator.size == 256
        assert generator.rotation_range == (-15, 15)

    def test_generator_call(self):
        """Generator call should return single matrix."""

        class MockSampleInfo:
            idx = 0
            idx_in_epoch = 0
            idx_in_batch = 0
            iteration = 0

        generator = DALIAffineGenerator(size=256, seed=42)
        matrix = generator(MockSampleInfo())

        assert matrix.shape == (2, 3)
        assert matrix.dtype == np.float32

    def test_generator_reset_changes_stream(self):
        """Reset should produce different affine samples for new epoch."""

        class MockSampleInfo:
            idx = 0
            idx_in_epoch = 0
            idx_in_batch = 0
            iteration = 0

        generator = DALIAffineGenerator(size=256, seed=42)
        m1 = generator(MockSampleInfo())
        generator.reset()
        m2 = generator(MockSampleInfo())

        assert not np.allclose(m1, m2)


class TestApplyWarpGridNumpy:
    """Tests for numpy-based warp application (CPU fallback)."""

    def test_apply_warp_identity_grid(self):
        """Identity grid should preserve image."""
        # Create identity grid
        h, w = 64, 64
        y = np.linspace(-1, 1, h)
        x = np.linspace(-1, 1, w)
        grid_x, grid_y = np.meshgrid(x, y)
        grid = np.stack([grid_x, grid_y], axis=-1).astype(np.float32)

        # Create test image
        image = np.random.rand(h, w, 3).astype(np.float32)

        # Apply warp
        warped = apply_warp_grid_numpy(image, grid)

        assert warped.shape == image.shape
        # Should be approximately the same
        np.testing.assert_array_almost_equal(warped, image, decimal=1)

    def test_apply_warp_grayscale(self):
        """Should handle grayscale images."""
        h, w = 64, 64
        y = np.linspace(-1, 1, h)
        x = np.linspace(-1, 1, w)
        grid_x, grid_y = np.meshgrid(x, y)
        grid = np.stack([grid_x, grid_y], axis=-1).astype(np.float32)

        image = np.random.rand(h, w).astype(np.float32)
        warped = apply_warp_grid_numpy(image, grid)

        assert warped.shape == image.shape

    def test_apply_warp_with_displacement(self):
        """Displaced grid should transform image."""
        h, w = 64, 64
        grid = gen_dali_warp_grid(64, batch_size=1)[0]

        image = np.random.rand(h, w, 3).astype(np.float32)
        warped = apply_warp_grid_numpy(image, grid)

        assert warped.shape == image.shape
        # Should be different from original
        assert not np.allclose(warped, image)


class TestDALIDataModuleFallback:
    """Tests for DALI DataModule fallback behavior."""

    def test_import_dali_loader(self):
        """Should be able to import dali_loader module."""
        from visagen.data.dali_loader import (
            DALIFaceDataModule,
            create_dali_datamodule,
        )

        assert DALIFaceDataModule is not None
        assert create_dali_datamodule is not None

    def test_datamodule_initialization(self, tmp_path):
        """DataModule should initialize with directories."""
        from visagen.data.dali_loader import DALIFaceDataModule

        # Create dummy directories
        src_dir = tmp_path / "src"
        dst_dir = tmp_path / "dst"
        src_dir.mkdir()
        dst_dir.mkdir()

        # Create dummy images
        for i in range(5):
            np.save(src_dir / f"img_{i}.npy", np.random.rand(256, 256, 3))
            np.save(dst_dir / f"img_{i}.npy", np.random.rand(256, 256, 3))

        dm = DALIFaceDataModule(
            src_dir=src_dir,
            dst_dir=dst_dir,
            batch_size=2,
            use_dali=False,  # Force fallback
        )

        assert dm.batch_size == 2
        assert not dm.using_dali

    def test_using_dali_property(self, tmp_path):
        """using_dali property should reflect actual usage."""
        from visagen.data.dali_loader import DALIFaceDataModule

        src_dir = tmp_path / "src"
        dst_dir = tmp_path / "dst"
        src_dir.mkdir()
        dst_dir.mkdir()

        # Force PyTorch fallback
        dm = DALIFaceDataModule(
            src_dir=src_dir,
            dst_dir=dst_dir,
            use_dali=False,
        )
        assert dm.using_dali is False

        # Check if DALI would be used
        dm_auto = DALIFaceDataModule(
            src_dir=src_dir,
            dst_dir=dst_dir,
            use_dali=None,  # Auto-detect
        )
        assert dm_auto.using_dali == DALI_AVAILABLE

    def test_create_dali_datamodule_pytorch_fallback_uses_datamodule_contract(
        self, tmp_path
    ):
        """Fallback path should map to FaceDataModule argument names correctly."""
        from visagen.data.dali_loader import create_dali_datamodule
        from visagen.data.datamodule import FaceDataModule

        src_dir = tmp_path / "src"
        dst_dir = tmp_path / "dst"
        src_dir.mkdir()
        dst_dir.mkdir()

        dm = create_dali_datamodule(
            src_dir=src_dir,
            dst_dir=dst_dir,
            batch_size=4,
            image_size=128,
            force_pytorch=True,
            num_workers=2,
            val_split=0.2,
            augmentation_config={"random_warp": False},
            uniform_yaw=True,
        )

        assert isinstance(dm, FaceDataModule)
        assert dm.target_size == 128
        assert dm.num_workers == 2
        assert dm.val_split == 0.2
        assert dm.uniform_yaw is True

    def test_dali_iterator_wrapper_returns_training_tuple_format(self):
        """Wrapper should expose (src_dict, dst_dict) expected by DFLModule."""
        from visagen.data.dali_loader import DALIIteratorWrapper

        class _FakeIterator:
            batch_size = 2

            def __iter__(self):
                return self

            def __next__(self):
                return [
                    {
                        "src_images": np.zeros((2, 3, 64, 64), dtype=np.float32),
                        "dst_images": np.ones((2, 3, 64, 64), dtype=np.float32),
                    }
                ]

            def reset(self):
                return None

            def epoch_size(self, _reader_name):
                return 8

        wrapped = DALIIteratorWrapper(_FakeIterator())
        src_dict, dst_dict = next(iter(wrapped))

        assert isinstance(src_dict, dict)
        assert isinstance(dst_dict, dict)
        assert "image" in src_dict
        assert "image" in dst_dict
        assert src_dict["image"].shape == (2, 3, 64, 64)


class TestDALIModuleExports:
    """Tests for module exports from __init__.py."""

    def test_all_exports_available(self):
        """All DALI exports should be importable."""
        from visagen.data import (
            DALI_AVAILABLE,
            DALIAffineGenerator,
            DALIFaceDataModule,
            DALIWarpGridGenerator,
            benchmark_dataloaders,
            check_dali_available,
            create_dali_datamodule,
            create_dali_iterator,
            gen_dali_affine_matrix,
            gen_dali_warp_grid,
        )

        # Just check they're not None
        assert DALI_AVAILABLE is not None or DALI_AVAILABLE is False
        assert check_dali_available is not None
        assert create_dali_iterator is not None
        assert gen_dali_warp_grid is not None
        assert DALIWarpGridGenerator is not None
        assert gen_dali_affine_matrix is not None
        assert DALIAffineGenerator is not None
        assert DALIFaceDataModule is not None
        assert create_dali_datamodule is not None
        assert benchmark_dataloaders is not None


@pytest.mark.skipif(not DALI_AVAILABLE, reason="DALI not installed")
class TestDALIPipelineIntegration:
    """Integration tests requiring DALI installation."""

    def test_create_dali_iterator(self, tmp_path):
        """Should create working DALI iterator."""
        from visagen.data.dali_pipeline import create_dali_iterator

        # Create test images
        src_dir = tmp_path / "src"
        dst_dir = tmp_path / "dst"
        src_dir.mkdir()
        dst_dir.mkdir()

        # Create actual JPEG files
        from PIL import Image

        for i in range(10):
            img = Image.fromarray(
                np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            )
            img.save(src_dir / f"img_{i}.jpg")
            img.save(dst_dir / f"img_{i}.jpg")

        iterator = create_dali_iterator(
            src_dir=src_dir,
            dst_dir=dst_dir,
            batch_size=2,
            image_size=128,
        )

        # Get one batch
        batch = next(iter(iterator))

        assert "src_images" in batch[0]
        assert "dst_images" in batch[0]
        assert batch[0]["src_images"].shape[0] == 2
        assert batch[0]["src_images"].shape[2:] == (128, 128)

    def test_dali_datamodule_training(self, tmp_path):
        """DALI DataModule should work with Lightning Trainer."""
        from PIL import Image

        from visagen.data.dali_loader import DALIFaceDataModule

        # Create test images
        src_dir = tmp_path / "src"
        dst_dir = tmp_path / "dst"
        src_dir.mkdir()
        dst_dir.mkdir()

        for i in range(20):
            img = Image.fromarray(
                np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            )
            img.save(src_dir / f"img_{i}.jpg")
            img.save(dst_dir / f"img_{i}.jpg")

        dm = DALIFaceDataModule(
            src_dir=src_dir,
            dst_dir=dst_dir,
            batch_size=4,
            image_size=128,
        )

        dm.setup("fit")
        loader = dm.train_dataloader()

        batch = next(iter(loader))
        assert batch is not None
