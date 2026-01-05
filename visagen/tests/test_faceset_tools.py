"""Tests for faceset tools (enhancer and resizer)."""

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from visagen.tools.faceset_resizer import (
    FACE_TYPE_MULTS,
    ResizeResult,
    calculate_target_resolution,
    resize_faceset,
    resize_single_face,
)


class TestFaceTypeMultipliers:
    """Tests for face type multiplier constants."""

    def test_multipliers_exist(self) -> None:
        """Test that all face types have multipliers."""
        expected = ["half_face", "mid_face", "full_face", "whole_face", "head"]
        for face_type in expected:
            assert face_type in FACE_TYPE_MULTS

    def test_multipliers_ordered(self) -> None:
        """Test that multipliers are in ascending order."""
        values = [
            FACE_TYPE_MULTS["half_face"],
            FACE_TYPE_MULTS["mid_face"],
            FACE_TYPE_MULTS["full_face"],
            FACE_TYPE_MULTS["whole_face"],
            FACE_TYPE_MULTS["head"],
        ]
        assert values == sorted(values)


class TestCalculateTargetResolution:
    """Tests for target resolution calculation."""

    def test_no_face_type_change(self) -> None:
        """Test resolution with no face type change."""
        result = calculate_target_resolution(256, 512, "full_face", None)
        assert result == 512

    def test_same_face_type(self) -> None:
        """Test resolution with same face type."""
        result = calculate_target_resolution(256, 512, "full_face", "full_face")
        assert result == 512

    def test_face_type_conversion(self) -> None:
        """Test resolution with face type conversion."""
        # half_face (1.0) to full_face (1.5) should increase resolution
        result = calculate_target_resolution(256, 256, "half_face", "full_face")
        expected = int(256 * 1.5 / 1.0)  # 384
        assert result == expected

    def test_missing_source_face_type(self) -> None:
        """Test with missing source face type."""
        result = calculate_target_resolution(256, 512, None, "full_face")
        assert result == 512


class TestResizeSingleFace:
    """Tests for single face resizing."""

    def test_resize_basic(self) -> None:
        """Test basic face resizing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.png"
            output_path = Path(tmpdir) / "output.png"

            # Create test image
            img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            cv2.imwrite(str(input_path), img)

            # Resize
            result = resize_single_face(
                input_path,
                output_path,
                target_size=128,
                preserve_metadata=False,
            )

            assert result is True
            assert output_path.exists()

            # Check output size
            output_img = cv2.imread(str(output_path))
            assert output_img.shape[:2] == (128, 128)

    def test_resize_upscale(self) -> None:
        """Test upscaling."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.png"
            output_path = Path(tmpdir) / "output.png"

            # Create small test image
            img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            cv2.imwrite(str(input_path), img)

            # Upscale
            result = resize_single_face(
                input_path,
                output_path,
                target_size=256,
                preserve_metadata=False,
            )

            assert result is True
            output_img = cv2.imread(str(output_path))
            assert output_img.shape[:2] == (256, 256)

    def test_resize_missing_input(self) -> None:
        """Test with missing input file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = resize_single_face(
                Path(tmpdir) / "nonexistent.png",
                Path(tmpdir) / "output.png",
                target_size=256,
            )
            assert result is False

    def test_resize_jpeg(self) -> None:
        """Test JPEG output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.jpg"
            output_path = Path(tmpdir) / "output.jpg"

            # Create test image
            img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            cv2.imwrite(str(input_path), img)

            result = resize_single_face(
                input_path,
                output_path,
                target_size=128,
                jpeg_quality=90,
                preserve_metadata=False,
            )

            assert result is True
            assert output_path.exists()


class TestResizeFaceset:
    """Tests for batch faceset resizing."""

    def test_resize_empty_directory(self) -> None:
        """Test with empty input directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "input"
            input_dir.mkdir()

            result = resize_faceset(input_dir, target_size=256)

            assert result.total_images == 0
            assert result.resized_count == 0

    def test_resize_missing_directory(self) -> None:
        """Test with non-existent directory."""
        with pytest.raises(FileNotFoundError):
            resize_faceset(Path("/nonexistent/path"), target_size=256)

    def test_resize_batch(self) -> None:
        """Test batch resizing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "input"
            input_dir.mkdir()

            # Create test images
            for i in range(5):
                img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
                cv2.imwrite(str(input_dir / f"face_{i:03d}.png"), img)

            # Resize batch
            result = resize_faceset(
                input_dir,
                target_size=128,
                num_workers=2,
            )

            assert result.total_images == 5
            assert result.resized_count == 5
            assert result.error_count == 0
            assert result.output_dir.exists()

            # Check output files
            output_files = list(result.output_dir.glob("*.png"))
            assert len(output_files) == 5

            # Check output sizes
            for f in output_files:
                img = cv2.imread(str(f))
                assert img.shape[:2] == (128, 128)

    def test_resize_skip_already_correct(self) -> None:
        """Test that already correct size images are skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "input"
            output_dir = Path(tmpdir) / "output"
            input_dir.mkdir()

            # Create test images at target size
            for i in range(3):
                img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
                cv2.imwrite(str(input_dir / f"face_{i:03d}.png"), img)

            # Resize to same size
            result = resize_faceset(
                input_dir,
                output_dir,
                target_size=256,
            )

            assert result.total_images == 3
            # Should be skipped since already correct size
            assert result.skipped_count == 3

    def test_resize_mixed_formats(self) -> None:
        """Test with mixed image formats."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "input"
            input_dir.mkdir()

            # Create PNG images
            for i in range(2):
                img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
                cv2.imwrite(str(input_dir / f"face_{i:03d}.png"), img)

            # Create JPEG images
            for i in range(2):
                img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
                cv2.imwrite(str(input_dir / f"face_{i:03d}.jpg"), img)

            result = resize_faceset(input_dir, target_size=128)

            assert result.total_images == 4
            assert result.resized_count == 4

    def test_resize_result_dataclass(self) -> None:
        """Test ResizeResult dataclass."""
        result = ResizeResult(
            total_images=10,
            resized_count=8,
            skipped_count=1,
            error_count=1,
            output_dir=Path("/tmp/output"),
        )
        assert result.total_images == 10
        assert result.resized_count == 8
        assert result.skipped_count == 1
        assert result.error_count == 1


class TestFacesetEnhancer:
    """Tests for faceset enhancer (mocked due to GFPGAN dependency)."""

    def test_enhance_result_import(self) -> None:
        """Test that EnhanceResult can be imported."""
        from visagen.tools.faceset_enhancer import EnhanceResult

        result = EnhanceResult(
            total_images=10,
            enhanced_count=9,
            skipped_count=0,
            error_count=1,
            output_dir=Path("/tmp/output"),
        )
        assert result.total_images == 10

    def test_gfpgan_availability_check(self) -> None:
        """Test GFPGAN availability is checked."""
        from visagen.tools.faceset_enhancer import is_gfpgan_available

        # Just verify the function exists and returns bool
        result = is_gfpgan_available()
        assert isinstance(result, bool)

    def test_enhance_faceset_no_gfpgan(self) -> None:
        """Test enhance_faceset without GFPGAN."""
        from visagen.tools.faceset_enhancer import enhance_faceset

        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "input"
            input_dir.mkdir()

            # Create test image
            img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            cv2.imwrite(str(input_dir / "face.png"), img)

            # Should raise if GFPGAN not available
            from visagen.tools.faceset_enhancer import is_gfpgan_available

            if not is_gfpgan_available():
                with pytest.raises(RuntimeError, match="GFPGAN"):
                    enhance_faceset(input_dir)
            else:
                # If GFPGAN is available, it should work
                result = enhance_faceset(input_dir, strength=0.5)
                assert result.total_images == 1

    def test_enhance_missing_directory(self) -> None:
        """Test enhance with non-existent directory."""
        from visagen.tools.faceset_enhancer import enhance_faceset

        with pytest.raises(FileNotFoundError):
            enhance_faceset(Path("/nonexistent/path"))
