"""
Tests for vision module components.

These tests verify the vision pipeline works correctly
without requiring actual model weights or GPU.
"""

import pytest
import numpy as np

from visagen.vision.face_type import FaceType, FACE_TYPE_TO_PADDING
from visagen.vision.aligner import (
    FaceAligner,
    AlignedFace,
    umeyama,
    transform_points,
    LANDMARKS_2D_NEW,
    LANDMARKS_68_3D,
)
from visagen.vision.dflimg import FaceMetadata, DFLImage


class TestFaceType:
    """Tests for FaceType enum."""

    def test_face_type_values(self):
        """Verify FaceType enum values."""
        assert FaceType.HALF == 0
        assert FaceType.MID_FULL == 1
        assert FaceType.FULL == 2
        assert FaceType.WHOLE_FACE == 3
        assert FaceType.HEAD == 4
        assert FaceType.HEAD_NO_ALIGN == 10

    def test_face_type_to_string(self):
        """Verify string conversion."""
        assert FaceType.to_string(FaceType.HALF) == "half"
        assert FaceType.to_string(FaceType.WHOLE_FACE) == "whole_face"
        assert FaceType.to_string(FaceType.HEAD) == "head"

    def test_face_type_from_string(self):
        """Verify string parsing."""
        assert FaceType.from_string("half") == FaceType.HALF
        assert FaceType.from_string("whole_face") == FaceType.WHOLE_FACE
        assert FaceType.from_string("HEAD") == FaceType.HEAD  # Case insensitive

    def test_face_type_padding_values(self):
        """Verify padding values are defined for all types."""
        for face_type in [FaceType.HALF, FaceType.MID_FULL, FaceType.FULL,
                          FaceType.WHOLE_FACE, FaceType.HEAD]:
            assert face_type in FACE_TYPE_TO_PADDING
            padding, remove_align = FACE_TYPE_TO_PADDING[face_type]
            assert isinstance(padding, float)
            assert isinstance(remove_align, bool)

    def test_face_type_padding_increasing(self):
        """Verify padding increases from HALF to HEAD."""
        paddings = [
            FACE_TYPE_TO_PADDING[FaceType.HALF][0],
            FACE_TYPE_TO_PADDING[FaceType.MID_FULL][0],
            FACE_TYPE_TO_PADDING[FaceType.FULL][0],
            FACE_TYPE_TO_PADDING[FaceType.WHOLE_FACE][0],
            FACE_TYPE_TO_PADDING[FaceType.HEAD][0],
        ]
        # Each should be >= previous
        for i in range(1, len(paddings)):
            assert paddings[i] >= paddings[i - 1]


class TestUmeyama:
    """Tests for Umeyama transform."""

    def test_identity_transform(self):
        """Same points should give identity transform."""
        points = np.array([
            [0, 0],
            [1, 0],
            [1, 1],
            [0, 1],
        ], dtype=np.float32)

        mat = umeyama(points, points, estimate_scale=True)

        # Should be close to identity (3x3)
        assert mat.shape == (3, 3)
        np.testing.assert_allclose(mat[:2, :2], np.eye(2), atol=1e-6)
        np.testing.assert_allclose(mat[:2, 2], [0, 0], atol=1e-6)

    def test_translation_only(self):
        """Pure translation should be detected."""
        src = np.array([[0, 0], [1, 0], [1, 1]], dtype=np.float32)
        dst = src + np.array([10, 20])

        mat = umeyama(src, dst, estimate_scale=False)

        # Rotation should be identity, translation should be [10, 20]
        np.testing.assert_allclose(mat[:2, :2], np.eye(2), atol=1e-6)
        np.testing.assert_allclose(mat[:2, 2], [10, 20], atol=1e-6)

    def test_scale_estimation(self):
        """Scale should be correctly estimated."""
        src = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
        dst = src * 2.0  # Scale by 2

        mat = umeyama(src, dst, estimate_scale=True)

        # Scale should be approximately 2
        scale = np.sqrt(mat[0, 0] ** 2 + mat[0, 1] ** 2)
        np.testing.assert_allclose(scale, 2.0, atol=1e-5)


class TestTransformPoints:
    """Tests for point transformation."""

    def test_identity_transform(self):
        """Identity matrix should not change points."""
        points = np.array([[10, 20], [30, 40]], dtype=np.float32)
        mat = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)

        result = transform_points(points, mat)

        np.testing.assert_allclose(result, points, atol=1e-6)

    def test_translation(self):
        """Translation should shift all points."""
        points = np.array([[0, 0], [10, 10]], dtype=np.float32)
        mat = np.array([[1, 0, 5], [0, 1, 10]], dtype=np.float32)

        result = transform_points(points, mat)

        expected = np.array([[5, 10], [15, 20]], dtype=np.float32)
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_inverse_transform(self):
        """Inverse should recover original points."""
        points = np.array([[10, 20], [30, 40]], dtype=np.float32)
        mat = np.array([[2, 0, 5], [0, 2, 10]], dtype=np.float32)

        forward = transform_points(points, mat, invert=False)
        backward = transform_points(forward, mat, invert=True)

        np.testing.assert_allclose(backward, points, atol=1e-5)


class TestLandmarkTemplates:
    """Tests for landmark templates."""

    def test_landmarks_2d_shape(self):
        """2D landmarks should have correct shape."""
        assert LANDMARKS_2D_NEW.shape == (33, 2)
        assert LANDMARKS_2D_NEW.dtype == np.float32

    def test_landmarks_2d_normalized(self):
        """2D landmarks should be in [0, 1] range."""
        assert np.all(LANDMARKS_2D_NEW >= 0)
        assert np.all(LANDMARKS_2D_NEW <= 1)

    def test_landmarks_68_3d_shape(self):
        """3D landmarks should have correct shape."""
        assert LANDMARKS_68_3D.shape == (68, 3)
        assert LANDMARKS_68_3D.dtype == np.float32


class TestFaceAligner:
    """Tests for FaceAligner class."""

    @pytest.fixture
    def aligner(self):
        """Create FaceAligner instance."""
        return FaceAligner()

    @pytest.fixture
    def dummy_landmarks_68(self):
        """Create dummy 68-point landmarks."""
        # Generate reasonable face landmarks in a 256x256 image
        np.random.seed(42)
        landmarks = np.zeros((68, 2), dtype=np.float32)

        # Jaw (0-16)
        for i in range(17):
            landmarks[i] = [50 + i * 10, 100 + abs(i - 8) * 3]

        # Eyebrows (17-26)
        for i in range(5):
            landmarks[17 + i] = [70 + i * 10, 70]
            landmarks[22 + i] = [140 + i * 10, 70]

        # Nose (27-35)
        for i in range(9):
            landmarks[27 + i] = [128, 80 + i * 10]

        # Eyes (36-47)
        for i in range(6):
            landmarks[36 + i] = [85 + i * 7, 85]
            landmarks[42 + i] = [148 + i * 7, 85]

        # Mouth (48-67)
        for i in range(20):
            angle = i * np.pi / 10
            landmarks[48 + i] = [128 + np.cos(angle) * 30, 180 + np.sin(angle) * 15]

        return landmarks

    def test_get_transform_mat_shape(self, aligner, dummy_landmarks_68):
        """Transform matrix should have correct shape."""
        mat = aligner.get_transform_mat(
            dummy_landmarks_68,
            output_size=256,
            face_type=FaceType.WHOLE_FACE,
        )

        assert mat.shape == (2, 3)
        assert mat.dtype in [np.float32, np.float64]

    def test_get_transform_mat_all_face_types(self, aligner, dummy_landmarks_68):
        """Transform should work for all face types."""
        for face_type in [FaceType.HALF, FaceType.MID_FULL, FaceType.FULL,
                          FaceType.WHOLE_FACE, FaceType.HEAD]:
            mat = aligner.get_transform_mat(
                dummy_landmarks_68,
                output_size=256,
                face_type=face_type,
            )
            assert mat.shape == (2, 3)

    def test_convert_106_to_68(self, aligner):
        """106 to 68 conversion should produce correct shape."""
        landmarks_106 = np.random.randn(106, 2).astype(np.float32)
        landmarks_68 = aligner.convert_106_to_68(landmarks_106)

        assert landmarks_68.shape == (68, 2)

    def test_estimate_pitch_yaw_roll_range(self, aligner, dummy_landmarks_68):
        """Pose estimation should return values in valid range."""
        pitch, yaw, roll = aligner.estimate_pitch_yaw_roll(dummy_landmarks_68, size=256)

        half_pi = np.pi / 2
        assert -half_pi <= pitch <= half_pi
        assert -half_pi <= yaw <= half_pi
        assert -half_pi <= roll <= half_pi


class TestFaceMetadata:
    """Tests for FaceMetadata dataclass."""

    @pytest.fixture
    def sample_metadata(self):
        """Create sample metadata."""
        return FaceMetadata(
            landmarks=np.random.randn(68, 2).astype(np.float32),
            source_landmarks=np.random.randn(68, 2).astype(np.float32),
            source_rect=(10, 20, 200, 220),
            source_filename="test.jpg",
            face_type="whole_face",
            image_to_face_mat=np.eye(2, 3, dtype=np.float32),
            eyebrows_expand_mod=1.0,
        )

    def test_to_dict(self, sample_metadata):
        """Metadata should convert to dict."""
        d = sample_metadata.to_dict()

        assert isinstance(d, dict)
        assert "landmarks" in d
        assert "source_landmarks" in d
        assert "source_rect" in d
        assert "source_filename" in d
        assert "face_type" in d
        assert "image_to_face_mat" in d

    def test_from_dict_roundtrip(self, sample_metadata):
        """Dict conversion should be reversible."""
        d = sample_metadata.to_dict()
        recovered = FaceMetadata.from_dict(d)

        np.testing.assert_allclose(
            recovered.landmarks, sample_metadata.landmarks, atol=1e-6
        )
        assert recovered.source_filename == sample_metadata.source_filename
        assert recovered.face_type == sample_metadata.face_type

    def test_from_dict_minimal(self):
        """Minimal dict should work with defaults."""
        minimal = {
            "landmarks": [[0, 0]] * 68,
            "image_to_face_mat": [[1, 0, 0], [0, 1, 0]],
        }

        metadata = FaceMetadata.from_dict(minimal)

        assert metadata.landmarks.shape == (68, 2)
        assert metadata.eyebrows_expand_mod == 1.0
        assert metadata.xseg_mask is None


class TestDFLImage:
    """Tests for DFLImage class."""

    def test_has_metadata_nonexistent_file(self, tmp_path):
        """Non-existent file should return False."""
        fake_path = tmp_path / "nonexistent.jpg"
        assert DFLImage.has_metadata(fake_path) is False

    def test_jpeg_markers(self):
        """JPEG markers should have correct values."""
        assert DFLImage._SOI == 0xD8
        assert DFLImage._EOI == 0xD9
        assert DFLImage._APP15 == 0xEF

    def test_parse_empty_data(self):
        """Empty data should return None."""
        result = DFLImage._parse_jpeg_metadata(b"")
        assert result is None

    def test_parse_invalid_jpeg(self):
        """Invalid JPEG should return None."""
        result = DFLImage._parse_jpeg_metadata(b"not a jpeg file")
        assert result is None
