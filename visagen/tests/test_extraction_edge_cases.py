"""Edge case tests for extraction module."""

import numpy as np
import pytest

from visagen.vision.aligner import FaceAligner


class TestAlignerValidation:
    """Validation tests for FaceAligner."""

    def test_nan_landmarks_rejected(self):
        """NaN landmarks should raise ValueError."""
        aligner = FaceAligner()
        landmarks = np.full((106, 2), np.nan)
        with pytest.raises(ValueError, match="NaN"):
            aligner.convert_106_to_68(landmarks)

    def test_inf_landmarks_rejected(self):
        """Infinite landmarks should raise ValueError."""
        aligner = FaceAligner()
        landmarks = np.full((106, 2), np.inf)
        with pytest.raises(ValueError, match="infinite"):
            aligner.convert_106_to_68(landmarks)

    def test_negative_inf_landmarks_rejected(self):
        """Negative infinite landmarks should raise ValueError."""
        aligner = FaceAligner()
        landmarks = np.full((106, 2), -np.inf)
        with pytest.raises(ValueError, match="infinite"):
            aligner.convert_106_to_68(landmarks)

    def test_valid_landmarks_pass(self):
        """Valid landmarks should not raise."""
        aligner = FaceAligner()
        landmarks = np.random.rand(106, 2) * 512
        result = aligner.convert_106_to_68(landmarks)
        assert result.shape == (68, 2)

    def test_5_point_landmarks_rejected(self):
        """5-point landmarks should raise ValueError."""
        aligner = FaceAligner()
        landmarks = np.random.rand(5, 2) * 512
        with pytest.raises(ValueError, match="5-point"):
            aligner.convert_106_to_68(landmarks)

    def test_wrong_shape_rejected(self):
        """Wrong shape landmarks should raise ValueError."""
        aligner = FaceAligner()
        landmarks = np.random.rand(106, 3)  # Wrong second dimension
        with pytest.raises(ValueError, match="shape"):
            aligner.convert_106_to_68(landmarks)


class TestExtractionProgressDataclass:
    """Tests for ExtractionProgress dataclass."""

    def test_progress_default_values(self):
        """Test default values for new fields."""
        from visagen.tools.extract_v2 import ExtractionProgress

        progress = ExtractionProgress(
            current_frame=10,
            total_frames=100,
            faces_extracted=5,
        )
        assert progress.fps == 0.0
        assert progress.elapsed_seconds == 0.0
        assert progress.eta_seconds == 0.0

    def test_progress_with_metrics(self):
        """Test progress with metrics values."""
        from visagen.tools.extract_v2 import ExtractionProgress

        progress = ExtractionProgress(
            current_frame=50,
            total_frames=100,
            faces_extracted=25,
            fps=15.5,
            elapsed_seconds=3.2,
            eta_seconds=3.3,
        )
        assert progress.fps == 15.5
        assert progress.elapsed_seconds == 3.2
        assert progress.eta_seconds == 3.3


class TestAppSettingsExtraction:
    """Tests for extraction settings in AppSettings."""

    def test_default_extraction_settings(self):
        """Test default extraction settings."""
        from visagen.gui.state.app_state import AppSettings

        settings = AppSettings()
        assert settings.extraction_output_size == 512
        assert settings.extraction_face_type == "whole_face"
        assert settings.extraction_min_confidence == 0.5
        assert settings.extraction_jpeg_quality == 95
        assert settings.extraction_auto_mask is True
        assert settings.segmentation_batch_size == 8
        assert settings.detector_warmup is True

    def test_extraction_settings_serialization(self):
        """Test extraction settings serialization."""
        from visagen.gui.state.app_state import AppSettings

        settings = AppSettings(
            extraction_output_size=256,
            extraction_face_type="full",
            extraction_min_confidence=0.7,
        )
        data = settings.to_dict()
        assert data["extraction_output_size"] == 256
        assert data["extraction_face_type"] == "full"
        assert data["extraction_min_confidence"] == 0.7

    def test_extraction_settings_deserialization(self):
        """Test extraction settings deserialization."""
        from visagen.gui.state.app_state import AppSettings

        data = {
            "device": "cuda",
            "extraction_output_size": 1024,
            "extraction_face_type": "head",
        }
        settings = AppSettings.from_dict(data)
        assert settings.device == "cuda"
        assert settings.extraction_output_size == 1024
        assert settings.extraction_face_type == "head"
