"""Contract tests for FrameProcessor detector invocation and filtering."""

from __future__ import annotations

from unittest.mock import Mock, patch

import numpy as np

from visagen.vision.detector import DetectedFace


def test_detect_called_without_threshold_and_filtered_by_confidence() -> None:
    """FrameProcessor should call detect() without threshold and filter afterwards."""
    from visagen.merger.frame_processor import FrameProcessor, FrameProcessorConfig

    config = FrameProcessorConfig(min_confidence=0.5, max_faces=10)

    with patch(
        "visagen.merger.frame_processor.FrameProcessor._load_model"
    ) as load_mock:
        load_mock.return_value = Mock()
        processor = FrameProcessor(Mock(), config=config, device="cpu")

    low_conf_face = DetectedFace(
        bbox=np.array([0, 0, 64, 64], dtype=np.float32),
        confidence=0.4,
        landmarks=np.random.rand(68, 2).astype(np.float32),
    )
    high_conf_face = DetectedFace(
        bbox=np.array([32, 32, 128, 128], dtype=np.float32),
        confidence=0.9,
        landmarks=np.random.rand(68, 2).astype(np.float32),
    )

    detector = Mock()
    detector.detect.return_value = [low_conf_face, high_conf_face]
    processor._detector = detector

    aligner = Mock()
    mat = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    aligner.align.return_value = (np.zeros((256, 256, 3), dtype=np.uint8), mat)
    aligner.get_transform_mat.return_value = mat
    aligner.face_type = "whole_face"
    processor._aligner = aligner
    object.__setattr__(
        processor,
        "_generate_mask",
        Mock(return_value=np.full((256, 256), 255.0, dtype=np.float32)),
    )

    frame = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    results = processor._detect_and_align(frame)

    assert len(results) == 1
    assert detector.detect.call_count == 1
    assert "threshold" not in detector.detect.call_args.kwargs
