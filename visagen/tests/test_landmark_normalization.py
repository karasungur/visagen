"""Tests for landmark normalization behavior in FrameProcessor."""

from __future__ import annotations

from unittest.mock import Mock, patch

import numpy as np
import pytest


def _make_processor():
    from visagen.merger.frame_processor import FrameProcessor

    with patch("visagen.merger.frame_processor.FrameProcessor._load_model") as load:
        load.return_value = Mock()
        return FrameProcessor(Mock(), device="cpu")


def test_normalize_landmarks_68_passthrough() -> None:
    processor = _make_processor()
    lm68 = np.random.rand(68, 2).astype(np.float32)
    out = processor._normalize_landmarks(lm68)
    assert out.shape == (68, 2)


def test_normalize_landmarks_106_to_68() -> None:
    processor = _make_processor()
    lm106 = np.random.rand(106, 2).astype(np.float32)
    out = processor._normalize_landmarks(lm106)
    assert out.shape == (68, 2)


def test_normalize_landmarks_5_point_raises() -> None:
    processor = _make_processor()
    lm5 = np.random.rand(5, 2).astype(np.float32)
    with pytest.raises(ValueError):
        processor._normalize_landmarks(lm5)
