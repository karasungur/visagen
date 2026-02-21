"""FrameProcessor color transfer mode behavior tests."""

from __future__ import annotations

from unittest.mock import Mock, patch

import numpy as np
import pytest


def _make_processor(mode: str):
    from visagen.merger.frame_processor import FrameProcessor, FrameProcessorConfig

    config = FrameProcessorConfig(color_transfer_mode=mode)
    with patch("visagen.merger.frame_processor.FrameProcessor._load_model") as load:
        load.return_value = Mock()
        return FrameProcessor(Mock(), config=config, device="cpu")


def test_apply_color_transfer_supports_known_mode() -> None:
    processor = _make_processor("rct")
    swapped = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    target = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)

    result = processor._apply_color_transfer(swapped, target)
    assert result.shape == swapped.shape
    assert result.dtype == np.uint8


def test_apply_color_transfer_raises_on_unsupported_mode() -> None:
    processor = _make_processor("unsupported-mode")
    swapped = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    target = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)

    with pytest.raises(ValueError):
        _ = processor._apply_color_transfer(swapped, target)
