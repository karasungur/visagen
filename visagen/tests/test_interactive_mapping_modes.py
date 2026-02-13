"""Tests for interactive merge mode and config mapping behavior."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def test_apply_config_maps_modes_to_expected_blend_and_color() -> None:
    from visagen.merger.frame_processor import FrameProcessorConfig
    from visagen.merger.interactive import InteractiveMerger

    merger = InteractiveMerger()

    class _StubProcessor:
        def __init__(self) -> None:
            self.config = FrameProcessorConfig()

    merger._processor = _StubProcessor()

    merger.session.config.mode = "hist-match"
    merger.session.config.color_transfer = "rct"
    merger.session.config.mask_mode = "dst"
    merger.session.config.hist_match_threshold = 211
    merger.session.config.face_scale = 7
    merger.session.config.sharpen_mode = "box"
    merger.session.config.sharpen_amount = 45

    merger._apply_config_to_processor()

    cfg = merger.processor.config
    assert cfg.blend_mode == "laplacian"
    assert cfg.color_transfer_mode == "hist-match"
    assert cfg.mask_mode == "dst"
    assert cfg.hist_match_threshold == 211
    assert cfg.face_scale == 7
    assert cfg.sharpen is True
    assert cfg.sharpen_amount == 0.45


def test_original_mode_returns_unprocessed_frame(tmp_path: Path) -> None:
    from visagen.merger.interactive import InteractiveMerger

    frame_bgr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    frame_path = tmp_path / "000001.png"
    cv2.imwrite(str(frame_path), frame_bgr)

    merger = InteractiveMerger()
    merger.frames = [frame_path]
    merger.session.current_idx = 0
    merger.session.config.mode = "original"

    out = merger.process_current_frame()
    assert out is not None
    expected = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    np.testing.assert_array_equal(out, expected)
