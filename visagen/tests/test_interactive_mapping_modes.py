"""Tests for interactive merge mode and config mapping behavior."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np


def test_apply_config_maps_modes_to_expected_blend_and_color() -> None:
    from visagen.merger.frame_processor import FrameProcessorConfig
    from visagen.merger.interactive import InteractiveMerger

    merger = InteractiveMerger()

    class _StubProcessor:
        def __init__(self) -> None:
            self.config = FrameProcessorConfig()

    merger._processor = cast(Any, _StubProcessor())

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


def test_export_all_supports_cancellation(tmp_path: Path) -> None:
    from visagen.merger.interactive import InteractiveMerger

    checkpoint = tmp_path / "model.ckpt"
    checkpoint.touch()
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    output_dir = tmp_path / "out"

    for i in range(10):
        frame = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        cv2.imwrite(str(frames_dir / f"{i:06d}.png"), frame)

    merger = InteractiveMerger(
        checkpoint_path=checkpoint,
        frames_dir=frames_dir,
        output_dir=output_dir,
    )
    success, message = merger.load_session()
    assert success, message

    def _fake_export_frame(idx: int) -> tuple[bool, str]:
        time.sleep(0.01)
        return True, f"frame_{idx}.png"

    merger.export_frame = _fake_export_frame  # type: ignore[assignment]

    stop_event = threading.Event()

    def _cancel_after_three(current: int, total: int) -> None:
        if current >= 3:
            stop_event.set()

    success, message, exported = merger.export_all(
        progress_callback=_cancel_after_three,
        stop_event=stop_event,
    )

    assert success is False
    assert "cancelled" in message.lower()
    assert exported < len(merger.frames)
