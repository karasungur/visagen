"""Tests for blending branch wiring in FrameProcessor."""

from __future__ import annotations

from unittest.mock import Mock, patch

import numpy as np


def test_laplacian_branch_uses_laplacian_pyramid_blend() -> None:
    """Laplacian blend mode must call laplacian_pyramid_blend helper."""
    from visagen.merger.frame_processor import FrameProcessor, FrameProcessorConfig

    config = FrameProcessorConfig(blend_mode="laplacian")
    with patch(
        "visagen.merger.frame_processor.FrameProcessor._load_model"
    ) as load_mock:
        load_mock.return_value = Mock()
        processor = FrameProcessor(Mock(), config=config, device="cpu")

    bg = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    fg = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    mask = np.ones((128, 128), dtype=np.float32)

    with patch("visagen.postprocess.blending.laplacian_pyramid_blend") as lap_mock:
        lap_mock.return_value = np.zeros((128, 128, 3), dtype=np.float32)
        out = processor._apply_blend(bg, fg, mask)

    assert lap_mock.call_count == 1
    assert out.dtype == np.uint8
    assert out.shape == bg.shape


def test_post_poisson_hist_match_runs_only_for_hist_match_mode() -> None:
    """Post-poisson color transfer should run only in seamless-hist-match path."""
    from visagen.merger.frame_processor import FrameProcessor, FrameProcessorConfig

    frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    swapped = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    mask = np.ones((64, 64), dtype=np.float32)
    metadata = {
        "matrix": np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32),
        "original_shape": frame.shape[:2],
    }

    with patch(
        "visagen.merger.frame_processor.FrameProcessor._load_model"
    ) as load_model:
        load_model.return_value = Mock()
        processor_rct = FrameProcessor(
            Mock(),
            config=FrameProcessorConfig(
                output_size=64,
                blend_mode="poisson",
                color_transfer_mode="rct",
            ),
            device="cpu",
        )
        processor_none = FrameProcessor(
            Mock(),
            config=FrameProcessorConfig(
                output_size=64,
                blend_mode="poisson",
                color_transfer_mode=None,
            ),
            device="cpu",
        )
        processor_hist = FrameProcessor(
            Mock(),
            config=FrameProcessorConfig(
                output_size=64,
                blend_mode="poisson",
                color_transfer_mode="hist-match",
            ),
            device="cpu",
        )

    with (
        patch.object(processor_rct, "_apply_blend", return_value=frame.copy()),
        patch.object(processor_rct, "_apply_color_transfer") as rct_ct_mock,
    ):
        _ = processor_rct._blend_to_frame(frame, swapped, metadata, mask)
    assert rct_ct_mock.call_count == 0

    with (
        patch.object(processor_none, "_apply_blend", return_value=frame.copy()),
        patch.object(processor_none, "_apply_color_transfer") as none_ct_mock,
    ):
        _ = processor_none._blend_to_frame(frame, swapped, metadata, mask)
    assert none_ct_mock.call_count == 0

    with (
        patch.object(processor_hist, "_apply_blend", return_value=frame.copy()),
        patch.object(
            processor_hist,
            "_apply_color_transfer",
            return_value=swapped.copy(),
        ) as hist_ct_mock,
    ):
        _ = processor_hist._blend_to_frame(frame, swapped, metadata, mask)
    assert hist_ct_mock.call_count == 1
