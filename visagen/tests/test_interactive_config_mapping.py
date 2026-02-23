"""Tests for interactive merge mode/config mapping."""

from pathlib import Path

from visagen.merger.interactive_config import (
    InteractiveMergerConfig,
    InteractiveMergerSession,
    map_merge_mode_to_processor,
)
from visagen.merger.merger import MergerConfig


def test_map_merge_mode_to_processor_supported_modes() -> None:
    assert map_merge_mode_to_processor("overlay") == ("laplacian", None, False)
    assert map_merge_mode_to_processor("hist-match") == (
        "laplacian",
        "hist-match",
        False,
    )
    assert map_merge_mode_to_processor("seamless") == ("poisson", None, False)
    assert map_merge_mode_to_processor("seamless-hist-match") == (
        "poisson",
        "hist-match",
        False,
    )


def test_map_merge_mode_to_processor_alias() -> None:
    assert map_merge_mode_to_processor("original") == ("laplacian", None, True)


def test_interactive_config_normalizes_mask_mode() -> None:
    config = InteractiveMergerConfig(mask_mode="learned_prd")
    assert config.mask_mode == "segmented"


def test_interactive_config_accepts_dst_mask_mode() -> None:
    config = InteractiveMergerConfig(mask_mode="dst")
    assert config.mask_mode == "dst"


def test_interactive_config_accepts_hist_match_color_transfer() -> None:
    config = InteractiveMergerConfig(color_transfer="hist-match")
    assert config.color_transfer == "hist-match"


def test_merger_config_from_session_uses_mode_mapping(tmp_path: Path) -> None:
    session_path = tmp_path / "session.json"

    session = InteractiveMergerSession(
        frames_dir="/tmp/frames",
        checkpoint_path="/tmp/model.ckpt",
        output_dir="/tmp/out",
        config=InteractiveMergerConfig(
            mode="seamless-hist-match",
            mask_mode="segmented",
            color_transfer="rct",
            hist_match_threshold=222,
            masked_hist_match=True,
        ),
    )
    session.to_json(session_path)

    config = MergerConfig.from_session(session_path)

    assert config.frame_processor_config is not None
    assert config.frame_processor_config.blend_mode == "poisson"
    assert config.frame_processor_config.color_transfer_mode == "hist-match"
    assert config.frame_processor_config.mask_mode == "segmented"
    assert config.frame_processor_config.hist_match_threshold == 222
