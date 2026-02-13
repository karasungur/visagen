"""Tests for GUI CLI command builders."""

from visagen.gui.command_builders import (
    build_extract_command,
    build_merge_command,
    build_train_command,
    build_video_create_command,
    build_video_cut_command,
    build_video_denoise_command,
    build_video_extract_command,
)


def test_build_extract_command_uses_extract_v2_positional_args() -> None:
    cmd = build_extract_command(
        "input.mp4",
        "out_dir",
        face_type="full",
        output_size=256,
        min_confidence=0.65,
    )

    assert cmd[2] == "visagen.tools.extract_v2"
    assert cmd[3] == "input.mp4"
    assert cmd[4] == "out_dir"
    assert "--face-type" in cmd
    assert "--size" in cmd
    assert "--min-confidence" in cmd


def test_build_merge_command_uses_positional_input_output() -> None:
    cmd = build_merge_command(
        "input.mp4",
        "output.mp4",
        "model.ckpt",
        color_transfer="rct",
        blend_mode="laplacian",
        restore_face=True,
        restore_strength=0.7,
        restore_model="1.4",
        codec="libx264",
        crf=18,
    )

    assert cmd[2] == "visagen.tools.merge"
    assert cmd[3] == "input.mp4"
    assert cmd[4] == "output.mp4"
    assert "--checkpoint" in cmd
    assert "--color-transfer" in cmd
    assert "--blend-mode" in cmd
    assert "--restore-face" in cmd
    assert "--restore-strength" in cmd
    assert "--restore-model" in cmd
    assert "--codec" in cmd
    assert "--crf" in cmd


def test_build_train_command_includes_core_paths_and_limits() -> None:
    cmd = build_train_command(
        "src/aligned",
        "dst/aligned",
        "model",
        batch_size=16,
        max_epochs=400,
    )

    assert cmd[2] == "visagen.tools.train"
    assert "--src-dir" in cmd
    assert "--dst-dir" in cmd
    assert "--output-dir" in cmd
    assert "--batch-size" in cmd
    assert "--max-epochs" in cmd


def test_build_train_command_supports_preset_overrides() -> None:
    cmd = build_train_command(
        "src/aligned",
        "dst/aligned",
        "model",
        batch_size=8,
        max_epochs=120,
        precision="16-mixed",
        lpips_weight=1.0,
    )

    assert "--precision" in cmd
    assert "--lpips-weight" in cmd


def test_build_video_extract_command_uses_positional_output() -> None:
    cmd = build_video_extract_command(
        "video.mp4", "frames", fps=25.0, output_format="jpg"
    )

    assert cmd[2] == "visagen.tools.video_ed"
    assert cmd[3] == "extract"
    assert cmd[4] == "video.mp4"
    assert cmd[5] == "frames"
    assert "--format" in cmd
    assert "--fps" in cmd


def test_build_video_create_command_uses_positional_output() -> None:
    cmd = build_video_create_command("frames", "out.mp4", fps=30.0, codec="libx264")

    assert cmd[3] == "create"
    assert cmd[4] == "frames"
    assert cmd[5] == "out.mp4"


def test_build_video_cut_command_uses_positional_output() -> None:
    cmd = build_video_cut_command(
        "input.mp4",
        "cut.mp4",
        start_time="00:00:01",
        end_time="00:00:05",
    )

    assert cmd[3] == "cut"
    assert cmd[4] == "input.mp4"
    assert cmd[5] == "cut.mp4"
    assert "--start" in cmd
    assert "--end" in cmd


def test_build_video_denoise_command_optional_output() -> None:
    cmd = build_video_denoise_command("frames", output_dir="denoised", factor=9)

    assert cmd[3] == "denoise"
    assert cmd[4] == "frames"
    assert "--factor" in cmd
    assert "--output" in cmd
