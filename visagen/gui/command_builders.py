"""Centralized CLI command builders used by Gradio tabs."""

from __future__ import annotations

import sys
from pathlib import Path


def build_extract_command(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    face_type: str = "whole_face",
    output_size: int = 512,
    min_confidence: float | None = None,
) -> list[str]:
    """Build command for `visagen.tools.extract_v2`."""
    cmd = [
        sys.executable,
        "-m",
        "visagen.tools.extract_v2",
        str(input_path),
        str(output_dir),
        "--face-type",
        face_type,
        "--size",
        str(int(output_size)),
    ]

    if min_confidence is not None:
        cmd.extend(["--min-confidence", str(float(min_confidence))])

    return cmd


def build_train_command(
    src_dir: str | Path,
    dst_dir: str | Path,
    output_dir: str | Path,
    *,
    batch_size: int,
    max_epochs: int,
    precision: str | None = None,
    lpips_weight: float | None = None,
) -> list[str]:
    """Build command for `visagen.tools.train`."""
    cmd = [
        sys.executable,
        "-m",
        "visagen.tools.train",
        "--src-dir",
        str(src_dir),
        "--dst-dir",
        str(dst_dir),
        "--output-dir",
        str(output_dir),
        "--batch-size",
        str(int(batch_size)),
        "--max-epochs",
        str(int(max_epochs)),
    ]

    if precision is not None:
        cmd.extend(["--precision", precision])
    if lpips_weight is not None:
        cmd.extend(["--lpips-weight", str(float(lpips_weight))])

    return cmd


def build_merge_command(
    input_path: str | Path,
    output_path: str | Path,
    checkpoint_path: str | Path,
    *,
    color_transfer: str | None = None,
    blend_mode: str | None = None,
    restore_face: bool = False,
    restore_strength: float | None = None,
    restore_model: str | float | None = None,
    codec: str | None = None,
    crf: int | None = None,
) -> list[str]:
    """Build command for `visagen.tools.merge`."""
    cmd = [
        sys.executable,
        "-m",
        "visagen.tools.merge",
        str(input_path),
        str(output_path),
        "--checkpoint",
        str(checkpoint_path),
    ]

    if color_transfer is not None:
        cmd.extend(["--color-transfer", color_transfer])
    if blend_mode is not None:
        cmd.extend(["--blend-mode", blend_mode])
    if restore_face:
        cmd.append("--restore-face")
        if restore_strength is not None:
            cmd.extend(["--restore-strength", str(float(restore_strength))])
        if restore_model is not None:
            cmd.extend(["--restore-model", str(restore_model)])
    if codec is not None:
        cmd.extend(["--codec", codec])
    if crf is not None:
        cmd.extend(["--crf", str(int(crf))])

    return cmd


def build_video_extract_command(
    input_video: str | Path,
    output_dir: str | Path,
    *,
    fps: float | None = None,
    output_format: str = "png",
) -> list[str]:
    """Build command for `visagen.tools.video_ed extract`."""
    cmd = [
        sys.executable,
        "-m",
        "visagen.tools.video_ed",
        "extract",
        str(input_video),
        str(output_dir),
        "--format",
        output_format,
    ]
    if fps is not None and fps > 0:
        cmd.extend(["--fps", str(float(fps))])
    return cmd


def build_video_create_command(
    input_dir: str | Path,
    output_video: str | Path,
    *,
    fps: float = 30.0,
    codec: str = "libx264",
    bitrate: str | None = None,
) -> list[str]:
    """Build command for `visagen.tools.video_ed create`."""
    cmd = [
        sys.executable,
        "-m",
        "visagen.tools.video_ed",
        "create",
        str(input_dir),
        str(output_video),
        "--fps",
        str(float(fps)),
        "--codec",
        codec,
    ]
    if bitrate:
        cmd.extend(["--bitrate", bitrate])
    return cmd


def build_video_cut_command(
    input_video: str | Path,
    output_video: str | Path,
    *,
    start_time: str,
    end_time: str,
    codec: str = "copy",
    audio_track_id: int = 0,
    bitrate: str | None = None,
) -> list[str]:
    """Build command for `visagen.tools.video_ed cut`."""
    cmd = [
        sys.executable,
        "-m",
        "visagen.tools.video_ed",
        "cut",
        str(input_video),
        str(output_video),
        "--start",
        start_time,
        "--end",
        end_time,
    ]
    if codec:
        cmd.extend(["--codec", codec])
    cmd.extend(["--audio-track-id", str(max(0, int(audio_track_id)))])
    if bitrate:
        cmd.extend(["--bitrate", bitrate])
    return cmd


def build_video_denoise_command(
    input_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    factor: int = 7,
) -> list[str]:
    """Build command for `visagen.tools.video_ed denoise`."""
    cmd = [
        sys.executable,
        "-m",
        "visagen.tools.video_ed",
        "denoise",
        str(input_dir),
        "--factor",
        str(int(factor)),
    ]
    if output_dir:
        cmd.extend(["--output", str(output_dir)])
    return cmd
