"""Tests for dataset cleaner tool."""

from pathlib import Path

import cv2
import numpy as np

from visagen.tools import dataset_cleaner


def _write_uniform_image(path: Path, value: int) -> None:
    """Write a simple uniform image."""
    image = np.full((64, 64, 3), value, dtype=np.uint8)
    cv2.imwrite(str(path), image)


def test_analyze_zero_byte_file(tmp_path: Path):
    """Zero-byte files should be flagged when broken checks are enabled."""
    broken = tmp_path / "broken.jpg"
    broken.write_bytes(b"")

    path, reasons, error = dataset_cleaner._analyze_file(
        broken,
        check_broken=True,
        black_threshold=None,
    )

    assert path == broken
    assert reasons == ["zero-byte"]
    assert error is None


def test_analyze_black_frame(tmp_path: Path):
    """Mostly black frame should be flagged by threshold."""
    black = tmp_path / "black.jpg"
    _write_uniform_image(black, 0)

    path, reasons, error = dataset_cleaner._analyze_file(
        black,
        check_broken=False,
        black_threshold=0.95,
    )

    assert path == black
    assert len(reasons) == 1
    assert reasons[0].startswith("black-ratio>=")
    assert error is None


def test_main_dry_run_keeps_files(tmp_path: Path):
    """Dry-run should detect candidates but not move files."""
    black = tmp_path / "black.jpg"
    bright = tmp_path / "bright.jpg"
    _write_uniform_image(black, 0)
    _write_uniform_image(bright, 255)

    rc = dataset_cleaner.main(
        [
            str(tmp_path),
            "--black-threshold",
            "0.95",
            "--exec-mode",
            "thread",
            "--jobs",
            "1",
            "--dry-run",
        ]
    )

    assert rc == 0
    assert black.exists()
    assert bright.exists()


def test_main_managed_trash_move(tmp_path: Path):
    """Cleaner should move matched files into managed trash when not dry-run."""
    broken = tmp_path / "broken.jpg"
    broken.write_bytes(b"")

    rc = dataset_cleaner.main(
        [
            str(tmp_path),
            "--broken",
            "--exec-mode",
            "thread",
            "--jobs",
            "1",
        ]
    )

    manifest = tmp_path / ".visagen_trash" / "manifest.jsonl"
    assert rc == 0
    assert not broken.exists()
    assert manifest.exists()


def test_main_custom_trash_collision(tmp_path: Path):
    """Custom trash directory should resolve name collisions safely."""
    broken = tmp_path / "broken.jpg"
    broken.write_bytes(b"")

    custom_trash = tmp_path / "custom_trash"
    custom_trash.mkdir(parents=True, exist_ok=True)
    (custom_trash / "broken.jpg").write_bytes(b"existing")

    rc = dataset_cleaner.main(
        [
            str(tmp_path),
            "--broken",
            "--trash-dir",
            str(custom_trash),
            "--exec-mode",
            "thread",
            "--jobs",
            "1",
        ]
    )

    assert rc == 0
    assert not broken.exists()
    assert (custom_trash / "broken.jpg").exists()
    assert (custom_trash / "broken_restored_1.jpg").exists()
