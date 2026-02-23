"""Parity tests for util command wrappers."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

from visagen.tools.util_compat import (
    export_faceset_mask,
    main,
    recover_original_aligned_filename,
    restore_faceset_metadata_folder,
    save_faceset_metadata_folder,
)
from visagen.vision.face_image import FaceImage, FaceMetadata


def _create_metadata(source_filename: str) -> FaceMetadata:
    return FaceMetadata(
        landmarks=np.zeros((68, 2), dtype=np.float32),
        source_landmarks=np.zeros((68, 2), dtype=np.float32),
        source_rect=(0, 0, 31, 31),
        source_filename=source_filename,
        face_type="whole_face",
        image_to_face_mat=np.eye(2, 3, dtype=np.float32),
    )


def _create_face_image(
    path: Path, source_filename: str, with_mask: bool = False
) -> None:
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[:, :] = 120
    metadata = _create_metadata(source_filename)

    if with_mask:
        mask = np.zeros((32, 32, 1), dtype=np.float32)
        mask[8:24, 8:24, 0] = 1.0
        FaceImage.set_xseg_mask(metadata, mask)

    FaceImage.save(path, image, metadata)


def test_save_restore_faceset_metadata_roundtrip(tmp_path: Path) -> None:
    faceset_dir = tmp_path / "faces"
    faceset_dir.mkdir()
    face_path = faceset_dir / "face.jpg"
    _create_face_image(face_path, source_filename="source.mp4")

    saved = save_faceset_metadata_folder(faceset_dir)
    assert saved == 1
    assert (faceset_dir / "meta.dat").exists()

    # Simulate external edit with wrong size.
    small = np.zeros((16, 16, 3), dtype=np.uint8)
    cv2.imwrite(str(face_path), small)

    restored = restore_faceset_metadata_folder(faceset_dir)
    assert restored == 1
    assert not (faceset_dir / "meta.dat").exists()

    image, metadata = FaceImage.load(face_path)
    assert image.shape[:2] == (32, 32)
    assert metadata is not None
    assert metadata.source_filename == "source.mp4"


def test_recover_original_aligned_filename(tmp_path: Path) -> None:
    faceset_dir = tmp_path / "faces"
    faceset_dir.mkdir()

    _create_face_image(faceset_dir / "a.jpg", source_filename="clip.mp4")
    _create_face_image(faceset_dir / "b.jpg", source_filename="clip.mp4")

    renamed = recover_original_aligned_filename(faceset_dir)
    assert renamed == 2
    assert (faceset_dir / "clip_0.jpg").exists()
    assert (faceset_dir / "clip_1.jpg").exists()


def test_export_faceset_mask_writes_sidecar_png(tmp_path: Path) -> None:
    faceset_dir = tmp_path / "faces"
    faceset_dir.mkdir()

    _create_face_image(
        faceset_dir / "masked.jpg", source_filename="masked.mp4", with_mask=True
    )

    exported = export_faceset_mask(faceset_dir)
    assert exported == 1
    assert (faceset_dir / "masked_mask.png").exists()


def test_cli_main_runs_selected_flag(monkeypatch, tmp_path: Path) -> None:
    faceset_dir = tmp_path / "faces"
    faceset_dir.mkdir()
    _create_face_image(faceset_dir / "face.jpg", source_filename="source.mp4")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "visagen-util",
            "--input-dir",
            str(faceset_dir),
            "--save-faceset-metadata",
        ],
    )
    main()

    assert (faceset_dir / "meta.dat").exists()
