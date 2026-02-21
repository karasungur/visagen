"""CLI tests for visagen.tools.mask_export."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

from visagen.tools.mask_export import main


def _create_image_and_mask(image_dir: Path, stem: str = "face") -> None:
    image_dir.mkdir(parents=True, exist_ok=True)
    image = np.zeros((48, 48, 3), dtype=np.uint8)
    image[8:40, 8:40] = 200
    cv2.imwrite(str(image_dir / f"{stem}.png"), image)

    mask = np.zeros((48, 48), dtype=np.uint8)
    mask[12:36, 12:36] = 255
    cv2.imwrite(str(image_dir / f"{stem}_mask.png"), mask)


def test_cli_export_and_import_labelme(monkeypatch, tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    _create_image_and_mask(image_dir, "face")

    labelme_dir = tmp_path / "labelme"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "visagen-mask-export",
            "export",
            "--input-dir",
            str(image_dir),
            "--output",
            str(labelme_dir),
            "--format",
            "labelme",
        ],
    )
    main()

    json_path = labelme_dir / "face.json"
    assert json_path.exists()

    imported_dir = tmp_path / "imported"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "visagen-mask-export",
            "import",
            "--input",
            str(labelme_dir),
            "--output-dir",
            str(imported_dir),
            "--format",
            "labelme",
        ],
    )
    main()

    assert (imported_dir / "face_mask.png").exists()


def test_cli_export_and_import_coco(monkeypatch, tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    _create_image_and_mask(image_dir, "face")

    coco_path = tmp_path / "annotations.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "visagen-mask-export",
            "export",
            "--input-dir",
            str(image_dir),
            "--output",
            str(coco_path),
            "--format",
            "coco",
        ],
    )
    main()

    assert coco_path.exists()
    data = json.loads(coco_path.read_text())
    assert data["annotations"]

    imported_dir = tmp_path / "imported"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "visagen-mask-export",
            "import",
            "--input",
            str(coco_path),
            "--output-dir",
            str(imported_dir),
            "--format",
            "coco",
        ],
    )
    main()

    assert (imported_dir / "face_mask.png").exists()
