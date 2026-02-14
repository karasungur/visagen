"""CLI for LabelMe/COCO mask annotation export and import."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import cv2
import numpy as np

from visagen.vision.mask_export import (
    export_coco,
    export_labelme,
    import_coco,
    import_labelme,
)

SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def _iter_images(input_dir: Path) -> list[Path]:
    files: list[Path] = []
    for ext in SUPPORTED_IMAGE_EXTS:
        files.extend(input_dir.glob(f"*{ext}"))
    return sorted(files)


def _normalize_binary_mask(mask: np.ndarray) -> np.ndarray:
    """Normalize mask to uint8 binary format {0, 255}."""
    if mask.ndim == 3:
        mask = mask[..., 0]

    if mask.dtype != np.uint8:
        mask = mask.astype(np.float32)
        max_val = float(mask.max()) if mask.size > 0 else 0.0
        if max_val <= 1.0:
            mask = mask * 255.0
        mask = np.clip(mask, 0.0, 255.0).astype(np.uint8)

    return cast(np.ndarray, np.where(mask > 127, 255, 0).astype(np.uint8))


def _load_mask(image_path: Path) -> np.ndarray | None:
    """Load mask from sidecar PNG first, then DFL metadata for JPEG."""
    sidecar = image_path.parent / f"{image_path.stem}_mask.png"
    if sidecar.exists():
        sidecar_mask = cv2.imread(str(sidecar), cv2.IMREAD_GRAYSCALE)
        if sidecar_mask is not None:
            return _normalize_binary_mask(sidecar_mask)

    if image_path.suffix.lower() not in {".jpg", ".jpeg"}:
        return None

    try:
        from visagen.vision.dflimg import DFLImage

        _image, metadata = DFLImage.load(image_path)
        if metadata is None:
            return None

        mask = DFLImage.get_xseg_mask(metadata)
        if mask is None:
            return None

        return _normalize_binary_mask(mask)
    except Exception:
        return None


def _export_annotations(args: argparse.Namespace) -> int:
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    images = _iter_images(input_dir)
    if not images:
        raise RuntimeError(f"No images found in {input_dir}")

    masks_by_name: dict[str, np.ndarray] = {}
    for image_path in images:
        mask = _load_mask(image_path)
        if mask is not None:
            masks_by_name[image_path.name] = mask

    if not masks_by_name:
        raise RuntimeError(
            "No masks found (expected *_mask.png sidecars or DFL embedded xseg masks)"
        )

    if args.format == "labelme":
        out_dir = Path(args.output)
        if out_dir.suffix.lower() == ".json":
            out_dir = out_dir.parent
        out_dir.mkdir(parents=True, exist_ok=True)

        exported = 0
        for image_path in images:
            mask = masks_by_name.get(image_path.name)
            if mask is None:
                continue
            export_labelme(
                image_path=image_path,
                mask=mask,
                output_path=out_dir / f"{image_path.stem}.json",
                label=args.label,
                include_image_data=bool(args.include_image_data),
                min_area=int(args.min_area),
            )
            exported += 1

        print(f"Exported {exported} LabelMe annotation files to {out_dir}")
        return exported

    out_file = Path(args.output)
    if out_file.suffix.lower() != ".json":
        out_file = out_file / "annotations.json"

    ordered_paths = [path for path in images if path.name in masks_by_name]
    ordered_masks = [masks_by_name[path.name] for path in ordered_paths]
    export_coco(
        image_paths=ordered_paths,
        masks=ordered_masks,
        output_path=out_file,
        categories=[args.label],
        min_area=int(args.min_area),
    )
    print(f"Exported COCO annotations for {len(ordered_paths)} images to {out_file}")
    return len(ordered_paths)


def _write_imported_mask(output_dir: Path, image_name: str, mask: np.ndarray) -> Path:
    """Write imported mask to sidecar PNG path in output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(image_name).stem
    mask_path = output_dir / f"{stem}_mask.png"
    cv2.imwrite(str(mask_path), _normalize_binary_mask(mask))
    return mask_path


def _import_annotations(args: argparse.Namespace) -> int:
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Annotation path not found: {input_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.format == "labelme":
        json_files = (
            [input_path] if input_path.is_file() else sorted(input_path.glob("*.json"))
        )
        if not json_files:
            raise RuntimeError(f"No LabelMe JSON files found in {input_path}")

        imported = 0
        for json_path in json_files:
            mask, metadata = import_labelme(json_path)
            image_name = metadata.get("image_path") or f"{json_path.stem}.jpg"
            _write_imported_mask(output_dir, image_name, mask)
            imported += 1

        print(f"Imported {imported} LabelMe masks to {output_dir}")
        return imported

    coco_file = input_path
    if coco_file.is_dir():
        coco_file = coco_file / "annotations.json"
    if not coco_file.exists():
        raise FileNotFoundError(f"COCO annotation file not found: {coco_file}")

    imported = 0
    for image_name, (mask, _labels) in import_coco(coco_file).items():
        _write_imported_mask(output_dir, image_name, mask)
        imported += 1

    print(f"Imported {imported} COCO masks to {output_dir}")
    return imported


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visagen mask annotation import/export utilities",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser(
        "export", help="Export masks to LabelMe or COCO"
    )
    export_parser.add_argument(
        "--input-dir", type=Path, required=True, help="Directory of face images"
    )
    export_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory or output JSON path",
    )
    export_parser.add_argument(
        "--format", choices=["labelme", "coco"], default="labelme"
    )
    export_parser.add_argument(
        "--label", type=str, default="face", help="Annotation label/category name"
    )
    export_parser.add_argument(
        "--include-image-data",
        action="store_true",
        help="Include base64 image data in LabelMe JSON",
    )
    export_parser.add_argument(
        "--min-area", type=int, default=100, help="Minimum polygon area"
    )

    import_parser = subparsers.add_parser(
        "import", help="Import masks from LabelMe or COCO"
    )
    import_parser.add_argument(
        "--input", type=Path, required=True, help="Annotation file or directory"
    )
    import_parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for *_mask.png files",
    )
    import_parser.add_argument(
        "--format", choices=["labelme", "coco"], default="labelme"
    )

    return parser


def main() -> None:
    """CLI entrypoint."""
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "export":
        _export_annotations(args)
        return

    if args.command == "import":
        _import_annotations(args)
        return

    parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
