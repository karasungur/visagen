"""Legacy util command parity wrappers for Visagen."""

from __future__ import annotations

import argparse
import pickle
import zipfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from visagen.vision.dflimg import DFLImage, FaceMetadata

SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def _iter_images(input_dir: Path) -> list[Path]:
    files: list[Path] = []
    for ext in SUPPORTED_IMAGE_EXTS:
        files.extend(input_dir.glob(f"*{ext}"))
    return sorted(files)


def add_landmarks_debug_images(input_path: Path) -> int:
    """Create *_debug.jpg files with landmark dots for DFL images."""
    created = 0
    for filepath in _iter_images(input_path):
        if filepath.stem.endswith("_debug"):
            continue

        image = cv2.imread(str(filepath), cv2.IMREAD_COLOR)
        if image is None:
            continue

        try:
            _img, metadata = DFLImage.load(filepath)
        except Exception:
            metadata = None

        if metadata is None:
            continue

        landmarks = np.asarray(metadata.landmarks, dtype=np.float32)
        if landmarks.ndim != 2 or landmarks.shape[1] != 2:
            continue

        debug_img = image.copy()
        for x, y in landmarks:
            cv2.circle(debug_img, (int(round(x)), int(round(y))), 1, (0, 255, 0), -1)

        output_file = filepath.parent / f"{filepath.stem}_debug.jpg"
        cv2.imwrite(str(output_file), debug_img, [cv2.IMWRITE_JPEG_QUALITY, 50])
        created += 1

    return created


def recover_original_aligned_filename(input_path: Path) -> int:
    """Rename aligned files based on DFL source_filename metadata."""
    entries: list[tuple[Path, str]] = []
    for filepath in _iter_images(input_path):
        try:
            _img, metadata = DFLImage.load(filepath)
        except Exception:
            metadata = None

        if metadata is None or not metadata.source_filename:
            continue

        source_stem = Path(metadata.source_filename).stem
        entries.append((filepath, source_stem))

    if not entries:
        return 0

    grouped: dict[str, list[Path]] = defaultdict(list)
    for filepath, source_stem in entries:
        grouped[source_stem].append(filepath)

    rename_map: dict[Path, Path] = {}
    for source_stem, files in grouped.items():
        for idx, src in enumerate(sorted(files)):
            rename_map[src] = src.parent / f"{source_stem}_{idx}{src.suffix.lower()}"

    # Two-phase rename to avoid collisions.
    temp_map: dict[Path, Path] = {}
    for src in sorted(rename_map):
        tmp = src.parent / f"{src.name}.visagen_tmp"
        src.rename(tmp)
        temp_map[tmp] = rename_map[src]

    renamed = 0
    for tmp, dst in temp_map.items():
        tmp.rename(dst)
        renamed += 1

    return renamed


def save_faceset_metadata_folder(input_path: Path) -> int:
    """Save DFL metadata for editable roundtrip as meta.dat."""
    metadata_path = input_path / "meta.dat"

    payload: dict[str, tuple[tuple[int, int, int], dict]] = {}
    for filepath in _iter_images(input_path):
        if filepath.suffix.lower() not in {".jpg", ".jpeg"}:
            continue
        try:
            image, metadata = DFLImage.load(filepath)
        except Exception:
            continue

        if metadata is None:
            continue

        payload[filepath.name] = (tuple(image.shape), metadata.to_dict())

    metadata_path.write_bytes(pickle.dumps(payload))
    return len(payload)


def restore_faceset_metadata_folder(input_path: Path) -> int:
    """Restore DFL metadata from meta.dat, resizing edited images back if needed."""
    metadata_path = input_path / "meta.dat"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    payload = pickle.loads(metadata_path.read_bytes())
    restored = 0

    for filename, (shape, meta_dict) in payload.items():
        filepath = input_path / filename
        if not filepath.exists():
            continue
        if filepath.suffix.lower() not in {".jpg", ".jpeg"}:
            continue

        image = cv2.imread(str(filepath), cv2.IMREAD_COLOR)
        if image is None:
            continue

        target_shape = tuple(int(v) for v in shape)
        if tuple(image.shape) != target_shape:
            image = cv2.resize(
                image,
                (target_shape[1], target_shape[0]),
                interpolation=cv2.INTER_LANCZOS4,
            )

        metadata = FaceMetadata.from_dict(meta_dict)
        DFLImage.save(filepath, image, metadata, quality=100)
        restored += 1

    metadata_path.unlink(missing_ok=True)
    return restored


def export_faceset_mask(input_dir: Path) -> int:
    """Export embedded xseg mask from DFL images to *_mask.png files."""
    exported = 0

    for filepath in _iter_images(input_dir):
        if filepath.stem.endswith("_mask"):
            continue
        if filepath.suffix.lower() not in {".jpg", ".jpeg"}:
            continue

        try:
            _image, metadata = DFLImage.load(filepath)
        except Exception:
            continue

        if metadata is None:
            continue

        mask = DFLImage.get_xseg_mask(metadata)
        if mask is None:
            continue

        if mask.ndim == 3:
            mask = mask[..., 0]
        mask_u8 = np.where(mask > 0.5, 255, 0).astype(np.uint8)
        output_path = filepath.parent / f"{filepath.stem}_mask.png"
        cv2.imwrite(str(output_path), mask_u8)
        exported += 1

    return exported


def pack_faceset(input_dir: Path) -> Path:
    """Pack faceset directory into a zip archive for transport/backups."""
    archive_path = input_dir.parent / f"{input_dir.name}.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in sorted(input_dir.rglob("*")):
            if file_path.is_file():
                zf.write(file_path, arcname=file_path.relative_to(input_dir))
    return archive_path


def unpack_faceset(input_dir: Path) -> int:
    """Unpack faceset zip archive back into input_dir."""
    archive_path = input_dir.parent / f"{input_dir.name}.zip"
    if not archive_path.exists():
        raise FileNotFoundError(f"Packed faceset not found: {archive_path}")

    input_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "r") as zf:
        zf.extractall(input_dir)
        return len(zf.namelist())


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Legacy util parity commands",
    )
    parser.add_argument(
        "--input-dir", type=Path, required=True, help="Faceset directory"
    )
    parser.add_argument(
        "--add-landmarks-debug-images",
        action="store_true",
        help="Add *_debug.jpg images with facial landmarks",
    )
    parser.add_argument(
        "--recover-original-aligned-filename",
        action="store_true",
        help="Recover filenames from DFL source_filename metadata",
    )
    parser.add_argument(
        "--save-faceset-metadata",
        action="store_true",
        help="Save metadata to meta.dat",
    )
    parser.add_argument(
        "--restore-faceset-metadata",
        action="store_true",
        help="Restore metadata from meta.dat",
    )
    parser.add_argument(
        "--pack-faceset",
        action="store_true",
        help="Pack faceset into <input-dir>.zip",
    )
    parser.add_argument(
        "--unpack-faceset",
        action="store_true",
        help="Unpack <input-dir>.zip into input directory",
    )
    parser.add_argument(
        "--export-faceset-mask",
        action="store_true",
        help="Export embedded xseg masks as *_mask.png",
    )
    return parser


def main() -> None:
    """CLI entrypoint."""
    parser = _build_parser()
    args = parser.parse_args()

    input_dir: Path = args.input_dir
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    actions_run = 0

    if args.add_landmarks_debug_images:
        created = add_landmarks_debug_images(input_dir)
        print(f"Created {created} landmark debug images")
        actions_run += 1

    if args.recover_original_aligned_filename:
        renamed = recover_original_aligned_filename(input_dir)
        print(f"Recovered {renamed} aligned filenames")
        actions_run += 1

    if args.save_faceset_metadata:
        saved = save_faceset_metadata_folder(input_dir)
        print(f"Saved metadata for {saved} files")
        actions_run += 1

    if args.restore_faceset_metadata:
        restored = restore_faceset_metadata_folder(input_dir)
        print(f"Restored metadata for {restored} files")
        actions_run += 1

    if args.pack_faceset:
        archive = pack_faceset(input_dir)
        print(f"Packed faceset to {archive}")
        actions_run += 1

    if args.unpack_faceset:
        count = unpack_faceset(input_dir)
        print(f"Unpacked {count} files into {input_dir}")
        actions_run += 1

    if args.export_faceset_mask:
        exported = export_faceset_mask(input_dir)
        print(f"Exported {exported} faceset masks")
        actions_run += 1

    if actions_run == 0:
        parser.error("At least one operation flag must be provided")


if __name__ == "__main__":
    main()
