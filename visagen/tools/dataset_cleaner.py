"""
Dataset cleaner for fast broken/black frame purging.

Usage:
    visagen-clean <input_dir> --broken --black-threshold 0.95 --dry-run
"""

from __future__ import annotations

import argparse
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from visagen.tools.dataset_trash import move_to_trash, resolve_collision_path


@dataclass
class CleanCandidate:
    """File candidate for cleanup."""

    filepath: Path
    reasons: list[str]


def _black_ratio(image: np.ndarray, threshold: int = 3) -> float:
    """Compute ratio of near-black pixels."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    black_count = int(np.sum(gray <= threshold))
    return float(black_count / gray.size)


def _analyze_file(
    filepath: Path,
    check_broken: bool,
    black_threshold: float | None,
) -> tuple[Path, list[str], str | None]:
    """Analyze a single file and return cleanup reasons."""
    reasons: list[str] = []
    try:
        if check_broken and filepath.stat().st_size == 0:
            reasons.append("zero-byte")
            return filepath, reasons, None

        image = cv2.imread(str(filepath))
        if image is None:
            if check_broken:
                reasons.append("unreadable")
            return filepath, reasons, None

        if black_threshold is not None:
            ratio = _black_ratio(image)
            if ratio >= black_threshold:
                reasons.append(f"black-ratio>={black_threshold:.2f}")

        return filepath, reasons, None
    except Exception as e:
        return filepath, reasons, str(e)


def get_image_paths(directory: Path) -> list[Path]:
    """Get all image paths from a directory."""
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
    paths: list[Path] = []

    for ext in extensions:
        paths.extend(directory.glob(f"*{ext}"))
        paths.extend(directory.glob(f"*{ext.upper()}"))

    return sorted(set(paths))


def parse_args(argv=None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Clean dataset by detecting broken and black frames.",
    )
    parser.add_argument("input", type=Path, help="Input dataset directory")
    parser.add_argument(
        "--broken",
        action="store_true",
        help="Mark broken files (0-byte or unreadable)",
    )
    parser.add_argument(
        "--black-threshold",
        type=float,
        default=None,
        help="Mark frames with black ratio >= threshold (0.0-1.0)",
    )
    parser.add_argument(
        "--trash-dir",
        type=Path,
        default=None,
        help="Custom trash directory (default: managed .visagen_trash)",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=None,
        help="Number of parallel workers (default: CPU count)",
    )
    parser.add_argument(
        "--exec-mode",
        type=str,
        choices=["auto", "process", "thread"],
        default="auto",
        help="Parallel execution mode (default: auto)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be cleaned without moving files",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose output",
    )
    args = parser.parse_args(argv)

    if not args.input.exists() or not args.input.is_dir():
        parser.error(f"Input directory not found: {args.input}")

    if not args.broken and args.black_threshold is None:
        parser.error(
            "At least one cleanup mode required: --broken and/or --black-threshold"
        )

    if args.black_threshold is not None and not (0.0 <= args.black_threshold <= 1.0):
        parser.error("--black-threshold must be between 0.0 and 1.0")

    return args


def main(argv=None) -> int:
    """CLI entry point."""
    args = parse_args(argv)
    image_paths = get_image_paths(args.input)

    print(f"Scanning {len(image_paths)} files in {args.input}")
    if not image_paths:
        print("No images found.")
        return 0

    if args.exec_mode == "thread":
        use_threads = True
    elif args.exec_mode == "process":
        use_threads = False
    else:
        # Auto prefers process mode when black-threshold (pixel math) is enabled.
        use_threads = args.black_threshold is None

    max_workers = args.jobs or (os.cpu_count() or 4)
    executor_class = ThreadPoolExecutor if use_threads else ProcessPoolExecutor

    candidates: list[CleanCandidate] = []
    errors: list[str] = []

    with executor_class(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _analyze_file,
                path,
                args.broken,
                args.black_threshold,
            ): path
            for path in image_paths
        }
        for future in as_completed(futures):
            path = futures[future]
            try:
                filepath, reasons, error = future.result()
            except Exception as e:
                errors.append(f"{path}: {e}")
                continue

            if error is not None:
                errors.append(f"{filepath}: {error}")
                continue
            if reasons:
                candidates.append(CleanCandidate(filepath=filepath, reasons=reasons))

    print(f"Marked for cleanup: {len(candidates)}")
    if errors:
        print(f"Analysis errors: {len(errors)}")

    if args.verbose:
        preview = candidates[:20]
        for item in preview:
            print(f"  - {item.filepath.name}: {', '.join(item.reasons)}")
        if len(candidates) > len(preview):
            print(f"  ... {len(candidates) - len(preview)} more")

    if args.dry_run or not candidates:
        print("[Dry run - no changes made]" if args.dry_run else "Nothing to clean.")
        return 0

    paths_to_clean = [item.filepath for item in candidates]
    if args.trash_dir is None:
        batch = move_to_trash(
            paths_to_clean,
            dataset_root=args.input,
            reason="dataset-cleaner",
        )
        print(
            "Cleanup moved to managed trash: "
            f"batch={batch.batch_id}, moved={getattr(batch, 'count_moved', batch.count)}, "
            f"missing={getattr(batch, 'count_missing', 0)}, "
            f"failed={getattr(batch, 'count_failed', 0)}"
        )
    else:
        args.trash_dir.mkdir(parents=True, exist_ok=True)
        moved = 0
        failed = 0
        for path in paths_to_clean:
            try:
                destination = args.trash_dir / path.name
                if destination.exists():
                    destination = resolve_collision_path(destination)
                shutil.move(str(path), str(destination))
                moved += 1
            except Exception:
                failed += 1
        print(
            f"Cleanup moved files to {args.trash_dir}: moved={moved}, failed={failed}"
        )

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
