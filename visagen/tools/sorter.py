"""
Visagen Sort - Face image sorting and filtering tool.

Usage:
    visagen-sort <input_dir> [--method METHOD] [--target N] [--output OUTPUT]

Examples:
    visagen-sort ./aligned_faces --method blur
    visagen-sort ./aligned_faces --method final --target 2000
    visagen-sort ./aligned_faces --method face-yaw --output sorted/
    visagen-sort ./aligned_faces --undo-last-trash
"""

import argparse
import shutil
import sys
from pathlib import Path


def get_sort_methods():
    """Get available sorting methods."""
    from visagen.sorting import (
        AbsDiffDissimilaritySorter,
        AbsDiffSorter,
        BlackPixelSorter,
        BlurFastSorter,
        BlurSorter,
        BrightnessSorter,
        FinalSorter,
        HistogramDissimilaritySorter,
        HistogramSimilaritySorter,
        HueSorter,
        IDDissimilaritySorter,
        IDSimilaritySorter,
        MotionBlurSorter,
        OneFaceSorter,
        OrigNameSorter,
        PitchSorter,
        SourceRectSorter,
        SSIMDissimilaritySorter,
        SSIMSimilaritySorter,
        YawSorter,
    )
    from visagen.sorting.composite import FinalFastSorter

    return {
        "blur": BlurSorter,
        "blur-fast": BlurFastSorter,
        "motion-blur": MotionBlurSorter,
        "face-yaw": YawSorter,
        "face-pitch": PitchSorter,
        "face-source-rect-size": SourceRectSorter,
        "hist": HistogramSimilaritySorter,
        "hist-dissim": HistogramDissimilaritySorter,
        "absdiff": AbsDiffSorter,
        "absdiff-dissim": AbsDiffDissimilaritySorter,
        "id-sim": IDSimilaritySorter,
        "id-dissim": IDDissimilaritySorter,
        "ssim": SSIMSimilaritySorter,
        "ssim-dissim": SSIMDissimilaritySorter,
        "brightness": BrightnessSorter,
        "hue": HueSorter,
        "black": BlackPixelSorter,
        "origname": OrigNameSorter,
        "oneface": OneFaceSorter,
        "final": FinalSorter,
        "final-fast": FinalFastSorter,
    }


def parse_args(argv=None):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Sort face images by various criteria",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  visagen-sort ./aligned_faces --method blur
  visagen-sort ./aligned_faces --method final --target 2000
  visagen-sort ./aligned_faces -m face-yaw -o sorted/
  visagen-sort ./aligned_faces -m hist --dry-run
  visagen-sort ./aligned_faces --undo-last-trash

Available methods:
  blur              Sort by image sharpness (CPBD)
  blur-fast         Sort by image sharpness (fast Laplacian)
  motion-blur       Sort by motion blur
  face-yaw          Sort by face yaw angle (left-right)
  face-pitch        Sort by face pitch angle (up-down)
  face-source-rect-size  Sort by face size in source image
  hist              Sort by histogram similarity (groups similar)
  hist-dissim       Sort by histogram dissimilarity (unique first)
  absdiff           Sort by absolute pixel difference (GPU, similar)
  absdiff-dissim    Sort by absolute pixel difference (GPU, dissimilar)
  id-sim            Sort by identity similarity (embedding based)
  id-dissim         Sort by identity dissimilarity (embedding outliers)
  ssim              Sort by SSIM similarity (groups similar)
  ssim-dissim       Sort by SSIM dissimilarity (outliers first)
  brightness        Sort by brightness
  hue               Sort by hue
  black             Sort by amount of black pixels
  origname          Sort by original source filename
  oneface           Filter to keep only single-face frames
  final             Select best faces with pose variety
  final-fast        Fast version of final (less accurate)
""",
    )

    parser.add_argument(
        "input",
        type=Path,
        help="Input directory containing aligned face images",
    )

    parser.add_argument(
        "-m",
        "--method",
        default="blur",
        help="Sorting method (default: blur)",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output directory (default: rename in place)",
    )

    parser.add_argument(
        "-t",
        "--target",
        type=int,
        default=2000,
        help="Target count for 'final' method (default: 2000)",
    )

    parser.add_argument(
        "--trash-dir",
        type=Path,
        default=None,
        help="Directory for discarded images (default: managed <input>/.visagen_trash)",
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
        "--exact-limit",
        type=int,
        default=None,
        help="Override exact O(n^2) cutoff for methods that support it",
    )

    parser.add_argument(
        "--undo-last-trash",
        action="store_true",
        help="Undo last managed trash batch for this dataset and exit",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )

    parser.add_argument(
        "--no-rename",
        action="store_true",
        help="Don't rename files with numeric prefix",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args(argv)

    # Validate input
    if not args.input.exists():
        parser.error(f"Input directory does not exist: {args.input}")

    if not args.input.is_dir():
        parser.error(f"Input is not a directory: {args.input}")

    if args.exact_limit is not None and args.exact_limit < 0:
        parser.error("--exact-limit must be >= 0")

    # Validate method
    if not args.undo_last_trash:
        methods = get_sort_methods()
        if args.method not in methods:
            parser.error(
                f"Unknown method: {args.method}\n"
                f"Available methods: {', '.join(methods.keys())}"
            )

    return args


def get_image_paths(directory: Path) -> list[Path]:
    """Get all image paths from a directory."""
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    paths: list[Path] = []

    for ext in extensions:
        paths.extend(directory.glob(f"*{ext}"))
        paths.extend(directory.glob(f"*{ext.upper()}"))

    return sorted(paths)


def apply_sort_result(
    result,
    input_dir: Path,
    output_dir: Path | None,
    trash_dir: Path | None,
    no_rename: bool = False,
    dry_run: bool = False,
    verbose: bool = False,
):
    """Apply sorting result by renaming/moving files."""
    from visagen.sorting.base import SortOutput
    from visagen.tools.dataset_trash import move_to_trash, resolve_collision_path

    sort_output: SortOutput = result

    # Handle trash
    if len(sort_output.trash_images) > 0:
        if trash_dir is None:
            print(f"Moving {len(sort_output.trash_images)} images to managed trash...")
            if not dry_run:
                batch = move_to_trash(
                    [item.filepath for item in sort_output.trash_images],
                    dataset_root=input_dir,
                    reason=f"sort:{sort_output.method}",
                )
                moved_count = getattr(batch, "count_moved", batch.count)
                missing_count = getattr(batch, "count_missing", 0)
                failed_count = getattr(batch, "count_failed", 0)
                print(
                    "  Trash batch: "
                    f"{batch.batch_id} | moved: {moved_count} | missing: {missing_count} "
                    f"| failed: {failed_count} | dir: {batch.trash_dir}"
                )
        else:
            if not dry_run:
                trash_dir.mkdir(parents=True, exist_ok=True)

            print(f"Moving {len(sort_output.trash_images)} images to {trash_dir}")

            for item in sort_output.trash_images:
                src = item.filepath
                dst = trash_dir / src.name
                if dst.exists():
                    dst = resolve_collision_path(dst)

                if verbose:
                    print(f"  Trash: {src.name}")

                if not dry_run:
                    try:
                        shutil.move(str(src), str(dst))
                    except Exception as e:
                        print(f"  Failed to move {src.name}: {e}")

    # Handle sorted images
    if len(sort_output.sorted_images) > 0:
        target_dir = output_dir if output_dir else input_dir

        if output_dir and not dry_run:
            output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Processing {len(sort_output.sorted_images)} sorted images")

        if no_rename:
            # Just copy/move without renaming
            if output_dir:
                for item in sort_output.sorted_images:
                    src = item.filepath
                    dst = output_dir / src.name

                    if verbose:
                        print(f"  Copy: {src.name}")

                    if not dry_run:
                        try:
                            shutil.copy2(str(src), str(dst))
                        except Exception as e:
                            print(f"  Failed to copy {src.name}: {e}")
        else:
            # Two-phase rename to avoid conflicts
            # Phase 1: Rename to temp names
            temp_names: list[tuple[Path, Path, Path]] = []

            for i, item in enumerate(sort_output.sorted_images):
                src = item.filepath
                temp_name = target_dir / f"_sort_temp_{i:08d}_{src.name}"
                final_name = target_dir / f"{i:08d}{src.suffix}"
                temp_names.append((src, temp_name, final_name))

            if verbose:
                print("  Phase 1: Renaming to temp names...")

            if not dry_run:
                for src, temp, _final in temp_names:
                    try:
                        if output_dir:
                            shutil.copy2(str(src), str(temp))
                        else:
                            src.rename(temp)
                    except Exception as e:
                        print(f"  Failed to rename {src.name}: {e}")

            # Phase 2: Rename to final names
            if verbose:
                print("  Phase 2: Renaming to final names...")

            if not dry_run:
                for _src, temp, final in temp_names:
                    try:
                        if temp.exists():
                            temp.rename(final)
                    except Exception as e:
                        print(f"  Failed to rename {temp.name}: {e}")

            if verbose:
                print(f"  Renamed {len(temp_names)} files")


def main(argv=None):
    """Main entry point."""
    args = parse_args(argv)

    if args.undo_last_trash:
        from visagen.tools.dataset_trash import undo_last_batch

        restored = undo_last_batch(args.input)
        if not restored.batch_id:
            print("No trash batch to undo.")
            return 0

        print(
            f"Undo {restored.batch_id}: restored={restored.restored}, "
            f"skipped={restored.skipped}, failed={restored.failed}"
        )
        if restored.errors:
            print(f"  First error: {restored.errors[0]}")
        return 0

    # Get image paths
    image_paths = get_image_paths(args.input)
    print(f"Found {len(image_paths)} images in {args.input}")

    if len(image_paths) == 0:
        print("No images found. Exiting.")
        return 0

    # Get sort method
    methods = get_sort_methods()
    sorter_cls = methods[args.method]

    # Initialize sorter
    if args.method in ("final", "final-fast"):
        kwargs = {"target_count": args.target}
        if args.exact_limit is not None:
            kwargs["exact_limit"] = args.exact_limit
        sorter = sorter_cls(**kwargs)
    elif args.method in (
        "hist",
        "hist-dissim",
        "absdiff",
        "absdiff-dissim",
        "id-sim",
        "id-dissim",
        "ssim",
        "ssim-dissim",
    ):
        kwargs = {}
        if args.exact_limit is not None:
            kwargs["exact_limit"] = args.exact_limit
        sorter = sorter_cls(**kwargs)
    else:
        sorter = sorter_cls()

    print(f"Sorting by: {sorter.description}")

    # Initialize processor
    from visagen.sorting.processor import ParallelSortProcessor

    if args.exec_mode == "thread":
        use_threads = True
    elif args.exec_mode == "process":
        use_threads = False
    else:
        profile = getattr(sorter, "execution_profile", "cpu_bound")
        use_threads = profile in ("io_bound", "gpu_bound")

    processor = ParallelSortProcessor(max_workers=args.jobs, use_threads=use_threads)

    # Run sorting
    result = sorter.sort(image_paths, processor)

    print("\nResults:")
    print(f"  Sorted: {len(result.sorted_images)}")
    print(f"  Trash: {len(result.trash_images)}")
    print(f"  Time: {result.elapsed_seconds:.1f}s")

    # Apply results
    if not args.dry_run:
        apply_sort_result(
            result,
            args.input,
            args.output,
            args.trash_dir,
            args.no_rename,
            args.dry_run,
            args.verbose,
        )
    else:
        print("\n[Dry run - no changes made]")

        if args.verbose:
            print("\nTop 10 sorted:")
            for i, item in enumerate(result.sorted_images[:10]):
                print(f"  {i + 1}. {item.filepath.name} (score: {item.score:.4f})")

            if len(result.trash_images) > 0:
                print(f"\nSample trash ({min(5, len(result.trash_images))}):")
                for item in result.trash_images[:5]:
                    reason = (
                        item.metadata.get("reason", "unknown")
                        if item.metadata
                        else "unknown"
                    )
                    print(f"  - {item.filepath.name}: {reason}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
