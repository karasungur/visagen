#!/usr/bin/env python3
"""
Visagen Merge CLI - Video face swap tool.

Usage:
    visagen-merge input.mp4 output.mp4 --checkpoint model.ckpt

Example:
    visagen-merge video.mp4 output.mp4 -c model.ckpt --color-transfer rct --blend-mode laplacian
"""

import argparse
import sys
from pathlib import Path

from visagen.merger.frame_processor import FrameProcessorConfig
from visagen.merger.merger import FaceMerger, MergerConfig


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Merge faces in video using trained Visagen model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  visagen-merge input.mp4 output.mp4 --checkpoint model.ckpt

  # With color transfer and blending options
  visagen-merge input.mp4 output.mp4 -c model.ckpt \\
      --color-transfer rct --blend-mode laplacian --mask-erode 5

  # Resume interrupted processing
  visagen-merge input.mp4 output.mp4 -c model.ckpt --resume

  # Process frame directory
  visagen-merge frames/ output/ -c model.ckpt
        """,
    )

    # Required arguments
    parser.add_argument(
        "input",
        type=Path,
        help="Input video file or directory of frames",
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Output video file or directory",
    )
    parser.add_argument(
        "--checkpoint",
        "-c",
        type=Path,
        required=True,
        help="Path to trained model checkpoint (.ckpt)",
    )

    # Detection options
    detection = parser.add_argument_group("Face Detection")
    detection.add_argument(
        "--min-confidence",
        type=float,
        default=0.5,
        help="Minimum face detection confidence (default: 0.5)",
    )
    detection.add_argument(
        "--max-faces",
        type=int,
        default=1,
        help="Maximum faces to process per frame (default: 1)",
    )
    detection.add_argument(
        "--face-type",
        type=str,
        default="whole_face",
        choices=["half", "mid_full", "full", "whole_face", "head"],
        help="Face alignment type (default: whole_face)",
    )
    detection.add_argument(
        "--output-size",
        type=int,
        default=256,
        help="Model input/output size (default: 256)",
    )

    # Color transfer options
    color = parser.add_argument_group("Color Transfer")
    color.add_argument(
        "--color-transfer",
        type=str,
        default="rct",
        choices=["rct", "lct", "sot", "none"],
        help="Color transfer mode (default: rct)",
    )

    # Blending options
    blend = parser.add_argument_group("Blending")
    blend.add_argument(
        "--blend-mode",
        type=str,
        default="laplacian",
        choices=["laplacian", "poisson", "feather"],
        help="Blending mode (default: laplacian)",
    )
    blend.add_argument(
        "--mask-erode",
        type=int,
        default=5,
        help="Mask erosion kernel size (default: 5)",
    )
    blend.add_argument(
        "--mask-blur",
        type=int,
        default=5,
        help="Mask blur kernel size (default: 5)",
    )

    # Video output options
    video = parser.add_argument_group("Video Output")
    video.add_argument(
        "--codec",
        type=str,
        default="libx264",
        help="Video codec (default: libx264)",
    )
    video.add_argument(
        "--crf",
        type=int,
        default=18,
        help="Constant rate factor 0-51, lower is better quality (default: 18)",
    )
    video.add_argument(
        "--preset",
        type=str,
        default="medium",
        choices=["ultrafast", "superfast", "veryfast", "faster", "fast",
                 "medium", "slow", "slower", "veryslow"],
        help="Encoding preset (default: medium)",
    )
    video.add_argument(
        "--no-audio",
        action="store_true",
        help="Don't copy audio track from source",
    )

    # Processing options
    proc = parser.add_argument_group("Processing")
    proc.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1)",
    )
    proc.add_argument(
        "--gpu",
        type=str,
        default=None,
        help="GPU IDs to use, comma-separated (default: auto)",
    )
    proc.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for inference: cuda, cpu (default: auto)",
    )

    # Resume options
    resume = parser.add_argument_group("Resume")
    resume.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previous interrupted run",
    )
    resume.add_argument(
        "--temp-dir",
        type=Path,
        default=None,
        help="Temporary directory for intermediate frames",
    )

    # Config file
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Load configuration from YAML file",
    )
    parser.add_argument(
        "--save-config",
        type=Path,
        default=None,
        help="Save configuration to YAML file and exit",
    )

    # Misc
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Quiet mode, minimal output",
    )

    return parser.parse_args()


def build_config(args: argparse.Namespace) -> MergerConfig:
    """Build MergerConfig from arguments."""
    # Load from YAML if provided
    if args.config and args.config.exists():
        config = MergerConfig.from_yaml(args.config)
        # Override with command line args
        config.input_path = args.input
        config.output_path = args.output
        config.checkpoint_path = args.checkpoint
        return config

    # Build frame processor config
    frame_config = FrameProcessorConfig(
        min_confidence=args.min_confidence,
        max_faces=args.max_faces,
        face_type=args.face_type,
        output_size=args.output_size,
        color_transfer_mode=args.color_transfer if args.color_transfer != "none" else None,
        blend_mode=args.blend_mode,
        mask_erode=args.mask_erode,
        mask_blur=args.mask_blur,
    )

    # Build merger config
    config = MergerConfig(
        input_path=args.input,
        output_path=args.output,
        checkpoint_path=args.checkpoint,
        frame_processor_config=frame_config,
        num_workers=args.workers,
        codec=args.codec,
        crf=args.crf,
        preset=args.preset,
        copy_audio=not args.no_audio,
        resume=args.resume,
        temp_dir=args.temp_dir,
        device=args.device,
    )

    return config


def main() -> int:
    """CLI entry point."""
    args = parse_args()

    # Set up logging
    import logging

    log_level = logging.DEBUG if args.verbose else logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Validate inputs
    if not args.input.exists():
        print(f"Error: Input not found: {args.input}", file=sys.stderr)
        return 1

    if not args.checkpoint.exists():
        print(f"Error: Checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 1

    # Build configuration
    try:
        config = build_config(args)
    except Exception as e:
        print(f"Error: Failed to build configuration: {e}", file=sys.stderr)
        return 1

    # Save config and exit if requested
    if args.save_config:
        try:
            config.to_yaml(args.save_config)
            print(f"Configuration saved to: {args.save_config}")
            return 0
        except Exception as e:
            print(f"Error: Failed to save configuration: {e}", file=sys.stderr)
            return 1

    # Progress callback
    def progress_callback(current: int, total: int) -> None:
        if not args.quiet:
            percent = current / total * 100 if total > 0 else 0
            bar_len = 30
            filled = int(bar_len * current / total) if total > 0 else 0
            bar = "█" * filled + "░" * (bar_len - filled)
            print(f"\r[{bar}] {percent:5.1f}% ({current}/{total})", end="", flush=True)

    # Run merger
    if not args.quiet:
        print(f"Visagen Merge")
        print(f"  Input: {args.input}")
        print(f"  Output: {args.output}")
        print(f"  Checkpoint: {args.checkpoint}")
        print(f"  Color transfer: {config.frame_processor_config.color_transfer_mode}")
        print(f"  Blend mode: {config.frame_processor_config.blend_mode}")
        print()

    try:
        merger = FaceMerger(config, progress_callback=progress_callback)
        stats = merger.run()

        if not args.quiet:
            print()  # New line after progress bar
            print()
            print("Processing complete!")
            print(f"  Total frames: {stats.total_frames}")
            print(f"  Processed: {stats.processed_frames}")
            if stats.skipped_frames > 0:
                print(f"  Skipped (resumed): {stats.skipped_frames}")
            if stats.failed_frames > 0:
                print(f"  Failed: {stats.failed_frames}")
            print(f"  Faces detected: {stats.faces_detected}")
            print(f"  Faces swapped: {stats.faces_swapped}")
            print(f"  Total time: {stats.total_time:.1f}s")
            print(f"  Average FPS: {stats.fps:.1f}")
            print()
            print(f"Output saved to: {config.output_path}")

        return 0

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        if config.resume:
            print("Progress saved. Run with --resume to continue.")
        return 130

    except FileNotFoundError as e:
        print(f"\nError: {e}", file=sys.stderr)
        return 1

    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
