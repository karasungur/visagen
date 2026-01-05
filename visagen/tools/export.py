#!/usr/bin/env python3
"""
Visagen Export CLI - ONNX/TensorRT export tool.

Usage:
    visagen-export checkpoint.ckpt -o model.onnx
    visagen-export checkpoint.ckpt -o model.onnx --opset 17 --dynamic --validate
    visagen-export model.onnx -o model.engine --format tensorrt --precision fp16

Examples:
    # Export to ONNX with validation
    visagen-export model.ckpt -o model.onnx --validate

    # Export with dynamic axes for variable batch size
    visagen-export model.ckpt -o model.onnx --dynamic

    # Build TensorRT engine from ONNX
    visagen-export model.onnx -o model.engine --format tensorrt --precision fp16
"""

import argparse
import logging
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Export Visagen model to ONNX/TensorRT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export checkpoint to ONNX
  visagen-export model.ckpt -o model.onnx

  # Export with validation
  visagen-export model.ckpt -o model.onnx --validate

  # Export with dynamic batch/resolution
  visagen-export model.ckpt -o model.onnx --dynamic

  # Build TensorRT engine from ONNX
  visagen-export model.onnx -o model.engine --format tensorrt

  # TensorRT with FP16 precision
  visagen-export model.onnx -o model.engine --format tensorrt --precision fp16
        """,
    )

    # Required arguments
    parser.add_argument(
        "input",
        type=Path,
        help="Input file: checkpoint (.ckpt) for ONNX export, ONNX (.onnx) for TensorRT",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output file path (.onnx or .engine)",
    )

    # Format selection
    parser.add_argument(
        "--format",
        "-f",
        type=str,
        choices=["onnx", "tensorrt"],
        default=None,
        help="Export format (auto-detected from output extension if not specified)",
    )

    # ONNX options
    onnx_group = parser.add_argument_group("ONNX Export Options")
    onnx_group.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (default: 17)",
    )
    onnx_group.add_argument(
        "--dynamic",
        action="store_true",
        help="Enable dynamic axes (batch, height, width)",
    )
    onnx_group.add_argument(
        "--no-optimize",
        action="store_true",
        help="Skip ONNX optimization (onnx-simplifier)",
    )
    onnx_group.add_argument(
        "--validate",
        action="store_true",
        help="Validate exported model against PyTorch",
    )

    # TensorRT options
    trt_group = parser.add_argument_group("TensorRT Options")
    trt_group.add_argument(
        "--precision",
        type=str,
        choices=["fp32", "fp16", "int8"],
        default="fp16",
        help="TensorRT precision (default: fp16)",
    )
    trt_group.add_argument(
        "--max-batch",
        type=int,
        default=8,
        help="Maximum batch size for TensorRT (default: 8)",
    )
    trt_group.add_argument(
        "--workspace",
        type=int,
        default=1,
        help="Workspace size in GB (default: 1)",
    )

    # Input shape options
    shape_group = parser.add_argument_group("Input Shape")
    shape_group.add_argument(
        "--input-size",
        type=int,
        default=256,
        help="Input image size (default: 256)",
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


def detect_format(args: argparse.Namespace) -> str:
    """Auto-detect export format from output extension."""
    if args.format:
        return args.format

    output_ext = args.output.suffix.lower()
    if output_ext == ".onnx":
        return "onnx"
    elif output_ext in (".engine", ".trt", ".plan"):
        return "tensorrt"
    else:
        # Infer from input
        input_ext = args.input.suffix.lower()
        if input_ext == ".ckpt":
            return "onnx"
        elif input_ext == ".onnx":
            return "tensorrt"
        else:
            print(
                f"Error: Cannot auto-detect format. "
                f"Use --format to specify 'onnx' or 'tensorrt'",
                file=sys.stderr,
            )
            sys.exit(1)


def export_onnx(args: argparse.Namespace) -> int:
    """Export model to ONNX format."""
    from visagen.export import ONNXExporter, ExportConfig

    # Build config
    config = ExportConfig(
        opset_version=args.opset,
        dynamic_axes=args.dynamic,
        optimize=not args.no_optimize,
        verbose=args.verbose,
    )

    # Input shape
    input_shape = (1, 3, args.input_size, args.input_size)

    print(f"Exporting to ONNX: {args.output}")
    print(f"  Opset version: {config.opset_version}")
    print(f"  Dynamic axes: {config.dynamic_axes}")
    print(f"  Optimize: {config.optimize}")
    print(f"  Input shape: {input_shape}")
    print()

    try:
        exporter = ONNXExporter(
            checkpoint_path=args.input,
            output_path=args.output,
            config=config,
            input_shape=input_shape,
        )

        output_path = exporter.export()
        print(f"ONNX model exported to: {output_path}")

        # Validate if requested
        if args.validate:
            print()
            print("Validating export...")
            result = exporter.validate()

            if result.passed:
                print(f"✓ Validation PASSED")
                print(f"  Max difference: {result.max_diff:.6f}")
                print(f"  Mean difference: {result.mean_diff:.6f}")
            else:
                print(f"✗ Validation FAILED")
                print(f"  Max difference: {result.max_diff:.6f}")
                print(f"  Mean difference: {result.mean_diff:.6f}")
                return 1

        return 0

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


def export_tensorrt(args: argparse.Namespace) -> int:
    """Build TensorRT engine from ONNX."""
    try:
        from visagen.export import TensorRTBuilder
        from visagen.export.tensorrt_builder import BuildConfig
    except ImportError:
        print(
            "Error: TensorRT is not installed. "
            "Install TensorRT from NVIDIA to use this feature.",
            file=sys.stderr,
        )
        return 1

    # Build config
    config = BuildConfig(
        precision=args.precision,
        max_batch_size=args.max_batch,
        workspace_size=args.workspace * (1 << 30),  # Convert GB to bytes
        min_shape=(1, 3, 128, 128),
        opt_shape=(1, 3, args.input_size, args.input_size),
        max_shape=(args.max_batch, 3, 512, 512),
    )

    print(f"Building TensorRT engine: {args.output}")
    print(f"  Input ONNX: {args.input}")
    print(f"  Precision: {config.precision}")
    print(f"  Max batch size: {config.max_batch_size}")
    print(f"  Workspace: {args.workspace} GB")
    print()

    try:
        builder = TensorRTBuilder(
            onnx_path=args.input,
            engine_path=args.output,
            config=config,
        )

        # Show platform info
        if args.verbose:
            info = builder.get_platform_info()
            print("Platform info:")
            print(f"  TensorRT version: {info.get('tensorrt_version', 'unknown')}")
            print(f"  FP16 support: {info.get('has_fast_fp16', False)}")
            print(f"  INT8 support: {info.get('has_fast_int8', False)}")
            print()

        engine_path = builder.build()
        print()
        print(f"TensorRT engine saved to: {engine_path}")

        return 0

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


def main() -> int:
    """CLI entry point."""
    args = parse_args()

    # Set up logging
    log_level = (
        logging.DEBUG
        if args.verbose
        else logging.WARNING
        if args.quiet
        else logging.INFO
    )
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Validate input exists
    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        return 1

    # Detect format
    export_format = detect_format(args)

    if not args.quiet:
        print(f"Visagen Export")
        print(f"  Format: {export_format.upper()}")
        print()

    # Run export
    if export_format == "onnx":
        return export_onnx(args)
    elif export_format == "tensorrt":
        return export_tensorrt(args)
    else:
        print(f"Error: Unknown format: {export_format}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
