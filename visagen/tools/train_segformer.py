#!/usr/bin/env python3
"""
CLI tool for SegFormer LoRA training.

Train LoRA adapters for face segmentation improvement.

Usage:
    python -m visagen.tools.train_segformer ./training_samples ./output

    # With custom parameters
    python -m visagen.tools.train_segformer ./samples ./output \\
        --epochs 100 --rank 16 --lr 0.0001
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def progress_callback(epoch: int, max_epochs: int, loss: float) -> None:
    """Print training progress."""
    bar_width = 30
    progress = epoch / max_epochs
    filled = int(bar_width * progress)
    bar = "=" * filled + "-" * (bar_width - filled)

    print(
        f"\rEpoch {epoch:3d}/{max_epochs} [{bar}] Loss: {loss:.4f}", end="", flush=True
    )
    if epoch == max_epochs:
        print()  # New line at completion


def main(args: list[str] | None = None) -> int:
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Train SegFormer LoRA adapters for face segmentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic training
    python -m visagen.tools.train_segformer ./training_samples ./lora_output

    # Custom training parameters
    python -m visagen.tools.train_segformer ./samples ./output \\
        --epochs 100 --rank 16 --lr 0.0001 --batch-size 8

    # Quick training with 10 samples
    python -m visagen.tools.train_segformer ./quick_samples ./output \\
        --epochs 30 --rank 4

Directory structure for samples_dir:
    samples_dir/
    ├── images/
    │   ├── sample_0001.jpg
    │   └── sample_0002.jpg
    └── masks/
        ├── sample_0001.png
        └── sample_0002.png
        """,
    )

    parser.add_argument(
        "samples_dir",
        type=Path,
        help="Directory containing training samples (images/ and masks/ subdirs)",
    )

    parser.add_argument(
        "output_dir",
        type=Path,
        help="Output directory for LoRA weights",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50)",
    )

    parser.add_argument(
        "--rank",
        type=int,
        default=8,
        help="LoRA rank (default: 8)",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=16.0,
        help="LoRA alpha scaling (default: 16.0)",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate (default: 0.0001)",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size (default: 4)",
    )

    parser.add_argument(
        "--ce-weight",
        type=float,
        default=1.0,
        help="Cross-entropy loss weight (default: 1.0)",
    )

    parser.add_argument(
        "--dice-weight",
        type=float,
        default=1.0,
        help="Dice loss weight (default: 1.0)",
    )

    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Validation data fraction (default: 0.1)",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="DataLoader workers (default: 0)",
    )

    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Path to base SegFormer model (default: auto-detect)",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )

    parsed_args = parser.parse_args(args)

    # Validate inputs
    if not parsed_args.samples_dir.exists():
        logger.error(f"Samples directory not found: {parsed_args.samples_dir}")
        return 1

    images_dir = parsed_args.samples_dir / "images"
    masks_dir = parsed_args.samples_dir / "masks"

    if not images_dir.exists():
        logger.error(f"Images subdirectory not found: {images_dir}")
        return 1

    if not masks_dir.exists():
        logger.error(f"Masks subdirectory not found: {masks_dir}")
        return 1

    # Count samples
    image_count = len(list(images_dir.glob("*.jpg"))) + len(
        list(images_dir.glob("*.png"))
    )
    if image_count == 0:
        logger.error("No image files found in images directory")
        return 1

    logger.info(f"Found {image_count} training samples")

    # Import training module (delayed to avoid slow import on --help)
    try:
        from visagen.training.segformer_finetune import (
            SegFormerFinetuneConfig,
            SegFormerTrainer,
        )
    except ImportError as e:
        logger.error(f"Failed to import training module: {e}")
        logger.error("Make sure PyTorch Lightning and transformers are installed")
        return 1

    # Create configuration
    config = SegFormerFinetuneConfig(
        learning_rate=parsed_args.lr,
        max_epochs=parsed_args.epochs,
        batch_size=parsed_args.batch_size,
        lora_rank=parsed_args.rank,
        lora_alpha=parsed_args.alpha,
        ce_weight=parsed_args.ce_weight,
        dice_weight=parsed_args.dice_weight,
        val_split=parsed_args.val_split,
        num_workers=parsed_args.workers,
    )

    logger.info("Training configuration:")
    logger.info(f"  Epochs: {config.max_epochs}")
    logger.info(f"  LoRA rank: {config.lora_rank}")
    logger.info(f"  LoRA alpha: {config.lora_alpha}")
    logger.info(f"  Learning rate: {config.learning_rate}")
    logger.info(f"  Batch size: {config.batch_size}")

    # Create trainer
    trainer = SegFormerTrainer(
        config=config,
        model_path=parsed_args.model_path,
    )

    # Run training
    logger.info("Starting training...")

    callback = None if parsed_args.quiet else progress_callback

    try:
        output_path = trainer.train(
            samples_dir=parsed_args.samples_dir,
            output_dir=parsed_args.output_dir,
            progress_callback=callback,
        )

        logger.info(f"Training complete! LoRA weights saved to: {output_path}")
        logger.info(
            f"Weights file size: {output_path.stat().st_size / 1024 / 1024:.1f} MB"
        )

        return 0

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return 130

    except Exception as e:
        logger.error(f"Training failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
