"""
Pretrain Script for Visagen.

CLI for pretraining face models on large datasets (FFHQ, CelebA, etc.)
with PyTorch Lightning.

Usage:
    visagen-pretrain --data-dir /path/to/ffhq --output-dir ./pretrain_model

    # With custom settings:
    visagen-pretrain \\
        --data-dir /path/to/celeba \\
        --output-dir ./pretrain_model \\
        --batch-size 32 \\
        --max-epochs 100 \\
        --precision 16-mixed

    # Resume from checkpoint:
    visagen-pretrain \\
        --data-dir /path/to/ffhq \\
        --output-dir ./pretrain_model \\
        --resume ./pretrain_model/checkpoints/last.ckpt
"""

import argparse
import sys
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger

from visagen.data.pretrain_datamodule import PretrainDataModule
from visagen.training.pretrain_module import PretrainModule


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Pretrain Visagen model on large face dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic pretraining on FFHQ
    visagen-pretrain --data-dir /data/ffhq/images1024x1024 --output-dir ./pretrain

    # With custom settings
    visagen-pretrain \\
        --data-dir /data/celeba \\
        --output-dir ./pretrain \\
        --batch-size 32 \\
        --max-epochs 100 \\
        --precision 16-mixed

    # Resume pretraining
    visagen-pretrain \\
        --data-dir /data/ffhq \\
        --output-dir ./pretrain \\
        --resume ./pretrain/checkpoints/last.ckpt

    # Self-pretrain on your own aligned faces
    visagen-pretrain \\
        --data-dir ./workspace/data_src/aligned \\
        --output-dir ./pretrain \\
        --no-recursive

After pretraining, use the checkpoint for fine-tuning:
    visagen-train \\
        --src-dir ./data_src/aligned \\
        --dst-dir ./data_dst/aligned \\
        --pretrain-from ./pretrain/checkpoints/last.ckpt
        """,
    )

    # Data arguments
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Path to pretrain dataset (FFHQ, CelebA, or aligned faces)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("workspace/pretrain"),
        help="Output directory for checkpoints and logs",
    )

    # Training arguments
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Training batch size (default: 16)",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=100,
        help="Maximum training epochs (default: 100)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="Training image size (default: 256)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader workers (default: 4)",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.05,
        help="Validation split ratio (default: 0.05)",
    )

    # Dataset format
    parser.add_argument(
        "--recursive",
        action="store_true",
        default=True,
        help="Scan subdirectories for images (default: True)",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Disable recursive scanning (for flat directories)",
    )

    # Model arguments
    parser.add_argument(
        "--encoder-dims",
        type=str,
        default="64,128,256,512",
        help="Encoder channel dimensions (default: 64,128,256,512)",
    )
    parser.add_argument(
        "--encoder-depths",
        type=str,
        default="2,2,4,2",
        help="Encoder block depths (default: 2,2,4,2)",
    )
    parser.add_argument(
        "--dssim-weight",
        type=float,
        default=10.0,
        help="DSSIM loss weight (default: 10.0)",
    )
    parser.add_argument(
        "--l1-weight",
        type=float,
        default=10.0,
        help="L1 loss weight (default: 10.0)",
    )

    # Hardware arguments
    parser.add_argument(
        "--devices",
        type=int,
        default=1,
        help="Number of GPUs/devices (default: 1)",
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        default="auto",
        choices=["auto", "cpu", "gpu", "cuda", "mps"],
        help="Accelerator type (default: auto)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="16-mixed",
        choices=["32", "16-mixed", "bf16-mixed"],
        help="Training precision (default: 16-mixed)",
    )

    # Checkpoint arguments
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume from checkpoint path",
    )
    parser.add_argument(
        "--save-top-k",
        type=int,
        default=3,
        help="Save top-k checkpoints by validation loss (default: 3)",
    )

    # Augmentation arguments
    parser.add_argument(
        "--enable-warp",
        action="store_true",
        help="Enable random warp augmentation (disabled by default for pretrain)",
    )

    # Logging
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    # Handle recursive flag
    if args.no_recursive:
        args.recursive = False

    # Validate input
    if not args.data_dir.exists():
        parser.error(f"Data directory does not exist: {args.data_dir}")

    return args


def main() -> int:
    """Main pretrain entry point."""
    args = parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Parse model dimensions
    encoder_dims = [int(d) for d in args.encoder_dims.split(",")]
    encoder_depths = [int(d) for d in args.encoder_depths.split(",")]
    decoder_dims = encoder_dims[::-1]  # Reverse for decoder

    # Augmentation config
    aug_config = {
        "random_flip_prob": 0.5,
        "random_warp": args.enable_warp,  # Disabled by default for pretrain
        "rotation_range": (-10, 10),
        "scale_range": (-0.05, 0.05),
        "translation_range": (-0.05, 0.05),
        "hsv_shift_amount": 0.1,
        "brightness_range": 0.1,
        "contrast_range": 0.1,
    }

    # Create DataModule
    print("=" * 60)
    print("VISAGEN PRETRAIN")
    print("=" * 60)
    print("\nLoading pretrain dataset from:")
    print(f"  {args.data_dir}")

    datamodule = PretrainDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target_size=args.image_size,
        val_split=args.val_split,
        recursive=args.recursive,
        augmentation_config=aug_config,
    )

    # Setup data to get sample counts
    datamodule.setup()
    print(f"  Total images: {datamodule.num_total_images:,}")
    print(f"  Training samples: {datamodule.num_train_samples:,}")
    print(f"  Validation samples: {datamodule.num_val_samples:,}")

    # Create Model
    print("\nModel configuration:")
    print(f"  Image size: {args.image_size}")
    print(f"  Encoder dims: {encoder_dims}")
    print(f"  Encoder depths: {encoder_depths}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  DSSIM weight: {args.dssim_weight}")
    print(f"  L1 weight: {args.l1_weight}")
    print("  Mode: Self-reconstruction (pretrain)")
    print(f"  Random warp: {'enabled' if args.enable_warp else 'disabled'}")

    model = PretrainModule(
        image_size=args.image_size,
        encoder_dims=encoder_dims,
        encoder_depths=encoder_depths,
        decoder_dims=decoder_dims,
        learning_rate=args.learning_rate,
        dssim_weight=args.dssim_weight,
        l1_weight=args.l1_weight,
        lpips_weight=0.0,  # Disabled for pretrain
        id_weight=0.0,  # Disabled for pretrain
    )

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=args.output_dir / "checkpoints",
            filename="pretrain-{epoch:04d}-{val_total:.4f}",
            monitor="val_total",
            mode="min",
            save_top_k=args.save_top_k,
            save_last=True,
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    # Add progress bar if rich is available
    try:
        callbacks.append(RichProgressBar())
    except Exception:
        pass  # Rich not available

    # Logger
    logger = TensorBoardLogger(
        save_dir=args.output_dir,
        name="logs",
        default_hp_metric=False,
    )

    # Trainer
    print("\nTraining configuration:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Max epochs: {args.max_epochs}")
    print(f"  Precision: {args.precision}")
    print(f"  Accelerator: {args.accelerator}")
    print(f"  Devices: {args.devices}")
    print(f"  Output: {args.output_dir}")

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=50,
        enable_progress_bar=True,
        gradient_clip_val=1.0,
    )

    # Train
    print("\n" + "=" * 60)
    print("Starting pretraining...")
    print("=" * 60 + "\n")

    if args.resume is not None:
        print(f"Resuming from: {args.resume}")
        trainer.fit(model, datamodule, ckpt_path=str(args.resume))
    else:
        trainer.fit(model, datamodule)

    print("\n" + "=" * 60)
    print("Pretraining complete!")
    print("=" * 60)
    print(f"\nPretrained model saved to: {args.output_dir / 'checkpoints'}")
    print(f"TensorBoard logs: {args.output_dir / 'logs'}")
    print("\nTo fine-tune on specific faces:")
    print("  visagen-train \\")
    print("    --src-dir ./data_src/aligned \\")
    print("    --dst-dir ./data_dst/aligned \\")
    print(f"    --pretrain-from {args.output_dir / 'checkpoints' / 'last.ckpt'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
