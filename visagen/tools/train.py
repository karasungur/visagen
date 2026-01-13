"""
Training Script for Visagen.

CLI for training face swapping models with PyTorch Lightning.

Usage:
    python -m visagen.tools.train \\
        --src-dir /path/to/data_src/aligned \\
        --dst-dir /path/to/data_dst/aligned \\
        --output-dir /path/to/workspace/model \\
        --batch-size 8 \\
        --max-epochs 500

    # Or with YAML config:
    python -m visagen.tools.train --config config/train.yaml
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

from visagen.data.datamodule import FaceDataModule
from visagen.training.callbacks import (
    AutoBackupCallback,
    PreviewCallback,
    TargetStepCallback,
)
from visagen.training.dfl_module import DFLModule
from visagen.utils.config import (
    load_config_with_validation,
    print_config_summary,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Visagen face swapping model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic training
    python -m visagen.tools.train \\
        --src-dir ./workspace/data_src/aligned \\
        --dst-dir ./workspace/data_dst/aligned

    # With custom settings
    python -m visagen.tools.train \\
        --src-dir ./data_src/aligned \\
        --dst-dir ./data_dst/aligned \\
        --batch-size 16 \\
        --max-epochs 1000 \\
        --precision 16-mixed

    # Resume from checkpoint
    python -m visagen.tools.train \\
        --src-dir ./data_src/aligned \\
        --dst-dir ./data_dst/aligned \\
        --resume ./workspace/model/checkpoints/last.ckpt
        """,
    )

    # Data arguments
    parser.add_argument(
        "--src-dir",
        type=Path,
        required=True,
        help="Source faces directory (aligned)",
    )
    parser.add_argument(
        "--dst-dir",
        type=Path,
        required=True,
        help="Destination faces directory (aligned)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("workspace/model"),
        help="Output directory for checkpoints and logs",
    )

    # Training arguments
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Training batch size (default: 8)",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=500,
        help="Maximum training epochs (default: 500)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)",
    )
    parser.add_argument(
        "--optimizer-type",
        type=str,
        default="adamw",
        choices=["adamw", "adabelief"],
        help="Optimizer type (default: adamw)",
    )
    parser.add_argument(
        "--lr-dropout",
        type=float,
        default=1.0,
        help="Learning rate dropout for AdaBelief (default: 1.0)",
    )
    parser.add_argument(
        "--lr-cos-period",
        type=int,
        default=0,
        help="Cosine LR period for AdaBelief (default: 0)",
    )
    parser.add_argument(
        "--clipnorm",
        type=float,
        default=0.0,
        help="Gradient clipping norm for AdaBelief (default: 0.0)",
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
        default=0.1,
        help="Validation split ratio (default: 0.1)",
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
    parser.add_argument(
        "--lpips-weight",
        type=float,
        default=0.0,
        help="LPIPS loss weight (default: 0.0, requires lpips package)",
    )
    parser.add_argument(
        "--eyes-mouth-weight",
        type=float,
        default=0.0,
        help="Eyes/Mouth priority loss weight (default: 0.0, requires landmarks)",
    )
    parser.add_argument(
        "--gaze-weight",
        type=float,
        default=0.0,
        help="Gaze consistency loss weight (default: 0.0, requires landmarks)",
    )
    parser.add_argument(
        "--face-style-weight",
        type=float,
        default=0.0,
        help="Face style loss weight (default: 0.0, learns dst face color)",
    )
    parser.add_argument(
        "--bg-style-weight",
        type=float,
        default=0.0,
        help="Background style loss weight (default: 0.0, learns dst background)",
    )
    parser.add_argument(
        "--true-face-power",
        type=float,
        default=0.0,
        help="True face discriminator power (default: 0.0, df architecture only)",
    )
    parser.add_argument(
        "--gan-power",
        type=float,
        default=0.0,
        help="GAN loss power (default: 0.0, enables adversarial training when > 0)",
    )

    # Experimental model arguments
    parser.add_argument(
        "--model-type",
        type=str,
        default="standard",
        choices=["standard", "diffusion", "eg3d"],
        help="Model architecture type (default: standard)",
    )
    parser.add_argument(
        "--texture-weight",
        type=float,
        default=0.0,
        help="Texture consistency loss weight for diffusion model (default: 0.0)",
    )
    parser.add_argument(
        "--use-pretrained-vae",
        action="store_true",
        default=True,
        help="Use pretrained SD VAE for diffusion model (default: True)",
    )
    parser.add_argument(
        "--no-pretrained-vae",
        action="store_true",
        help="Use lite encoder instead of pretrained VAE",
    )
    parser.add_argument(
        "--eg3d-latent-dim",
        type=int,
        default=512,
        help="Latent dimension for EG3D model (default: 512)",
    )
    parser.add_argument(
        "--eg3d-plane-channels",
        type=int,
        default=32,
        help="Number of channels per tri-plane for EG3D (default: 32)",
    )
    parser.add_argument(
        "--eg3d-render-resolution",
        type=int,
        default=64,
        help="Neural render resolution for EG3D (default: 64)",
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
        default="32",
        choices=["32", "16-mixed", "bf16-mixed"],
        help="Training precision (default: 32)",
    )

    # Checkpoint arguments
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume from checkpoint path",
    )
    parser.add_argument(
        "--pretrain-from",
        type=Path,
        default=None,
        help="Load pretrained checkpoint weights (resets epoch counter to 0)",
    )
    parser.add_argument(
        "--save-top-k",
        type=int,
        default=3,
        help="Save top-k checkpoints by validation loss (default: 3)",
    )

    # Config file
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML configuration file (overrides CLI args)",
    )

    # Augmentation arguments
    parser.add_argument(
        "--no-augmentation",
        action="store_true",
        help="Disable augmentation (for debugging)",
    )
    parser.add_argument(
        "--no-warp",
        action="store_true",
        help="Disable warp augmentation",
    )
    parser.add_argument(
        "--uniform-yaw",
        action="store_true",
        help="Enable uniform yaw sampling",
    )
    parser.add_argument(
        "--masked-training",
        action="store_true",
        help="Enable masked training (blur out non-face area)",
    )

    # Preview arguments
    parser.add_argument(
        "--preview-interval",
        type=int,
        default=500,
        help="Generate preview every N steps (default: 500)",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Disable preview generation",
    )
    parser.add_argument(
        "--save-preview-history",
        action="store_true",
        help="Save historical previews (step_*.png)",
    )

    # Auto-backup arguments
    parser.add_argument(
        "--backup-interval",
        type=int,
        default=0,
        help="Create backup every N steps (0 = disabled, default: 0)",
    )
    parser.add_argument(
        "--backup-minutes",
        type=int,
        default=0,
        help="Create backup every N minutes (0 = use step interval)",
    )
    parser.add_argument(
        "--max-backups",
        type=int,
        default=5,
        help="Maximum number of rotating backups to keep (default: 5)",
    )

    # Target stopping arguments
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Stop training at this step (0 = unlimited, default: 0)",
    )
    parser.add_argument(
        "--target-loss",
        type=float,
        default=None,
        help="Stop training when loss reaches this value (default: None)",
    )

    return parser.parse_args()


def load_config(config_path: Path) -> dict:
    """Load YAML configuration file."""
    try:
        import yaml
    except ImportError:
        print(
            "Error: PyYAML required for config files. Install with: pip install pyyaml"
        )
        sys.exit(1)

    with open(config_path) as f:
        return yaml.safe_load(f)


def main() -> int:
    """Main training entry point."""
    args = parse_args()

    # Default values for CLI > Config priority checking
    cli_defaults = {
        "batch_size": 8,
        "max_epochs": 500,
        "learning_rate": 1e-4,
        "optimizer_type": "adamw",
        "lr_dropout": 1.0,
        "lr_cos_period": 0,
        "clipnorm": 0.0,
        "image_size": 256,
        "num_workers": 4,
        "val_split": 0.1,
        "dssim_weight": 10.0,
        "l1_weight": 10.0,
        "lpips_weight": 0.0,
        "eyes_mouth_weight": 0.0,
        "gaze_weight": 0.0,
        "face_style_weight": 0.0,
        "bg_style_weight": 0.0,
        "true_face_power": 0.0,
        "gan_power": 0.0,
        "texture_weight": 0.0,
        "preview_interval": 500,
        "backup_interval": 0,
        "backup_minutes": 0,
        "max_backups": 5,
        "max_steps": 0,
        "devices": 1,
        "accelerator": "auto",
        "precision": "32",
        "model_type": "standard",
        "encoder_dims": "64,128,256,512",
        "encoder_depths": "2,2,4,2",
        "save_top_k": 3,
        "eg3d_latent_dim": 512,
        "eg3d_plane_channels": 32,
        "eg3d_render_resolution": 64,
        "uniform_yaw": False,
        "masked_training": False,
    }

    # Load YAML config if provided
    if args.config is not None:
        if not args.config.exists():
            print(f"Error: Config file not found: {args.config}")
            return 1

        config, errors = load_config_with_validation(args.config)
        if errors:
            print("Config validation errors:")
            for error in errors:
                print(f"  - {error}")
            return 1

        # Merge: CLI > Config > Defaults
        # Only use config value if CLI is still at default
        for key, value in config.items():
            key_normalized = key.replace("-", "_")
            if hasattr(args, key_normalized):
                current_value = getattr(args, key_normalized)
                default_value = cli_defaults.get(key_normalized)
                # Only apply config if CLI value equals default
                if current_value == default_value:
                    setattr(args, key_normalized, value)

        # Print config summary
        print_config_summary(config)

    # Validate directories
    if not args.src_dir.exists():
        print(f"Error: Source directory not found: {args.src_dir}")
        return 1
    if not args.dst_dir.exists():
        print(f"Error: Destination directory not found: {args.dst_dir}")
        return 1

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Parse model dimensions
    encoder_dims = [int(d) for d in args.encoder_dims.split(",")]
    encoder_depths = [int(d) for d in args.encoder_depths.split(",")]
    decoder_dims = encoder_dims[::-1]  # Reverse for decoder

    # Augmentation config
    aug_config = None
    if not args.no_augmentation:
        aug_config = {
            "random_flip_prob": 0.4,
            "random_warp": not args.no_warp,
            "rotation_range": (-10, 10),
            "scale_range": (-0.05, 0.05),
            "translation_range": (-0.05, 0.05),
            "hsv_shift_amount": 0.1,
            "brightness_range": 0.1,
            "contrast_range": 0.1,
        }

    # Create DataModule
    print("Loading data from:")
    print(f"  Source: {args.src_dir}")
    print(f"  Destination: {args.dst_dir}")

    datamodule = FaceDataModule(
        src_dir=args.src_dir,
        dst_dir=args.dst_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target_size=args.image_size,
        val_split=args.val_split,
        augmentation_config=aug_config,
        uniform_yaw=args.uniform_yaw,
    )

    # Create Model
    print("\nModel configuration:")
    print(f"  Model type: {args.model_type}")
    print(f"  Image size: {args.image_size}")
    print(f"  Encoder dims: {encoder_dims}")
    print(f"  Encoder depths: {encoder_depths}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  DSSIM weight: {args.dssim_weight}")
    print(f"  L1 weight: {args.l1_weight}")
    print(f"  LPIPS weight: {args.lpips_weight}")
    print(f"  Eyes/Mouth weight: {args.eyes_mouth_weight}")
    print(f"  Gaze weight: {args.gaze_weight}")
    print(f"  Face style weight: {args.face_style_weight}")
    print(f"  Background style weight: {args.bg_style_weight}")
    print(f"  True face power: {args.true_face_power}")
    print(f"  GAN power: {args.gan_power}")
    print(f"  Uniform yaw: {args.uniform_yaw}")
    print(f"  Masked training: {args.masked_training}")
    if args.model_type == "diffusion":
        print(f"  Texture weight: {args.texture_weight}")
        print(f"  Pretrained VAE: {not args.no_pretrained_vae}")
    elif args.model_type == "eg3d":
        print(f"  EG3D latent dim: {args.eg3d_latent_dim}")
        print(f"  EG3D plane channels: {args.eg3d_plane_channels}")
        print(f"  EG3D render resolution: {args.eg3d_render_resolution}")

    if args.pretrain_from is not None:
        # Load pretrained weights for fine-tuning
        if not args.pretrain_from.exists():
            print(f"Error: Pretrained checkpoint not found: {args.pretrain_from}")
            return 1

        print(f"  Pretrained from: {args.pretrain_from}")

        from visagen.training.pretrain_module import PretrainModule

        model = PretrainModule.load_for_finetune(
            checkpoint_path=args.pretrain_from,
            strict=False,
            # Override with CLI settings
            image_size=args.image_size,
            encoder_dims=encoder_dims,
            encoder_depths=encoder_depths,
            decoder_dims=decoder_dims,
            learning_rate=args.learning_rate,
            dssim_weight=args.dssim_weight,
            l1_weight=args.l1_weight,
            lpips_weight=args.lpips_weight,
            # Optimizer config
            optimizer_type=args.optimizer_type,
            lr_dropout=args.lr_dropout,
            lr_cos_period=args.lr_cos_period,
            clipnorm=args.clipnorm,
        )
        print("  Mode: Fine-tuning from pretrained weights (epoch resets to 0)")
    else:
        model = DFLModule(
            image_size=args.image_size,
            encoder_dims=encoder_dims,
            encoder_depths=encoder_depths,
            decoder_dims=decoder_dims,
            learning_rate=args.learning_rate,
            dssim_weight=args.dssim_weight,
            l1_weight=args.l1_weight,
            lpips_weight=args.lpips_weight,
            eyes_mouth_weight=args.eyes_mouth_weight,
            gaze_weight=args.gaze_weight,
            face_style_weight=args.face_style_weight,
            bg_style_weight=args.bg_style_weight,
            true_face_power=args.true_face_power,
            gan_power=args.gan_power,
            # Optimizer config
            optimizer_type=args.optimizer_type,
            lr_dropout=args.lr_dropout,
            lr_cos_period=args.lr_cos_period,
            clipnorm=args.clipnorm,
            # Experimental model parameters
            model_type=args.model_type,
            diffusion_texture_weight=args.texture_weight,
            use_pretrained_vae=not args.no_pretrained_vae,
            eg3d_latent_dim=args.eg3d_latent_dim,
            eg3d_plane_channels=args.eg3d_plane_channels,
            eg3d_render_resolution=args.eg3d_render_resolution,
            # Masked training
            blur_out_mask=args.masked_training,
        )

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=args.output_dir / "checkpoints",
            filename="visagen-{epoch:04d}-{val_total:.4f}",
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

    # Add preview callback
    if not args.no_preview:
        preview_callback = PreviewCallback(
            preview_dir=args.output_dir / "previews",
            interval=args.preview_interval,
            num_samples=4,
            save_history=args.save_preview_history,
        )
        callbacks.append(preview_callback)

    # Add auto-backup callback
    if args.backup_interval > 0 or args.backup_minutes > 0:
        backup_callback = AutoBackupCallback(
            backup_dir=args.output_dir / "backups",
            interval_steps=args.backup_interval if args.backup_interval > 0 else 0,
            interval_minutes=args.backup_minutes if args.backup_minutes > 0 else None,
            max_backups=args.max_backups,
        )
        callbacks.append(backup_callback)

    # Add target step/loss callback
    if args.max_steps > 0 or args.target_loss is not None:
        target_callback = TargetStepCallback(
            target_step=args.max_steps,
            target_loss=args.target_loss,
        )
        callbacks.append(target_callback)

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
    if not args.no_preview:
        print(f"  Preview interval: {args.preview_interval} steps")
    if args.backup_interval > 0:
        print(f"  Backup interval: {args.backup_interval} steps")
    if args.backup_minutes > 0:
        print(f"  Backup interval: {args.backup_minutes} minutes")
    if args.max_steps > 0:
        print(f"  Target steps: {args.max_steps}")
    if args.target_loss is not None:
        print(f"  Target loss: {args.target_loss}")

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

    # Setup data to get sample counts
    datamodule.setup()
    print("\nDataset:")
    print(f"  Training samples: {datamodule.num_train_samples}")
    print(f"  Validation samples: {datamodule.num_val_samples}")

    # Train
    print("\nStarting training...")
    if args.resume is not None:
        print(f"Resuming from: {args.resume}")

        # Show checkpoint hyperparameters
        try:
            import torch

            ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
            if "hyper_parameters" in ckpt:
                hparams = ckpt["hyper_parameters"]
                print("\nCheckpoint hyperparameters:")
                for key, value in sorted(hparams.items()):
                    print(f"  {key}: {value}")
                print()
        except Exception as e:
            print(f"Note: Could not read checkpoint hyperparameters: {e}")

        trainer.fit(model, datamodule, ckpt_path=str(args.resume))
    else:
        trainer.fit(model, datamodule)

    print("\nTraining complete!")
    print(f"Checkpoints saved to: {args.output_dir / 'checkpoints'}")
    print(f"Logs saved to: {args.output_dir / 'logs'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
