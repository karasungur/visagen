"""
Hyperparameter Tuning CLI for Visagen.

CLI for running Optuna-based hyperparameter optimization.

Usage:
    visagen-tune \\
        --src-dir /path/to/data_src/aligned \\
        --dst-dir /path/to/data_dst/aligned \\
        --output-dir /path/to/workspace/optuna \\
        --n-trials 20 \\
        --epochs-per-trial 50

    # Resume previous study:
    visagen-tune \\
        --src-dir ./data_src/aligned \\
        --dst-dir ./data_dst/aligned \\
        --output-dir ./workspace/optuna \\
        --study-name my_study \\
        --n-trials 10  # Additional trials
"""

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Hyperparameter tuning for Visagen",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic tuning
    visagen-tune \\
        --src-dir ./workspace/data_src/aligned \\
        --dst-dir ./workspace/data_dst/aligned

    # With custom settings
    visagen-tune \\
        --src-dir ./data_src/aligned \\
        --dst-dir ./data_dst/aligned \\
        --n-trials 50 \\
        --epochs-per-trial 100 \\
        --enable-gan-search

    # Resume from previous study
    visagen-tune \\
        --src-dir ./data_src/aligned \\
        --dst-dir ./data_dst/aligned \\
        --study-name visagen_hpo \\
        --n-trials 10
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
        default=Path("workspace/optuna"),
        help="Output directory for study database and configs (default: workspace/optuna)",
    )

    # Tuning arguments
    parser.add_argument(
        "--study-name",
        type=str,
        default="visagen_hpo",
        help="Optuna study name (default: visagen_hpo)",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=20,
        help="Number of trials to run (default: 20)",
    )
    parser.add_argument(
        "--epochs-per-trial",
        type=int,
        default=50,
        help="Maximum epochs per trial (default: 50)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Timeout in seconds (default: None)",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel trials (default: 1, sequential)",
    )

    # Search space customization
    parser.add_argument(
        "--lr-min",
        type=float,
        default=1e-5,
        help="Minimum learning rate (default: 1e-5)",
    )
    parser.add_argument(
        "--lr-max",
        type=float,
        default=1e-3,
        help="Maximum learning rate (default: 1e-3)",
    )
    parser.add_argument(
        "--enable-gan-search",
        action="store_true",
        help="Enable GAN power hyperparameter search",
    )
    parser.add_argument(
        "--enable-lpips-search",
        action="store_true",
        help="Enable LPIPS weight hyperparameter search",
    )

    # Data arguments
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Training batch size (default: 8)",
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

    # Hardware arguments
    parser.add_argument(
        "--accelerator",
        type=str,
        default="auto",
        choices=["auto", "cpu", "gpu", "cuda", "mps"],
        help="Accelerator type (default: auto)",
    )
    parser.add_argument(
        "--devices",
        type=int,
        default=1,
        help="Number of devices (default: 1)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="32",
        choices=["32", "16-mixed", "bf16-mixed"],
        help="Training precision (default: 32)",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point for hyperparameter tuning."""
    args = parse_args()

    # Check optuna is installed
    try:
        import optuna
    except ImportError:
        print("Error: Optuna is required for tuning.")
        print("Install with: pip install 'visagen[tuning]'")
        return 1

    # Validate directories
    if not args.src_dir.exists():
        print(f"Error: Source directory not found: {args.src_dir}")
        return 1
    if not args.dst_dir.exists():
        print(f"Error: Destination directory not found: {args.dst_dir}")
        return 1

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Import after validation to speed up CLI response
    from visagen.data.datamodule import FaceDataModule
    from visagen.tuning.optuna_tuner import OptunaTuner, TuningConfig

    print("=" * 60)
    print("VISAGEN HYPERPARAMETER TUNING")
    print("=" * 60)
    print("\nData:")
    print(f"  Source: {args.src_dir}")
    print(f"  Destination: {args.dst_dir}")
    print("\nTuning settings:")
    print(f"  Study name: {args.study_name}")
    print(f"  Trials: {args.n_trials}")
    print(f"  Epochs/trial: {args.epochs_per_trial}")
    print(f"  Learning rate range: [{args.lr_min}, {args.lr_max}]")
    print(f"  GAN search: {'enabled' if args.enable_gan_search else 'disabled'}")
    print(f"  LPIPS search: {'enabled' if args.enable_lpips_search else 'disabled'}")
    print("\nHardware:")
    print(f"  Accelerator: {args.accelerator}")
    print(f"  Devices: {args.devices}")
    print(f"  Precision: {args.precision}")
    print(f"  Batch size: {args.batch_size}")
    print(f"\nOutput: {args.output_dir}")
    print("=" * 60)

    # Create DataModule
    print("\nLoading data...")
    datamodule = FaceDataModule(
        src_dir=args.src_dir,
        dst_dir=args.dst_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target_size=args.image_size,
        val_split=args.val_split,
    )

    # Setup data to validate
    datamodule.setup()
    print(f"  Training samples: {datamodule.num_train_samples}")
    print(f"  Validation samples: {datamodule.num_val_samples}")

    # Create tuning config
    config = TuningConfig(
        learning_rate_range=(args.lr_min, args.lr_max),
        gan_power_range=(0.0, 0.5) if args.enable_gan_search else (0.0, 0.0),
        lpips_weight_range=(0.0, 5.0) if args.enable_lpips_search else (0.0, 0.0),
    )

    # Create tuner
    storage_path = args.output_dir / f"{args.study_name}.db"
    tuner = OptunaTuner(
        study_name=args.study_name,
        storage_path=storage_path,
    )

    print(f"\nStudy database: {storage_path}")

    # Check for existing trials
    existing_trials = len(tuner.study.trials)
    if existing_trials > 0:
        print(f"Resuming study with {existing_trials} existing trials")
        print(f"Current best value: {tuner.study.best_value:.6f}")

    # Run optimization
    print(f"\nStarting optimization ({args.n_trials} trials)...")
    print("-" * 60)

    try:
        tuner.optimize(
            datamodule=datamodule,
            config=config,
            n_trials=args.n_trials,
            max_epochs=args.epochs_per_trial,
            timeout=args.timeout,
            n_jobs=args.n_jobs,
            accelerator=args.accelerator,
            devices=args.devices,
            precision=args.precision,
            show_progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n\nOptimization interrupted by user.")

    # Print summary
    tuner.print_summary()

    # Export best config
    config_path = args.output_dir / "best_config.yaml"
    tuner.export_best_config(config_path)
    print(f"Best configuration saved to: {config_path}")

    # Save trials dataframe
    try:
        df = tuner.get_trials_dataframe()
        csv_path = args.output_dir / "trials.csv"
        df.to_csv(csv_path, index=False)
        print(f"Trials history saved to: {csv_path}")
    except Exception:
        pass  # pandas not available

    print("\nTuning complete!")
    print("\nTo train with best params, use:")
    print(f"  visagen-train --config {config_path} \\")
    print(f"    --src-dir {args.src_dir} \\")
    print(f"    --dst-dir {args.dst_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
