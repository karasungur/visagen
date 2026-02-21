"""
Optuna Hyperparameter Tuner for Visagen.

Provides automated hyperparameter optimization using Optuna with
PyTorch Lightning integration.

Features:
    - TuningConfig: Dataclass for defining search spaces
    - OptunaTuner: Main tuner class with study management
    - PyTorchLightningPruningCallback integration
    - SQLite study persistence for resume support
    - YAML config export for best parameters
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import optuna
import pytorch_lightning as pl
import yaml  # type: ignore[import-untyped]
from optuna_integration import PyTorchLightningPruningCallback
from pytorch_lightning.callbacks import EarlyStopping

logger = logging.getLogger(__name__)


@dataclass
class TuningConfig:
    """
    Configuration for hyperparameter search space.

    Defines the ranges and options for hyperparameters to optimize.
    All range tuples are (min, max) inclusive.

    Args:
        learning_rate_range: Learning rate range (log scale). Default: (1e-5, 1e-3).
        weight_decay_range: Weight decay range. Default: (0.001, 0.1).
        dssim_weight_range: DSSIM loss weight range. Default: (1.0, 20.0).
        l1_weight_range: L1 loss weight range. Default: (1.0, 20.0).
        drop_path_rate_range: Stochastic depth rate range. Default: (0.0, 0.3).
        gan_power_range: GAN loss weight range. Default: (0.0, 0.0) (disabled).
        batch_size_options: Batch size options to try. Default: None (use fixed).
        encoder_dims_options: Encoder dimension options. Default: None.

    Example:
        >>> config = TuningConfig(
        ...     learning_rate_range=(1e-5, 1e-3),
        ...     gan_power_range=(0.0, 0.5),  # Enable GAN search
        ... )
    """

    learning_rate_range: tuple[float, float] = (1e-5, 1e-3)
    weight_decay_range: tuple[float, float] = (0.001, 0.1)
    dssim_weight_range: tuple[float, float] = (1.0, 20.0)
    l1_weight_range: tuple[float, float] = (1.0, 20.0)
    drop_path_rate_range: tuple[float, float] = (0.0, 0.3)
    lpips_weight_range: tuple[float, float] = (0.0, 5.0)
    gan_power_range: tuple[float, float] = (0.0, 0.0)  # Disabled by default

    # Categorical options (None = don't tune)
    batch_size_options: list[int] | None = None
    encoder_dims_options: list[list[int]] | None = None

    def is_gan_enabled(self) -> bool:
        """Check if GAN hyperparameter search is enabled."""
        return self.gan_power_range[1] > 0


class OptunaTuner:
    """
    Optuna-based hyperparameter tuner for Visagen.

    Manages Optuna study lifecycle, creates objective functions,
    and provides utilities for result analysis.

    Args:
        study_name: Name of the Optuna study. Default: "visagen_hpo".
        storage_path: Path to SQLite database for persistence.
            If None, uses in-memory storage. Default: None.
        direction: Optimization direction. Default: "minimize".
        pruner: Optuna pruner for early stopping.
            Default: MedianPruner(n_startup_trials=5, n_warmup_steps=10).
        sampler: Optuna sampler for parameter selection.
            Default: TPESampler().

    Example:
        >>> tuner = OptunaTuner(
        ...     study_name="visagen_hpo",
        ...     storage_path=Path("./optuna/study.db"),
        ... )
        >>> study = tuner.optimize(
        ...     datamodule=datamodule,
        ...     config=TuningConfig(),
        ...     n_trials=20,
        ...     max_epochs=50,
        ... )
        >>> print(f"Best params: {study.best_params}")
    """

    def __init__(
        self,
        study_name: str = "visagen_hpo",
        storage_path: Path | None = None,
        direction: str = "minimize",
        pruner: optuna.pruners.BasePruner | None = None,
        sampler: optuna.samplers.BaseSampler | None = None,
    ) -> None:
        self.study_name = study_name
        self.storage_path = storage_path
        self.direction = direction

        # Default pruner: MedianPruner
        if pruner is None:
            pruner = optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=10,
                interval_steps=1,
            )
        self.pruner = pruner

        # Default sampler: TPE
        if sampler is None:
            sampler = optuna.samplers.TPESampler(seed=42)
        self.sampler = sampler

        # Create storage URL if path provided
        if storage_path is not None:
            storage_path = Path(storage_path)
            storage_path.parent.mkdir(parents=True, exist_ok=True)
            self.storage: str | None = f"sqlite:///{storage_path}"
        else:
            self.storage = None

        # Study will be created on first optimize call
        self._study: optuna.Study | None = None

    @property
    def study(self) -> optuna.Study:
        """Get or create the Optuna study."""
        if self._study is None:
            self._study = optuna.create_study(
                study_name=self.study_name,
                storage=self.storage,
                direction=self.direction,
                pruner=self.pruner,
                sampler=self.sampler,
                load_if_exists=True,
            )
        return self._study

    def _suggest_hyperparameters(
        self,
        trial: optuna.Trial,
        config: TuningConfig,
    ) -> dict[str, Any]:
        """
        Sample hyperparameters for a trial.

        Args:
            trial: Optuna trial object.
            config: Tuning configuration with search spaces.

        Returns:
            Dictionary of suggested hyperparameters.
        """
        params: dict[str, Any] = {}

        # Learning rate (log scale)
        params["learning_rate"] = trial.suggest_float(
            "learning_rate",
            config.learning_rate_range[0],
            config.learning_rate_range[1],
            log=True,
        )

        # Weight decay
        params["weight_decay"] = trial.suggest_float(
            "weight_decay",
            config.weight_decay_range[0],
            config.weight_decay_range[1],
        )

        # DSSIM weight
        params["dssim_weight"] = trial.suggest_float(
            "dssim_weight",
            config.dssim_weight_range[0],
            config.dssim_weight_range[1],
        )

        # L1 weight
        params["l1_weight"] = trial.suggest_float(
            "l1_weight",
            config.l1_weight_range[0],
            config.l1_weight_range[1],
        )

        # Drop path rate
        params["drop_path_rate"] = trial.suggest_float(
            "drop_path_rate",
            config.drop_path_rate_range[0],
            config.drop_path_rate_range[1],
        )

        # LPIPS weight (often 0)
        if config.lpips_weight_range[1] > 0:
            params["lpips_weight"] = trial.suggest_float(
                "lpips_weight",
                config.lpips_weight_range[0],
                config.lpips_weight_range[1],
            )

        # GAN power (optional)
        if config.is_gan_enabled():
            params["gan_power"] = trial.suggest_float(
                "gan_power",
                config.gan_power_range[0],
                config.gan_power_range[1],
            )

        # Batch size (categorical)
        if config.batch_size_options is not None:
            params["batch_size"] = trial.suggest_categorical(
                "batch_size",
                config.batch_size_options,
            )

        # Encoder dims (categorical)
        if config.encoder_dims_options is not None:
            dims_idx = trial.suggest_categorical(
                "encoder_dims_idx",
                list(range(len(config.encoder_dims_options))),
            )
            params["encoder_dims"] = config.encoder_dims_options[dims_idx]

        return params

    def create_objective(
        self,
        datamodule: "pl.LightningDataModule",
        config: TuningConfig,
        max_epochs: int = 50,
        accelerator: str = "auto",
        devices: int = 1,
        precision: str = "32",
        gradient_clip_val: float = 1.0,
        monitor_metric: str = "val_total",
    ) -> Callable[[optuna.Trial], float]:
        """
        Create an Optuna objective function.

        Args:
            datamodule: PyTorch Lightning DataModule for training.
            config: Tuning configuration with search spaces.
            max_epochs: Maximum epochs per trial. Default: 50.
            accelerator: Training accelerator. Default: "auto".
            devices: Number of devices. Default: 1.
            precision: Training precision. Default: "32".
            gradient_clip_val: Gradient clipping value. Default: 1.0.
            monitor_metric: Metric to optimize. Default: "val_total".

        Returns:
            Objective function for Optuna optimization.
        """
        from visagen.training.dfl_module import DFLModule

        def objective(trial: optuna.Trial) -> float:
            # Suggest hyperparameters
            params = self._suggest_hyperparameters(trial, config)

            logger.info(f"Trial {trial.number}: {params}")

            # Handle batch size if tuned
            if "batch_size" in params:
                batch_size = params.pop("batch_size")
                cast(Any, datamodule).batch_size = batch_size

            # Create model with suggested params
            model = DFLModule(**params)

            # Pruning callback
            pruning_callback = PyTorchLightningPruningCallback(
                trial,
                monitor=monitor_metric,
            )

            # Early stopping callback
            early_stop = EarlyStopping(
                monitor=monitor_metric,
                patience=10,
                mode="min",
            )

            # Trainer
            trainer = pl.Trainer(
                max_epochs=max_epochs,
                accelerator=accelerator,
                devices=devices,
                precision=cast(Any, precision),
                callbacks=[pruning_callback, early_stop],
                enable_progress_bar=False,
                enable_model_summary=False,
                logger=False,
                gradient_clip_val=gradient_clip_val,
            )

            # Train
            try:
                trainer.fit(model, datamodule)
            except optuna.TrialPruned:
                raise

            # Return best validation loss
            if trainer.callback_metrics.get(monitor_metric) is not None:
                return trainer.callback_metrics[monitor_metric].item()
            else:
                # Fallback: return a high value if metric not found
                return float("inf")

        return objective

    def optimize(
        self,
        datamodule: "pl.LightningDataModule",
        config: TuningConfig,
        n_trials: int = 20,
        max_epochs: int = 50,
        timeout: int | None = None,
        n_jobs: int = 1,
        accelerator: str = "auto",
        devices: int = 1,
        precision: str = "32",
        show_progress_bar: bool = True,
    ) -> optuna.Study:
        """
        Run hyperparameter optimization.

        Args:
            datamodule: PyTorch Lightning DataModule.
            config: Tuning configuration.
            n_trials: Number of trials to run. Default: 20.
            max_epochs: Maximum epochs per trial. Default: 50.
            timeout: Timeout in seconds. Default: None.
            n_jobs: Number of parallel jobs. Default: 1.
            accelerator: Training accelerator. Default: "auto".
            devices: Number of devices. Default: 1.
            precision: Training precision. Default: "32".
            show_progress_bar: Show Optuna progress bar. Default: True.

        Returns:
            Completed Optuna study.
        """
        objective = self.create_objective(
            datamodule=datamodule,
            config=config,
            max_epochs=max_epochs,
            accelerator=accelerator,
            devices=devices,
            precision=precision,
        )

        self.study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=n_jobs,
            show_progress_bar=show_progress_bar,
        )

        return self.study

    def get_best_params(self) -> dict[str, Any]:
        """
        Get the best hyperparameters found.

        Returns:
            Dictionary of best hyperparameters.

        Raises:
            ValueError: If no trials have been completed.
        """
        if len(self.study.trials) == 0:
            raise ValueError("No trials completed. Run optimize() first.")

        return cast(dict[str, Any], self.study.best_params)

    def get_best_value(self) -> float:
        """
        Get the best objective value found.

        Returns:
            Best objective value.
        """
        return cast(float, self.study.best_value)

    def get_trials_dataframe(self) -> Any:
        """
        Get trials as a pandas DataFrame.

        Returns:
            DataFrame with trial information.
        """
        return self.study.trials_dataframe()

    def export_best_config(
        self,
        output_path: Path,
        include_study_info: bool = True,
    ) -> None:
        """
        Export best configuration to YAML file.

        Args:
            output_path: Path for output YAML file.
            include_study_info: Include study metadata. Default: True.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        config = {
            "hyperparameters": self.get_best_params(),
        }

        if include_study_info:
            config["study_info"] = {
                "study_name": self.study_name,
                "best_value": float(self.get_best_value()),
                "n_trials": len(self.study.trials),
                "best_trial_number": self.study.best_trial.number,
            }

        with open(output_path, "w") as f:
            yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Best config exported to: {output_path}")

    def print_summary(self) -> None:
        """Print optimization summary to console."""
        print("\n" + "=" * 60)
        print("OPTUNA OPTIMIZATION SUMMARY")
        print("=" * 60)
        print(f"Study name: {self.study_name}")
        print(f"Total trials: {len(self.study.trials)}")
        print(f"Best trial: #{self.study.best_trial.number}")
        print(f"Best value: {self.study.best_value:.6f}")
        print("\nBest hyperparameters:")
        for key, value in self.study.best_params.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.6g}")
            else:
                print(f"  {key}: {value}")
        print("=" * 60 + "\n")
