"""
Visagen Tuning - Hyperparameter Optimization.

Provides Optuna-based hyperparameter optimization for DFLModule training.

Example:
    >>> from visagen.tuning import OptunaTuner, TuningConfig
    >>> config = TuningConfig(learning_rate_range=(1e-5, 1e-3))
    >>> tuner = OptunaTuner(study_name="my_study")
    >>> study = tuner.optimize(datamodule, config, n_trials=20)
    >>> print(study.best_params)
"""

from visagen.tuning.optuna_tuner import OptunaTuner, TuningConfig

__all__ = ["OptunaTuner", "TuningConfig"]
