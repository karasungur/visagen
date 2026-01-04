"""Tests for Optuna hyperparameter tuning."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from dataclasses import asdict

# Skip all tests if optuna not installed
optuna = pytest.importorskip("optuna")

from visagen.tuning.optuna_tuner import OptunaTuner, TuningConfig


class TestTuningConfig:
    """Tests for TuningConfig dataclass."""

    def test_default_values(self):
        """Test default parameter ranges."""
        config = TuningConfig()

        assert config.learning_rate_range == (1e-5, 1e-3)
        assert config.weight_decay_range == (0.001, 0.1)
        assert config.dssim_weight_range == (1.0, 20.0)
        assert config.l1_weight_range == (1.0, 20.0)
        assert config.drop_path_rate_range == (0.0, 0.3)
        assert config.gan_power_range == (0.0, 0.0)  # Disabled by default

    def test_custom_ranges(self):
        """Test custom parameter ranges."""
        config = TuningConfig(
            learning_rate_range=(1e-6, 1e-2),
            gan_power_range=(0.0, 1.0),
            batch_size_options=[4, 8, 16],
        )

        assert config.learning_rate_range == (1e-6, 1e-2)
        assert config.gan_power_range == (0.0, 1.0)
        assert config.batch_size_options == [4, 8, 16]

    def test_is_gan_enabled_false(self):
        """Test GAN disabled by default."""
        config = TuningConfig()
        assert config.is_gan_enabled() is False

    def test_is_gan_enabled_true(self):
        """Test GAN enabled when range > 0."""
        config = TuningConfig(gan_power_range=(0.0, 0.5))
        assert config.is_gan_enabled() is True

    def test_dataclass_asdict(self):
        """Test conversion to dict."""
        config = TuningConfig()
        config_dict = asdict(config)

        assert "learning_rate_range" in config_dict
        assert "dssim_weight_range" in config_dict


class TestOptunaTuner:
    """Tests for OptunaTuner."""

    def test_init_default(self):
        """Test default initialization."""
        tuner = OptunaTuner()

        assert tuner.study_name == "visagen_hpo"
        assert tuner.storage is None
        assert tuner.direction == "minimize"

    def test_init_with_storage(self, tmp_path):
        """Test initialization with storage path."""
        storage_path = tmp_path / "study.db"
        tuner = OptunaTuner(
            study_name="test_study",
            storage_path=storage_path,
        )

        assert tuner.study_name == "test_study"
        assert "sqlite:///" in tuner.storage
        assert str(storage_path) in tuner.storage

    def test_study_property_creates_study(self):
        """Test study is created on first access."""
        tuner = OptunaTuner(study_name="test_auto_create")

        # Study should be None before access
        assert tuner._study is None

        # Access study property
        study = tuner.study

        # Should now be created
        assert study is not None
        assert tuner._study is not None
        assert study.study_name == "test_auto_create"

    def test_study_load_if_exists(self, tmp_path):
        """Test study is loaded if exists."""
        storage_path = tmp_path / "persist.db"

        # Create first tuner and study
        tuner1 = OptunaTuner(
            study_name="persist_test",
            storage_path=storage_path,
        )
        _ = tuner1.study  # Create study

        # Create second tuner with same storage
        tuner2 = OptunaTuner(
            study_name="persist_test",
            storage_path=storage_path,
        )
        study2 = tuner2.study

        assert study2.study_name == "persist_test"

    def test_suggest_hyperparameters(self):
        """Test hyperparameter suggestion."""
        tuner = OptunaTuner()
        config = TuningConfig()

        # Create mock trial
        mock_trial = Mock(spec=optuna.Trial)
        mock_trial.suggest_float = Mock(return_value=0.001)
        mock_trial.suggest_categorical = Mock(return_value=8)

        params = tuner._suggest_hyperparameters(mock_trial, config)

        # Check required params are present
        assert "learning_rate" in params
        assert "weight_decay" in params
        assert "dssim_weight" in params
        assert "l1_weight" in params
        assert "drop_path_rate" in params

        # GAN should not be present (disabled by default)
        assert "gan_power" not in params

    def test_suggest_hyperparameters_with_gan(self):
        """Test hyperparameter suggestion with GAN enabled."""
        tuner = OptunaTuner()
        config = TuningConfig(gan_power_range=(0.0, 0.5))

        mock_trial = Mock(spec=optuna.Trial)
        mock_trial.suggest_float = Mock(return_value=0.1)

        params = tuner._suggest_hyperparameters(mock_trial, config)

        assert "gan_power" in params

    def test_suggest_hyperparameters_with_batch_size(self):
        """Test hyperparameter suggestion with batch size options."""
        tuner = OptunaTuner()
        config = TuningConfig(batch_size_options=[4, 8, 16])

        mock_trial = Mock(spec=optuna.Trial)
        mock_trial.suggest_float = Mock(return_value=0.001)
        mock_trial.suggest_categorical = Mock(return_value=8)

        params = tuner._suggest_hyperparameters(mock_trial, config)

        assert "batch_size" in params
        assert params["batch_size"] == 8

    def test_get_best_params_no_trials(self):
        """Test error when no trials completed."""
        tuner = OptunaTuner(study_name="empty_study")

        with pytest.raises(ValueError, match="No trials completed"):
            tuner.get_best_params()

    def test_export_best_config(self, tmp_path):
        """Test config export to YAML."""
        tuner = OptunaTuner(study_name="export_test")

        # Create a trial manually
        def dummy_objective(trial):
            return trial.suggest_float("x", 0, 1)

        tuner.study.optimize(dummy_objective, n_trials=1)

        # Export config
        output_path = tmp_path / "config.yaml"
        tuner.export_best_config(output_path)

        assert output_path.exists()

        # Read and verify content
        import yaml
        with open(output_path) as f:
            config = yaml.safe_load(f)

        assert "hyperparameters" in config
        assert "study_info" in config
        assert config["study_info"]["n_trials"] == 1

    def test_export_best_config_no_study_info(self, tmp_path):
        """Test config export without study info."""
        tuner = OptunaTuner(study_name="export_test_minimal")

        def dummy_objective(trial):
            return trial.suggest_float("x", 0, 1)

        tuner.study.optimize(dummy_objective, n_trials=1)

        output_path = tmp_path / "config_minimal.yaml"
        tuner.export_best_config(output_path, include_study_info=False)

        import yaml
        with open(output_path) as f:
            config = yaml.safe_load(f)

        assert "hyperparameters" in config
        assert "study_info" not in config

    def test_print_summary(self, capsys):
        """Test summary printing."""
        tuner = OptunaTuner(study_name="summary_test")

        def dummy_objective(trial):
            x = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
            return x

        tuner.study.optimize(dummy_objective, n_trials=2)
        tuner.print_summary()

        captured = capsys.readouterr()
        assert "OPTUNA OPTIMIZATION SUMMARY" in captured.out
        assert "summary_test" in captured.out
        assert "Best trial" in captured.out


class TestOptunaTunerIntegration:
    """Integration tests for OptunaTuner (require more dependencies)."""

    @pytest.mark.integration
    def test_create_objective(self):
        """Test objective function creation."""
        # This would require actual DFLModule and DataModule
        # Marked as integration test
        pass

    @pytest.mark.integration
    def test_full_optimization_run(self):
        """Test complete optimization run."""
        # Would require actual training data
        # Marked as integration test
        pass


class TestPrunerAndSampler:
    """Tests for pruner and sampler configuration."""

    def test_default_pruner(self):
        """Test default MedianPruner."""
        tuner = OptunaTuner()

        assert isinstance(tuner.pruner, optuna.pruners.MedianPruner)

    def test_custom_pruner(self):
        """Test custom pruner."""
        custom_pruner = optuna.pruners.SuccessiveHalvingPruner()
        tuner = OptunaTuner(pruner=custom_pruner)

        assert isinstance(tuner.pruner, optuna.pruners.SuccessiveHalvingPruner)

    def test_default_sampler(self):
        """Test default TPESampler."""
        tuner = OptunaTuner()

        assert isinstance(tuner.sampler, optuna.samplers.TPESampler)

    def test_custom_sampler(self):
        """Test custom sampler."""
        custom_sampler = optuna.samplers.RandomSampler()
        tuner = OptunaTuner(sampler=custom_sampler)

        assert isinstance(tuner.sampler, optuna.samplers.RandomSampler)
