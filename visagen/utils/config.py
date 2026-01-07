"""
Configuration validation and utilities for Visagen.

This module provides:
- CONFIG_SCHEMA: Schema definition for all training configuration options
- validate_config(): Validate a configuration dictionary
- load_config_with_validation(): Load and validate a YAML config file
- print_config_summary(): Print a formatted configuration summary
- get_default_config(): Get default configuration values
"""

from pathlib import Path
from typing import Any

# Required fields for training config
REQUIRED_FIELDS = ["src_dir", "dst_dir"]

# Field type definitions with validation rules
CONFIG_SCHEMA: dict[str, dict[str, Any]] = {
    # ==========================================================================
    # Paths
    # ==========================================================================
    "src_dir": {"type": "path", "required": True},
    "dst_dir": {"type": "path", "required": True},
    "output_dir": {"type": "path", "default": "workspace/model"},
    "resume": {"type": "path", "default": None},
    "pretrain_from": {"type": "path", "default": None},
    "config": {"type": "path", "default": None},
    # ==========================================================================
    # Training
    # ==========================================================================
    "batch_size": {"type": "int", "min": 1, "max": 128, "default": 8},
    "max_epochs": {"type": "int", "min": 1, "default": 500},
    "learning_rate": {"type": "float", "min": 1e-7, "max": 1.0, "default": 1e-4},
    "image_size": {"type": "int", "choices": [64, 128, 256, 512], "default": 256},
    "num_workers": {"type": "int", "min": 0, "max": 32, "default": 4},
    "val_split": {"type": "float", "min": 0.0, "max": 0.5, "default": 0.1},
    # ==========================================================================
    # Model
    # ==========================================================================
    "model_type": {
        "type": "str",
        "choices": ["standard", "diffusion", "eg3d"],
        "default": "standard",
    },
    "encoder_dims": {"type": "str", "default": "64,128,256,512"},
    "encoder_depths": {"type": "str", "default": "2,2,4,2"},
    # Diffusion model specific
    "texture_weight": {"type": "float", "min": 0.0, "default": 0.0},
    "use_pretrained_vae": {"type": "bool", "default": True},
    "no_pretrained_vae": {"type": "bool", "default": False},
    # EG3D model specific
    "eg3d_latent_dim": {"type": "int", "min": 64, "max": 2048, "default": 512},
    "eg3d_plane_channels": {"type": "int", "min": 8, "max": 128, "default": 32},
    "eg3d_render_resolution": {"type": "int", "min": 16, "max": 256, "default": 64},
    # ==========================================================================
    # Loss weights
    # ==========================================================================
    "dssim_weight": {"type": "float", "min": 0.0, "default": 10.0},
    "l1_weight": {"type": "float", "min": 0.0, "default": 10.0},
    "lpips_weight": {"type": "float", "min": 0.0, "default": 0.0},
    "eyes_mouth_weight": {"type": "float", "min": 0.0, "default": 0.0},
    "gaze_weight": {"type": "float", "min": 0.0, "default": 0.0},
    # ==========================================================================
    # Augmentation
    # ==========================================================================
    "no_augmentation": {"type": "bool", "default": False},
    "no_warp": {"type": "bool", "default": False},
    # ==========================================================================
    # Preview & Backup
    # ==========================================================================
    "preview_interval": {"type": "int", "min": 1, "default": 500},
    "no_preview": {"type": "bool", "default": False},
    "save_preview_history": {"type": "bool", "default": False},
    "backup_interval": {"type": "int", "min": 0, "default": 0},
    "backup_minutes": {"type": "int", "min": 0, "default": 0},
    "max_backups": {"type": "int", "min": 1, "max": 100, "default": 5},
    # ==========================================================================
    # Target
    # ==========================================================================
    "max_steps": {"type": "int", "min": 0, "default": 0},
    "target_loss": {"type": "float", "min": 0.0, "default": None, "nullable": True},
    # ==========================================================================
    # Hardware
    # ==========================================================================
    "devices": {"type": "int", "min": 1, "default": 1},
    "accelerator": {
        "type": "str",
        "choices": ["auto", "cpu", "gpu", "cuda", "mps"],
        "default": "auto",
    },
    "precision": {
        "type": "str",
        "choices": ["32", "16-mixed", "bf16-mixed"],
        "default": "32",
    },
    # ==========================================================================
    # Checkpoint
    # ==========================================================================
    "save_top_k": {"type": "int", "min": 1, "max": 100, "default": 3},
}


def validate_config(config: dict[str, Any]) -> list[str]:
    """
    Validate configuration dictionary against schema.

    Args:
        config: Configuration dictionary to validate.

    Returns:
        List of error messages (empty if valid).

    Example:
        >>> errors = validate_config({"batch_size": -1})
        >>> print(errors)
        ['batch_size: value -1 below minimum 1']
    """
    errors: list[str] = []

    # Check required fields
    for field in REQUIRED_FIELDS:
        if field not in config:
            errors.append(f"Missing required field: {field}")

    # Validate each field
    for key, value in config.items():
        if key not in CONFIG_SCHEMA:
            continue  # Allow unknown fields (forward compatibility)

        schema = CONFIG_SCHEMA[key]

        # Handle nullable fields
        if value is None:
            if schema.get("nullable", False):
                continue
            if schema.get("default") is None:
                continue

        # Type validation
        if schema["type"] == "int":
            if not isinstance(value, int) or isinstance(value, bool):
                errors.append(f"{key}: expected int, got {type(value).__name__}")
                continue
            if "min" in schema and value < schema["min"]:
                errors.append(f"{key}: value {value} below minimum {schema['min']}")
            if "max" in schema and value > schema["max"]:
                errors.append(f"{key}: value {value} above maximum {schema['max']}")
            if "choices" in schema and value not in schema["choices"]:
                errors.append(f"{key}: value {value} not in {schema['choices']}")

        elif schema["type"] == "float":
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                errors.append(f"{key}: expected float, got {type(value).__name__}")
                continue
            if "min" in schema and value < schema["min"]:
                errors.append(f"{key}: value {value} below minimum {schema['min']}")
            if "max" in schema and value > schema["max"]:
                errors.append(f"{key}: value {value} above maximum {schema['max']}")

        elif schema["type"] == "str":
            if not isinstance(value, str):
                errors.append(f"{key}: expected str, got {type(value).__name__}")
                continue
            if "choices" in schema and value not in schema["choices"]:
                errors.append(f"{key}: value '{value}' not in {schema['choices']}")

        elif schema["type"] == "bool":
            if not isinstance(value, bool):
                errors.append(f"{key}: expected bool, got {type(value).__name__}")

        elif schema["type"] == "path":
            if value is not None and not isinstance(value, (str, Path)):
                errors.append(f"{key}: expected path, got {type(value).__name__}")

    return errors


def load_config_with_validation(config_path: Path | str) -> tuple[dict, list[str]]:
    """
    Load and validate YAML config file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Tuple of (config_dict, error_list).
        If errors is non-empty, the config may be invalid.

    Example:
        >>> config, errors = load_config_with_validation("config/train.yaml")
        >>> if errors:
        ...     print("Errors:", errors)
        ... else:
        ...     print("Config loaded successfully")
    """
    try:
        import yaml
    except ImportError:
        return {}, ["PyYAML not installed. Run: pip install pyyaml"]

    config_path = Path(config_path)

    if not config_path.exists():
        return {}, [f"Config file not found: {config_path}"]

    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
    except Exception as e:
        return {}, [f"Error parsing YAML: {e}"]

    if config is None:
        config = {}

    errors = validate_config(config)
    return config, errors


def get_default_config() -> dict[str, Any]:
    """
    Get default configuration values.

    Returns:
        Dictionary with all default values from CONFIG_SCHEMA.

    Example:
        >>> defaults = get_default_config()
        >>> print(defaults["batch_size"])
        8
    """
    return {
        key: schema.get("default")
        for key, schema in CONFIG_SCHEMA.items()
        if "default" in schema
    }


def print_config_summary(config: dict[str, Any]) -> None:
    """
    Print configuration summary to console.

    Groups configuration values by category and prints them
    in a formatted, easy-to-read manner.

    Args:
        config: Configuration dictionary to print.
    """
    print("\n" + "=" * 60)
    print(" Configuration Summary (from config file)")
    print("=" * 60)

    # Group by category
    categories = {
        "Paths": ["src_dir", "dst_dir", "output_dir", "resume", "pretrain_from"],
        "Training": [
            "batch_size",
            "max_epochs",
            "learning_rate",
            "image_size",
            "num_workers",
            "val_split",
        ],
        "Model": ["model_type", "encoder_dims", "encoder_depths"],
        "Loss Weights": [
            "dssim_weight",
            "l1_weight",
            "lpips_weight",
            "eyes_mouth_weight",
            "gaze_weight",
        ],
        "Preview & Backup": [
            "preview_interval",
            "backup_interval",
            "backup_minutes",
            "max_backups",
        ],
        "Target": ["max_steps", "target_loss"],
        "Hardware": ["devices", "accelerator", "precision"],
    }

    for category, fields in categories.items():
        values = [(f, config.get(f)) for f in fields if f in config]
        if values:
            print(f"\n{category}:")
            for field, value in values:
                print(f"  {field}: {value}")

    print("\n" + "=" * 60 + "\n")
