"""Training preset management."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class TrainingPreset:
    """Training parameters preset."""

    name: str
    # Basic parameters
    batch_size: int = 8
    max_epochs: int = 500
    learning_rate: float = 1e-4
    # Loss weights
    dssim_weight: float = 10.0
    l1_weight: float = 10.0
    lpips_weight: float = 0.0
    # GAN and model
    gan_power: float = 0.0
    precision: str = "16-mixed"
    model_type: str = "standard"
    # Experimental
    texture_weight: float = 0.0
    use_pretrained_vae: bool = True
    # Advanced loss weights
    eyes_mouth_weight: float = 0.0
    gaze_weight: float = 0.0
    true_face_power: float = 0.0
    face_style_weight: float = 0.0
    bg_style_weight: float = 0.0
    id_weight: float = 0.0
    # Temporal parameters
    temporal_enabled: bool = False
    temporal_power: float = 0.1
    temporal_consistency_weight: float = 1.0
    # Advanced
    uniform_yaw: bool = False
    masked_training: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainingPreset:
        """Deserialize from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def save(self, path: Path) -> None:
        """Save preset to JSON file."""
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path) -> TrainingPreset:
        """Load preset from JSON file."""
        data = json.loads(path.read_text())
        return cls.from_dict(data)


class PresetManager:
    """Preset management for training configurations."""

    PRESETS_DIR = Path("./presets")

    # Built-in presets
    BUILTIN_PRESETS: dict[str, TrainingPreset] = {
        "quick": TrainingPreset(
            name="Quick Training",
            max_epochs=100,
            batch_size=16,
            precision="16-mixed",
        ),
        "quality": TrainingPreset(
            name="High Quality",
            max_epochs=1000,
            batch_size=4,
            lpips_weight=1.0,
        ),
        "balanced": TrainingPreset(
            name="Balanced",
            max_epochs=500,
            batch_size=8,
        ),
        "gan": TrainingPreset(
            name="GAN Enhanced",
            max_epochs=800,
            batch_size=6,
            gan_power=0.1,
        ),
    }

    def __init__(self, presets_dir: Path | str | None = None) -> None:
        """
        Initialize preset manager.

        Args:
            presets_dir: Optional custom directory for user presets.
        """
        self.presets_dir = Path(presets_dir) if presets_dir else self.PRESETS_DIR
        self.presets_dir.mkdir(parents=True, exist_ok=True)

    def list_presets(self) -> list[tuple[str, str]]:
        """
        Return list of (display_name, key) tuples.

        Builtin presets are prefixed with 📦, user presets with 👤.
        """
        presets: list[tuple[str, str]] = []

        # Builtin presets first
        for key, preset in self.BUILTIN_PRESETS.items():
            presets.append((f"📦 {preset.name}", f"builtin:{key}"))

        # User presets
        for path in sorted(self.presets_dir.glob("*.json")):
            try:
                preset = TrainingPreset.load(path)
                presets.append((f"👤 {preset.name}", f"user:{path.stem}"))
            except Exception:
                continue

        return presets

    def load_preset(self, key: str) -> TrainingPreset | None:
        """
        Load preset by key.

        Args:
            key: Preset key in format "builtin:name" or "user:name".

        Returns:
            TrainingPreset or None if not found.
        """
        if not key:
            return None

        if key.startswith("builtin:"):
            name = key.split(":", 1)[1]
            return self.BUILTIN_PRESETS.get(name)

        if key.startswith("user:"):
            name = key.split(":", 1)[1]
            path = self.presets_dir / f"{name}.json"
            if path.exists():
                return TrainingPreset.load(path)

        return None

    def save_preset(self, preset: TrainingPreset) -> str:
        """
        Save user preset.

        Args:
            preset: Preset to save.

        Returns:
            Key of saved preset.
        """
        # Create safe filename from name
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in preset.name)
        path = self.presets_dir / f"{safe_name}.json"
        preset.save(path)
        return f"user:{safe_name}"

    def delete_preset(self, key: str) -> bool:
        """
        Delete user preset.

        Args:
            key: Preset key (must be user preset).

        Returns:
            True if deleted, False otherwise.
        """
        if not key.startswith("user:"):
            return False  # Can't delete builtin

        name = key.split(":", 1)[1]
        path = self.presets_dir / f"{name}.json"

        if path.exists():
            path.unlink()
            return True

        return False

    def get_preset_names(self) -> list[str]:
        """Get just the display names for dropdown."""
        return [name for name, _ in self.list_presets()]
