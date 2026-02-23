"""
Interactive Merger Configuration for Visagen.

Provides dataclass-based configuration for the interactive merger UI,
allowing real-time parameter adjustment during face merging.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Available merge modes
MERGE_MODES = {
    "original": "Show original frame without merge",
    "overlay": "Overlay predicted face onto destination (default)",
    "hist-match": "Match histogram of predicted to destination",
    "seamless": "Use OpenCV seamlessClone for blending",
    "seamless-hist-match": "Seamless clone with histogram matching",
}

# Mode aliases mapped to supported modes
MERGE_MODE_ALIASES = {
    "raw-rgb": "overlay",
    "raw-predict": "overlay",
}

# Available mask modes supported by FrameProcessor
MASK_MODES = {
    "full": "Full face mask (1.0 everywhere)",
    "convex_hull": "Convex hull from landmarks",
    "dst": "Destination-driven mask",
    "segmented": "SegFormer face segmentation (default)",
}

MASK_MODE_ALIASES = {
    "learned_prd": "segmented",
    "learned_dst": "segmented",
    "learned_prd_x_dst": "segmented",
    "learned_prd_plus_dst": "segmented",
    "segformer_prd": "segmented",
    "segformer_dst": "segmented",
    "segformer_prd_x_dst": "segmented",
    "all_combined": "segmented",
}

# Available color transfer modes
COLOR_TRANSFER_MODES = {
    "none": "No color transfer",
    "rct": "Reinhard Color Transfer (default)",
    "lct": "Linear Color Transfer",
    "mkl": "Monge-Kantorovitch Linear",
    "idt": "Iterative Distribution Transfer",
    "sot": "Sliced Optimal Transport",
    "mix": "Mixed LCT+SOT (best quality)",
    "hist-match": "Histogram matching",
}

# Available sharpen modes
SHARPEN_MODES = {
    "none": "No sharpening",
    "box": "Box filter sharpening",
    "gaussian": "Gaussian sharpening",
}

# Merge mode -> FrameProcessor mapping
_MODE_TO_PROCESSOR: dict[str, tuple[str, str | None] | tuple[str, str | None, bool]] = {
    "original": ("laplacian", None, True),
    "overlay": ("laplacian", None),
    "hist-match": ("laplacian", "hist-match"),
    "seamless": ("poisson", None),
    "seamless-hist-match": ("poisson", "hist-match"),
}


def map_merge_mode_to_processor(mode: str) -> tuple[str, str | None, bool]:
    """
    Map interactive merge mode to (blend_mode, forced_color_transfer).

    Returns:
        Tuple where:
        - blend_mode is one of FrameProcessor blend modes.
        - forced_color_transfer is an optional color transfer override.
        - passthrough_original indicates original frame bypass.
    """
    normalized = mode.lower()
    normalized = MERGE_MODE_ALIASES.get(normalized, normalized)
    mapped = _MODE_TO_PROCESSOR.get(normalized, ("laplacian", None))
    if len(mapped) == 2:
        blend_mode, forced_color_transfer = mapped
        return blend_mode, forced_color_transfer, False
    blend_mode, forced_color_transfer, passthrough_original = mapped
    return blend_mode, forced_color_transfer, passthrough_original


@dataclass
class InteractiveMergerConfig:
    """
    Configuration for interactive face merging.

    Controls all aspects of the face swap process including
    mode selection, mask processing, color transfer, and enhancement.

    Attributes:
        mode: Merge mode (overlay, hist-match, seamless, etc.)
        mask_mode: Mask generation mode (full, convex_hull, dst, segmented)
        color_transfer: Color transfer algorithm
        erode_mask: Mask erosion amount (-100 to 100, negative = dilate)
        blur_mask: Mask blur amount (0 to 100)
        face_scale: Face scale adjustment (-50 to 50)
        sharpen_mode: Sharpening algorithm (none, box, gaussian)
        sharpen_amount: Sharpen/blur intensity (-100 to 100)
        hist_match_threshold: Histogram match threshold (0 to 255)
        restore_face: Enable GFPGAN face restoration
        restore_strength: Restoration blend strength (0.0 to 1.0)
    """

    # Mode settings
    mode: str = "overlay"
    mask_mode: str = "segmented"
    color_transfer: str = "rct"

    # Mask parameters
    erode_mask: int = 0  # -100..100 (negative = dilate)
    blur_mask: int = 10  # 0..100

    # Face adjustments
    face_scale: int = 0  # -50..50

    # Sharpening
    sharpen_mode: str = "none"
    sharpen_amount: int = 0  # -100..100

    # Histogram matching
    hist_match_threshold: int = 238  # 0..255
    masked_hist_match: bool = True  # Use mask for histogram matching

    # Face restoration (GFPGAN)
    restore_face: bool = False
    restore_strength: float = 0.5  # 0.0..1.0

    # Super resolution
    # 0 = disabled, 1-100 = 4x upscale blend power
    super_resolution_power: int = 0  # 0..100

    # Motion blur (for temporal consistency)
    motion_blur_power: int = 0  # 0..100

    # Image degradation effects
    image_denoise_power: int = 0  # 0..500
    bicubic_degrade_power: int = 0  # 0..100
    color_degrade_power: int = 0  # 0..100

    def __post_init__(self) -> None:
        """Validate configuration values."""
        self._validate()

    def _validate(self) -> None:
        """Validate all configuration values are within bounds."""
        # Normalize aliases before validation
        self.mode = MERGE_MODE_ALIASES.get(self.mode, self.mode)
        self.mask_mode = MASK_MODE_ALIASES.get(self.mask_mode, self.mask_mode)

        # Mode validation
        if self.mode not in MERGE_MODES:
            raise ValueError(
                f"Invalid mode: {self.mode}. Must be one of {list(MERGE_MODES.keys())}"
            )

        if self.mask_mode not in MASK_MODES:
            raise ValueError(
                f"Invalid mask_mode: {self.mask_mode}. Must be one of {list(MASK_MODES.keys())}"
            )

        if self.color_transfer not in COLOR_TRANSFER_MODES:
            raise ValueError(
                f"Invalid color_transfer: {self.color_transfer}. Must be one of {list(COLOR_TRANSFER_MODES.keys())}"
            )

        if self.sharpen_mode not in SHARPEN_MODES:
            raise ValueError(
                f"Invalid sharpen_mode: {self.sharpen_mode}. Must be one of {list(SHARPEN_MODES.keys())}"
            )

        # Range validation
        self.erode_mask = max(-100, min(100, self.erode_mask))
        self.blur_mask = max(0, min(100, self.blur_mask))
        self.face_scale = max(-50, min(50, self.face_scale))
        self.sharpen_amount = max(-100, min(100, self.sharpen_amount))
        self.hist_match_threshold = max(0, min(255, self.hist_match_threshold))
        self.restore_strength = max(0.0, min(1.0, self.restore_strength))
        self.super_resolution_power = max(0, min(100, self.super_resolution_power))
        self.motion_blur_power = max(0, min(100, self.motion_blur_power))
        self.image_denoise_power = max(0, min(500, self.image_denoise_power))
        self.bicubic_degrade_power = max(0, min(100, self.bicubic_degrade_power))
        self.color_degrade_power = max(0, min(100, self.color_degrade_power))

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InteractiveMergerConfig":
        """Create configuration from dictionary."""
        # Filter out unknown keys for forward compatibility
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered_data)

    def to_json(self, path: Path | str) -> None:
        """Save configuration to JSON file."""
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def from_json(cls, path: Path | str) -> "InteractiveMergerConfig":
        """Load configuration from JSON file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        data = json.loads(path.read_text())
        return cls.from_dict(data)

    def copy(self) -> "InteractiveMergerConfig":
        """Create a copy of this configuration."""
        return InteractiveMergerConfig.from_dict(self.to_dict())

    def to_status_string(self) -> str:
        """Generate a human-readable status string."""
        parts = [
            f"Mode: {self.mode}",
            f"Mask: {self.mask_mode}",
            f"Color: {self.color_transfer}",
        ]

        if self.erode_mask != 0:
            parts.append(f"Erode: {self.erode_mask}")
        if self.blur_mask != 0:
            parts.append(f"Blur: {self.blur_mask}")
        if self.face_scale != 0:
            parts.append(f"Scale: {self.face_scale}")
        if self.sharpen_mode != "none":
            parts.append(f"Sharpen: {self.sharpen_mode} ({self.sharpen_amount})")
        if self.restore_face:
            parts.append(f"Restore: {self.restore_strength:.1f}")
        if self.super_resolution_power > 0:
            parts.append(f"SuperRes: {self.super_resolution_power}%")
        if self.motion_blur_power > 0:
            parts.append(f"MotionBlur: {self.motion_blur_power}%")

        return " | ".join(parts)


@dataclass
class InteractiveMergerSession:
    """
    Session state for interactive merger.

    Stores the current session including frame paths, current position,
    and global configuration.

    Attributes:
        frames_dir: Directory containing input frames
        checkpoint_path: Path to model checkpoint
        output_dir: Directory for output frames
        current_idx: Current frame index
        total_frames: Total number of frames
        config: Current merger configuration
    """

    frames_dir: str = ""
    checkpoint_path: str = ""
    output_dir: str = "./output"
    current_idx: int = 0
    total_frames: int = 0
    config: InteractiveMergerConfig = field(default_factory=InteractiveMergerConfig)

    def to_dict(self) -> dict[str, Any]:
        """Convert session to dictionary for serialization."""
        return {
            "frames_dir": self.frames_dir,
            "checkpoint_path": self.checkpoint_path,
            "output_dir": self.output_dir,
            "current_idx": self.current_idx,
            "total_frames": self.total_frames,
            "config": self.config.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InteractiveMergerSession":
        """Create session from dictionary."""
        config_data = data.pop("config", {})
        config = InteractiveMergerConfig.from_dict(config_data)
        return cls(config=config, **data)

    def to_json(self, path: Path | str) -> None:
        """Save session to JSON file."""
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def from_json(cls, path: Path | str) -> "InteractiveMergerSession":
        """Load session from JSON file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Session file not found: {path}")
        data = json.loads(path.read_text())
        return cls.from_dict(data)
