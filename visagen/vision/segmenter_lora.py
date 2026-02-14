"""
LoRA Adapter for SegFormer.

Provides Low-Rank Adaptation (LoRA) for efficient fine-tuning
of SegFormer face parsing model.
"""

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

from visagen.vision.segmenter import (
    DEFAULT_MODEL_PATH,
    FACE_COMPONENTS,
    LABEL_TO_ID,
    FaceSegmenter,
    SegmentationResult,
)


@dataclass
class LoRAConfig:
    """
    Configuration for LoRA adapters.

    Attributes:
        rank: Rank of the low-rank decomposition. Default: 8.
        alpha: Scaling factor for LoRA. Default: 16.0.
        dropout: Dropout probability for LoRA layers. Default: 0.1.
        target_modules: List of module name patterns to apply LoRA to.
    """

    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.1
    target_modules: list[str] = field(
        default_factory=lambda: [
            "attention.self.query",
            "attention.self.key",
            "attention.self.value",
        ]
    )


class LoRALayer(nn.Module):
    """
    Low-Rank Adaptation layer.

    Implements W' = W + BA where B and A are low-rank matrices.
    This allows efficient fine-tuning with minimal parameters.

    Args:
        original_layer: The original linear layer to adapt.
        rank: Rank of the low-rank decomposition.
        alpha: Scaling factor for the adaptation.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        original_layer: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = original_layer.in_features
        out_features = original_layer.out_features

        # Freeze original layer
        for param in self.original_layer.parameters():
            param.requires_grad = False

        # LoRA matrices: B @ A (out, in) = (out, rank) @ (rank, in)
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # Dropout
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        # Initialize A with Kaiming uniform and B with zeros
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with LoRA adaptation."""
        # Original forward
        original_output = self.original_layer(x)

        # LoRA forward: x @ A^T @ B^T * scaling
        lora_output = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        lora_output = lora_output * self.scaling

        return cast(torch.Tensor, original_output + lora_output)

    def merge_weights(self) -> None:
        """Merge LoRA weights into original layer for inference."""
        with torch.no_grad():
            # W' = W + B @ A * scaling
            delta_w = (self.lora_B @ self.lora_A) * self.scaling
            self.original_layer.weight.add_(delta_w)

    def get_lora_params(self) -> dict[str, Any]:
        """Get LoRA parameters for saving."""
        return {
            "lora_A": self.lora_A.data.clone(),
            "lora_B": self.lora_B.data.clone(),
            "rank": self.rank,
            "alpha": self.alpha,
        }

    def load_lora_params(self, params: dict[str, torch.Tensor]) -> None:
        """Load LoRA parameters."""
        self.lora_A.data.copy_(params["lora_A"])
        self.lora_B.data.copy_(params["lora_B"])


class SegFormerLoRA(nn.Module):
    """
    SegFormer model with LoRA adapters.

    Wraps a SegFormer model and adds LoRA layers to attention
    modules for efficient fine-tuning.

    Args:
        model: Pre-trained SegFormer model.
        config: LoRA configuration.
    """

    def __init__(
        self,
        model: SegformerForSemanticSegmentation,
        config: LoRAConfig | None = None,
    ) -> None:
        super().__init__()

        self.model = model
        self.config = config or LoRAConfig()
        self.lora_layers: dict[str, LoRALayer] = {}

        # Apply LoRA to target modules
        self._apply_lora()

    def _apply_lora(self) -> None:
        """Apply LoRA to target modules in the model."""
        for name, module in self.model.named_modules():
            # Check if this module matches any target pattern
            if self._should_apply_lora(name, module):
                self._replace_with_lora(name, module)

    def _should_apply_lora(self, name: str, module: nn.Module) -> bool:
        """Check if LoRA should be applied to this module."""
        if not isinstance(module, nn.Linear):
            return False

        for pattern in self.config.target_modules:
            if pattern in name:
                return True

        return False

    def _replace_with_lora(self, name: str, module: nn.Linear) -> None:
        """Replace a linear layer with LoRA-wrapped version."""
        lora_layer = LoRALayer(
            original_layer=module,
            rank=self.config.rank,
            alpha=self.config.alpha,
            dropout=self.config.dropout,
        )

        # Store reference
        self.lora_layers[name] = lora_layer

        # Replace in model
        parts = name.split(".")
        parent = self.model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], lora_layer)

    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass through the model."""
        return cast(dict[str, torch.Tensor], self.model(pixel_values=pixel_values, labels=labels))

    def get_trainable_params(self) -> Iterator[nn.Parameter]:
        """Get only LoRA trainable parameters."""
        for lora_layer in self.lora_layers.values():
            yield lora_layer.lora_A
            yield lora_layer.lora_B

    def get_trainable_param_count(self) -> int:
        """Get count of trainable parameters."""
        return sum(p.numel() for p in self.get_trainable_params())

    def save_lora_weights(self, path: Path | str) -> None:
        """
        Save LoRA weights to file.

        Args:
            path: Output path for weights file (.pt).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "config": {
                "rank": self.config.rank,
                "alpha": self.config.alpha,
                "dropout": self.config.dropout,
                "target_modules": self.config.target_modules,
            },
            "lora_weights": {},
        }

        for name, lora_layer in self.lora_layers.items():
            state["lora_weights"][name] = lora_layer.get_lora_params()

        torch.save(state, path)

    def load_lora_weights(self, path: Path | str) -> None:
        """
        Load LoRA weights from file.

        Args:
            path: Path to weights file (.pt).
        """
        path = Path(path)
        state = torch.load(path, map_location="cpu", weights_only=True)

        for name, params in state["lora_weights"].items():
            if name in self.lora_layers:
                self.lora_layers[name].load_lora_params(params)

    def merge_and_unload(self) -> SegformerForSemanticSegmentation:
        """
        Merge LoRA weights and return original model.

        Returns:
            Original model with merged LoRA weights.
        """
        for lora_layer in self.lora_layers.values():
            lora_layer.merge_weights()

        return self.model


class FaceSegmenterLoRA(FaceSegmenter):
    """
    Extended FaceSegmenter with LoRA support.

    Allows loading project-specific LoRA weights for improved
    segmentation on specific face types or conditions.

    Args:
        model_name: Model path or HuggingFace identifier.
        device: Device for inference.
        use_half: Use FP16 for faster inference.
        lora_weights: Optional path to LoRA weights file.
        lora_config: Optional LoRA configuration.

    Example:
        >>> segmenter = FaceSegmenterLoRA(lora_weights="./lora/project.pt")
        >>> result = segmenter.segment(face_image)
    """

    def __init__(
        self,
        model_name: str | Path | None = None,
        device: str | None = None,
        use_half: bool = True,
        lora_weights: Path | str | None = None,
        lora_config: LoRAConfig | None = None,
    ) -> None:
        # Initialize base segmenter (loads model)
        super().__init__(
            model_name=model_name,
            device=device,
            use_half=False,  # Don't use half until after LoRA setup
        )

        self._lora_enabled = False
        self._lora_model: SegFormerLoRA | None = None

        # Apply LoRA if config or weights provided
        if lora_config is not None or lora_weights is not None:
            self._setup_lora(lora_config or LoRAConfig())

            if lora_weights is not None:
                self.load_lora_weights(lora_weights)

            self._lora_enabled = True

        # Convert to half precision if requested
        if use_half and device == "cuda":
            self.model.half()
            self.use_half = True

    def _setup_lora(self, config: LoRAConfig) -> None:
        """Set up LoRA adapters on the model."""
        self._lora_model = SegFormerLoRA(self.model, config)
        self.model = self._lora_model.model

    @property
    def lora_enabled(self) -> bool:
        """Check if LoRA is enabled."""
        return self._lora_enabled

    @property
    def lora_model(self) -> SegFormerLoRA | None:
        """Get the LoRA-wrapped model."""
        return self._lora_model

    def load_lora_weights(self, path: Path | str) -> None:
        """
        Load LoRA weights from file.

        Args:
            path: Path to LoRA weights file.
        """
        if self._lora_model is None:
            self._setup_lora(LoRAConfig())

        assert self._lora_model is not None
        self._lora_model.load_lora_weights(path)
        self._lora_enabled = True

    def save_lora_weights(self, path: Path | str) -> None:
        """
        Save LoRA weights to file.

        Args:
            path: Output path for weights file.
        """
        if self._lora_model is None:
            raise RuntimeError("LoRA not enabled. Nothing to save.")

        self._lora_model.save_lora_weights(path)

    def get_trainable_params(self) -> Iterator[nn.Parameter]:
        """Get trainable LoRA parameters."""
        if self._lora_model is None:
            return iter([])

        return self._lora_model.get_trainable_params()

    def get_trainable_param_count(self) -> int:
        """Get count of trainable parameters."""
        if self._lora_model is None:
            return 0

        return self._lora_model.get_trainable_param_count()

    def train_mode(self) -> None:
        """Set model to training mode."""
        self.model.train()

    def eval_mode(self) -> None:
        """Set model to evaluation mode."""
        self.model.eval()


def create_lora_segmenter(
    lora_weights: Path | str | None = None,
    rank: int = 8,
    alpha: float = 16.0,
    device: str | None = None,
) -> FaceSegmenterLoRA:
    """
    Factory function to create a LoRA-enabled segmenter.

    Args:
        lora_weights: Optional path to pre-trained LoRA weights.
        rank: LoRA rank. Default: 8.
        alpha: LoRA alpha scaling. Default: 16.0.
        device: Compute device. Default: auto-detect.

    Returns:
        FaceSegmenterLoRA instance.
    """
    config = LoRAConfig(rank=rank, alpha=alpha)

    return FaceSegmenterLoRA(
        device=device,
        lora_weights=lora_weights,
        lora_config=config,
    )
