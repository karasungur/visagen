"""
Export-friendly model wrapper for Visagen.

Provides a wrapper that combines encoder and decoder into a single
module suitable for ONNX/TensorRT export.

The wrapper handles:
- Tuple outputs from encoder (features, latent)
- Skip connection preparation for decoder
- Single tensor input/output interface
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from visagen.training import TrainingModule


class ExportableModel(nn.Module):
    """
    Wrapper that combines encoder and decoder for ONNX export.

    This wrapper provides a clean single-input, single-output interface
    that is compatible with ONNX and TensorRT export requirements.

    The TrainingModule has:
    - Encoder that returns (features_list, latent)
    - Decoder that takes (latent, skip_features)

    This wrapper flattens this into:
    - Input: image tensor (B, C, H, W)
    - Output: reconstructed image tensor (B, C, H, W)

    Args:
        encoder: ConvNeXt encoder module.
        decoder: Decoder module.

    Example:
        >>> model = ExportableModel.from_checkpoint("model.ckpt")
        >>> model.eval()
        >>> output = model(torch.randn(1, 3, 256, 256))
        >>> output.shape
        torch.Size([1, 3, 256, 256])
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for export.

        Returns image tensor only (mask discarded for export).

        Args:
            x: Input tensor of shape (B, 3, H, W).

        Returns:
            Output tensor of shape (B, 3, H, W).
        """
        # Encode - returns (features_list, latent)
        features, latent = self.encoder(x)

        # Prepare skip connections
        # features: [stage0, stage1, stage2, stage3]
        # decoder needs: [stage2, stage1, stage0, stage0] (deep to shallow)
        skip_features = features[:-1][::-1] + [features[0]]

        # Decode
        output = self.decoder(latent, skip_features)

        if isinstance(output, tuple):
            output = output[0]  # export only image, not mask

        result: torch.Tensor = output
        return result

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        map_location: str | None = "cpu",
        strict: bool = True,
    ) -> "ExportableModel":
        """
        Load model from Lightning checkpoint.

        Args:
            checkpoint_path: Path to .ckpt file.
            map_location: Device to load model to. Default: "cpu".
            strict: Whether to strictly enforce matching keys. Default: True.

        Returns:
            ExportableModel instance ready for export.

        Raises:
            FileNotFoundError: If checkpoint file doesn't exist.
            RuntimeError: If checkpoint loading fails.
        """
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        logger.info(f"Loading checkpoint from {checkpoint_path}")

        from visagen.training import TrainingModule

        # Load Lightning module
        module = TrainingModule.load_from_checkpoint(
            str(checkpoint_path),
            map_location=map_location,
            strict=strict,
        )
        assert module.encoder is not None
        assert module.decoder is not None

        # Extract encoder and decoder
        model = cls(
            encoder=module.encoder,
            decoder=module.decoder,
        )

        logger.info("Model loaded successfully")

        return model

    @classmethod
    def from_module(cls, module: "TrainingModule") -> "ExportableModel":
        """
        Create from existing TrainingModule instance.

        Args:
            module: TrainingModule instance.

        Returns:
            ExportableModel instance.
        """
        assert module.encoder is not None
        assert module.decoder is not None
        return cls(
            encoder=module.encoder,
            decoder=module.decoder,
        )

    def get_input_shape(self) -> tuple:
        """Get expected input shape (B, C, H, W)."""
        return (1, 3, 256, 256)

    def get_output_shape(self) -> tuple:
        """Get expected output shape (B, C, H, W)."""
        return (1, 3, 256, 256)
