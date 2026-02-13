"""Neural Color Transfer using VGG features.

Deep learning based color transfer that matches color statistics
in feature space for more semantic color matching.

This module provides neural network based color transfer that goes beyond
simple statistical matching by using learned feature representations to
match colors in a perceptually meaningful way.

Features:
    - VGG-based feature extraction for semantic color matching
    - Gram matrix computation for style representation
    - Histogram matching in feature space
    - Optional luminance preservation

References:
    - Deep Photo Style Transfer (Luan et al., 2017)
    - Neural Style Transfer (Gatys et al., 2016)
    - A Closed-form Solution to Photorealistic Image Stylization
"""

from __future__ import annotations

import logging
from typing import Literal

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Check torch availability (separate from torchvision)
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Check torchvision availability for VGG-based features
try:
    from torchvision import models

    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False


def is_neural_color_available() -> bool:
    """Check if neural color transfer is available (requires torch)."""
    return TORCH_AVAILABLE


def is_vgg_available() -> bool:
    """Check if VGG-based features are available (requires torchvision)."""
    return TORCH_AVAILABLE and TORCHVISION_AVAILABLE


if TORCH_AVAILABLE and TORCHVISION_AVAILABLE:

    class VGGFeatureExtractor(nn.Module):
        """Extract VGG features for color matching.

        Uses pretrained VGG19 to extract features at specified layers
        for computing color statistics in feature space.

        Args:
            layers: List of layer names to extract features from.
                   Default: ["relu1_1", "relu2_1", "relu3_1"].
            use_input_norm: Normalize input to VGG expected range. Default: True.
        """

        # VGG19 layer name to index mapping
        LAYER_MAP = {
            "relu1_1": 1,
            "relu1_2": 3,
            "relu2_1": 6,
            "relu2_2": 8,
            "relu3_1": 11,
            "relu3_2": 13,
            "relu3_3": 15,
            "relu3_4": 17,
            "relu4_1": 20,
            "relu4_2": 22,
            "relu4_3": 24,
            "relu4_4": 26,
            "relu5_1": 29,
            "relu5_2": 31,
            "relu5_3": 33,
            "relu5_4": 35,
        }

        def __init__(
            self,
            layers: list[str] | None = None,
            use_input_norm: bool = True,
        ) -> None:
            super().__init__()

            if layers is None:
                layers = ["relu1_1", "relu2_1", "relu3_1"]

            self.layers = layers
            self.use_input_norm = use_input_norm

            # Load pretrained VGG19
            try:
                vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
            except Exception as exc:
                logger.warning(
                    "Failed to load pretrained VGG19 weights for neural color transfer; "
                    "falling back to untrained VGG19. Reason: %s",
                    exc,
                )
                vgg = models.vgg19(weights=None)
            self.features = vgg.features

            # Freeze all parameters
            for param in self.parameters():
                param.requires_grad = False

            # Register normalization buffers
            if use_input_norm:
                mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
                self.register_buffer("mean", mean)
                self.register_buffer("std", std)

            # Get max layer index needed
            self.max_layer = max(self.LAYER_MAP[layer] for layer in layers)

        def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
            """Extract features from specified layers.

            Args:
                x: Input image (B, 3, H, W) in [0, 1] range, RGB order.

            Returns:
                Dictionary mapping layer names to feature tensors.
            """
            # Normalize input
            if self.use_input_norm:
                x = (x - self.mean) / self.std

            features = {}
            for i, layer in enumerate(self.features):
                x = layer(x)
                # Check if this layer should be extracted
                for layer_name, layer_idx in self.LAYER_MAP.items():
                    if i == layer_idx and layer_name in self.layers:
                        features[layer_name] = x

                # Early exit if we've extracted all needed layers
                if i >= self.max_layer:
                    break

            return features


def compute_gram_matrix(features: torch.Tensor) -> torch.Tensor:
    """Compute Gram matrix for style representation.

    The Gram matrix captures the correlations between different filter
    responses, providing a style representation of the input.

    Args:
        features: Feature tensor (B, C, H, W).

    Returns:
        Gram matrix (B, C, C).
    """
    b, c, h, w = features.shape
    features_flat = features.view(b, c, h * w)
    gram = torch.bmm(features_flat, features_flat.transpose(1, 2))
    return gram / (c * h * w)


def compute_mean_std(features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute channel-wise mean and standard deviation.

    Args:
        features: Feature tensor (B, C, H, W).

    Returns:
        Tuple of (mean, std) with shape (B, C, 1, 1).
    """
    mean = features.mean(dim=[2, 3], keepdim=True)
    std = features.std(dim=[2, 3], keepdim=True) + 1e-8
    return mean, std


def adaptive_instance_normalization(
    content: torch.Tensor,
    style: torch.Tensor,
) -> torch.Tensor:
    """Apply Adaptive Instance Normalization.

    Normalizes content features and applies style statistics.

    Args:
        content: Content feature tensor (B, C, H, W).
        style: Style feature tensor (B, C, H, W).

    Returns:
        Stylized feature tensor (B, C, H, W).
    """
    content_mean, content_std = compute_mean_std(content)
    style_mean, style_std = compute_mean_std(style)

    normalized = (content - content_mean) / content_std
    return normalized * style_std + style_mean


def match_histograms_channel(
    source: np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    """Match histogram of source to reference for a single channel.

    Args:
        source: Source channel (H, W) float32.
        reference: Reference channel (H, W) float32.

    Returns:
        Matched channel (H, W) float32.
    """
    # Compute histograms
    src_values, src_unique_indices, src_counts = np.unique(
        source.ravel(), return_inverse=True, return_counts=True
    )
    ref_values, ref_counts = np.unique(reference.ravel(), return_counts=True)

    # Compute CDFs
    src_cdf = np.cumsum(src_counts).astype(np.float64)
    src_cdf /= src_cdf[-1]
    ref_cdf = np.cumsum(ref_counts).astype(np.float64)
    ref_cdf /= ref_cdf[-1]

    # Map source values to reference values
    interp_values = np.interp(src_cdf, ref_cdf, ref_values)
    matched = interp_values[src_unique_indices].reshape(source.shape)

    return matched.astype(np.float32)


def neural_color_transfer(
    target: np.ndarray,
    reference: np.ndarray,
    *,
    strength: float = 1.0,
    preserve_luminance: bool = True,
    mode: Literal["histogram", "statistics", "gram"] = "statistics",
    device: str | None = None,
) -> np.ndarray:
    """Apply neural color transfer.

    Transfers color from reference image to target using neural network
    features for semantic color matching.

    Args:
        target: Target image to modify (H, W, 3) BGR float32 [0, 1].
        reference: Reference image for color (H, W, 3) BGR float32 [0, 1].
        strength: Transfer strength (0.0-1.0). Default: 1.0.
        preserve_luminance: Keep target luminance. Default: True.
        mode: Transfer mode:
            - "histogram": Histogram matching in LAB space
            - "statistics": Mean/std matching (fast)
            - "gram": Gram matrix based (more semantic, requires torchvision)
        device: Torch device. Default: auto-detect.

    Returns:
        Color-transferred image (H, W, 3) BGR float32 [0, 1].

    Raises:
        RuntimeError: If PyTorch is not available (for histogram/statistics)
                      or torchvision is not available (for gram mode).
    """
    # Check mode-specific requirements
    if mode == "gram":
        if not TORCHVISION_AVAILABLE:
            raise RuntimeError(
                "torchvision is required for gram mode. "
                "Install with: pip install torchvision"
            )
    elif mode in ("histogram", "statistics"):
        # These modes work without torchvision but still need some torch features
        # Actually histogram and statistics modes use numpy, not torch
        pass
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if device is None and TORCH_AVAILABLE:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Convert BGR to RGB for processing
    target_rgb = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
    reference_rgb = cv2.cvtColor(reference, cv2.COLOR_BGR2RGB)

    if preserve_luminance:
        # Convert to LAB and save target luminance
        target_lab = cv2.cvtColor(
            (target * 255).astype(np.uint8), cv2.COLOR_BGR2LAB
        ).astype(np.float32)
        target_l = target_lab[:, :, 0].copy()

    if mode == "histogram":
        # Simple histogram matching in LAB space
        result = _histogram_match_lab(target, reference)

    elif mode == "statistics":
        # Fast mean/std matching
        result = _statistics_match(target_rgb, reference_rgb)
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

    elif mode == "gram":
        # VGG feature based matching
        result = _gram_based_transfer(target_rgb, reference_rgb, device)
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Restore luminance if requested
    if preserve_luminance:
        result_lab = cv2.cvtColor(
            (np.clip(result, 0, 1) * 255).astype(np.uint8), cv2.COLOR_BGR2LAB
        ).astype(np.float32)
        result_lab[:, :, 0] = target_l
        result = (
            cv2.cvtColor(result_lab.astype(np.uint8), cv2.COLOR_LAB2BGR).astype(
                np.float32
            )
            / 255.0
        )

    # Apply strength blending
    if strength < 1.0:
        result = target * (1 - strength) + result * strength

    return np.clip(result, 0, 1).astype(np.float32)


def _histogram_match_lab(
    target: np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    """Match histograms in LAB color space.

    Args:
        target: Target image BGR float32 [0, 1].
        reference: Reference image BGR float32 [0, 1].

    Returns:
        Matched image BGR float32 [0, 1].
    """
    # Convert to LAB
    target_lab = (
        cv2.cvtColor((target * 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(
            np.float32
        )
        / 255.0
    )
    reference_lab = (
        cv2.cvtColor((reference * 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(
            np.float32
        )
        / 255.0
    )

    # Match each channel
    result_lab = np.zeros_like(target_lab)
    for i in range(3):
        result_lab[:, :, i] = match_histograms_channel(
            target_lab[:, :, i], reference_lab[:, :, i]
        )

    # Convert back to BGR
    result = (
        cv2.cvtColor(
            (np.clip(result_lab, 0, 1) * 255).astype(np.uint8), cv2.COLOR_LAB2BGR
        ).astype(np.float32)
        / 255.0
    )

    return result


def _statistics_match(
    target: np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    """Match mean and standard deviation per channel.

    Args:
        target: Target image RGB float32 [0, 1].
        reference: Reference image RGB float32 [0, 1].

    Returns:
        Matched image RGB float32 [0, 1].
    """
    result = np.zeros_like(target)

    for i in range(3):
        t_mean = target[:, :, i].mean()
        t_std = target[:, :, i].std() + 1e-8
        r_mean = reference[:, :, i].mean()
        r_std = reference[:, :, i].std() + 1e-8

        # Normalize and rescale
        result[:, :, i] = (target[:, :, i] - t_mean) / t_std * r_std + r_mean

    return np.clip(result, 0, 1).astype(np.float32)


def _gram_based_transfer(
    target: np.ndarray,
    reference: np.ndarray,
    device: str,
) -> np.ndarray:
    """VGG Gram matrix based color transfer.

    Uses VGG features to compute style statistics and applies them
    to the target image through iterative optimization.

    Args:
        target: Target image RGB float32 [0, 1].
        reference: Reference image RGB float32 [0, 1].
        device: Torch device.

    Returns:
        Matched image RGB float32 [0, 1].
    """
    # For efficiency, we use a simplified approach:
    # Extract features, match statistics, and blend
    device_obj = torch.device(device)

    # Prepare tensors
    target_tensor = (
        torch.from_numpy(target).permute(2, 0, 1).unsqueeze(0).to(device_obj)
    )
    reference_tensor = torch.from_numpy(reference).permute(2, 0, 1).unsqueeze(0)

    # Resize reference to match target if needed
    if reference_tensor.shape[2:] != target_tensor.shape[2:]:
        reference_tensor = F.interpolate(
            reference_tensor,
            size=target_tensor.shape[2:],
            mode="bilinear",
            align_corners=False,
        )
    reference_tensor = reference_tensor.to(device_obj)

    # Load VGG feature extractor
    vgg = VGGFeatureExtractor(layers=["relu1_1", "relu2_1", "relu3_1"]).to(device_obj)
    vgg.eval()

    with torch.no_grad():
        # Extract features
        target_features = vgg(target_tensor)
        reference_features = vgg(reference_tensor)

        # Apply AdaIN to each layer and blend back
        result = target_tensor.clone()

        for layer_name in target_features:
            t_feat = target_features[layer_name]
            r_feat = reference_features[layer_name]

            # Match statistics
            stylized_feat = adaptive_instance_normalization(t_feat, r_feat)

            # Compute difference and upsample to image size
            diff = stylized_feat - t_feat
            diff_upsampled = F.interpolate(
                diff,
                size=target_tensor.shape[2:],
                mode="bilinear",
                align_corners=False,
            )

            # Blend with result (small weight per layer)
            weight = 0.1 / len(target_features)
            result = result + weight * diff_upsampled.mean(dim=1, keepdim=True)

    # Convert back to numpy
    result = result.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return np.clip(result, 0, 1).astype(np.float32)
