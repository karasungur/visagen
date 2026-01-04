"""
Face Augmentation Pipeline.

Modern PyTorch-based augmentations for face training.
Matches legacy DeepFaceLab augmentation behavior.
"""

import math
import random
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from visagen.data.warp import gen_warp_params, warp_by_params


class FaceAugmentationPipeline(nn.Module):
    """
    Complete face augmentation pipeline.

    Applies geometric and color augmentations consistently to
    face images and masks. Can run on GPU for faster processing.

    Args:
        target_size: Output image size. Default: 256.
        random_flip_prob: Horizontal flip probability. Default: 0.4.
        random_warp: Enable grid-based warping. Default: True.
        rotation_range: Min/max rotation in degrees. Default: (-10, 10).
        scale_range: Min/max scale deviation. Default: (-0.05, 0.05).
        translation_range: Min/max translation (relative). Default: (-0.05, 0.05).
        hsv_shift_amount: HSV color shift amount. Default: 0.1.
        brightness_range: Brightness adjustment range. Default: 0.1.
        contrast_range: Contrast adjustment range. Default: 0.1.
        apply_geometric: Enable geometric augmentations. Default: True.
        apply_color: Enable color augmentations. Default: True.

    Example:
        >>> pipeline = FaceAugmentationPipeline(target_size=256)
        >>> image = torch.randn(3, 256, 256)
        >>> augmented, _ = pipeline(image)
        >>> augmented.shape
        torch.Size([3, 256, 256])
    """

    def __init__(
        self,
        target_size: int = 256,
        # Geometric
        random_flip_prob: float = 0.4,
        random_warp: bool = True,
        rotation_range: Tuple[float, float] = (-10, 10),
        scale_range: Tuple[float, float] = (-0.05, 0.05),
        translation_range: Tuple[float, float] = (-0.05, 0.05),
        # Color
        hsv_shift_amount: float = 0.1,
        brightness_range: float = 0.1,
        contrast_range: float = 0.1,
        # Control
        apply_geometric: bool = True,
        apply_color: bool = True,
    ) -> None:
        super().__init__()
        self.target_size = target_size
        self.random_flip_prob = random_flip_prob
        self.random_warp = random_warp
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.translation_range = translation_range
        self.hsv_shift_amount = hsv_shift_amount
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.apply_geometric = apply_geometric
        self.apply_color = apply_color

    def forward(
        self,
        image: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply augmentations to face image and optional mask.

        Geometric transforms are applied consistently to both image and mask.
        Color transforms are only applied to the image.

        Args:
            image: Image tensor (C, H, W) or (B, C, H, W) in [0, 1] or [-1, 1].
            mask: Optional mask tensor (1, H, W) or (B, 1, H, W) in [0, 1].

        Returns:
            Tuple of (augmented_image, augmented_mask).
        """
        # Track input format
        was_batched = image.dim() == 4
        was_normalized = image.min() < -0.5

        # Add batch dimension if needed
        if not was_batched:
            image = image.unsqueeze(0)
            if mask is not None:
                mask = mask.unsqueeze(0)

        # Convert to [0, 1] for processing
        if was_normalized:
            image = (image + 1) / 2

        # Apply geometric augmentations
        if self.apply_geometric:
            image, mask = self._apply_geometric(image, mask)

        # Apply color augmentations (only to image)
        if self.apply_color:
            image = self._apply_color(image)

        # Ensure valid range
        image = torch.clamp(image, 0, 1)
        if mask is not None:
            mask = torch.clamp(mask, 0, 1)

        # Restore original format
        if was_normalized:
            image = image * 2 - 1

        if not was_batched:
            image = image.squeeze(0)
            if mask is not None:
                mask = mask.squeeze(0)

        return image, mask

    def _apply_geometric(
        self,
        image: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Apply geometric augmentations."""
        b, c, h, w = image.shape

        # Random flip
        if random.random() < self.random_flip_prob:
            image = torch.flip(image, dims=[-1])
            if mask is not None:
                mask = torch.flip(mask, dims=[-1])

        # Random warp
        if self.random_warp:
            warp_params = gen_warp_params(self.target_size)
            image = warp_by_params(image, warp_params)
            if mask is not None:
                mask = warp_by_params(mask, warp_params)

        # Random affine (rotation, scale, translation)
        if any([
            self.rotation_range != (0, 0),
            self.scale_range != (0, 0),
            self.translation_range != (0, 0),
        ]):
            affine_matrix = self._gen_affine_matrix(h, w)
            image = self._apply_affine(image, affine_matrix)
            if mask is not None:
                mask = self._apply_affine(mask, affine_matrix)

        return image, mask

    def _gen_affine_matrix(self, h: int, w: int) -> torch.Tensor:
        """Generate random affine transformation matrix."""
        # Random parameters
        rotation = random.uniform(*self.rotation_range)
        scale = 1.0 + random.uniform(*self.scale_range)
        tx = random.uniform(*self.translation_range)
        ty = random.uniform(*self.translation_range)

        # Build transformation matrix
        angle_rad = math.radians(rotation)
        cos_a = math.cos(angle_rad) * scale
        sin_a = math.sin(angle_rad) * scale

        # Affine matrix for torch affine_grid (maps output to input)
        # theta format: [[a, b, tx], [c, d, ty]]
        matrix = torch.tensor([
            [cos_a, -sin_a, tx],
            [sin_a, cos_a, ty],
        ], dtype=torch.float32)

        return matrix

    def _apply_affine(
        self,
        tensor: torch.Tensor,
        matrix: torch.Tensor,
    ) -> torch.Tensor:
        """Apply affine transformation using grid_sample."""
        b, c, h, w = tensor.shape

        # Expand matrix for batch
        theta = matrix.unsqueeze(0).expand(b, -1, -1).to(tensor.device, tensor.dtype)

        # Create sampling grid
        grid = F.affine_grid(theta, [b, c, h, w], align_corners=True)

        # Apply transformation
        return F.grid_sample(
            tensor,
            grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True,
        )

    def _apply_color(self, image: torch.Tensor) -> torch.Tensor:
        """Apply color augmentations."""
        # HSV shift
        if self.hsv_shift_amount > 0:
            image = self._random_hsv_shift(image, self.hsv_shift_amount)

        # Brightness
        if self.brightness_range > 0:
            brightness = 1.0 + random.uniform(-self.brightness_range, self.brightness_range)
            image = image * brightness

        # Contrast
        if self.contrast_range > 0:
            contrast = 1.0 + random.uniform(-self.contrast_range, self.contrast_range)
            mean = image.mean(dim=[-2, -1], keepdim=True)
            image = (image - mean) * contrast + mean

        return image

    def _random_hsv_shift(
        self,
        image: torch.Tensor,
        amount: float,
    ) -> torch.Tensor:
        """
        Apply random HSV color shift.

        Args:
            image: RGB image tensor (B, 3, H, W) in [0, 1].
            amount: Shift amount (0-1).

        Returns:
            Color-shifted image.
        """
        # Convert RGB to HSV
        hsv = self._rgb_to_hsv(image)

        # Apply random shifts
        h_shift = random.uniform(-amount, amount)
        s_shift = random.uniform(-amount, amount)
        v_shift = random.uniform(-amount, amount)

        # Shift channels
        hsv[:, 0, :, :] = (hsv[:, 0, :, :] + h_shift) % 1.0
        hsv[:, 1, :, :] = torch.clamp(hsv[:, 1, :, :] + s_shift, 0, 1)
        hsv[:, 2, :, :] = torch.clamp(hsv[:, 2, :, :] + v_shift, 0, 1)

        # Convert back to RGB
        return self._hsv_to_rgb(hsv)

    @staticmethod
    def _rgb_to_hsv(rgb: torch.Tensor) -> torch.Tensor:
        """Convert RGB to HSV color space."""
        r, g, b = rgb[:, 0, :, :], rgb[:, 1, :, :], rgb[:, 2, :, :]

        max_val, max_idx = rgb.max(dim=1)
        min_val = rgb.min(dim=1)[0]
        diff = max_val - min_val

        # Value
        v = max_val

        # Saturation
        s = torch.where(max_val > 0, diff / (max_val + 1e-8), torch.zeros_like(max_val))

        # Hue
        h = torch.zeros_like(max_val)

        # When max is R
        mask_r = (max_idx == 0) & (diff > 0)
        h[mask_r] = ((g[mask_r] - b[mask_r]) / (diff[mask_r] + 1e-8)) % 6

        # When max is G
        mask_g = (max_idx == 1) & (diff > 0)
        h[mask_g] = (b[mask_g] - r[mask_g]) / (diff[mask_g] + 1e-8) + 2

        # When max is B
        mask_b = (max_idx == 2) & (diff > 0)
        h[mask_b] = (r[mask_b] - g[mask_b]) / (diff[mask_b] + 1e-8) + 4

        h = h / 6.0  # Normalize to [0, 1]
        h = h % 1.0  # Handle negative values

        return torch.stack([h, s, v], dim=1)

    @staticmethod
    def _hsv_to_rgb(hsv: torch.Tensor) -> torch.Tensor:
        """Convert HSV to RGB color space."""
        h, s, v = hsv[:, 0, :, :], hsv[:, 1, :, :], hsv[:, 2, :, :]

        h = h * 6.0
        i = h.floor()
        f = h - i
        p = v * (1 - s)
        q = v * (1 - s * f)
        t = v * (1 - s * (1 - f))

        i = i.long() % 6

        # Build RGB based on sector
        rgb = torch.zeros_like(hsv)

        mask0 = i == 0
        mask1 = i == 1
        mask2 = i == 2
        mask3 = i == 3
        mask4 = i == 4
        mask5 = i == 5

        rgb[:, 0, :, :] = torch.where(mask0 | mask5, v, torch.where(mask1, q, torch.where(mask4, t, p)))
        rgb[:, 1, :, :] = torch.where(mask0, t, torch.where(mask1 | mask2, v, torch.where(mask3, q, p)))
        rgb[:, 2, :, :] = torch.where(mask2, t, torch.where(mask3 | mask4, v, torch.where(mask5, q, p)))

        return rgb


class SimpleAugmentation:
    """
    Simple augmentation for validation/inference.

    Only applies deterministic transformations without randomness.
    """

    def __init__(self, target_size: int = 256) -> None:
        self.target_size = target_size

    def __call__(
        self,
        image: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Apply deterministic transformations only."""
        # Just ensure correct size
        if image.shape[-1] != self.target_size:
            image = F.interpolate(
                image.unsqueeze(0) if image.dim() == 3 else image,
                size=(self.target_size, self.target_size),
                mode='bilinear',
                align_corners=True,
            )
            if image.dim() == 4 and image.shape[0] == 1:
                image = image.squeeze(0)

        if mask is not None and mask.shape[-1] != self.target_size:
            mask = F.interpolate(
                mask.unsqueeze(0) if mask.dim() == 3 else mask,
                size=(self.target_size, self.target_size),
                mode='bilinear',
                align_corners=True,
            )
            if mask.dim() == 4 and mask.shape[0] == 1:
                mask = mask.squeeze(0)

        return image, mask
