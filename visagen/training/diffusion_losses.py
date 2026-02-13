"""
Diffusion-specific loss functions.

This module provides loss functions specifically designed for
training the DiffusionAutoEncoder model, including texture
consistency and perceptual losses.

Example:
    >>> from visagen.training.diffusion_losses import DiffusionLoss
    >>> loss_fn = DiffusionLoss()
    >>> total, losses = loss_fn(pred, target)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextureConsistencyLoss(nn.Module):
    """
    Ensures texture details are preserved during generation.

    Uses Gram matrix comparison for style consistency, which
    captures the correlation between features at different
    positions, encoding texture information.

    Args:
        layers: VGG layer indices to extract features from.
            Default uses layers that capture different texture scales.

    Example:
        >>> loss_fn = TextureConsistencyLoss()
        >>> loss = loss_fn(pred_image, target_image)
    """

    def __init__(self, layers: list[int] | None = None) -> None:
        super().__init__()
        self.layers = layers or [4, 9, 16, 23]  # VGG-19 layers

        # Lazy load VGG to avoid import overhead
        self._vgg = None

        # Register ImageNet normalization constants
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def _get_vgg(self, device: torch.device) -> nn.Module:
        """Lazy load VGG-19 model."""
        if self._vgg is None:
            try:
                from torchvision.models import VGG19_Weights, vgg19

                try:
                    vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
                except Exception:
                    # Offline/sandbox environments may block weight downloads.
                    vgg = vgg19(weights=None).features
                vgg = vgg.eval().to(device)
                for p in vgg.parameters():
                    p.requires_grad = False
                self._vgg = vgg
            except ImportError:
                raise ImportError(
                    "torchvision is required for TextureConsistencyLoss. "
                    "Install with: pip install torchvision"
                )
        return self._vgg.to(device)

    def gram_matrix(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute Gram matrix for style representation.

        The Gram matrix captures correlations between features,
        encoding texture information independent of spatial layout.

        Args:
            x: Feature tensor (B, C, H, W).

        Returns:
            Gram matrix (B, C, C).
        """
        B, C, H, W = x.shape
        features = x.view(B, C, H * W)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram / (C * H * W)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute texture consistency loss.

        Args:
            pred: Predicted image (B, 3, H, W) in [-1, 1].
            target: Target image (B, 3, H, W) in [-1, 1].

        Returns:
            Scalar texture loss.
        """
        vgg = self._get_vgg(pred.device)

        # Normalize to ImageNet range [0, 1] then apply normalization
        pred_norm = (pred * 0.5 + 0.5 - self.mean.to(pred.device)) / self.std.to(
            pred.device
        )
        target_norm = (target * 0.5 + 0.5 - self.mean.to(pred.device)) / self.std.to(
            pred.device
        )

        loss = torch.tensor(0.0, device=pred.device)
        x_pred, x_target = pred_norm, target_norm

        for i, layer in enumerate(vgg):
            x_pred = layer(x_pred)
            x_target = layer(x_target)

            if i in self.layers:
                gram_pred = self.gram_matrix(x_pred)
                gram_target = self.gram_matrix(x_target)
                loss = loss + F.mse_loss(gram_pred, gram_target)

        return loss / len(self.layers)


class PerceptualTextureLoss(nn.Module):
    """
    Combined perceptual and texture loss.

    Combines L1 feature matching loss with Gram matrix texture
    loss for balanced structure and texture preservation.

    Args:
        layers: VGG layer indices for feature extraction.
        texture_weight: Weight for texture loss (default: 1.0).
        perceptual_weight: Weight for perceptual loss (default: 1.0).
    """

    def __init__(
        self,
        layers: list[int] | None = None,
        texture_weight: float = 1.0,
        perceptual_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.layers = layers or [4, 9, 16, 23]
        self.texture_weight = texture_weight
        self.perceptual_weight = perceptual_weight

        self._vgg = None

        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def _get_vgg(self, device: torch.device) -> nn.Module:
        """Lazy load VGG-19 model."""
        if self._vgg is None:
            from torchvision.models import VGG19_Weights, vgg19

            try:
                vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
            except Exception:
                # Offline/sandbox environments may block weight downloads.
                vgg = vgg19(weights=None).features
            vgg = vgg.eval().to(device)
            for p in vgg.parameters():
                p.requires_grad = False
            self._vgg = vgg
        return self._vgg.to(device)

    def gram_matrix(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Gram matrix."""
        B, C, H, W = x.shape
        features = x.view(B, C, H * W)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram / (C * H * W)

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Compute combined perceptual and texture loss.

        Args:
            pred: Predicted image (B, 3, H, W) in [-1, 1].
            target: Target image (B, 3, H, W) in [-1, 1].

        Returns:
            Tuple of (total_loss, loss_dict).
        """
        vgg = self._get_vgg(pred.device)

        pred_norm = (pred * 0.5 + 0.5 - self.mean.to(pred.device)) / self.std.to(
            pred.device
        )
        target_norm = (target * 0.5 + 0.5 - self.mean.to(pred.device)) / self.std.to(
            pred.device
        )

        perceptual_loss = torch.tensor(0.0, device=pred.device)
        texture_loss = torch.tensor(0.0, device=pred.device)

        x_pred, x_target = pred_norm, target_norm

        for i, layer in enumerate(vgg):
            x_pred = layer(x_pred)
            x_target = layer(x_target)

            if i in self.layers:
                # Perceptual loss (feature matching)
                perceptual_loss = perceptual_loss + F.l1_loss(x_pred, x_target)

                # Texture loss (Gram matrix)
                gram_pred = self.gram_matrix(x_pred)
                gram_target = self.gram_matrix(x_target)
                texture_loss = texture_loss + F.mse_loss(gram_pred, gram_target)

        perceptual_loss = perceptual_loss / len(self.layers)
        texture_loss = texture_loss / len(self.layers)

        total = (
            self.perceptual_weight * perceptual_loss
            + self.texture_weight * texture_loss
        )

        losses = {
            "perceptual": perceptual_loss,
            "texture": texture_loss,
            "total": total,
        }

        return total, losses


class DiffusionLoss(nn.Module):
    """
    Combined loss for Diffusion AutoEncoder training.

    Combines reconstruction, texture consistency, and optional
    perceptual losses for training the hybrid encoder.

    Args:
        reconstruction_weight: Weight for L1 reconstruction loss.
        texture_weight: Weight for texture consistency loss.
        perceptual_weight: Weight for perceptual loss (0 to disable).

    Example:
        >>> loss_fn = DiffusionLoss()
        >>> total, losses = loss_fn(pred, target)
        >>> print(losses.keys())
        dict_keys(['reconstruction', 'texture', 'total'])
    """

    def __init__(
        self,
        reconstruction_weight: float = 10.0,
        texture_weight: float = 5.0,
        perceptual_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.reconstruction_weight = reconstruction_weight
        self.texture_weight = texture_weight
        self.perceptual_weight = perceptual_weight

        if texture_weight > 0 or perceptual_weight > 0:
            self.texture_loss = TextureConsistencyLoss()
        else:
            self.texture_loss = None

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Compute combined diffusion loss.

        Args:
            pred: Predicted image (B, 3, H, W).
            target: Target image (B, 3, H, W).

        Returns:
            Tuple of (total_loss, loss_dict).
        """
        losses: dict[str, torch.Tensor] = {}

        # L1 reconstruction
        losses["reconstruction"] = F.l1_loss(pred, target)

        total = self.reconstruction_weight * losses["reconstruction"]

        # Texture consistency
        if self.texture_loss is not None and self.texture_weight > 0:
            losses["texture"] = self.texture_loss(pred, target)
            total = total + self.texture_weight * losses["texture"]

        losses["total"] = total
        return total, losses
