"""
Code Discriminator for true_face_power training.

Port of DeepFaceLab's code discriminator from Model_SAEHD to PyTorch.
Used to enforce source identity preservation in latent space.

The code discriminator distinguishes between:
- Destination codes (real) - should be classified as real
- Source codes (fake) - should be classified as fake after training

This forces the generator to produce source codes that look like
destination codes in the latent space, improving identity preservation.
"""

from typing import cast

import torch
import torch.nn as nn


class CodeDiscriminator(nn.Module):
    """
    Discriminator for latent codes to enforce source identity.

    Used with true_face_power training option to make swapped faces
    better preserve the source identity in latent space.

    Architecture:
        - 3x Conv2d layers with LeakyReLU
        - AdaptiveAvgPool2d to 1x1
        - Flatten + Linear to single output

    Args:
        in_ch: Input channels (typically ae_dims). Default: 256.
        hidden_ch: Hidden layer channels. Default: 256.
        num_layers: Number of conv layers. Default: 3.
        use_spectral_norm: Apply spectral normalization. Default: False.

    Example:
        >>> disc = CodeDiscriminator(in_ch=256)
        >>> code = torch.randn(4, 256, 8, 8)
        >>> output = disc(code)
        >>> output.shape
        torch.Size([4, 1])
    """

    def __init__(
        self,
        in_ch: int = 256,
        hidden_ch: int = 256,
        num_layers: int = 3,
        use_spectral_norm: bool = False,
    ) -> None:
        super().__init__()

        self.in_ch = in_ch
        self.hidden_ch = hidden_ch
        self.num_layers = num_layers

        layers: list[nn.Module] = []

        # First conv: in_ch -> hidden_ch
        conv = nn.Conv2d(in_ch, hidden_ch, kernel_size=3, padding=1)
        if use_spectral_norm:
            conv = nn.utils.spectral_norm(conv)
        layers.append(conv)
        layers.append(nn.LeakyReLU(0.1, inplace=True))

        # Middle conv layers: hidden_ch -> hidden_ch
        for _ in range(num_layers - 1):
            conv = nn.Conv2d(hidden_ch, hidden_ch, kernel_size=3, padding=1)
            if use_spectral_norm:
                conv = nn.utils.spectral_norm(conv)
            layers.append(conv)
            layers.append(nn.LeakyReLU(0.1, inplace=True))

        # Global average pooling and flatten
        layers.append(nn.AdaptiveAvgPool2d(1))
        layers.append(nn.Flatten())

        # Final linear layer
        linear = nn.Linear(hidden_ch, 1)
        if use_spectral_norm:
            linear = nn.utils.spectral_norm(linear)
        layers.append(linear)

        self.net = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.1, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.1, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, code: torch.Tensor) -> torch.Tensor:
        """
        Classify latent code as real or fake.

        Args:
            code: Latent code tensor (B, C, H, W).

        Returns:
            Discrimination score (B, 1). Higher = more real.
        """
        return cast(torch.Tensor, self.net(code))


def code_discriminator_loss(
    disc: CodeDiscriminator,
    src_code: torch.Tensor,
    dst_code: torch.Tensor,
    loss_type: str = "bce",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute code discriminator losses.

    Args:
        disc: CodeDiscriminator module.
        src_code: Source identity codes (should be fake).
        dst_code: Destination identity codes (should be real).
        loss_type: Loss type, "bce" or "hinge". Default: "bce".

    Returns:
        Tuple of (generator_loss, discriminator_loss).

    Example:
        >>> disc = CodeDiscriminator()
        >>> src_code = torch.randn(4, 256, 8, 8)
        >>> dst_code = torch.randn(4, 256, 8, 8)
        >>> g_loss, d_loss = code_discriminator_loss(disc, src_code, dst_code)
    """
    # Get discriminator outputs
    src_pred = disc(src_code)
    dst_pred = disc(dst_code.detach())

    if loss_type == "bce":
        # BCE loss
        criterion = nn.BCEWithLogitsLoss()
        ones = torch.ones_like(src_pred)
        zeros = torch.zeros_like(src_pred)

        # Generator loss: src should look real
        g_loss = criterion(src_pred, ones)

        # Discriminator loss: dst=real, src=fake
        d_loss_real = criterion(dst_pred, ones)
        d_loss_fake = criterion(disc(src_code.detach()), zeros)
        d_loss = 0.5 * (d_loss_real + d_loss_fake)

    elif loss_type == "hinge":
        # Hinge loss
        # Generator: maximize src_pred
        g_loss = -src_pred.mean()

        # Discriminator: hinge loss
        d_loss_real = torch.relu(1.0 - dst_pred).mean()
        d_loss_fake = torch.relu(1.0 + disc(src_code.detach())).mean()
        d_loss = 0.5 * (d_loss_real + d_loss_fake)

    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    return g_loss, d_loss
