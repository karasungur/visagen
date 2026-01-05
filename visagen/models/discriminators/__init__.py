"""
Discriminator models for GAN training.

Provides PatchGAN discriminators for adversarial training:
- PatchDiscriminator: Simple patch-based discriminator
- UNetPatchDiscriminator: U-Net based with multi-scale outputs
- MultiScaleDiscriminator: Multiple discriminators at different scales
"""

from visagen.models.discriminators.patch_discriminator import (
    MultiScaleDiscriminator,
    PatchDiscriminator,
    ResidualBlock,
    UNetPatchDiscriminator,
)

__all__ = [
    "PatchDiscriminator",
    "UNetPatchDiscriminator",
    "MultiScaleDiscriminator",
    "ResidualBlock",
]
