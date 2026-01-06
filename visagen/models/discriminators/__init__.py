"""
Discriminator models for GAN training.

Provides PatchGAN discriminators for adversarial training:
- PatchDiscriminator: Simple patch-based discriminator
- UNetPatchDiscriminator: U-Net based with multi-scale outputs
- MultiScaleDiscriminator: Multiple discriminators at different scales
- TemporalDiscriminator: 3D Conv for temporal consistency
- TemporalPatchDiscriminator: Combined temporal + spatial
"""

from visagen.models.discriminators.patch_discriminator import (
    MultiScaleDiscriminator,
    PatchDiscriminator,
    ResidualBlock,
    UNetPatchDiscriminator,
)
from visagen.models.discriminators.temporal_discriminator import (
    LightweightTemporalDiscriminator,
    ResidualBlock3D,
    TemporalDiscriminator,
    TemporalPatchDiscriminator,
)

__all__ = [
    "PatchDiscriminator",
    "UNetPatchDiscriminator",
    "MultiScaleDiscriminator",
    "ResidualBlock",
    "TemporalDiscriminator",
    "TemporalPatchDiscriminator",
    "LightweightTemporalDiscriminator",
    "ResidualBlock3D",
]
