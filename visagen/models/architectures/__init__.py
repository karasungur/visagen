"""
Model architectures for Visagen.

This module contains various face swapping model architectures
ported from DeepFaceLab legacy codebase to modern PyTorch.

Available Architectures:
    - DFArchi: Direct Face architecture with separate src/dst decoders
    - LIAEArchi: Lightweight Inter-AB-B with shared decoder
    - AMPArchi: Amplified Morphable Portrait with morph control
"""

from visagen.models.architectures.amp import AMPArchi, morph_code
from visagen.models.architectures.df import DFArchi
from visagen.models.architectures.liae import LIAEArchi

__all__ = [
    "DFArchi",
    "LIAEArchi",
    "AMPArchi",
    "morph_code",
]
