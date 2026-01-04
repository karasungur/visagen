"""
Random Noise Dataset for testing.

Generates random tensors for forward pass validation
without requiring actual image data.
"""

import torch
from torch.utils.data import Dataset
from typing import Tuple, Optional


class RandomNoiseDataset(Dataset):
    """
    Dataset that generates random noise tensors.

    Useful for testing model forward passes and training loops
    without needing actual data.

    Args:
        size: Number of samples in the dataset. Default: 1000.
        image_size: Height and width of generated images. Default: 256.
        channels: Number of image channels. Default: 3.
        seed: Random seed for reproducibility. Default: None.

    Example:
        >>> dataset = RandomNoiseDataset(size=100, image_size=256)
        >>> src, dst = dataset[0]
        >>> src.shape, dst.shape
        (torch.Size([3, 256, 256]), torch.Size([3, 256, 256]))
    """

    def __init__(
        self,
        size: int = 1000,
        image_size: int = 256,
        channels: int = 3,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.size = size
        self.image_size = image_size
        self.channels = channels
        self.seed = seed

        if seed is not None:
            torch.manual_seed(seed)

    def __len__(self) -> int:
        """Return dataset size."""
        return self.size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a random sample pair.

        Args:
            idx: Sample index (used for seeding if needed).

        Returns:
            Tuple of (source_image, target_image) as random tensors
            in range [-1, 1].
        """
        # Generate deterministic noise based on index for reproducibility
        generator = torch.Generator()
        if self.seed is not None:
            generator.manual_seed(self.seed + idx)
        else:
            generator.manual_seed(idx)

        # Source image (random noise)
        src = torch.rand(
            self.channels,
            self.image_size,
            self.image_size,
            generator=generator,
        )
        # Normalize to [-1, 1]
        src = src * 2 - 1

        # Target image (different random noise for now)
        dst = torch.rand(
            self.channels,
            self.image_size,
            self.image_size,
            generator=generator,
        )
        dst = dst * 2 - 1

        return src, dst
