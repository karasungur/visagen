"""
LRU Cache for Segmentation Results.

Provides caching mechanism to avoid redundant segmentation
computations for the same images.
"""

import hashlib
import threading
from collections import OrderedDict
from typing import Any, TypeVar

import numpy as np

T = TypeVar("T")


class LRUCache:
    """
    Generic Least Recently Used (LRU) cache.

    Automatically evicts the least recently used items when
    the maximum size is reached.

    Args:
        max_size: Maximum number of items to store. Default: 100.

    Example:
        >>> cache = LRUCache(max_size=10)
        >>> cache.put(image, result, "config_string")
        >>> cached = cache.get(image, "config_string")
    """

    def __init__(self, max_size: int = 100) -> None:
        self.max_size = max_size
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._lock = threading.RLock()

    def _compute_hash(self, image: np.ndarray, config_str: str = "") -> str:
        """
        Compute hash for image and config combination.

        Uses SHA-256 for secure and collision-resistant hashing.

        Args:
            image: Image array to hash.
            config_str: Optional configuration string to include.

        Returns:
            SHA-256 hash string.
        """
        # Use image bytes and shape for hashing
        image_bytes = image.tobytes()
        shape_bytes = str(image.shape).encode()
        combined = image_bytes + shape_bytes + config_str.encode()
        return hashlib.sha256(combined).hexdigest()

    def get(self, image: np.ndarray, config_str: str = "") -> Any | None:
        """
        Get cached value for image.

        Args:
            image: Image to lookup.
            config_str: Configuration string used during caching.

        Returns:
            Cached value if found, None otherwise.
        """
        key = self._compute_hash(image, config_str)
        with self._lock:
            if key in self._cache:
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                self._hits += 1
                return self._cache[key]
            self._misses += 1
            return None

    def put(self, image: np.ndarray, value: Any, config_str: str = "") -> None:
        """
        Store value in cache.

        Args:
            image: Image key.
            value: Value to cache.
            config_str: Configuration string.
        """
        key = self._compute_hash(image, config_str)

        with self._lock:
            # Evict if necessary
            while len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)

            self._cache[key] = value
            self._cache.move_to_end(key)

    def clear(self) -> None:
        """Clear all cached items."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    def __len__(self) -> int:
        """Return number of cached items."""
        return len(self._cache)

    def __contains__(self, key: str) -> bool:
        """Check if key is in cache."""
        return key in self._cache

    @property
    def hit_rate(self) -> float:
        """Get cache hit rate (0.0 to 1.0)."""
        with self._lock:
            total = self._hits + self._misses
            if total == 0:
                return 0.0
            return self._hits / total

    @property
    def stats(self) -> dict[str, int | float]:
        """Get cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0.0
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
            }


class SegmentationCache(LRUCache):
    """
    Specialized cache for segmentation results.

    Extends LRUCache with segmentation-specific optimizations
    like reduced memory footprint for mask storage.

    Args:
        max_size: Maximum number of cached results. Default: 100.
        store_masks_only: Store only masks, not full results. Default: False.

    Example:
        >>> cache = SegmentationCache(max_size=50)
        >>> result = segmenter.segment(image)
        >>> cache.put(image, result)
        >>> cached = cache.get(image)
    """

    def __init__(
        self,
        max_size: int = 100,
        store_masks_only: bool = False,
    ) -> None:
        super().__init__(max_size)
        self.store_masks_only = store_masks_only

    def put_result(
        self,
        image: np.ndarray,
        result: Any,
        return_soft_mask: bool = False,
        anti_alias: bool = False,
        threshold_mode: str = "fixed",
        threshold_value: float = 0.5,
    ) -> None:
        """
        Store segmentation result with automatic config string.

        Args:
            image: Input image.
            result: Segmentation result.
            return_soft_mask: Whether soft mask was requested.
            anti_alias: Whether anti-aliasing was applied.
            threshold_mode: Threshold mode used.
            threshold_value: Threshold value used.
        """
        config_str = (
            f"soft={return_soft_mask},"
            f"aa={anti_alias},"
            f"tm={threshold_mode},"
            f"tv={threshold_value:.2f}"
        )
        self.put(image, result, config_str)

    def get_result(
        self,
        image: np.ndarray,
        return_soft_mask: bool = False,
        anti_alias: bool = False,
        threshold_mode: str = "fixed",
        threshold_value: float = 0.5,
    ) -> Any | None:
        """
        Get cached segmentation result.

        Args:
            image: Input image.
            return_soft_mask: Whether soft mask is requested.
            anti_alias: Whether anti-aliasing is requested.
            threshold_mode: Threshold mode requested.
            threshold_value: Threshold value requested.

        Returns:
            Cached result if found, None otherwise.
        """
        config_str = (
            f"soft={return_soft_mask},"
            f"aa={anti_alias},"
            f"tm={threshold_mode},"
            f"tv={threshold_value:.2f}"
        )
        return self.get(image, config_str)


class ImageHashCache:
    """
    Fast image hash cache for duplicate detection.

    Uses perceptual hashing to detect similar images quickly
    without storing full image data.

    Args:
        max_size: Maximum number of hashes to store. Default: 1000.
    """

    def __init__(self, max_size: int = 1000) -> None:
        self.max_size = max_size
        self._hashes: OrderedDict[str, str] = OrderedDict()

    def compute_phash(self, image: np.ndarray, hash_size: int = 8) -> str:
        """
        Compute perceptual hash of image.

        Args:
            image: Input image.
            hash_size: Size of hash grid. Default: 8.

        Returns:
            Hex string of perceptual hash.
        """
        import cv2

        # Resize to small square
        small = cv2.resize(image, (hash_size + 1, hash_size))

        # Convert to grayscale if needed
        if len(small.shape) == 3:
            small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        # Compute gradient
        diff = small[:, 1:] > small[:, :-1]

        # Convert to hex
        return "".join(format(byte, "02x") for byte in np.packbits(diff.flatten()))

    def add(self, image: np.ndarray, identifier: str) -> None:
        """
        Add image hash with identifier.

        Args:
            image: Image to hash.
            identifier: Unique identifier for this image.
        """
        phash = self.compute_phash(image)

        while len(self._hashes) >= self.max_size:
            self._hashes.popitem(last=False)

        self._hashes[phash] = identifier

    def find_similar(self, image: np.ndarray, max_distance: int = 5) -> str | None:
        """
        Find similar image by hash.

        Args:
            image: Image to search for.
            max_distance: Maximum Hamming distance for match.

        Returns:
            Identifier of similar image if found, None otherwise.
        """
        target_hash = self.compute_phash(image)

        for stored_hash, identifier in self._hashes.items():
            if self._hamming_distance(target_hash, stored_hash) <= max_distance:
                return identifier

        return None

    @staticmethod
    def _hamming_distance(hash1: str, hash2: str) -> int:
        """Compute Hamming distance between two hex hash strings."""
        if len(hash1) != len(hash2):
            return max(len(hash1), len(hash2)) * 4

        distance = 0
        for c1, c2 in zip(hash1, hash2, strict=True):
            xor = int(c1, 16) ^ int(c2, 16)
            distance += bin(xor).count("1")

        return distance

    def clear(self) -> None:
        """Clear all stored hashes."""
        self._hashes.clear()
