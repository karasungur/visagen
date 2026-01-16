"""
Input/Output utilities for Visagen.

Provides secure, atomic, and thread-safe file operations using portalocker.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import portalocker

logger = logging.getLogger(__name__)


def read_json_locked(file_path: str | Path) -> dict[str, Any] | None:
    """
    Read a JSON file using a shared lock.

    Args:
        file_path: Path to the JSON file.

    Returns:
        Dictionary containing JSON data, or None if file doesn't exist or is invalid.
    """
    path = Path(file_path)
    if not path.exists():
        return None

    try:
        with portalocker.Lock(
            path, mode="r", timeout=5, flags=portalocker.LOCK_SH
        ) as f:
            data = json.load(f)
            if isinstance(data, dict):
                return data
            logger.warning(f"JSON data in {path} is not a dictionary")
            return None
    except (json.JSONDecodeError, portalocker.exceptions.LockException, OSError) as e:
        logger.warning(f"Failed to read locked JSON file {path}: {e}")
        return None


def write_json_locked(file_path: str | Path, data: dict[str, Any]) -> bool:
    """
    Write a JSON file atomically using an exclusive lock.

    Args:
        file_path: Path to the destination file.
        data: Data to be written as JSON.

    Returns:
        True if write was successful, False otherwise.
    """
    path = Path(file_path)
    try:
        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        # Write to a temporary file first, then rename for atomicity
        temp_path = path.with_suffix(".tmp")
        with portalocker.Lock(
            temp_path, mode="w", timeout=5, flags=portalocker.LOCK_EX
        ) as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())

        # Atomic rename (on POSIX systems)
        temp_path.replace(path)
        return True
    except (TypeError, portalocker.exceptions.LockException, OSError) as e:
        logger.error(f"Failed to write locked JSON file {path}: {e}")
        if "temp_path" in locals() and temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass
        return False
