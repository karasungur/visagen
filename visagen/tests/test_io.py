"""Tests for secure IO utilities."""

import json
from pathlib import Path

from visagen.utils.io import read_json_locked, write_json_locked


def test_write_read_json_locked(tmp_path: Path):
    """Test basic write and read with locking."""
    test_file = tmp_path / "test.json"
    data = {"key": "value", "count": 42}

    # Test write
    assert write_json_locked(test_file, data) is True
    assert test_file.exists()

    # Test read
    read_data = read_json_locked(test_file)
    assert read_data == data


def test_read_nonexistent_file():
    """Test reading a file that doesn't exist."""
    assert read_json_locked("nonexistent_file_12345.json") is None


def test_read_invalid_json(tmp_path: Path):
    """Test reading an invalid JSON file."""
    test_file = tmp_path / "invalid.json"
    test_file.write_text("not a json")
    assert read_json_locked(test_file) is None


def test_write_atomic(tmp_path: Path):
    """Test that writing is atomic (via temp file)."""
    test_file = tmp_path / "atomic.json"
    data = {"a": 1}
    assert write_json_locked(test_file, data) is True

    # Check that it actually wrote valid JSON
    with open(test_file) as f:
        assert json.load(f) == data
