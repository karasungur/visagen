"""Tests for BatchTab helper logic."""

from visagen.gui.batch import BatchItem, BatchStatus
from visagen.gui.tabs.batch import BatchTab


def test_pending_merge_requires_checkpoint_detects_merge_item() -> None:
    items = [
        BatchItem(operation="extract", status=BatchStatus.PENDING),
        BatchItem(operation="merge", status=BatchStatus.PENDING),
    ]

    assert BatchTab._pending_merge_requires_checkpoint(items) is True


def test_pending_merge_requires_checkpoint_ignores_non_merge_items() -> None:
    items = [
        BatchItem(operation="extract", status=BatchStatus.PENDING),
        BatchItem(operation="extract", status=BatchStatus.RUNNING),
    ]

    assert BatchTab._pending_merge_requires_checkpoint(items) is False
