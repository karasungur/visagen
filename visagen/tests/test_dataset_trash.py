"""Tests for dataset trash manager."""

from pathlib import Path

from visagen.tools.dataset_trash import (
    list_batches,
    move_to_trash,
    resolve_collision_path,
    undo_last_batch,
)


def test_move_to_trash_and_undo(tmp_path: Path):
    """Files moved to trash should be restorable via undo."""
    dataset = tmp_path / "aligned"
    dataset.mkdir(parents=True, exist_ok=True)

    f1 = dataset / "a.jpg"
    f2 = dataset / "b.jpg"
    f1.write_bytes(b"a")
    f2.write_bytes(b"b")

    batch = move_to_trash([f1, f2], dataset_root=dataset, reason="test-delete")

    assert batch.count == 2
    assert batch.count_moved == 2
    assert batch.count_missing == 0
    assert batch.count_failed == 0
    assert batch.errors == []
    assert not f1.exists()
    assert not f2.exists()
    assert batch.manifest_path.exists()

    batches = list_batches(dataset)
    assert batch.batch_id in batches

    restored = undo_last_batch(dataset)
    assert restored.batch_id == batch.batch_id
    assert restored.restored == 2
    assert restored.failed == 0
    assert restored.errors == []
    assert f1.exists()
    assert f2.exists()


def test_move_to_trash_counts_missing(tmp_path: Path):
    """Missing files should be tracked without failing the batch."""
    dataset = tmp_path / "aligned"
    dataset.mkdir(parents=True, exist_ok=True)

    existing = dataset / "keep.jpg"
    missing = dataset / "missing.jpg"
    existing.write_bytes(b"x")

    batch = move_to_trash(
        [existing, missing], dataset_root=dataset, reason="test-delete"
    )

    assert batch.count == 1
    assert batch.count_moved == 1
    assert batch.count_missing == 1
    assert batch.count_failed == 0
    assert batch.errors == []


def test_undo_failure_keeps_manifest_entry(tmp_path: Path, monkeypatch):
    """Failed restore should remain undoable on the next attempt."""
    dataset = tmp_path / "aligned"
    dataset.mkdir(parents=True, exist_ok=True)

    target = dataset / "x.jpg"
    target.write_bytes(b"x")

    batch = move_to_trash([target], dataset_root=dataset, reason="test-delete")
    assert batch.count_moved == 1

    original_rename = Path.rename

    def flaky_rename(self: Path, dst: Path) -> Path:
        # Fail only when restoring from trash back to dataset.
        if self.parent == batch.trash_dir:
            raise OSError("restore blocked")
        return original_rename(self, dst)

    monkeypatch.setattr(Path, "rename", flaky_rename)

    result = undo_last_batch(dataset)
    assert result.batch_id == batch.batch_id
    assert result.restored == 0
    assert result.failed == 1
    assert len(result.errors) == 1

    # Batch entry must remain in manifest for a future retry.
    assert batch.batch_id in list_batches(dataset)


def test_resolve_collision_path(tmp_path: Path):
    """Collision helper should generate deterministic incremented filename."""
    target = tmp_path / "dup.jpg"
    target.write_bytes(b"a")

    resolved = resolve_collision_path(target)
    assert resolved.name == "dup_restored_1.jpg"

    resolved.write_bytes(b"b")
    resolved_next = resolve_collision_path(target)
    assert resolved_next.name == "dup_restored_2.jpg"
