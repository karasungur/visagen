"""Tests for dataset trash manager."""

from pathlib import Path

from visagen.tools.dataset_trash import list_batches, move_to_trash, undo_last_batch


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
    assert not f1.exists()
    assert not f2.exists()
    assert batch.manifest_path.exists()

    batches = list_batches(dataset)
    assert batch.batch_id in batches

    restored = undo_last_batch(dataset)
    assert restored.batch_id == batch.batch_id
    assert restored.restored == 2
    assert f1.exists()
    assert f2.exists()
