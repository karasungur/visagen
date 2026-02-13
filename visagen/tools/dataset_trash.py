"""
Dataset trash manager with batch undo support.

All delete operations are implemented as moves to a managed trash directory
with manifest entries. Batches can be restored in reverse order.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import portalocker


@dataclass
class TrashBatch:
    """Result information for a trash move operation."""

    batch_id: str
    count: int
    trash_dir: Path
    manifest_path: Path
    count_moved: int = 0
    count_missing: int = 0
    count_failed: int = 0
    errors: list[str] = field(default_factory=list)


@dataclass
class UndoResult:
    """Result information for an undo restore operation."""

    batch_id: str
    restored: int
    skipped: int
    failed: int = 0
    errors: list[str] = field(default_factory=list)


def _now_utc_iso() -> str:
    """Get current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat()


def resolve_collision_path(path: Path) -> Path:
    """Resolve destination collision by appending deterministic suffix."""
    if not path.exists():
        return path

    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    i = 1
    while True:
        candidate = parent / f"{stem}_restored_{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1


def _resolve_restore_collision(path: Path) -> Path:
    """Backward-compatible internal alias."""
    return resolve_collision_path(path)


def _read_manifest(manifest_path: Path) -> list[dict]:
    """Read manifest JSONL entries."""
    if not manifest_path.exists():
        return []
    entries: list[dict] = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except Exception:
            continue
    return entries


def _append_manifest(manifest_path: Path, entries: list[dict]) -> None:
    """Append entries to JSONL manifest."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = manifest_path.with_suffix(f"{manifest_path.suffix}.lock")
    with portalocker.Lock(str(lock_path), mode="a+", timeout=30):
        with manifest_path.open("a", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=True) + "\n")


def _rewrite_manifest(manifest_path: Path, entries: list[dict]) -> None:
    """Rewrite JSONL manifest with lock protection."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = manifest_path.with_suffix(f"{manifest_path.suffix}.lock")
    with portalocker.Lock(str(lock_path), mode="a+", timeout=30):
        with manifest_path.open("w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=True) + "\n")


def move_to_trash(
    paths: list[Path],
    dataset_root: Path,
    reason: str = "user-delete",
) -> TrashBatch:
    """
    Move files to managed trash and record manifest.

    Args:
        paths: Files to move.
        dataset_root: Dataset root directory.
        reason: Reason label for manifest entries.

    Returns:
        TrashBatch metadata.
    """
    dataset_root = Path(dataset_root)
    trash_root = dataset_root / ".visagen_trash"
    batch_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid4().hex[:8]
    )
    batch_dir = trash_root / batch_id
    manifest_path = trash_root / "manifest.jsonl"

    batch_dir.mkdir(parents=True, exist_ok=True)
    timestamp = _now_utc_iso()
    moved_entries: list[dict] = []
    moved_count = 0
    missing_count = 0
    failed_count = 0
    errors: list[str] = []

    for src in paths:
        src = Path(src)
        if not src.exists():
            missing_count += 1
            continue
        dst = batch_dir / src.name
        if dst.exists():
            dst = resolve_collision_path(dst)
        try:
            src.rename(dst)
            moved_count += 1
            moved_entries.append(
                {
                    "batch_id": batch_id,
                    "timestamp": timestamp,
                    "reason": reason,
                    "src": str(src),
                    "dst": str(dst),
                }
            )
        except Exception as e:
            failed_count += 1
            errors.append(f"{src}: {e}")

    if moved_entries:
        _append_manifest(manifest_path, moved_entries)

    return TrashBatch(
        batch_id=batch_id,
        count=moved_count,
        trash_dir=batch_dir,
        manifest_path=manifest_path,
        count_moved=moved_count,
        count_missing=missing_count,
        count_failed=failed_count,
        errors=errors,
    )


def list_batches(dataset_root: Path) -> list[str]:
    """List batch IDs found in manifest, newest first."""
    manifest = Path(dataset_root) / ".visagen_trash" / "manifest.jsonl"
    entries = _read_manifest(manifest)
    seen: set[str] = set()
    ordered: list[str] = []
    for entry in reversed(entries):
        batch_id = entry.get("batch_id")
        if isinstance(batch_id, str) and batch_id not in seen:
            seen.add(batch_id)
            ordered.append(batch_id)
    return ordered


def undo_last_batch(dataset_root: Path) -> UndoResult:
    """
    Restore files from the last trash batch.

    Args:
        dataset_root: Dataset root directory.

    Returns:
        UndoResult with restored/skipped counts.
    """
    dataset_root = Path(dataset_root)
    manifest_path = dataset_root / ".visagen_trash" / "manifest.jsonl"
    entries = _read_manifest(manifest_path)
    if not entries:
        return UndoResult(batch_id="", restored=0, skipped=0)

    batches = list_batches(dataset_root)
    if not batches:
        return UndoResult(batch_id="", restored=0, skipped=0)
    batch_id = batches[0]
    restored = 0
    skipped = 0
    failed = 0
    errors: list[str] = []
    remaining_entries: list[dict] = []

    for entry in entries:
        if entry.get("batch_id") != batch_id:
            remaining_entries.append(entry)
            continue

        src_value = entry.get("src")
        dst_value = entry.get("dst")
        if not isinstance(src_value, str) or not isinstance(dst_value, str):
            skipped += 1
            continue

        src = Path(src_value)
        dst = Path(dst_value)
        if not dst.exists():
            skipped += 1
            continue
        try:
            src.parent.mkdir(parents=True, exist_ok=True)
            restore_target = resolve_collision_path(src)
            dst.rename(restore_target)
            restored += 1
        except Exception as e:
            failed += 1
            errors.append(f"{dst} -> {src}: {e}")
            # Keep failed entries in manifest for retry.
            remaining_entries.append(entry)

    _rewrite_manifest(manifest_path, remaining_entries)

    return UndoResult(
        batch_id=batch_id,
        restored=restored,
        skipped=skipped,
        failed=failed,
        errors=errors,
    )
