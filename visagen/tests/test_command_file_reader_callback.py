"""Tests for live command callback parameter synchronization."""

from __future__ import annotations


class _TrainerStub:
    def __init__(self, step: int = 100) -> None:
        self.global_step = step


class _TemporalLossStub:
    def __init__(self, weight: float = 1.0) -> None:
        self.weight = weight


class _ModuleStub:
    def __init__(self) -> None:
        self.temporal_consistency_weight = 1.0
        self.temporal_consistency_loss: _TemporalLossStub | None = _TemporalLossStub(
            weight=1.0
        )


def test_temporal_weight_update_syncs_loss_object() -> None:
    """Updating temporal_consistency_weight should also update loss.weight."""
    from visagen.training.callbacks import CommandFileReaderCallback

    callback = CommandFileReaderCallback(command_file="unused.json")
    trainer = _TrainerStub()
    module = _ModuleStub()

    callback._update_module_param(
        trainer=trainer,  # type: ignore[arg-type]
        pl_module=module,  # type: ignore[arg-type]
        key="temporal_consistency_weight",
        value=2.5,
    )

    assert module.temporal_consistency_weight == 2.5
    assert module.temporal_consistency_loss is not None
    assert module.temporal_consistency_loss.weight == 2.5


def test_temporal_weight_update_without_loss_object_is_safe() -> None:
    """Callback should not fail when temporal loss object is missing."""
    from visagen.training.callbacks import CommandFileReaderCallback

    callback = CommandFileReaderCallback(command_file="unused.json")
    trainer = _TrainerStub()
    module = _ModuleStub()
    module.temporal_consistency_loss = None

    callback._update_module_param(
        trainer=trainer,  # type: ignore[arg-type]
        pl_module=module,  # type: ignore[arg-type]
        key="temporal_consistency_weight",
        value=3.0,
    )

    assert module.temporal_consistency_weight == 3.0
