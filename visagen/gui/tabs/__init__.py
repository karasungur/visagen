"""GUI tabs package with lazy-loaded tab classes."""

from importlib import import_module
from typing import Any

from .base import BaseTab, TabProtocol

_TAB_MODULES: dict[str, str] = {
    "BatchTab": "batch",
    "CompareTab": "compare",
    "ExportTab": "export",
    "ExtractTab": "extract",
    "FacesetToolsTab": "faceset_tools",
    "InferenceTab": "inference",
    "InteractiveMergeTab": "interactive_merge",
    "MergeTab": "merge",
    "PostprocessTab": "postprocess",
    "SettingsTab": "settings",
    "SortTab": "sort",
    "TrainingTab": "training",
    "VideoToolsTab": "video_tools",
    "WizardTab": "wizard",
}


def __getattr__(name: str) -> Any:
    """Lazily resolve tab classes to avoid eager optional GUI imports."""
    if name in _TAB_MODULES:
        module = import_module(f"{__name__}.{_TAB_MODULES[name]}")
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "BaseTab",
    "BatchTab",
    "CompareTab",
    "ExportTab",
    "ExtractTab",
    "FacesetToolsTab",
    "InferenceTab",
    "InteractiveMergeTab",
    "MergeTab",
    "PostprocessTab",
    "SettingsTab",
    "SortTab",
    "TabProtocol",
    "TrainingTab",
    "VideoToolsTab",
    "WizardTab",
]
