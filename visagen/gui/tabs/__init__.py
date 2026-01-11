"""GUI Tabs package."""

from .base import BaseTab, TabProtocol
from .batch import BatchTab
from .compare import CompareTab
from .export import ExportTab
from .extract import ExtractTab
from .faceset_tools import FacesetToolsTab
from .inference import InferenceTab
from .interactive_merge import InteractiveMergeTab
from .merge import MergeTab
from .postprocess import PostprocessTab
from .settings import SettingsTab
from .sort import SortTab
from .training import TrainingTab
from .video_tools import VideoToolsTab
from .wizard import WizardTab

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
