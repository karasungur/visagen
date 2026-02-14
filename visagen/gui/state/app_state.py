"""Application state management."""

from __future__ import annotations

import queue
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class AppSettings:
    """Persisted application settings."""

    device: str = "auto"
    gpu_id: int = 0
    default_batch_size: int = 8
    num_workers: int = 4
    workspace_dir: str = "./workspace"
    locale: str = "en"

    # Extraction settings
    extraction_output_size: int = 512
    extraction_face_type: str = "whole_face"
    extraction_min_confidence: float = 0.5
    extraction_jpeg_quality: int = 95
    extraction_auto_mask: bool = True
    segmentation_batch_size: int = 8
    detector_warmup: bool = True

    # Interactive Merger defaults
    last_merger_session_path: str | None = None
    auto_load_last_session: bool = True
    merger_default_mode: str = "overlay"
    merger_default_mask_mode: str = "segmented"
    merger_default_color_transfer: str = "rct"
    merger_default_erode: int = 0
    merger_default_blur: int = 10

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "device": self.device,
            "gpu_id": self.gpu_id,
            "default_batch_size": self.default_batch_size,
            "num_workers": self.num_workers,
            "workspace_dir": self.workspace_dir,
            "locale": self.locale,
            "extraction_output_size": self.extraction_output_size,
            "extraction_face_type": self.extraction_face_type,
            "extraction_min_confidence": self.extraction_min_confidence,
            "extraction_jpeg_quality": self.extraction_jpeg_quality,
            "extraction_auto_mask": self.extraction_auto_mask,
            "segmentation_batch_size": self.segmentation_batch_size,
            "detector_warmup": self.detector_warmup,
            "last_merger_session_path": self.last_merger_session_path,
            "auto_load_last_session": self.auto_load_last_session,
            "merger_default_mode": self.merger_default_mode,
            "merger_default_mask_mode": self.merger_default_mask_mode,
            "merger_default_color_transfer": self.merger_default_color_transfer,
            "merger_default_erode": self.merger_default_erode,
            "merger_default_blur": self.merger_default_blur,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AppSettings:
        """Deserialize from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def save(self, path: Path | str) -> None:
        """Save settings to JSON file."""
        import json

        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path | str) -> AppSettings:
        """Load settings from JSON file."""
        import json

        path = Path(path)
        if not path.exists():
            return cls()
        data = json.loads(path.read_text())
        return cls.from_dict(data)


@dataclass
class ModelState:
    """Current model state."""

    model: Any = None  # Loaded PyTorch model
    model_path: str | None = None  # Path to current checkpoint
    device: str = "cpu"

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None

    def load(self, checkpoint_path: str, device: str = "auto") -> str:
        """
        Load model from checkpoint.

        Args:
            checkpoint_path: Path to .ckpt file.
            device: Target device ('auto', 'cuda', 'cpu').

        Returns:
            Status message.
        """
        if not checkpoint_path or not Path(checkpoint_path).exists():
            return "Error: Checkpoint file not found"

        try:
            import torch

            from visagen.training.dfl_module import DFLModule

            # Load model
            self.model = DFLModule.load_from_checkpoint(
                checkpoint_path,
                map_location="cpu",
            )
            self.model.eval()

            # Determine device
            if device == "auto":
                target_device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                target_device = device

            # Move to device
            if target_device == "cuda" and torch.cuda.is_available():
                self.model = self.model.cuda()
                self.device = "cuda"
            else:
                self.model = self.model.cpu()
                self.device = "cpu"

            self.model_path = checkpoint_path
            return f"Model loaded: {Path(checkpoint_path).name} ({self.device})"

        except Exception as e:
            self.model = None
            self.model_path = None
            return f"Error loading model: {e}"

    def unload(self) -> str:
        """Unload model and free memory."""
        if self.model is not None:
            del self.model
            self.model = None
            self.model_path = None

            # Clear GPU cache
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

            return "Model unloaded"
        return "No model loaded"


@dataclass
class ProcessState:
    """State of running subprocesses."""

    training: subprocess.Popen | None = None
    merge: subprocess.Popen | None = None
    sort: subprocess.Popen | None = None
    export: subprocess.Popen | None = None
    extract: subprocess.Popen | None = None
    video_tools: subprocess.Popen | None = None
    faceset_tools: subprocess.Popen | None = None
    benchmark: subprocess.Popen | None = None

    training_queue: queue.Queue = field(default_factory=queue.Queue)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def get(self, name: str) -> subprocess.Popen | None:
        """Get process handle by name."""
        with self._lock:
            return getattr(self, name, None)

    def is_running(self, name: str) -> bool:
        """Return True when the named process exists and is still alive."""
        proc = self.get(name)
        return proc is not None and proc.poll() is None

    def launch(
        self,
        name: str,
        cmd: list[str],
        **popen_kwargs: Any,
    ) -> subprocess.Popen | None:
        """
        Launch subprocess for named slot if there is no active process.

        Returns:
            New process handle, or None when a live process already exists.
        """
        with self._lock:
            current = getattr(self, name, None)
            if current is not None:
                if current.poll() is None:
                    return None
                # Clear stale exited handle before launching replacement.
                setattr(self, name, None)

            process = subprocess.Popen(cmd, **popen_kwargs)
            setattr(self, name, process)
            return process

    def clear_if(self, name: str, expected: subprocess.Popen | None) -> bool:
        """
        Clear process slot only when current handle matches `expected`.

        This prevents one caller from clearing a newer process that started later.
        """
        with self._lock:
            current = getattr(self, name, None)
            if current is expected:
                setattr(self, name, None)
                return True
            return False

    def terminate_all(self) -> None:
        """Terminate all running processes."""
        with self._lock:
            procs = {
                proc_name: getattr(self, proc_name)
                for proc_name in [
                    "training",
                    "merge",
                    "sort",
                    "export",
                    "extract",
                    "video_tools",
                    "faceset_tools",
                    "benchmark",
                ]
            }
            for proc_name in procs:
                setattr(self, proc_name, None)

        for proc in procs.values():
            if proc is None:
                continue
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

    def terminate(self, name: str) -> bool:
        """Terminate a specific process by name."""
        with self._lock:
            proc = getattr(self, name, None)
            if proc is None:
                return False
            setattr(self, name, None)

        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        return True


@dataclass
class AppState:
    """
    Central application state container.

    Passed to all tabs via dependency injection.
    Contains model state, process management, and settings.
    """

    settings: AppSettings = field(default_factory=AppSettings)
    model: ModelState = field(default_factory=ModelState)
    processes: ProcessState = field(default_factory=ProcessState)

    # Optional lazy-loaded components
    _restorer: Any = field(default=None, repr=False)
    _interactive_merger: Any = field(default=None, repr=False)

    @classmethod
    def create(cls, settings_path: Path | str | None = None) -> AppState:
        """
        Factory method to create AppState with optional settings loading.

        Args:
            settings_path: Optional path to settings JSON file.
                If omitted, tries `./settings.json`.

        Returns:
            Initialized AppState instance.
        """
        state = cls()

        if settings_path:
            path = Path(settings_path)
        else:
            path = Path("./settings.json")

        if path.exists():
            state.settings = AppSettings.load(path)

        return state

    def cleanup(self) -> None:
        """Clean up all resources."""
        self.processes.terminate_all()
        self.model.unload()
        self._restorer = None
        self._interactive_merger = None
