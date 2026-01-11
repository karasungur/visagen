"""System monitoring components for RAM, VRAM, and GPU usage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import gradio as gr

from .base import BaseComponent, ComponentConfig

if TYPE_CHECKING:
    from visagen.gui.i18n import I18n


@dataclass
class SystemMonitorConfig(ComponentConfig):
    """Configuration for system monitor."""

    show_ram: bool = True
    show_vram: bool = True
    show_gpu_util: bool = True
    compact: bool = False


class SystemMonitor(BaseComponent):
    """
    Real-time system resource monitor.

    Displays:
    - RAM usage (used/total GB, percentage)
    - VRAM usage (if CUDA available)
    - GPU utilization (if available)
    """

    def __init__(
        self,
        config: SystemMonitorConfig,
        i18n: I18n,
    ) -> None:
        super().__init__(config, i18n)
        self.monitor_config = config

    def build(self) -> gr.HTML:
        """Build system monitor HTML component."""
        return gr.HTML(
            value=self.get_stats_html(),
            elem_id=self.config.get_elem_id(),
            elem_classes=["system-monitor", *self.config.elem_classes],
        )

    def get_stats(self) -> dict:
        """
        Collect current system statistics.

        Returns:
            Dictionary with RAM, VRAM, and GPU stats.
        """
        stats = {
            "ram_percent": 0.0,
            "ram_used_gb": 0.0,
            "ram_total_gb": 0.0,
            "vram_percent": 0.0,
            "vram_used_gb": 0.0,
            "vram_total_gb": 0.0,
            "gpu_util": 0.0,
            "gpu_name": "N/A",
            "cuda_available": False,
        }

        # RAM stats
        try:
            import psutil

            mem = psutil.virtual_memory()
            stats["ram_percent"] = mem.percent
            stats["ram_used_gb"] = mem.used / (1024**3)
            stats["ram_total_gb"] = mem.total / (1024**3)
        except ImportError:
            pass

        # GPU/VRAM stats
        try:
            import torch

            if torch.cuda.is_available():
                stats["cuda_available"] = True
                stats["gpu_name"] = torch.cuda.get_device_name(0)

                # VRAM
                allocated = torch.cuda.memory_allocated(0)
                total = torch.cuda.get_device_properties(0).total_memory
                stats["vram_used_gb"] = allocated / (1024**3)
                stats["vram_total_gb"] = total / (1024**3)
                stats["vram_percent"] = (allocated / total) * 100 if total > 0 else 0

                # GPU utilization (requires pynvml)
                try:
                    import pynvml

                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    stats["gpu_util"] = util.gpu
                    pynvml.nvmlShutdown()
                except Exception:
                    pass
        except ImportError:
            pass

        return stats

    def get_stats_html(self) -> str:
        """
        Generate HTML representation of system stats.

        Returns:
            HTML string with styled stats display.
        """
        stats = self.get_stats()

        if self.monitor_config.compact:
            return self._render_compact(stats)
        return self._render_full(stats)

    def _render_full(self, stats: dict) -> str:
        """Render full stats view."""
        ram_bar = self._progress_bar(
            stats["ram_percent"],
            f"RAM: {stats['ram_used_gb']:.1f} / {stats['ram_total_gb']:.1f} GB",
            self._get_color(stats["ram_percent"]),
        )

        vram_section = ""
        if self.monitor_config.show_vram and stats["cuda_available"]:
            vram_bar = self._progress_bar(
                stats["vram_percent"],
                f"VRAM: {stats['vram_used_gb']:.1f} / {stats['vram_total_gb']:.1f} GB",
                self._get_color(stats["vram_percent"]),
            )
            gpu_info = f"""
            <div style="font-size: 11px; color: #64748b; margin-bottom: 4px;">
                🎮 {stats["gpu_name"]}
            </div>
            """
            vram_section = gpu_info + vram_bar

        gpu_util_section = ""
        if (
            self.monitor_config.show_gpu_util
            and stats["cuda_available"]
            and stats["gpu_util"] > 0
        ):
            gpu_util_section = self._progress_bar(
                stats["gpu_util"],
                f"GPU: {stats['gpu_util']:.0f}%",
                self._get_color(stats["gpu_util"]),
            )

        return f"""
        <div style="
            padding: 12px;
            background: #f8fafc;
            border-radius: 8px;
            border: 1px solid #e2e8f0;
        ">
            <div style="font-weight: 600; margin-bottom: 8px; color: #334155;">
                📊 System Monitor
            </div>
            {ram_bar}
            {vram_section}
            {gpu_util_section}
        </div>
        """

    def _render_compact(self, stats: dict) -> str:
        """Render compact stats view (single line)."""
        ram_text = f"RAM: {stats['ram_percent']:.0f}%"
        vram_text = (
            f"VRAM: {stats['vram_percent']:.0f}%" if stats["cuda_available"] else ""
        )
        gpu_text = f"GPU: {stats['gpu_util']:.0f}%" if stats["gpu_util"] > 0 else ""

        parts = [p for p in [ram_text, vram_text, gpu_text] if p]
        text = " | ".join(parts)

        return f"""
        <div style="
            display: inline-flex;
            gap: 8px;
            padding: 4px 12px;
            background: #f1f5f9;
            border-radius: 16px;
            font-size: 12px;
            color: #475569;
        ">
            📊 {text}
        </div>
        """

    def _progress_bar(self, percent: float, label: str, color: str) -> str:
        """Generate a small progress bar."""
        return f"""
        <div style="margin-bottom: 8px;">
            <div style="
                display: flex;
                justify-content: space-between;
                font-size: 11px;
                color: #64748b;
                margin-bottom: 2px;
            ">
                <span>{label}</span>
                <span>{percent:.1f}%</span>
            </div>
            <div style="
                width: 100%;
                height: 8px;
                background: #e2e8f0;
                border-radius: 4px;
                overflow: hidden;
            ">
                <div style="
                    width: {min(100, percent)}%;
                    height: 100%;
                    background: {color};
                    border-radius: 4px;
                "></div>
            </div>
        </div>
        """

    def _get_color(self, percent: float) -> str:
        """Get color based on usage percentage."""
        if percent >= 90:
            return "#ef4444"  # Red
        elif percent >= 70:
            return "#f59e0b"  # Yellow/Orange
        else:
            return "#22c55e"  # Green


def get_system_stats_text() -> str:
    """
    Get system stats as plain text (for status bars, etc.)

    Returns:
        Simple text representation of system stats.
    """
    try:
        import psutil

        ram = psutil.virtual_memory()
        ram_text = f"RAM: {ram.percent:.0f}%"
    except ImportError:
        ram_text = "RAM: N/A"

    vram_text = ""
    try:
        import torch

        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0)
            total = torch.cuda.get_device_properties(0).total_memory
            vram_percent = (allocated / total) * 100 if total > 0 else 0
            vram_text = f" | VRAM: {vram_percent:.0f}%"
    except ImportError:
        pass

    return f"{ram_text}{vram_text}"
