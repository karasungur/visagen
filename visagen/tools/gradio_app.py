"""
Gradio Web Interface for Visagen.

Provides a user-friendly web UI for:
- Training model configuration and execution
- Single image face swap inference
- Face extraction from images/videos
- Color transfer and blending demos
- Application settings

Usage:
    visagen-gui --port 7860 --share
"""

import argparse
import sys
from typing import TYPE_CHECKING

from visagen.gui.i18n import I18n
from visagen.gui.state.app_state import AppState
from visagen.gui.tabs.benchmark import BenchmarkTab
from visagen.gui.tabs.export import ExportTab
from visagen.gui.tabs.extract import ExtractTab
from visagen.gui.tabs.faceset_tools import FacesetToolsTab
from visagen.gui.tabs.inference import InferenceTab
from visagen.gui.tabs.interactive_merge import InteractiveMergeTab
from visagen.gui.tabs.merge import MergeTab
from visagen.gui.tabs.postprocess import PostprocessTab
from visagen.gui.tabs.settings import SettingsTab
from visagen.gui.tabs.sort import SortTab
from visagen.gui.tabs.training import TrainingTab
from visagen.gui.tabs.video_tools import VideoToolsTab

# Check gradio availability
try:
    import gradio as gr

    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False

if TYPE_CHECKING:
    pass


def create_app() -> "gr.Blocks":
    """Create Gradio application using modular architecture."""
    if not GRADIO_AVAILABLE:
        raise ImportError(
            "Gradio is required. Install with: pip install 'visagen[gui]'"
        )

    # Initialize centralized state and i18n
    state = AppState.create()
    i18n = I18n()

    with gr.Blocks(
        title="Visagen - Face Swapping Framework",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown("# Visagen")
        gr.Markdown("Modern Face Swapping Framework with PyTorch Lightning")

        # Instantiate and create modular tabs
        TrainingTab(state, i18n).create()
        InferenceTab(state, i18n).create()
        ExtractTab(state, i18n).create()
        MergeTab(state, i18n).create()
        InteractiveMergeTab(state, i18n).create()
        SortTab(state, i18n).create()
        ExportTab(state, i18n).create()
        VideoToolsTab(state, i18n).create()
        FacesetToolsTab(state, i18n).create()
        PostprocessTab(state, i18n).create()
        BenchmarkTab(state, i18n).create()
        SettingsTab(state, i18n).create()

        gr.Markdown("---")
        gr.Markdown(
            "Made with PyTorch Lightning | "
            "[GitHub](https://github.com/karasungur/visagen)"
        )

    # Register cleanup on close if possible (Gradio doesn't have a direct close hook per se,
    # but we rely on process termination in AppState if needed manually)
    return demo


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Visagen Gradio Web Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Server host (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Server port (default: 7860)",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public shareable link",
    )
    parser.add_argument(
        "--auth",
        type=str,
        default=None,
        help="Username:password for basic auth",
    )

    return parser.parse_args()


def main() -> int:
    """CLI entry point for Gradio app."""
    if not GRADIO_AVAILABLE:
        print("Error: Gradio is required for the GUI.")
        print("Install with: pip install 'visagen[gui]'")
        return 1

    args = parse_args()

    print("=" * 50)
    print("VISAGEN WEB INTERFACE")
    print("=" * 50)
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Share: {args.share}")
    print("=" * 50)

    try:
        app = create_app()

        # Parse auth if provided
        auth = None
        if args.auth:
            parts = args.auth.split(":", 1)
            if len(parts) == 2:
                auth = (parts[0], parts[1])

        app.launch(
            server_name=args.host,
            server_port=args.port,
            share=args.share,
            auth=auth,
        )
    except KeyboardInterrupt:
        print("\nStopping...")
        return 0
    except Exception as e:
        print(f"\nError: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
