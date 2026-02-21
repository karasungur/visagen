"""Main application factory."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    import gradio as gr

from visagen.gui.i18n import I18n
from visagen.gui.state import AppState
from visagen.gui.tabs import (
    BatchTab,
    CompareTab,
    ExportTab,
    ExtractTab,
    FacesetToolsTab,
    InferenceTab,
    InteractiveMergeTab,
    MergeTab,
    PostprocessTab,
    SettingsTab,
    SortTab,
    TrainingTab,
    VideoToolsTab,
    WizardTab,
)
from visagen.gui.tabs.mask_editor import MaskEditorTab
from visagen.gui.theme import create_visagen_theme, get_theme_css


def create_app(
    settings_path: Path | str | None = None,
    locale: str = "en",
    dark_mode: bool = False,
) -> gr.Blocks:
    """
    Create the Visagen Gradio application.

    Args:
        settings_path: Optional path to settings JSON file.
        locale: Initial locale code.
        dark_mode: Whether to use dark theme.

    Returns:
        Configured gr.Blocks application.
    """
    import gradio as gr

    from visagen.gui.theme import create_dark_theme

    settings_file = Path(settings_path) if settings_path else None

    # Initialize state and i18n
    state = AppState.create(settings_file)
    i18n = I18n(locale=locale)

    # Tab classes in DeepFaceLab workflow order
    # 1. Extract → 2. Sort → 3. Train → 4. Merge → 5. Export
    tab_classes = [
        WizardTab,  # Quick Start wizard for new users
        ExtractTab,  # 1. Video → Frames → Faces
        SortTab,  # 2. Filter/sort faces
        FacesetToolsTab,  # 2.5. Face set utilities
        MaskEditorTab,  # 2.6. Mask editing with LoRA fine-tuning
        TrainingTab,  # 3. Train model
        InferenceTab,  # 3.5. Single image test
        CompareTab,  # 3.6. Model comparison
        MergeTab,  # 4. Process video
        InteractiveMergeTab,  # 4.5. Interactive merging
        BatchTab,  # 4.6. Batch processing
        PostprocessTab,  # 5. Post-processing
        ExportTab,  # 6. Export model
        VideoToolsTab,  # Utilities
        SettingsTab,  # Settings (last)
    ]

    # Select theme based on mode
    theme = create_dark_theme() if dark_mode else create_visagen_theme()

    # Get custom CSS
    custom_css = get_theme_css()

    with gr.Blocks(
        title=i18n.t("app.title"),
    ) as app:
        # Store css for launch
        app._visagen_css = custom_css
        app._visagen_theme = theme

        # Header with branding
        with gr.Row(elem_classes=["header"]):
            gr.Markdown(
                f"""
                # 🎭 Visagen
                *{i18n.t("app.subtitle")}*
                """
            )

        # Create all tabs
        with gr.Tabs():
            for tab_cls in tab_classes:
                if tab_cls is SettingsTab:
                    tab_instance = tab_cls(state, i18n, settings_path=settings_file)
                else:
                    tab_instance = tab_cls(state, i18n)
                tab_instance.create()

        # Footer
        gr.Markdown(
            f"""
            ---
            <div class="footer">
                {i18n.t("app.footer")}
            </div>
            """,
            elem_classes=["footer-container"],
        )

    return cast(gr.Blocks, app)


def main() -> int:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Visagen Web Interface")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--locale", default="en")
    parser.add_argument(
        "--settings",
        type=str,
        default=None,
        help="Path to settings file (default: ./settings.json if present)",
    )
    parser.add_argument("--dark", action="store_true", help="Use dark theme")

    args = parser.parse_args()

    app = create_app(
        settings_path=args.settings,
        locale=args.locale,
        dark_mode=args.dark,
    )

    # Get custom CSS from app
    css = getattr(app, "_visagen_css", None)
    theme = getattr(app, "_visagen_theme", None)

    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        theme=theme,
        css=css,
    )

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
