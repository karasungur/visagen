"""Wizard mode - step-by-step guided workflow for new users."""

from __future__ import annotations

import subprocess
from collections.abc import Generator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import gradio as gr

from visagen.gui.command_builders import (
    build_extract_command,
    build_merge_command,
    build_train_command,
)
from visagen.gui.components import (
    PathInput,
    PathInputConfig,
)
from visagen.gui.tabs.base import BaseTab

if TYPE_CHECKING:
    from visagen.gui.i18n import I18n
    from visagen.gui.state.app_state import AppState


class WizardTab(BaseTab):
    """
    Step-by-step wizard for new users.

    Guides users through the complete face swap workflow:
    1. Upload source/destination videos
    2. Extract faces
    3. Train model
    4. Apply to video

    This provides a simplified, guided experience for beginners.
    """

    WIZARD_STEPS = [
        ("upload", "1️⃣", "Upload"),
        ("extract", "2️⃣", "Extract"),
        ("train", "3️⃣", "Train"),
        ("apply", "4️⃣", "Apply"),
    ]

    def __init__(self, app_state: AppState, i18n: I18n) -> None:
        """Initialize wizard tab with step tracking."""
        super().__init__(app_state, i18n)
        self._current_step = 0
        self._completed_steps: set[int] = set()
        self._workspace_dir: Path = Path("./wizard_workspace")

    @property
    def id(self) -> str:
        return "wizard"

    def _render_wizard_steps(
        self, current: int = 0, completed: set[int] | None = None
    ) -> str:
        """Render wizard step indicator HTML."""
        completed = completed or set()
        steps_html = []

        for i, (key, icon, _label) in enumerate(self.WIZARD_STEPS):
            is_current = i == current
            is_completed = i in completed

            if is_completed:
                bg_color = "#dcfce7"
                text_color = "#166534"
                border_color = "#22c55e"
                status_icon = " ✓"
            elif is_current:
                bg_color = "#8b5cf6"
                text_color = "white"
                border_color = "#8b5cf6"
                status_icon = ""
            else:
                bg_color = "#f1f5f9"
                text_color = "#64748b"
                border_color = "#e2e8f0"
                status_icon = ""

            # Get translated label
            translated_label = self.t(f"steps.{key}")

            step_html = f"""
            <div style="
                display: flex;
                align-items: center;
                gap: 6px;
                padding: 10px 16px;
                background: {bg_color};
                color: {text_color};
                border: 2px solid {border_color};
                border-radius: 24px;
                font-size: 14px;
                font-weight: {"600" if is_current else "500"};
            ">
                <span>{icon}</span>
                <span>{translated_label}{status_icon}</span>
            </div>
            """
            steps_html.append(step_html)

            if i < len(self.WIZARD_STEPS) - 1:
                steps_html.append(
                    '<div style="color: #94a3b8; font-size: 18px;">→</div>'
                )

        return f"""
        <div style="
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
            padding: 20px;
            background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
            border-radius: 16px;
            border: 1px solid #e2e8f0;
            flex-wrap: wrap;
            margin-bottom: 20px;
        ">
            {"".join(steps_html)}
        </div>
        """

    def _build_content(self) -> dict[str, Any]:
        """Build wizard tab UI with step-by-step sections."""
        components: dict[str, Any] = {}

        gr.Markdown(f"### 🧙 {self.t('title')}")
        gr.Markdown(self.t("description"))

        # Workflow indicator at top
        components["workflow"] = gr.HTML(value=self._render_wizard_steps(0, set()))

        # ========== Step 1: Upload Videos ==========
        with gr.Group(visible=True) as step1:
            gr.Markdown(f"#### {self.t('step1.title')}")
            gr.Markdown(self.t("step1.description"))

            with gr.Row():
                with gr.Column():
                    components["src_video"] = gr.File(
                        label=self.t("step1.src_video"),
                        file_types=[".mp4", ".avi", ".mov", ".mkv"],
                    )
                with gr.Column():
                    components["dst_video"] = gr.File(
                        label=self.t("step1.dst_video"),
                        file_types=[".mp4", ".avi", ".mov", ".mkv"],
                    )

            components["step1_status"] = gr.Textbox(
                label=self.t("step1.status"),
                interactive=False,
                value="",
            )

            with gr.Row():
                components["step1_next"] = gr.Button(
                    self.t("next_step"),
                    variant="primary",
                    size="lg",
                )
        components["step1_group"] = step1

        # ========== Step 2: Extract Faces ==========
        with gr.Group(visible=False) as step2:
            gr.Markdown(f"#### {self.t('step2.title')}")
            gr.Markdown(self.t("step2.description"))

            with gr.Row():
                with gr.Column():
                    components["extract_face_type"] = gr.Dropdown(
                        label=self.t("step2.face_type"),
                        choices=["whole_face", "full", "head"],
                        value="whole_face",
                    )
                with gr.Column():
                    components["extract_size"] = gr.Slider(
                        label=self.t("step2.output_size"),
                        minimum=128,
                        maximum=512,
                        value=256,
                        step=64,
                    )

            components["extract_log"] = gr.Textbox(
                label=self.t("step2.log"),
                lines=8,
                interactive=False,
            )

            with gr.Row():
                components["step2_back"] = gr.Button(self.t("back"))
                components["step2_extract"] = gr.Button(
                    self.t("step2.extract"),
                    variant="primary",
                )
                components["step2_stop"] = gr.Button(
                    self.i18n.t("common.stop"),
                    variant="stop",
                )
                components["step2_next"] = gr.Button(
                    self.t("next_step"),
                    interactive=False,
                )
        components["step2_group"] = step2

        # ========== Step 3: Train Model ==========
        with gr.Group(visible=False) as step3:
            gr.Markdown(f"#### {self.t('step3.title')}")
            gr.Markdown(self.t("step3.description"))

            with gr.Row():
                with gr.Column():
                    components["train_epochs"] = gr.Slider(
                        label=self.t("step3.epochs"),
                        minimum=50,
                        maximum=1000,
                        value=200,
                        step=50,
                    )
                with gr.Column():
                    components["train_batch"] = gr.Slider(
                        label=self.t("step3.batch_size"),
                        minimum=2,
                        maximum=16,
                        value=8,
                        step=2,
                    )

            with gr.Row():
                components["train_preset"] = gr.Dropdown(
                    label=self.t("step3.preset"),
                    choices=[
                        ("🚀 Quick Training", "quick"),
                        ("⚖️ Balanced", "balanced"),
                        ("✨ High Quality", "quality"),
                    ],
                    value="balanced",
                )

            components["train_log"] = gr.Textbox(
                label=self.t("step3.log"),
                lines=10,
                interactive=False,
            )

            components["train_preview"] = gr.Image(
                label=self.t("step3.preview"),
                interactive=False,
                visible=False,
            )

            with gr.Row():
                components["step3_back"] = gr.Button(self.t("back"))
                components["step3_train"] = gr.Button(
                    self.t("step3.train"),
                    variant="primary",
                )
                components["step3_stop"] = gr.Button(
                    self.t("step3.stop"),
                    variant="stop",
                )
                components["step3_next"] = gr.Button(
                    self.t("next_step"),
                    interactive=False,
                )
        components["step3_group"] = step3

        # ========== Step 4: Apply to Video ==========
        with gr.Group(visible=False) as step4:
            gr.Markdown(f"#### {self.t('step4.title')}")
            gr.Markdown(self.t("step4.description"))

            with gr.Row():
                components["output_path"] = PathInput(
                    PathInputConfig(
                        key="wizard.output_video",
                        path_type="file",
                        default="./output_video.mp4",
                    ),
                    self.i18n,
                ).build()

            with gr.Row():
                with gr.Column():
                    components["merge_color_transfer"] = gr.Dropdown(
                        label=self.t("step4.color_transfer"),
                        choices=[
                            "rct",
                            "lct",
                            "sot",
                            "mkl",
                            "idt",
                            "mix",
                            "hist-match",
                            "neural",
                            "none",
                        ],
                        value="rct",
                    )
                with gr.Column():
                    components["merge_blend"] = gr.Dropdown(
                        label=self.t("step4.blend_mode"),
                        choices=["laplacian", "poisson", "feather"],
                        value="laplacian",
                    )

            components["merge_log"] = gr.Textbox(
                label=self.t("step4.log"),
                lines=8,
                interactive=False,
            )

            with gr.Row():
                components["step4_back"] = gr.Button(self.t("back"))
                components["step4_apply"] = gr.Button(
                    self.t("step4.apply"),
                    variant="primary",
                )
                components["step4_stop"] = gr.Button(
                    self.i18n.t("common.stop"),
                    variant="stop",
                )

            components["final_status"] = gr.Textbox(
                label=self.t("step4.status"),
                interactive=False,
                value="",
            )

            components["final_video"] = gr.Video(
                label=self.t("step4.result"),
                visible=False,
            )
        components["step4_group"] = step4

        return components

    def _setup_events(self, c: dict[str, Any]) -> None:
        """Wire up wizard navigation and processing events."""

        # ========== Navigation Helpers ==========
        def goto_step(step: int, completed: set[int] | None = None) -> tuple:
            """Navigate to a specific step."""
            completed = completed or self._completed_steps
            workflow_html = self._render_wizard_steps(step, completed)
            return (
                workflow_html,
                gr.update(visible=(step == 0)),  # step1
                gr.update(visible=(step == 1)),  # step2
                gr.update(visible=(step == 2)),  # step3
                gr.update(visible=(step == 3)),  # step4
            )

        # ========== Step 1: Validate and proceed ==========
        def validate_step1(src_video, dst_video):
            """Validate video uploads and proceed to step 2."""
            if not src_video or not dst_video:
                return (
                    self._render_wizard_steps(0, set()),
                    gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    self.t("step1.error_missing_videos"),
                )

            # Create workspace directories
            self._workspace_dir.mkdir(parents=True, exist_ok=True)
            (self._workspace_dir / "data_src").mkdir(exist_ok=True)
            (self._workspace_dir / "data_dst").mkdir(exist_ok=True)
            (self._workspace_dir / "data_src" / "aligned").mkdir(exist_ok=True)
            (self._workspace_dir / "data_dst" / "aligned").mkdir(exist_ok=True)
            (self._workspace_dir / "model").mkdir(exist_ok=True)

            # Copy videos to workspace
            import shutil

            src_path = src_video.name if hasattr(src_video, "name") else str(src_video)
            dst_path = dst_video.name if hasattr(dst_video, "name") else str(dst_video)

            shutil.copy(src_path, self._workspace_dir / "data_src" / "source.mp4")
            shutil.copy(dst_path, self._workspace_dir / "data_dst" / "destination.mp4")

            self._completed_steps.add(0)
            self._current_step = 1

            return (
                self._render_wizard_steps(1, self._completed_steps),
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=False),
                self.t("step1.success"),
            )

        c["step1_next"].click(
            fn=validate_step1,
            inputs=[c["src_video"], c["dst_video"]],
            outputs=[
                c["workflow"],
                c["step1_group"],
                c["step2_group"],
                c["step3_group"],
                c["step4_group"],
                c["step1_status"],
            ],
        )

        # ========== Step 2: Extract Faces ==========
        def run_extraction(
            face_type: str, output_size: int
        ) -> Generator[tuple, None, None]:
            """Run face extraction for both source and destination."""
            log_lines = []
            stop_codes = {-15, -9, 143, 137}

            def run_stage(cmd: list[str]) -> Generator[tuple, None, bool]:
                process: subprocess.Popen | None = None
                try:
                    process = self.state.processes.launch(
                        "extract",
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1,
                    )
                    if process is None:
                        log_lines.append(
                            "Extraction is already running. Stop it first."
                        )
                        yield (
                            "\n".join(log_lines[-50:]),
                            gr.update(interactive=False),
                        )
                        return False

                    if process.stdout:
                        for line in iter(process.stdout.readline, ""):
                            log_lines.append(line.strip())
                            yield (
                                "\n".join(log_lines[-50:]),
                                gr.update(interactive=False),
                            )
                            if process.poll() is not None:
                                break

                    remaining, _ = process.communicate()
                    if remaining:
                        log_lines.extend(
                            line.strip()
                            for line in remaining.splitlines()
                            if line.strip()
                        )
                        yield (
                            "\n".join(log_lines[-50:]),
                            gr.update(interactive=False),
                        )

                    exit_code = process.returncode
                    if exit_code == 0:
                        return True
                    if exit_code in stop_codes:
                        log_lines.append(self.i18n.t("status.extraction_stopped"))
                    else:
                        log_lines.append(
                            f"Extraction failed with exit code: {exit_code}"
                        )
                    yield (
                        "\n".join(log_lines[-50:]),
                        gr.update(interactive=False),
                    )
                    return False
                except Exception as e:
                    log_lines.append(f"Error: {e}")
                    yield (
                        "\n".join(log_lines[-50:]),
                        gr.update(interactive=False),
                    )
                    return False
                finally:
                    if process is not None:
                        if process.poll() is None:
                            process.terminate()
                            try:
                                process.wait(timeout=5)
                            except subprocess.TimeoutExpired:
                                process.kill()
                                process.wait()
                        self.state.processes.clear_if("extract", process)

            # Extract source faces
            log_lines.append("=== Extracting source faces ===")
            yield (
                "\n".join(log_lines),
                gr.update(interactive=False),
            )

            src_video = self._workspace_dir / "data_src" / "source.mp4"
            src_output = self._workspace_dir / "data_src" / "aligned"

            cmd = build_extract_command(
                src_video,
                src_output,
                face_type=face_type,
                output_size=output_size,
            )
            log_lines.append(f"Resolved argv: {' '.join(cmd)}")

            src_ok = yield from run_stage(cmd)
            if not src_ok:
                return

            # Extract destination faces
            log_lines.append("")
            log_lines.append("=== Extracting destination faces ===")
            yield (
                "\n".join(log_lines[-50:]),
                gr.update(interactive=False),
            )

            dst_video = self._workspace_dir / "data_dst" / "destination.mp4"
            dst_output = self._workspace_dir / "data_dst" / "aligned"

            cmd = build_extract_command(
                dst_video,
                dst_output,
                face_type=face_type,
                output_size=output_size,
            )
            log_lines.append(f"Resolved argv: {' '.join(cmd)}")

            dst_ok = yield from run_stage(cmd)
            if not dst_ok:
                return

            log_lines.append("")
            log_lines.append("✅ Extraction completed!")
            self._completed_steps.add(1)

            yield (
                "\n".join(log_lines[-50:]),
                gr.update(interactive=True),
            )

        c["step2_extract"].click(
            fn=run_extraction,
            inputs=[c["extract_face_type"], c["extract_size"]],
            outputs=[c["extract_log"], c["step2_next"]],
        )

        def stop_extraction() -> str:
            """Stop extraction process."""
            if self.state.processes.terminate("extract"):
                return self.i18n.t("status.extraction_stopped")
            return self.i18n.t("status.no_extraction")

        c["step2_stop"].click(
            fn=stop_extraction,
            outputs=[c["extract_log"]],
        )

        # Step 2 navigation
        def back_to_step1():
            self._current_step = 0
            return goto_step(0, self._completed_steps)

        c["step2_back"].click(
            fn=back_to_step1,
            outputs=[
                c["workflow"],
                c["step1_group"],
                c["step2_group"],
                c["step3_group"],
                c["step4_group"],
            ],
        )

        def next_to_step3():
            self._current_step = 2
            return goto_step(2, self._completed_steps)

        c["step2_next"].click(
            fn=next_to_step3,
            outputs=[
                c["workflow"],
                c["step1_group"],
                c["step2_group"],
                c["step3_group"],
                c["step4_group"],
            ],
        )

        # ========== Step 3: Train Model ==========
        def run_training(
            epochs: int,
            batch_size: int,
            preset: str,
        ) -> Generator[tuple, None, None]:
            """Run model training."""
            log_lines = []
            log_lines.append(f"=== Starting training ({preset} preset) ===")
            log_lines.append(f"Epochs: {epochs}, Batch Size: {batch_size}")

            yield (
                "\n".join(log_lines),
                gr.update(interactive=False),
                gr.update(visible=False),
            )

            src_dir = self._workspace_dir / "data_src" / "aligned"
            dst_dir = self._workspace_dir / "data_dst" / "aligned"
            model_dir = self._workspace_dir / "model"

            precision: str | None = None
            lpips_weight: float | None = None
            if preset == "quick":
                precision = "16-mixed"
            elif preset == "quality":
                lpips_weight = 1.0
            cmd = build_train_command(
                src_dir,
                dst_dir,
                model_dir,
                batch_size=batch_size,
                max_epochs=epochs,
                precision=precision,
                lpips_weight=lpips_weight,
            )
            log_lines.append(f"Resolved argv: {' '.join(cmd)}")

            process: subprocess.Popen | None = None
            training_completed = False
            try:
                process = self.state.processes.launch(
                    "training",
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                if process is None:
                    log_lines.append("Training is already running. Stop it first.")
                    yield (
                        "\n".join(log_lines[-50:]),
                        gr.update(interactive=False),
                        gr.update(visible=False),
                    )
                    return

                if process.stdout:
                    for line in iter(process.stdout.readline, ""):
                        log_lines.append(line.strip())
                        yield (
                            "\n".join(log_lines[-50:]),
                            gr.update(interactive=False),
                            gr.update(visible=False),
                        )
                        if process.poll() is not None:
                            break

                process.wait()
                exit_code = process.returncode
                if exit_code == 0:
                    training_completed = True
                elif exit_code in {-15, -9, 143, 137}:
                    log_lines.append(self.t("step3.stopped"))
                else:
                    log_lines.append(f"Training failed with exit code: {exit_code}")
            except Exception as e:
                log_lines.append(f"Error: {e}")
            finally:
                if process is not None:
                    if process.poll() is None:
                        process.terminate()
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                    self.state.processes.clear_if("training", process)

            if training_completed:
                log_lines.append("")
                log_lines.append("✅ Training completed!")
                self._completed_steps.add(2)

            yield (
                "\n".join(log_lines[-50:]),
                gr.update(interactive=training_completed),
                gr.update(visible=False),
            )

        c["step3_train"].click(
            fn=run_training,
            inputs=[c["train_epochs"], c["train_batch"], c["train_preset"]],
            outputs=[c["train_log"], c["step3_next"], c["train_preview"]],
        )

        def stop_training():
            """Stop training process."""
            if self.state.processes.terminate("training"):
                return self.t("step3.stopped")
            return self.i18n.t("status.no_training")

        c["step3_stop"].click(
            fn=stop_training,
            outputs=[c["train_log"]],
        )

        # Step 3 navigation
        def back_to_step2():
            self._current_step = 1
            return goto_step(1, self._completed_steps)

        c["step3_back"].click(
            fn=back_to_step2,
            outputs=[
                c["workflow"],
                c["step1_group"],
                c["step2_group"],
                c["step3_group"],
                c["step4_group"],
            ],
        )

        def next_to_step4():
            self._current_step = 3
            return goto_step(3, self._completed_steps)

        c["step3_next"].click(
            fn=next_to_step4,
            outputs=[
                c["workflow"],
                c["step1_group"],
                c["step2_group"],
                c["step3_group"],
                c["step4_group"],
            ],
        )

        # ========== Step 4: Apply to Video ==========
        def run_merge(
            output_path: str,
            color_transfer: str,
            blend_mode: str,
        ) -> Generator[tuple, None, None]:
            """Run video merge/application."""
            log_lines = []
            log_lines.append("=== Applying face swap to video ===")

            yield (
                "\n".join(log_lines),
                "",
                gr.update(visible=False),
            )

            checkpoint = self._workspace_dir / "model" / "checkpoints" / "last.ckpt"
            input_video = self._workspace_dir / "data_dst" / "destination.mp4"

            if not checkpoint.exists():
                log_lines.append("❌ Error: No trained model found!")
                yield (
                    "\n".join(log_lines),
                    self.t("step4.error_no_model"),
                    gr.update(visible=False),
                )
                return

            cmd = build_merge_command(
                input_video,
                output_path,
                checkpoint,
                color_transfer=color_transfer,
                blend_mode=blend_mode,
            )
            log_lines.append(f"Resolved argv: {' '.join(cmd)}")

            process: subprocess.Popen | None = None
            try:
                process = self.state.processes.launch(
                    "merge",
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                if process is None:
                    log_lines.append("Merge is already running. Stop it first.")
                    yield (
                        "\n".join(log_lines[-50:]),
                        "",
                        gr.update(visible=False),
                    )
                    return

                if process.stdout:
                    for line in iter(process.stdout.readline, ""):
                        log_lines.append(line.strip())
                        yield (
                            "\n".join(log_lines[-50:]),
                            "",
                            gr.update(visible=False),
                        )
                        if process.poll() is not None:
                            break

                remaining, _ = process.communicate()
                if remaining:
                    log_lines.extend(
                        line.strip() for line in remaining.splitlines() if line.strip()
                    )

                exit_code = process.returncode
                if exit_code == 0:
                    log_lines.append("")
                    log_lines.append("✅ Video creation completed!")
                    self._completed_steps.add(3)

                    yield (
                        "\n".join(log_lines[-50:]),
                        self.t("step4.success", path=output_path),
                        gr.update(visible=True, value=output_path),
                    )
                    return
                if exit_code in {-15, -9, 143, 137}:
                    log_lines.append(self.i18n.t("status.stopped"))
                else:
                    log_lines.append(f"Merge failed with exit code: {exit_code}")

                yield (
                    "\n".join(log_lines[-50:]),
                    "",
                    gr.update(visible=False),
                )
            except Exception as e:
                log_lines.append(f"Error: {e}")
                yield (
                    "\n".join(log_lines[-50:]),
                    "",
                    gr.update(visible=False),
                )
            finally:
                if process is not None:
                    if process.poll() is None:
                        process.terminate()
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                    self.state.processes.clear_if("merge", process)

        c["step4_apply"].click(
            fn=run_merge,
            inputs=[c["output_path"], c["merge_color_transfer"], c["merge_blend"]],
            outputs=[c["merge_log"], c["final_status"], c["final_video"]],
        )

        def stop_merge() -> str:
            """Stop merge process."""
            if self.state.processes.terminate("merge"):
                return self.i18n.t("status.stopped")
            return self.i18n.t("status.no_merge")

        c["step4_stop"].click(
            fn=stop_merge,
            outputs=[c["merge_log"]],
        )

        # Step 4 navigation
        def back_to_step3():
            self._current_step = 2
            return goto_step(2, self._completed_steps)

        c["step4_back"].click(
            fn=back_to_step3,
            outputs=[
                c["workflow"],
                c["step1_group"],
                c["step2_group"],
                c["step3_group"],
                c["step4_group"],
            ],
        )
