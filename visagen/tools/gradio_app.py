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
import queue
import subprocess
import sys
from collections.abc import Generator
from pathlib import Path
from typing import Any

import numpy as np

# Check gradio availability
try:
    import gradio as gr

    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False


class GradioApp:
    """
    Gradio application state and handlers.

    Manages model loading, inference, and training processes.
    """

    def __init__(self) -> None:
        self.model = None
        self.model_path: str | None = None

        # Subprocess management
        self.training_process: subprocess.Popen | None = None
        self.merge_process: subprocess.Popen | None = None
        self.sort_process: subprocess.Popen | None = None
        self.export_process: subprocess.Popen | None = None

        self.training_queue: queue.Queue = queue.Queue()
        self.device = "auto"
        self.settings = {
            "device": "auto",
            "default_batch_size": 8,
            "num_workers": 4,
            "workspace_dir": "./workspace",
        }

        # Lazy-loaded components
        self._restorer = None

    def load_model(self, checkpoint_path: str) -> str:
        """
        Load model from checkpoint.

        Args:
            checkpoint_path: Path to model checkpoint.

        Returns:
            Status message.
        """
        if not checkpoint_path or not Path(checkpoint_path).exists():
            return "Error: Checkpoint file not found"

        try:
            import torch

            from visagen.training.dfl_module import DFLModule

            self.model = DFLModule.load_from_checkpoint(
                checkpoint_path,
                map_location="cpu",
            )
            self.model.eval()

            # Move to GPU if available
            if torch.cuda.is_available():
                self.model = self.model.cuda()

            self.model_path = checkpoint_path
            return f"Model loaded: {Path(checkpoint_path).name}"

        except Exception as e:
            return f"Error loading model: {e}"

    def unload_model(self) -> str:
        """Unload model to free memory."""
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

    def swap_face(
        self,
        source_img: np.ndarray,
        target_img: np.ndarray,
    ) -> np.ndarray:
        """
        Perform face swap inference.

        Args:
            source_img: Source face image (H, W, C) uint8.
            target_img: Target face image (H, W, C) uint8 (unused for now).

        Returns:
            Swapped face image (H, W, C) uint8.
        """
        if self.model is None:
            raise gr.Error("No model loaded. Please load a checkpoint first.")

        if source_img is None:
            raise gr.Error("Please provide a source image.")

        try:
            import cv2
            import torch

            # Preprocess: resize and normalize
            img = cv2.resize(source_img, (256, 256))
            img = img.astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
            img = torch.from_numpy(img).unsqueeze(0)  # Add batch dim

            if next(self.model.parameters()).is_cuda:
                img = img.cuda()

            # Inference
            with torch.no_grad():
                output = self.model(img)

            # Postprocess
            output = output.squeeze(0).cpu().numpy()
            output = np.transpose(output, (1, 2, 0))  # CHW -> HWC
            output = np.clip(output * 255, 0, 255).astype(np.uint8)

            # Resize back to original size if needed
            if source_img.shape[:2] != (256, 256):
                output = cv2.resize(output, (source_img.shape[1], source_img.shape[0]))

            return output

        except Exception as e:
            raise gr.Error(f"Inference failed: {e}")

    def start_training(
        self,
        src_dir: str,
        dst_dir: str,
        output_dir: str,
        batch_size: int,
        max_epochs: int,
        learning_rate: float,
        dssim_weight: float,
        l1_weight: float,
        lpips_weight: float,
        gan_power: float,
        precision: str,
        resume_ckpt: str,
    ) -> Generator[str, None, None]:
        """
        Start training process with progress updates.

        Yields log lines as training progresses.
        """
        # Validate inputs
        if not src_dir or not Path(src_dir).exists():
            yield "Error: Source directory not found"
            return
        if not dst_dir or not Path(dst_dir).exists():
            yield "Error: Destination directory not found"
            return

        # Build command
        cmd = [
            sys.executable,
            "-m",
            "visagen.tools.train",
            "--src-dir",
            src_dir,
            "--dst-dir",
            dst_dir,
            "--output-dir",
            output_dir or "./workspace/model",
            "--batch-size",
            str(int(batch_size)),
            "--max-epochs",
            str(int(max_epochs)),
            "--learning-rate",
            str(learning_rate),
            "--dssim-weight",
            str(dssim_weight),
            "--l1-weight",
            str(l1_weight),
            "--lpips-weight",
            str(lpips_weight),
            "--precision",
            precision,
        ]

        if resume_ckpt and Path(resume_ckpt).exists():
            cmd.extend(["--resume", resume_ckpt])

        yield f"Starting training...\n$ {' '.join(cmd)}\n"

        # Launch subprocess
        try:
            self.training_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            # Stream output
            for line in iter(self.training_process.stdout.readline, ""):
                if line:
                    yield line
                if self.training_process.poll() is not None:
                    break

            # Get remaining output
            remaining, _ = self.training_process.communicate()
            if remaining:
                yield remaining

            exit_code = self.training_process.returncode
            if exit_code == 0:
                yield "\n\nTraining completed successfully!"
            else:
                yield f"\n\nTraining exited with code {exit_code}"

        except Exception as e:
            yield f"\n\nError: {e}"

        finally:
            self.training_process = None

    def stop_training(self) -> str:
        """Stop running training process."""
        if self.training_process is not None:
            self.training_process.terminate()
            try:
                self.training_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.training_process.kill()
            self.training_process = None
            return "Training stopped."
        return "No training in progress."

    def apply_color_transfer(
        self,
        source: np.ndarray,
        target: np.ndarray,
        mode: str,
    ) -> np.ndarray:
        """
        Apply color transfer.

        Args:
            source: Source image (color reference).
            target: Target image (to modify).
            mode: Color transfer mode.

        Returns:
            Color-transferred image.
        """
        if source is None or target is None:
            raise gr.Error("Please provide both source and target images.")

        try:
            from visagen.postprocess import color_transfer

            # Ensure float32 [0, 1] and BGR
            if source.dtype == np.uint8:
                source = source.astype(np.float32) / 255.0
            if target.dtype == np.uint8:
                target = target.astype(np.float32) / 255.0

            # Gradio provides RGB, convert to BGR for OpenCV-based functions
            import cv2

            source_bgr = cv2.cvtColor(source, cv2.COLOR_RGB2BGR)
            target_bgr = cv2.cvtColor(target, cv2.COLOR_RGB2BGR)

            result_bgr = color_transfer(mode, target_bgr, source_bgr)

            # Convert back to RGB
            result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

            return (result_rgb * 255).astype(np.uint8)

        except Exception as e:
            raise gr.Error(f"Color transfer failed: {e}")

    def apply_blend(
        self,
        foreground: np.ndarray,
        background: np.ndarray,
        mask: np.ndarray,
        mode: str,
    ) -> np.ndarray:
        """
        Apply image blending.

        Args:
            foreground: Foreground image.
            background: Background image.
            mask: Blending mask.
            mode: Blend mode.

        Returns:
            Blended image.
        """
        if foreground is None or background is None or mask is None:
            raise gr.Error("Please provide foreground, background, and mask images.")

        try:
            import cv2

            from visagen.postprocess import blend

            # Ensure float32 [0, 1]
            if foreground.dtype == np.uint8:
                foreground = foreground.astype(np.float32) / 255.0
            if background.dtype == np.uint8:
                background = background.astype(np.float32) / 255.0
            if mask.dtype == np.uint8:
                mask = mask.astype(np.float32) / 255.0

            # Handle mask dimensions
            if mask.ndim == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)

            # Resize mask to match images if needed
            if mask.shape[:2] != foreground.shape[:2]:
                mask = cv2.resize(mask, (foreground.shape[1], foreground.shape[0]))

            result = blend(mode, foreground, background, mask)

            return (result * 255).astype(np.uint8)

        except Exception as e:
            raise gr.Error(f"Blending failed: {e}")

    def run_extraction(
        self,
        input_path: str,
        output_dir: str,
        face_type: str,
        output_size: int,
        min_confidence: float,
    ) -> Generator[str, None, None]:
        """Run face extraction."""
        if not input_path or not Path(input_path).exists():
            yield "Error: Input path not found"
            return

        cmd = [
            sys.executable,
            "-m",
            "visagen.tools.extract_v2",
            "--input",
            input_path,
            "--output",
            output_dir or "./workspace/extracted",
            "--face-type",
            face_type,
            "--output-size",
            str(int(output_size)),
            "--min-confidence",
            str(min_confidence),
        ]

        yield f"Starting extraction...\n$ {' '.join(cmd)}\n"

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )

            for line in iter(process.stdout.readline, ""):
                if line:
                    yield line

            exit_code = process.wait()
            if exit_code == 0:
                yield "\n\nExtraction completed!"
            else:
                yield f"\n\nExtraction failed with code {exit_code}"

        except Exception as e:
            yield f"\n\nError: {e}"

    def save_settings(
        self,
        device: str,
        gpu_id: int,
        default_batch_size: int,
        num_workers: int,
        workspace_dir: str,
    ) -> str:
        """Save application settings."""
        self.settings.update(
            {
                "device": device,
                "gpu_id": int(gpu_id),
                "default_batch_size": int(default_batch_size),
                "num_workers": int(num_workers),
                "workspace_dir": workspace_dir,
            }
        )
        return "Settings saved."

    def apply_face_restoration(
        self,
        image: np.ndarray,
        strength: float,
        mode: str,
        model_version: float,
        gpen_model_size: int,
    ) -> np.ndarray:
        """
        Apply face restoration to image.

        Args:
            image: Input face image (H, W, 3) uint8 RGB.
            strength: Restoration strength (0.0-1.0).
            mode: Restoration mode ('gfpgan' or 'gpen').
            model_version: GFPGAN version (1.2, 1.3, 1.4).
            gpen_model_size: GPEN model size (256, 512, 1024).

        Returns:
            Restored face image (H, W, 3) uint8 RGB.
        """
        if image is None:
            raise gr.Error("Please provide an image.")

        try:
            import cv2

            # Gradio provides RGB, restoration expects BGR
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if mode == "gpen":
                from visagen.postprocess.gpen import (
                    is_gpen_available,
                    restore_face_gpen,
                )

                if not is_gpen_available():
                    raise gr.Error(
                        "GPEN not available. Requires PyTorch. "
                        "Install with: pip install torch"
                    )

                restored_bgr = restore_face_gpen(
                    image_bgr,
                    strength=strength,
                    model_size=gpen_model_size,
                )
            else:
                # Default: GFPGAN
                from visagen.postprocess.restore import (
                    is_gfpgan_available,
                    restore_face,
                )

                if not is_gfpgan_available():
                    raise gr.Error(
                        "GFPGAN not installed. "
                        "Install with: pip install 'visagen[restore]'"
                    )

                restored_bgr = restore_face(
                    image_bgr,
                    strength=strength,
                    model_version=model_version,
                )

            # Convert back to RGB
            restored_rgb = cv2.cvtColor(restored_bgr, cv2.COLOR_BGR2RGB)

            return restored_rgb

        except ImportError:
            raise gr.Error(
                "Restore module not available. "
                "Install with: pip install 'visagen[restore]'"
            )
        except Exception as e:
            raise gr.Error(f"Face restoration failed: {e}")

    def apply_neural_color_transfer(
        self,
        source: np.ndarray,
        target: np.ndarray,
        mode: str,
        strength: float,
        preserve_luminance: bool,
    ) -> np.ndarray:
        """
        Apply neural color transfer.

        Args:
            source: Style reference image (H, W, 3) uint8 RGB.
            target: Target image to modify (H, W, 3) uint8 RGB.
            mode: Transfer mode ('histogram', 'statistics', 'gram').
            strength: Transfer strength (0.0-1.0).
            preserve_luminance: Keep target luminance.

        Returns:
            Color-transferred image (H, W, 3) uint8 RGB.
        """
        if source is None or target is None:
            raise gr.Error("Please provide both source and target images.")

        try:
            import cv2

            from visagen.postprocess.neural_color import (
                is_neural_color_available,
                neural_color_transfer,
            )

            if not is_neural_color_available():
                raise gr.Error(
                    "Neural color transfer not available. "
                    "Requires PyTorch. Install with: pip install torch"
                )

            # Convert RGB to BGR float32 [0, 1]
            source_bgr = (
                cv2.cvtColor(source, cv2.COLOR_RGB2BGR).astype(np.float32) / 255
            )
            target_bgr = (
                cv2.cvtColor(target, cv2.COLOR_RGB2BGR).astype(np.float32) / 255
            )

            # Apply transfer
            result_bgr = neural_color_transfer(
                target_bgr,
                source_bgr,
                mode=mode,
                strength=strength,
                preserve_luminance=preserve_luminance,
            )

            # Convert back to RGB uint8
            result_rgb = cv2.cvtColor(
                (result_bgr * 255).clip(0, 255).astype(np.uint8),
                cv2.COLOR_BGR2RGB,
            )

            return result_rgb

        except Exception as e:
            raise gr.Error(f"Neural color transfer failed: {e}")

    def segment_and_export_mask(
        self,
        image: np.ndarray,
        export_format: str,
        label: str,
    ) -> tuple[np.ndarray | None, str | None]:
        """
        Segment face and export mask to annotation format.

        Args:
            image: Input image (H, W, 3) uint8 RGB.
            export_format: Export format ('labelme' or 'coco').
            label: Label name for the mask.

        Returns:
            Tuple of (mask visualization, exported file path).
        """
        if image is None:
            raise gr.Error("Please provide an image.")

        try:
            import tempfile

            import cv2

            from visagen.vision.mask_export import export_coco, export_labelme
            from visagen.vision.segmenter import FaceSegmenter

            # Convert RGB to BGR
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Segment
            segmenter = FaceSegmenter()
            mask = segmenter.segment(image_bgr)

            if mask is None:
                raise gr.Error("Segmentation failed. No face detected.")

            # Create temp file for export
            suffix = ".json"
            with tempfile.NamedTemporaryFile(
                suffix=suffix, delete=False, mode="w"
            ) as f:
                temp_path = f.name

            # Export based on format
            from pathlib import Path

            if export_format == "coco":
                export_coco(
                    [Path("input.jpg")],
                    [mask],
                    Path(temp_path),
                    categories=[{"id": 1, "name": label}],
                )
            else:
                # Default: labelme
                export_labelme(
                    Path("input.jpg"),
                    mask,
                    Path(temp_path),
                    label=label,
                )

            # Create mask visualization (colorized)
            mask_vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
            mask_vis[:, :, 1] = 0  # Remove green
            mask_vis[:, :, 2] = 0  # Remove blue, keep red

            return mask_vis, temp_path

        except Exception as e:
            raise gr.Error(f"Mask export failed: {e}")

    def run_merge(
        self,
        input_video: str,
        output_video: str,
        checkpoint: str,
        color_transfer: str,
        blend_mode: str,
        restore_face: bool,
        restore_strength: float,
        restore_version: float,
        codec: str,
        crf: int,
    ) -> Generator[str, None, None]:
        """
        Run video merge using visagen-merge CLI.

        Yields log lines as processing progresses.
        """
        # Validate inputs
        if not input_video or not Path(input_video).exists():
            yield "Error: Input video not found"
            return
        if not checkpoint or not Path(checkpoint).exists():
            yield "Error: Checkpoint not found"
            return

        # Build command
        cmd = [
            sys.executable,
            "-m",
            "visagen.tools.merge",
            str(input_video),
            str(output_video) if output_video else "./output.mp4",
            "--checkpoint",
            str(checkpoint),
            "--color-transfer",
            color_transfer if color_transfer != "none" else "none",
            "--blend-mode",
            blend_mode,
            "--codec",
            codec,
            "--crf",
            str(int(crf)),
        ]

        # Add restoration options
        if restore_face:
            cmd.append("--restore-face")
            cmd.extend(["--restore-strength", str(restore_strength)])
            cmd.extend(["--restore-model", str(restore_version)])

        yield f"Starting merge...\n$ {' '.join(cmd)}\n"

        try:
            self.merge_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            for line in iter(self.merge_process.stdout.readline, ""):
                if line:
                    yield line
                if self.merge_process.poll() is not None:
                    break

            remaining, _ = self.merge_process.communicate()
            if remaining:
                yield remaining

            exit_code = self.merge_process.returncode
            if exit_code == 0:
                yield "\n\nMerge completed successfully!"
            else:
                yield f"\n\nMerge exited with code {exit_code}"

        except Exception as e:
            yield f"\n\nError: {e}"

        finally:
            self.merge_process = None

    def stop_merge(self) -> str:
        """Stop running merge process."""
        if self.merge_process is not None:
            self.merge_process.terminate()
            try:
                self.merge_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.merge_process.kill()
            self.merge_process = None
            return "Merge stopped."
        return "No merge in progress."

    def run_sort(
        self,
        input_dir: str,
        output_dir: str,
        method: str,
        target_count: int,
        dry_run: bool,
    ) -> Generator[str, None, None]:
        """
        Run dataset sorting using visagen-sort CLI.

        Yields log lines as sorting progresses.
        """
        if not input_dir or not Path(input_dir).exists():
            yield "Error: Input directory not found"
            return

        # Build command
        cmd = [
            sys.executable,
            "-m",
            "visagen.tools.sorter",
            str(input_dir),
            "--method",
            method,
            "--verbose",
        ]

        if output_dir:
            cmd.extend(["--output-dir", str(output_dir)])

        if method in ("final", "final-fast"):
            cmd.extend(["--target-count", str(int(target_count))])

        if dry_run:
            cmd.append("--dry-run")

        yield f"Starting sorting...\n$ {' '.join(cmd)}\n"

        try:
            self.sort_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            for line in iter(self.sort_process.stdout.readline, ""):
                if line:
                    yield line
                if self.sort_process.poll() is not None:
                    break

            remaining, _ = self.sort_process.communicate()
            if remaining:
                yield remaining

            exit_code = self.sort_process.returncode
            if exit_code == 0:
                yield "\n\nSorting completed!"
            else:
                yield f"\n\nSorting exited with code {exit_code}"

        except Exception as e:
            yield f"\n\nError: {e}"

        finally:
            self.sort_process = None

    def stop_sort(self) -> str:
        """Stop running sort process."""
        if self.sort_process is not None:
            self.sort_process.terminate()
            try:
                self.sort_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.sort_process.kill()
            self.sort_process = None
            return "Sorting stopped."
        return "No sorting in progress."

    def run_export(
        self,
        input_path: str,
        output_path: str,
        export_format: str,
        precision: str,
        validate: bool,
    ) -> Generator[str, None, None]:
        """
        Run model export using visagen-export CLI.

        Yields log lines as export progresses.
        """
        if not input_path or not Path(input_path).exists():
            yield "Error: Input file not found"
            return

        # Build command
        cmd = [
            sys.executable,
            "-m",
            "visagen.tools.export",
            str(input_path),
            "-o",
            str(output_path) if output_path else "./model.onnx",
            "--format",
            export_format,
            "--precision",
            precision,
        ]

        if validate:
            cmd.append("--validate")

        yield f"Starting export...\n$ {' '.join(cmd)}\n"

        try:
            self.export_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            for line in iter(self.export_process.stdout.readline, ""):
                if line:
                    yield line
                if self.export_process.poll() is not None:
                    break

            remaining, _ = self.export_process.communicate()
            if remaining:
                yield remaining

            exit_code = self.export_process.returncode
            if exit_code == 0:
                yield "\n\nExport completed successfully!"
            else:
                yield f"\n\nExport exited with code {exit_code}"

        except Exception as e:
            yield f"\n\nError: {e}"

        finally:
            self.export_process = None

    def stop_export(self) -> str:
        """Stop running export process."""
        if self.export_process is not None:
            self.export_process.terminate()
            try:
                self.export_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.export_process.kill()
            self.export_process = None
            return "Export stopped."
        return "No export in progress."

    def run_extract_frames(
        self,
        input_video: str,
        output_dir: str,
        fps: float | None,
        output_format: str,
    ) -> Generator[str, None, None]:
        """Run frame extraction from video."""
        if not input_video or not Path(input_video).exists():
            yield "Error: Input video not found"
            return

        cmd = [
            sys.executable,
            "-m",
            "visagen.tools.video_ed",
            "extract",
            str(input_video),
            "--output",
            output_dir or "./frames",
            "--format",
            output_format,
        ]

        if fps and fps > 0:
            cmd.extend(["--fps", str(fps)])

        yield f"Starting frame extraction...\n$ {' '.join(cmd)}\n"

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )

            for line in iter(process.stdout.readline, ""):
                if line:
                    yield line

            exit_code = process.wait()
            if exit_code == 0:
                yield "\n\nExtraction completed!"
            else:
                yield f"\n\nExtraction failed with code {exit_code}"

        except Exception as e:
            yield f"\n\nError: {e}"

    def run_create_video(
        self,
        input_dir: str,
        output_video: str,
        fps: float,
        codec: str,
        bitrate: str,
    ) -> Generator[str, None, None]:
        """Run video creation from frames."""
        if not input_dir or not Path(input_dir).exists():
            yield "Error: Input directory not found"
            return

        cmd = [
            sys.executable,
            "-m",
            "visagen.tools.video_ed",
            "create",
            str(input_dir),
            "--output",
            output_video or "./output.mp4",
            "--fps",
            str(fps),
            "--codec",
            codec,
            "--bitrate",
            bitrate,
        ]

        yield f"Starting video creation...\n$ {' '.join(cmd)}\n"

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )

            for line in iter(process.stdout.readline, ""):
                if line:
                    yield line

            exit_code = process.wait()
            if exit_code == 0:
                yield "\n\nVideo created successfully!"
            else:
                yield f"\n\nVideo creation failed with code {exit_code}"

        except Exception as e:
            yield f"\n\nError: {e}"

    def run_cut_video(
        self,
        input_video: str,
        output_video: str,
        start_time: str,
        end_time: str,
    ) -> Generator[str, None, None]:
        """Run video cutting."""
        if not input_video or not Path(input_video).exists():
            yield "Error: Input video not found"
            return

        cmd = [
            sys.executable,
            "-m",
            "visagen.tools.video_ed",
            "cut",
            str(input_video),
            "--output",
            output_video or "./cut_output.mp4",
            "--start",
            start_time,
            "--end",
            end_time,
        ]

        yield f"Starting video cut...\n$ {' '.join(cmd)}\n"

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )

            for line in iter(process.stdout.readline, ""):
                if line:
                    yield line

            exit_code = process.wait()
            if exit_code == 0:
                yield "\n\nVideo cut completed!"
            else:
                yield f"\n\nVideo cut failed with code {exit_code}"

        except Exception as e:
            yield f"\n\nError: {e}"

    def run_denoise_sequence(
        self,
        input_dir: str,
        output_dir: str,
        factor: int,
    ) -> Generator[str, None, None]:
        """Run temporal denoising on frame sequence."""
        if not input_dir or not Path(input_dir).exists():
            yield "Error: Input directory not found"
            return

        cmd = [
            sys.executable,
            "-m",
            "visagen.tools.video_ed",
            "denoise",
            str(input_dir),
            "--factor",
            str(factor),
        ]

        if output_dir:
            cmd.extend(["--output", output_dir])

        yield f"Starting temporal denoising...\n$ {' '.join(cmd)}\n"

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )

            for line in iter(process.stdout.readline, ""):
                if line:
                    yield line

            exit_code = process.wait()
            if exit_code == 0:
                yield "\n\nDenoising completed!"
            else:
                yield f"\n\nDenoising failed with code {exit_code}"

        except Exception as e:
            yield f"\n\nError: {e}"

    def run_enhance_faceset(
        self,
        input_dir: str,
        output_dir: str,
        strength: float,
        model_version: float,
    ) -> Generator[str, None, None]:
        """Run faceset enhancement with GFPGAN."""
        if not input_dir or not Path(input_dir).exists():
            yield "Error: Input directory not found"
            return

        cmd = [
            sys.executable,
            "-m",
            "visagen.tools.faceset_enhancer",
            str(input_dir),
            "--strength",
            str(strength),
            "--model-version",
            str(model_version),
        ]

        if output_dir:
            cmd.extend(["--output", output_dir])

        yield f"Starting faceset enhancement...\n$ {' '.join(cmd)}\n"

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )

            for line in iter(process.stdout.readline, ""):
                if line:
                    yield line

            exit_code = process.wait()
            if exit_code == 0:
                yield "\n\nEnhancement completed!"
            else:
                yield f"\n\nEnhancement failed with code {exit_code}"

        except Exception as e:
            yield f"\n\nError: {e}"

    def run_resize_faceset(
        self,
        input_dir: str,
        output_dir: str,
        target_size: int,
        face_type: str | None,
        interpolation: str,
    ) -> Generator[str, None, None]:
        """Run faceset resizing."""
        if not input_dir or not Path(input_dir).exists():
            yield "Error: Input directory not found"
            return

        cmd = [
            sys.executable,
            "-m",
            "visagen.tools.faceset_resizer",
            str(input_dir),
            "--size",
            str(target_size),
            "--interpolation",
            interpolation,
        ]

        if output_dir:
            cmd.extend(["--output", output_dir])

        if face_type and face_type != "keep":
            cmd.extend(["--face-type", face_type])

        yield f"Starting faceset resize...\n$ {' '.join(cmd)}\n"

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )

            for line in iter(process.stdout.readline, ""):
                if line:
                    yield line

            exit_code = process.wait()
            if exit_code == 0:
                yield "\n\nResize completed!"
            else:
                yield f"\n\nResize failed with code {exit_code}"

        except Exception as e:
            yield f"\n\nError: {e}"


def create_training_tab(app: GradioApp) -> dict[str, Any]:
    """Create training configuration and execution tab."""
    with gr.Tab("Training"):
        gr.Markdown("### Model Training")

        with gr.Row():
            with gr.Column():
                src_dir = gr.Textbox(
                    label="Source Directory",
                    placeholder="./workspace/data_src/aligned",
                    info="Directory containing source face images",
                )
                dst_dir = gr.Textbox(
                    label="Destination Directory",
                    placeholder="./workspace/data_dst/aligned",
                    info="Directory containing destination face images",
                )
                output_dir = gr.Textbox(
                    label="Output Directory",
                    value="./workspace/model",
                    info="Directory for checkpoints and logs",
                )

            with gr.Column():
                batch_size = gr.Slider(
                    1,
                    32,
                    value=8,
                    step=1,
                    label="Batch Size",
                )
                max_epochs = gr.Slider(
                    10,
                    2000,
                    value=500,
                    step=10,
                    label="Max Epochs",
                )
                learning_rate = gr.Number(
                    value=1e-4,
                    label="Learning Rate",
                )

        with gr.Row():
            with gr.Column():
                dssim_weight = gr.Slider(
                    0,
                    30,
                    value=10.0,
                    label="DSSIM Weight",
                )
                l1_weight = gr.Slider(
                    0,
                    30,
                    value=10.0,
                    label="L1 Weight",
                )
                lpips_weight = gr.Slider(
                    0,
                    10,
                    value=0.0,
                    label="LPIPS Weight",
                    info="Requires lpips package",
                )

            with gr.Column():
                gan_power = gr.Slider(
                    0,
                    1.0,
                    value=0.0,
                    label="GAN Power",
                    info="0 = disabled, > 0 = adversarial training",
                )
                precision = gr.Dropdown(
                    ["32", "16-mixed", "bf16-mixed"],
                    value="32",
                    label="Precision",
                )

        with gr.Row():
            resume_ckpt = gr.Textbox(
                label="Resume from Checkpoint (optional)",
                placeholder="./workspace/model/checkpoints/last.ckpt",
            )

        with gr.Row():
            train_btn = gr.Button("Start Training", variant="primary")
            stop_btn = gr.Button("Stop Training", variant="stop")

        training_log = gr.Textbox(
            label="Training Log",
            lines=15,
            max_lines=30,
            interactive=False,
        )

        # Event handlers
        train_btn.click(
            fn=app.start_training,
            inputs=[
                src_dir,
                dst_dir,
                output_dir,
                batch_size,
                max_epochs,
                learning_rate,
                dssim_weight,
                l1_weight,
                lpips_weight,
                gan_power,
                precision,
                resume_ckpt,
            ],
            outputs=training_log,
        )

        stop_btn.click(
            fn=app.stop_training,
            outputs=training_log,
        )

    return {
        "src_dir": src_dir,
        "dst_dir": dst_dir,
        "output_dir": output_dir,
    }


def create_inference_tab(app: GradioApp) -> dict[str, Any]:
    """Create single image face swap inference tab."""
    with gr.Tab("Inference"):
        gr.Markdown("### Face Swap Inference")

        with gr.Row():
            checkpoint_path = gr.Textbox(
                label="Model Checkpoint",
                placeholder="./workspace/model/checkpoints/last.ckpt",
            )
            load_btn = gr.Button("Load Model")
            unload_btn = gr.Button("Unload Model")

        model_status = gr.Textbox(
            label="Model Status",
            value="No model loaded",
            interactive=False,
        )

        with gr.Row():
            source_image = gr.Image(
                label="Source Face",
                type="numpy",
            )
            target_image = gr.Image(
                label="Target Face (reference)",
                type="numpy",
            )
            output_image = gr.Image(
                label="Result",
                type="numpy",
            )

        with gr.Row():
            swap_btn = gr.Button("Swap Face", variant="primary")

        # Event handlers
        load_btn.click(
            fn=app.load_model,
            inputs=checkpoint_path,
            outputs=model_status,
        )

        unload_btn.click(
            fn=app.unload_model,
            outputs=model_status,
        )

        swap_btn.click(
            fn=app.swap_face,
            inputs=[source_image, target_image],
            outputs=output_image,
        )

    return {}


def create_extract_tab(app: GradioApp) -> dict[str, Any]:
    """Create face extraction tab."""
    with gr.Tab("Extract"):
        gr.Markdown("### Face Extraction")
        gr.Markdown("Extract faces from images or videos for training.")

        with gr.Row():
            input_path = gr.Textbox(
                label="Input (image, video, or directory)",
                placeholder="./input_video.mp4",
            )
            output_dir = gr.Textbox(
                label="Output Directory",
                value="./workspace/extracted",
            )

        with gr.Row():
            face_type = gr.Dropdown(
                ["whole_face", "full", "mid_full", "half", "head"],
                value="whole_face",
                label="Face Type",
            )
            output_size = gr.Slider(
                128,
                1024,
                value=512,
                step=64,
                label="Output Size",
            )
            min_confidence = gr.Slider(
                0.1,
                1.0,
                value=0.5,
                step=0.05,
                label="Min Confidence",
            )

        with gr.Row():
            extract_btn = gr.Button("Extract Faces", variant="primary")

        extract_log = gr.Textbox(
            label="Extraction Log",
            lines=10,
            max_lines=20,
            interactive=False,
        )

        extract_btn.click(
            fn=app.run_extraction,
            inputs=[input_path, output_dir, face_type, output_size, min_confidence],
            outputs=extract_log,
        )

        gr.Markdown("---")
        gr.Markdown("### Mask Export")
        gr.Markdown(
            "Segment faces and export masks to LabelMe/COCO format for external editing."
        )

        with gr.Row():
            mask_input = gr.Image(
                label="Input Image",
                type="numpy",
            )
            mask_preview = gr.Image(
                label="Segmentation Mask",
                type="numpy",
            )

        with gr.Row():
            mask_format = gr.Dropdown(
                ["labelme", "coco"],
                value="labelme",
                label="Export Format",
                info="LabelMe=per-image JSON, COCO=dataset-wide JSON",
            )
            mask_label = gr.Textbox(
                value="face",
                label="Label Name",
                info="Label for the segmented region",
            )
            mask_export_btn = gr.Button("Segment & Export")

        mask_output = gr.File(label="Exported Annotation File")

        mask_export_btn.click(
            fn=app.segment_and_export_mask,
            inputs=[mask_input, mask_format, mask_label],
            outputs=[mask_preview, mask_output],
        )

    return {}


def create_merge_tab(app: GradioApp) -> dict[str, Any]:
    """Create video merge processing tab."""
    with gr.Tab("Merge"):
        gr.Markdown("### Video Face Swap")
        gr.Markdown(
            "Process videos using trained models with customizable blending and color transfer."
        )

        with gr.Row():
            with gr.Column():
                input_video = gr.Textbox(
                    label="Input Video",
                    placeholder="./input.mp4",
                    info="Path to source video file",
                )
                output_video = gr.Textbox(
                    label="Output Video",
                    placeholder="./output.mp4",
                    info="Path for processed video output",
                )
                merge_checkpoint = gr.Textbox(
                    label="Model Checkpoint",
                    placeholder="./workspace/model/checkpoints/last.ckpt",
                )

            with gr.Column():
                color_transfer = gr.Dropdown(
                    ["rct", "lct", "sot", "none"],
                    value="rct",
                    label="Color Transfer Mode",
                    info="RCT=Reinhard, LCT=Linear, SOT=Sliced OT",
                )
                blend_mode = gr.Dropdown(
                    ["laplacian", "poisson", "feather"],
                    value="laplacian",
                    label="Blend Mode",
                    info="Laplacian=pyramid, Poisson=seamless, Feather=alpha",
                )

        # Face Restoration Section
        gr.Markdown("#### Face Restoration")
        with gr.Row():
            merge_restore_face = gr.Checkbox(
                label="Enable GFPGAN",
                value=False,
                info="Enhance face quality with GFPGAN",
            )
            merge_restore_strength = gr.Slider(
                0.0,
                1.0,
                value=0.5,
                step=0.1,
                label="Restoration Strength",
            )
            merge_restore_version = gr.Dropdown(
                [1.2, 1.3, 1.4],
                value=1.4,
                label="GFPGAN Version",
            )

        # Video Encoding Section
        gr.Markdown("#### Video Encoding")
        with gr.Row():
            codec = gr.Dropdown(
                ["auto", "libx264", "libx265", "h264_nvenc", "hevc_nvenc"],
                value="auto",
                label="Encoder",
                info="'auto' selects NVENC if available",
            )
            crf = gr.Slider(
                0,
                51,
                value=18,
                step=1,
                label="Quality (CRF)",
                info="Lower = better quality, higher file size",
            )

        with gr.Row():
            merge_btn = gr.Button("Start Merge", variant="primary")
            stop_merge_btn = gr.Button("Stop", variant="stop")

        merge_log = gr.Textbox(
            label="Merge Log",
            lines=15,
            max_lines=30,
            interactive=False,
        )

        # Event handlers
        merge_btn.click(
            fn=app.run_merge,
            inputs=[
                input_video,
                output_video,
                merge_checkpoint,
                color_transfer,
                blend_mode,
                merge_restore_face,
                merge_restore_strength,
                merge_restore_version,
                codec,
                crf,
            ],
            outputs=merge_log,
        )

        stop_merge_btn.click(
            fn=app.stop_merge,
            outputs=merge_log,
        )

    return {}


def create_sort_tab(app: GradioApp) -> dict[str, Any]:
    """Create dataset sorting tab."""
    with gr.Tab("Sort"):
        gr.Markdown("### Dataset Sorting")
        gr.Markdown("Sort and filter face images by various criteria.")

        with gr.Row():
            with gr.Column():
                sort_input = gr.Textbox(
                    label="Input Directory",
                    placeholder="./workspace/data_src/aligned",
                    info="Directory containing aligned face images",
                )
                sort_output = gr.Textbox(
                    label="Output Directory (optional)",
                    placeholder="Leave empty to sort in place",
                    info="Optional output directory for sorted images",
                )

            with gr.Column():
                sort_method = gr.Dropdown(
                    [
                        "blur",
                        "motion-blur",
                        "face-yaw",
                        "face-pitch",
                        "face-source-rect-size",
                        "hist",
                        "hist-dissim",
                        "brightness",
                        "hue",
                        "black",
                        "origname",
                        "oneface",
                        "final",
                        "final-fast",
                    ],
                    value="blur",
                    label="Sort Method",
                    info="Select sorting/filtering algorithm",
                )
                target_count = gr.Slider(
                    100,
                    10000,
                    value=2000,
                    step=100,
                    label="Target Count",
                    info="Used only for 'final' and 'final-fast' methods",
                )
                dry_run = gr.Checkbox(
                    label="Dry Run (Preview)",
                    value=True,
                    info="Show what would happen without making changes",
                )

        with gr.Row():
            sort_btn = gr.Button("Start Sorting", variant="primary")
            stop_sort_btn = gr.Button("Stop", variant="stop")

        sort_log = gr.Textbox(
            label="Sorting Log",
            lines=15,
            max_lines=30,
            interactive=False,
        )

        # Event handlers
        sort_btn.click(
            fn=app.run_sort,
            inputs=[sort_input, sort_output, sort_method, target_count, dry_run],
            outputs=sort_log,
        )

        stop_sort_btn.click(
            fn=app.stop_sort,
            outputs=sort_log,
        )

    return {}


def create_export_tab(app: GradioApp) -> dict[str, Any]:
    """Create model export tab."""
    with gr.Tab("Export"):
        gr.Markdown("### Model Export")
        gr.Markdown(
            "Export trained models to ONNX or TensorRT for optimized inference."
        )

        with gr.Row():
            with gr.Column():
                export_input = gr.Textbox(
                    label="Input Path",
                    placeholder="./workspace/model/checkpoints/last.ckpt",
                    info="Checkpoint (.ckpt) for ONNX, or ONNX (.onnx) for TensorRT",
                )
                export_output = gr.Textbox(
                    label="Output Path",
                    placeholder="./model.onnx",
                    info="Output file path (.onnx or .engine)",
                )

            with gr.Column():
                export_format = gr.Dropdown(
                    ["onnx", "tensorrt"],
                    value="onnx",
                    label="Export Format",
                    info="ONNX for cross-platform, TensorRT for NVIDIA GPUs",
                )
                export_precision = gr.Dropdown(
                    ["fp32", "fp16", "int8"],
                    value="fp16",
                    label="Precision",
                    info="FP16 recommended for balance of speed and quality",
                )
                export_validate = gr.Checkbox(
                    label="Validate Export",
                    value=True,
                    info="Compare exported model against PyTorch original",
                )

        with gr.Row():
            export_btn = gr.Button("Export Model", variant="primary")
            stop_export_btn = gr.Button("Stop", variant="stop")

        export_log = gr.Textbox(
            label="Export Log",
            lines=15,
            max_lines=30,
            interactive=False,
        )

        # Event handlers
        export_btn.click(
            fn=app.run_export,
            inputs=[
                export_input,
                export_output,
                export_format,
                export_precision,
                export_validate,
            ],
            outputs=export_log,
        )

        stop_export_btn.click(
            fn=app.stop_export,
            outputs=export_log,
        )

    return {}


def create_postprocess_tab(app: GradioApp) -> dict[str, Any]:
    """Create color transfer and blending demo tab."""
    with gr.Tab("Postprocess"):
        gr.Markdown("### Color Transfer Demo")

        with gr.Row():
            ct_source = gr.Image(
                label="Source (color reference)",
                type="numpy",
            )
            ct_target = gr.Image(
                label="Target (to modify)",
                type="numpy",
            )
            ct_result = gr.Image(
                label="Result",
                type="numpy",
            )

        with gr.Row():
            ct_mode = gr.Dropdown(
                ["rct", "lct", "sot", "mkl", "idt"],
                value="rct",
                label="Color Transfer Mode",
                info="RCT=Reinhard, LCT=Linear, SOT=Sliced OT, MKL=Monge-Kantorovitch, IDT=Iterative",
            )
            ct_btn = gr.Button("Apply Color Transfer")

        ct_btn.click(
            fn=app.apply_color_transfer,
            inputs=[ct_source, ct_target, ct_mode],
            outputs=ct_result,
        )

        gr.Markdown("---")
        gr.Markdown("### Blending Demo")

        with gr.Row():
            bl_fg = gr.Image(
                label="Foreground",
                type="numpy",
            )
            bl_bg = gr.Image(
                label="Background",
                type="numpy",
            )
            bl_mask = gr.Image(
                label="Mask",
                type="numpy",
            )
            bl_result = gr.Image(
                label="Result",
                type="numpy",
            )

        with gr.Row():
            bl_mode = gr.Dropdown(
                ["laplacian", "poisson", "feather"],
                value="laplacian",
                label="Blend Mode",
                info="Laplacian=multi-band pyramid, Poisson=seamless clone, Feather=alpha blend",
            )
            bl_btn = gr.Button("Blend Images")

        bl_btn.click(
            fn=app.apply_blend,
            inputs=[bl_fg, bl_bg, bl_mask, bl_mode],
            outputs=bl_result,
        )

        gr.Markdown("---")
        gr.Markdown("### Face Restoration Demo")
        gr.Markdown("Enhance face quality using GFPGAN or GPEN.")

        with gr.Row():
            restore_input = gr.Image(
                label="Input Face",
                type="numpy",
            )
            restore_result = gr.Image(
                label="Restored Face",
                type="numpy",
            )

        with gr.Row():
            restore_mode = gr.Dropdown(
                ["gfpgan", "gpen"],
                value="gfpgan",
                label="Restoration Mode",
                info="GFPGAN: Best quality, GPEN: Better structure preservation",
            )
            restore_strength = gr.Slider(
                0.0,
                1.0,
                value=0.5,
                step=0.1,
                label="Restoration Strength",
                info="0 = original, 1 = fully restored",
            )

        with gr.Row():
            restore_version = gr.Dropdown(
                [1.2, 1.3, 1.4],
                value=1.4,
                label="GFPGAN Version",
                info="Only used when mode is GFPGAN",
            )
            gpen_model_size = gr.Dropdown(
                [256, 512, 1024],
                value=512,
                label="GPEN Model Size",
                info="Only used when mode is GPEN. Larger = better quality, slower",
            )
            restore_btn = gr.Button("Restore Face")

        restore_btn.click(
            fn=app.apply_face_restoration,
            inputs=[
                restore_input,
                restore_strength,
                restore_mode,
                restore_version,
                gpen_model_size,
            ],
            outputs=restore_result,
        )

        gr.Markdown("---")
        gr.Markdown("### Neural Color Transfer")
        gr.Markdown("VGG-based semantic color matching for more realistic results.")

        with gr.Row():
            nct_source = gr.Image(
                label="Style Reference (color source)",
                type="numpy",
            )
            nct_target = gr.Image(
                label="Target Image (to modify)",
                type="numpy",
            )
            nct_result = gr.Image(
                label="Result",
                type="numpy",
            )

        with gr.Row():
            nct_mode = gr.Dropdown(
                ["histogram", "statistics", "gram"],
                value="histogram",
                label="Transfer Mode",
                info="histogram=LAB space, statistics=mean/std, gram=style (requires torchvision)",
            )
            nct_strength = gr.Slider(
                0.0,
                1.0,
                value=0.8,
                step=0.1,
                label="Transfer Strength",
            )
            nct_preserve_lum = gr.Checkbox(
                value=True,
                label="Preserve Luminance",
            )
            nct_btn = gr.Button("Apply Neural Color")

        nct_btn.click(
            fn=app.apply_neural_color_transfer,
            inputs=[nct_source, nct_target, nct_mode, nct_strength, nct_preserve_lum],
            outputs=nct_result,
        )

    return {}


def create_settings_tab(app: GradioApp) -> dict[str, Any]:
    """Create application settings tab."""
    with gr.Tab("Settings"):
        gr.Markdown("### Application Settings")

        with gr.Row():
            device = gr.Dropdown(
                ["auto", "cuda", "cpu", "mps"],
                value="auto",
                label="Device",
            )
            gpu_id = gr.Number(
                value=0,
                label="GPU ID (if multiple GPUs)",
            )

        with gr.Row():
            default_batch_size = gr.Slider(
                1,
                32,
                value=8,
                step=1,
                label="Default Batch Size",
            )
            num_workers = gr.Slider(
                0,
                16,
                value=4,
                step=1,
                label="DataLoader Workers",
            )

        with gr.Row():
            workspace_dir = gr.Textbox(
                value="./workspace",
                label="Default Workspace Directory",
            )

        save_settings_btn = gr.Button("Save Settings")
        settings_status = gr.Textbox(
            value="",
            label="Status",
            interactive=False,
        )

        save_settings_btn.click(
            fn=app.save_settings,
            inputs=[device, gpu_id, default_batch_size, num_workers, workspace_dir],
            outputs=settings_status,
        )

    return {}


def create_video_tools_tab(app: GradioApp) -> dict[str, Any]:
    """Create video editing tools tab."""
    with gr.Tab("Video Tools"):
        gr.Markdown("### Video Editing Tools")
        gr.Markdown("Tools for video-to-frame and frame-to-video conversion.")

        # Extract Frames Section
        gr.Markdown("#### Extract Frames from Video")
        with gr.Row():
            with gr.Column():
                extract_input = gr.Textbox(
                    label="Input Video",
                    placeholder="./input.mp4",
                    info="Path to video file",
                )
                extract_output = gr.Textbox(
                    label="Output Directory",
                    placeholder="./frames",
                    info="Directory to save extracted frames",
                )
            with gr.Column():
                extract_fps = gr.Number(
                    label="FPS (0 = original)",
                    value=0,
                    info="Target frame rate (0 to keep original)",
                )
                extract_format = gr.Dropdown(
                    ["png", "jpg"],
                    value="png",
                    label="Output Format",
                )

        extract_btn = gr.Button("Extract Frames", variant="primary")
        extract_log = gr.Textbox(
            label="Log",
            lines=8,
            max_lines=15,
            interactive=False,
        )

        extract_btn.click(
            fn=app.run_extract_frames,
            inputs=[extract_input, extract_output, extract_fps, extract_format],
            outputs=extract_log,
        )

        gr.Markdown("---")

        # Create Video Section
        gr.Markdown("#### Create Video from Frames")
        with gr.Row():
            with gr.Column():
                create_input = gr.Textbox(
                    label="Input Directory",
                    placeholder="./frames",
                    info="Directory containing image sequence",
                )
                create_output = gr.Textbox(
                    label="Output Video",
                    placeholder="./output.mp4",
                    info="Output video path",
                )
            with gr.Column():
                create_fps = gr.Number(
                    label="FPS",
                    value=30,
                    info="Output video frame rate",
                )
                create_codec = gr.Dropdown(
                    ["libx264", "libx265", "h264_nvenc", "hevc_nvenc"],
                    value="libx264",
                    label="Codec",
                )
                create_bitrate = gr.Textbox(
                    label="Bitrate",
                    value="16M",
                    info="Video bitrate (e.g., 16M, 25M)",
                )

        create_btn = gr.Button("Create Video", variant="primary")
        create_log = gr.Textbox(
            label="Log",
            lines=8,
            max_lines=15,
            interactive=False,
        )

        create_btn.click(
            fn=app.run_create_video,
            inputs=[
                create_input,
                create_output,
                create_fps,
                create_codec,
                create_bitrate,
            ],
            outputs=create_log,
        )

        gr.Markdown("---")

        # Cut Video Section
        gr.Markdown("#### Cut Video Segment")
        with gr.Row():
            with gr.Column():
                cut_input = gr.Textbox(
                    label="Input Video",
                    placeholder="./input.mp4",
                )
                cut_output = gr.Textbox(
                    label="Output Video",
                    placeholder="./cut_output.mp4",
                )
            with gr.Column():
                cut_start = gr.Textbox(
                    label="Start Time",
                    value="00:00:00",
                    info="Format: HH:MM:SS or seconds",
                )
                cut_end = gr.Textbox(
                    label="End Time",
                    value="00:00:10",
                    info="Format: HH:MM:SS or seconds",
                )

        cut_btn = gr.Button("Cut Video", variant="primary")
        cut_log = gr.Textbox(
            label="Log",
            lines=8,
            max_lines=15,
            interactive=False,
        )

        cut_btn.click(
            fn=app.run_cut_video,
            inputs=[cut_input, cut_output, cut_start, cut_end],
            outputs=cut_log,
        )

        gr.Markdown("---")

        # Denoise Section
        gr.Markdown("#### Temporal Denoising")
        gr.Markdown("Apply temporal denoising to reduce flickering in frame sequences.")
        with gr.Row():
            with gr.Column():
                denoise_input = gr.Textbox(
                    label="Input Directory",
                    placeholder="./frames",
                    info="Directory containing image sequence",
                )
                denoise_output = gr.Textbox(
                    label="Output Directory (optional)",
                    placeholder="Leave empty for in-place",
                )
            with gr.Column():
                denoise_factor = gr.Slider(
                    3,
                    15,
                    value=7,
                    step=2,
                    label="Denoise Factor",
                    info="Temporal window size (must be odd)",
                )

        denoise_btn = gr.Button("Apply Denoising", variant="primary")
        denoise_log = gr.Textbox(
            label="Log",
            lines=8,
            max_lines=15,
            interactive=False,
        )

        denoise_btn.click(
            fn=app.run_denoise_sequence,
            inputs=[denoise_input, denoise_output, denoise_factor],
            outputs=denoise_log,
        )

    return {}


def create_faceset_tools_tab(app: GradioApp) -> dict[str, Any]:
    """Create faceset processing tools tab."""
    with gr.Tab("Faceset Tools"):
        gr.Markdown("### Faceset Processing Tools")
        gr.Markdown("Tools for enhancing and resizing face datasets.")

        # Faceset Enhancer Section
        gr.Markdown("#### Face Enhancement (GFPGAN)")
        gr.Markdown("Enhance face quality using GFPGAN restoration.")
        with gr.Row():
            with gr.Column():
                enhance_input = gr.Textbox(
                    label="Input Directory",
                    placeholder="./workspace/data_src/aligned",
                    info="Directory containing face images",
                )
                enhance_output = gr.Textbox(
                    label="Output Directory (optional)",
                    placeholder="Leave empty for auto-naming",
                    info="Output directory (default: input_enhanced)",
                )
            with gr.Column():
                enhance_strength = gr.Slider(
                    0.0,
                    1.0,
                    value=0.5,
                    step=0.1,
                    label="Enhancement Strength",
                    info="0 = original, 1 = fully enhanced",
                )
                enhance_model = gr.Dropdown(
                    [1.2, 1.3, 1.4],
                    value=1.4,
                    label="GFPGAN Version",
                )

        enhance_btn = gr.Button("Enhance Faceset", variant="primary")
        enhance_log = gr.Textbox(
            label="Log",
            lines=10,
            max_lines=20,
            interactive=False,
        )

        enhance_btn.click(
            fn=app.run_enhance_faceset,
            inputs=[enhance_input, enhance_output, enhance_strength, enhance_model],
            outputs=enhance_log,
        )

        gr.Markdown("---")

        # Faceset Resizer Section
        gr.Markdown("#### Faceset Resizing")
        gr.Markdown("Resize face images with DFL metadata preservation.")
        with gr.Row():
            with gr.Column():
                resize_input = gr.Textbox(
                    label="Input Directory",
                    placeholder="./workspace/data_src/aligned",
                    info="Directory containing face images",
                )
                resize_output = gr.Textbox(
                    label="Output Directory (optional)",
                    placeholder="Leave empty for auto-naming",
                    info="Output directory (default: input_SIZE)",
                )
            with gr.Column():
                resize_size = gr.Slider(
                    128,
                    1024,
                    value=256,
                    step=64,
                    label="Target Size",
                    info="Output image size (width = height)",
                )
                resize_face_type = gr.Dropdown(
                    [
                        "keep",
                        "half_face",
                        "mid_face",
                        "full_face",
                        "whole_face",
                        "head",
                    ],
                    value="keep",
                    label="Face Type",
                    info="Target face type (keep = preserve original)",
                )
                resize_interp = gr.Dropdown(
                    ["lanczos", "cubic", "linear", "nearest"],
                    value="lanczos",
                    label="Interpolation",
                )

        resize_btn = gr.Button("Resize Faceset", variant="primary")
        resize_log = gr.Textbox(
            label="Log",
            lines=10,
            max_lines=20,
            interactive=False,
        )

        resize_btn.click(
            fn=app.run_resize_faceset,
            inputs=[
                resize_input,
                resize_output,
                resize_size,
                resize_face_type,
                resize_interp,
            ],
            outputs=resize_log,
        )

    return {}


def create_app() -> "gr.Blocks":
    """Create Gradio application."""
    if not GRADIO_AVAILABLE:
        raise ImportError(
            "Gradio is required. Install with: pip install 'visagen[gui]'"
        )

    app_state = GradioApp()

    with gr.Blocks(
        title="Visagen - Face Swapping Framework",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown("# Visagen")
        gr.Markdown("Modern Face Swapping Framework with PyTorch Lightning")

        # Create tabs
        create_training_tab(app_state)
        create_inference_tab(app_state)
        create_extract_tab(app_state)
        create_merge_tab(app_state)
        create_sort_tab(app_state)
        create_export_tab(app_state)
        create_video_tools_tab(app_state)
        create_faceset_tools_tab(app_state)
        create_postprocess_tab(app_state)
        create_settings_tab(app_state)

        gr.Markdown("---")
        gr.Markdown(
            "Made with PyTorch Lightning | "
            "[GitHub](https://github.com/karasungur/visagen)"
        )

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

    return 0


if __name__ == "__main__":
    sys.exit(main())
