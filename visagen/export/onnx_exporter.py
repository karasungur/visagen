"""
ONNX Export utilities for Visagen.

Provides ONNXExporter class for converting PyTorch models to ONNX format
with optimization and validation capabilities.

Features:
    - Dynamic axes support (batch, height, width)
    - ONNX optimization via onnx-simplifier
    - Export validation against PyTorch
    - Configurable opset version
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from visagen.export.model_wrapper import ExportableModel

if TYPE_CHECKING:
    from visagen.export.validation import ValidationResult

logger = logging.getLogger(__name__)


@dataclass
class ExportConfig:
    """
    ONNX export configuration.

    Attributes:
        opset_version: ONNX opset version. Default: 17.
        dynamic_axes: Enable dynamic batch/height/width. Default: True.
        optimize: Run onnx-simplifier optimization. Default: True.
        input_names: Names for input tensors. Default: ["input"].
        output_names: Names for output tensors. Default: ["output"].
        verbose: Verbose export logging. Default: False.
    """

    opset_version: int = 17
    dynamic_axes: bool = True
    optimize: bool = True
    input_names: list[str] = field(default_factory=lambda: ["input"])
    output_names: list[str] = field(default_factory=lambda: ["output"])
    verbose: bool = False


class ONNXExporter:
    """
    Export PyTorch model to ONNX format.

    Handles the complete export pipeline:
    1. Load model from checkpoint
    2. Create dummy input
    3. Export with torch.onnx.export
    4. Optimize with onnx-simplifier
    5. Validate against PyTorch output

    Args:
        checkpoint_path: Path to Lightning checkpoint (.ckpt).
        output_path: Path for output ONNX file (.onnx).
        config: Export configuration. Default: None (use defaults).
        input_shape: Input tensor shape. Default: (1, 3, 256, 256).

    Example:
        >>> exporter = ONNXExporter("model.ckpt", "model.onnx")
        >>> exporter.export()
        PosixPath('model.onnx')
        >>> result = exporter.validate()
        >>> print(f"Max diff: {result.max_diff:.6f}")
    """

    def __init__(
        self,
        checkpoint_path: Path,
        output_path: Path,
        config: ExportConfig | None = None,
        input_shape: tuple[int, ...] = (1, 3, 256, 256),
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.output_path = Path(output_path)
        self.config = config or ExportConfig()
        self.input_shape = input_shape
        self._model: ExportableModel | None = None

    @property
    def model(self) -> ExportableModel:
        """Get loaded model (lazy loading)."""
        if self._model is None:
            self._model = ExportableModel.from_checkpoint(self.checkpoint_path)
            self._model.eval()
        return self._model

    def export(self) -> Path:
        """
        Export model to ONNX format.

        Returns:
            Path to exported ONNX file.

        Raises:
            RuntimeError: If export fails.
        """
        # Ensure output directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Exporting model to ONNX: {self.output_path}")

        # Load and prepare model
        model = self.model
        model.eval()

        # Create dummy input
        dummy_input = torch.randn(*self.input_shape)

        # Prepare dynamic axes
        dynamic_axes = None
        if self.config.dynamic_axes:
            dynamic_axes = {
                self.config.input_names[0]: {
                    0: "batch",
                    2: "height",
                    3: "width",
                },
                self.config.output_names[0]: {
                    0: "batch",
                    2: "height",
                    3: "width",
                },
            }
            logger.info("Dynamic axes enabled for batch, height, width")

        # Export
        with torch.no_grad():
            torch.onnx.export(
                model,
                (dummy_input,),
                str(self.output_path),
                opset_version=self.config.opset_version,
                input_names=self.config.input_names,
                output_names=self.config.output_names,
                dynamic_axes=dynamic_axes,
                do_constant_folding=True,
                verbose=self.config.verbose,
            )

        logger.info(f"ONNX model exported (opset {self.config.opset_version})")

        # Optimize if requested
        if self.config.optimize:
            self._optimize()

        # Log file size
        file_size_mb = self.output_path.stat().st_size / (1024 * 1024)
        logger.info(f"ONNX file size: {file_size_mb:.2f} MB")

        return self.output_path

    def _optimize(self) -> None:
        """
        Optimize ONNX model with onnx-simplifier.

        Performs:
        - Constant folding
        - Dead code elimination
        - Redundant node removal
        """
        try:
            import onnx
            from onnxsim import simplify

            logger.info("Optimizing ONNX model with onnx-simplifier...")

            # Load model
            model = onnx.load(str(self.output_path))

            # Check model validity
            onnx.checker.check_model(model)

            # Simplify
            model_opt, check = simplify(
                model,
                check_n=3,  # Number of check iterations
            )

            if check:
                onnx.save(model_opt, str(self.output_path))
                logger.info("ONNX optimization completed successfully")
            else:
                logger.warning("ONNX simplification check failed, keeping original")

        except ImportError:
            logger.warning(
                "onnx-simplifier not installed, skipping optimization. "
                "Install with: pip install onnx-simplifier"
            )
        except Exception as e:
            logger.warning(f"ONNX optimization failed: {e}")

    def validate(
        self,
        rtol: float = 1e-3,
        atol: float = 1e-5,
        num_tests: int = 3,
    ) -> "ValidationResult":
        """
        Validate ONNX output against PyTorch.

        Runs multiple random inputs through both models and compares outputs.

        Args:
            rtol: Relative tolerance for comparison. Default: 1e-3.
            atol: Absolute tolerance for comparison. Default: 1e-5.
            num_tests: Number of random inputs to test. Default: 3.

        Returns:
            ValidationResult with comparison metrics.

        Raises:
            FileNotFoundError: If ONNX file doesn't exist.
        """
        if not self.output_path.exists():
            raise FileNotFoundError(
                f"ONNX file not found: {self.output_path}. Run export() first."
            )

        from visagen.export.validation import validate_export

        logger.info(f"Validating ONNX export with {num_tests} random inputs...")

        # Run validation with random inputs
        max_diffs = []
        mean_diffs = []

        for i in range(num_tests):
            test_input = torch.randn(*self.input_shape)
            result = validate_export(
                self.model,
                self.output_path,
                test_input,
                rtol=rtol,
                atol=atol,
            )
            max_diffs.append(result.max_diff)
            mean_diffs.append(result.mean_diff)

            if not result.passed:
                logger.warning(
                    f"Validation test {i + 1} failed: max_diff={result.max_diff:.6f}"
                )

        # Aggregate results
        from visagen.export.validation import ValidationResult

        overall_max = max(max_diffs)
        overall_mean = sum(mean_diffs) / len(mean_diffs)
        overall_passed = overall_max <= atol + rtol * overall_max

        final_result = ValidationResult(
            passed=overall_passed,
            max_diff=overall_max,
            mean_diff=overall_mean,
            rtol=rtol,
            atol=atol,
        )

        if final_result.passed:
            logger.info(f"Validation PASSED (max diff: {final_result.max_diff:.6f})")
        else:
            logger.error(f"Validation FAILED (max diff: {final_result.max_diff:.6f})")

        return final_result

    def export_with_metadata(
        self,
        metadata: dict | None = None,
    ) -> Path:
        """
        Export model with custom metadata.

        Args:
            metadata: Custom metadata to embed in ONNX model.

        Returns:
            Path to exported ONNX file.
        """
        # First do standard export
        self.export()

        # Add metadata if provided
        if metadata:
            try:
                import onnx

                model = onnx.load(str(self.output_path))

                for key, value in metadata.items():
                    meta = model.metadata_props.add()
                    meta.key = str(key)
                    meta.value = str(value)

                onnx.save(model, str(self.output_path))
                logger.info(f"Added {len(metadata)} metadata entries")

            except Exception as e:
                logger.warning(f"Failed to add metadata: {e}")

        return self.output_path
