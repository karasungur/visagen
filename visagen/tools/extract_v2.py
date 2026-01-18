"""
Modern Face Extraction Tool.

Extract and align faces from images and videos using
InsightFace detection and SegFormer segmentation.
"""

import argparse
import logging
import sys
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from visagen.vision.aligner import FaceAligner
from visagen.vision.detector import FaceDetector
from visagen.vision.dflimg import DFLImage, FaceMetadata
from visagen.vision.face_type import FaceType
from visagen.vision.segmenter import FaceSegmenter

logger = logging.getLogger(__name__)

# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Supported video extensions
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".webm"}


@dataclass
class ExtractedFace:
    """
    Result of face extraction.

    Attributes:
        image: Aligned face image (BGR, uint8).
        metadata: DFL-compatible face metadata.
        confidence: Detection confidence score.
        face_index: Index of face in source (for multi-face).
    """

    image: np.ndarray
    metadata: FaceMetadata
    confidence: float
    face_index: int = 0


@dataclass
class ExtractionProgress:
    """Progress information for extraction."""

    current_frame: int
    total_frames: int
    faces_extracted: int
    current_face: ExtractedFace | None = None
    source_name: str = ""


class FaceExtractor:
    """
    Extract and align faces from images and videos.

    Uses InsightFace SCRFD for detection, AntelopeV2 for landmarks,
    and SegFormer for segmentation masks.

    Args:
        output_size: Size of extracted face images. Default: 512.
        face_type: Face type for alignment. Default: WHOLE_FACE.
        jpeg_quality: JPEG quality for saved images. Default: 95.
        min_confidence: Minimum detection confidence. Default: 0.5.
        device: Device for inference. Default: auto-detect.

    Example:
        >>> extractor = FaceExtractor(output_size=512)
        >>> faces = extractor.extract_from_image(Path("photo.jpg"))
        >>> for face in faces:
        ...     print(f"Face {face.face_index}: {face.confidence:.2f}")
    """

    def __init__(
        self,
        output_size: int = 512,
        face_type: FaceType = FaceType.WHOLE_FACE,
        jpeg_quality: int = 95,
        min_confidence: float = 0.5,
        device: str | None = None,
        model_name: str = "antelopev2",
    ) -> None:
        self.output_size = output_size
        self.face_type = face_type
        self.jpeg_quality = jpeg_quality
        self.min_confidence = min_confidence
        self._stop_requested = False

        # Initialize vision components
        self.detector = FaceDetector(model_name=model_name, device=device)
        self.aligner = FaceAligner()
        self.segmenter: FaceSegmenter | None = None  # Lazy load

    def request_stop(self) -> None:
        """Request extraction to stop gracefully."""
        self._stop_requested = True

    def reset_stop(self) -> None:
        """Reset stop flag for new extraction."""
        self._stop_requested = False

    def _ensure_segmenter(self) -> FaceSegmenter:
        """Lazy-load segmenter on first use."""
        if self.segmenter is None:
            self.segmenter = FaceSegmenter()
        return self.segmenter

    def extract_from_image(
        self,
        image_path: Path,
        with_mask: bool = True,
    ) -> list[ExtractedFace]:
        """
        Extract all faces from a single image.

        Args:
            image_path: Path to source image.
            with_mask: Generate segmentation masks. Default: True.

        Returns:
            List of ExtractedFace objects.
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Load image
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        return self._extract_from_array(
            image,
            source_filename=image_path.name,
            with_mask=with_mask,
        )

    def _extract_from_array(
        self,
        image: np.ndarray,
        source_filename: str,
        with_mask: bool = True,
    ) -> list[ExtractedFace]:
        """Extract faces from numpy array."""
        results = []

        # Detect faces
        faces = self.detector.detect(image)

        for idx, face in enumerate(faces):
            # Filter by confidence
            if face.confidence < self.min_confidence:
                continue

            # Get 68-point landmarks from 106-point
            landmarks_106 = face.landmarks
            landmarks_68 = self.aligner.convert_106_to_68(landmarks_106)

            # Align face
            aligned = self.aligner.align_face(
                image,
                landmarks_68,
                self.face_type,
                self.output_size,
            )

            # Generate segmentation mask if requested
            xseg_mask = None
            if with_mask:
                segmenter = self._ensure_segmenter()
                seg_result = segmenter.segment(aligned.image)
                # Compress mask for storage
                ret, buf = cv2.imencode(".png", seg_result.mask)
                if ret:
                    xseg_mask = buf.tobytes()

            # Build metadata
            x1, y1, x2, y2 = map(int, face.bbox)
            metadata = FaceMetadata(
                landmarks=aligned.landmarks,
                source_landmarks=landmarks_68,
                source_rect=(x1, y1, x2, y2),
                source_filename=source_filename,
                face_type=FaceType.to_string(self.face_type),
                image_to_face_mat=aligned.transform_matrix,
                eyebrows_expand_mod=1.0,
                xseg_mask=xseg_mask,
            )

            results.append(
                ExtractedFace(
                    image=aligned.image,
                    metadata=metadata,
                    confidence=face.confidence,
                    face_index=idx,
                )
            )

        return results

    def extract_from_video(
        self,
        video_path: Path,
        output_dir: Path,
        frame_skip: int = 1,
        with_mask: bool = True,
        max_frames: int | None = None,
    ) -> int:
        """
        Extract faces from video file.

        Args:
            video_path: Path to source video.
            output_dir: Directory for extracted faces.
            frame_skip: Process every Nth frame. Default: 1 (all frames).
            with_mask: Generate segmentation masks. Default: True.
            max_frames: Maximum frames to process. Default: None (all).

        Returns:
            Number of faces extracted.
        """
        video_path = Path(video_path)
        output_dir = Path(output_dir)

        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        output_dir.mkdir(parents=True, exist_ok=True)

        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if max_frames:
            total_frames = min(total_frames, max_frames * frame_skip)

        face_count = 0
        frame_idx = 0

        try:
            with tqdm(total=total_frames // frame_skip, desc="Extracting") as pbar:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if max_frames and frame_idx >= max_frames * frame_skip:
                        break

                    # Skip frames
                    if frame_idx % frame_skip != 0:
                        frame_idx += 1
                        continue

                    # Extract faces
                    source_name = f"{video_path.stem}_{frame_idx:06d}"
                    faces = self._extract_from_array(
                        frame,
                        source_filename=source_name,
                        with_mask=with_mask,
                    )

                    # Save extracted faces
                    for face in faces:
                        output_name = f"{source_name}_{face.face_index}.jpg"
                        output_path = output_dir / output_name

                        DFLImage.save(
                            output_path,
                            face.image,
                            face.metadata,
                            quality=self.jpeg_quality,
                        )
                        face_count += 1

                    frame_idx += 1
                    pbar.update(1)

        finally:
            cap.release()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

        return face_count

    def extract_streaming(
        self,
        video_path: Path,
        output_dir: Path,
        frame_skip: int = 1,
        with_mask: bool = True,
        max_frames: int | None = None,
    ) -> Generator[tuple[ExtractedFace, ExtractionProgress], None, int]:
        """
        Extract faces with streaming progress.

        Yields:
            Tuple of (ExtractedFace, ExtractionProgress) for each face.

        Returns:
            Total number of faces extracted.
        """
        self.reset_stop()
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if max_frames:
            total_frames = min(total_frames, max_frames * frame_skip)

        face_count = 0
        frame_idx = 0

        try:
            while frame_idx < total_frames:
                if self._stop_requested:
                    break

                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % frame_skip != 0:
                    frame_idx += 1
                    continue

                source_name = f"{video_path.stem}_{frame_idx:06d}"
                faces = self._extract_from_array(frame, source_name, with_mask)

                for face in faces:
                    face_count += 1

                    # Save to disk
                    output_name = f"{source_name}_{face.face_index}.jpg"
                    DFLImage.save(
                        output_dir / output_name,
                        face.image,
                        face.metadata,
                        quality=self.jpeg_quality,
                    )

                    progress = ExtractionProgress(
                        current_frame=frame_idx,
                        total_frames=total_frames,
                        faces_extracted=face_count,
                        current_face=face,
                        source_name=source_name,
                    )

                    yield (face, progress)

                frame_idx += 1

        finally:
            cap.release()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

        return face_count

    def extract_from_directory(
        self,
        input_dir: Path,
        output_dir: Path,
        with_mask: bool = True,
        recursive: bool = False,
    ) -> int:
        """
        Extract faces from all images in a directory.

        Args:
            input_dir: Directory containing source images.
            output_dir: Directory for extracted faces.
            with_mask: Generate segmentation masks. Default: True.
            recursive: Search subdirectories. Default: False.

        Returns:
            Number of faces extracted.
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)

        if not input_dir.exists():
            raise FileNotFoundError(f"Directory not found: {input_dir}")

        output_dir.mkdir(parents=True, exist_ok=True)

        # Find all images
        pattern = "**/*" if recursive else "*"
        image_files = [
            f for f in input_dir.glob(pattern) if f.suffix.lower() in IMAGE_EXTENSIONS
        ]

        face_count = 0

        for image_path in tqdm(image_files, desc="Processing images"):
            try:
                faces = self.extract_from_image(image_path, with_mask=with_mask)

                for face in faces:
                    output_name = f"{image_path.stem}_{face.face_index}.jpg"
                    output_path = output_dir / output_name

                    DFLImage.save(
                        output_path,
                        face.image,
                        face.metadata,
                        quality=self.jpeg_quality,
                    )
                    face_count += 1

            except Exception as e:
                print(f"Error processing {image_path}: {e}", file=sys.stderr)
                continue

        return face_count

    def cleanup(self) -> None:
        """Cleanup GPU resources."""
        try:
            if hasattr(self, "detector"):
                del self.detector
            if hasattr(self, "aligner"):
                del self.aligner
            if hasattr(self, "segmenter") and self.segmenter is not None:
                del self.segmenter

            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def __del__(self) -> None:
        """Cleanup on deletion."""
        self.cleanup()


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Extract faces from images and videos.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "input",
        type=Path,
        help="Input image, video, or directory",
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Output directory for extracted faces",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=512,
        help="Output face size (default: 512)",
    )
    parser.add_argument(
        "--face-type",
        type=str,
        default="whole_face",
        choices=["half", "mid_full", "full", "whole_face", "head"],
        help="Face type for alignment (default: whole_face)",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=95,
        help="JPEG quality 1-100 (default: 95)",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.5,
        help="Minimum detection confidence (default: 0.5)",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=1,
        help="Process every Nth frame for videos (default: 1)",
    )
    parser.add_argument(
        "--no-mask",
        action="store_true",
        help="Skip segmentation mask generation",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search subdirectories for images",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for inference (cuda/cpu, default: auto)",
    )

    args = parser.parse_args()

    # Parse face type
    face_type_map = {
        "half": FaceType.HALF,
        "mid_full": FaceType.MID_FULL,
        "full": FaceType.FULL,
        "whole_face": FaceType.WHOLE_FACE,
        "head": FaceType.HEAD,
    }
    face_type = face_type_map[args.face_type]

    # Create extractor
    extractor = FaceExtractor(
        output_size=args.size,
        face_type=face_type,
        jpeg_quality=args.quality,
        min_confidence=args.min_confidence,
        device=args.device,
    )

    input_path = args.input
    output_path = args.output
    with_mask = not args.no_mask

    # Determine input type and process
    if input_path.is_file():
        suffix = input_path.suffix.lower()

        if suffix in IMAGE_EXTENSIONS:
            # Single image
            output_path.mkdir(parents=True, exist_ok=True)
            faces = extractor.extract_from_image(input_path, with_mask=with_mask)

            for face in faces:
                out_name = f"{input_path.stem}_{face.face_index}.jpg"
                DFLImage.save(
                    output_path / out_name,
                    face.image,
                    face.metadata,
                    quality=args.quality,
                )

            print(f"Extracted {len(faces)} face(s) from {input_path.name}")
            return 0

        elif suffix in VIDEO_EXTENSIONS:
            # Video file
            count = extractor.extract_from_video(
                input_path,
                output_path,
                frame_skip=args.frame_skip,
                with_mask=with_mask,
            )
            print(f"Extracted {count} face(s) from {input_path.name}")
            return 0

        else:
            print(f"Unsupported file type: {suffix}", file=sys.stderr)
            return 1

    elif input_path.is_dir():
        # Directory of images
        count = extractor.extract_from_directory(
            input_path,
            output_path,
            with_mask=with_mask,
            recursive=args.recursive,
        )
        print(f"Extracted {count} face(s) from {input_path}")
        return 0

    else:
        print(f"Input not found: {input_path}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
