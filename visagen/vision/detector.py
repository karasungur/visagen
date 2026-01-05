"""
Face Detection using InsightFace SCRFD.

SCRFD (Sample and Computation Redistribution for Face Detection) provides
fast and accurate face detection, especially for small faces.
"""

from dataclasses import dataclass

import numpy as np

try:
    from insightface.app import FaceAnalysis
    from insightface.data import get_image as ins_get_image

    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False


@dataclass
class DetectedFace:
    """
    Detected face information.

    Attributes:
        bbox: Bounding box as (x1, y1, x2, y2).
        confidence: Detection confidence score (0-1).
        landmarks: 5-point facial landmarks (eyes, nose, mouth corners).
        embedding: Face embedding vector (if available).
    """

    bbox: np.ndarray  # Shape: (4,) - x1, y1, x2, y2
    confidence: float
    landmarks: np.ndarray | None = None  # Shape: (5, 2) - 5 keypoints
    embedding: np.ndarray | None = None  # Shape: (512,) - ArcFace embedding

    @property
    def x1(self) -> int:
        return int(self.bbox[0])

    @property
    def y1(self) -> int:
        return int(self.bbox[1])

    @property
    def x2(self) -> int:
        return int(self.bbox[2])

    @property
    def y2(self) -> int:
        return int(self.bbox[3])

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center(self) -> tuple[int, int]:
        return (self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2

    def to_rect(self) -> tuple[int, int, int, int]:
        """Return bbox as (left, top, right, bottom) tuple."""
        return (self.x1, self.y1, self.x2, self.y2)


class FaceDetector:
    """
    Face detector using InsightFace SCRFD.

    Uses the 'buffalo_l' model pack which includes:
    - SCRFD face detection
    - 2D106 landmark detection
    - ArcFace recognition (optional)

    Args:
        model_name: InsightFace model pack name. Default: "buffalo_l".
        ctx_id: GPU device ID (-1 for CPU). Default: 0.
        det_thresh: Detection confidence threshold. Default: 0.5.
        det_size: Detection input size. Default: (640, 640).

    Example:
        >>> detector = FaceDetector()
        >>> image = cv2.imread("photo.jpg")
        >>> faces = detector.detect(image)
        >>> for face in faces:
        ...     print(f"Face at {face.bbox} with confidence {face.confidence:.2f}")
    """

    def __init__(
        self,
        model_name: str = "buffalo_l",
        ctx_id: int = 0,
        det_thresh: float = 0.5,
        det_size: tuple[int, int] = (640, 640),
    ) -> None:
        if not INSIGHTFACE_AVAILABLE:
            raise ImportError(
                "InsightFace is not installed. "
                "Install with: pip install insightface onnxruntime-gpu"
            )

        self.model_name = model_name
        self.ctx_id = ctx_id
        self.det_thresh = det_thresh
        self.det_size = det_size

        # Initialize face analysis
        self._app = FaceAnalysis(
            name=model_name,
            providers=self._get_providers(ctx_id),
        )
        self._app.prepare(ctx_id=ctx_id, det_thresh=det_thresh, det_size=det_size)

    def _get_providers(self, ctx_id: int) -> list[str]:
        """Get ONNX Runtime execution providers based on device."""
        if ctx_id >= 0:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    def detect(
        self,
        image: np.ndarray,
        max_faces: int | None = None,
        sort_by_size: bool = True,
    ) -> list[DetectedFace]:
        """
        Detect faces in an image.

        Args:
            image: Input image as numpy array (BGR format).
            max_faces: Maximum number of faces to return. Default: None (all).
            sort_by_size: Sort faces by area (largest first). Default: True.

        Returns:
            List of DetectedFace objects.
        """
        # Run detection
        faces = self._app.get(image)

        # Convert to DetectedFace objects
        detected = []
        for face in faces:
            detected.append(
                DetectedFace(
                    bbox=face.bbox.astype(np.float32),
                    confidence=float(face.det_score),
                    landmarks=face.kps if hasattr(face, "kps") else None,
                    embedding=face.embedding if hasattr(face, "embedding") else None,
                )
            )

        # Sort by size (largest first)
        if sort_by_size:
            detected.sort(key=lambda f: f.area, reverse=True)

        # Limit number of faces
        if max_faces is not None and max_faces > 0:
            detected = detected[:max_faces]

        return detected

    def detect_with_rotation(
        self,
        image: np.ndarray,
        max_faces: int | None = None,
        rotations: list[int] = None,
    ) -> tuple[list[DetectedFace], int]:
        """
        Detect faces with automatic rotation handling.

        Tries multiple rotations if no faces are found in the original orientation.

        Args:
            image: Input image as numpy array (BGR format).
            max_faces: Maximum number of faces to return.
            rotations: List of rotation angles to try. Default: [0, 90, 270, 180].

        Returns:
            Tuple of (detected faces, rotation angle used).
        """
        if rotations is None:
            rotations = [0, 90, 270, 180]

        for rotation in rotations:
            rotated = self._rotate_image(image, rotation)
            faces = self.detect(rotated, max_faces=max_faces)

            if len(faces) > 0:
                # Adjust coordinates back to original orientation
                if rotation != 0:
                    faces = self._unrotate_faces(faces, image.shape, rotation)
                return faces, rotation

        return [], 0

    def _rotate_image(self, image: np.ndarray, angle: int) -> np.ndarray:
        """Rotate image by 0, 90, 180, or 270 degrees."""
        if angle == 0:
            return image
        elif angle == 90:
            return image.swapaxes(0, 1)[:, ::-1, :]
        elif angle == 180:
            return image[::-1, ::-1, :]
        elif angle == 270:
            return image.swapaxes(0, 1)[::-1, :, :]
        return image

    def _unrotate_faces(
        self,
        faces: list[DetectedFace],
        original_shape: tuple[int, int, int],
        rotation: int,
    ) -> list[DetectedFace]:
        """Transform face coordinates back to original image orientation."""
        h, w = original_shape[:2]
        result = []

        for face in faces:
            x1, y1, x2, y2 = face.bbox
            landmarks = face.landmarks.copy() if face.landmarks is not None else None

            if rotation == 90:
                new_bbox = np.array([y1, h - x2, y2, h - x1])
                if landmarks is not None:
                    new_landmarks = landmarks[:, ::-1].copy()
                    new_landmarks[:, 1] = h - new_landmarks[:, 1]
                    landmarks = new_landmarks
            elif rotation == 180:
                new_bbox = np.array([w - x2, h - y2, w - x1, h - y1])
                if landmarks is not None:
                    landmarks[:, 0] = w - landmarks[:, 0]
                    landmarks[:, 1] = h - landmarks[:, 1]
            elif rotation == 270:
                new_bbox = np.array([w - y2, x1, w - y1, x2])
                if landmarks is not None:
                    new_landmarks = landmarks[:, ::-1].copy()
                    new_landmarks[:, 0] = w - new_landmarks[:, 0]
                    landmarks = new_landmarks
            else:
                new_bbox = face.bbox

            result.append(
                DetectedFace(
                    bbox=new_bbox,
                    confidence=face.confidence,
                    landmarks=landmarks,
                    embedding=face.embedding,
                )
            )

        return result

    def set_threshold(self, threshold: float) -> None:
        """Update detection confidence threshold."""
        self.det_thresh = threshold
        self._app.prepare(
            ctx_id=self.ctx_id,
            det_thresh=threshold,
            det_size=self.det_size,
        )
