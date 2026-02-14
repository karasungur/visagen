"""
Face Alignment using 3D Landmarks and Umeyama Transform.

Provides face alignment using AntelopeV2 106-point landmarks
and Umeyama similarity transform, compatible with legacy DFL format.
"""

import logging
import math
from dataclasses import dataclass
from typing import cast

import cv2
import numpy as np
import numpy.linalg as npla

from visagen.vision.face_type import FACE_TYPE_TO_PADDING, FaceType

logger = logging.getLogger(__name__)

# Standard 2D landmarks for alignment (normalized coordinates)
# These define the "ideal" face position in output image
LANDMARKS_2D_NEW = np.array(
    [
        [0.000213256, 0.106454],  # 17
        [0.0752622, 0.038915],  # 18
        [0.18113, 0.0187482],  # 19
        [0.29077, 0.0344891],  # 20
        [0.393397, 0.0773906],  # 21
        [0.586856, 0.0773906],  # 22
        [0.689483, 0.0344891],  # 23
        [0.799124, 0.0187482],  # 24
        [0.904991, 0.038915],  # 25
        [0.98004, 0.106454],  # 26
        [0.490127, 0.203352],  # 27
        [0.490127, 0.307009],  # 28
        [0.490127, 0.409805],  # 29
        [0.490127, 0.515625],  # 30
        [0.36688, 0.587326],  # 31
        [0.426036, 0.609345],  # 32
        [0.490127, 0.628106],  # 33
        [0.554217, 0.609345],  # 34
        [0.613373, 0.587326],  # 35
        [0.121737, 0.216423],  # 36
        [0.187122, 0.178758],  # 37
        [0.265825, 0.179852],  # 38
        [0.334606, 0.231733],  # 39
        [0.260918, 0.245099],  # 40
        [0.182743, 0.244077],  # 41
        [0.645647, 0.231733],  # 42
        [0.714428, 0.179852],  # 43
        [0.793132, 0.178758],  # 44
        [0.858516, 0.216423],  # 45
        [0.79751, 0.244077],  # 46
        [0.719335, 0.245099],  # 47
        [0.254149, 0.780233],  # 48
        [0.726104, 0.780233],  # 54
    ],
    dtype=np.float32,
)

# 3D landmarks for pose estimation (68 points)
LANDMARKS_68_3D = np.array(
    [
        [-73.393523, -29.801432, 47.667532],
        [-72.775014, -10.949766, 45.909403],
        [-70.533638, 7.929818, 44.842580],
        [-66.850058, 26.074280, 43.141114],
        [-59.790187, 42.564390, 38.635298],
        [-48.368973, 56.481080, 30.750622],
        [-34.121101, 67.246992, 18.456453],
        [-17.875411, 75.056892, 3.609035],
        [0.098749, 77.061286, -0.881698],
        [17.477031, 74.758448, 5.181201],
        [32.648966, 66.929021, 19.176563],
        [46.372358, 56.311389, 30.770570],
        [57.343480, 42.419126, 37.628629],
        [64.388482, 25.455880, 40.886309],
        [68.212038, 6.990805, 42.281449],
        [70.486405, -11.666193, 44.142567],
        [71.375822, -30.365191, 47.140426],
        [-61.119406, -49.361602, 14.254422],
        [-51.287588, -58.769795, 7.268147],
        [-37.804800, -61.996155, 0.442051],
        [-24.022754, -61.033399, -6.606501],
        [-11.635713, -56.686759, -11.967398],
        [12.056636, -57.391033, -12.051204],
        [25.106256, -61.902186, -7.315098],
        [38.338588, -62.777713, -1.022953],
        [51.191007, -59.302347, 5.349435],
        [60.053851, -50.190255, 11.615746],
        [0.653940, -42.193790, -13.380835],
        [0.804809, -30.993721, -21.150853],
        [0.992204, -19.944596, -29.284036],
        [1.226783, -8.414541, -36.948060],
        [-14.772472, 2.598255, -20.132003],
        [-7.180239, 4.751589, -23.536684],
        [0.555920, 6.562900, -25.944448],
        [8.272499, 4.661005, -23.695741],
        [15.214351, 2.643046, -20.858157],
        [-46.047290, -37.471411, 7.037989],
        [-37.674688, -42.730510, 3.021217],
        [-27.883856, -42.711517, 1.353629],
        [-19.648268, -36.754742, -0.111088],
        [-28.272965, -35.134493, -0.147273],
        [-38.082418, -34.919043, 1.476612],
        [19.265868, -37.032306, -0.665746],
        [27.894191, -43.342445, 0.247660],
        [37.437529, -43.110822, 1.696435],
        [45.170805, -38.086515, 4.894163],
        [38.196454, -35.532024, 0.282961],
        [28.764989, -35.484289, -1.172675],
        [-28.916267, 28.612716, -2.240310],
        [-17.533194, 22.172187, -15.934335],
        [-6.684590, 19.029051, -22.611355],
        [0.381001, 20.721118, -23.748437],
        [8.375443, 19.035460, -22.721995],
        [18.876618, 22.394109, -15.610679],
        [28.794412, 28.079924, -3.217393],
        [19.057574, 36.298248, -14.987997],
        [8.956375, 39.634575, -22.554245],
        [0.381549, 40.395647, -23.591626],
        [-7.428895, 39.836405, -22.406106],
        [-18.160634, 36.677899, -15.121907],
        [-24.377490, 28.677771, -4.785684],
        [-6.897633, 25.475976, -20.893742],
        [0.340663, 26.014269, -22.220479],
        [8.444722, 25.326198, -21.025520],
        [24.474473, 28.323008, -5.712776],
        [8.449166, 30.596216, -20.671489],
        [0.205322, 31.408738, -21.903670],
        [-7.198266, 30.844876, -20.328022],
    ],
    dtype=np.float32,
)


def umeyama(
    src: np.ndarray, dst: np.ndarray, estimate_scale: bool = True
) -> np.ndarray:
    """
    Estimate N-D similarity transformation with or without scaling.

    Parameters:
        src: Source points, shape (n, d).
        dst: Destination points, shape (n, d).
        estimate_scale: Whether to estimate scaling. Default: True.

    Returns:
        Transformation matrix of shape (d+1, d+1).
    """
    num = src.shape[0]
    dim = src.shape[1]

    # Compute mean of src and dst
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)

    # Subtract mean from src and dst
    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    # Covariance matrix
    A = dst_demean.T @ src_demean / num

    d = np.ones((dim,), dtype=np.float64)
    if npla.det(A) < 0:
        d[dim - 1] = -1

    T = np.eye(dim + 1, dtype=np.float64)

    U, S, V = npla.svd(A)

    # Rotation
    rank = npla.matrix_rank(A)
    if rank == 0:
        return np.nan * T

    if rank == dim - 1:
        if npla.det(U) * npla.det(V) > 0:
            T[:dim, :dim] = U @ V
        else:
            s = d[dim - 1]
            d[dim - 1] = -1
            T[:dim, :dim] = U @ np.diag(d) @ V
            d[dim - 1] = s
    else:
        T[:dim, :dim] = U @ np.diag(d) @ V

    # Scale
    if estimate_scale:
        scale = 1.0 / src_demean.var(axis=0).sum() * (S @ d)
    else:
        scale = 1.0

    T[:dim, dim] = dst_mean - scale * (T[:dim, :dim] @ src_mean.T)
    T[:dim, :dim] *= scale

    return T


def transform_points(
    points: np.ndarray,
    mat: np.ndarray,
    invert: bool = False,
) -> np.ndarray:
    """
    Apply affine transformation to points.

    Args:
        points: Points array of shape (n, 2).
        mat: Affine transform matrix of shape (2, 3).
        invert: Whether to invert the transform. Default: False.

    Returns:
        Transformed points of shape (n, 2).
    """
    if invert:
        mat = cv2.invertAffineTransform(mat)

    points = np.expand_dims(points, axis=1)
    points = cv2.transform(points, mat)
    points = np.squeeze(points)

    return points


@dataclass
class AlignedFace:
    """
    Aligned face result.

    Attributes:
        image: Aligned face image.
        landmarks: Landmarks in aligned image space.
        source_landmarks: Original landmarks in source image.
        transform_matrix: Affine transform from source to aligned.
        inverse_matrix: Affine transform from aligned to source.
    """

    image: np.ndarray
    landmarks: np.ndarray
    source_landmarks: np.ndarray
    transform_matrix: np.ndarray
    inverse_matrix: np.ndarray


class FaceAligner:
    """
    Face alignment using 3D landmarks and Umeyama transform.

    Aligns faces to a canonical position using detected landmarks,
    supporting various face types (HALF, FULL, WHOLE_FACE, HEAD).

    Example:
        >>> aligner = FaceAligner()
        >>> aligned = aligner.align_face(image, landmarks, FaceType.WHOLE_FACE, 512)
        >>> aligned_image = aligned.image
    """

    def __init__(
        self,
        output_size: int = 256,
        face_type: str | FaceType = FaceType.WHOLE_FACE,
    ) -> None:
        """
        Initialize face aligner.

        Args:
            output_size: Output image size (square). Default: 256.
            face_type: Face type determining crop region. Default: WHOLE_FACE.
        """
        self._template = LANDMARKS_2D_NEW
        self.output_size = output_size
        self.face_type = (
            FaceType.from_string(face_type) if isinstance(face_type, str) else face_type
        )

    def get_transform_mat(
        self,
        image_landmarks: np.ndarray,
        output_size: int,
        face_type: FaceType,
        scale: float = 1.0,
    ) -> np.ndarray:
        """
        Compute affine transform matrix for face alignment.

        Args:
            image_landmarks: 68-point facial landmarks.
            output_size: Output image size (square).
            face_type: Face type determining crop region.
            scale: Additional scaling factor. Default: 1.0.

        Returns:
            Affine transform matrix of shape (2, 3).
        """
        if not isinstance(image_landmarks, np.ndarray):
            image_landmarks = np.array(image_landmarks)

        # Get padding and alignment flag for face type
        padding, remove_align = FACE_TYPE_TO_PADDING.get(face_type, (0.0, False))

        # Estimate landmarks transform from global to local aligned space [0..1]
        # Use landmarks 17-48 (eyebrows, eyes, nose) + point 54 (mouth corner)
        src_points = np.concatenate([image_landmarks[17:49], image_landmarks[54:55]])
        mat = umeyama(src_points, self._template, True)[:2]

        # Get corner points in global space
        g_p = transform_points(
            np.array([(0, 0), (1, 0), (1, 1), (0, 1), (0.5, 0.5)], dtype=np.float32),
            mat,
            invert=True,
        )
        g_c = g_p[4]  # Center point

        # Calculate diagonal vectors between corners in global space
        tb_diag_vec = (g_p[2] - g_p[0]).astype(np.float32)
        tb_diag_vec /= npla.norm(tb_diag_vec)
        bt_diag_vec = (g_p[1] - g_p[3]).astype(np.float32)
        bt_diag_vec /= npla.norm(bt_diag_vec)

        # Calculate modifier of diagonal vectors for scale and padding
        mod = (1.0 / scale) * (
            npla.norm(g_p[0] - g_p[2]) * (padding * np.sqrt(2.0) + 0.5)
        )

        # Adjust center based on face type
        if face_type == FaceType.WHOLE_FACE:
            # Adjust vertical offset, 7% below to cover more forehead
            vec = (g_p[0] - g_p[3]).astype(np.float32)
            vec_len = npla.norm(vec)
            vec /= vec_len
            g_c = g_c + vec * vec_len * 0.07

        elif face_type in (FaceType.HEAD, FaceType.HEAD_NO_ALIGN):
            # Adjust for HEAD type using yaw estimation
            yaw = self._estimate_averaged_yaw(
                transform_points(image_landmarks, mat, invert=False)
            )

            hvec = (g_p[0] - g_p[1]).astype(np.float32)
            hvec_len = npla.norm(hvec)
            hvec /= hvec_len

            # Damp near zero
            yaw *= np.abs(math.tanh(yaw * 2))
            g_c = g_c - hvec * (yaw * hvec_len / 2.0)

            # Adjust vertical offset, 50% below
            vvec = (g_p[0] - g_p[3]).astype(np.float32)
            vvec_len = npla.norm(vvec)
            vvec /= vvec_len
            g_c = g_c + vvec * vvec_len * 0.50

        # Calculate 3 points in global space for affine transform
        if not remove_align:
            l_t = np.array(
                [
                    g_c - tb_diag_vec * mod,
                    g_c + bt_diag_vec * mod,
                    g_c + tb_diag_vec * mod,
                ]
            )
        else:
            # Remove alignment - face centered but not rotated
            l_t = np.array(
                [
                    g_c - tb_diag_vec * mod,
                    g_c + bt_diag_vec * mod,
                    g_c + tb_diag_vec * mod,
                    g_c - bt_diag_vec * mod,
                ]
            )

            # Get area of face square in global space
            area = self._polygon_area(l_t[:, 0], l_t[:, 1])
            side = np.float32(math.sqrt(area) / 2)

            # Calculate 3 points with unrotated square
            l_t = np.array(
                [
                    g_c + [-side, -side],
                    g_c + [side, -side],
                    g_c + [side, side],
                ]
            )

        # Calculate affine transform from 3 global points to output size
        pts2 = np.array(
            [
                (0, 0),
                (output_size, 0),
                (output_size, output_size),
            ],
            dtype=np.float32,
        )
        mat = cv2.getAffineTransform(l_t.astype(np.float32), pts2.astype(np.float32))

        return cast(np.ndarray, mat)

    def align(
        self,
        image: np.ndarray,
        landmarks: np.ndarray,
        scale: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Align a face using instance configuration.

        Simplified interface that uses the output_size and face_type
        set during initialization.

        Args:
            image: Source image (BGR format).
            landmarks: 68-point facial landmarks.
            scale: Additional scaling factor. Default: 1.0.

        Returns:
            Tuple of (aligned_image, transform_matrix).
        """
        mat = self.get_transform_mat(landmarks, self.output_size, self.face_type, scale)

        # Warp image
        aligned_image = cv2.warpAffine(
            image,
            mat,
            (self.output_size, self.output_size),
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_REPLICATE,
        )

        return aligned_image, mat

    def align_face(
        self,
        image: np.ndarray,
        landmarks: np.ndarray,
        face_type: FaceType,
        output_size: int,
        scale: float = 1.0,
    ) -> AlignedFace:
        """
        Align a face to canonical position.

        Args:
            image: Source image (BGR format).
            landmarks: 68-point facial landmarks.
            face_type: Face type determining crop region.
            output_size: Output image size (square).
            scale: Additional scaling factor. Default: 1.0.

        Returns:
            AlignedFace containing aligned image and metadata.
        """
        mat = self.get_transform_mat(landmarks, output_size, face_type, scale)
        inv_mat = cv2.invertAffineTransform(mat)

        # Warp image
        aligned_image = cv2.warpAffine(
            image,
            mat,
            (output_size, output_size),
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_REPLICATE,
        )

        # Transform landmarks to aligned space
        aligned_landmarks = transform_points(landmarks, mat, invert=False)

        return AlignedFace(
            image=aligned_image,
            landmarks=aligned_landmarks,
            source_landmarks=landmarks,
            transform_matrix=mat,
            inverse_matrix=inv_mat,
        )

    def estimate_pitch_yaw_roll(
        self,
        aligned_landmarks: np.ndarray,
        size: int = 256,
    ) -> tuple[float, float, float]:
        """
        Estimate face pose (pitch, yaw, roll) from aligned landmarks.

        Args:
            aligned_landmarks: 68-point landmarks in aligned image space.
            size: Size of the aligned image. Default: 256.

        Returns:
            Tuple of (pitch, yaw, roll) in radians [-pi/2, pi/2].
        """
        shape = (size, size)
        focal_length = shape[1]
        camera_center = (shape[1] / 2, shape[0] / 2)

        camera_matrix = np.array(
            [
                [focal_length, 0, camera_center[0]],
                [0, focal_length, camera_center[1]],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )

        # Use points 0-26 (jaw + eyebrows) and 30-35 (nose)
        object_points = np.concatenate(
            [
                LANDMARKS_68_3D[:27],
                LANDMARKS_68_3D[30:36],
            ],
            axis=0,
        )

        image_points = np.concatenate(
            [
                aligned_landmarks[:27],
                aligned_landmarks[30:36],
            ],
            axis=0,
        ).astype(np.float32)

        _, rotation_vector, _ = cv2.solvePnP(
            object_points,
            image_points,
            camera_matrix,
            np.zeros((4, 1)),
        )

        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        pitch, yaw, roll = self._rotation_matrix_to_euler(rotation_matrix)

        half_pi = math.pi / 2.0
        pitch = np.clip(pitch, -half_pi, half_pi)
        yaw = np.clip(yaw, -half_pi, half_pi)
        roll = np.clip(roll, -half_pi, half_pi)

        return -pitch, yaw, roll

    def _estimate_averaged_yaw(self, landmarks: np.ndarray) -> float:
        """Estimate yaw angle from landmarks using averaging method."""
        if not isinstance(landmarks, np.ndarray):
            landmarks = np.array(landmarks)

        l = (
            (landmarks[27][0] - landmarks[0][0])
            + (landmarks[28][0] - landmarks[1][0])
            + (landmarks[29][0] - landmarks[2][0])
        ) / 3.0

        r = (
            (landmarks[16][0] - landmarks[27][0])
            + (landmarks[15][0] - landmarks[28][0])
            + (landmarks[14][0] - landmarks[29][0])
        ) / 3.0

        return float(r - l)

    def _polygon_area(self, x: np.ndarray, y: np.ndarray) -> float:
        """Calculate polygon area using shoelace formula."""
        return float(0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))

    def _rotation_matrix_to_euler(self, R: np.ndarray) -> tuple[float, float, float]:
        """Convert rotation matrix to Euler angles (pitch, yaw, roll)."""
        sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        singular = sy < 1e-6

        if not singular:
            x = math.atan2(R[2, 1], R[2, 2])
            y = math.atan2(-R[2, 0], sy)
            z = math.atan2(R[1, 0], R[0, 0])
        else:
            x = math.atan2(-R[1, 2], R[1, 1])
            y = math.atan2(-R[2, 0], sy)
            z = 0

        return x, y, z

    def convert_106_to_68(self, landmarks_106: np.ndarray) -> np.ndarray:
        """
        Convert 106-point landmarks (InsightFace) to 68-point format (DFL).

        Args:
            landmarks_106: 106-point landmarks array.

        Returns:
            68-point landmarks array.
        """
        if not isinstance(landmarks_106, np.ndarray):
            landmarks_106 = np.array(landmarks_106)

        if landmarks_106.ndim != 2 or landmarks_106.shape[1] != 2:
            raise ValueError(
                f"landmarks_106 must have shape (N, 2), got {landmarks_106.shape}"
            )

        num_points = landmarks_106.shape[0]
        if num_points == 5:
            raise ValueError(
                "Received 5-point landmarks instead of 106-point. "
                "Cannot convert to 68-point format."
            )
        elif num_points != 106:
            logger.warning(
                f"Expected 106-point landmarks, got {num_points}. "
                f"Conversion may produce errors."
            )

        # NaN and Inf validation
        if np.any(np.isnan(landmarks_106)):
            raise ValueError("Landmarks contain NaN values")
        if np.any(np.isinf(landmarks_106)):
            raise ValueError("Landmarks contain infinite values")

        # Bounds check (warning for extreme values)
        if np.any(landmarks_106 < -10000) or np.any(landmarks_106 > 10000):
            logger.warning(
                "Landmarks contain extreme values, results may be unreliable"
            )

        # Mapping from 106 to 68 points
        # This is an approximate mapping
        idx_map = [
            # Jaw (17 points: 0-16)
            0,
            2,
            4,
            6,
            8,
            10,
            12,
            14,
            16,
            18,
            20,
            22,
            24,
            26,
            28,
            30,
            32,
            # Right eyebrow (5 points: 17-21)
            33,
            34,
            35,
            36,
            37,
            # Left eyebrow (5 points: 22-26)
            42,
            43,
            44,
            45,
            46,
            # Nose (9 points: 27-35)
            51,
            52,
            53,
            54,
            55,
            56,
            57,
            58,
            59,
            # Right eye (6 points: 36-41)
            60,
            61,
            63,
            64,
            65,
            67,
            # Left eye (6 points: 42-47)
            68,
            69,
            71,
            72,
            73,
            75,
            # Mouth outer (12 points: 48-59)
            76,
            77,
            78,
            79,
            80,
            81,
            82,
            83,
            84,
            85,
            86,
            87,
            # Mouth inner (8 points: 60-67)
            88,
            89,
            90,
            91,
            92,
            93,
            94,
            95,
        ]

        return landmarks_106[idx_map]

    def get_face_type_transform_mat(
        self,
        landmarks: np.ndarray,
        output_size: int,
        source_face_type: FaceType,
        target_face_type: FaceType,
    ) -> np.ndarray:
        """
        Compute transformation matrix between face types.

        Used when the model was trained on a different face type than
        the input image. Transforms from source face type space to
        target face type space.

        Args:
            landmarks: 68-point facial landmarks.
            output_size: Output image size (square).
            source_face_type: Face type of the input image.
            target_face_type: Target face type to transform to.

        Returns:
            Affine transform matrix of shape (2, 3).
        """
        if source_face_type == target_face_type:
            return np.eye(2, 3, dtype=np.float32)

        # Get transform matrices for both face types
        source_mat = self.get_transform_mat(landmarks, output_size, source_face_type)
        target_mat = self.get_transform_mat(landmarks, output_size, target_face_type)

        # Transform corners from target space to global, then to source space
        corners = np.array([(0, 0), (output_size, 0), (0, output_size)], dtype=np.float32)

        # Target corners in global space
        global_pts = transform_points(corners, target_mat, invert=True)

        # Global points in source space
        source_pts = transform_points(global_pts, source_mat)

        # Affine transform from source_pts to corners
        return cast(
            np.ndarray,
            cv2.getAffineTransform(source_pts.astype(np.float32), corners.astype(np.float32)),
        )

    def warp_between_face_types(
        self,
        image: np.ndarray,
        landmarks: np.ndarray,
        source_face_type: FaceType,
        target_face_type: FaceType,
    ) -> np.ndarray:
        """
        Warp image from one face type space to another.

        Args:
            image: Input image aligned to source_face_type.
            landmarks: 68-point landmarks in source image space.
            source_face_type: Current face type of the image.
            target_face_type: Desired face type after warping.

        Returns:
            Warped image in target face type space.
        """
        if source_face_type == target_face_type:
            return cast(np.ndarray, image.copy())

        h, w = image.shape[:2]
        mat = self.get_face_type_transform_mat(
            landmarks, w, source_face_type, target_face_type
        )

        return cast(
            np.ndarray,
            cv2.warpAffine(
                image,
                mat,
                (w, w),
                flags=cv2.INTER_LANCZOS4,
                borderMode=cv2.BORDER_REPLICATE,
            ),
        )
