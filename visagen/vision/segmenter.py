"""
Face Segmentation using SegFormer.

Provides semantic face parsing using SegFormer model
fine-tuned on CelebAMask-HQ dataset.
"""

import numpy as np
import cv2
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation


# CelebAMask-HQ label mapping (19 classes)
CELEBAMASK_LABELS = {
    0: "background",
    1: "skin",
    2: "nose",
    3: "eye_glasses",
    4: "left_eye",
    5: "right_eye",
    6: "left_brow",
    7: "right_brow",
    8: "left_ear",
    9: "right_ear",
    10: "mouth",
    11: "upper_lip",
    12: "lower_lip",
    13: "hair",
    14: "hat",
    15: "earring",
    16: "necklace",
    17: "neck",
    18: "cloth",
}

# Inverse mapping for lookup
LABEL_TO_ID = {v: k for k, v in CELEBAMASK_LABELS.items()}

# Face mask components (excluding background, hair, hat, cloth, accessories)
FACE_COMPONENTS = {
    "skin", "nose", "left_eye", "right_eye",
    "left_brow", "right_brow", "left_ear", "right_ear",
    "mouth", "upper_lip", "lower_lip",
}


@dataclass
class SegmentationResult:
    """
    Face segmentation result.

    Attributes:
        mask: Binary face mask (H, W), values 0 or 255.
        parsing: Full parsing map with class indices (H, W).
        confidence: Mean confidence score for face region.
    """

    mask: np.ndarray
    parsing: np.ndarray
    confidence: float


class FaceSegmenter:
    """
    SegFormer-based face segmentation.

    Uses SegFormer model fine-tuned on CelebAMask-HQ for
    semantic face parsing with 19 classes.

    Args:
        model_name: HuggingFace model identifier.
            Default: "jonathandinu/face-parsing".
        device: Device for inference. Default: auto-detect.
        use_half: Use FP16 for faster inference. Default: True on CUDA.

    Example:
        >>> segmenter = FaceSegmenter()
        >>> result = segmenter.segment(aligned_face)
        >>> mask = result.mask  # Binary face mask
    """

    def __init__(
        self,
        model_name: str = "jonathandinu/face-parsing",
        device: Optional[str] = None,
        use_half: bool = True,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.use_half = use_half and device == "cuda"

        # Load model and processor
        self.processor = SegformerImageProcessor.from_pretrained(model_name)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()

        if self.use_half:
            self.model.half()

    def segment(
        self,
        face_image: np.ndarray,
        return_soft_mask: bool = False,
    ) -> SegmentationResult:
        """
        Segment face from aligned face image.

        Args:
            face_image: Aligned face image (BGR, uint8).
            return_soft_mask: Return soft probabilities instead of binary.
                Default: False.

        Returns:
            SegmentationResult with binary mask and full parsing.
        """
        h, w = face_image.shape[:2]

        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

        # Preprocess
        inputs = self.processor(images=rgb_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        if self.use_half:
            inputs["pixel_values"] = inputs["pixel_values"].half()

        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        # Upsample to original size
        logits = F.interpolate(
            logits,
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )

        # Get predictions
        probs = F.softmax(logits, dim=1)
        parsing = logits.argmax(dim=1).squeeze().cpu().numpy()

        # Create face mask from face components
        face_mask = np.zeros((h, w), dtype=np.float32)
        for component in FACE_COMPONENTS:
            if component in LABEL_TO_ID:
                component_id = LABEL_TO_ID[component]
                face_mask += probs[0, component_id].cpu().numpy()

        # Calculate confidence as mean probability in face region
        confidence = float(face_mask[face_mask > 0.5].mean()) if (face_mask > 0.5).any() else 0.0

        # Convert to binary mask
        if not return_soft_mask:
            face_mask = (face_mask > 0.5).astype(np.uint8) * 255

        return SegmentationResult(
            mask=face_mask,
            parsing=parsing.astype(np.uint8),
            confidence=confidence,
        )

    def get_parsing(
        self,
        face_image: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Get detailed parsing masks for each face component.

        Args:
            face_image: Aligned face image (BGR, uint8).

        Returns:
            Dictionary mapping component names to binary masks.
        """
        h, w = face_image.shape[:2]

        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

        # Preprocess
        inputs = self.processor(images=rgb_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        if self.use_half:
            inputs["pixel_values"] = inputs["pixel_values"].half()

        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        # Upsample to original size
        logits = F.interpolate(
            logits,
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )

        # Get per-class masks
        parsing = logits.argmax(dim=1).squeeze().cpu().numpy()

        result = {}
        for label_id, label_name in CELEBAMASK_LABELS.items():
            mask = (parsing == label_id).astype(np.uint8) * 255
            if mask.any():
                result[label_name] = mask

        return result

    def get_hair_mask(self, face_image: np.ndarray) -> np.ndarray:
        """
        Get hair-only mask from face image.

        Args:
            face_image: Aligned face image (BGR, uint8).

        Returns:
            Binary hair mask (H, W), values 0 or 255.
        """
        parsing = self.get_parsing(face_image)
        hair_mask = parsing.get("hair", np.zeros(face_image.shape[:2], dtype=np.uint8))
        return hair_mask

    def get_face_and_hair_mask(self, face_image: np.ndarray) -> np.ndarray:
        """
        Get combined face and hair mask.

        Args:
            face_image: Aligned face image (BGR, uint8).

        Returns:
            Binary mask (H, W), values 0 or 255.
        """
        result = self.segment(face_image)
        parsing = self.get_parsing(face_image)

        combined = result.mask.copy()
        if "hair" in parsing:
            combined = np.maximum(combined, parsing["hair"])

        return combined
