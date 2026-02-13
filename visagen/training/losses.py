"""
Loss Functions for Visagen Training.

Implements modern perceptual loss functions:
- DSSIM: Differentiable Structural Similarity
- LPIPS: Learned Perceptual Image Patch Similarity
- IDLoss: ArcFace-based Identity Preservation
- EyesMouthLoss: Priority loss for facial details
- StyleLoss: Gram matrix based style transfer
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# =============================================================================
# Gaussian Blur Utilities (Kornia Fallback)
# =============================================================================


def _create_gaussian_kernel_1d(
    size: int,
    sigma: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Create 1D Gaussian kernel."""
    coords = torch.arange(size, dtype=torch.float32, device=device)
    coords -= (size - 1) / 2.0
    g = torch.exp(-0.5 * (coords / sigma) ** 2)
    return (g / g.sum()).to(dtype=dtype)


def _gaussian_blur2d_native(
    x: torch.Tensor,
    kernel_size: int | tuple[int, int],
    sigma: float | tuple[float, float],
) -> torch.Tensor:
    """PyTorch native Gaussian blur (depthwise conv2d)."""
    # Parse kernel size
    if isinstance(kernel_size, tuple):
        ksize_h, ksize_w = kernel_size
    else:
        ksize_h = ksize_w = kernel_size

    # Parse sigma
    if isinstance(sigma, tuple):
        sigma_h, sigma_w = sigma
    else:
        sigma_h = sigma_w = sigma

    # Validate kernel sizes (must be odd and >= 3)
    ksize_h = max(3, ksize_h if ksize_h % 2 == 1 else ksize_h + 1)
    ksize_w = max(3, ksize_w if ksize_w % 2 == 1 else ksize_w + 1)

    # Create 1D kernels
    kernel_h = _create_gaussian_kernel_1d(ksize_h, sigma_h, x.device, x.dtype)
    kernel_w = _create_gaussian_kernel_1d(ksize_w, sigma_w, x.device, x.dtype)

    # 2D kernel via outer product
    kernel_2d = kernel_h.unsqueeze(1) @ kernel_w.unsqueeze(0)
    kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

    # Expand for all channels (depthwise)
    c = x.shape[1]
    kernel_2d = kernel_2d.repeat(c, 1, 1, 1)  # (C, 1, H, W)

    # Padding
    padding_h = ksize_h // 2
    padding_w = ksize_w // 2

    return F.conv2d(x, kernel_2d, padding=(padding_h, padding_w), groups=c)


def safe_gaussian_blur2d(
    x: torch.Tensor,
    kernel_size: int | tuple[int, int],
    sigma: float | tuple[float, float],
) -> torch.Tensor:
    """
    Safe Gaussian blur with automatic kornia fallback.

    Tries kornia.filters.gaussian_blur2d first, falls back to
    native PyTorch implementation if kornia is not available.

    Args:
        x: Input tensor (B, C, H, W)
        kernel_size: Kernel size (odd int or tuple)
        sigma: Gaussian standard deviation

    Returns:
        Blurred tensor (B, C, H, W)
    """
    try:
        import kornia.filters

        # Kornia expects kernel_size as tuple and sigma as (B, 2) tensor
        if isinstance(kernel_size, int):
            ks = (kernel_size, kernel_size)
        else:
            ks = kernel_size

        # Build sigma tensor with shape (B, 2) for kornia
        b = x.shape[0]
        if isinstance(sigma, (int, float)):
            sig = torch.tensor([[sigma, sigma]], device=x.device, dtype=x.dtype)
        elif isinstance(sigma, tuple):
            sig = torch.tensor([list(sigma)], device=x.device, dtype=x.dtype)
        else:
            sig = sigma

        # Expand to batch size if needed
        if sig.dim() == 2 and sig.shape[0] == 1 and b > 1:
            sig = sig.expand(b, -1)

        return kornia.filters.gaussian_blur2d(x, ks, sig)
    except ImportError:
        return _gaussian_blur2d_native(x, kernel_size, sigma)


# =============================================================================
# Loss Functions
# =============================================================================


class DSSIMLoss(nn.Module):
    """
    Differentiable Structural Similarity Loss.

    Computes DSSIM = (1 - SSIM) / 2, where SSIM measures
    luminance, contrast, and structure similarity.

    Args:
        filter_size: Gaussian filter size. Default: 11.
        filter_sigma: Gaussian filter sigma. Default: 1.5.
        k1: Luminance stability constant. Default: 0.01.
        k2: Contrast stability constant. Default: 0.03.
        max_val: Maximum pixel value. Default: 1.0.

    Example:
        >>> loss_fn = DSSIMLoss()
        >>> pred = torch.randn(2, 3, 256, 256)
        >>> target = torch.randn(2, 3, 256, 256)
        >>> loss = loss_fn(pred, target)
    """

    def __init__(
        self,
        filter_size: int = 11,
        filter_sigma: float = 1.5,
        k1: float = 0.01,
        k2: float = 0.03,
        max_val: float = 1.0,
    ) -> None:
        super().__init__()
        self.filter_size = filter_size
        self.filter_sigma = filter_sigma
        self.k1 = k1
        self.k2 = k2
        self.max_val = max_val

        # Create Gaussian kernel
        kernel = self._create_gaussian_kernel(filter_size, filter_sigma)
        self.register_buffer("kernel", kernel)

    def _create_gaussian_kernel(self, size: int, sigma: float) -> torch.Tensor:
        """Create 2D Gaussian kernel using softmax normalization for compatibility."""
        coords = torch.arange(size, dtype=torch.float32)
        coords -= (size - 1) / 2.0

        # Use softmax normalization for kernel (matches reference implementation)
        g = coords**2
        g = g * (-0.5 / (sigma**2))

        # 2D kernel via outer sum
        g_2d = g.unsqueeze(0) + g.unsqueeze(1)  # (size, size)
        g_2d = g_2d.view(-1)  # Flatten for softmax
        g_2d = torch.softmax(g_2d, dim=0)  # Softmax normalization
        g_2d = g_2d.view(size, size)

        return g_2d.unsqueeze(0).unsqueeze(0)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute DSSIM loss.

        Args:
            pred: Predicted image (B, C, H, W).
            target: Target image (B, C, H, W).

        Returns:
            DSSIM loss value (scalar or per-batch).
        """
        c = pred.shape[1]

        # Constants with numerical stability
        EPS = 1e-8
        c1 = (self.k1 * self.max_val) ** 2
        c2 = (self.k2 * self.max_val) ** 2

        # Expand kernel for all channels (ensure dtype matches for AMP)
        kernel = self.kernel.to(dtype=pred.dtype).repeat(c, 1, 1, 1)

        # Compute means using depthwise convolution
        mu1 = F.conv2d(pred, kernel, padding=0, groups=c)
        mu2 = F.conv2d(target, kernel, padding=0, groups=c)

        mu1_sq = mu1**2
        mu2_sq = mu2**2
        mu1_mu2 = mu1 * mu2

        # Compute variances with clamping for numerical stability
        sigma1_sq = torch.clamp(
            F.conv2d(pred**2, kernel, padding=0, groups=c) - mu1_sq, min=0.0
        )
        sigma2_sq = torch.clamp(
            F.conv2d(target**2, kernel, padding=0, groups=c) - mu2_sq, min=0.0
        )
        sigma12 = F.conv2d(pred * target, kernel, padding=0, groups=c) - mu1_mu2

        # SSIM formula with epsilon for stability
        luminance = (2 * mu1_mu2 + c1) / (mu1_sq + mu2_sq + c1 + EPS)
        contrast_structure = (2 * sigma12 + c2) / (sigma1_sq + sigma2_sq + c2 + EPS)

        # Clamp SSIM to valid range
        ssim_map = torch.clamp(luminance * contrast_structure, -1.0, 1.0)

        # DSSIM = (1 - SSIM) / 2
        dssim = (1.0 - ssim_map.mean(dim=[2, 3])) / 2.0

        return dssim.mean()


class MultiScaleDSSIMLoss(nn.Module):
    """
    Multi-scale DSSIM loss.

    Computes DSSIM at multiple filter sizes for better
    structural similarity across scales.

    Args:
        resolution: Image resolution for dynamic filter size calculation. Default: 256.
        filter_sizes: List of filter sizes. Default: None (calculated from resolution).
        weights: Weights for each scale. Default: equal weights.
    """

    def __init__(
        self,
        resolution: int = 256,
        filter_sizes: tuple[int, ...] | None = None,
        weights: tuple[float, ...] | None = None,
    ) -> None:
        super().__init__()

        if filter_sizes is None:
            filter_sizes = (
                max(3, int(resolution / 11.6)) | 1,
                max(3, int(resolution / 23.2)) | 1,
            )

        if weights is None:
            weights = tuple(1.0 / len(filter_sizes) for _ in filter_sizes)

        self.weights = weights
        self.losses = nn.ModuleList([DSSIMLoss(filter_size=fs) for fs in filter_sizes])

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute multi-scale DSSIM loss."""
        total = 0.0
        for weight, loss_fn in zip(self.weights, self.losses, strict=False):
            total = total + weight * loss_fn(pred, target)
        return total


class LPIPSLoss(nn.Module):
    """
    Learned Perceptual Image Patch Similarity Loss.

    Uses pre-trained VGG or AlexNet features to compute
    perceptual similarity.

    Args:
        net: Network to use ('vgg' or 'alex'). Default: 'vgg'.
        use_gpu: Use GPU if available. Default: True.

    Note:
        Requires lpips package: pip install lpips
    """

    def __init__(
        self,
        net: str = "vgg",
        use_gpu: bool = True,
    ) -> None:
        super().__init__()
        self.net = net
        self._lpips = None
        self._use_gpu = use_gpu
        self._cached_device = None

    def _ensure_lpips(self) -> None:
        """Lazy load LPIPS model."""
        if self._lpips is None:
            try:
                import lpips

                try:
                    self._lpips = lpips.LPIPS(net=self.net)
                except Exception:
                    # Offline/sandbox environments may block pretrained downloads.
                    # Fall back to random perceptual network so loss remains usable.
                    self._lpips = lpips.LPIPS(
                        net=self.net,
                        pretrained=False,
                        pnet_rand=True,
                        verbose=False,
                    )
                if self._use_gpu and torch.cuda.is_available():
                    self._lpips = self._lpips.cuda()
                self._lpips.eval()
                for param in self._lpips.parameters():
                    param.requires_grad = False
            except ImportError:
                raise ImportError(
                    "lpips package required. Install with: pip install lpips"
                )

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute LPIPS loss.

        Args:
            pred: Predicted image (B, C, H, W) in [-1, 1].
            target: Target image (B, C, H, W) in [-1, 1].

        Returns:
            LPIPS loss value.
        """
        self._ensure_lpips()

        # Only move model if device changed (cache device for efficiency)
        if self._cached_device != pred.device:
            self._lpips = self._lpips.to(pred.device)
            self._cached_device = pred.device

        return self._lpips(pred, target).mean()


class IDLoss(nn.Module):
    """
    Identity Preservation Loss using ArcFace embeddings.

    Computes cosine similarity between face embeddings to
    ensure identity is preserved during face swapping.

    Args:
        model_path: Path to ArcFace model. Default: None (uses insightface).

    Note:
        Requires insightface package for default model.
    """

    def __init__(self, model_path: str | None = None) -> None:
        super().__init__()
        self.model_path = model_path
        self._model = None

    def _ensure_model(self, device: torch.device) -> None:
        """Lazy load ArcFace model."""
        if self._model is None:
            try:
                from insightface.app import FaceAnalysis

                app = FaceAnalysis(
                    name="buffalo_l",
                    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                )
                app.prepare(ctx_id=0 if device.type == "cuda" else -1)
                self._model = app
            except ImportError:
                raise ImportError(
                    "insightface package required. Install with: "
                    "pip install insightface onnxruntime-gpu"
                )

    def _get_embedding(self, img: torch.Tensor) -> torch.Tensor:
        """Extract face embedding from image tensor."""
        import numpy as np

        # Convert to numpy (B, C, H, W) -> (B, H, W, C)
        img_np = img.detach().cpu().permute(0, 2, 3, 1).numpy()

        # Denormalize from [-1, 1] to [0, 255]
        img_np = ((img_np + 1) * 127.5).astype(np.uint8)

        embeddings = []
        for i in range(img_np.shape[0]):
            # BGR format expected
            img_bgr = img_np[i, :, :, ::-1].copy()
            faces = self._model.get(img_bgr)
            if len(faces) > 0:
                embeddings.append(torch.from_numpy(faces[0].embedding))
            else:
                logger.warning(f"No face detected in IDLoss at batch {i}")
                # Small random embedding to maintain gradient flow
                embeddings.append(torch.randn(512) * 0.01)

        return torch.stack(embeddings).to(img.device)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute identity loss.

        Args:
            pred: Predicted face (B, C, H, W) in [-1, 1].
            target: Target face (B, C, H, W) in [-1, 1].

        Returns:
            Identity loss (1 - cosine_similarity).
        """
        self._ensure_model(pred.device)

        pred_emb = self._get_embedding(pred)
        target_emb = self._get_embedding(target)

        # Check for degenerate embeddings
        pred_norm = torch.norm(pred_emb, dim=1)
        target_norm = torch.norm(target_emb, dim=1)
        valid_mask = (pred_norm > 1e-4) & (target_norm > 1e-4)

        if not valid_mask.any():
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        # Cosine similarity only for valid embeddings
        cos_sim = F.cosine_similarity(
            pred_emb[valid_mask], target_emb[valid_mask], dim=1
        )

        # Loss = 1 - similarity (so 0 = identical, 2 = opposite)
        return (1.0 - cos_sim).mean()


class EyesMouthLoss(nn.Module):
    """
    Priority loss for eyes and mouth regions.

    Applies higher weight to eye and mouth regions based on
    landmark positions to improve detail preservation.

    Args:
        weight_multiplier: Multiplier for eye/mouth regions. Default: 30.0.
        base_loss: Base loss function. Default: L1.
    """

    def __init__(
        self,
        weight_multiplier: float = 300.0,
        use_l1: bool = True,
    ) -> None:
        super().__init__()
        self.weight_multiplier = weight_multiplier
        self.use_l1 = use_l1

    def _create_region_mask(
        self,
        landmarks: torch.Tensor,
        image_size: tuple[int, int],
        region_indices: tuple[int, ...],
        radius: int = 15,
    ) -> torch.Tensor:
        """Create soft mask for facial region."""
        batch_size = landmarks.shape[0]
        h, w = image_size

        mask = torch.zeros(batch_size, 1, h, w, device=landmarks.device)

        # Get region center from landmarks
        for b in range(batch_size):
            for idx in region_indices:
                if idx < landmarks.shape[1]:
                    cx, cy = landmarks[b, idx].long()
                    cx = torch.clamp(cx, 0, w - 1)
                    cy = torch.clamp(cy, 0, h - 1)

                    # Create circular mask
                    y, x = torch.meshgrid(
                        torch.arange(h, device=landmarks.device),
                        torch.arange(w, device=landmarks.device),
                        indexing="ij",
                    )
                    dist = ((x - cx) ** 2 + (y - cy) ** 2).float().sqrt()
                    region = torch.clamp(1.0 - dist / radius, 0, 1)
                    mask[b, 0] = torch.maximum(mask[b, 0], region)

        return mask

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        landmarks: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute eyes/mouth priority loss.

        Args:
            pred: Predicted face (B, C, H, W).
            target: Target face (B, C, H, W).
            landmarks: 68-point landmarks (B, 68, 2). Optional.

        Returns:
            Weighted loss value.
        """
        # If no landmarks, use uniform weighting
        if landmarks is None:
            if self.use_l1:
                return F.l1_loss(pred, target)
            return F.mse_loss(pred, target)

        h, w = pred.shape[2:]

        # Eye indices (36-47 in 68-point format)
        eye_indices = tuple(range(36, 48))
        # Mouth indices (48-67 in 68-point format)
        mouth_indices = tuple(range(48, 68))

        eye_mask = self._create_region_mask(landmarks, (h, w), eye_indices)
        mouth_mask = self._create_region_mask(landmarks, (h, w), mouth_indices)

        # Combined priority mask
        priority_mask = torch.clamp(eye_mask + mouth_mask, 0, 1)

        # Compute base loss
        if self.use_l1:
            pixel_loss = torch.abs(pred - target)
        else:
            pixel_loss = (pred - target) ** 2

        # Apply priority weighting
        weighted_loss = pixel_loss * (
            1.0 + priority_mask * (self.weight_multiplier - 1.0)
        )

        return weighted_loss.mean()


class GazeLoss(nn.Module):
    """
    Gaze direction consistency loss for eyes.

    Ensures eye regions maintain consistent appearance and gaze direction
    between source and target faces. Uses landmark-based eye extraction
    and computes similarity in the eye regions.

    Args:
        eye_size: Size to resize eye patches for comparison. Default: 32.
        use_perceptual: Use perceptual loss for eyes. Default: False.

    Example:
        >>> loss_fn = GazeLoss()
        >>> pred = torch.randn(2, 3, 256, 256)
        >>> target = torch.randn(2, 3, 256, 256)
        >>> landmarks = torch.randn(2, 68, 2) * 256  # 68-point landmarks
        >>> loss = loss_fn(pred, target, landmarks)
    """

    # 68-point landmark indices for eyes
    LEFT_EYE_INDICES = list(range(36, 42))  # 36-41
    RIGHT_EYE_INDICES = list(range(42, 48))  # 42-47

    def __init__(
        self,
        eye_size: int = 32,
        use_perceptual: bool = False,
    ) -> None:
        super().__init__()
        self.eye_size = eye_size
        self.use_perceptual = use_perceptual

    def _get_eye_bbox(
        self,
        landmarks: torch.Tensor,
        indices: list[int],
        padding: float = 0.3,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get bounding box for eye region from landmarks.

        Args:
            landmarks: (B, 68, 2) landmark coordinates.
            indices: List of landmark indices for the eye.
            padding: Padding ratio around eye. Default: 0.3.

        Returns:
            Tuple of (x1, y1, x2, y2) tensors of shape (B,).
        """
        # Extract eye landmarks
        eye_pts = landmarks[:, indices, :]  # (B, 6, 2)

        # Get bounding box
        x_min = eye_pts[:, :, 0].min(dim=1).values
        x_max = eye_pts[:, :, 0].max(dim=1).values
        y_min = eye_pts[:, :, 1].min(dim=1).values
        y_max = eye_pts[:, :, 1].max(dim=1).values

        # Add padding
        width = x_max - x_min
        height = y_max - y_min

        x1 = x_min - width * padding
        x2 = x_max + width * padding
        y1 = y_min - height * padding
        y2 = y_max + height * padding

        return x1, y1, x2, y2

    def _extract_eye_patch(
        self,
        images: torch.Tensor,
        x1: torch.Tensor,
        y1: torch.Tensor,
        x2: torch.Tensor,
        y2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract and resize eye patches from images using differentiable grid sampling.

        Args:
            images: (B, C, H, W) input images.
            x1, y1, x2, y2: Bounding box coordinates (B,).

        Returns:
            (B, C, eye_size, eye_size) eye patches.
        """
        batch_size, channels, height, width = images.shape
        device = images.device

        # Create sampling grid for each batch item
        patches = []
        for b in range(batch_size):
            # Normalize coordinates to [-1, 1] range for grid_sample
            # grid_sample expects (x, y) in [-1, 1] where -1 is left/top, 1 is right/bottom
            bx1 = x1[b].clamp(0, width - 1)
            by1 = y1[b].clamp(0, height - 1)
            bx2 = x2[b].clamp(0, width - 1)
            by2 = y2[b].clamp(0, height - 1)

            # Ensure minimum box size of 2 pixels
            MIN_BOX_SIZE = 2
            box_width = bx2 - bx1
            box_height = by2 - by1

            if box_width < MIN_BOX_SIZE or box_height < MIN_BOX_SIZE:
                # Expand from center to minimum size
                center_x = (bx1 + bx2) / 2
                center_y = (by1 + by2) / 2
                half_size = MIN_BOX_SIZE / 2
                bx1 = torch.clamp(center_x - half_size, min=0)
                bx2 = bx1 + MIN_BOX_SIZE
                by1 = torch.clamp(center_y - half_size, min=0)
                by2 = by1 + MIN_BOX_SIZE
                # Ensure still within bounds
                bx2 = torch.clamp(bx2, max=width - 1)
                by2 = torch.clamp(by2, max=height - 1)

            # Ensure valid box (at least 1 pixel)
            if (bx2 - bx1) < 1 or (by2 - by1) < 1:
                # Invalid bounding box - use center region of image to preserve gradients
                # Create a default grid sampling from center to maintain differentiability
                center_grid = torch.zeros(
                    1, self.eye_size, self.eye_size, 2, device=device
                )
                patch = F.grid_sample(
                    images[b : b + 1],
                    center_grid,
                    mode="bilinear",
                    padding_mode="border",
                    align_corners=True,
                ).squeeze(0)
                patches.append(patch)
                continue

            # Create normalized grid
            # grid values in [-1, 1], where -1 maps to 0 and 1 maps to width-1 or height-1
            x_norm_start = (bx1 / (width - 1)) * 2 - 1
            x_norm_end = (bx2 / (width - 1)) * 2 - 1
            y_norm_start = (by1 / (height - 1)) * 2 - 1
            y_norm_end = (by2 / (height - 1)) * 2 - 1

            # Create 2D grid
            y_coords = torch.linspace(
                y_norm_start.item(),
                y_norm_end.item(),
                self.eye_size,
                device=device,
            )
            x_coords = torch.linspace(
                x_norm_start.item(),
                x_norm_end.item(),
                self.eye_size,
                device=device,
            )

            # Create meshgrid (H, W, 2) -> (1, H, W, 2)
            grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
            grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)

            # Sample using grid_sample (preserves gradients)
            patch = F.grid_sample(
                images[b : b + 1],
                grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            ).squeeze(0)

            patches.append(patch)

        return torch.stack(patches, dim=0)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        landmarks: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute gaze/eye consistency loss.

        Args:
            pred: Predicted face (B, C, H, W).
            target: Target face (B, C, H, W).
            landmarks: 68-point landmarks (B, 68, 2). Required.

        Returns:
            Gaze loss value (scalar).
        """
        if landmarks is None:
            # Fall back to simple L1 if no landmarks
            return F.l1_loss(pred, target)

        # Get eye bounding boxes
        left_x1, left_y1, left_x2, left_y2 = self._get_eye_bbox(
            landmarks, self.LEFT_EYE_INDICES
        )
        right_x1, right_y1, right_x2, right_y2 = self._get_eye_bbox(
            landmarks, self.RIGHT_EYE_INDICES
        )

        # Extract eye patches from pred and target
        pred_left = self._extract_eye_patch(pred, left_x1, left_y1, left_x2, left_y2)
        pred_right = self._extract_eye_patch(
            pred, right_x1, right_y1, right_x2, right_y2
        )
        target_left = self._extract_eye_patch(
            target, left_x1, left_y1, left_x2, left_y2
        )
        target_right = self._extract_eye_patch(
            target, right_x1, right_y1, right_x2, right_y2
        )

        # Compute loss for both eyes
        loss_left = F.l1_loss(pred_left, target_left)
        loss_right = F.l1_loss(pred_right, target_right)

        return (loss_left + loss_right) / 2.0


class MomentStyleLoss(nn.Module):
    """
    Moment-matching style loss.

    Computes style difference using first-order (mean) and second-order (std)
    statistics instead of Gram matrices.

    Args:
        gaussian_blur_radius: Blur radius before computing moments. Default: 0.
        loss_weight: Weight multiplier for the loss. Default: 1.0.
    """

    def __init__(
        self,
        gaussian_blur_radius: float = 0.0,
        loss_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.blur_radius = gaussian_blur_radius
        self.loss_weight = loss_weight

    def forward(
        self,
        content: torch.Tensor,
        style: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute moment-matching style loss.

        Args:
            content: Content features (B, C, H, W).
            style: Style features (B, C, H, W).

        Returns:
            Style loss value.
        """
        if content.shape[1] != style.shape[1]:
            raise ValueError("content and style must have same number of channels")

        # Optional Gaussian blur
        if self.blur_radius > 0:
            kernel_size = max(3, int(2 * 2 * self.blur_radius))
            if kernel_size % 2 == 0:
                kernel_size += 1
            sigma = self.blur_radius
            content = safe_gaussian_blur2d(
                content, (kernel_size, kernel_size), (sigma, sigma)
            )
            style = safe_gaussian_blur2d(
                style, (kernel_size, kernel_size), (sigma, sigma)
            )

        # Compute spatial statistics
        c_mean = content.mean(dim=[2, 3], keepdim=True)
        s_mean = style.mean(dim=[2, 3], keepdim=True)

        c_var = content.var(dim=[2, 3], keepdim=True)
        s_var = style.var(dim=[2, 3], keepdim=True)

        c_std = torch.sqrt(c_var + 1e-5)
        s_std = torch.sqrt(s_var + 1e-5)

        # Mean and std losses
        mean_loss = ((c_mean - s_mean) ** 2).sum(dim=[1, 2, 3])
        std_loss = ((c_std - s_std) ** 2).sum(dim=[1, 2, 3])

        # Normalize by channel count
        nc = content.shape[1]
        loss = (mean_loss + std_loss) * (self.loss_weight / nc)

        return loss.mean()


class StyleLoss(nn.Module):
    """
    Gram matrix based style transfer loss.

    Computes style difference using Gram matrices of feature maps.

    Args:
        gaussian_blur_radius: Blur radius before Gram computation. Default: 0.
    """

    def __init__(self, gaussian_blur_radius: float = 0.0) -> None:
        super().__init__()
        self.blur_radius = gaussian_blur_radius

    def _gram_matrix(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Gram matrix with optional blur."""
        # Apply Gaussian blur if blur_radius > 0
        if self.blur_radius > 0:
            kernel_size = int(self.blur_radius) * 2 + 1
            sigma = self.blur_radius / 3.0
            x = safe_gaussian_blur2d(x, (kernel_size, kernel_size), (sigma, sigma))

        b, c, h, w = x.shape
        # Ensure contiguous memory for efficient matrix multiplication
        features = x.reshape(b, c, h * w).contiguous()
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram / (c * h * w)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute style loss.

        Args:
            pred: Predicted image (B, C, H, W).
            target: Target image (B, C, H, W).
            mask: Optional mask (B, 1, H, W).

        Returns:
            Style loss value.
        """
        if mask is not None:
            pred = pred * mask
            target = target * mask

        gram_pred = self._gram_matrix(pred)
        gram_target = self._gram_matrix(target)

        return F.mse_loss(gram_pred, gram_target)


class CombinedLoss(nn.Module):
    """
    Combined loss function for face swapping training.

    Combines multiple losses with configurable weights.

    Args:
        dssim_weight: Weight for DSSIM loss. Default: 10.0.
        l1_weight: Weight for L1 loss. Default: 10.0.
        lpips_weight: Weight for LPIPS loss. Default: 0.0.
        id_weight: Weight for ID loss. Default: 0.0.
        eyes_mouth_weight: Weight for eyes/mouth loss. Default: 0.0.
        gaze_weight: Weight for gaze/eye loss. Default: 0.0.
        use_multiscale_dssim: Use multi-scale DSSIM. Default: True.
    """

    def __init__(
        self,
        dssim_weight: float = 10.0,
        l1_weight: float = 10.0,
        lpips_weight: float = 0.0,
        id_weight: float = 0.0,
        eyes_mouth_weight: float = 0.0,
        gaze_weight: float = 0.0,
        use_multiscale_dssim: bool = True,
    ) -> None:
        super().__init__()

        self.dssim_weight = dssim_weight
        self.l1_weight = l1_weight
        self.lpips_weight = lpips_weight
        self.id_weight = id_weight
        self.eyes_mouth_weight = eyes_mouth_weight
        self.gaze_weight = gaze_weight

        # Initialize loss functions
        if use_multiscale_dssim:
            self.dssim = MultiScaleDSSIMLoss()
        else:
            self.dssim = DSSIMLoss()

        self.lpips = LPIPSLoss() if lpips_weight > 0 else None
        self.id_loss = IDLoss() if id_weight > 0 else None
        self.eyes_mouth = EyesMouthLoss() if eyes_mouth_weight > 0 else None
        self.gaze_loss = GazeLoss() if gaze_weight > 0 else None

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        landmarks: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute combined loss.

        Args:
            pred: Predicted image (B, C, H, W).
            target: Target image (B, C, H, W).
            landmarks: Optional landmarks for eyes/mouth loss.

        Returns:
            Tuple of (total_loss, loss_dict).
        """
        losses = {}
        total = torch.tensor(0.0, device=pred.device)

        # DSSIM
        if self.dssim_weight > 0:
            loss_dssim = self.dssim(pred, target)
            losses["dssim"] = loss_dssim
            total = total + self.dssim_weight * loss_dssim

        # L1
        if self.l1_weight > 0:
            loss_l1 = F.l1_loss(pred, target)
            losses["l1"] = loss_l1
            total = total + self.l1_weight * loss_l1

        # LPIPS
        if self.lpips is not None and self.lpips_weight > 0:
            loss_lpips = self.lpips(pred, target)
            losses["lpips"] = loss_lpips
            total = total + self.lpips_weight * loss_lpips

        # ID Loss
        if self.id_loss is not None and self.id_weight > 0:
            loss_id = self.id_loss(pred, target)
            losses["id"] = loss_id
            total = total + self.id_weight * loss_id

        # Eyes/Mouth
        if self.eyes_mouth is not None and self.eyes_mouth_weight > 0:
            loss_em = self.eyes_mouth(pred, target, landmarks)
            losses["eyes_mouth"] = loss_em
            total = total + self.eyes_mouth_weight * loss_em

        # Gaze/Eye
        if self.gaze_loss is not None and self.gaze_weight > 0:
            loss_gaze = self.gaze_loss(pred, target, landmarks)
            losses["gaze"] = loss_gaze
            total = total + self.gaze_weight * loss_gaze

        losses["total"] = total
        return total, losses


class GANLoss(nn.Module):
    """
    GAN Loss for generator training.

    Supports multiple GAN loss modes for flexibility in training dynamics.

    Args:
        mode: Loss mode - 'vanilla', 'lsgan', or 'hinge'. Default: 'vanilla'.
        target_real_label: Target label for real images. Default: 1.0.
        target_fake_label: Target label for fake images. Default: 0.0.

    Example:
        >>> gan_loss = GANLoss(mode='vanilla')
        >>> d_fake = discriminator(generated_image)
        >>> g_loss = gan_loss(d_fake, target_is_real=True)  # Generator wants D to classify as real
    """

    def __init__(
        self,
        mode: str = "vanilla",
        target_real_label: float = 1.0,
        target_fake_label: float = 0.0,
    ) -> None:
        super().__init__()

        self.mode = mode
        self.register_buffer("real_label", torch.tensor(target_real_label))
        self.register_buffer("fake_label", torch.tensor(target_fake_label))

        if mode == "vanilla":
            self.loss_fn = nn.BCEWithLogitsLoss()
        elif mode == "lsgan":
            self.loss_fn = nn.MSELoss()
        elif mode == "hinge":
            self.loss_fn = None  # Custom implementation
        else:
            raise ValueError(f"Unknown GAN loss mode: {mode}")

    def _get_target_tensor(
        self, prediction: torch.Tensor, target_is_real: bool
    ) -> torch.Tensor:
        """Create target tensor with same shape as prediction."""
        target_val = self.real_label if target_is_real else self.fake_label
        return target_val.expand_as(prediction)

    def forward(
        self,
        prediction: torch.Tensor,
        target_is_real: bool,
    ) -> torch.Tensor:
        """
        Compute GAN loss.

        Args:
            prediction: Discriminator output logits.
            target_is_real: Whether target should be real (True) or fake (False).

        Returns:
            Loss value (scalar).
        """
        if self.mode == "hinge":
            if target_is_real:
                return -prediction.mean()
            else:
                return prediction.mean()
        else:
            target = self._get_target_tensor(prediction, target_is_real)
            return self.loss_fn(prediction, target)


class DiscriminatorLoss(nn.Module):
    """
    Discriminator loss combining real and fake classification losses.

    Computes: (loss_real + loss_fake) / 2

    The discriminator learns to output high values for real images
    and low values for generated (fake) images.

    Args:
        mode: Loss mode - 'vanilla', 'lsgan', or 'hinge'. Default: 'vanilla'.

    Example:
        >>> d_loss_fn = DiscriminatorLoss(mode='vanilla')
        >>> d_real = discriminator(real_image)
        >>> d_fake = discriminator(fake_image.detach())
        >>> d_loss = d_loss_fn(d_real, d_fake)
    """

    def __init__(self, mode: str = "vanilla") -> None:
        super().__init__()

        self.mode = mode
        self.gan_loss = GANLoss(mode=mode)

    def forward(
        self,
        d_real: torch.Tensor,
        d_fake: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute discriminator loss.

        Args:
            d_real: Discriminator output for real images.
            d_fake: Discriminator output for fake/generated images.

        Returns:
            Combined discriminator loss (scalar).
        """
        if self.mode == "hinge":
            # Hinge loss: max(0, 1 - D(real)) + max(0, 1 + D(fake))
            loss_real = F.relu(1.0 - d_real).mean()
            loss_fake = F.relu(1.0 + d_fake).mean()
        else:
            loss_real = self.gan_loss(d_real, target_is_real=True)
            loss_fake = self.gan_loss(d_fake, target_is_real=False)

        return (loss_real + loss_fake) * 0.5


class TotalVariationLoss(nn.Module):
    """
    Total Variation loss to suppress noise and artifacts.

    Encourages spatial smoothness by penalizing differences
    between neighboring pixels. Useful in GAN training to
    reduce random bright dots and artifacts.

    Args:
        weight: Loss weight multiplier. Default: 1e-6.

    Example:
        >>> tv_loss = TotalVariationLoss(weight=1e-6)
        >>> loss = tv_loss(generated_image)
    """

    def __init__(self, weight: float = 1e-6) -> None:
        super().__init__()
        self.weight = weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute total variation loss.

        Args:
            x: Input tensor (B, C, H, W).

        Returns:
            TV loss value (scalar).
        """
        # Horizontal differences
        diff_h = x[:, :, 1:, :] - x[:, :, :-1, :]
        # Vertical differences
        diff_w = x[:, :, :, 1:] - x[:, :, :, :-1]

        return self.weight * (diff_h.pow(2).mean() + diff_w.pow(2).mean())


# =============================================================================
# Temporal Losses for Video Consistency
# =============================================================================


class TemporalConsistencyLoss(nn.Module):
    """
    Frame-to-frame consistency loss for temporal smoothness.

    Penalizes large differences between consecutive frames to encourage
    smooth transitions and reduce flickering artifacts in video.

    Args:
        mode: Loss mode - 'l1', 'l2', or 'ssim'. Default: 'l1'.
        weight: Loss weight multiplier. Default: 1.0.

    Example:
        >>> loss_fn = TemporalConsistencyLoss()
        >>> sequence = torch.randn(2, 3, 5, 256, 256)  # (B, C, T, H, W)
        >>> loss = loss_fn(sequence)
    """

    def __init__(
        self,
        mode: str = "l1",
        weight: float = 1.0,
    ) -> None:
        super().__init__()
        if mode not in ("l1", "l2", "ssim"):
            raise ValueError(f"Unknown mode: {mode}. Use 'l1', 'l2', or 'ssim'.")
        self.mode = mode
        self.weight = weight
        # Lazy-loaded DSSIM for SSIM mode
        self._dssim_loss = None

    @property
    def dssim_loss(self) -> "DSSIMLoss":
        """Lazy load DSSIM loss for SSIM mode."""
        if self._dssim_loss is None:
            self._dssim_loss = DSSIMLoss()
        return self._dssim_loss

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        """
        Compute temporal consistency loss.

        Args:
            sequence: Input sequence (B, C, T, H, W).

        Returns:
            Temporal consistency loss (scalar).
        """
        if self.mode == "ssim":
            # Vectorized DSSIM: process all frame pairs at once
            B, C, T, H, W = sequence.shape
            if T < 2:
                return torch.tensor(0.0, device=sequence.device, requires_grad=True)

            # Reshape to process all frame pairs in single forward
            # frames_t: (B*(T-1), C, H, W), frames_t1: (B*(T-1), C, H, W)
            frames_t = (
                sequence[:, :, :-1].permute(0, 2, 1, 3, 4).reshape(B * (T - 1), C, H, W)
            )
            frames_t1 = (
                sequence[:, :, 1:].permute(0, 2, 1, 3, 4).reshape(B * (T - 1), C, H, W)
            )
            loss = self.dssim_loss(frames_t, frames_t1)
        else:
            # Compute frame-to-frame differences
            # sequence[:, :, 1:] - sequence[:, :, :-1] gives T-1 difference frames
            diff = sequence[:, :, 1:, :, :] - sequence[:, :, :-1, :, :]

            if self.mode == "l1":
                loss = diff.abs().mean()
            else:  # l2
                loss = diff.pow(2).mean()

        return self.weight * loss


class TemporalGANLoss(nn.Module):
    """
    GAN loss for temporal discriminator.

    Same interface as GANLoss but designed for temporal consistency
    discrimination scores.

    Args:
        mode: Loss mode - 'vanilla', 'lsgan', or 'hinge'. Default: 'vanilla'.
        target_real_label: Target label for real sequences. Default: 1.0.
        target_fake_label: Target label for fake sequences. Default: 0.0.

    Example:
        >>> loss_fn = TemporalGANLoss(mode='vanilla')
        >>> temporal_score = torch.randn(2, 1)
        >>> loss = loss_fn(temporal_score, target_is_real=True)
    """

    def __init__(
        self,
        mode: str = "vanilla",
        target_real_label: float = 1.0,
        target_fake_label: float = 0.0,
    ) -> None:
        super().__init__()

        self.mode = mode
        self.register_buffer("real_label", torch.tensor(target_real_label))
        self.register_buffer("fake_label", torch.tensor(target_fake_label))

        if mode == "vanilla":
            self.loss_fn = nn.BCEWithLogitsLoss()
        elif mode == "lsgan":
            self.loss_fn = nn.MSELoss()
        elif mode == "hinge":
            self.loss_fn = None
        else:
            raise ValueError(f"Unknown temporal GAN loss mode: {mode}")

    def _get_target_tensor(
        self, prediction: torch.Tensor, target_is_real: bool
    ) -> torch.Tensor:
        """Create target tensor with same shape as prediction."""
        target_val = self.real_label if target_is_real else self.fake_label
        return target_val.expand_as(prediction)

    def forward(
        self,
        prediction: torch.Tensor,
        target_is_real: bool,
    ) -> torch.Tensor:
        """
        Compute temporal GAN loss.

        Args:
            prediction: Temporal discriminator output (B, 1).
            target_is_real: Whether target should be real (True) or fake (False).

        Returns:
            Loss value (scalar).
        """
        if self.mode == "hinge":
            if target_is_real:
                return -prediction.mean()
            else:
                return prediction.mean()
        else:
            target = self._get_target_tensor(prediction, target_is_real)
            return self.loss_fn(prediction, target)


class TemporalDiscriminatorLoss(nn.Module):
    """
    Discriminator loss for temporal training.

    Combines real sequence (from video) vs fake sequence (generated).
    Computes: (loss_real + loss_fake) / 2

    Args:
        mode: Loss mode - 'vanilla', 'lsgan', or 'hinge'. Default: 'vanilla'.

    Example:
        >>> loss_fn = TemporalDiscriminatorLoss(mode='vanilla')
        >>> d_real = temporal_disc(real_sequence)
        >>> d_fake = temporal_disc(fake_sequence.detach())
        >>> loss = loss_fn(d_real, d_fake)
    """

    def __init__(self, mode: str = "vanilla") -> None:
        super().__init__()

        self.mode = mode
        self.gan_loss = TemporalGANLoss(mode=mode)

    def forward(
        self,
        d_real: torch.Tensor,
        d_fake: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute temporal discriminator loss.

        Args:
            d_real: Discriminator output for real sequences.
            d_fake: Discriminator output for fake/generated sequences.

        Returns:
            Combined discriminator loss (scalar).
        """
        if self.mode == "hinge":
            # Hinge loss: max(0, 1 - D(real)) + max(0, 1 + D(fake))
            loss_real = F.relu(1.0 - d_real).mean()
            loss_fake = F.relu(1.0 + d_fake).mean()
        else:
            loss_real = self.gan_loss(d_real, target_is_real=True)
            loss_fake = self.gan_loss(d_fake, target_is_real=False)

        return (loss_real + loss_fake) * 0.5


# =============================================================================
# Style Losses for Face/Background (DeepFaceLab Legacy)
# =============================================================================


def gram_matrix(x: torch.Tensor) -> torch.Tensor:
    """
    Compute Gram matrix for style loss.

    The Gram matrix captures feature correlations and is used
    to measure style similarity between images.

    Args:
        x: Feature tensor (B, C, H, W).

    Returns:
        Gram matrix (B, C, C) normalized by spatial dimensions.
    """
    b, c, h, w = x.shape
    features = x.view(b, c, -1)
    gram = torch.bmm(features, features.transpose(1, 2))
    return gram / (c * h * w)


def face_style_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    pred_mask: torch.Tensor,
    target_mask: torch.Tensor,
    blur_radius: int = 32,
    weight: float = 1.0,
) -> torch.Tensor:
    """
    Face region style loss using Gram matrices.

    Port of DeepFaceLab's face_style_power from Model_SAEHD.
    Computes style similarity between predicted swapped face
    and target destination face using Gram matrices.

    Args:
        pred: Predicted swapped face (B, C, H, W).
        target: Target destination face (B, C, H, W).
        pred_mask: Mask for predicted face (B, 1, H, W).
        target_mask: Mask for target face (B, 1, H, W).
        blur_radius: Gaussian blur radius before Gram computation. Default: 32.
        weight: Loss weight multiplier. Default: 1.0.

    Returns:
        Face style loss (scalar).

    Example:
        >>> pred = torch.randn(2, 3, 256, 256)
        >>> target = torch.randn(2, 3, 256, 256)
        >>> mask = torch.ones(2, 1, 256, 256)
        >>> loss = face_style_loss(pred, target, mask, mask)
    """
    # Apply masks
    pred_masked = pred * pred_mask
    target_masked = target * target_mask

    # Optional blur for smoother style matching
    if blur_radius > 0:
        kernel_size = blur_radius * 2 + 1
        sigma = blur_radius / 3.0
        pred_masked = safe_gaussian_blur2d(
            pred_masked, (kernel_size, kernel_size), (sigma, sigma)
        )
        target_masked = safe_gaussian_blur2d(
            target_masked, (kernel_size, kernel_size), (sigma, sigma)
        )

    # Compute Gram matrices
    gram_pred = gram_matrix(pred_masked)
    gram_target = gram_matrix(target_masked.detach())

    return weight * F.mse_loss(gram_pred, gram_target)


def bg_style_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    weight: float = 1.0,
    resolution: int = 256,
) -> torch.Tensor:
    """
    Background region style loss using DSSIM + L2.

    Port of DeepFaceLab's bg_style_power from Model_SAEHD.
    Computes style similarity for background (non-face) regions
    using a combination of DSSIM and L2 loss.

    Args:
        pred: Predicted image (B, C, H, W).
        target: Target image (B, C, H, W).
        mask: Face mask (B, 1, H, W). 1=face, 0=background.
        weight: Loss weight multiplier. Default: 1.0.
        resolution: Image resolution for DSSIM filter size. Default: 256.

    Returns:
        Background style loss (scalar).

    Example:
        >>> pred = torch.randn(2, 3, 256, 256)
        >>> target = torch.randn(2, 3, 256, 256)
        >>> mask = torch.ones(2, 1, 256, 256)
        >>> mask[:, :, 128:, :] = 0  # Bottom half is background
        >>> loss = bg_style_loss(pred, target, mask)
    """
    # Get background regions
    anti_mask = 1.0 - mask
    pred_bg = pred * anti_mask
    target_bg = target.detach() * anti_mask

    # DSSIM component
    # Filter size ~22 for 256 resolution (legacy formula)
    filter_size = max(3, int(resolution / 11.6)) | 1  # Ensure odd
    dssim_fn = DSSIMLoss(filter_size=filter_size)
    dssim = dssim_fn(pred_bg, target_bg)

    # L2 component
    l2 = F.mse_loss(pred_bg, target_bg)

    # Combined loss with legacy weight multiplier
    return weight * 10.0 * (dssim + l2)
