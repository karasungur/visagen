"""
Loss Functions for Visagen Training.

Implements modern perceptual loss functions:
- DSSIM: Differentiable Structural Similarity
- LPIPS: Learned Perceptual Image Patch Similarity
- IDLoss: ArcFace-based Identity Preservation
- EyesMouthLoss: Priority loss for facial details
- StyleLoss: Gram matrix based style transfer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        """Create 2D Gaussian kernel."""
        coords = torch.arange(size, dtype=torch.float32)
        coords -= (size - 1) / 2.0

        g = coords**2
        g = torch.exp(-g / (2 * sigma**2))

        g_2d = g.unsqueeze(0) * g.unsqueeze(1)
        g_2d = g_2d / g_2d.sum()

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

        # Expand kernel for all channels
        kernel = self.kernel.repeat(c, 1, 1, 1)

        # Constants
        c1 = (self.k1 * self.max_val) ** 2
        c2 = (self.k2 * self.max_val) ** 2

        # Compute means using depthwise convolution
        mu1 = F.conv2d(pred, kernel, padding=0, groups=c)
        mu2 = F.conv2d(target, kernel, padding=0, groups=c)

        mu1_sq = mu1**2
        mu2_sq = mu2**2
        mu1_mu2 = mu1 * mu2

        # Compute variances
        sigma1_sq = F.conv2d(pred**2, kernel, padding=0, groups=c) - mu1_sq
        sigma2_sq = F.conv2d(target**2, kernel, padding=0, groups=c) - mu2_sq
        sigma12 = F.conv2d(pred * target, kernel, padding=0, groups=c) - mu1_mu2

        # SSIM formula
        luminance = (2 * mu1_mu2 + c1) / (mu1_sq + mu2_sq + c1)
        contrast_structure = (2 * sigma12 + c2) / (sigma1_sq + sigma2_sq + c2)

        ssim_map = luminance * contrast_structure

        # DSSIM = (1 - SSIM) / 2
        dssim = (1.0 - ssim_map.mean(dim=[2, 3])) / 2.0

        return dssim.mean()


class MultiScaleDSSIMLoss(nn.Module):
    """
    Multi-scale DSSIM loss.

    Computes DSSIM at multiple filter sizes for better
    structural similarity across scales.

    Args:
        filter_sizes: List of filter sizes. Default: [11, 23].
        weights: Weights for each scale. Default: equal weights.
    """

    def __init__(
        self,
        filter_sizes: tuple[int, ...] = (11, 23),
        weights: tuple[float, ...] | None = None,
    ) -> None:
        super().__init__()

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

    def _ensure_lpips(self) -> None:
        """Lazy load LPIPS model."""
        if self._lpips is None:
            try:
                import lpips

                self._lpips = lpips.LPIPS(net=self.net)
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

        # Move model to same device as input
        if self._lpips.parameters().__next__().device != pred.device:
            self._lpips = self._lpips.to(pred.device)

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
                # Return zero embedding if no face detected
                embeddings.append(torch.zeros(512))

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

        # Cosine similarity
        cos_sim = F.cosine_similarity(pred_emb, target_emb, dim=1)

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
        weight_multiplier: float = 30.0,
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
        """Compute Gram matrix."""
        b, c, h, w = x.shape
        features = x.view(b, c, h * w)
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
        use_multiscale_dssim: Use multi-scale DSSIM. Default: True.
    """

    def __init__(
        self,
        dssim_weight: float = 10.0,
        l1_weight: float = 10.0,
        lpips_weight: float = 0.0,
        id_weight: float = 0.0,
        eyes_mouth_weight: float = 0.0,
        use_multiscale_dssim: bool = True,
    ) -> None:
        super().__init__()

        self.dssim_weight = dssim_weight
        self.l1_weight = l1_weight
        self.lpips_weight = lpips_weight
        self.id_weight = id_weight
        self.eyes_mouth_weight = eyes_mouth_weight

        # Initialize loss functions
        if use_multiscale_dssim:
            self.dssim = MultiScaleDSSIMLoss()
        else:
            self.dssim = DSSIMLoss()

        self.lpips = LPIPSLoss() if lpips_weight > 0 else None
        self.id_loss = IDLoss() if id_weight > 0 else None
        self.eyes_mouth = EyesMouthLoss() if eyes_mouth_weight > 0 else None

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
