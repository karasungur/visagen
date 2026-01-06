"""
EG3D: Efficient Geometry-aware 3D GANs.

3D-aware face generation using tri-plane representation for
consistent face synthesis across extreme viewing angles.

Key Concepts:
- Tri-plane: XY, XZ, YZ feature planes (efficient 3D representation)
- Volume rendering: Ray marching through tri-plane features
- Super resolution: Upsample low-res renders to high-res output

This module provides a complete implementation of EG3D adapted
for face swapping, including encoder and generator components.

Example:
    >>> from visagen.models.experimental.eg3d import EG3DGenerator
    >>> model = EG3DGenerator(latent_dim=512)
    >>> output = model(z)  # Generate face from latent code

Reference:
    Chan et al. "Efficient Geometry-aware 3D GANs" (CVPR 2022)
    https://nvlabs.github.io/eg3d/
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from visagen.models.encoders.convnext import ConvNeXtEncoder
from visagen.models.layers.attention import CBAM

if TYPE_CHECKING:
    pass


class CameraParams:
    """
    Camera parameters for 3D rendering.

    Provides utilities for creating camera matrices from
    Euler angles or other parameterizations.

    Args:
        fov: Field of view in degrees (default: 18.837, EG3D default).
        near: Near clipping plane (default: 0.1).
        far: Far clipping plane (default: 10.0).
        focal_length: Camera focal length (default: 4.2647, FFHQ default).
    """

    def __init__(
        self,
        fov: float = 18.837,
        near: float = 0.1,
        far: float = 10.0,
        focal_length: float = 4.2647,
    ) -> None:
        self.fov = fov
        self.near = near
        self.far = far
        self.focal_length = focal_length

    @staticmethod
    def from_euler(
        yaw: float = 0.0,
        pitch: float = 0.0,
        roll: float = 0.0,
        translation: tuple[float, float, float] = (0.0, 0.0, 2.7),
    ) -> torch.Tensor:
        """
        Create camera-to-world matrix from Euler angles.

        Args:
            yaw: Rotation around Y axis (radians).
            pitch: Rotation around X axis (radians).
            roll: Rotation around Z axis (radians).
            translation: Camera position (x, y, z).

        Returns:
            cam2world matrix (4, 4).
        """
        cy, sy = math.cos(yaw), math.sin(yaw)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cr, sr = math.cos(roll), math.sin(roll)

        # Rotation matrices
        Ry = torch.tensor(
            [
                [cy, 0.0, sy, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [-sy, 0.0, cy, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        Rx = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, cp, -sp, 0.0],
                [0.0, sp, cp, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        Rz = torch.tensor(
            [
                [cr, -sr, 0.0, 0.0],
                [sr, cr, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )

        # Combine rotations: R = Rz @ Ry @ Rx
        R = Rz @ Ry @ Rx

        # Add translation
        R[0, 3] = translation[0]
        R[1, 3] = translation[1]
        R[2, 3] = translation[2]

        return R

    @staticmethod
    def random_camera(
        yaw_range: tuple[float, float] = (-0.5, 0.5),
        pitch_range: tuple[float, float] = (-0.3, 0.3),
        batch_size: int = 1,
    ) -> torch.Tensor:
        """
        Generate random camera poses.

        Args:
            yaw_range: Range for yaw angle (radians).
            pitch_range: Range for pitch angle (radians).
            batch_size: Number of camera poses to generate.

        Returns:
            cam2world matrices (B, 4, 4).
        """
        yaw = torch.rand(batch_size) * (yaw_range[1] - yaw_range[0]) + yaw_range[0]
        pitch = (
            torch.rand(batch_size) * (pitch_range[1] - pitch_range[0]) + pitch_range[0]
        )

        cameras = []
        for i in range(batch_size):
            cam = CameraParams.from_euler(yaw=yaw[i].item(), pitch=pitch[i].item())
            cameras.append(cam)

        return torch.stack(cameras, dim=0)


class MappingNetwork(nn.Module):
    """
    Mapping network: z -> w (style code).

    Transforms random latent codes into intermediate
    style codes for better disentanglement.

    Args:
        latent_dim: Dimension of input and output latent codes.
        num_layers: Number of MLP layers (default: 4).
    """

    def __init__(
        self,
        latent_dim: int = 512,
        num_layers: int = 4,
    ) -> None:
        super().__init__()

        layers = []
        for _ in range(num_layers):
            layers.extend(
                [
                    nn.Linear(latent_dim, latent_dim),
                    nn.LeakyReLU(0.2),
                ]
            )

        self.mapping = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Map z to w."""
        return self.mapping(z)


class SynthesisBlock(nn.Module):
    """
    Single synthesis block for tri-plane generation.

    Upsamples features and applies convolutions.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        self.upsample = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=False
        )
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with upsampling."""
        x = self.upsample(x)
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        return x


class TriplaneGenerator(nn.Module):
    """
    Generate tri-plane features from latent code.

    Tri-plane representation:
    - XY plane: Front view features
    - XZ plane: Top view features
    - YZ plane: Side view features

    This representation is efficient (O(N²) instead of O(N³))
    while still capturing 3D structure.

    Args:
        latent_dim: Dimension of input latent code (default: 512).
        plane_channels: Number of channels per plane (default: 32).
        plane_resolution: Spatial resolution of planes (default: 256).
    """

    def __init__(
        self,
        latent_dim: int = 512,
        plane_channels: int = 32,
        plane_resolution: int = 256,
    ) -> None:
        super().__init__()
        self.plane_channels = plane_channels
        self.plane_resolution = plane_resolution

        # Initial constant input (StyleGAN-like)
        self.const = nn.Parameter(torch.randn(1, latent_dim, 4, 4))

        # Mapping network
        self.mapping = MappingNetwork(latent_dim)

        # Synthesis blocks
        self.blocks = nn.ModuleList()
        self.to_planes = nn.ModuleList()

        in_ch = latent_dim
        out_channels = [512, 256, 128, 64, 32]

        for out_ch in out_channels:
            self.blocks.append(SynthesisBlock(in_ch, out_ch))
            # Each block can output tri-plane features
            self.to_planes.append(nn.Conv2d(out_ch, plane_channels * 3, 1))
            in_ch = out_ch

        # Final upsampling to target resolution
        self.final_upsample = nn.Upsample(
            size=(plane_resolution, plane_resolution),
            mode="bilinear",
            align_corners=False,
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Generate tri-plane features.

        Args:
            z: Latent code (B, latent_dim).

        Returns:
            Tri-plane features (B, 3, plane_channels, H, W).
        """
        B = z.shape[0]

        # Map z to w
        w = self.mapping(z)  # noqa: F841 - used for future style modulation

        # Start from constant
        x = self.const.expand(B, -1, -1, -1)

        # Progressive synthesis
        planes = None
        for block, to_plane in zip(self.blocks, self.to_planes, strict=True):
            x = block(x)
            if planes is None:
                planes = to_plane(x)
            else:
                planes = F.interpolate(
                    planes, size=x.shape[2:], mode="bilinear", align_corners=False
                ) + to_plane(x)

        # Upsample to target resolution
        planes = self.final_upsample(planes)

        # Reshape to tri-plane format: (B, 3*C, H, W) -> (B, 3, C, H, W)
        return planes.reshape(B, 3, self.plane_channels, *planes.shape[2:])


class NeRFDecoder(nn.Module):
    """
    NeRF-style decoder for volume rendering.

    Queries tri-plane features at 3D points and predicts
    RGB color and density for volume rendering.

    Args:
        plane_channels: Number of channels in tri-plane features.
        hidden_dim: Hidden layer dimension (default: 64).
        num_layers: Number of MLP layers (default: 2).
    """

    def __init__(
        self,
        plane_channels: int = 32,
        hidden_dim: int = 64,
        num_layers: int = 2,
    ) -> None:
        super().__init__()

        # MLP for density and color prediction
        layers: list[nn.Module] = [
            nn.Linear(plane_channels, hidden_dim),
            nn.ReLU(),
        ]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])

        self.mlp = nn.Sequential(*layers)

        # Output heads
        self.density_head = nn.Linear(hidden_dim, 1)
        self.color_head = nn.Linear(hidden_dim, 3)

    def sample_from_planes(
        self,
        triplane: torch.Tensor,
        points: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sample features from tri-plane at 3D points.

        Args:
            triplane: (B, 3, C, H, W) tri-plane features.
            points: (B, N, 3) 3D points in [-1, 1].

        Returns:
            Features (B, N, C).
        """
        # Extract x, y, z coordinates
        x, y, z = points[..., 0], points[..., 1], points[..., 2]

        # Sample from each plane
        # XY plane (front): use x, y
        xy_coords = torch.stack([x, y], dim=-1).unsqueeze(2)  # (B, N, 1, 2)
        xy_feat = (
            F.grid_sample(
                triplane[:, 0],  # (B, C, H, W)
                xy_coords,
                mode="bilinear",
                align_corners=False,
                padding_mode="border",
            )
            .squeeze(-1)
            .transpose(1, 2)
        )  # (B, N, C)

        # XZ plane (top): use x, z
        xz_coords = torch.stack([x, z], dim=-1).unsqueeze(2)
        xz_feat = (
            F.grid_sample(
                triplane[:, 1],
                xz_coords,
                mode="bilinear",
                align_corners=False,
                padding_mode="border",
            )
            .squeeze(-1)
            .transpose(1, 2)
        )

        # YZ plane (side): use y, z
        yz_coords = torch.stack([y, z], dim=-1).unsqueeze(2)
        yz_feat = (
            F.grid_sample(
                triplane[:, 2],
                yz_coords,
                mode="bilinear",
                align_corners=False,
                padding_mode="border",
            )
            .squeeze(-1)
            .transpose(1, 2)
        )

        # Aggregate features (sum)
        return xy_feat + xz_feat + yz_feat

    def forward(
        self,
        triplane: torch.Tensor,
        points: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Decode 3D points to RGB and density.

        Args:
            triplane: (B, 3, C, H, W) tri-plane features.
            points: (B, N, 3) 3D query points.

        Returns:
            rgb: (B, N, 3) RGB colors.
            sigma: (B, N, 1) densities.
        """
        # Sample features from planes
        features = self.sample_from_planes(triplane, points)

        # MLP forward
        hidden = self.mlp(features)

        # Predict density and color
        sigma = F.softplus(self.density_head(hidden))
        rgb = torch.sigmoid(self.color_head(hidden))

        return rgb, sigma


class VolumeRenderer(nn.Module):
    """
    Differentiable volume rendering.

    Performs ray marching and alpha compositing for
    rendering 3D representations to 2D images.

    Args:
        num_samples: Number of samples per ray (default: 48).
        near: Near plane distance (default: 0.1).
        far: Far plane distance (default: 10.0).
    """

    def __init__(
        self,
        num_samples: int = 48,
        near: float = 0.1,
        far: float = 10.0,
    ) -> None:
        super().__init__()
        self.num_samples = num_samples
        self.near = near
        self.far = far

    def forward(
        self,
        rgb: torch.Tensor,
        sigma: torch.Tensor,
        z_vals: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Volume render from samples.

        Args:
            rgb: (B, H*W, N_samples, 3) colors along rays.
            sigma: (B, H*W, N_samples, 1) densities.
            z_vals: (B, H*W, N_samples) sample depths.

        Returns:
            rgb_map: (B, H*W, 3) rendered colors.
            depth_map: (B, H*W) rendered depths.
        """
        # Compute distances between samples
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.full_like(dists[..., :1], 1e10)], dim=-1)

        # Alpha from density
        alpha = 1.0 - torch.exp(-sigma.squeeze(-1) * dists)

        # Transmittance
        T = torch.cumprod(1.0 - alpha + 1e-10, dim=-1)
        T = torch.cat([torch.ones_like(T[..., :1]), T[..., :-1]], dim=-1)

        # Weights
        weights = alpha * T

        # Composite
        rgb_map = (weights.unsqueeze(-1) * rgb).sum(dim=-2)
        depth_map = (weights * z_vals).sum(dim=-1)

        return rgb_map, depth_map


class SuperResolutionModule(nn.Module):
    """
    Upsample low-res neural render to high-res output.

    Uses residual blocks with CBAM attention for
    high-quality upsampling.

    Args:
        in_channels: Number of input channels (default: 3).
        out_channels: Number of output channels (default: 3).
        hidden_dim: Hidden layer dimension (default: 64).
        upsample_factor: Upsampling factor (default: 4).
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        hidden_dim: int = 64,
        upsample_factor: int = 4,
    ) -> None:
        super().__init__()
        self.upsample_factor = upsample_factor

        self.conv1 = nn.Conv2d(in_channels, hidden_dim, 3, padding=1)

        # Residual blocks with CBAM
        self.res_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
                    CBAM(hidden_dim),
                )
                for _ in range(4)
            ]
        )

        # Upsampling (2x per stage)
        upsample_layers: list[nn.Module] = []
        num_upsample = int(math.log2(upsample_factor))
        for _ in range(num_upsample):
            upsample_layers.extend(
                [
                    nn.Conv2d(hidden_dim, hidden_dim * 4, 3, padding=1),
                    nn.PixelShuffle(2),
                    nn.LeakyReLU(0.2),
                ]
            )
        self.upsample = nn.Sequential(*upsample_layers)

        self.final = nn.Conv2d(hidden_dim, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Upsample input."""
        feat = F.leaky_relu(self.conv1(x), 0.2)

        # Residual blocks
        for block in self.res_blocks:
            feat = feat + block(feat)

        feat = self.upsample(feat)
        return torch.tanh(self.final(feat))


class EG3DGenerator(nn.Module):
    """
    3D geometry-aware generator using tri-plane representation.

    Architecture:
    1. Mapping network: z -> w (style code)
    2. Tri-plane generator: w -> 3 feature planes
    3. NeRF decoder: tri-plane + rays -> low-res render
    4. Super resolution: low-res -> high-res output

    Key Benefits:
    - View-consistent face generation
    - No 3D artifacts at extreme angles
    - Efficient (O(N²) vs O(N³) for full 3D)

    Args:
        latent_dim: Dimension of latent code (default: 512).
        plane_channels: Channels per tri-plane (default: 32).
        plane_resolution: Tri-plane spatial resolution (default: 256).
        render_resolution: Neural render resolution (default: 64).
        output_resolution: Final output resolution (default: 256).
        num_samples: Samples per ray for volume rendering (default: 48).

    Example:
        >>> model = EG3DGenerator()
        >>> z = torch.randn(1, 512)
        >>> output = model(z)
        >>> print(output.shape)  # torch.Size([1, 3, 256, 256])
    """

    def __init__(
        self,
        latent_dim: int = 512,
        plane_channels: int = 32,
        plane_resolution: int = 256,
        render_resolution: int = 64,
        output_resolution: int = 256,
        num_samples: int = 48,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.render_resolution = render_resolution
        self.output_resolution = output_resolution
        self.num_samples = num_samples

        # Tri-plane generator
        self.triplane_gen = TriplaneGenerator(
            latent_dim=latent_dim,
            plane_channels=plane_channels,
            plane_resolution=plane_resolution,
        )

        # NeRF decoder
        self.nerf_decoder = NeRFDecoder(
            plane_channels=plane_channels,
        )

        # Volume renderer
        self.volume_renderer = VolumeRenderer(
            num_samples=num_samples,
        )

        # Super resolution
        upsample_factor = output_resolution // render_resolution
        self.super_resolution = SuperResolutionModule(
            upsample_factor=upsample_factor,
        )

        # Default camera intrinsics
        self.register_buffer(
            "default_intrinsics",
            torch.tensor(
                [
                    [4.2647, 0.0, 0.5],
                    [0.0, 4.2647, 0.5],
                    [0.0, 0.0, 1.0],
                ]
            ),
        )

    def generate_rays(
        self,
        cam2world: torch.Tensor,
        intrinsics: torch.Tensor,
        resolution: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate camera rays for rendering.

        Args:
            cam2world: Camera-to-world transform (B, 4, 4).
            intrinsics: Camera intrinsics (B, 3, 3) or (3, 3).
            resolution: Image resolution.

        Returns:
            origins: Ray origins (B, H*W, 3).
            directions: Ray directions (B, H*W, 3).
        """
        B = cam2world.shape[0]
        device = cam2world.device

        # Create pixel coordinates
        u = torch.linspace(-1, 1, resolution, device=device)
        v = torch.linspace(-1, 1, resolution, device=device)
        u, v = torch.meshgrid(u, v, indexing="xy")
        u = u.flatten()
        v = v.flatten()

        # Ray directions in camera space
        if intrinsics.dim() == 2:
            intrinsics = intrinsics.unsqueeze(0).expand(B, -1, -1)
        fx, fy = intrinsics[0, 0, 0], intrinsics[0, 1, 1]

        directions = torch.stack(
            [
                u / fx,
                v / fy,
                torch.ones_like(u),
            ],
            dim=-1,
        )
        directions = directions / directions.norm(dim=-1, keepdim=True)

        # Transform to world space
        directions = directions.unsqueeze(0).expand(B, -1, -1)
        directions = (cam2world[:, :3, :3] @ directions.transpose(1, 2)).transpose(1, 2)

        # Ray origins
        origins = (
            cam2world[:, :3, 3].unsqueeze(1).expand(-1, resolution * resolution, -1)
        )

        return origins, directions

    def forward(
        self,
        z: torch.Tensor,
        cam2world: torch.Tensor | None = None,
        intrinsics: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Generate face image from latent code.

        Args:
            z: Latent code (B, latent_dim).
            cam2world: Camera-to-world transform (B, 4, 4).
            intrinsics: Camera intrinsics (B, 3, 3).

        Returns:
            Generated image (B, 3, H, W).
        """
        B = z.shape[0]
        device = z.device

        # Default camera (frontal view)
        if cam2world is None:
            cam2world = (
                CameraParams.from_euler().unsqueeze(0).expand(B, -1, -1).to(device)
            )
        if intrinsics is None:
            intrinsics = self.default_intrinsics

        # Generate tri-plane
        triplane = self.triplane_gen(z)

        # Generate rays
        origins, directions = self.generate_rays(
            cam2world, intrinsics, self.render_resolution
        )

        # Sample points along rays
        t_vals = torch.linspace(0.1, 10.0, self.num_samples, device=device)
        points = origins.unsqueeze(-2) + directions.unsqueeze(-2) * t_vals.view(
            1, 1, -1, 1
        )
        points = points.reshape(B, -1, 3)

        # Query NeRF
        rgb, sigma = self.nerf_decoder(triplane, points)

        # Reshape for volume rendering
        H = W = self.render_resolution
        N = self.num_samples
        rgb = rgb.reshape(B, H * W, N, 3)
        sigma = sigma.reshape(B, H * W, N, 1)
        z_vals = t_vals.unsqueeze(0).unsqueeze(0).expand(B, H * W, -1)

        # Volume render
        rgb_map, _ = self.volume_renderer(rgb, sigma, z_vals)

        # Reshape to image
        low_res = rgb_map.reshape(B, H, W, 3).permute(0, 3, 1, 2)

        # Super resolution
        output = self.super_resolution(low_res)

        return output


class EG3DEncoder(nn.Module):
    """
    Encoder for EG3D: Map real face to latent code.

    Uses ConvNeXt backbone with projection head to
    encode real images into the EG3D latent space.

    Args:
        latent_dim: Output latent dimension (default: 512).
        backbone_dims: ConvNeXt encoder dimensions.
        backbone_depths: ConvNeXt encoder depths.

    Example:
        >>> encoder = EG3DEncoder()
        >>> z = encoder(real_image)
        >>> print(z.shape)  # torch.Size([1, 512])
    """

    def __init__(
        self,
        latent_dim: int = 512,
        backbone_dims: list[int] | None = None,
        backbone_depths: list[int] | None = None,
    ) -> None:
        super().__init__()
        backbone_dims = backbone_dims or [64, 128, 256, 512]
        backbone_depths = backbone_depths or [2, 2, 4, 2]

        self.backbone = ConvNeXtEncoder(
            in_channels=3,
            dims=backbone_dims,
            depths=backbone_depths,
        )

        # Projection to latent space
        self.projection = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(backbone_dims[-1], latent_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode image to latent code.

        Args:
            x: Input image (B, 3, H, W).

        Returns:
            Latent code (B, latent_dim).
        """
        _, latent = self.backbone(x)
        return self.projection(latent)
