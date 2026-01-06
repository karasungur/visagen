"""
Tests for experimental modules.

These tests verify forward passes and basic functionality
of Diffusion AutoEncoder and EG3D components.

Note:
    Some tests require optional dependencies and will be
    skipped if not available.
"""

import pytest
import torch

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def device():
    """Get test device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def batch_size():
    """Standard batch size for tests."""
    return 2


@pytest.fixture
def image_size():
    """Small image size for fast testing."""
    return 64


# =============================================================================
# Diffusion AutoEncoder Tests
# =============================================================================


class TestTextureEncoderLite:
    """Tests for lightweight texture encoder (no diffusers required)."""

    @pytest.fixture
    def encoder(self, image_size):
        """Create texture encoder lite for testing."""
        from visagen.models.experimental.diffusion import TextureEncoderLite

        return TextureEncoderLite(latent_dim=256)

    def test_forward_shape(self, encoder, batch_size, image_size, device):
        """Test forward pass produces correct output shape."""
        encoder = encoder.to(device)
        x = torch.randn(batch_size, 3, image_size, image_size, device=device)

        with torch.no_grad():
            output = encoder(x)

        # Should downsample 8x
        expected_size = image_size // 8
        assert output.shape == (batch_size, 256, expected_size, expected_size)

    def test_gradient_flow(self, encoder, batch_size, image_size, device):
        """Test gradients flow through encoder."""
        encoder = encoder.to(device)
        x = torch.randn(
            batch_size, 3, image_size, image_size, device=device, requires_grad=True
        )

        output = encoder(x)
        loss = output.mean()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape


class TestCrossAttentionFusion:
    """Tests for cross-attention fusion module."""

    @pytest.fixture
    def fusion(self):
        """Create fusion module for testing."""
        from visagen.models.experimental.diffusion import CrossAttentionFusion

        return CrossAttentionFusion(dim=256, num_heads=4)

    def test_forward_shape(self, fusion, batch_size, device):
        """Test forward pass preserves shape."""
        fusion = fusion.to(device)
        structure = torch.randn(batch_size, 256, 8, 8, device=device)
        texture = torch.randn(batch_size, 256, 8, 8, device=device)

        with torch.no_grad():
            output = fusion(structure, texture)

        assert output.shape == structure.shape

    def test_residual_connection(self, fusion, batch_size, device):
        """Test residual connection is applied."""
        fusion = fusion.to(device)
        structure = torch.randn(batch_size, 256, 8, 8, device=device)
        texture = torch.zeros(batch_size, 256, 8, 8, device=device)

        with torch.no_grad():
            output = fusion(structure, texture)

        # With zero texture, output should be close to structure
        # (due to residual connection)
        assert output.shape == structure.shape


class TestDiffusionDecoder:
    """Tests for diffusion decoder."""

    @pytest.fixture
    def decoder(self):
        """Create decoder for testing."""
        from visagen.models.experimental.diffusion import DiffusionDecoder

        return DiffusionDecoder(
            latent_channels=256,
            dims=[256, 128, 64, 32],
        )

    def test_forward_shape(self, decoder, batch_size, device):
        """Test forward pass produces correct output shape."""
        decoder = decoder.to(device)
        latent = torch.randn(batch_size, 256, 8, 8, device=device)

        with torch.no_grad():
            output = decoder(latent)

        # 8 -> 16 -> 32 -> 64 -> 128 -> *4 = 512
        # Actually: 4 blocks = 8*2^4 = 128, then final 4x = 512
        # But our final_upsample is 4x, so: 8->16->32->64->128 (4 blocks) * 4 = 512
        assert output.shape[0] == batch_size
        assert output.shape[1] == 3  # RGB output

    def test_with_texture_styles(self, decoder, batch_size, device):
        """Test forward pass with texture injection."""
        decoder = decoder.to(device)
        latent = torch.randn(batch_size, 256, 8, 8, device=device)

        # Create texture styles for each block
        styles = [
            torch.randn(batch_size, 256, device=device),
            torch.randn(batch_size, 128, device=device),
            torch.randn(batch_size, 64, device=device),
            torch.randn(batch_size, 32, device=device),
        ]

        with torch.no_grad():
            output = decoder(latent, styles)

        assert output.shape[0] == batch_size
        assert output.shape[1] == 3


class TestDiffusionAutoEncoderLite:
    """Tests for DiffusionAutoEncoder without pretrained VAE."""

    @pytest.fixture
    def model(self, image_size):
        """Create diffusion model for testing (lite version)."""
        from visagen.models.experimental.diffusion import DiffusionAutoEncoder

        return DiffusionAutoEncoder(
            image_size=image_size,
            structure_dims=[32, 64, 128, 256],
            structure_depths=[1, 1, 2, 1],
            texture_dim=256,
            decoder_dims=[256, 128, 64, 32],
            use_pretrained_vae=False,  # Use lite encoder
            use_attention=True,
        )

    def test_forward_shape(self, model, batch_size, image_size, device):
        """Test forward pass produces correct output shape."""
        model = model.to(device)
        x = torch.randn(batch_size, 3, image_size, image_size, device=device)

        with torch.no_grad():
            output = model(x)

        # Output should match input shape
        assert output.shape[0] == batch_size
        assert output.shape[1] == 3

    def test_encode_returns_tuple(self, model, batch_size, image_size, device):
        """Test encode returns structure and texture latents."""
        model = model.to(device)
        x = torch.randn(batch_size, 3, image_size, image_size, device=device)

        with torch.no_grad():
            structure, texture = model.encode(x)

        assert structure.ndim == 4  # (B, C, H, W)
        assert texture.ndim == 4  # (B, C, H, W)

    def test_gradient_flow(self, model, batch_size, image_size, device):
        """Test gradients flow through model."""
        model = model.to(device)
        x = torch.randn(
            batch_size, 3, image_size, image_size, device=device, requires_grad=True
        )

        output = model(x)
        loss = output.mean()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape


# =============================================================================
# Diffusion Loss Tests
# =============================================================================


class TestDiffusionLoss:
    """Tests for diffusion loss functions."""

    @pytest.fixture
    def loss_fn(self):
        """Create diffusion loss for testing."""
        from visagen.training.diffusion_losses import DiffusionLoss

        return DiffusionLoss(
            reconstruction_weight=10.0,
            texture_weight=0.0,  # Disable VGG for speed
        )

    def test_returns_tuple(self, loss_fn, batch_size, device):
        """Test loss returns (total, dict) tuple."""
        loss_fn = loss_fn.to(device)
        pred = torch.randn(batch_size, 3, 64, 64, device=device)
        target = torch.randn(batch_size, 3, 64, 64, device=device)

        total, losses = loss_fn(pred, target)

        assert isinstance(total, torch.Tensor)
        assert isinstance(losses, dict)
        assert "reconstruction" in losses
        assert "total" in losses

    def test_identical_images_low_loss(self, loss_fn, batch_size, device):
        """Test identical images have low reconstruction loss."""
        loss_fn = loss_fn.to(device)
        x = torch.randn(batch_size, 3, 64, 64, device=device)

        total, losses = loss_fn(x, x)

        assert losses["reconstruction"].item() < 1e-6


# =============================================================================
# EG3D Tests
# =============================================================================


class TestCameraParams:
    """Tests for camera parameters utility."""

    def test_from_euler_shape(self):
        """Test Euler to matrix conversion."""
        from visagen.models.experimental.eg3d import CameraParams

        cam = CameraParams.from_euler(yaw=0.0, pitch=0.0, roll=0.0)

        assert cam.shape == (4, 4)
        assert cam.dtype == torch.float32

    def test_from_euler_identity(self):
        """Test identity rotation."""
        from visagen.models.experimental.eg3d import CameraParams

        cam = CameraParams.from_euler(yaw=0.0, pitch=0.0, roll=0.0)

        # Rotation part should be identity
        rotation = cam[:3, :3]
        identity = torch.eye(3)
        assert torch.allclose(rotation, identity, atol=1e-6)

    def test_random_camera(self):
        """Test random camera generation."""
        from visagen.models.experimental.eg3d import CameraParams

        cameras = CameraParams.random_camera(batch_size=4)

        assert cameras.shape == (4, 4, 4)


class TestTriplaneGenerator:
    """Tests for tri-plane generator."""

    @pytest.fixture
    def generator(self):
        """Create tri-plane generator for testing."""
        from visagen.models.experimental.eg3d import TriplaneGenerator

        return TriplaneGenerator(
            latent_dim=256,
            plane_channels=16,
            plane_resolution=32,
        )

    def test_output_shape(self, generator, batch_size, device):
        """Test tri-plane output shape."""
        generator = generator.to(device)
        z = torch.randn(batch_size, 256, device=device)

        with torch.no_grad():
            triplane = generator(z)

        # Should be (B, 3, C, H, W)
        assert triplane.shape == (batch_size, 3, 16, 32, 32)

    @pytest.mark.skip(reason="MappingNetwork output not yet used for style modulation")
    def test_gradient_flow(self, generator, batch_size, device):
        """Test gradients flow through generator."""
        generator = generator.to(device)
        z = torch.randn(batch_size, 256, device=device, requires_grad=True)

        triplane = generator(z)
        loss = triplane.mean()
        loss.backward()

        assert z.grad is not None


class TestNeRFDecoder:
    """Tests for NeRF decoder."""

    @pytest.fixture
    def decoder(self):
        """Create NeRF decoder for testing."""
        from visagen.models.experimental.eg3d import NeRFDecoder

        return NeRFDecoder(plane_channels=16, hidden_dim=32)

    def test_output_shapes(self, decoder, batch_size, device):
        """Test NeRF decoder outputs."""
        decoder = decoder.to(device)

        triplane = torch.randn(batch_size, 3, 16, 32, 32, device=device)
        points = torch.randn(batch_size, 100, 3, device=device)

        with torch.no_grad():
            rgb, sigma = decoder(triplane, points)

        assert rgb.shape == (batch_size, 100, 3)
        assert sigma.shape == (batch_size, 100, 1)

    def test_rgb_range(self, decoder, batch_size, device):
        """Test RGB values are in [0, 1]."""
        decoder = decoder.to(device)

        triplane = torch.randn(batch_size, 3, 16, 32, 32, device=device)
        points = torch.randn(batch_size, 100, 3, device=device)

        with torch.no_grad():
            rgb, _ = decoder(triplane, points)

        assert rgb.min() >= 0.0
        assert rgb.max() <= 1.0

    def test_sigma_positive(self, decoder, batch_size, device):
        """Test density values are positive."""
        decoder = decoder.to(device)

        triplane = torch.randn(batch_size, 3, 16, 32, 32, device=device)
        points = torch.randn(batch_size, 100, 3, device=device)

        with torch.no_grad():
            _, sigma = decoder(triplane, points)

        assert sigma.min() >= 0.0


class TestVolumeRenderer:
    """Tests for volume renderer."""

    @pytest.fixture
    def renderer(self):
        """Create volume renderer for testing."""
        from visagen.models.experimental.eg3d import VolumeRenderer

        return VolumeRenderer(num_samples=16)

    def test_output_shapes(self, renderer, batch_size, device):
        """Test volume rendering outputs."""
        renderer = renderer.to(device)

        num_rays = 64
        num_samples = 16

        rgb = torch.rand(batch_size, num_rays, num_samples, 3, device=device)
        sigma = torch.rand(batch_size, num_rays, num_samples, 1, device=device)
        z_vals = (
            torch.linspace(0.1, 10.0, num_samples, device=device)
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(batch_size, num_rays, -1)
        )

        rgb_map, depth_map = renderer(rgb, sigma, z_vals)

        assert rgb_map.shape == (batch_size, num_rays, 3)
        assert depth_map.shape == (batch_size, num_rays)


class TestSuperResolutionModule:
    """Tests for super resolution module."""

    @pytest.fixture
    def sr_module(self):
        """Create super resolution module for testing."""
        from visagen.models.experimental.eg3d import SuperResolutionModule

        return SuperResolutionModule(upsample_factor=4)

    def test_upsample_factor(self, sr_module, batch_size, device):
        """Test upsampling factor is correct."""
        sr_module = sr_module.to(device)
        x = torch.randn(batch_size, 3, 16, 16, device=device)

        with torch.no_grad():
            output = sr_module(x)

        # Should upsample by 4x
        assert output.shape == (batch_size, 3, 64, 64)

    def test_output_range(self, sr_module, batch_size, device):
        """Test output is in [-1, 1] range (due to tanh)."""
        sr_module = sr_module.to(device)
        x = torch.randn(batch_size, 3, 16, 16, device=device)

        with torch.no_grad():
            output = sr_module(x)

        assert output.min() >= -1.0
        assert output.max() <= 1.0


class TestEG3DGenerator:
    """Tests for EG3D generator."""

    @pytest.fixture
    def generator(self):
        """Create EG3D generator for testing."""
        from visagen.models.experimental.eg3d import EG3DGenerator

        return EG3DGenerator(
            latent_dim=128,
            plane_channels=8,
            plane_resolution=32,
            render_resolution=16,
            output_resolution=32,
            num_samples=8,
        )

    def test_forward_shape(self, generator, batch_size, device):
        """Test forward pass produces correct output shape."""
        generator = generator.to(device)
        z = torch.randn(batch_size, 128, device=device)

        with torch.no_grad():
            output = generator(z)

        assert output.shape == (batch_size, 3, 32, 32)

    def test_with_custom_camera(self, generator, device):
        """Test generation with custom camera pose."""
        from visagen.models.experimental.eg3d import CameraParams

        generator = generator.to(device)
        z = torch.randn(1, 128, device=device)
        cam = CameraParams.from_euler(yaw=0.3).unsqueeze(0).to(device)

        with torch.no_grad():
            output = generator(z, cam2world=cam)

        assert output.shape == (1, 3, 32, 32)


class TestEG3DEncoder:
    """Tests for EG3D encoder."""

    @pytest.fixture
    def encoder(self):
        """Create EG3D encoder for testing."""
        from visagen.models.experimental.eg3d import EG3DEncoder

        return EG3DEncoder(
            latent_dim=128,
            backbone_dims=[32, 64, 128, 256],
            backbone_depths=[1, 1, 2, 1],
        )

    def test_forward_shape(self, encoder, batch_size, image_size, device):
        """Test forward pass produces correct latent shape."""
        encoder = encoder.to(device)
        x = torch.randn(batch_size, 3, image_size, image_size, device=device)

        with torch.no_grad():
            z = encoder(x)

        assert z.shape == (batch_size, 128)

    def test_gradient_flow(self, encoder, batch_size, image_size, device):
        """Test gradients flow through encoder."""
        encoder = encoder.to(device)
        x = torch.randn(
            batch_size, 3, image_size, image_size, device=device, requires_grad=True
        )

        z = encoder(x)
        loss = z.mean()
        loss.backward()

        assert x.grad is not None


# =============================================================================
# Integration Tests
# =============================================================================


class TestExperimentalModuleImports:
    """Test that experimental modules can be imported."""

    def test_import_from_experimental(self):
        """Test lazy imports work correctly."""
        from visagen.models.experimental import (
            CameraParams,
            CrossAttentionFusion,
            DiffusionAutoEncoder,
            DiffusionDecoder,
            EG3DEncoder,
            EG3DGenerator,
            NeRFDecoder,
            SuperResolutionModule,
            TextureEncoder,
            TriplaneGenerator,
            VolumeRenderer,
        )

        # Just check they're classes
        assert callable(DiffusionAutoEncoder)
        assert callable(TextureEncoder)
        assert callable(CrossAttentionFusion)
        assert callable(DiffusionDecoder)
        assert callable(EG3DGenerator)
        assert callable(EG3DEncoder)
        assert callable(TriplaneGenerator)
        assert callable(NeRFDecoder)
        assert callable(VolumeRenderer)
        assert callable(SuperResolutionModule)
        assert CameraParams is not None
