import tempfile
from pathlib import Path

import numpy as np
import torch

from sharp.utils.gaussians import Gaussians3D, SceneMetaData, save_ply


def test_gaussians3d_creation():
    """Test creating a Gaussians3D object."""
    num_gaussians = 100
    gaussians = Gaussians3D(
        mean_vectors=torch.randn(1, num_gaussians, 3),
        singular_values=torch.rand(1, num_gaussians, 3),
        quaternions=torch.randn(1, num_gaussians, 4),
        colors=torch.rand(1, num_gaussians, 3),
        opacities=torch.rand(1, num_gaussians),
    )

    assert gaussians.mean_vectors.shape == (1, num_gaussians, 3)
    assert gaussians.singular_values.shape == (1, num_gaussians, 3)
    assert gaussians.quaternions.shape == (1, num_gaussians, 4)
    assert gaussians.colors.shape == (1, num_gaussians, 3)
    assert gaussians.opacities.shape == (1, num_gaussians)


def test_gaussians3d_to_device():
    """Test moving Gaussians3D to a device."""
    num_gaussians = 50
    gaussians = Gaussians3D(
        mean_vectors=torch.randn(1, num_gaussians, 3),
        singular_values=torch.rand(1, num_gaussians, 3),
        quaternions=torch.randn(1, num_gaussians, 4),
        colors=torch.rand(1, num_gaussians, 3),
        opacities=torch.rand(1, num_gaussians),
    )

    gaussians_cpu = gaussians.to(torch.device("cpu"))
    assert gaussians_cpu.mean_vectors.device.type == "cpu"
    assert gaussians_cpu.singular_values.device.type == "cpu"


def test_scene_metadata():
    """Test creating SceneMetaData."""
    metadata = SceneMetaData(
        focal_length_px=1000.0,
        resolution_px=(1920, 1080),
        color_space="linearRGB",
    )

    assert metadata.focal_length_px == 1000.0
    assert metadata.resolution_px == (1920, 1080)
    assert metadata.color_space == "linearRGB"


def test_save_ply(tmp_path):
    """Test saving Gaussians to PLY file."""
    num_gaussians = 10
    gaussians = Gaussians3D(
        mean_vectors=torch.randn(1, num_gaussians, 3),
        singular_values=torch.rand(1, num_gaussians, 3),
        quaternions=torch.randn(1, num_gaussians, 4),
        colors=torch.rand(1, num_gaussians, 3),
        opacities=torch.rand(1, num_gaussians),
    )

    output_path = tmp_path / "test_gaussians.ply"
    save_ply(gaussians, f_px=1000.0, image_shape=(1920, 1080), path=output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0

