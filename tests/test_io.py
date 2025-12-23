import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from sharp.utils import io


def test_load_rgb(tmp_path):
    """Test loading an RGB image."""
    test_image_path = tmp_path / "test.jpg"
    img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    img.save(test_image_path)

    image, icc_profile, f_px = io.load_rgb(test_image_path)

    assert image.shape == (100, 100, 3)
    assert image.dtype == np.uint8
    assert f_px > 0
    assert isinstance(icc_profile, (list, type(None)))


def test_get_supported_image_extensions():
    """Test getting supported image extensions."""
    extensions = io.get_supported_image_extensions()
    assert isinstance(extensions, list)
    assert len(extensions) > 0
    assert ".jpg" in extensions or ".JPG" in extensions
    assert ".png" in extensions or ".PNG" in extensions


def test_get_supported_video_extensions():
    """Test getting supported video extensions."""
    extensions = io.get_supported_video_extensions()
    assert isinstance(extensions, list)
    assert ".mp4" in extensions or ".MP4" in extensions


def test_save_image(tmp_path):
    """Test saving an image."""
    test_image_path = tmp_path / "test_output.jpg"
    img_array = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)

    io.save_image(img_array, test_image_path)

    assert test_image_path.exists()
    loaded_image, _, _ = io.load_rgb(test_image_path)
    assert loaded_image.shape == (50, 50, 3)


def test_convert_focallength():
    """Test focal length conversion."""
    f_px = io.convert_focallength(1920, 1080, 30.0)
    assert f_px > 0
    assert isinstance(f_px, float)


def test_load_example_image():
    """Test loading the example image from tests/data directory."""
    example_path = Path(__file__).parent / "data" / "example.jpg"
    if example_path.exists():
        image, _, f_px = io.load_rgb(example_path)
        assert image.ndim == 3
        assert image.shape[2] == 3
        assert f_px > 0

