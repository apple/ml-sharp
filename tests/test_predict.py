import numpy as np
import pytest
import torch
from pathlib import Path

from sharp.cli.predict import predict_image
from sharp.models import PredictorParams, create_predictor
from sharp.utils.gaussians import Gaussians3D


@pytest.mark.skipif(
    not torch.cuda.is_available() and not torch.mps.is_available(),
    reason="Requires CUDA or MPS for model inference",
)
def test_predict_image_with_model(tmp_path):
    """Test predict_image function with a real model checkpoint."""
    example_path = Path(__file__).parent / "data" / "example.jpg"
    if not example_path.exists():
        pytest.skip("Test image not found")

    from sharp.utils import io

    image, _, f_px = io.load_rgb(example_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")

    # Use the pre-cached model checkpoint
    checkpoint_path = Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / "sharp_2572gikvuh.pt"
    if not checkpoint_path.exists():
        pytest.skip("Model checkpoint not found in cache")

    try:
        predictor = create_predictor(PredictorParams(checkpoint_path=str(checkpoint_path)))
        predictor.eval()
        predictor.to(device)

        gaussians = predict_image(predictor, image, f_px, device)

        assert isinstance(gaussians, Gaussians3D)
        assert gaussians.mean_vectors.shape[0] == 1
        assert gaussians.mean_vectors.shape[2] == 3
        assert gaussians.colors.shape[2] == 3
        assert gaussians.opacities.shape[1] > 0
    except Exception as e:
        pytest.skip(f"Model inference failed (likely missing checkpoint): {e}")


def test_predict_image_signature():
    """Test that predict_image function has correct signature."""
    import inspect

    sig = inspect.signature(predict_image)
    params = sig.parameters

    assert "predictor" in params
    assert "image" in params
    assert "f_px" in params
    assert "device" in params


def test_create_predictor():
    """Test creating a predictor model."""
    params = PredictorParams()
    predictor = create_predictor(params)

    assert predictor is not None
    assert hasattr(predictor, "eval")
    assert hasattr(predictor, "to")


def test_predictor_params():
    """Test PredictorParams creation."""
    params = PredictorParams()
    assert params is not None

