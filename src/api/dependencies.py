"""Contains dependencies for the API application.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch

from sharp.models import PredictorParams, RGBGaussianPredictor, create_predictor

LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = Path(__file__).parent.parent.parent / "model" / "sharp_2572gikvuh.pt"


class ModelCache:
    """Singleton class for caching loaded models."""

    _instance: ModelCache|None = None
    _models: dict[str, RGBGaussianPredictor] = {}

    def __new__(cls) -> ModelCache:
        """Create or return the singleton instance."""
        if cls._instance is None:
            cls._instance = super(ModelCache, cls).__new__(cls)
        return cls._instance

    def get_model(
        self, checkpoint_path: Path | None = None, device: str = "cuda"
    ) -> RGBGaussianPredictor:
        """Get a model from cache or load it if not present.

        Args:
            checkpoint_path: Path to the model checkpoint. If None, uses default model.
            device: Device to load the model on.

        Returns:
            The loaded model.
        """
        # Create a unique key for the model based on checkpoint path
        model_key = str(checkpoint_path) if checkpoint_path else "default"

        # If model is already cached, return it
        if model_key in self._models:
            LOGGER.info(f"Using cached model: {model_key}")
            return self._models[model_key]

        # Load checkpoint
        LOGGER.info(f"Loading model: {model_key}")
        # Always use the default model path
        final_checkpoint_path = DEFAULT_MODEL_PATH
        LOGGER.info(f"Loading checkpoint from {final_checkpoint_path}")
        state_dict = torch.load(final_checkpoint_path, weights_only=True)

        # Create and configure model
        gaussian_predictor = create_predictor(PredictorParams())
        gaussian_predictor.load_state_dict(state_dict)
        gaussian_predictor.eval()
        gaussian_predictor.to(device)

        # Cache the model
        self._models[model_key] = gaussian_predictor
        LOGGER.info(f"Model cached successfully: {model_key}")

        return gaussian_predictor


def get_model_cache() -> ModelCache:
    """Dependency function to get the model cache instance."""
    return ModelCache()
