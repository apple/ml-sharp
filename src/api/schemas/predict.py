"""Contains Pydantic models for prediction API.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

from typing import Dict, Optional, Any

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Model for prediction request parameters."""

    device: Optional[str] = Field(
        default="default",
        description="Device to run inference on. Options: ['cpu', 'mps', 'cuda']",
        pattern="^(default|cpu|mps|cuda)$"
    )
    render: bool = Field(
        default=False,
        description="Whether to render the 3D Gaussians into a video"
    )
    checkpoint_path: Optional[str] = Field(
        default=None,
        description="Path to a custom model checkpoint"
    )


class PredictResult(BaseModel):
    """Model for prediction result data."""

    gaussians_ply_path: str = Field(
        description="Path to the generated PLY file containing 3D Gaussians"
    )
    render_video_path: Optional[str] = Field(
        default=None,
        description="Path to the rendered video (if rendering was enabled)"
    )
    inference_time: float = Field(
        description="Inference time in seconds"
    )
    image_dimensions: Dict[str, int] = Field(
        description="Dimensions of the input image"
    )
    num_gaussians: int = Field(
        description="Number of 3D Gaussians generated"
    )


class PredictResponse(BaseModel):
    """Model for prediction API response."""

    success: bool = Field(
        description="Whether the prediction was successful"
    )
    message: str = Field(
        description="Response message"
    )
    result: Optional[PredictResult] = Field(
        default=None,
        description="Prediction result data (if successful)"
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message (if unsuccessful)"
    )


class HealthResponse(BaseModel):
    """Model for health check API response."""

    status: str = Field(
        description="Health status"
    )
    models_loaded: int = Field(
        description="Number of models currently loaded in cache"
    )
    timestamp: str = Field(
        description="Current timestamp"
    )
