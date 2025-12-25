"""Contains Pydantic models for prediction API.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

from typing import Dict, Optional, Any, List
from enum import Enum
from datetime import datetime

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


class BatchPredictRequest(BaseModel):
    """Model for batch prediction request parameters."""

    device: Optional[str] = Field(
        default="default",
        description="Device to run inference on. Options: ['cpu', 'mps', 'cuda']",
        pattern="^(default|cpu|mps|cuda)$"
    )
    render: bool = Field(
        default=False,
        description="Whether to render the 3D Gaussians into a video"
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
    filename: str = Field(
        description="Original filename of the input image"
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


class TaskStatusEnum(str, Enum):
    """Enum for task status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskStatus(BaseModel):
    """Model for task status data."""

    task_id: str = Field(
        description="Unique identifier for the task"
    )
    status: TaskStatusEnum = Field(
        description="Current status of the task"
    )
    created_at: datetime = Field(
        description="Time when the task was created"
    )
    started_at: Optional[datetime] = Field(
        default=None,
        description="Time when the task started processing"
    )
    completed_at: Optional[datetime] = Field(
        default=None,
        description="Time when the task completed"
    )
    total_images: int = Field(
        description="Total number of images to process"
    )
    processed_images: int = Field(
        description="Number of images already processed"
    )
    current_image: Optional[str] = Field(
        default=None,
        description="Filename of the current image being processed"
    )
    progress: float = Field(
        description="Progress percentage (0-100)"
    )
    estimated_time_remaining: Optional[float] = Field(
        default=None,
        description="Estimated time remaining in seconds"
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if task failed"
    )


class BatchPredictResult(BaseModel):
    """Model for batch prediction result data."""

    task_id: str = Field(
        description="Unique identifier for the task"
    )
    status: TaskStatusEnum = Field(
        description="Final status of the task"
    )
    results: List[PredictResult] = Field(
        description="List of prediction results for each image"
    )
    total_inference_time: float = Field(
        description="Total inference time in seconds"
    )


class TaskResult(BaseModel):
    """Model for task result response."""

    task_id: str = Field(
        description="Unique identifier for the task"
    )
    status: TaskStatusEnum = Field(
        description="Current status of the task"
    )
    result: Optional[BatchPredictResult] = Field(
        default=None,
        description="Batch prediction result if task completed"
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if task failed"
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
