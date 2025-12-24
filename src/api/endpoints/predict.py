"""Contains prediction API endpoint implementation.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass, field

import numpy as np
import torch
from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse

from sharp.models.predictor import RGBGaussianPredictor
from sharp.utils import io
from sharp.utils.gaussians import Gaussians3D, SceneMetaData, save_ply, unproject_gaussians

from ..dependencies import ModelCache, get_model_cache
from ..schemas import (
    PredictRequest, PredictResponse, PredictResult,
    BatchPredictRequest, TaskStatus, TaskStatusEnum, BatchPredictResult, TaskResult
)
from .render import render_gaussians

LOGGER = logging.getLogger(__name__)
router = APIRouter()

# Create directory for results
# RESULTS_DIR = Path("/tmp/ml-sharp/results")
RESULTS_DIR = Path(__file__).parent.parent.parent.parent / "tmp" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class BatchTask:
    """Dataclass for storing batch prediction task information."""
    task_id: str
    request: BatchPredictRequest
    images: List[Dict[str, Any]]
    status: TaskStatusEnum
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    processed_images: int = 0
    current_image: str | None = None
    results: list[PredictResult] = field(default_factory=list)
    total_inference_time: float = 0.0
    error: str | None = None
    estimated_time_remaining: float | None = None

    @property
    def progress(self) -> float:
        """Calculate progress percentage."""
        if not self.images:
            return 100.0
        return min(100.0, (self.processed_images / len(self.images)) * 100.0)

    def to_status(self) -> TaskStatus:
        """Convert to TaskStatus model."""
        return TaskStatus(
            task_id=self.task_id,
            status=self.status,
            created_at=self.created_at,
            started_at=self.started_at,
            completed_at=self.completed_at,
            total_images=len(self.images),
            processed_images=self.processed_images,
            current_image=self.current_image,
            progress=self.progress,
            estimated_time_remaining=self.estimated_time_remaining,
            error=self.error
        )


# In-memory task storage
TASKS: dict[str, BatchTask] = {}

# Background task queue
TASK_QUEUE = asyncio.Queue()


def get_device(device_str: str) -> torch.device:
    """Get the appropriate torch device based on string input.

    Args:
        device_str: Device string. Options: ['default', 'cpu', 'mps', 'cuda']

    Returns:
        The corresponding torch device.
    """
    if device_str == "default":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device_str)


@torch.no_grad()
def predict_image(
    predictor: RGBGaussianPredictor,
    image: np.ndarray,
    f_px: float,
    device: torch.device,
) -> Gaussians3D:
    """Predict Gaussians from an image.

    Args:
        predictor: The loaded model.
        image: The input image as a numpy array.
        f_px: The focal length in pixels.
        device: The device to run inference on.

    Returns:
        The generated 3D Gaussians.
    """
    internal_shape = (1536, 1536)

    LOGGER.info("Running preprocessing.")
    image_pt = torch.from_numpy(image.copy()).float().to(device).permute(2, 0, 1) / 255.0
    _, height, width = image_pt.shape
    disparity_factor = torch.tensor([f_px / width]).float().to(device)

    # Resize image for model input
    image_resized_pt = torch.nn.functional.interpolate(
        image_pt[None],
        size=(internal_shape[1], internal_shape[0]),
        mode="bilinear",
        align_corners=True,
    )

    # Predict Gaussians in the NDC space.
    LOGGER.info("Running inference.")
    gaussians_ndc = predictor(image_resized_pt, disparity_factor)

    LOGGER.info("Running postprocessing.")
    intrinsics = (
        torch.tensor(
            [
                [f_px, 0, width / 2, 0],
                [0, f_px, height / 2, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        .float()
        .to(device)
    )
    intrinsics_resized = intrinsics.clone()
    intrinsics_resized[0] *= internal_shape[0] / width
    intrinsics_resized[1] *= internal_shape[1] / height

    # Convert Gaussians to metrics space.
    gaussians = unproject_gaussians(
        gaussians_ndc, torch.eye(4).to(device), intrinsics_resized, internal_shape
    )

    return gaussians


async def process_batch_task(
    task: BatchTask,
    model_cache: ModelCache,
):
    """Process a batch prediction task in the background.

    Args:
        task: The batch task to process.
        model_cache: Model cache dependency.
    """
    try:
        # Update task status to running
        task.status = TaskStatusEnum.RUNNING
        task.started_at = datetime.now()

        # Get device
        device = get_device(task.request.device or "default")
        LOGGER.info(f"Batch task {task.task_id} using device: {device}")

        # Validate rendering option
        if task.request.render and device.type != "cuda":
            LOGGER.warning(f"Task {task.task_id}: Can only run rendering with gsplat on CUDA. Rendering is disabled.")
            task.request.render = False

        # Get model from cache
        checkpoint_path = Path(task.request.checkpoint_path) if task.request.checkpoint_path else None
        model = model_cache.get_model(checkpoint_path, str(device))

        # Process each image
        total_start_time = time.time()
        image_times = []

        for image_data in task.images:
            try:
                # Update current image
                task.current_image = image_data["filename"]
                image_start_time = time.time()

                # Process image
                img_pil = image_data["img_pil"]
                img = np.asarray(img_pil)
                
                # Convert to RGB if single channel
                if img.ndim < 3 or img.shape[2] == 1:
                    img = np.dstack((img, img, img))
                
                # Remove alpha channel if present
                img = img[:, :, :3]
                
                # Calculate focal length in pixels
                height, width = img.shape[:2]
                f_px = image_data["f_px"]
                image = img

                # Run inference
                gaussians = predict_image(model, image, f_px, device)

                # Generate unique filename
                unique_id = str(uuid.uuid4())[:8]
                filename = f"{unique_id}_{image_data['filename']}".replace(" ", "_").replace(".", "_")

                # Save PLY file
                ply_path = RESULTS_DIR / f"{filename}.ply"
                save_ply(gaussians, f_px, (height, width), ply_path)

                # Optional rendering
                render_video_path = None
                if task.request.render:
                    video_path = RESULTS_DIR / f"{filename}.mp4"
                    intrinsics = torch.tensor(
                        [
                            [f_px, 0, (width - 1) / 2.0, 0],
                            [0, f_px, (height - 1) / 2.0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1],
                        ],
                        device=device,
                        dtype=torch.float32,
                    )
                    metadata = SceneMetaData(intrinsics[0, 0].item(), (width, height), "linearRGB")
                    render_gaussians(gaussians, metadata, video_path)
                    render_video_path = f"/results/{video_path.name}"

                # Calculate image inference time
                image_time = time.time() - image_start_time
                image_times.append(image_time)

                # Create result
                result = PredictResult(
                    gaussians_ply_path=f"/results/{ply_path.name}",
                    render_video_path=render_video_path,
                    inference_time=image_time,
                    image_dimensions={"height": height, "width": width},
                    num_gaussians=gaussians.mean_vectors.shape[0],
                    filename=image_data["filename"]
                )

                # Add to results
                task.results.append(result)
                task.processed_images += 1

                # Update estimated time remaining
                if len(image_times) > 0:
                    avg_time_per_image = sum(image_times) / len(image_times)
                    remaining_images = len(task.images) - task.processed_images
                    task.estimated_time_remaining = avg_time_per_image * remaining_images

            except Exception as e:
                LOGGER.exception(f"Task {task.task_id}: Error processing image {image_data['filename']}: {str(e)}")
                # Continue processing other images
                task.processed_images += 1
                continue

        # Calculate total inference time
        task.total_inference_time = time.time() - total_start_time

        # Update task status to completed
        task.status = TaskStatusEnum.COMPLETED
        task.completed_at = datetime.now()
        task.current_image = None
        task.estimated_time_remaining = 0.0

        LOGGER.info(f"Batch task {task.task_id} completed successfully")

    except Exception as e:
        LOGGER.exception(f"Batch task {task.task_id} failed: {str(e)}")
        # Update task status to failed
        task.status = TaskStatusEnum.FAILED
        task.completed_at = datetime.now()
        task.error = str(e)


@router.post("/predict", response_model=PredictResponse)
async def predict(
    file: UploadFile = File(...),
    request: PredictRequest = Depends(),
    model_cache: ModelCache = Depends(get_model_cache),
):
    """Predict 3D Gaussians from an input image.

    Args:
        file: The input image file.
        request: Prediction request parameters.
        model_cache: Model cache dependency.

    Returns:
        The prediction response.
    """
    try:
        # Start inference timer
        start_time = time.time()

        # Validate file type
        content_type = file.content_type
        filename = file.filename or ""
        
        # Check if content_type is available and starts with 'image/'
        is_image = False
        if content_type and content_type.startswith("image/"):
            is_image = True
        # If content_type is not available, check file extension
        elif filename.lower().endswith((".jpg", ".jpeg", ".png", ".gif", ".bmp")):
            is_image = True
        
        if not is_image:
            return PredictResponse(
                success=False,
                message="Invalid file type. Please upload an image file.",
                error="Invalid file type"
            )

        # Read and validate image
        image_bytes = await file.read()
        
        # Convert bytes to PIL Image
        from PIL import Image
        import io as io_module
        img_pil = Image.open(io_module.BytesIO(image_bytes))
        
        # Extract EXIF data
        img_exif = io.extract_exif(img_pil)
        
        # Rotate the image if needed
        auto_rotate = True
        if auto_rotate:
            exif_orientation = img_exif.get("Orientation", 1)
            if exif_orientation == 3:
                img_pil = img_pil.rotate(180)
            elif exif_orientation == 6:
                img_pil = img_pil.rotate(270)
            elif exif_orientation == 8:
                img_pil = img_pil.rotate(90)
        
        # Extract focal length
        f_35mm = img_exif.get("FocalLengthIn35mmFilm", img_exif.get("FocalLenIn35mmFilm", None))
        if f_35mm is None or f_35mm < 1:
            f_35mm = img_exif.get("FocalLength", None)
            if f_35mm is None:
                LOGGER.warn("Did not find focallength in exif data - Setting to 30mm.")
                f_35mm = 30.0
            if f_35mm < 10.0:
                LOGGER.info("Found focal length below 10mm, assuming it's not for 35mm.")
                f_35mm *= 8.4
        
        # Convert to numpy array
        img = np.asarray(img_pil)
        
        # Convert to RGB if single channel
        if img.ndim < 3 or img.shape[2] == 1:
            img = np.dstack((img, img, img))
        
        # Remove alpha channel if present
        img = img[:, :, :3]
        
        # Calculate focal length in pixels
        height, width = img.shape[:2]
        f_px = io.convert_focallength(width, height, f_35mm)
        image = img  # Rename to match original variable name

        # Get device
        device = get_device(request.device or "default")
        LOGGER.info(f"Using device: {device}")

        # Validate rendering option
        if request.render and device.type != "cuda":
            LOGGER.warning("Can only run rendering with gsplat on CUDA. Rendering is disabled.")
            request.render = False

        # Get model from cache
        checkpoint_path = Path(request.checkpoint_path) if request.checkpoint_path else None
        model = model_cache.get_model(checkpoint_path, str(device))

        # Run inference
        gaussians = predict_image(model, image, f_px, device)

        # Calculate inference time
        inference_time = time.time() - start_time

        # Generate unique filename
        unique_id = str(uuid.uuid4())[:8]
        filename = f"{unique_id}_{file.filename}".replace(" ", "_").replace(".", "_")

        # Save PLY file
        ply_path = RESULTS_DIR / f"{filename}.ply"
        save_ply(gaussians, f_px, (height, width), ply_path)

        # Optional rendering
        render_video_path = None
        if request.render:
            video_path = RESULTS_DIR / f"{filename}.mp4"
            intrinsics = torch.tensor(
                [
                    [f_px, 0, (width - 1) / 2.0, 0],
                    [0, f_px, (height - 1) / 2.0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ],
                device=device,
                dtype=torch.float32,
            )
            metadata = SceneMetaData(intrinsics[0, 0].item(), (width, height), "linearRGB")
            render_gaussians(gaussians, metadata, video_path)
            render_video_path = f"/results/{video_path.name}"

        # Prepare response
        result = PredictResult(
            gaussians_ply_path=f"/results/{ply_path.name}",
            render_video_path=render_video_path,
            inference_time=inference_time,
            image_dimensions={"height": height, "width": width},
            num_gaussians=gaussians.mean_vectors.shape[0],
            filename=file.filename or ""
        )

        return PredictResponse(
            success=True,
            message="Prediction completed successfully.",
            result=result
        )

    except Exception as e:
        LOGGER.exception(f"Prediction failed: {str(e)}")
        return PredictResponse(
            success=False,
            message="Prediction failed.",
            error=str(e)
        )


@router.post("/predict/batch", response_model=dict[str, str])
async def batch_predict(
    files: list[UploadFile] = File(...),
    request: BatchPredictRequest = Depends(),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    model_cache: ModelCache = Depends(get_model_cache),
):
    """Predict 3D Gaussians from multiple input images (batch processing).

    Args:
        files: List of input image files.
        request: Batch prediction request parameters.
        background_tasks: FastAPI background tasks dependency.
        model_cache: Model cache dependency.

    Returns:
        Task ID for tracking progress.
    """
    try:
        # Validate files
        if not files:
            return JSONResponse(
                status_code=400,
                content={"error": "No files uploaded"}
            )

        # Process uploaded files
        images = []
        from PIL import Image
        import io as io_module

        for file in files:
            # Validate file type
            content_type = file.content_type
            filename = file.filename or ""
            
            # Check if content_type is available and starts with 'image/'
            is_image = False
            if content_type and content_type.startswith("image/"):
                is_image = True
            # If content_type is not available, check file extension
            elif filename.lower().endswith((".jpg", ".jpeg", ".png", ".gif", ".bmp")):
                is_image = True
            
            if not is_image:
                return JSONResponse(
                    status_code=400,
                    content={"error": f"Invalid file type: {filename}"}
                )

            # Read and process image
            image_bytes = await file.read()
            img_pil = Image.open(io_module.BytesIO(image_bytes))
            
            # Extract EXIF data
            img_exif = io.extract_exif(img_pil)
            
            # Rotate the image if needed
            auto_rotate = True
            if auto_rotate:
                exif_orientation = img_exif.get("Orientation", 1)
                if exif_orientation == 3:
                    img_pil = img_pil.rotate(180)
                elif exif_orientation == 6:
                    img_pil = img_pil.rotate(270)
                elif exif_orientation == 8:
                    img_pil = img_pil.rotate(90)
            
            # Extract focal length
            f_35mm = img_exif.get("FocalLengthIn35mmFilm", img_exif.get("FocalLenIn35mmFilm", None))
            if f_35mm is None or f_35mm < 1:
                f_35mm = img_exif.get("FocalLength", None)
                if f_35mm is None:
                    LOGGER.warn(f"Did not find focallength in exif data for {filename} - Setting to 30mm.")
                    f_35mm = 30.0
                if f_35mm < 10.0:
                    LOGGER.info(f"Found focal length below 10mm for {filename}, assuming it's not for 35mm.")
                    f_35mm *= 8.4
            
            # Calculate focal length in pixels
            width, height = img_pil.size
            f_px = io.convert_focallength(width, height, f_35mm)

            # Add to images list
            images.append({
                "file": file,
                "filename": filename,
                "img_pil": img_pil,
                "f_px": f_px
            })

        # Create batch task
        task_id = str(uuid.uuid4())
        task = BatchTask(
            task_id=task_id,
            request=request,
            images=images,
            status=TaskStatusEnum.PENDING,
            created_at=datetime.now()
        )

        # Store task
        TASKS[task_id] = task

        # Add to background tasks
        background_tasks.add_task(process_batch_task, task, model_cache)

        # Return task ID
        return {"task_id": task_id}

    except Exception as e:
        LOGGER.exception(f"Batch prediction failed: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Batch prediction failed: {str(e)}"}
        )


@router.get("/predict/batch/{task_id}/status", response_model=TaskStatus)
async def get_batch_status(task_id: str):
    """Get the status of a batch prediction task.

    Args:
        task_id: The ID of the batch task.

    Returns:
        Task status information.
    """
    if task_id not in TASKS:
        raise HTTPException(status_code=404, detail="Task not found")

    task = TASKS[task_id]
    return task.to_status()


@router.get("/predict/batch/{task_id}/results", response_model=TaskResult)
async def get_batch_results(task_id: str):
    """Get the results of a batch prediction task.

    Args:
        task_id: The ID of the batch task.

    Returns:
        Task result information.
    """
    if task_id not in TASKS:
        raise HTTPException(status_code=404, detail="Task not found")

    task = TASKS[task_id]

    if task.status == TaskStatusEnum.COMPLETED:
        batch_result = BatchPredictResult(
            task_id=task.task_id,
            status=task.status,
            results=task.results,
            total_inference_time=task.total_inference_time
        )
        return TaskResult(
            task_id=task_id,
            status=task.status,
            result=batch_result
        )
    elif task.status == TaskStatusEnum.FAILED:
        return TaskResult(
            task_id=task_id,
            status=task.status,
            error=task.error
        )
    else:
        return TaskResult(
            task_id=task_id,
            status=task.status
        )
