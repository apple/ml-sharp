"""Contains prediction API endpoint implementation.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from fastapi import APIRouter, Depends, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from sharp.models import RGBGaussianPredictor
from sharp.utils import io
from sharp.utils.gaussians import Gaussians3D, SceneMetaData, save_ply, unproject_gaussians

from ..dependencies import ModelCache, get_model_cache
from ..schemas import PredictRequest, PredictResponse, PredictResult
from .render import render_gaussians

LOGGER = logging.getLogger(__name__)
router = APIRouter()

# Create directory for results
# RESULTS_DIR = Path("/tmp/ml-sharp/results")
RESULTS_DIR = Path(__file__).parent.parent.parent / "tmp" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


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
        import uuid
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
