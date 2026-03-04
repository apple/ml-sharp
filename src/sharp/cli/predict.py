"""Contains `sharp predict` CLI implementation.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

import logging
from pathlib import Path

import click
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data

from sharp.models import (
    PredictorParams,
    RGBGaussianPredictor,
    create_predictor,
)
from sharp.utils import io
from sharp.utils import logging as logging_utils
from sharp.utils.gaussians import (
    Gaussians3D,
    SceneMetaData,
    save_ply,
    unproject_gaussians,
)

from .render import render_gaussians

LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL_URL = "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"


@click.command()
@click.option(
    "-i",
    "--input-path",
    type=click.Path(path_type=Path, exists=True),
    help="Path to an image or containing a list of images.",
    required=True,
)
@click.option(
    "-o",
    "--output-path",
    type=click.Path(path_type=Path, file_okay=False),
    help="Path to save the predicted Gaussians and renderings.",
    required=True,
)
@click.option(
    "-c",
    "--checkpoint-path",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
    help="Path to the .pt checkpoint. If not provided, downloads the default model automatically.",
    required=False,
)
@click.option(
    "--render/--no-render",
    "with_rendering",
    is_flag=True,
    default=False,
    help="Whether to render trajectory for checkpoint.",
)
@click.option(
    "--device",
    type=str,
    default="default",
    help="Device to run on. ['cpu', 'mps', 'cuda']",
)
@click.option(
    "--precision",
    type=click.Choice(["float32", "bfloat16", "float16"]),
    default="float32",
    help="Inference precision. bfloat16/float16 reduce memory and may be faster.",
)
@click.option("-v", "--verbose", is_flag=True, help="Activate debug logs.")
def predict_cli(
    input_path: Path,
    output_path: Path,
    checkpoint_path: Path,
    with_rendering: bool,
    device: str,
    precision: str,
    verbose: bool,
):
    """Predict Gaussians from input images."""
    logging_utils.configure(logging.DEBUG if verbose else logging.INFO)

    extensions = io.get_supported_image_extensions()

    image_paths = []
    if input_path.is_file():
        if input_path.suffix in extensions:
            image_paths = [input_path]
    else:
        for ext in extensions:
            image_paths.extend(list(input_path.glob(f"**/*{ext}")))

    if len(image_paths) == 0:
        LOGGER.info("No valid images found. Input was %s.", input_path)
        return

    LOGGER.info("Processing %d valid image files.", len(image_paths))

    if device == "default":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    LOGGER.info("Using device %s", device)

    if with_rendering and device != "cuda":
        LOGGER.warning("Can only run rendering with gsplat on CUDA. Rendering is disabled.")
        with_rendering = False

    # Load or download checkpoint
    if checkpoint_path is None:
        LOGGER.info("No checkpoint provided. Downloading default model from %s", DEFAULT_MODEL_URL)
        state_dict = torch.hub.load_state_dict_from_url(DEFAULT_MODEL_URL, progress=True)
    else:
        LOGGER.info("Loading checkpoint from %s", checkpoint_path)
        state_dict = torch.load(checkpoint_path, weights_only=True)

    gaussian_predictor = create_predictor(PredictorParams())
    gaussian_predictor.load_state_dict(state_dict)
    gaussian_predictor.eval()
    gaussian_predictor.to(device)

    output_path.mkdir(exist_ok=True, parents=True)

    for image_path in image_paths:
        LOGGER.info("Processing %s", image_path)
        image, _, f_px = io.load_rgb(image_path)
        height, width = image.shape[:2]
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
        gaussians = predict_image(
            gaussian_predictor, image, f_px, torch.device(device), precision
        )

        LOGGER.info("Saving 3DGS to %s", output_path)
        save_ply(gaussians, f_px, (height, width), output_path / f"{image_path.stem}.ply")

        if with_rendering:
            output_video_path = (output_path / image_path.stem).with_suffix(".mp4")
            LOGGER.info("Rendering trajectory to %s", output_video_path)

            metadata = SceneMetaData(intrinsics[0, 0].item(), (width, height), "linearRGB")
            render_gaussians(gaussians, metadata, output_video_path)


@torch.no_grad()
def predict_image(
    predictor: RGBGaussianPredictor,
    image: np.ndarray,
    f_px: float,
    device: torch.device,
    precision: str = "float32",
) -> Gaussians3D:
    """Predict Gaussians from an image."""
    internal_shape = (1536, 1536)

    LOGGER.info("Running preprocessing.")
    image_pt = torch.from_numpy(image.copy()).float().to(device).permute(2, 0, 1) / 255.0
    _, height, width = image_pt.shape
    disparity_factor = torch.tensor([f_px / width]).float().to(device)

    image_resized_pt = F.interpolate(
        image_pt[None],
        size=(internal_shape[1], internal_shape[0]),
        mode="bilinear",
        align_corners=True,
    )

    # Predict Gaussians in the NDC space.
    LOGGER.info("Running inference.")

    # Selective precision casting: only cast heavy encoder/backbone modules to
    # the target dtype. Lightweight modules (init_model, prediction_head,
    # gaussian_composer) stay in float32 for numerical stability. Forward hooks
    # cast inputs on entry and outputs back to float32 on exit.
    use_autocast = precision != "float32"
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16}
    autocast_dtype = dtype_map.get(precision, torch.float32)

    cast_modules: list[torch.nn.Module] = []
    hooks: list[torch.utils.hooks.RemovableHandle] = []
    if use_autocast:

        def _cast_input_tensor(obj):
            if isinstance(obj, torch.Tensor) and obj.is_floating_point():
                return obj.to(autocast_dtype)
            if isinstance(obj, list):
                return [_cast_input_tensor(x) for x in obj]
            if isinstance(obj, tuple) and hasattr(obj, "_asdict"):
                fields = {k: _cast_input_tensor(v) for k, v in obj._asdict().items()}
                return type(obj)(**fields)
            if isinstance(obj, tuple):
                return tuple(_cast_input_tensor(x) for x in obj)
            return obj

        def _cast_inputs(_mod, args, kwargs):
            new_args = tuple(_cast_input_tensor(a) for a in args)
            new_kwargs = {k: _cast_input_tensor(v) for k, v in kwargs.items()}
            return new_args, new_kwargs

        def _cast_output_to_float(obj):
            if isinstance(obj, torch.Tensor) and obj.is_floating_point():
                return obj.float()
            if isinstance(obj, list):
                return [_cast_output_to_float(x) for x in obj]
            if isinstance(obj, dict):
                return {k: _cast_output_to_float(v) for k, v in obj.items()}
            if isinstance(obj, tuple) and hasattr(obj, "_asdict"):
                fields = {k: _cast_output_to_float(v) for k, v in obj._asdict().items()}
                return type(obj)(**fields)
            if isinstance(obj, tuple):
                return tuple(_cast_output_to_float(x) for x in obj)
            return obj

        def _cast_outputs_hook(_mod, _input, output):
            return _cast_output_to_float(output)

        for name in ("monodepth_model", "feature_model"):
            mod = getattr(predictor, name, None)
            if mod is not None:
                mod.to(autocast_dtype)
                cast_modules.append(mod)
                hooks.append(mod.register_forward_pre_hook(_cast_inputs, with_kwargs=True))
                hooks.append(mod.register_forward_hook(_cast_outputs_hook))

    try:
        gaussians_ndc = predictor(image_resized_pt, disparity_factor)
    finally:
        for hook in hooks:
            hook.remove()
        for mod in cast_modules:
            mod.float()

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
