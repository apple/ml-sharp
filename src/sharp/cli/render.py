"""Contains `sharp render` CLI implementation.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

import logging
from pathlib import Path

import click
import torch
import torch.utils.data

from sharp.utils import camera, gsplat, io
from sharp.utils import logging as logging_utils
from sharp.utils.gaussians import Gaussians3D, SceneMetaData, load_ply

LOGGER = logging.getLogger(__name__)


@click.command()
@click.option(
    "-i",
    "--input-path",
    type=click.Path(exists=True, path_type=Path),
    help="Path to the ply or a list of plys.",
    required=True,
)
@click.option(
    "-o",
    "--output-path",
    type=click.Path(path_type=Path, file_okay=False),
    help="Path to save the rendered videos.",
    required=True,
)
@click.option("-v", "--verbose", is_flag=True, help="Activate debug logs.")
def render_cli(input_path: Path, output_path: Path, verbose: bool):
    """Predict Gaussians from input images."""
    logging_utils.configure(logging.DEBUG if verbose else logging.INFO)

    if not torch.cuda.is_available():
        LOGGER.error("Rendering a checkpoint requires CUDA.")
        exit(1)

    output_path.mkdir(exist_ok=True, parents=True)

    params = camera.TrajectoryParams()

    if input_path.suffix == ".ply":
        scene_paths = [input_path]
    elif input_path.is_dir():
        scene_paths = list(input_path.glob("*.ply"))
    else:
        LOGGER.error("Input path must be either directory or single PLY file.")
        exit(1)

    for scene_path in scene_paths:
        LOGGER.info("Rendering %s", scene_path)
        gaussians, metadata = load_ply(scene_path)
        render_gaussians(
            gaussians=gaussians,
            metadata=metadata,
            params=params,
            output_path=(output_path / scene_path.stem).with_suffix(".mp4"),
        )


def render_gaussians(
    gaussians: Gaussians3D,
    metadata: SceneMetaData,
    output_path: Path,
    params: camera.TrajectoryParams | None = None,
) -> None:
    """Render a single gaussian checkpoint file."""
    (width, height) = metadata.resolution_px
    f_px = metadata.focal_length_px

    if params is None:
        params = camera.TrajectoryParams()

    if not torch.cuda.is_available():
        raise RuntimeError("Rendering a checkpoint requires CUDA.")

    device = torch.device("cuda")

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
    camera_model = camera.create_camera_model(
        gaussians, intrinsics, resolution_px=metadata.resolution_px
    )

    trajectory = camera.create_eye_trajectory(
        gaussians, params, resolution_px=metadata.resolution_px, f_px=f_px
    )
    renderer = gsplat.GSplatRenderer(color_space=metadata.color_space)
    video_writer = io.VideoWriter(output_path)

    for _, eye_position in enumerate(trajectory):
        camera_info = camera_model.compute(eye_position)
        rendering_output = renderer(
            gaussians.to(device),
            extrinsics=camera_info.extrinsics[None].to(device),
            intrinsics=camera_info.intrinsics[None].to(device),
            image_width=camera_info.width,
            image_height=camera_info.height,
        )
        color = (rendering_output.color[0].permute(1, 2, 0) * 255.0).to(dtype=torch.uint8)
        depth = rendering_output.depth[0]
        video_writer.add_frame(color, depth)
    video_writer.close()
