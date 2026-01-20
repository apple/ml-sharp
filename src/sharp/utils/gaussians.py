"""Contains basic data structures and functionality for 3D Gaussians.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal, NamedTuple

import numpy as np
import torch
from plyfile import PlyData, PlyElement

from sharp.utils import color_space as cs_utils
from sharp.utils import linalg

LOGGER = logging.getLogger(__name__)


BackgroundColor = Literal["black", "white", "random_color", "random_pixel"]


class Gaussians3D(NamedTuple):
    """Represents a collection of 3D Gaussians."""

    mean_vectors: torch.Tensor
    singular_values: torch.Tensor
    quaternions: torch.Tensor
    colors: torch.Tensor
    opacities: torch.Tensor

    def to(self, device: torch.device) -> Gaussians3D:
        """Move Gaussians to device."""
        return Gaussians3D(
            mean_vectors=self.mean_vectors.to(device),
            singular_values=self.singular_values.to(device),
            quaternions=self.quaternions.to(device),
            colors=self.colors.to(device),
            opacities=self.opacities.to(device),
        )


class SceneMetaData(NamedTuple):
    """Meta data about Gaussian scene."""

    focal_length_px: float
    resolution_px: tuple[int, int]
    color_space: cs_utils.ColorSpace


def get_unprojection_matrix(
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    image_shape: tuple[int, int],
) -> torch.Tensor:
    """Compute unprojection matrix to transform Gaussians to Euclidean space.

    Args:
        extrinsics: The 4x4 extrinsics matrix of the camera view.
        intrinsics: The 4x4 intrinsics matrix of the camera view.
        image_shape: The (width, height) of the input image.

    Returns:
        A 4x4 matrix to transform Gaussians from NDC space to Euclidean space.
    """
    device = intrinsics.device
    image_width, image_height = image_shape
    # This matrix converts OpenCV pixel coordinates to NDC coordinates where
    # (-1, 1) denotes the top left and (1, 1) the bottom right of the image.
    #
    # Note that premultiplying the intrinsics with ndc_matrix typically yields a matrix
    # that simply scales the x-axis by 2 * focal_length / image_width and the y-axis by
    # 2 * focal_length / image_height.
    ndc_matrix = torch.tensor(
        [
            [2.0 / image_width, 0.0, -1.0, 0.0],
            [0.0, 2.0 / image_height, -1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        device=device,
    )
    return torch.linalg.inv(ndc_matrix @ intrinsics @ extrinsics)


def unproject_gaussians(
    gaussians_ndc: Gaussians3D,
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    image_shape: tuple[int, int],
) -> Gaussians3D:
    """Unproject Gaussians from NDC space to world coordinates."""
    unprojection_matrix = get_unprojection_matrix(extrinsics, intrinsics, image_shape)
    gaussians = apply_transform(gaussians_ndc, unprojection_matrix[:3])
    return gaussians


def apply_transform(gaussians: Gaussians3D, transform: torch.Tensor) -> Gaussians3D:
    """Apply an affine transformation to 3D Gaussians.

    Args:
        gaussians: The Gaussians to transform.
        transform: An affine transform with shape 3x4.

    Returns:
        The transformed Gaussians.

    Note: This operation is not differentiable.
    """
    transform_linear = transform[..., :3, :3]
    transform_offset = transform[..., :3, 3]

    mean_vectors = gaussians.mean_vectors @ transform_linear.T + transform_offset
    covariance_matrices = compose_covariance_matrices(
        gaussians.quaternions, gaussians.singular_values
    )
    covariance_matrices = (
        transform_linear @ covariance_matrices @ transform_linear.transpose(-1, -2)
    )
    quaternions, singular_values = decompose_covariance_matrices(covariance_matrices)

    return Gaussians3D(
        mean_vectors=mean_vectors,
        singular_values=singular_values,
        quaternions=quaternions,
        colors=gaussians.colors,
        opacities=gaussians.opacities,
    )


def decompose_covariance_matrices(
    covariance_matrices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decompose 3D covariance matrices into quaternions and singular values.

    Args:
        covariance_matrices: The covariance matrices to decompose.

    Returns:
        Quaternion and singular values corresponding to the orientation and scales of
        the diagonalized matrix.

    Note: This operation is not differentiable.
    """
    device = covariance_matrices.device
    dtype = covariance_matrices.dtype

    # We convert to fp64 to avoid numerical errors.
    covariance_matrices = covariance_matrices.detach().cpu().to(torch.float64)
    rotations, singular_values_2, _ = torch.linalg.svd(covariance_matrices)

    # NOTE: in SVD, it is possible that U and VT are both reflections.
    # We need to correct them.
    batch_idx, gaussian_idx = torch.where(torch.linalg.det(rotations) < 0)
    num_reflections = len(gaussian_idx)
    if num_reflections > 0:
        LOGGER.warning(
            "Received %d reflection matrices from SVD. Flipping them to rotations.",
            num_reflections,
        )
        # Flip the last column of reflection and make it a rotation.
        rotations[batch_idx, gaussian_idx, :, -1] *= -1
    quaternions = linalg.quaternions_from_rotation_matrices(rotations)
    quaternions = quaternions.to(dtype=dtype, device=device)
    singular_values = singular_values_2.sqrt().to(dtype=dtype, device=device)
    return quaternions, singular_values


def compose_covariance_matrices(
    quaternions: torch.Tensor, singular_values: torch.Tensor
) -> torch.Tensor:
    """Compose 3D covariance matrices into quaternions and singular values.

    Args:
        quaternions: The quaternions describing the principal basis.
        singular_values: The scales of the diagonalized matrix.

    Returns:
        The 3x3 covariances matrices.
    """
    device = quaternions.device
    rotations = linalg.rotation_matrices_from_quaternions(quaternions)
    diagonal_matrix = torch.eye(3, device=device) * singular_values[..., :, None]
    return rotations @ diagonal_matrix.square() @ rotations.transpose(-1, -2)


def convert_spherical_harmonics_to_rgb(sh0: torch.Tensor) -> torch.Tensor:
    """Convert degree-0 spherical harmonics to RGB.

    Reference:
        https://en.wikipedia.org/wiki/Table_of_spherical_harmonics
    """
    coeff_degree0 = np.sqrt(1.0 / (4.0 * np.pi))
    return sh0 * coeff_degree0 + 0.5


def convert_rgb_to_spherical_harmonics(rgb: torch.Tensor) -> torch.Tensor:
    """Convert RGB to degree-0 spherical harmonics.

    Reference:
        https://en.wikipedia.org/wiki/Table_of_spherical_harmonics
    """
    coeff_degree0 = np.sqrt(1.0 / (4.0 * np.pi))
    return (rgb - 0.5) / coeff_degree0


def load_ply(path: Path) -> tuple[Gaussians3D, SceneMetaData]:
    """Loads a ply from a file."""
    plydata = PlyData.read(path)

    vertices = next(filter(lambda x: x.name == "vertex", plydata.elements))

    properties = ["x", "y", "z"]
    properties.extend([f"f_dc_{i}" for i in range(3)])
    properties.extend([f"scale_{i}" for i in range(3)])
    properties.extend([f"rot_{i}" for i in range(3)])

    for prop in properties:
        if prop not in vertices:
            raise KeyError(f"Incompatible ply file: property {prop} not found in ply elements.")
    mean_vectors = np.stack(
        (
            np.asarray(vertices["x"]),
            np.asarray(vertices["y"]),
            np.asarray(vertices["z"]),
        ),
        axis=1,
    )

    scale_logits = np.stack(
        (
            np.asarray(vertices["scale_0"]),
            np.asarray(vertices["scale_1"]),
            np.asarray(vertices["scale_2"]),
        ),
        axis=1,
    )

    quaternions = np.stack(
        (
            np.asarray(vertices["rot_0"]),
            np.asarray(vertices["rot_1"]),
            np.asarray(vertices["rot_2"]),
            np.asarray(vertices["rot_3"]),
        ),
        axis=1,
    )

    spherical_harmonics_deg0 = np.stack(
        (
            np.asarray(vertices["f_dc_0"]),
            np.asarray(vertices["f_dc_1"]),
            np.asarray(vertices["f_dc_2"]),
        ),
        axis=1,
    )

    colors = convert_spherical_harmonics_to_rgb(spherical_harmonics_deg0)

    opacity_logits = np.asarray(vertices["opacity"])[..., None]

    supplement_elements = [element for element in plydata.elements if element.name != "vertex"]
    supplement_data: dict[str, Any] = {}
    supplement_keys = ["extrinsic", "intrinsic", "color_space", "image_size"]

    for element in supplement_elements:
        for key in supplement_keys:
            if key not in supplement_data and key in element:
                supplement_data[key] = np.asarray(element[key])

    # Parse intrinsics and image_size.
    if "intrinsic" in supplement_data:
        intrinsics_data = supplement_data["intrinsic"]

        # Legacy: image_size is contained in intrinsic element.
        if "image_size" not in supplement_data:
            if len(intrinsics_data) != 4:
                raise ValueError(
                    "Expect legacy intrinsics with len=4 containing image size, "
                    f"but received len={len(intrinsics_data)}"
                )
            focal_length_px = (intrinsics_data[0], intrinsics_data[1])
            width = int(intrinsics_data[2])
            height = int(intrinsics_data[3])

        else:
            if len(intrinsics_data) != 9:
                raise ValueError(
                    "Expect 9 elements in intrinsics, " f"but received {len(intrinsics_data)}."
                )
            intrinsics_matrix = intrinsics_data.reshape((3, 3))
            focal_length_px = (intrinsics_matrix[0, 0], intrinsics_matrix[1, 1])

            image_size_data = supplement_data["image_size"]
            width = image_size_data[0]
            height = image_size_data[1]

    # Default to VGA resolution: focal length = 512, image size = (640, 480).
    else:
        focal_length_px = (512, 512)
        width = 640
        height = 480

    # Parse extrinsics.
    extrinsics_data = supplement_data.get("extrinsic", np.eye(4).flatten())
    extrinsics_matrix = np.eye(4)

    # Legacy: extrinsics store 12 elements.
    if len(extrinsics_data) == 12:
        extrinsics_matrix[:3] = extrinsics_data.reshape((3, 4))
        extrinsics_matrix[:3, :3] = extrinsics_matrix[:3, :3].copy().T
    elif len(extrinsics_data) == 16:
        extrinsics_matrix[:] = extrinsics_data.reshape((4, 4))
    else:
        raise ValueError(f"Unrecognized extrinsics matrix shape {len(extrinsics_data)}")

    # Parse color space.
    color_space_index = supplement_data.get("color_space", 1)
    color_space = cs_utils.decode_color_space(color_space_index)
    colors = torch.from_numpy(colors).view(1, -1, 3).float()

    if color_space == "sRGB":
        # Convert to linearRGB for proper alpha blending.
        colors = cs_utils.sRGB2linearRGB(colors.flatten(0, 1)).view(1, -1, 3)
        color_space = "linearRGB"

    mean_vectors = torch.from_numpy(mean_vectors).view(1, -1, 3).float()
    quaternions = torch.from_numpy(quaternions).view(1, -1, 4).float()
    singular_values = torch.exp(torch.from_numpy(scale_logits).view(1, -1, 3)).float()
    opacities = torch.sigmoid(torch.from_numpy(opacity_logits).view(1, -1)).float()

    gaussians = Gaussians3D(
        mean_vectors=mean_vectors,
        quaternions=quaternions,
        singular_values=singular_values,
        opacities=opacities,
        colors=colors,
    )
    metadata = SceneMetaData(focal_length_px[0], (width, height), color_space)
    return gaussians, metadata


@torch.no_grad()
def save_ply(
    gaussians: Gaussians3D, f_px: float, image_shape: tuple[int, int], path: Path
) -> PlyData:
    """Save a predicted Gaussian3D to a ply file."""

    def _inverse_sigmoid(tensor: torch.Tensor) -> torch.Tensor:
        return torch.log(tensor / (1.0 - tensor))

    xyz = gaussians.mean_vectors.flatten(0, 1).detach().cpu()
    scale_logits = torch.log(gaussians.singular_values).flatten(0, 1).detach().cpu()
    quaternions = gaussians.quaternions.flatten(0, 1).detach().cpu()
    opacity_logits = _inverse_sigmoid(gaussians.opacities).flatten(0, 1).unsqueeze(-1).detach().cpu()

    # SHARP takes an image, convert it to sRGB color space as input,
    # and predicts linearRGB Gaussians as output.
    # The SHARP renderer would blend linearRGB Gaussians and convert rendered images and videos
    # back to sRGB for the best display quality.
    #
    # However, public renderers do not have such linear2sRGB conversions after rendering.
    # If they render linearRGB Gaussians as-is, the output would be dark without Gamma correction.
    #
    # To make it compatible to public renderers, we force convert linearRGB to sRGB during export.
    # - The SHARP renderer will still handle conversions properly.
    # - Public renderers will be mostly working fine when regarding sRGB images as linearRGB images,
    #   although for the best performance, it is recommended to apply the conversions.
    colors_linear = gaussians.colors.flatten(0, 1).detach().cpu()
    colors = convert_rgb_to_spherical_harmonics(cs_utils.linearRGB2sRGB(colors_linear))
    color_space_index = cs_utils.encode_color_space("sRGB")

    attributes = torch.cat(
        (
            xyz,
            colors,
            opacity_logits,
            scale_logits,
            quaternions,
        ),
        dim=1,
    )

    dtype_full = [
        (attribute, "f4")
        for attribute in ["x", "y", "z"]
        + [f"f_dc_{i}" for i in range(3)]
        + ["opacity"]
        + [f"scale_{i}" for i in range(3)]
        + [f"rot_{i}" for i in range(4)]
    ]

    num_gaussians = len(xyz)
    attributes_np = attributes.numpy()
    elements = np.core.records.fromarrays(attributes_np.T, dtype=dtype_full)
    vertex_elements = PlyElement.describe(elements, "vertex")

    # Load image-wise metadata.
    image_height, image_width = image_shape

    # Export image size.
    dtype_image_size = [("image_size", "u4")]
    image_size_array = np.empty(2, dtype=dtype_image_size)
    image_size_array[:] = np.array([image_width, image_height])
    image_size_element = PlyElement.describe(image_size_array, "image_size")

    # Export intrinsics.
    dtype_intrinsic = [("intrinsic", "f4")]
    intrinsic_array = np.empty(9, dtype=dtype_intrinsic)
    intrinsic = np.array(
        [
            f_px,
            0,
            image_width * 0.5,
            0,
            f_px,
            image_height * 0.5,
            0,
            0,
            1,
        ]
    )
    intrinsic_array[:] = intrinsic.flatten()
    intrinsic_element = PlyElement.describe(intrinsic_array, "intrinsic")

    # Export dummy extrinsics.
    dtype_extrinsic = [("extrinsic", "f4")]
    extrinsic_array = np.empty(16, dtype=dtype_extrinsic)
    extrinsic_array[:] = np.eye(4).flatten()
    extrinsic_element = PlyElement.describe(extrinsic_array, "extrinsic")

    # Export number of frames and particles per frame.
    dtype_frames = [("frame", "i4")]
    frame_array = np.empty(2, dtype=dtype_frames)
    frame_array[:] = np.array([1, num_gaussians], dtype=np.int32)
    frame_element = PlyElement.describe(frame_array, "frame")

    # Export disparity ranges for transform.
    dtype_disparity = [("disparity", "f4")]
    disparity_array = np.empty(2, dtype=dtype_disparity)

    disparity = 1.0 / gaussians.mean_vectors[0, ..., -1]
    quantiles = (
        torch.quantile(disparity, q=torch.tensor([0.1, 0.9], device=disparity.device))
        .float()
        .cpu()
        .numpy()
    )
    disparity_array[:] = quantiles
    disparity_element = PlyElement.describe(disparity_array, "disparity")

    # Export colorspace.
    dtype_color_space = [("color_space", "u1")]
    color_space_array = np.empty(1, dtype=dtype_color_space)
    color_space_array[:] = np.array([color_space_index]).flatten()
    color_space_element = PlyElement.describe(color_space_array, "color_space")

    dtype_version = [("version", "u1")]
    version_array = np.empty(3, dtype=dtype_version)
    version_array[:] = np.array([1, 5, 0], dtype=np.uint8).flatten()
    version_element = PlyElement.describe(version_array, "version")

    plydata = PlyData(
        [
            vertex_elements,
            extrinsic_element,
            intrinsic_element,
            image_size_element,
            frame_element,
            disparity_element,
            color_space_element,
            version_element,
        ],
        byte_order='<'
    )

    with open(path, 'wb') as f:
        plydata.write(f)
    return plydata
