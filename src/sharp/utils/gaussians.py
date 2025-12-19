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

    xyz = gaussians.mean_vectors.flatten(0, 1)
    scale_logits = torch.log(gaussians.singular_values).flatten(0, 1)
    quaternions = gaussians.quaternions.flatten(0, 1)

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
    colors = convert_rgb_to_spherical_harmonics(
        cs_utils.linearRGB2sRGB(gaussians.colors.flatten(0, 1))
    )
    color_space_index = cs_utils.encode_color_space("sRGB")

    # Store opacity logits.
    opacity_logits = _inverse_sigmoid(gaussians.opacities).flatten(0, 1).unsqueeze(-1)

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
    elements = np.empty(num_gaussians, dtype=dtype_full)
    elements[:] = list(map(tuple, attributes.detach().cpu().numpy()))
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
        ]
    )

    plydata.write(path)
    return plydata


@torch.no_grad()
def save_splat(
    gaussians: Gaussians3D, f_px: float, image_shape: tuple[int, int], path: Path
) -> None:
    """Save Gaussians to .splat format (compact binary format for web viewers).

    The .splat format is a simple binary format used by web-based 3DGS viewers.
    Each Gaussian is stored as 32 bytes:
    - 12 bytes: xyz position (3 x float32)
    - 12 bytes: scales (3 x float32)
    - 4 bytes: RGBA color (4 x uint8)
    - 4 bytes: quaternion rotation (4 x uint8, encoded as (q * 128 + 128))

    Gaussians are sorted by size * opacity (descending) for progressive rendering.
    """
    xyz = gaussians.mean_vectors.flatten(0, 1).cpu().numpy()
    scales = gaussians.singular_values.flatten(0, 1).cpu().numpy()
    quats = gaussians.quaternions.flatten(0, 1).cpu().numpy()
    colors_rgb = cs_utils.linearRGB2sRGB(gaussians.colors.flatten(0, 1)).cpu().numpy()
    opacities = gaussians.opacities.flatten(0, 1).cpu().numpy()

    # Sort by size * opacity (descending) for progressive rendering
    sort_idx = np.argsort(-(scales.prod(axis=1) * opacities))

    # Normalize quaternions
    quats = quats / np.linalg.norm(quats, axis=1, keepdims=True)

    with open(path, "wb") as f:
        for i in sort_idx:
            f.write(xyz[i].astype(np.float32).tobytes())
            f.write(scales[i].astype(np.float32).tobytes())
            rgba = np.concatenate([colors_rgb[i], [opacities[i]]])
            f.write((rgba * 255).clip(0, 255).astype(np.uint8).tobytes())
            f.write((quats[i] * 128 + 128).clip(0, 255).astype(np.uint8).tobytes())


@torch.no_grad()
def save_sog(
    gaussians: Gaussians3D, f_px: float, image_shape: tuple[int, int], path: Path
) -> None:
    """Save Gaussians to SOG format (Spatially Ordered Gaussians).

    SOG is a highly compressed format using quantization and WebP images.
    Typically 15-20x smaller than PLY. The format stores data in a ZIP archive
    containing WebP images for positions, rotations, scales, and colors.

    Reference: https://github.com/aras-p/sog-format
    """
    import io
    import json
    import math
    import zipfile

    from PIL import Image

    xyz = gaussians.mean_vectors.flatten(0, 1).cpu().numpy()
    scales = gaussians.singular_values.flatten(0, 1).cpu().numpy()
    quats = gaussians.quaternions.flatten(0, 1).cpu().numpy()
    colors_linear = gaussians.colors.flatten(0, 1).cpu().numpy()
    opacities = gaussians.opacities.flatten(0, 1).cpu().numpy()

    num_gaussians = len(xyz)

    # Compute image dimensions (roughly square)
    img_width = int(math.ceil(math.sqrt(num_gaussians)))
    img_height = int(math.ceil(num_gaussians / img_width))
    total_pixels = img_width * img_height

    # Pad arrays to fill image
    def pad_array(arr: np.ndarray, total: int) -> np.ndarray:
        if len(arr) < total:
            pad_shape = (total - len(arr),) + arr.shape[1:]
            return np.concatenate([arr, np.zeros(pad_shape, dtype=arr.dtype)])
        return arr

    xyz = pad_array(xyz, total_pixels)
    scales = pad_array(scales, total_pixels)
    quats = pad_array(quats, total_pixels)
    colors_linear = pad_array(colors_linear, total_pixels)
    opacities = pad_array(opacities, total_pixels)

    # Normalize quaternions
    quats = quats / (np.linalg.norm(quats, axis=1, keepdims=True) + 1e-8)

    # === 1. Encode positions (16-bit per axis with symmetric log transform) ===
    def symlog(x: np.ndarray) -> np.ndarray:
        return np.sign(x) * np.log1p(np.abs(x))

    xyz_log = symlog(xyz)
    mins = xyz_log.min(axis=0)
    maxs = xyz_log.max(axis=0)

    # Avoid division by zero
    ranges = maxs - mins
    ranges = np.where(ranges < 1e-8, 1.0, ranges)

    # Quantize to 16-bit
    xyz_norm = (xyz_log - mins) / ranges
    xyz_q16 = (xyz_norm * 65535).clip(0, 65535).astype(np.uint16)

    means_l = (xyz_q16 & 0xFF).astype(np.uint8)
    means_u = (xyz_q16 >> 8).astype(np.uint8)

    # === 2. Encode quaternions (smallest-three, 26-bit) ===
    def encode_quaternion(q: np.ndarray) -> np.ndarray:
        """Encode quaternion using smallest-three method."""
        # Find largest component
        abs_q = np.abs(q)
        mode = np.argmax(abs_q, axis=1)

        # Ensure the largest component is positive
        signs = np.sign(q[np.arange(len(q)), mode])
        q = q * signs[:, None]

        # Extract the three smallest components
        result = np.zeros((len(q), 4), dtype=np.uint8)
        sqrt2_inv = 1.0 / math.sqrt(2)

        for i in range(len(q)):
            m = mode[i]
            # Get indices of the three kept components
            kept = [j for j in range(4) if j != m]
            vals = q[i, kept]
            # Quantize from [-sqrt2/2, sqrt2/2] to [0, 255]
            encoded = ((vals * sqrt2_inv + 0.5) * 255).clip(0, 255).astype(np.uint8)
            result[i, :3] = encoded
            result[i, 3] = 252 + m  # Mode in alpha channel

        return result

    quats_encoded = encode_quaternion(quats)

    # === 3. Build scale codebook (256 entries) ===
    # SOG stores scales in LOG space - the renderer does exp(codebook[idx])
    scales_log = np.log(np.maximum(scales, 1e-10))
    scales_log_flat = scales_log.flatten()

    # Use percentiles for codebook (in log space)
    percentiles = np.linspace(0, 100, 256)
    scale_codebook = np.percentile(scales_log_flat, percentiles).astype(np.float32)

    # Quantize values to nearest codebook entry
    def quantize_to_codebook(values: np.ndarray, codebook: np.ndarray) -> np.ndarray:
        indices = np.searchsorted(codebook, values)
        indices = np.clip(indices, 0, len(codebook) - 1)
        # Check if previous index is closer
        prev_indices = np.clip(indices - 1, 0, len(codebook) - 1)
        dist_curr = np.abs(values - codebook[indices])
        dist_prev = np.abs(values - codebook[prev_indices])
        use_prev = (dist_prev < dist_curr) & (indices > 0)
        indices = np.where(use_prev, prev_indices, indices)
        return indices.astype(np.uint8)

    scales_q = np.stack(
        [
            quantize_to_codebook(scales_log[:, 0], scale_codebook),
            quantize_to_codebook(scales_log[:, 1], scale_codebook),
            quantize_to_codebook(scales_log[:, 2], scale_codebook),
        ],
        axis=1,
    )

    # === 4. Build SH0 codebook and encode colors ===
    SH_C0 = 0.28209479177387814
    sh0_coeffs = (colors_linear - 0.5) / SH_C0
    sh0_flat = sh0_coeffs.flatten()

    sh0_percentiles = np.linspace(0, 100, 256)
    sh0_codebook = np.percentile(sh0_flat, sh0_percentiles).astype(np.float32)

    sh0_r = quantize_to_codebook(sh0_coeffs[:, 0], sh0_codebook)
    sh0_g = quantize_to_codebook(sh0_coeffs[:, 1], sh0_codebook)
    sh0_b = quantize_to_codebook(sh0_coeffs[:, 2], sh0_codebook)
    sh0_a = (opacities * 255).clip(0, 255).astype(np.uint8)

    # === 5. Create images ===
    def create_image(data: np.ndarray, width: int, height: int) -> Image.Image:
        data = data.reshape(height, width, -1)
        if data.shape[2] == 3:
            return Image.fromarray(data, mode="RGB")
        elif data.shape[2] == 4:
            return Image.fromarray(data, mode="RGBA")
        else:
            raise ValueError(f"Unexpected channel count: {data.shape[2]}")

    means_l_img = create_image(means_l, img_width, img_height)
    means_u_img = create_image(means_u, img_width, img_height)
    quats_img = create_image(quats_encoded, img_width, img_height)
    scales_img = create_image(scales_q, img_width, img_height)

    sh0_data = np.stack([sh0_r, sh0_g, sh0_b, sh0_a], axis=1)
    sh0_img = create_image(sh0_data, img_width, img_height)

    # === 6. Create meta.json ===
    meta = {
        "version": 2,
        "count": num_gaussians,
        "antialias": False,
        "means": {
            "mins": mins.tolist(),
            "maxs": maxs.tolist(),
            "files": ["means_l.webp", "means_u.webp"],
        },
        "scales": {"codebook": scale_codebook.tolist(), "files": ["scales.webp"]},
        "quats": {"files": ["quats.webp"]},
        "sh0": {"codebook": sh0_codebook.tolist(), "files": ["sh0.webp"]},
    }

    # === 7. Save as ZIP archive ===
    path = Path(path)
    if path.suffix.lower() != ".sog":
        path = path.with_suffix(".sog")

    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        # Save images as lossless WebP
        for name, img in [
            ("means_l.webp", means_l_img),
            ("means_u.webp", means_u_img),
            ("quats.webp", quats_img),
            ("scales.webp", scales_img),
            ("sh0.webp", sh0_img),
        ]:
            buf = io.BytesIO()
            img.save(buf, format="WEBP", lossless=True)
            zf.writestr(name, buf.getvalue())

        # Save meta.json
        zf.writestr("meta.json", json.dumps(meta, indent=2))
