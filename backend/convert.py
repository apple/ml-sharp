"""Gaussian to .splat conversion utilities.

Provides both:
- gaussians_to_splat(): Direct in-memory conversion from Gaussians3D tensors
  to .splat bytes. This is the fast path -- no PLY round-trip, no disk I/O,
  pure vectorized numpy. Use this for all pipelines.
- process_ply_to_splat(): Legacy PLY-file-based conversion (kept as fallback).
"""

import numpy as np
import torch
from pathlib import Path


def _linearRGB_to_sRGB_numpy(linear: np.ndarray) -> np.ndarray:
    """Convert linearRGB to sRGB using the standard transfer function.

    Numpy equivalent of sharp.utils.color_space.linearRGB2sRGB.
    """
    THRESHOLD = 0.0031308
    srgb = np.where(
        linear <= THRESHOLD,
        linear * 12.92,
        1.055 * np.power(np.clip(linear, THRESHOLD, None), 1.0 / 2.4) - 0.055,
    )
    return srgb


def gaussians_to_splat(gaussians, sort: bool = True) -> bytes:
    """Convert Gaussians3D directly to .splat binary format in memory.

    This bypasses the PLY round-trip entirely, which is the critical
    optimization: save_ply() creates 1 Python tuple per gaussian (fatal at
    28M+ gaussians), then PlyData serializes/deserializes the whole thing.
    This function does everything with vectorized numpy in seconds.

    Args:
        gaussians: Gaussians3D namedtuple with tensors:
            - mean_vectors: (1, N, 3) positions
            - singular_values: (1, N, 3) scales (already exp'd)
            - quaternions: (1, N, 4) rotations (w, x, y, z)
            - colors: (1, N, 3) linearRGB in [0, 1]
            - opacities: (1, N) in [0, 1]
        sort: Whether to sort by scale*opacity for progressive rendering.

    Returns:
        Raw bytes in .splat format (32 bytes per gaussian).
    """
    # Move all tensors to CPU and convert to numpy in one batch
    positions = gaussians.mean_vectors.detach().cpu().flatten(0, 1).numpy()   # (N, 3)
    scales = gaussians.singular_values.detach().cpu().flatten(0, 1).numpy()   # (N, 3)
    quats = gaussians.quaternions.detach().cpu().flatten(0, 1).numpy()        # (N, 4)
    colors_linear = gaussians.colors.detach().cpu().flatten(0, 1).numpy()     # (N, 3)
    opacities = gaussians.opacities.detach().cpu().flatten(0, 1).numpy()      # (N,)

    count = positions.shape[0]

    # Sort by scale * opacity for progressive rendering quality
    if sort and count > 0:
        scale_product = scales[:, 0] * scales[:, 1] * scales[:, 2]
        sort_key = -(scale_product * opacities)
        sorted_indices = np.argsort(sort_key)

        positions = positions[sorted_indices]
        scales = scales[sorted_indices]
        quats = quats[sorted_indices]
        colors_linear = colors_linear[sorted_indices]
        opacities = opacities[sorted_indices]

    # 1. Positions: float32 (already correct)
    positions = positions.astype(np.float32)

    # 2. Scales: float32 (already exp'd from singular_values)
    scales = scales.astype(np.float32)

    # 3. Colors: linearRGB -> sRGB -> uint8
    colors_srgb = _linearRGB_to_sRGB_numpy(colors_linear)
    alpha = opacities
    rgba = np.column_stack([colors_srgb, alpha])
    rgba_u8 = (rgba * 255.0).clip(0, 255).astype(np.uint8)

    # 4. Rotations: normalize quaternions -> quantize to uint8
    norms = np.linalg.norm(quats, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)  # avoid division by zero
    quats = quats / norms
    quats_u8 = (quats * 128.0 + 128.0).clip(0, 255).astype(np.uint8)

    # Pack into structured array: pos(3f4), scale(3f4), color(4u1), rot(4u1)
    splat_dtype = np.dtype([
        ('pos', '3f4'),
        ('scale', '3f4'),
        ('color', '4u1'),
        ('rot', '4u1'),
    ])

    data = np.empty(count, dtype=splat_dtype)
    data['pos'] = positions
    data['scale'] = scales
    data['color'] = rgba_u8
    data['rot'] = quats_u8

    return data.tobytes()


def process_ply_to_splat(ply_file_path):
    """Convert a .ply file to .splat format (legacy path).

    For new code, prefer gaussians_to_splat() which skips the PLY round-trip.
    This is kept for standalone PLY file conversion use cases.
    """
    from plyfile import PlyData

    ply_path_str = str(ply_file_path) if isinstance(ply_file_path, Path) else ply_file_path
    plydata = PlyData.read(ply_path_str)
    vert = plydata["vertex"]

    count = len(vert["x"])

    # Read all properties into contiguous arrays at once
    x = np.array(vert["x"], dtype=np.float32)
    y = np.array(vert["y"], dtype=np.float32)
    z = np.array(vert["z"], dtype=np.float32)
    scale_0 = np.array(vert["scale_0"], dtype=np.float32)
    scale_1 = np.array(vert["scale_1"], dtype=np.float32)
    scale_2 = np.array(vert["scale_2"], dtype=np.float32)
    rot_0 = np.array(vert["rot_0"], dtype=np.float32)
    rot_1 = np.array(vert["rot_1"], dtype=np.float32)
    rot_2 = np.array(vert["rot_2"], dtype=np.float32)
    rot_3 = np.array(vert["rot_3"], dtype=np.float32)
    f_dc_0 = np.array(vert["f_dc_0"], dtype=np.float32)
    f_dc_1 = np.array(vert["f_dc_1"], dtype=np.float32)
    f_dc_2 = np.array(vert["f_dc_2"], dtype=np.float32)
    opacity = np.array(vert["opacity"], dtype=np.float32)

    # Sort by scale/opacity for quality
    sorted_indices = np.argsort(
        -np.exp(scale_0 + scale_1 + scale_2)
        / (1.0 + np.exp(-opacity))
    )

    # Apply sort
    x = x[sorted_indices]; y = y[sorted_indices]; z = z[sorted_indices]
    scale_0 = scale_0[sorted_indices]; scale_1 = scale_1[sorted_indices]; scale_2 = scale_2[sorted_indices]
    rot_0 = rot_0[sorted_indices]; rot_1 = rot_1[sorted_indices]
    rot_2 = rot_2[sorted_indices]; rot_3 = rot_3[sorted_indices]
    f_dc_0 = f_dc_0[sorted_indices]; f_dc_1 = f_dc_1[sorted_indices]; f_dc_2 = f_dc_2[sorted_indices]
    opacity = opacity[sorted_indices]

    # 1. Positions
    positions = np.column_stack((x, y, z))

    # 2. Scales
    scales = np.exp(np.column_stack((scale_0, scale_1, scale_2)))

    # 3. Colors: SH -> RGB -> uint8
    SH_C0 = 0.28209479177387814
    r = 0.5 + SH_C0 * f_dc_0
    g = 0.5 + SH_C0 * f_dc_1
    b = 0.5 + SH_C0 * f_dc_2
    a = 1.0 / (1.0 + np.exp(-opacity))
    colors = (np.column_stack((r, g, b, a)) * 255.0).clip(0, 255).astype(np.uint8)

    # 4. Rotations
    rots = np.column_stack((rot_0, rot_1, rot_2, rot_3))
    norms = np.linalg.norm(rots, axis=1, keepdims=True)
    rots = rots / np.maximum(norms, 1e-10)
    rots = (rots * 128.0 + 128.0).clip(0, 255).astype(np.uint8)

    # Pack
    splat_dtype = np.dtype([
        ('pos', '3f4'),
        ('scale', '3f4'),
        ('color', '4u1'),
        ('rot', '4u1'),
    ])

    data = np.empty(count, dtype=splat_dtype)
    data['pos'] = positions
    data['scale'] = scales
    data['color'] = colors
    data['rot'] = rots

    return data.tobytes()
