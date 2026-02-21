"""Equirectangular to cubemap conversion.

Extracts 6 perspective cubemap faces from a 360 equirectangular image
using pure numpy with bilinear sampling.
"""

import numpy as np
from typing import Dict, Tuple


# Face rotation matrices (world_from_camera).
# In OpenCV convention: x-right, y-down, z-forward.
# Each matrix rotates the camera's local +Z axis to the face's look direction.
FACE_ROTATIONS: Dict[str, np.ndarray] = {
    # Front: looking along +Z (identity)
    "front": np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]),
    # Back: looking along -Z (180 deg around Y)
    "back": np.array([
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, -1.0],
    ]),
    # Left: looking along -X (-90 deg around Y)
    "left": np.array([
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
    ]),
    # Right: looking along +X (+90 deg around Y)
    "right": np.array([
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
    ]),
    # Up: looking along -Y (+90 deg around X)
    "up": np.array([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ]),
    # Down: looking along +Y (-90 deg around X)
    "down": np.array([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
    ]),
}

# Ordered list of face names for consistent iteration
FACE_NAMES = ["front", "back", "left", "right", "up", "down"]


def _bilinear_sample(image: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Bilinear sampling from an image with horizontal wrapping.

    Args:
        image: Source image, shape (H, W, C), dtype uint8 or float.
        x: Horizontal pixel coordinates (float), shape (N,) or (H_out, W_out).
        y: Vertical pixel coordinates (float), shape (N,) or (H_out, W_out).

    Returns:
        Sampled values with same spatial shape as x/y and C channels.
    """
    H, W = image.shape[:2]

    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1

    # Horizontal wrapping (equirectangular wraps around)
    x0_w = x0 % W
    x1_w = x1 % W

    # Vertical clamping (poles don't wrap)
    y0_c = np.clip(y0, 0, H - 1)
    y1_c = np.clip(y1, 0, H - 1)

    # Fractional parts
    fx = x - np.floor(x)
    fy = y - np.floor(y)

    # Weights
    wa = (1.0 - fx) * (1.0 - fy)
    wb = fx * (1.0 - fy)
    wc = (1.0 - fx) * fy
    wd = fx * fy

    # Sample and interpolate
    img_float = image.astype(np.float64)
    result = (
        wa[..., None] * img_float[y0_c, x0_w]
        + wb[..., None] * img_float[y0_c, x1_w]
        + wc[..., None] * img_float[y1_c, x0_w]
        + wd[..., None] * img_float[y1_c, x1_w]
    )

    return result.astype(image.dtype)


def _extract_face(
    equirect: np.ndarray, face_name: str, face_size: int
) -> np.ndarray:
    """Extract a single cubemap face from an equirectangular image.

    Args:
        equirect: Equirectangular image, shape (H, W, C).
        face_name: One of 'front', 'back', 'left', 'right', 'up', 'down'.
        face_size: Output face resolution (face_size x face_size).

    Returns:
        Cubemap face image, shape (face_size, face_size, C).
    """
    H, W = equirect.shape[:2]
    R = FACE_ROTATIONS[face_name]

    # Create pixel grid for the cubemap face
    # For 90-degree FOV: tan(45deg) = 1, so normalized coords range [-1, 1]
    u = np.linspace(-1.0 + 1.0 / face_size, 1.0 - 1.0 / face_size, face_size)
    v = np.linspace(-1.0 + 1.0 / face_size, 1.0 - 1.0 / face_size, face_size)
    uu, vv = np.meshgrid(u, v)  # (face_size, face_size)

    # Local ray directions (camera looks along +Z)
    # Shape: (face_size, face_size, 3)
    rays_local = np.stack([uu, vv, np.ones_like(uu)], axis=-1)

    # Normalize rays
    norms = np.linalg.norm(rays_local, axis=-1, keepdims=True)
    rays_local = rays_local / norms

    # Rotate to world frame: rays_world = R @ rays_local
    # rays_local: (face_size, face_size, 3) -> reshape to (N, 3)
    shape = rays_local.shape[:2]
    rays_flat = rays_local.reshape(-1, 3)
    rays_world = (R @ rays_flat.T).T  # (N, 3)
    rays_world = rays_world.reshape(*shape, 3)

    dx = rays_world[..., 0]
    dy = rays_world[..., 1]
    dz = rays_world[..., 2]

    # Convert to spherical coordinates
    # longitude: angle in XZ plane from +Z axis
    lon = np.arctan2(dx, dz)  # [-pi, pi]
    # latitude: angle from horizon (positive = up = -Y in OpenCV)
    lat = np.arctan2(-dy, np.sqrt(dx ** 2 + dz ** 2))  # [-pi/2, pi/2]

    # Map to equirectangular pixel coordinates
    # lon=-pi -> u=0, lon=0 -> u=W/2, lon=pi -> u=W
    eq_x = (lon / (2.0 * np.pi) + 0.5) * W
    # lat=pi/2 -> v=0 (top), lat=0 -> v=H/2, lat=-pi/2 -> v=H (bottom)
    eq_y = (0.5 - lat / np.pi) * H

    # Bilinear sample
    face_image = _bilinear_sample(equirect, eq_x, eq_y)

    return face_image


def equirect_to_cubemap(
    image: np.ndarray, face_size: int = 0
) -> Dict[str, np.ndarray]:
    """Convert an equirectangular image to 6 cubemap faces.

    Args:
        image: Equirectangular image, shape (H, W, C). Should have ~2:1 aspect ratio.
        face_size: Resolution of each cubemap face. If 0, defaults to W/4.

    Returns:
        Dictionary mapping face names to face images.
        Keys: 'front', 'back', 'left', 'right', 'up', 'down'.
        Each value is a numpy array of shape (face_size, face_size, C).
    """
    H, W = image.shape[:2]

    if face_size <= 0:
        face_size = W // 4

    faces = {}
    for name in FACE_NAMES:
        faces[name] = _extract_face(image, name, face_size)

    return faces


def get_face_rotation_4x4(face_name: str) -> np.ndarray:
    """Get the 4x4 affine matrix (world_from_camera) for a cubemap face.

    This is the rotation-only affine matrix used to transform Gaussians
    from a face's camera coordinate frame to the shared world frame.

    Args:
        face_name: One of 'front', 'back', 'left', 'right', 'up', 'down'.

    Returns:
        4x4 affine matrix with the 3x3 rotation and zero translation.
    """
    R = FACE_ROTATIONS[face_name]
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    return T


def get_cubemap_focal_length(face_size: int) -> float:
    """Compute the focal length in pixels for a 90-degree FOV cubemap face.

    For 90-degree FOV: f_px = face_size / (2 * tan(45deg)) = face_size / 2.

    Args:
        face_size: The resolution of the cubemap face.

    Returns:
        Focal length in pixels.
    """
    return face_size / 2.0
