"""Gaussian transformation and merging utilities for 360 pipeline.

Transforms Gaussians from individual cubemap face camera frames into a
shared world frame and merges them into a single Gaussian set.

Performance notes:
- apply_rigid_transform() uses quaternion multiplication instead of SVD,
  which is orders of magnitude faster for rotation+translation transforms.
- merge_gaussians() concatenates on CPU to avoid GPU memory buildup.
"""

import torch
import numpy as np
from typing import List

from sharp.utils.gaussians import Gaussians3D
from sharp.utils.linalg import (
    quaternions_from_rotation_matrices,
    quaternion_product,
)
from backend.equirect_to_cubemap import FACE_ROTATIONS, FACE_NAMES


def apply_rigid_transform(
    gaussians: Gaussians3D, transform: torch.Tensor
) -> Gaussians3D:
    """Apply a rigid (rotation + translation) transform without SVD.

    This is the fast path for transforms that contain only rotation and
    translation (no scaling or shearing). It replaces the general
    apply_transform() which decomposes and recomposes covariance matrices
    via SVD -- catastrophically slow at scale (28M+ gaussians).

    For a rigid transform:
    - Positions are rotated and translated normally.
    - Quaternions are composed via Hamilton product (closed-form, vectorized).
    - Singular values (scales) are unchanged under rotation.
    - Colors and opacities are unchanged.

    Args:
        gaussians: The Gaussians3D to transform.
        transform: A 3x4 affine transform [R | t] where R is a pure
                   rotation matrix (orthogonal, det=1).

    Returns:
        Transformed Gaussians3D on the same device as input.
    """
    R = transform[..., :3, :3]
    t = transform[..., :3, 3]

    # 1. Transform positions: pos_new = pos @ R^T + t
    mean_vectors = gaussians.mean_vectors @ R.T + t

    # 2. Transform quaternions via Hamilton product: q_new = q_R * q_old
    # Convert the single rotation matrix to a quaternion (trivially fast).
    q_R = quaternions_from_rotation_matrices(
        R.unsqueeze(0)
    )  # (1, 4)
    q_R = q_R.unsqueeze(0)  # (1, 1, 4) -- broadcasts with (1, N, 4)
    quaternions = quaternion_product(q_R, gaussians.quaternions)

    return Gaussians3D(
        mean_vectors=mean_vectors,
        singular_values=gaussians.singular_values,
        quaternions=quaternions,
        colors=gaussians.colors,
        opacities=gaussians.opacities,
    )


def get_face_transform(face_name: str, device: torch.device) -> torch.Tensor:
    """Get the 3x4 affine transform (world_from_camera) for a cubemap face.

    The transform is rotation-only (no translation), since all cubemap
    faces share the same camera origin.

    Args:
        face_name: One of 'front', 'back', 'left', 'right', 'up', 'down'.
        device: Torch device for the output tensor.

    Returns:
        3x4 affine transform tensor.
    """
    R = FACE_ROTATIONS[face_name]
    transform = np.zeros((3, 4), dtype=np.float64)
    transform[:3, :3] = R
    return torch.from_numpy(transform).float().to(device)


def transform_gaussians(
    gaussians: Gaussians3D, face_name: str, device: torch.device
) -> Gaussians3D:
    """Transform Gaussians from a face's camera frame to the world frame.

    Uses apply_rigid_transform (quaternion multiplication) instead of the
    general apply_transform (SVD decomposition) since cubemap face
    transforms are pure rotations.

    Args:
        gaussians: Gaussians3D in the face's local camera coordinates.
        face_name: The cubemap face name.
        device: Torch device.

    Returns:
        Gaussians3D transformed to the shared world coordinate frame.
    """
    transform = get_face_transform(face_name, device)
    return apply_rigid_transform(gaussians, transform)


def merge_gaussians(gaussians_list: List[Gaussians3D]) -> Gaussians3D:
    """Merge multiple Gaussians3D into a single set by concatenation.

    All input Gaussians should already be in the same coordinate frame
    (i.e., world frame after transformation).

    Args:
        gaussians_list: List of Gaussians3D to merge.

    Returns:
        Single Gaussians3D containing all Gaussians from all inputs,
        concatenated along dim=1 (the Gaussian count dimension).
    """
    if len(gaussians_list) == 0:
        raise ValueError("Cannot merge empty list of Gaussians")

    if len(gaussians_list) == 1:
        return gaussians_list[0]

    return Gaussians3D(
        mean_vectors=torch.cat(
            [g.mean_vectors for g in gaussians_list], dim=1
        ),
        singular_values=torch.cat(
            [g.singular_values for g in gaussians_list], dim=1
        ),
        quaternions=torch.cat(
            [g.quaternions for g in gaussians_list], dim=1
        ),
        colors=torch.cat(
            [g.colors for g in gaussians_list], dim=1
        ),
        opacities=torch.cat(
            [g.opacities for g in gaussians_list], dim=1
        ),
    )
