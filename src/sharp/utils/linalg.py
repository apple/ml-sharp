"""Contains linear algebra related utility functions.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation


def rotation_matrices_from_quaternions(quaternions: torch.Tensor) -> torch.Tensor:
    """Convert batch of quaternions into rotations matrices.

    Args:
        quaternions: The quaternions convert to matrices.

    Returns:
        The rotations matrices corresponding to the (normalized) quaternions.
    """
    device = quaternions.device
    shape = quaternions.shape[:-1]

    quaternions = quaternions / torch.linalg.norm(quaternions, dim=-1, keepdim=True)
    real_part = quaternions[..., 0]
    vector_part = quaternions[..., 1:]

    vector_cross = get_cross_product_matrix(vector_part)
    real_part = real_part[..., None, None]

    matrix_outer = vector_part[..., :, None] * vector_part[..., None, :]
    matrix_diag = real_part.square() * eyes(3, shape=shape, device=device)
    matrix_cross_1 = 2 * real_part * vector_cross
    matrix_cross_2 = vector_cross @ vector_cross

    return matrix_outer + matrix_diag + matrix_cross_1 + matrix_cross_2


def quaternions_from_rotation_matrices(
    matrices: torch.Tensor,
    use_gpu: bool = True,
) -> torch.Tensor:
    """Convert batch of rotation matrices to quaternions.

    Args:
        matrices: The matrices to convert to quaternions.
        use_gpu: If True and matrices are on CUDA, use pure PyTorch GPU implementation
                 for ~300x faster conversion. If False, use scipy on CPU (original behavior).

    Returns:
        The quaternions corresponding to the rotation matrices (w, x, y, z convention).

    Note: The GPU implementation is not differentiable but provides significant speedup
          for large batches (e.g., 2M+ gaussians). Set use_gpu=False for maximum numerical
          precision or when working with small batches on CPU.
    """
    if not matrices.shape[-2:] == (3, 3):
        raise ValueError(f"matrices have invalid shape {matrices.shape}")

    if use_gpu and matrices.is_cuda:
        return _quaternions_from_rotation_matrices_gpu(matrices)
    return _quaternions_from_rotation_matrices_cpu(matrices)


def _quaternions_from_rotation_matrices_cpu(matrices: torch.Tensor) -> torch.Tensor:
    """CPU implementation using scipy (original behavior)."""
    matrices_np = matrices.detach().cpu().numpy()
    quaternions_np = Rotation.from_matrix(matrices_np.reshape(-1, 3, 3)).as_quat()
    # We use a convention where the w component is at the start of the quaternion.
    quaternions_np = quaternions_np[:, [3, 0, 1, 2]]
    quaternions_np = quaternions_np.reshape(matrices_np.shape[:-2] + (4,))
    return torch.as_tensor(quaternions_np, device=matrices.device, dtype=matrices.dtype)


def _quaternions_from_rotation_matrices_gpu(matrices: torch.Tensor) -> torch.Tensor:
    """Pure PyTorch GPU implementation using Shepperd's method.

    Reference:
        https://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
    """
    # Flatten batch dimensions
    original_shape = matrices.shape[:-2]
    matrices = matrices.reshape(-1, 3, 3)
    batch_size = matrices.shape[0]

    # Allocate output
    quaternions = torch.zeros(batch_size, 4, device=matrices.device, dtype=matrices.dtype)

    # Extract matrix elements
    m00, m01, m02 = matrices[:, 0, 0], matrices[:, 0, 1], matrices[:, 0, 2]
    m10, m11, m12 = matrices[:, 1, 0], matrices[:, 1, 1], matrices[:, 1, 2]
    m20, m21, m22 = matrices[:, 2, 0], matrices[:, 2, 1], matrices[:, 2, 2]

    # Compute trace
    trace = m00 + m11 + m22

    # Case 1: trace > 0
    mask1 = trace > 0
    s1 = torch.sqrt(trace[mask1] + 1.0) * 2  # s = 4 * w
    quaternions[mask1, 0] = 0.25 * s1  # w
    quaternions[mask1, 1] = (m21[mask1] - m12[mask1]) / s1  # x
    quaternions[mask1, 2] = (m02[mask1] - m20[mask1]) / s1  # y
    quaternions[mask1, 3] = (m10[mask1] - m01[mask1]) / s1  # z

    # Case 2: m00 > m11 and m00 > m22
    mask2 = (~mask1) & (m00 > m11) & (m00 > m22)
    s2 = torch.sqrt(1.0 + m00[mask2] - m11[mask2] - m22[mask2]) * 2  # s = 4 * x
    quaternions[mask2, 0] = (m21[mask2] - m12[mask2]) / s2  # w
    quaternions[mask2, 1] = 0.25 * s2  # x
    quaternions[mask2, 2] = (m01[mask2] + m10[mask2]) / s2  # y
    quaternions[mask2, 3] = (m02[mask2] + m20[mask2]) / s2  # z

    # Case 3: m11 > m22
    mask3 = (~mask1) & (~mask2) & (m11 > m22)
    s3 = torch.sqrt(1.0 + m11[mask3] - m00[mask3] - m22[mask3]) * 2  # s = 4 * y
    quaternions[mask3, 0] = (m02[mask3] - m20[mask3]) / s3  # w
    quaternions[mask3, 1] = (m01[mask3] + m10[mask3]) / s3  # x
    quaternions[mask3, 2] = 0.25 * s3  # y
    quaternions[mask3, 3] = (m12[mask3] + m21[mask3]) / s3  # z

    # Case 4: else (m22 is largest)
    mask4 = (~mask1) & (~mask2) & (~mask3)
    s4 = torch.sqrt(1.0 + m22[mask4] - m00[mask4] - m11[mask4]) * 2  # s = 4 * z
    quaternions[mask4, 0] = (m10[mask4] - m01[mask4]) / s4  # w
    quaternions[mask4, 1] = (m02[mask4] + m20[mask4]) / s4  # x
    quaternions[mask4, 2] = (m12[mask4] + m21[mask4]) / s4  # y
    quaternions[mask4, 3] = 0.25 * s4  # z

    # Normalize to ensure unit quaternions
    quaternions = F.normalize(quaternions, dim=-1)

    # Reshape back to original batch shape
    return quaternions.reshape(original_shape + (4,))


def get_cross_product_matrix(vectors: torch.Tensor) -> torch.Tensor:
    """Generate cross product matrix for vector exterior product."""
    if not vectors.shape[-1] == 3:
        raise ValueError("Only 3-dimensional vectors are supported")
    device = vectors.device
    shape = vectors.shape[:-1]
    unit_basis = eyes(3, shape=shape, device=device)
    # We compute the matrix by multiplying each column of unit_basis with the
    # corresponding vector.
    return torch.cross(vectors[..., :, None], unit_basis, dim=-2)


def eyes(
    dim: int, shape: tuple[int, ...], device: torch.device | str | None = None
) -> torch.Tensor:
    """Create batch of identity matrices."""
    return torch.eye(dim, device=device).broadcast_to(shape + (dim, dim)).clone()


def quaternion_product(q1, q2):
    """Compute dot product between two quaternions."""
    real_1 = q1[..., :1]
    real_2 = q2[..., :1]
    vector_1 = q1[..., 1:]
    vector_2 = q2[..., 1:]

    real_out = real_1 * real_2 - (vector_1 * vector_2).sum(dim=-1, keepdim=True)
    vector_out = real_1 * vector_2 + real_2 * vector_1 + torch.cross(vector_1, vector_2)
    return torch.concatenate([real_out, vector_out], dim=-1)


def quaternion_conj(q):
    """Get conjugate of a quaternion."""
    real = q[..., :1]
    vector = q[..., 1:]
    return torch.concatenate([real, -vector], dim=-1)


def project(u: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Project tensor u to unit basis a."""
    unit_u = F.normalize(u, dim=-1)
    inner_prod = (unit_u * basis).sum(dim=-1, keepdim=True)
    return inner_prod * u
