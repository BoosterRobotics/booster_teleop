






from __future__ import annotations

import logging
import math
import numpy as np
import torch
import torch.nn.functional
from typing import Literal


logger = logging.getLogger(__name__)



@torch.jit.script
def scale_transform(x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:

    offset = (lower + upper) * 0.5

    return 2 * (x - offset) / (upper - lower)


@torch.jit.script
def unscale_transform(x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:

    offset = (lower + upper) * 0.5

    return x * (upper - lower) * 0.5 + offset


@torch.jit.script
def saturate(x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    return torch.max(torch.min(x, upper), lower)


@torch.jit.script
def normalize(x: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    return x / x.norm(p=2, dim=-1).clamp(min=eps, max=None).unsqueeze(-1)


@torch.jit.script
def wrap_to_pi(angles: torch.Tensor) -> torch.Tensor:

    wrapped_angle = (angles + torch.pi) % (2 * torch.pi)


    return torch.where((wrapped_angle == 0) & (angles > 0), torch.pi, wrapped_angle - torch.pi)


@torch.jit.script
def copysign(mag: float, other: torch.Tensor) -> torch.Tensor:
    mag_torch = abs(mag) * torch.ones_like(other)
    return torch.copysign(mag_torch, other)




@torch.jit.script
def quat_unique(q: torch.Tensor) -> torch.Tensor:
    return torch.where(q[..., 0:1] < 0, -q, q)


@torch.jit.script
def matrix_from_quat(quaternions: torch.Tensor) -> torch.Tensor:
    r, i, j, k = torch.unbind(quaternions, -1)

    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def convert_quat(quat: torch.Tensor | np.ndarray, to: Literal["xyzw", "wxyz"] = "xyzw") -> torch.Tensor | np.ndarray:

    if quat.shape[-1] != 4:
        msg = f"Expected input quaternion shape mismatch: {quat.shape} != (..., 4)."
        raise ValueError(msg)
    if to not in ["xyzw", "wxyz"]:
        msg = f"Expected input argument `to` to be 'xyzw' or 'wxyz'. Received: {to}."
        raise ValueError(msg)

    if isinstance(quat, np.ndarray):

        if to == "xyzw":

            return np.roll(quat, -1, axis=-1)
        else:

            return np.roll(quat, 1, axis=-1)
    else:

        if not isinstance(quat, torch.Tensor):
            quat = torch.tensor(quat, dtype=float)

        if to == "xyzw":

            return quat.roll(-1, dims=-1)
        else:

            return quat.roll(1, dims=-1)


@torch.jit.script
def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    shape = q.shape
    q = q.reshape(-1, 4)
    return torch.cat((q[..., 0:1], -q[..., 1:]), dim=-1).view(shape)


@torch.jit.script
def quat_inv(q: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    return quat_conjugate(q) / q.pow(2).sum(dim=-1, keepdim=True).clamp(min=eps)


@torch.jit.script
def quat_from_euler_xyz(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)

    qw = cy * cr * cp + sy * sr * sp
    qx = cy * sr * cp - sy * cr * sp
    qy = cy * cr * sp + sy * sr * cp
    qz = sy * cr * cp - cy * sr * sp

    return torch.stack([qw, qx, qy, qz], dim=-1)


@torch.jit.script
def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


@torch.jit.script
def quat_from_matrix(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(matrix.reshape(batch_dim + (9,)), dim=-1)

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )


    quat_by_rijk = torch.stack(
        [

            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),

            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),

            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),

            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )



    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))



    return quat_candidates[torch.nn.functional.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :].reshape(
        batch_dim + (4,)
    )


def _axis_angle_rotation(axis: Literal["X", "Y", "Z"], angle: torch.Tensor) -> torch.Tensor:
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError("letter must be either X, Y or Z.")

    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))


def matrix_from_euler(euler_angles: torch.Tensor, convention: str) -> torch.Tensor:
    if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    matrices = [_axis_angle_rotation(c, e) for c, e in zip(convention, torch.unbind(euler_angles, -1))]

    return torch.matmul(torch.matmul(matrices[0], matrices[1]), matrices[2])


@torch.jit.script
def euler_xyz_from_quat(
    quat: torch.Tensor, wrap_to_2pi: bool = False
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q_w, q_x, q_y, q_z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

    sin_roll = 2.0 * (q_w * q_x + q_y * q_z)
    cos_roll = 1 - 2 * (q_x * q_x + q_y * q_y)
    roll = torch.atan2(sin_roll, cos_roll)


    sin_pitch = 2.0 * (q_w * q_y - q_z * q_x)
    pitch = torch.where(torch.abs(sin_pitch) >= 1, copysign(torch.pi / 2.0, sin_pitch), torch.asin(sin_pitch))


    sin_yaw = 2.0 * (q_w * q_z + q_x * q_y)
    cos_yaw = 1 - 2 * (q_y * q_y + q_z * q_z)
    yaw = torch.atan2(sin_yaw, cos_yaw)

    if wrap_to_2pi:
        return roll % (2 * torch.pi), pitch % (2 * torch.pi), yaw % (2 * torch.pi)
    return roll, pitch, yaw


@torch.jit.script
def axis_angle_from_quat(quat: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:






    quat = quat * (1.0 - 2.0 * (quat[..., 0:1] < 0.0))
    mag = torch.linalg.norm(quat[..., 1:], dim=-1)
    half_angle = torch.atan2(mag, quat[..., 0])
    angle = 2.0 * half_angle

    sin_half_angles_over_angles = torch.where(
        angle.abs() > eps, torch.sin(half_angle) / angle, 0.5 - angle * angle / 48
    )
    return quat[..., 1:4] / sin_half_angles_over_angles.unsqueeze(-1)


@torch.jit.script
def quat_from_angle_axis(angle: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    theta = (angle / 2).unsqueeze(-1)
    xyz = normalize(axis) * theta.sin()
    w = theta.cos()
    return normalize(torch.cat([w, xyz], dim=-1))


@torch.jit.script
def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:

    if q1.shape != q2.shape:
        msg = f"Expected input quaternion shape mismatch: {q1.shape} != {q2.shape}."
        raise ValueError(msg)

    shape = q1.shape
    q1 = q1.reshape(-1, 4)
    q2 = q2.reshape(-1, 4)

    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]

    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    return torch.stack([w, x, y, z], dim=-1).view(shape)


def quat_mul_np(x: np.ndarray, y: np.ndarray, scalar_first: bool = True) -> np.ndarray:
    x = np.asarray(x)
    y = np.asarray(y)
    if not scalar_first:
        x = x[..., [3, 0, 1, 2]]
        y = y[..., [3, 0, 1, 2]]

    x0, x1, x2, x3 = x[..., 0:1], x[..., 1:2], x[..., 2:3], x[..., 3:4]
    y0, y1, y2, y3 = y[..., 0:1], y[..., 1:2], y[..., 2:3], y[..., 3:4]

    res = np.concatenate(
        [
            x0 * y0 - x1 * y1 - x2 * y2 - x3 * y3,
            x0 * y1 + x1 * y0 + x2 * y3 - x3 * y2,
            x0 * y2 - x1 * y3 + x2 * y0 + x3 * y1,
            x0 * y3 + x1 * y2 - x2 * y1 + x3 * y0,
        ],
        axis=-1,
    )

    if not scalar_first:
        res = res[..., [1, 2, 3, 0]]
    return res


@torch.jit.script
def yaw_quat(quat: torch.Tensor) -> torch.Tensor:
    shape = quat.shape
    quat_yaw = quat.view(-1, 4)
    qw = quat_yaw[:, 0]
    qx = quat_yaw[:, 1]
    qy = quat_yaw[:, 2]
    qz = quat_yaw[:, 3]
    yaw = torch.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    quat_yaw = torch.zeros_like(quat_yaw)
    quat_yaw[:, 3] = torch.sin(yaw / 2)
    quat_yaw[:, 0] = torch.cos(yaw / 2)
    quat_yaw = normalize(quat_yaw)
    return quat_yaw.view(shape)


@torch.jit.script
def quat_box_minus(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    quat_diff = quat_mul(q1, quat_conjugate(q2))
    return axis_angle_from_quat(quat_diff)


@torch.jit.script
def quat_box_plus(q: torch.Tensor, delta: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    delta_norm = torch.clamp_min(torch.linalg.norm(delta, dim=-1, keepdim=True), min=eps)
    delta_quat = quat_from_angle_axis(delta_norm.squeeze(-1), delta / delta_norm)
    new_quat = quat_mul(delta_quat, q)
    return quat_unique(new_quat)


@torch.jit.script
def quat_apply(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:

    shape = vec.shape

    quat = quat.reshape(-1, 4)
    vec = vec.reshape(-1, 3)

    xyz = quat[:, 1:]
    t = xyz.cross(vec, dim=-1) * 2
    return (vec + quat[:, 0:1] * t + xyz.cross(t, dim=-1)).view(shape)


@torch.jit.script
def quat_apply_inverse(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:

    shape = vec.shape

    quat = quat.reshape(-1, 4)
    vec = vec.reshape(-1, 3)

    xyz = quat[:, 1:]
    t = xyz.cross(vec, dim=-1) * 2
    return (vec - quat[:, 0:1] * t + xyz.cross(t, dim=-1)).view(shape)


@torch.jit.script
def quat_apply_yaw(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    quat_yaw = yaw_quat(quat)
    return quat_apply(quat_yaw, vec)


def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:

    logger.warning(
        "The function 'quat_rotate' will be deprecated in favor of the faster method 'quat_apply'."
        " Please use 'quat_apply' instead...."
    )
    return quat_apply(q, v)


def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    logger.warning(
        "The function 'quat_rotate_inverse' will be deprecated in favor of the faster method 'quat_apply_inverse'."
        " Please use 'quat_apply_inverse' instead...."
    )
    return quat_apply_inverse(q, v)


@torch.jit.script
def quat_error_magnitude(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    axis_angle_error = quat_box_minus(q1, q2)
    return torch.norm(axis_angle_error, dim=-1)


@torch.jit.script
def skew_symmetric_matrix(vec: torch.Tensor) -> torch.Tensor:

    if vec.shape[-1] != 3:
        raise ValueError(f"Expected input vector shape mismatch: {vec.shape} != (..., 3).")

    if vec.ndim == 1:
        vec = vec.unsqueeze(0)

    skew_sym_mat = torch.zeros(vec.shape[0], 3, 3, device=vec.device, dtype=vec.dtype)
    skew_sym_mat[:, 0, 1] = -vec[:, 2]
    skew_sym_mat[:, 0, 2] = vec[:, 1]
    skew_sym_mat[:, 1, 2] = -vec[:, 0]
    skew_sym_mat[:, 1, 0] = vec[:, 2]
    skew_sym_mat[:, 2, 0] = -vec[:, 1]
    skew_sym_mat[:, 2, 1] = vec[:, 0]

    return skew_sym_mat




def is_identity_pose(pos: torch.tensor, rot: torch.tensor) -> bool:

    pos_identity = torch.zeros_like(pos)
    rot_identity = torch.zeros_like(rot)
    rot_identity[..., 0] = 1

    return torch.allclose(pos, pos_identity) and torch.allclose(rot, rot_identity)


@torch.jit.script
def combine_frame_transforms(
    t01: torch.Tensor, q01: torch.Tensor, t12: torch.Tensor | None = None, q12: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:

    if q12 is not None:
        q02 = quat_mul(q01, q12)
    else:
        q02 = q01

    if t12 is not None:
        t02 = t01 + quat_apply(q01, t12)
    else:
        t02 = t01

    return t02, q02


def rigid_body_twist_transform(
    v0: torch.Tensor, w0: torch.Tensor, t01: torch.Tensor, q01: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    w1 = quat_rotate_inverse(q01, w0)
    v1 = quat_rotate_inverse(q01, v0 + torch.cross(w0, t01, dim=-1))
    return v1, w1


@torch.jit.script
def subtract_frame_transforms(
    t01: torch.Tensor, q01: torch.Tensor, t02: torch.Tensor | None = None, q02: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:

    q10 = quat_inv(q01)
    if q02 is not None:
        q12 = quat_mul(q10, q02)
    else:
        q12 = q10

    if t02 is not None:
        t12 = quat_apply(q10, t02 - t01)
    else:
        t12 = quat_apply(q10, -t01)
    return t12, q12



def compute_pose_error(
    t01: torch.Tensor,
    q01: torch.Tensor,
    t02: torch.Tensor,
    q02: torch.Tensor,
    rot_error_type: Literal["quat", "axis_angle"] = "axis_angle",
) -> tuple[torch.Tensor, torch.Tensor]:



    source_quat_norm = quat_mul(q01, quat_conjugate(q01))[:, 0]

    source_quat_inv = quat_conjugate(q01) / source_quat_norm.unsqueeze(-1)

    quat_error = quat_mul(q02, source_quat_inv)


    pos_error = t02 - t01


    if rot_error_type == "quat":
        return pos_error, quat_error
    elif rot_error_type == "axis_angle":

        axis_angle_error = axis_angle_from_quat(quat_error)
        return pos_error, axis_angle_error
    else:
        raise ValueError(f"Unsupported orientation error type: {rot_error_type}. Valid: 'quat', 'axis_angle'.")


@torch.jit.script
def apply_delta_pose(
    source_pos: torch.Tensor, source_rot: torch.Tensor, delta_pose: torch.Tensor, eps: float = 1.0e-6
) -> tuple[torch.Tensor, torch.Tensor]:

    num_poses = source_pos.shape[0]
    device = source_pos.device


    target_pos = source_pos + delta_pose[:, 0:3]

    rot_actions = delta_pose[:, 3:6]
    angle = torch.linalg.vector_norm(rot_actions, dim=1)
    axis = rot_actions / angle.unsqueeze(-1)

    identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).repeat(num_poses, 1)
    rot_delta_quat = torch.where(
        angle.unsqueeze(-1).repeat(1, 4) > eps, quat_from_angle_axis(angle, axis), identity_quat
    )

    target_rot = quat_mul(rot_delta_quat, source_rot)

    return target_pos, target_rot


@torch.jit.script
def transform_points(
    points: torch.Tensor, pos: torch.Tensor | None = None, quat: torch.Tensor | None = None
) -> torch.Tensor:
    points_batch = points.clone()

    is_batched = points_batch.dim() == 3

    if points_batch.dim() == 2:
        points_batch = points_batch[None]
    if points_batch.dim() != 3:
        raise ValueError(f"Expected points to have dim = 2 or dim = 3: got shape {points.shape}")
    if not (pos is None or pos.dim() == 1 or pos.dim() == 2):
        raise ValueError(f"Expected pos to have dim = 1 or dim = 2: got shape {pos.shape}")
    if not (quat is None or quat.dim() == 1 or quat.dim() == 2):
        raise ValueError(f"Expected quat to have dim = 1 or dim = 2: got shape {quat.shape}")

    if quat is not None:

        rot_mat = matrix_from_quat(quat)
        if rot_mat.dim() == 2:
            rot_mat = rot_mat[None]


        points_batch = torch.matmul(rot_mat, points_batch.transpose_(1, 2))

        points_batch = points_batch.transpose_(1, 2)

    if pos is not None:

        if pos.dim() == 1:
            pos = pos[None, None, :]
        else:
            pos = pos[:, None, :]

        points_batch += pos

    if not is_batched:
        points_batch = points_batch.squeeze(0)

    return points_batch




@torch.jit.script
def orthogonalize_perspective_depth(depth: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:

    perspective_depth_batch = depth.clone()
    intrinsics_batch = intrinsics.clone()


    is_batched = perspective_depth_batch.dim() == 4 or (
        perspective_depth_batch.dim() == 3 and perspective_depth_batch.shape[-1] != 1
    )


    add_last_dim = False
    if perspective_depth_batch.dim() == 4 and perspective_depth_batch.shape[-1] == 1:
        add_last_dim = True
        perspective_depth_batch = perspective_depth_batch.squeeze(dim=3)
    if perspective_depth_batch.dim() == 3 and perspective_depth_batch.shape[-1] == 1:
        add_last_dim = True
        perspective_depth_batch = perspective_depth_batch.squeeze(dim=2)

    if perspective_depth_batch.dim() == 2:
        perspective_depth_batch = perspective_depth_batch[None]

    if intrinsics_batch.dim() == 2:
        intrinsics_batch = intrinsics_batch[None]

    if is_batched and intrinsics_batch.shape[0] == 1:
        intrinsics_batch = intrinsics_batch.expand(perspective_depth_batch.shape[0], -1, -1)


    if perspective_depth_batch.dim() != 3:
        raise ValueError(f"Expected depth images to have 2, 3, or 4 dimensions; got {depth.shape}.")
    if intrinsics_batch.dim() != 3:
        raise ValueError(f"Expected intrinsics to have shape (3, 3) or (N, 3, 3); got {intrinsics.shape}.")


    im_height, im_width = perspective_depth_batch.shape[1:]


    fx = intrinsics_batch[:, 0, 0].view(-1, 1, 1)
    fy = intrinsics_batch[:, 1, 1].view(-1, 1, 1)
    cx = intrinsics_batch[:, 0, 2].view(-1, 1, 1)
    cy = intrinsics_batch[:, 1, 2].view(-1, 1, 1)


    u_grid = torch.arange(im_width, device=depth.device, dtype=depth.dtype)
    v_grid = torch.arange(im_height, device=depth.device, dtype=depth.dtype)
    u_grid, v_grid = torch.meshgrid(u_grid, v_grid, indexing="xy")


    u_grid = u_grid.unsqueeze(0).expand(perspective_depth_batch.shape[0], -1, -1)
    v_grid = v_grid.unsqueeze(0).expand(perspective_depth_batch.shape[0], -1, -1)


    x_term = ((u_grid - cx) / fx) ** 2
    y_term = ((v_grid - cy) / fy) ** 2


    orthogonal_depth = perspective_depth_batch / torch.sqrt(1 + x_term + y_term)


    if add_last_dim:
        orthogonal_depth = orthogonal_depth.unsqueeze(-1)


    if not is_batched:
        orthogonal_depth = orthogonal_depth.squeeze(0)

    return orthogonal_depth


@torch.jit.script
def unproject_depth(depth: torch.Tensor, intrinsics: torch.Tensor, is_ortho: bool = True) -> torch.Tensor:

    intrinsics_batch = intrinsics.clone()

    if not is_ortho:
        depth_batch = orthogonalize_perspective_depth(depth, intrinsics)
    else:
        depth_batch = depth.clone()


    is_batched = depth_batch.dim() == 4 or (depth_batch.dim() == 3 and depth_batch.shape[-1] != 1)

    if depth_batch.dim() == 3 and depth_batch.shape[-1] == 1:
        depth_batch = depth_batch.squeeze(dim=2)
    if depth_batch.dim() == 2:
        depth_batch = depth_batch[None]
    if depth_batch.dim() == 4 and depth_batch.shape[-1] == 1:
        depth_batch = depth_batch.squeeze(dim=3)
    if intrinsics_batch.dim() == 2:
        intrinsics_batch = intrinsics_batch[None]

    if depth_batch.dim() != 3:
        raise ValueError(f"Expected depth images to have dim = 2 or 3 or 4: got shape {depth.shape}")
    if intrinsics_batch.dim() != 3:
        raise ValueError(f"Expected intrinsics to have shape (3, 3) or (N, 3, 3): got shape {intrinsics.shape}")


    im_height, im_width = depth_batch.shape[1:]

    indices_u = torch.arange(im_width, device=depth.device, dtype=depth.dtype)
    indices_v = torch.arange(im_height, device=depth.device, dtype=depth.dtype)
    img_indices = torch.stack(torch.meshgrid([indices_u, indices_v], indexing="ij"), dim=0).reshape(2, -1)
    pixels = torch.nn.functional.pad(img_indices, (0, 0, 0, 1), mode="constant", value=1.0)
    pixels = pixels.unsqueeze(0)


    points = torch.matmul(torch.inverse(intrinsics_batch), pixels)
    points = points / points[:, -1, :].unsqueeze(1)

    depth_batch = depth_batch.transpose_(1, 2).reshape(depth_batch.shape[0], -1).unsqueeze(2)
    depth_batch = depth_batch.expand(-1, -1, 3)

    points_xyz = points.transpose_(1, 2) * depth_batch


    if not is_batched:
        points_xyz = points_xyz.squeeze(0)

    return points_xyz


@torch.jit.script
def project_points(points: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:

    points_batch = points.clone()
    intrinsics_batch = intrinsics.clone()


    is_batched = points_batch.dim() == 2

    if points_batch.dim() == 2:
        points_batch = points_batch[None]
    if intrinsics_batch.dim() == 2:
        intrinsics_batch = intrinsics_batch[None]

    if points_batch.dim() != 3:
        raise ValueError(f"Expected points to have dim = 3: got shape {points.shape}.")
    if intrinsics_batch.dim() != 3:
        raise ValueError(f"Expected intrinsics to have shape (3, 3) or (N, 3, 3): got shape {intrinsics.shape}.")


    points_2d = torch.matmul(intrinsics_batch, points_batch.transpose(1, 2))
    points_2d = points_2d / points_2d[:, -1, :].unsqueeze(1)
    points_2d = points_2d.transpose_(1, 2)

    points_2d[:, :, -1] = points_batch[:, :, -1]


    if not is_batched:
        points_2d = points_2d.squeeze(0)

    return points_2d




@torch.jit.script
def default_orientation(num: int, device: str) -> torch.Tensor:
    quat = torch.zeros((num, 4), dtype=torch.float, device=device)
    quat[..., 0] = 1.0

    return quat


@torch.jit.script
def random_orientation(num: int, device: str) -> torch.Tensor:

    quat = torch.randn((num, 4), dtype=torch.float, device=device)

    return torch.nn.functional.normalize(quat, p=2.0, dim=-1, eps=1e-12)


@torch.jit.script
def random_yaw_orientation(num: int, device: str) -> torch.Tensor:
    roll = torch.zeros(num, dtype=torch.float, device=device)
    pitch = torch.zeros(num, dtype=torch.float, device=device)
    yaw = 2 * torch.pi * torch.rand(num, dtype=torch.float, device=device)

    return quat_from_euler_xyz(roll, pitch, yaw)


def sample_triangle(lower: float, upper: float, size: int | tuple[int, ...], device: str) -> torch.Tensor:

    if isinstance(size, int):
        size = (size,)

    r = 2 * torch.rand(*size, device=device) - 1

    r = torch.where(r < 0.0, -torch.sqrt(-r), torch.sqrt(r))

    r = (r + 1.0) / 2.0

    return (upper - lower) * r + lower


def sample_uniform(
    lower: torch.Tensor | float, upper: torch.Tensor | float, size: int | tuple[int, ...], device: str
) -> torch.Tensor:

    if isinstance(size, int):
        size = (size,)

    return torch.rand(*size, device=device) * (upper - lower) + lower


def sample_log_uniform(
    lower: torch.Tensor | float, upper: torch.Tensor | float, size: int | tuple[int, ...], device: str
) -> torch.Tensor:

    if not isinstance(lower, torch.Tensor):
        lower = torch.tensor(lower, dtype=torch.float, device=device)
    if not isinstance(upper, torch.Tensor):
        upper = torch.tensor(upper, dtype=torch.float, device=device)

    return torch.exp(sample_uniform(torch.log(lower), torch.log(upper), size, device))


def sample_gaussian(
    mean: torch.Tensor | float, std: torch.Tensor | float, size: int | tuple[int, ...], device: str
) -> torch.Tensor:
    if isinstance(mean, float):
        if isinstance(size, int):
            size = (size,)
        return torch.normal(mean=mean, std=std, size=size).to(device=device)
    else:
        return torch.normal(mean=mean, std=std).to(device=device)


def sample_cylinder(
    radius: float, h_range: tuple[float, float], size: int | tuple[int, ...], device: str
) -> torch.Tensor:

    angles = (torch.rand(size, device=device) * 2 - 1) * torch.pi
    h_min, h_max = h_range

    if isinstance(size, int):
        size = (size, 3)
    else:
        size += (3,)

    xyz = torch.zeros(size, device=device)
    xyz[..., 0] = radius * torch.cos(angles)
    xyz[..., 1] = radius * torch.sin(angles)
    xyz[..., 2].uniform_(h_min, h_max)

    return xyz




def convert_camera_frame_orientation_convention(
    orientation: torch.Tensor,
    origin: Literal["opengl", "ros", "world"] = "opengl",
    target: Literal["opengl", "ros", "world"] = "ros",
) -> torch.Tensor:
    if target == origin:
        return orientation.clone()


    if origin == "ros":

        rotm = matrix_from_quat(orientation)
        rotm[:, :, 2] = -rotm[:, :, 2]
        rotm[:, :, 1] = -rotm[:, :, 1]

        quat_gl = quat_from_matrix(rotm)
    elif origin == "world":

        rotm = matrix_from_quat(orientation)
        rotm = torch.matmul(
            rotm,
            matrix_from_euler(torch.tensor([math.pi / 2, -math.pi / 2, 0], device=orientation.device), "XYZ"),
        )

        quat_gl = quat_from_matrix(rotm)
    else:
        quat_gl = orientation


    if target == "ros":

        rotm = matrix_from_quat(quat_gl)
        rotm[:, :, 2] = -rotm[:, :, 2]
        rotm[:, :, 1] = -rotm[:, :, 1]
        return quat_from_matrix(rotm)
    elif target == "world":

        rotm = matrix_from_quat(quat_gl)
        rotm = torch.matmul(
            rotm,
            matrix_from_euler(torch.tensor([math.pi / 2, -math.pi / 2, 0], device=orientation.device), "XYZ").T,
        )
        return quat_from_matrix(rotm)
    else:
        return quat_gl.clone()


def create_rotation_matrix_from_view(
    eyes: torch.Tensor,
    targets: torch.Tensor,
    up_axis: Literal["Y", "Z"] = "Z",
    device: str = "cpu",
) -> torch.Tensor:
    if up_axis == "Y":
        up_axis_vec = torch.tensor((0, 1, 0), device=device, dtype=torch.float32).repeat(eyes.shape[0], 1)
    elif up_axis == "Z":
        up_axis_vec = torch.tensor((0, 0, 1), device=device, dtype=torch.float32).repeat(eyes.shape[0], 1)
    else:
        raise ValueError(f"Invalid up axis: {up_axis}. Valid options are 'Y' and 'Z'.")


    z_axis = -torch.nn.functional.normalize(targets - eyes, eps=1e-5)
    x_axis = torch.nn.functional.normalize(torch.cross(up_axis_vec, z_axis, dim=1), eps=1e-5)
    y_axis = torch.nn.functional.normalize(torch.cross(z_axis, x_axis, dim=1), eps=1e-5)
    is_close = torch.isclose(x_axis, torch.tensor(0.0), atol=5e-3).all(dim=1, keepdim=True)
    if is_close.any():
        replacement = torch.nn.functional.normalize(torch.cross(y_axis, z_axis, dim=1), eps=1e-5)
        x_axis = torch.where(is_close, replacement, x_axis)
    R = torch.cat((x_axis[:, None, :], y_axis[:, None, :], z_axis[:, None, :]), dim=1)
    return R.transpose(1, 2)


def make_pose(pos: torch.Tensor, rot: torch.Tensor) -> torch.Tensor:
    assert isinstance(pos, torch.Tensor), "Input must be a torch tensor"
    assert isinstance(rot, torch.Tensor), "Input must be a torch tensor"
    assert pos.shape[:-1] == rot.shape[:-2]
    assert pos.shape[-1] == rot.shape[-2] == rot.shape[-1] == 3
    pose = torch.zeros(pos.shape[:-1] + (4, 4), dtype=pos.dtype, device=pos.device)
    pose[..., :3, :3] = rot
    pose[..., :3, 3] = pos
    pose[..., 3, 3] = 1.0
    return pose


def unmake_pose(pose: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    assert isinstance(pose, torch.Tensor), "Input must be a torch tensor"
    return pose[..., :3, 3], pose[..., :3, :3]


def pose_inv(pose: torch.Tensor) -> torch.Tensor:
    assert isinstance(pose, torch.Tensor), "Input must be a torch tensor"
    num_axes = len(pose.shape)
    assert num_axes >= 2

    inv_pose = torch.zeros_like(pose)


    inv_pose[..., :3, :3] = pose[..., :3, :3].transpose(-1, -2)


    inv_pose[..., :3, 3] = torch.matmul(-inv_pose[..., :3, :3], pose[..., :3, 3:4])[..., 0]
    inv_pose[..., 3, 3] = 1.0
    return inv_pose


def pose_in_A_to_pose_in_B(pose_in_A: torch.Tensor, pose_A_in_B: torch.Tensor) -> torch.Tensor:
    assert isinstance(pose_in_A, torch.Tensor), "Input must be a torch tensor"
    assert isinstance(pose_A_in_B, torch.Tensor), "Input must be a torch tensor"
    return torch.matmul(pose_A_in_B, pose_in_A)


def quat_slerp(q1: torch.Tensor, q2: torch.Tensor, tau: float) -> torch.Tensor:
    assert isinstance(q1, torch.Tensor), "Input must be a torch tensor"
    assert isinstance(q2, torch.Tensor), "Input must be a torch tensor"
    if tau == 0.0:
        return q1
    elif tau == 1.0:
        return q2
    d = torch.dot(q1, q2)
    if abs(abs(d) - 1.0) < torch.finfo(q1.dtype).eps * 4.0:
        return q1
    if d < 0.0:

        d = -d
        q2 *= -1.0
    angle = torch.acos(torch.clamp(d, -1, 1))
    if abs(angle) < torch.finfo(q1.dtype).eps * 4.0:
        return q1
    isin = 1.0 / torch.sin(angle)
    q1 = q1 * torch.sin((1.0 - tau) * angle) * isin
    q2 = q2 * torch.sin(tau * angle) * isin
    q1 = q1 + q2
    return q1


def interpolate_rotations(R1: torch.Tensor, R2: torch.Tensor, num_steps: int, axis_angle: bool = True) -> torch.Tensor:
    assert isinstance(R1, torch.Tensor), "Input must be a torch tensor"
    assert isinstance(R2, torch.Tensor), "Input must be a torch tensor"
    if axis_angle:

        delta_rot_mat = torch.matmul(R2, R1.transpose(-1, -2))
        delta_quat = quat_from_matrix(delta_rot_mat)
        delta_axis_angle = axis_angle_from_quat(delta_quat)


        delta_angle = torch.linalg.norm(delta_axis_angle)


        rot_step_size = delta_angle / num_steps


        if delta_angle < 0.05:

            rot_steps = torch.stack([R2 for _ in range(num_steps)])
        else:

            delta_axis = delta_axis_angle / delta_angle
            delta_rot_steps = [
                matrix_from_quat(quat_from_angle_axis(i * rot_step_size, delta_axis)) for i in range(num_steps)
            ]
            rot_steps = torch.stack([torch.matmul(delta_rot_steps[i], R1) for i in range(num_steps)])
    else:
        q1 = quat_from_matrix(R1)
        q2 = quat_from_matrix(R2)
        rot_steps = torch.stack(
            [matrix_from_quat(quat_slerp(q1, q2, tau=float(i) / num_steps)) for i in range(num_steps)]
        )


    rot_steps = torch.cat([rot_steps, R2[None]], dim=0)

    return rot_steps


def interpolate_poses(
    pose_1: torch.Tensor,
    pose_2: torch.Tensor,
    num_steps: int = None,
    step_size: float = None,
    perturb: bool = False,
) -> tuple[torch.Tensor, int]:
    assert isinstance(pose_1, torch.Tensor), "Input must be a torch tensor"
    assert isinstance(pose_2, torch.Tensor), "Input must be a torch tensor"
    assert step_size is None or num_steps is None

    pos1, rot1 = unmake_pose(pose_1)
    pos2, rot2 = unmake_pose(pose_2)

    if num_steps == 0:

        return (
            torch.cat([pos1[None], pos2[None]], dim=0),
            torch.cat([rot1[None], rot2[None]], dim=0),
            num_steps,
        )

    delta_pos = pos2 - pos1
    if num_steps is None:
        assert torch.norm(delta_pos) > 0
        num_steps = math.ceil(torch.norm(delta_pos) / step_size)

    num_steps += 1
    assert num_steps >= 2


    pos_step_size = delta_pos / num_steps
    grid = torch.arange(num_steps, dtype=torch.float32)
    if perturb:

        perturbations = torch.rand(num_steps - 2) - 0.5
        grid[1:-1] += perturbations
    pos_steps = torch.stack([pos1 + grid[i] * pos_step_size for i in range(num_steps)])


    pos_steps = torch.cat([pos_steps, pos2[None]], dim=0)


    rot_steps = interpolate_rotations(R1=rot1, R2=rot2, num_steps=num_steps, axis_angle=True)

    pose_steps = make_pose(pos_steps, rot_steps)
    return pose_steps, num_steps - 1


def transform_poses_from_frame_A_to_frame_B(
    src_poses: torch.Tensor, frame_A: torch.Tensor, frame_B: torch.Tensor
) -> torch.Tensor:

    src_poses_rel_frame_B = pose_in_A_to_pose_in_B(
        pose_in_A=src_poses,
        pose_A_in_B=pose_inv(frame_B[None]),
    )


    transformed_poses = pose_in_A_to_pose_in_B(
        pose_in_A=src_poses_rel_frame_B,
        pose_A_in_B=frame_A[None],
    )
    return transformed_poses


def generate_random_rotation(rot_boundary: float = (2 * math.pi)) -> torch.Tensor:
    angles = torch.rand(3) * rot_boundary
    Rx = torch.tensor(
        [[1, 0, 0], [0, torch.cos(angles[0]), -torch.sin(angles[0])], [0, torch.sin(angles[0]), torch.cos(angles[0])]]
    )

    Ry = torch.tensor(
        [[torch.cos(angles[1]), 0, torch.sin(angles[1])], [0, 1, 0], [-torch.sin(angles[1]), 0, torch.cos(angles[1])]]
    )

    Rz = torch.tensor(
        [[torch.cos(angles[2]), -torch.sin(angles[2]), 0], [torch.sin(angles[2]), torch.cos(angles[2]), 0], [0, 0, 1]]
    )


    R = torch.matmul(torch.matmul(Rz, Ry), Rx)
    return R


def generate_random_translation(pos_boundary: float = 1) -> torch.Tensor:
    return torch.rand(3) * 2 * pos_boundary - pos_boundary


def generate_random_transformation_matrix(pos_boundary: float = 1, rot_boundary: float = (2 * math.pi)) -> torch.Tensor:
    R = generate_random_rotation(rot_boundary)
    translation = generate_random_translation(pos_boundary)


    T = torch.eye(4)
    T[:3, :3] = R
    T[:3, 3] = translation

    return T
