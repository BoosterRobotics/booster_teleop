from __future__ import annotations

from dataclasses import replace

import numpy as np

from booster_deploy.types.retarget_types import ReferenceFrame50HzNs


def _slerp_wxyz(start: np.ndarray, end: np.ndarray, ratio: float) -> np.ndarray:
    a = np.asarray(start, dtype=np.float64)
    b = np.asarray(end, dtype=np.float64)
    a = a / max(float(np.linalg.norm(a)), 1e-12)
    b = b / max(float(np.linalg.norm(b)), 1e-12)
    dot = float(np.dot(a, b))
    if dot < 0.0:
        b = -b
        dot = -dot
    dot = min(max(dot, -1.0), 1.0)
    if dot > 0.9995:
        out = a + float(ratio) * (b - a)
        return out / max(float(np.linalg.norm(out)), 1e-12)
    theta = float(np.arccos(dot))
    sin_theta = float(np.sin(theta))
    out = (
        np.sin((1.0 - ratio) * theta) / sin_theta * a
        + np.sin(float(ratio) * theta) / sin_theta * b
    )
    return out / max(float(np.linalg.norm(out)), 1e-12)


# Log the already computed transition arrays without changing the blend.
def _blend_identity(held,target,weight,elapsed,duration,tangent,**arrays):
    from booster_deploy.telemetry.postprocess import identity,output
    lineage=identity('blend',[held.identity,target.identity],weight=weight,
        parameters={'elapsed_s':elapsed,'duration_s':duration,'tangent':tangent,'method':'smoothstep_velocity_tangent'})
    output(lineage,source_ids=tuple(dict.fromkeys(held.source_ids+target.source_ids)),**arrays)
    return lineage


def blend_reference_windows(
    start: list[ReferenceFrame50HzNs],
    live: list[ReferenceFrame50HzNs],
    *,
    elapsed_s: float,
    duration_s: float,
) -> tuple[list[ReferenceFrame50HzNs], bool]:

    if len(start) != len(live) or not start:
        raise ValueError("reference transition windows must have equal non-zero length")
    duration = float(duration_s)
    if not np.isfinite(duration) or duration < 0.0:
        raise ValueError("reference transition duration must be finite and >= 0")
    if duration == 0.0 or elapsed_s >= duration:
        return list(live), True
    u = min(max(float(elapsed_s) / duration, 0.0), 1.0)
    alpha = 3.0 * u**2 - 2.0 * u**3
    alpha_dot = (6.0 * u - 6.0 * u**2) / duration
    tangent = duration * u * (1.0 - u) ** 2
    tangent_dot = 1.0 - 4.0 * u + 3.0 * u**2
    output: list[ReferenceFrame50HzNs] = []
    for held, target in zip(start, live):
        held_q = np.asarray(held.joint_pos, dtype=np.float64)
        target_q = np.asarray(target.joint_pos, dtype=np.float64)
        held_v = np.asarray(held.joint_vel, dtype=np.float64)
        target_v = np.asarray(target.joint_vel, dtype=np.float64)
        joint_pos = (1.0 - alpha) * held_q + alpha * target_q + tangent * held_v
        joint_vel = (
            tangent_dot * held_v
            + alpha * target_v
            + alpha_dot * (target_q - held_q)
        )
        root_pos = (1.0 - alpha) * np.asarray(held.root_pos) + alpha * np.asarray(target.root_pos)
        root_quat = _slerp_wxyz(held.root_quat, target.root_quat, alpha)
        output.append(
            replace(
                target,
                identity=_blend_identity(held,target,alpha,elapsed_s,duration,tangent,
                    joint_pos=np.asarray(joint_pos,dtype=np.float64),joint_vel=np.asarray(joint_vel,dtype=np.float64),
                    root_pos=np.asarray(root_pos,dtype=np.float64),root_quat=np.asarray(root_quat,dtype=np.float64)),
                source_ids=tuple(dict.fromkeys(held.source_ids+target.source_ids)),
                joint_pos=np.asarray(joint_pos, dtype=np.float64),
                joint_vel=np.asarray(joint_vel, dtype=np.float64),
                root_pos=np.asarray(root_pos, dtype=np.float64),
                root_quat=np.asarray(root_quat, dtype=np.float64),
            )
        )
    return output, u >= 1.0
