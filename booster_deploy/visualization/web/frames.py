"""Render-frame adapters shared by the web hub and tests."""
from __future__ import annotations

from typing import Any, Iterable

import mujoco
import numpy as np

from booster_deploy.robots.t2_contract import CONTROL_JOINT_NAMES
from booster_deploy.types.human_frame import HumanFrame
from booster_deploy.types.retarget_types import ReferenceFrame50HzNs
from booster_deploy.visualization.revo2_combined_qpos import Revo2CombinedQposMapper


def _flat(values: np.ndarray) -> list[float]:
    return np.asarray(values, dtype=np.float32).reshape(-1).tolist()


def t2_body_qpos(model: mujoco.MjModel, compact: np.ndarray) -> np.ndarray:
    """Expand root + 31 control-order joints; leave unmeasured fingers at rest."""
    compact = np.asarray(compact, dtype=np.float64).reshape(-1)
    if compact.size != 7 + len(CONTROL_JOINT_NAMES):
        raise ValueError(f"Invalid T2 body qpos shape {compact.shape}")
    qpos = model.qpos0.copy()
    qpos[:7] = compact[:7]
    for name, value in zip(CONTROL_JOINT_NAMES, compact[7:], strict=True):
        qpos[model.jnt_qposadr[model.joint(name).id]] = value
    return qpos


def qpos_frame_payload(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos: np.ndarray,
    *,
    seq: int,
    sample_ns: int | None = None,
) -> dict[str, Any]:
    q = np.asarray(qpos, dtype=np.float64).reshape(-1)
    if q.shape != (int(model.nq),):
        raise ValueError(f"qpos shape {q.shape} != ({int(model.nq)},)")
    data.qpos[:] = q
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    return {
        "kind": "robot",
        "seq": int(seq),
        "sample_ns": None if sample_ns is None else int(sample_ns),
        "root_pos": _flat(q[:3]),
        "root_quat": _flat(q[3:7]),
        "body_pos": _flat(data.xpos),
        "body_quat": _flat(data.xquat),
    }


def reference_frame_payload(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    frame: ReferenceFrame50HzNs,
    *,
    hand_mapper: Revo2CombinedQposMapper | None = None,
) -> dict[str, Any]:
    joints = np.asarray(frame.joint_pos, dtype=np.float64).reshape(-1)
    qpos = np.concatenate((
        np.asarray(frame.root_pos, dtype=np.float64).reshape(3),
        np.asarray(frame.root_quat, dtype=np.float64).reshape(4),
        joints,
    ))
    if hand_mapper is not None and frame.hand_frame is not None:
        qpos = hand_mapper.compose(qpos, frame.hand_frame)
    else:
        qpos = t2_body_qpos(model, qpos)
    payload = qpos_frame_payload(
        model, data, qpos, seq=int(frame.sample_idx), sample_ns=int(frame.sample_ns)
    )
    payload["robot_name"] = str(frame.robot_name)
    payload["body_seq"] = getattr(frame, "body_seq", None)
    return payload


def human_frame_payload(
    frame: HumanFrame,
    *,
    seq: int,
    connections: Iterable[tuple[str, str]] = (),
    sample_ns: int | None = None,
) -> dict[str, Any]:
    names = sorted(frame)
    indices = {name: i for i, name in enumerate(names)}
    positions = np.stack([np.asarray(frame[name][0], dtype=np.float32).reshape(3) for name in names])
    # PICO drops joints to NaN while it re-acquires tracking (most visibly right
    # after entering teleop).  Publishing those poisons the camera and the browser
    # silently draws nothing, which reads as "the skeleton disappeared".
    if not np.isfinite(positions).all():
        raise ValueError("human frame contains non-finite joint positions")
    edges = [
        [indices[parent], indices[child]]
        for parent, child in connections
        if parent in indices and child in indices
    ]
    return {
        "kind": "human",
        "seq": int(seq),
        "sample_ns": None if sample_ns is None else int(sample_ns),
        "names": names,
        "positions": _flat(positions),
        "edges": edges,
    }
