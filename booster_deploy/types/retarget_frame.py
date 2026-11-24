
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class RetargetFrame:

    qpos: NDArray[np.float64]
    root_pos: NDArray[np.float64]
    root_quat: NDArray[np.float64]
    joint_pos: NDArray[np.float64]
    joint_vel: NDArray[np.float64] | None
    robot_name: str
    source: str = ""
    timestamp_s: Optional[float] = None
    seq: Optional[int] = None

    @classmethod
    def from_qpos(
        cls,
        qpos: NDArray[np.float64],
        *,
        robot_name: str,
        source: str = "",
        timestamp_s: Optional[float] = None,
        seq: Optional[int] = None,
        joint_vel: NDArray[np.float64] | None = None,
    ) -> "RetargetFrame":

        qpos_np = np.asarray(qpos, dtype=np.float64).copy()
        if qpos_np.shape[0] < 7:
            raise ValueError(f"qpos must contain at least root pos and quat, got shape {qpos_np.shape}")
        joint_pos = qpos_np[7:].copy()
        joint_vel_np = None
        if joint_vel is not None:
            joint_vel_np = np.asarray(joint_vel, dtype=np.float64).copy()
            if joint_vel_np.shape != joint_pos.shape:
                raise ValueError(f"joint_vel shape {joint_vel_np.shape} does not match joint_pos shape {joint_pos.shape}")
        return cls(
            qpos=qpos_np,
            root_pos=qpos_np[:3].copy(),
            root_quat=qpos_np[3:7].copy(),
            joint_pos=joint_pos,
            joint_vel=joint_vel_np,
            robot_name=robot_name,
            source=source,
            timestamp_s=timestamp_s,
            seq=seq,
        )
