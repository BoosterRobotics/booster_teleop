
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RobotState:

    qpos: np.ndarray
    qvel: np.ndarray
    quat: np.ndarray
    ang_vel: np.ndarray
    timestamp: float
    base_pos: np.ndarray | None = None
    base_lin_vel: np.ndarray | None = None
    real2sim_joint_indexes: np.ndarray | None = None
    sim2real_joint_indexes: np.ndarray | None = None

    def __post_init__(self) -> None:

        self.qpos = np.asarray(self.qpos)
        self.qvel = np.asarray(self.qvel)
        self.quat = np.asarray(self.quat)
        self.ang_vel = np.asarray(self.ang_vel)
        if self.base_pos is not None:
            self.base_pos = np.asarray(self.base_pos)
        if self.base_lin_vel is not None:
            self.base_lin_vel = np.asarray(self.base_lin_vel)

        num_joints = int(self.qpos.reshape(-1).shape[0])
        if self.real2sim_joint_indexes is None:
            self.real2sim_joint_indexes = np.arange(num_joints, dtype=np.int32)
        else:
            self.real2sim_joint_indexes = np.asarray(self.real2sim_joint_indexes, dtype=np.int32).reshape(-1)
        if self.sim2real_joint_indexes is None:
            self.sim2real_joint_indexes = np.arange(num_joints, dtype=np.int32)
        else:
            self.sim2real_joint_indexes = np.asarray(self.sim2real_joint_indexes, dtype=np.int32).reshape(-1)

    @property
    def joint_pos(self) -> np.ndarray:

        return np.asarray(self.qpos)

    @property
    def joint_vel(self) -> np.ndarray:

        return np.asarray(self.qvel)

    @property
    def root_pos_w(self) -> np.ndarray:

        if self.base_pos is None:
            return np.zeros(3, dtype=np.float32)
        return np.asarray(self.base_pos)

    @property
    def root_quat_w(self) -> np.ndarray:

        return np.asarray(self.quat)

    @property
    def root_ang_vel_b(self) -> np.ndarray:

        return np.asarray(self.ang_vel)

    @property
    def root_lin_vel_b(self) -> np.ndarray:

        if self.base_lin_vel is None:
            return np.zeros(3, dtype=np.float32)
        return np.asarray(self.base_lin_vel)
