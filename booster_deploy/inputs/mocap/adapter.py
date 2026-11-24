
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from scipy.spatial.transform import Rotation as R

from booster_deploy.types import HumanFrame
from booster_deploy.types.mocap_types import MocapJoints
from booster_deploy.utils.isaaclab.math import quat_mul_np


MOCAP_HIERARCHY: dict[str, str] = {
    "Hips": "world",
    "RightUpLeg": "Hips",
    "RightLeg": "RightUpLeg",
    "RightFoot": "RightLeg",
    "RightTiptoe": "RightFoot",
    "LeftUpLeg": "Hips",
    "LeftLeg": "LeftUpLeg",
    "LeftFoot": "LeftLeg",
    "LeftTiptoe": "LeftFoot",
    "Spine": "Hips",
    "Spine1": "Spine",
    "Spine2": "Spine1",
    "Neck": "Spine2",
    "Neck1": "Neck",
    "Head": "Neck1",
    "Head1": "Head",
    "RightShoulder": "Spine2",
    "RightArm": "RightShoulder",
    "RightForeArm": "RightArm",
    "RightHand": "RightForeArm",
    "RightHandThumb1": "RightHand",
    "RightHandThumb2": "RightHandThumb1",
    "RightHandThumb3": "RightHandThumb2",
    "RightInHandIndex": "RightHand",
    "RightHandIndex1": "RightInHandIndex",
    "RightHandIndex2": "RightHandIndex1",
    "RightHandIndex3": "RightHandIndex2",
    "RightInHandMiddle": "RightHand",
    "RightHandMiddle1": "RightInHandMiddle",
    "RightHandMiddle2": "RightHandMiddle1",
    "RightHandMiddle3": "RightHandMiddle2",
    "RightInHandRing": "RightHand",
    "RightHandRing1": "RightInHandRing",
    "RightHandRing2": "RightHandRing1",
    "RightHandRing3": "RightHandRing2",
    "RightInHandPinky": "RightHand",
    "RightHandPinky1": "RightInHandPinky",
    "RightHandPinky2": "RightHandPinky1",
    "RightHandPinky3": "RightHandPinky2",
    "LeftShoulder": "Spine2",
    "LeftArm": "LeftShoulder",
    "LeftForeArm": "LeftArm",
    "LeftHand": "LeftForeArm",
    "LeftHandThumb1": "LeftHand",
    "LeftHandThumb2": "LeftHandThumb1",
    "LeftHandThumb3": "LeftHandThumb2",
    "LeftInHandIndex": "LeftHand",
    "LeftHandIndex1": "LeftInHandIndex",
    "LeftHandIndex2": "LeftHandIndex1",
    "LeftHandIndex3": "LeftHandIndex2",
    "LeftInHandMiddle": "LeftHand",
    "LeftHandMiddle1": "LeftInHandMiddle",
    "LeftHandMiddle2": "LeftHandMiddle1",
    "LeftHandMiddle3": "LeftHandMiddle2",
    "LeftInHandRing": "LeftHand",
    "LeftHandRing1": "LeftInHandRing",
    "LeftHandRing2": "LeftHandRing1",
    "LeftHandRing3": "LeftHandRing2",
    "LeftInHandPinky": "LeftHand",
    "LeftHandPinky1": "LeftInHandPinky",
    "LeftHandPinky2": "LeftHandPinky1",
    "LeftHandPinky3": "LeftHandPinky2",
}

MOCAP_JOINT_NAMES: tuple[str, ...] = tuple(MOCAP_HIERARCHY.keys())


PNLINK_TO_DEPLOY_MATRIX = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
PNLINK_TO_DEPLOY_QUAT = R.from_matrix(PNLINK_TO_DEPLOY_MATRIX).as_quat(scalar_first=True)


def skeleton_connections(hierarchy: Mapping[str, str] = MOCAP_HIERARCHY) -> tuple[tuple[str, str], ...]:

    return tuple((parent, child) for child, parent in hierarchy.items() if parent != "world")


class PNLinkMocapAdapter:

    def __init__(self, hierarchy: Mapping[str, str] = MOCAP_HIERARCHY) -> None:

        self.hierarchy = dict(hierarchy)
        self.joint_names = self._ordered_joint_names(self.hierarchy)
        self._name_to_index = {name: idx for idx, name in enumerate(self.joint_names)}
        self.parent_indices = self._build_parent_indices()

    @staticmethod
    def _ordered_joint_names(hierarchy: Mapping[str, str]) -> list[str]:

        ordered: list[str] = []
        visited: set[str] = set()

        def visit(name: str) -> None:
            if name in visited:
                return
            parent = hierarchy.get(name, "world")
            if parent != "world":
                visit(parent)
            visited.add(name)
            ordered.append(name)

        for joint_name in hierarchy:
            visit(joint_name)
        return ordered

    def _build_parent_indices(self) -> list[int]:

        parent_indices: list[int] = []
        for joint_name in self.joint_names:
            parent_name = self.hierarchy.get(joint_name, "world")
            parent_indices.append(-1 if parent_name == "world" else self._name_to_index[parent_name])
        return parent_indices

    def joints_to_human_frame(self, joints: MocapJoints) -> HumanFrame:

        local_pos = np.zeros((len(self.joint_names), 3), dtype=np.float64)
        local_quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64), (len(self.joint_names), 1))

        for joint_name, joint_data in joints.items():
            idx = self._name_to_index.get(joint_name)
            if idx is None:
                continue
            if "local_pos" in joint_data:
                local_pos[idx] = np.asarray(joint_data["local_pos"], dtype=np.float64) * 0.01
            if "local_rot" in joint_data:
                quat = np.asarray(joint_data["local_rot"], dtype=np.float64)
                if np.dot(quat, quat) > 1e-12:
                    local_quat[idx] = quat / np.linalg.norm(quat)

        global_pos = np.zeros_like(local_pos)
        global_quat = np.zeros_like(local_quat)
        for idx, parent_idx in enumerate(self.parent_indices):
            if parent_idx < 0:
                global_pos[idx] = local_pos[idx]
                global_quat[idx] = local_quat[idx]
                continue
            parent_quat = global_quat[parent_idx]
            global_pos[idx] = global_pos[parent_idx] + _rotate_vector(parent_quat, local_pos[idx])
            global_quat[idx] = quat_mul_np(parent_quat, local_quat[idx], scalar_first=True)

        frame: HumanFrame = {}
        for idx, joint_name in enumerate(self.joint_names):
            pos = global_pos[idx] @ PNLINK_TO_DEPLOY_MATRIX.T
            quat = quat_mul_np(PNLINK_TO_DEPLOY_QUAT, global_quat[idx], scalar_first=True)
            frame[joint_name] = (
                np.asarray(pos, dtype=np.float64),
                np.asarray(quat, dtype=np.float64),
            )

        _add_foot_mod(frame)
        return frame


def add_synthetic_tip_joints(
    joints: MocapJoints,
    *,
    quat_mode: str = "wxyz_identity",
) -> MocapJoints:

    out = {name: dict(value) for name, value in joints.items()}
    quat = [1.0, 0.0, 0.0, 0.0] if quat_mode == "wxyz_identity" else [0.0, 0.0, 0.0, 1.0]
    out.setdefault("Head1", {"local_pos": [0.0, 16.450001, 0.0], "local_rot": quat})
    out.setdefault("LeftTiptoe", {"local_pos": [0.0, -7.85, 14.28], "local_rot": quat})
    out.setdefault("RightTiptoe", {"local_pos": [0.0, -7.85, 14.28], "local_rot": quat})
    return out


def _rotate_vector(quat_wxyz: np.ndarray, vec: np.ndarray) -> np.ndarray:

    pure = np.array([0.0, vec[0], vec[1], vec[2]], dtype=np.float64)
    inv = quat_wxyz * np.array([1.0, -1.0, -1.0, -1.0], dtype=np.float64)
    return quat_mul_np(quat_mul_np(quat_wxyz, pure, scalar_first=True), inv, scalar_first=True)[1:4]


def _add_foot_mod(frame: HumanFrame) -> None:

    if "LeftFoot" in frame:
        frame["LeftFootMod"] = (
            frame["LeftFoot"][0].copy(),
            frame.get("LeftTiptoe", frame["LeftFoot"])[1].copy(),
        )
    if "RightFoot" in frame:
        frame["RightFootMod"] = (
            frame["RightFoot"][0].copy(),
            frame.get("RightTiptoe", frame["RightFoot"])[1].copy(),
        )


__all__ = [
    "MOCAP_HIERARCHY",
    "MOCAP_JOINT_NAMES",
    "PNLINK_TO_DEPLOY_MATRIX",
    "PNLINK_TO_DEPLOY_QUAT",
    "PNLinkMocapAdapter",
    "add_synthetic_tip_joints",
    "skeleton_connections",
]
