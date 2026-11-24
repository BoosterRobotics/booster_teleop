
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from booster_deploy.types import HumanFrame
from booster_deploy.utils.isaaclab.math import quat_mul_np


INPUT_TO_DEPLOY_MATRIX = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
INPUT_TO_DEPLOY_QUAT = np.array([np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0], dtype=np.float64)

BODY_JOINT_NAMES = [
    "Pelvis", "Left_Hip", "Right_Hip", "Spine1", "Left_Knee", "Right_Knee",
    "Spine2", "Left_Ankle", "Right_Ankle", "Spine3", "Left_Foot", "Right_Foot",
    "Neck", "Left_Collar", "Right_Collar", "Head", "Left_Shoulder", "Right_Shoulder",
    "Left_Elbow", "Right_Elbow", "Left_Wrist", "Right_Wrist", "Left_Hand", "Right_Hand",
]

BODY_JOINT_PARENTS = np.array(
    [
        -1,
        0, 0, 0, 1, 2,
        3, 4, 5, 6, 7, 8,
        9, 12, 12, 12, 13, 14,
        16, 17, 18, 19, 20, 21,
    ],
    dtype=np.int32,
)


def transform_xrobot_frame(frame: HumanFrame) -> HumanFrame:

    transformed: HumanFrame = {}
    for body_name, (pos, quat) in frame.items():
        orientation = quat_mul_np(
            INPUT_TO_DEPLOY_QUAT,
            np.asarray(quat, dtype=np.float64),
            scalar_first=True,
        )
        position = np.asarray(pos, dtype=np.float64) @ INPUT_TO_DEPLOY_MATRIX.T
        transformed[body_name] = (position, np.asarray(orientation, dtype=np.float64))
    return transformed


def body_poses_to_frame(body_poses: NDArray[np.float64]) -> HumanFrame:

    raw: HumanFrame = {}
    for i, joint_name in enumerate(BODY_JOINT_NAMES):
        pos = np.asarray([body_poses[i][0], body_poses[i][1], body_poses[i][2]], dtype=np.float64)
        quat_wxyz = np.asarray([body_poses[i][6], body_poses[i][3], body_poses[i][4], body_poses[i][5]], dtype=np.float64)
        raw[joint_name] = (pos, quat_wxyz)
    return transform_xrobot_frame(raw)


def body_poses_to_human_frame(body_poses):

    return body_poses_to_frame(body_poses)


def pico_skeleton_connections() -> tuple[tuple[str, str], ...]:

    pairs: list[tuple[str, str]] = []
    for idx, parent in enumerate(BODY_JOINT_PARENTS):
        if parent >= 0:
            pairs.append((BODY_JOINT_NAMES[int(parent)], BODY_JOINT_NAMES[idx]))
    return tuple(pairs)


__all__ = [
    "BODY_JOINT_NAMES",
    "BODY_JOINT_PARENTS",
    "INPUT_TO_DEPLOY_MATRIX",
    "INPUT_TO_DEPLOY_QUAT",
    "body_poses_to_human_frame",
    "pico_skeleton_connections",
    "transform_xrobot_frame",
]
