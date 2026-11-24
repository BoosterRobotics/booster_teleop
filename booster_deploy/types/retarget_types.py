
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from booster_deploy.types.human_frame import HumanFrame
from booster_deploy.types.hand_retarget_types import HandRetargetFrameNs


class RetargetInputFrameProtocol(Protocol):

    seq: int
    recv_ns: int
    human_format: str
    human_frame: HumanFrame


@dataclass(frozen=True, slots=True)
class RetargetQposFrameNs:

    body_seq: int
    body_recv_ns: int
    retarget_recv_ns: int
    retarget_done_ns: int
    robot_name: str
    robot_qpos: NDArray[np.float64]
    source_ids: tuple = ()
    offline_sample_ns: int | None = None
    offline_cycle: int = 0
    clock_domain: str | None = None


@dataclass(frozen=True, slots=True)
class ReferenceFrame50HzNs:

    sample_idx: int
    sample_ns: int
    body_seq: int
    body_recv_ns: int
    retarget_done_ns: int
    robot_name: str
    joint_pos: NDArray[np.float64]
    joint_vel: NDArray[np.float64]
    root_pos: NDArray[np.float64]
    root_quat: NDArray[np.float64]
    identity: dict | None = None
    human_format: str | None = None
    human_frame: HumanFrame | None = None
    hand_frame: HandRetargetFrameNs | None = None
    source_ids: tuple = ()
    offline_sample_ns: int | None = None
    offline_cycle: int = 0
    clock_domain: str | None = None


class ReferenceFrame50HzProtocol(Protocol):

    sample_idx: int
    sample_ns: int
    robot_name: str
    joint_pos: NDArray[np.float64]
    joint_vel: NDArray[np.float64]
    root_pos: NDArray[np.float64]
    root_quat: NDArray[np.float64]
