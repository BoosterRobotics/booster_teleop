
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class HandRetargetCommandNs:

    hand_side: str
    robot_name: str
    joint_names: tuple[str, ...]
    joint_pos: NDArray[np.float64]
    qpos: NDArray[np.float64]
    loss: float | None = None


@dataclass(frozen=True, slots=True)
class HandRetargetFrameNs:

    body_seq: int
    body_recv_ns: int
    retarget_recv_ns: int
    retarget_done_ns: int
    human_format: str
    commands: tuple[HandRetargetCommandNs, ...]


class HandRetargetFrameProtocol(Protocol):

    body_seq: int
    body_recv_ns: int
    retarget_recv_ns: int
    retarget_done_ns: int
    human_format: str
    commands: tuple[HandRetargetCommandNs, ...]


__all__ = [
    "HandRetargetCommandNs",
    "HandRetargetFrameNs",
    "HandRetargetFrameProtocol",
]
