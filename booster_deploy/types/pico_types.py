
from __future__ import annotations

from dataclasses import dataclass
from enum import IntFlag

import numpy as np
from numpy.typing import NDArray

from booster_deploy.types.human_frame import HumanFrame


class PicoButtonBits(IntFlag):

    NONE = 0
    X = 1 << 0
    Y = 1 << 1
    A = 1 << 2
    B = 1 << 3
    LEFT_AXIS_CLICK = 1 << 4
    RIGHT_AXIS_CLICK = 1 << 5
    LEFT_MENU = 1 << 6
    RIGHT_MENU = 1 << 7


@dataclass(frozen=True, slots=True)
class PicoBodyRawFrameNs:

    seq: int
    body_recv_ns: int
    source_ns: int | None
    poses: NDArray[np.float64]
    publish_ns: int | None = None
    sdk_recv_ns: int | None = None
    stream_id: str | None = None
    clock_domain: str | None = None
    source_ids: tuple = ()
    offline_sample_ns: int | None = None
    offline_cycle: int = 0
    identity: dict | None = None


@dataclass(frozen=True, slots=True)
class PicoRetargetInputFrameNs:

    body_seq: int
    body_recv_ns: int
    convert_done_ns: int
    human_format: str
    human_frame: HumanFrame
    source_ids: tuple = ()
    offline_sample_ns: int | None = None
    offline_cycle: int = 0
    native_index: int | None = None
    loop: int | None = None
    stream_id: str | None = None
    clock_domain: str | None = None

    @property
    def seq(self) -> int:

        return self.body_seq

    @property
    def recv_ns(self) -> int:

        return self.body_recv_ns


@dataclass(frozen=True, slots=True)
class PicoControlFrameNs:

    seq: int
    recv_ns: int
    source_ns: int | None
    buttons: PicoButtonBits
    left_axis: tuple[float, float]
    right_axis: tuple[float, float]
    left_trigger: float = 0.0
    right_trigger: float = 0.0
    left_grip: float = 0.0
    right_grip: float = 0.0
