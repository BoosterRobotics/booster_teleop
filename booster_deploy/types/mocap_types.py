
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from booster_deploy.types.human_frame import HumanFrame


MocapJoints = dict[str, dict[str, Any]]


@dataclass(frozen=True, slots=True)
class MocapBodyRawFrameNs:

    seq: int
    body_recv_ns: int
    source_ns: int | None
    raw_joints: MocapJoints
    stream_id: str | None = None
    source_epoch: int | None = None
    clock_domain: str | None = None
    body_unique_known: bool = False


@dataclass(frozen=True, slots=True)
class MocapRetargetInputFrameNs:

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
    source_epoch: int | None = None
    clock_domain: str | None = None
    source_ns: int | None = None
    body_unique_known: bool = False

    @property
    def seq(self) -> int:

        return self.body_seq

    @property
    def recv_ns(self) -> int:

        return self.body_recv_ns


__all__ = [
    "MocapBodyRawFrameNs",
    "MocapJoints",
    "MocapRetargetInputFrameNs",
]
