
from __future__ import annotations

from typing import Any

import numpy as np

from booster_deploy.types.hand_retarget_types import (
    HandRetargetCommandNs,
    HandRetargetFrameNs,
)


def encode_hand_retarget_frame(frame: HandRetargetFrameNs) -> bytes:

    import msgpack

    payload: dict[str, Any] = {
        "body_seq": int(frame.body_seq),
        "body_recv_ns": int(frame.body_recv_ns),
        "retarget_recv_ns": int(frame.retarget_recv_ns),
        "retarget_done_ns": int(frame.retarget_done_ns),
        "human_format": str(frame.human_format),
        "commands": [_encode_command(command) for command in frame.commands],
    }
    return msgpack.packb(payload, use_bin_type=True)


def decode_hand_retarget_frame(payload: bytes) -> HandRetargetFrameNs:

    import msgpack

    raw = msgpack.unpackb(payload, raw=False)
    return HandRetargetFrameNs(
        body_seq=int(raw.get("body_seq", 0)),
        body_recv_ns=int(raw.get("body_recv_ns", 0)),
        retarget_recv_ns=int(raw.get("retarget_recv_ns", 0)),
        retarget_done_ns=int(raw.get("retarget_done_ns", 0)),
        human_format=str(raw.get("human_format", "")),
        commands=tuple(_decode_command(item) for item in raw.get("commands", []) or []),
    )


def _encode_command(command: HandRetargetCommandNs) -> dict[str, Any]:
    return {
        "hand_side": str(command.hand_side),
        "robot_name": str(command.robot_name),
        "joint_names": [str(name) for name in command.joint_names],
        "joint_pos": np.asarray(command.joint_pos, dtype=np.float64).tolist(),
        "qpos": np.asarray(command.qpos, dtype=np.float64).tolist(),
        "loss": None if command.loss is None else float(command.loss),
    }


def _decode_command(payload: dict[str, Any]) -> HandRetargetCommandNs:
    return HandRetargetCommandNs(
        hand_side=str(payload.get("hand_side", "")),
        robot_name=str(payload.get("robot_name", "")),
        joint_names=tuple(str(name) for name in payload.get("joint_names", []) or []),
        joint_pos=np.asarray(payload.get("joint_pos", []), dtype=np.float64),
        qpos=np.asarray(payload.get("qpos", []), dtype=np.float64),
        loss=None if payload.get("loss") is None else float(payload["loss"]),
    )


__all__ = [
    "decode_hand_retarget_frame",
    "encode_hand_retarget_frame",
]
