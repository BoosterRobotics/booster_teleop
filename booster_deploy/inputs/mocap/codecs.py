
from __future__ import annotations

from typing import Any

import numpy as np

from booster_deploy.types import HumanFrame
from booster_deploy.types.mocap_types import MocapRetargetInputFrameNs


def encode_mocap_retarget_input(frame: MocapRetargetInputFrameNs) -> bytes:

    import msgpack

    payload: dict[str, Any] = {
        "stream_id": frame.stream_id,
        "source_epoch": frame.source_epoch,
        "clock_domain": frame.clock_domain,
        "source_ns": frame.source_ns,
        "body_unique_known": frame.body_unique_known,
        "body_seq": int(frame.body_seq),
        "source_ids": frame.source_ids,
        "native_index": frame.native_index, "loop": frame.loop,
        "body_recv_ns": int(frame.body_recv_ns),
        "offline_sample_ns": frame.offline_sample_ns,
        "offline_cycle": frame.offline_cycle,
        "convert_done_ns": int(frame.convert_done_ns),
        "human_format": str(frame.human_format),
        "human_frame": _encode_human_frame(frame.human_frame),
    }
    return msgpack.packb(payload, use_bin_type=True)


def decode_mocap_retarget_input(payload: bytes) -> MocapRetargetInputFrameNs:

    import msgpack

    raw = msgpack.unpackb(payload, raw=False)
    return MocapRetargetInputFrameNs(
        stream_id=raw.get("stream_id"),
        source_epoch=raw.get("source_epoch"),
        clock_domain=raw.get("clock_domain"),
        source_ns=raw.get("source_ns"),
        body_unique_known=bool(raw.get("body_unique_known", False)),
        body_seq=int(raw["body_seq"]),
        source_ids=tuple(tuple(x) for x in raw.get("source_ids", ())),
        offline_sample_ns=raw.get("offline_sample_ns"),
        offline_cycle=int(raw.get("offline_cycle",0)),
        native_index=raw.get("native_index"), loop=raw.get("loop"),
        body_recv_ns=int(raw.get("body_recv_ns", 0)),
        convert_done_ns=int(raw.get("convert_done_ns", 0)),
        human_format=str(raw.get("human_format", "")),
        human_frame=_decode_human_frame(raw.get("human_frame", {}) or {}),
    )


def _encode_human_frame(frame: HumanFrame) -> dict[str, list[list[float]]]:

    payload: dict[str, list[list[float]]] = {}
    for name, (pos, quat) in frame.items():
        payload[str(name)] = [
            np.asarray(pos, dtype=np.float64).tolist(),
            np.asarray(quat, dtype=np.float64).tolist(),
        ]
    return payload


def _decode_human_frame(payload: dict[str, Any]) -> HumanFrame:

    frame: HumanFrame = {}
    for name, value in payload.items():
        pos, quat = value
        frame[str(name)] = (
            np.asarray(pos, dtype=np.float64),
            np.asarray(quat, dtype=np.float64),
        )
    return frame


__all__ = [
    "decode_mocap_retarget_input",
    "encode_mocap_retarget_input",
]
