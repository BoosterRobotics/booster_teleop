
from __future__ import annotations

from typing import Any

import numpy as np

from booster_deploy.types import HumanFrame

from booster_deploy.types.pico_types import PicoButtonBits, PicoControlFrameNs, PicoRetargetInputFrameNs


def encode_pico_retarget_input(frame: PicoRetargetInputFrameNs) -> bytes:

    import msgpack

    payload: dict[str, Any] = {
        "body_seq": int(frame.body_seq),
        "source_ids": frame.source_ids,
        "native_index": frame.native_index, "loop": frame.loop,
        "stream_id": frame.stream_id,
        "clock_domain": frame.clock_domain,
        "body_recv_ns": int(frame.body_recv_ns),
        "offline_sample_ns": frame.offline_sample_ns,
        "offline_cycle": frame.offline_cycle,
        "convert_done_ns": int(frame.convert_done_ns),
        "human_format": str(frame.human_format),
        "human_frame": _encode_human_frame(frame.human_frame),
    }
    return msgpack.packb(payload, use_bin_type=True)


def decode_pico_retarget_input(payload: bytes) -> PicoRetargetInputFrameNs:

    import msgpack

    raw = msgpack.unpackb(payload, raw=False)
    return PicoRetargetInputFrameNs(
        body_seq=int(raw["body_seq"]),
        source_ids=tuple(tuple(x) for x in raw.get("source_ids", ())),
        offline_sample_ns=raw.get("offline_sample_ns"),
        offline_cycle=int(raw.get("offline_cycle",0)),
        native_index=raw.get("native_index"), loop=raw.get("loop"),
        stream_id=raw.get("stream_id"),
        clock_domain=raw.get("clock_domain"),
        body_recv_ns=int(raw["body_recv_ns"]),
        convert_done_ns=int(raw["convert_done_ns"]),
        human_format=str(raw["human_format"]),
        human_frame=_decode_human_frame(raw["human_frame"]),
    )


def encode_pico_control(frame: PicoControlFrameNs) -> bytes:

    import msgpack

    payload = {
        "seq": int(frame.seq),
        "recv_ns": int(frame.recv_ns),
        "source_ns": frame.source_ns if frame.source_ns is None else int(frame.source_ns),
        "buttons": int(frame.buttons),
        "left_axis": [float(frame.left_axis[0]), float(frame.left_axis[1])],
        "right_axis": [float(frame.right_axis[0]), float(frame.right_axis[1])],
        "left_trigger": float(frame.left_trigger),
        "right_trigger": float(frame.right_trigger),
        "left_grip": float(frame.left_grip),
        "right_grip": float(frame.right_grip),
    }
    return msgpack.packb(payload, use_bin_type=True)


def decode_pico_control(payload: bytes) -> PicoControlFrameNs:

    import msgpack

    raw = msgpack.unpackb(payload, raw=False)
    return PicoControlFrameNs(
        seq=int(raw["seq"]),
        recv_ns=int(raw["recv_ns"]),
        source_ns=None if raw.get("source_ns") is None else int(raw["source_ns"]),
        buttons=PicoButtonBits(int(raw.get("buttons", 0))),
        left_axis=_axis_pair(raw.get("left_axis")),
        right_axis=_axis_pair(raw.get("right_axis")),
        left_trigger=float(raw.get("left_trigger", 0.0)),
        right_trigger=float(raw.get("right_trigger", 0.0)),
        left_grip=float(raw.get("left_grip", 0.0)),
        right_grip=float(raw.get("right_grip", 0.0)),
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


def _axis_pair(raw: object) -> tuple[float, float]:

    if raw in (None, "", "null"):
        return (0.0, 0.0)
    if not isinstance(raw, (list, tuple)) or len(raw) < 2:
        return (0.0, 0.0)
    return (float(raw[0]), float(raw[1]))
