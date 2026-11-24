"""Small versioned msgpack protocol for simulation snapshots."""
from __future__ import annotations

from typing import Any

import msgpack
import numpy as np

VERSION = 1
TOPIC = b"webviz.sim"


def _pack_array(value: np.ndarray | None) -> dict[str, Any] | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    return {"shape": list(arr.shape), "data": arr.tobytes()}


def _unpack_array(value: dict[str, Any] | None) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.frombuffer(value["data"], dtype=np.float32)
    return arr.reshape(tuple(int(v) for v in value["shape"])).copy()


def encode_sim_snapshot(
    *, seq: int, sample_ns: int, qpos: np.ndarray,
    reference_qpos: np.ndarray | None = None,
) -> bytes:
    return msgpack.packb({
        "version": VERSION,
        "seq": int(seq),
        "sample_ns": int(sample_ns),
        "qpos": _pack_array(qpos),
        "reference_qpos": _pack_array(reference_qpos),
    }, use_bin_type=True)


def decode_sim_snapshot(payload: bytes) -> dict[str, Any]:
    data = msgpack.unpackb(payload, raw=False)
    if int(data.get("version", -1)) != VERSION:
        raise ValueError(f"unsupported web visualization version: {data.get('version')}")
    return {
        "version": VERSION,
        "seq": int(data["seq"]),
        "sample_ns": int(data["sample_ns"]),
        "qpos": _unpack_array(data["qpos"]),
        "reference_qpos": _unpack_array(data.get("reference_qpos")),
    }
