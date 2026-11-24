"""JSON wire codec for the pre-retarget PICO stream."""

import json
from typing import Any


PICO_RAW_TOPIC = b"pico_rt.raw"
PICO_RAW_VERSION = 1


def encode_pico_raw(packet: dict[str, Any]) -> bytes:
    return json.dumps(
        {"pico_raw_version": PICO_RAW_VERSION, **packet},
        separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def decode_pico_raw(payload: bytes) -> dict[str, Any]:
    packet = json.loads(payload)
    if not isinstance(packet, dict) or packet.get("pico_raw_version") != PICO_RAW_VERSION:
        raise ValueError("unsupported PICO raw packet version")
    if packet.get("kind") not in {"metadata", "sdk", "body", "control"}:
        raise ValueError("unsupported PICO raw packet kind")
    if not isinstance(packet.get("stream_id"), str) or not packet["stream_id"]:
        raise ValueError("missing PICO raw stream ID")
    for key, minimum in (("packet_seq", 1), ("source_drops", 0)):
        value = packet.get(key)
        if type(value) is not int or value < minimum:
            raise ValueError(f"invalid PICO raw {key}")
    return packet


