"""Minimal server-to-browser WebSocket helpers (RFC 6455)."""
from __future__ import annotations

import base64
import hashlib
import struct

_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


def accept_value(client_key: str) -> str:
    raw = hashlib.sha1((client_key.strip() + _GUID).encode("ascii")).digest()
    return base64.b64encode(raw).decode("ascii")


def encode_text_frame(payload: bytes) -> bytes:
    """Encode an unmasked final text frame; servers must not mask frames."""
    size = len(payload)
    if size < 126:
        head = bytes((0x81, size))
    elif size <= 0xFFFF:
        head = bytes((0x81, 126)) + struct.pack("!H", size)
    else:
        head = bytes((0x81, 127)) + struct.pack("!Q", size)
    return head + payload
