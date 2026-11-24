"""Thread-safe latest-only frame store for browser visualization."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class _Entry:
    value: dict[str, Any]
    at: float


class LatestFrameStore:
    """Keep one frame per stream; slow clients never create a backlog."""

    def __init__(self, *, stale_after_s: float = 0.5) -> None:
        self.stale_after_s = float(stale_after_s)
        self._lock = threading.Lock()
        self._entries: dict[str, _Entry] = {}

    def publish(self, stream: str, frame: dict[str, Any], *, at: float | None = None) -> None:
        stamp = time.monotonic() if at is None else float(at)
        with self._lock:
            self._entries[str(stream)] = _Entry(dict(frame), stamp)

    def discard(self, *streams: str) -> None:
        with self._lock:
            for stream in streams:
                self._entries.pop(stream, None)

    def snapshot(self, *, now: float | None = None) -> dict[str, Any]:
        stamp = time.monotonic() if now is None else float(now)
        with self._lock:
            entries = dict(self._entries)
        streams: dict[str, dict[str, Any]] = {}
        status: dict[str, dict[str, Any]] = {}
        for name, entry in entries.items():
            age = max(0.0, stamp - entry.at)
            streams[name] = entry.value
            status[name] = {"age_s": age, "stale": age > self.stale_after_s, "received_mono_s": entry.at}
        return {"version": 1, "streams": streams, "status": status}
