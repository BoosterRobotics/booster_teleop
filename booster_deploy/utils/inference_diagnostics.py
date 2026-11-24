
from __future__ import annotations

from collections import deque
from collections.abc import Callable
import json
import logging
import os
from pathlib import Path
import threading
import time
from typing import Any

from booster_deploy.utils.project_path import resolve_project_path


MONITOR_PATH_ENV = "BOOSTER_RT_MONITOR_PATH"
logger = logging.getLogger("booster_deploy")


class InferenceDiagnostics:

    def __init__(
        self,
        *,
        output_path: str | os.PathLike[str] | None = None,
        window_s: float = 5.0,
        publish_hz: float = 2.0,
        clock_ns: Callable[[], int] = time.perf_counter_ns,
    ) -> None:
        self.output_path = resolve_project_path(output_path) if output_path else None
        self.window_s = max(float(window_s), 0.1)
        self.publish_hz = max(float(publish_hz), 0.1)
        self._clock_ns = clock_ns
        self._lock = threading.Lock()
        self._inference_times_ns: deque[int] = deque()
        self._continuity_events: deque[tuple[int, str, int]] = deque()
        self._previous_base_idx: int | None = None
        self._latest_base_idx: int | None = None
        self._latest_delta: int | None = None
        self._inference_total = 0
        self._continuous_total = 0
        self._hold_total = 0
        self._skip_total = 0
        self._skipped_frames_total = 0
        self._running = True
        self._closed = False
        self._stop_event = threading.Event()
        self._publisher_thread: threading.Thread | None = None

        if self.output_path is not None:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            self._write_snapshot()
            self._publisher_thread = threading.Thread(
                target=self._publish_loop,
                name="inference-diagnostics-publisher",
                daemon=True,
            )
            self._publisher_thread.start()

    @classmethod
    def from_environment(cls) -> "InferenceDiagnostics":

        output_path = os.getenv(MONITOR_PATH_ENV) or None
        try:
            return cls(output_path=output_path)
        except OSError as exc:
            logger.warning("RT inference monitor disabled: %s", exc)
            return cls()

    def reset_continuity(self) -> None:

        with self._lock:
            self._previous_base_idx = None
            self._latest_delta = None

    def record_success(
        self,
        base_idx: int | None,
        *,
        track_continuity: bool,
    ) -> None:

        now_ns = int(self._clock_ns())
        with self._lock:
            self._inference_total += 1
            self._inference_times_ns.append(now_ns)
            self._latest_base_idx = None if base_idx is None else int(base_idx)

            if not track_continuity or base_idx is None:
                self._previous_base_idx = None
                self._latest_delta = None
                self._prune_locked(now_ns)
                return

            current_idx = int(base_idx)
            previous_idx = self._previous_base_idx
            self._previous_base_idx = current_idx
            if previous_idx is None:
                self._latest_delta = None
                self._prune_locked(now_ns)
                return

            delta = current_idx - previous_idx
            if delta < 0:


                self._latest_delta = None
                self._prune_locked(now_ns)
                return

            self._latest_delta = delta
            if delta == 0:
                kind = "hold"
                skipped_frames = 0
                self._hold_total += 1
            elif delta == 1:
                kind = "continuous"
                skipped_frames = 0
                self._continuous_total += 1
            else:
                kind = "skip"
                skipped_frames = delta - 1
                self._skip_total += 1
                self._skipped_frames_total += skipped_frames
            self._continuity_events.append((now_ns, kind, skipped_frames))
            self._prune_locked(now_ns)

    def snapshot(self) -> dict[str, Any]:

        now_ns = int(self._clock_ns())
        with self._lock:
            self._prune_locked(now_ns)
            times = tuple(self._inference_times_ns)
            events = tuple(self._continuity_events)
            if len(times) >= 2 and times[-1] > times[0]:
                inference_hz = (len(times) - 1) * 1e9 / (times[-1] - times[0])
            else:
                inference_hz = 0.0

            rolling_continuous = sum(kind == "continuous" for _, kind, _ in events)
            rolling_hold = sum(kind == "hold" for _, kind, _ in events)
            rolling_skip = sum(kind == "skip" for _, kind, _ in events)
            rolling_skipped_frames = sum(skipped for _, _, skipped in events)
            rolling_transitions = rolling_continuous + rolling_hold + rolling_skip
            total_transitions = self._continuous_total + self._hold_total + self._skip_total

            return {
                "schema_version": 1,
                "pid": os.getpid(),
                "running": self._running,
                "updated_monotonic_ns": now_ns,
                "window_s": self.window_s,
                "inference": {
                    "hz": inference_hz,
                    "count_window": len(times),
                    "total": self._inference_total,
                    "base_idx": self._latest_base_idx,
                    "delta": self._latest_delta,
                },
                "continuity": {
                    "rolling": self._continuity_payload(
                        transitions=rolling_transitions,
                        continuous=rolling_continuous,
                        hold=rolling_hold,
                        skip=rolling_skip,
                        skipped_frames=rolling_skipped_frames,
                    ),
                    "total": self._continuity_payload(
                        transitions=total_transitions,
                        continuous=self._continuous_total,
                        hold=self._hold_total,
                        skip=self._skip_total,
                        skipped_frames=self._skipped_frames_total,
                    ),
                },
            }

    def close(self) -> None:

        if self._closed:
            return
        self._closed = True
        with self._lock:
            self._running = False
        self._stop_event.set()
        if self._publisher_thread is not None:
            self._publisher_thread.join(timeout=1.0)
            self._publisher_thread = None
        if self.output_path is not None:
            try:
                self._write_snapshot()
            except OSError:
                pass

    @staticmethod
    def _continuity_payload(
        *,
        transitions: int,
        continuous: int,
        hold: int,
        skip: int,
        skipped_frames: int,
    ) -> dict[str, float | int]:
        denominator = max(int(transitions), 1)
        return {
            "transitions": int(transitions),
            "continuous": int(continuous),
            "hold": int(hold),
            "hold_rate": float(hold) / denominator if transitions else 0.0,
            "skip": int(skip),
            "skip_rate": float(skip) / denominator if transitions else 0.0,
            "skipped_frames": int(skipped_frames),
        }

    def _prune_locked(self, now_ns: int) -> None:
        cutoff_ns = now_ns - int(self.window_s * 1e9)
        while self._inference_times_ns and self._inference_times_ns[0] < cutoff_ns:
            self._inference_times_ns.popleft()
        while self._continuity_events and self._continuity_events[0][0] < cutoff_ns:
            self._continuity_events.popleft()

    def _publish_loop(self) -> None:
        interval_s = 1.0 / self.publish_hz
        while not self._stop_event.wait(interval_s):
            try:
                self._write_snapshot()
            except OSError:

                continue

    def _write_snapshot(self) -> None:
        if self.output_path is None:
            return
        payload = self.snapshot()
        temp_path = self.output_path.with_name(
            f".{self.output_path.name}.{os.getpid()}.tmp"
        )
        try:
            with temp_path.open("w", encoding="utf-8") as stream:
                json.dump(payload, stream, separators=(",", ":"))
            os.replace(temp_path, self.output_path)
        finally:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass


__all__ = ["InferenceDiagnostics", "MONITOR_PATH_ENV"]
