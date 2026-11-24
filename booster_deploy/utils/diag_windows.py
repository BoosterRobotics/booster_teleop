"""Sliding-window statistics for the per-pane diagnostic records (v2).

Design contract (2026-10-27, revised with the operator):

* Every stage of the teleop chain is a **serial link on one machine**, so all
  timing is ``time.monotonic_ns()``. Wall clock is a human label only and never
  enters a latency computation.
* Three orthogonal questions, three orthogonal metrics:
  - *is it keeping up?*  -> ``RateWindow.hz`` over the last second
  - *is it stuttering?*  -> inter-event gap p95/max + spike count + ``stall_ms``
  - *is it losing frames?* -> sequence-gap counter (``SeqTracker``), never a
    time-based estimate (upstream rates legitimately vary, e.g. PICO 50-90 Hz)
* Medians/means of inter-event gaps are deliberately **not** reported: they stay
  healthy while the stream stalls, so they carry no diagnostic information.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass


def _quantile(sorted_values: list[float], fraction: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = fraction * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


@dataclass(frozen=True)
class LatencyStats:
    """Quantile summary of one latency series (milliseconds)."""

    count: int
    p50: float
    p95: float
    maximum: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "count": int(self.count),
            "p50": round(self.p50, 3),
            "p95": round(self.p95, 3),
            "max": round(self.maximum, 3),
        }


@dataclass(frozen=True)
class GapStats:
    """Inter-event spacing summary: only the tail and the spike count matter."""

    count: int
    p95: float
    maximum: float
    spikes: int
    long: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "count": int(self.count),
            "p95": round(self.p95, 3),
            "max": round(self.maximum, 3),
            "spikes": int(self.spikes),
            "long": int(self.long),
        }


EMPTY_LATENCY = LatencyStats(0, 0.0, 0.0, 0.0)
EMPTY_GAP = GapStats(0, 0.0, 0.0, 0, 0)


class RateWindow:
    """Event counter that reports the rate over the trailing window."""

    def __init__(self, window_s: float = 1.0) -> None:
        self.window_ns = max(int(float(window_s) * 1e9), 1)
        self._times: deque[int] = deque()
        self.total = 0

    def mark(self, now_ns: int) -> None:
        now_ns = int(now_ns)
        self._times.append(now_ns)
        self.total += 1
        self._prune(now_ns)

    def _prune(self, now_ns: int) -> None:
        cutoff = int(now_ns) - self.window_ns
        times = self._times
        while times and times[0] < cutoff:
            times.popleft()

    def hz(self, now_ns: int) -> float:
        self._prune(now_ns)
        n = len(self._times)
        if n == 0:
            return 0.0
        if n >= 3:
            span_ns = self._times[-1] - self._times[0]
            if span_ns > 0:
                return (n - 1) * 1e9 / span_ns
        return n / (self.window_ns / 1e9)

    def count(self, now_ns: int) -> int:
        self._prune(now_ns)
        return len(self._times)


class IntervalWindow:
    """Event timestamps: rate over 1 s, gap tail over 5 s, stall since last event."""

    def __init__(self, rate_window_s: float = 1.0, gap_window_s: float = 5.0,
                 spike_factor: float = 3.0) -> None:
        self.rate = RateWindow(rate_window_s)
        self.gap_window_ns = max(int(float(gap_window_s) * 1e9), 1)
        self.spike_factor = float(spike_factor)
        self._marks: deque[int] = deque()
        self._gaps: deque[tuple[int, float]] = deque()  # (stamp, gap_ms)
        self.last_mark_ns: int | None = None

    def mark(self, now_ns: int) -> None:
        now_ns = int(now_ns)
        if self.last_mark_ns is not None:
            delta_ns = now_ns - self.last_mark_ns
            # A silence longer than the gap window is a session boundary (clip
            # start/stop, leaving Tracking), not a stutter: ``stall_ms`` still
            # reports how long the pane was idle.
            if delta_ns <= self.gap_window_ns:
                self._gaps.append((now_ns, delta_ns / 1e6))
        self.last_mark_ns = now_ns
        self.rate.mark(now_ns)
        self._prune(now_ns)

    def _prune(self, now_ns: int) -> None:
        cutoff = int(now_ns) - self.gap_window_ns
        gaps = self._gaps
        while gaps and gaps[0][0] < cutoff:
            gaps.popleft()

    def reset(self) -> None:
        """Forget history: a new clip/session must not inherit the idle gap."""

        self._gaps.clear()
        self.last_mark_ns = None

    @property
    def total(self) -> int:
        return self.rate.total

    def hz(self, now_ns: int) -> float:
        return self.rate.hz(now_ns)

    def stall_ms(self, now_ns: int) -> float | None:
        if self.last_mark_ns is None:
            return None
        return max((int(now_ns) - self.last_mark_ns) / 1e6, 0.0)

    def gap_stats(self, now_ns: int) -> GapStats:
        self._prune(now_ns)
        values = sorted(gap for _, gap in self._gaps)
        if not values:
            return EMPTY_GAP
        median = _quantile(values, 0.5)
        threshold = self.spike_factor * median if median > 0.0 else 0.0
        spikes = sum(1 for value in values if threshold and value > threshold)
        long_gaps = sum(1 for value in values if median > 0.0 and value > 1.5 * median)
        return GapStats(
            count=len(values),
            p95=_quantile(values, 0.95),
            maximum=float(values[-1]),
            spikes=spikes,
            long=long_gaps,
        )


class SampleWindow:
    """Numeric series (latency / work time) with quantiles over a 5 s window."""

    def __init__(self, window_s: float = 5.0) -> None:
        self.window_ns = max(int(float(window_s) * 1e9), 1)
        self._values: deque[tuple[int, float]] = deque()

    def add(self, value: float, now_ns: int) -> None:
        now_ns = int(now_ns)
        self._values.append((now_ns, float(value)))
        self._prune(now_ns)

    def _prune(self, now_ns: int) -> None:
        cutoff = int(now_ns) - self.window_ns
        values = self._values
        while values and values[0][0] < cutoff:
            values.popleft()

    def stats(self, now_ns: int) -> LatencyStats:
        self._prune(now_ns)
        values = sorted(value for _, value in self._values)
        if not values:
            return EMPTY_LATENCY
        return LatencyStats(
            count=len(values),
            p50=_quantile(values, 0.5),
            p95=_quantile(values, 0.95),
            maximum=float(values[-1]),
        )


class SeqTracker:
    """Count missing frames from a monotone sequence number.

    A gap is ``seq - last - 1``. This is exact and rate-independent: upstream
    rates may legitimately swing (PICO 50-90 Hz) without being a fault, so a
    time-based "expected vs received" estimate would raise false alarms.
    """

    def __init__(self) -> None:
        self.last: int | None = None
        self.missing = 0
        self.duplicates = 0
        self.resets = 0

    def observe(self, seq: int | None) -> int:
        if seq is None:
            return 0
        seq = int(seq)
        if self.last is None:
            self.last = seq
            return 0
        if seq == self.last:
            self.duplicates += 1
            return 0
        if seq < self.last:
            # Stream restarted (new publisher / replayed clip): not a loss.
            self.resets += 1
            self.last = seq
            return 0
        missing = seq - self.last - 1
        if missing:
            self.missing += missing
        self.last = seq
        return missing

    def as_dict(self) -> dict[str, int]:
        return {"missing": int(self.missing), "dups": int(self.duplicates),
                "resets": int(self.resets)}


__all__ = [
    "EMPTY_GAP",
    "EMPTY_LATENCY",
    "GapStats",
    "IntervalWindow",
    "LatencyStats",
    "RateWindow",
    "SampleWindow",
    "SeqTracker",
]
