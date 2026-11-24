
from __future__ import annotations

from collections.abc import Callable
from time import perf_counter, sleep


class PeriodicDeadline:

    def __init__(
        self,
        period_s: float,
        *,
        clock: Callable[[], float] = perf_counter,
        sleeper: Callable[[float], None] = sleep,
    ) -> None:
        period_s = float(period_s)
        if period_s <= 0.0:
            raise ValueError(f"period_s must be positive, got {period_s}")
        self.period_s = period_s
        self._clock = clock
        self._sleeper = sleeper
        self._next_start_s = clock()
        self._deadline_s: float | None = None

    def wait(self) -> float:

        now = self._clock()
        if now < self._next_start_s:
            self._sleeper(self._next_start_s - now)
        started_s = self._clock()
        self._deadline_s = self._next_start_s + self.period_s
        return started_s

    def complete(self) -> float:

        if self._deadline_s is None:
            raise RuntimeError("PeriodicDeadline.complete() called before wait()")
        finished_s = self._clock()
        lateness_s = max(finished_s - self._deadline_s, 0.0)
        self._next_start_s = finished_s if lateness_s > 0.0 else self._deadline_s
        self._deadline_s = None
        return lateness_s
