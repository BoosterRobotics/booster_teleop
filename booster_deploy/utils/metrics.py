
from __future__ import annotations

import time
import logging

import numpy as np

from .synced_array import SyncedArray

logger = logging.getLogger("booster_metrics")


class SyncedMetrics:

    def __init__(self, name: str, max_events: int = 10000):

        self.name = name
        self.max_events = int(max_events)

        self._arr = SyncedArray(
            f"metric_{name}",
            shape=(self.max_events + 2,),
            dtype="float64",
        )

    def mark(self) -> None:
        def _updater(buf: np.ndarray):


            write_pos = int(buf[0])
            total = int(buf[1])
            buf[2 + write_pos] = time.perf_counter()
            write_pos = (write_pos + 1) % self.max_events
            buf[0] = float(write_pos)
            buf[1] = float(total + 1)

        self._arr.modify_in_place(_updater)

    def compute(self):
        data = self._arr.read()
        write_pos = int(data[0])
        total = int(data[1])
        if total < 2:
            return {
                "count": int(total),
                "freq_hz": 0.0,
                "mean_period_s": None,
                "min_period_s": None,
                "max_period_s": None,
            }

        if total < self.max_events:
            ts = data[2:2 + total]
        else:

            if write_pos == 0:
                ts = data[2:2 + self.max_events]
            else:
                part1 = data[2 + write_pos:2 + self.max_events]
                part2 = data[2:2 + write_pos]
                ts = np.concatenate([part1, part2])

        periods = np.diff(ts)
        mean_p = float(np.mean(periods))
        return {
            "count": int(min(total, self.max_events)),
            "freq_hz": 1.0 / mean_p if mean_p > 0 else float("inf"),
            "mean_period_s": mean_p,
            "min_period_s": float(np.min(periods)),
            "max_period_s": float(np.max(periods)),
        }

    def cleanup(self) -> None:

        try:
            self._arr.cleanup()
        except Exception:
            pass


__all__ = ["SyncedMetrics"]
