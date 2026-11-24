
from __future__ import annotations

import sys
from time import perf_counter
from typing import TextIO


def format_eta(seconds: float) -> str:

    seconds_i = int(max(0.0, seconds))
    h = seconds_i // 3600
    m = (seconds_i % 3600) // 60
    s = seconds_i % 60
    if h > 0:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def print_progress(
    label: str,
    step: int,
    total: int,
    t0: float,
    *,
    width: int = 36,
    stream: TextIO | None = None,
) -> None:
    if total <= 0:
        return
    if stream is None:
        stream = sys.stdout

    step_c = max(0, min(int(step), int(total)))
    total_i = int(total)
    filled = int(width * step_c / total_i)
    bar = "█" * filled + "-" * (width - filled)
    pct = 100.0 * step_c / total_i

    eta_str = "--:--"
    if step_c > 0:
        elapsed = perf_counter() - t0
        sec_per_step = elapsed / step_c
        remaining = (total_i - step_c) * sec_per_step
        eta_str = format_eta(remaining)

    stream.write(f"\r[{label}] [{bar}] {step_c}/{total_i} {pct:5.1f}%  ETA {eta_str}")
    stream.flush()


def clear_progress(*, stream: TextIO | None = None) -> None:
    if stream is None:
        stream = sys.stdout
    stream.write("\r\033[2K\r")
    stream.flush()

