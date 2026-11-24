
from __future__ import annotations

import sys
from dataclasses import dataclass
from time import perf_counter
from typing import Sequence, TextIO


@dataclass(frozen=True)
class PanelSection:

    title: str
    lines: Sequence[str]


class LiveTerminalPanel:

    def __init__(
        self,
        *,
        refresh_hz: float = 10.0,
        stream: TextIO | None = None,
        enabled: bool | None = None,
    ) -> None:

        self._stream = stream if stream is not None else sys.stdout
        self._refresh_interval = 1.0 / max(float(refresh_hz), 1e-6)
        self._enabled = self._stream.isatty() if enabled is None else bool(enabled) and self._stream.isatty()
        self._last_render_time = 0.0
        self._printed_lines = 0

    def render(
        self,
        title: str,
        sections: Sequence[PanelSection],
        *,
        force: bool = False,
    ) -> None:

        if not self._enabled:
            return

        now = perf_counter()
        if not force and now - self._last_render_time < self._refresh_interval:
            return

        lines = self._format_lines(title, sections)
        if self._printed_lines > 0:
            self._stream.write(f"\x1b[{self._printed_lines}F\x1b[J")

        self._stream.write("\n".join(lines))
        self._stream.write("\n")
        self._stream.flush()
        self._printed_lines = len(lines)
        self._last_render_time = now

    def close(self) -> None:

        if not self._enabled or self._printed_lines <= 0:
            return
        self._stream.write("\n")
        self._stream.flush()
        self._printed_lines = 0

    @staticmethod
    def _format_lines(title: str, sections: Sequence[PanelSection]) -> list[str]:

        width = max(64, len(title) + 4)
        lines = [f" {title} ".center(width, "=")]
        for section in sections:
            if not section.lines:
                continue
            lines.append(f"[{section.title}]")
            lines.extend(section.lines)
        return lines
