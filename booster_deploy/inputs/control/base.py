
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class RawControllerState:

    axes: dict[str, float] = field(default_factory=dict)
    buttons: frozenset[str] = frozenset()
    source: str = ""
    timestamp_s: float | None = None

    def axis(self, name: str, default: float = 0.0) -> float:

        return float(self.axes.get(name, default))

    def pressed(self, name: str) -> bool:

        return name in self.buttons


class RawInputProvider(Protocol):

    @property
    def source(self) -> str:

        ...

    def get_state(self) -> RawControllerState:

        ...

    def close(self) -> None:

        ...
