
from __future__ import annotations

from dataclasses import dataclass

from .base import RawControllerState, RawInputProvider


@dataclass(frozen=True)
class LocomotionCommand:

    vx: float = 0.0
    vy: float = 0.0
    wz: float = 0.0
    reset: bool = False
    stop: bool = False
    source: str = ""


@dataclass(frozen=True)
class LocomotionMapperConfig:

    max_vx: float = 1.0
    max_vy: float = 1.0
    max_wz: float = 1.2
    deadzone: float = 0.08
    invert_vx: bool = False
    invert_vy: bool = False
    invert_wz: bool = False
    left_y_mode: str = "bipolar"


class LocomotionMapper:

    def __init__(self, config: LocomotionMapperConfig | None = None) -> None:

        self.config = config or LocomotionMapperConfig()
        self._keyboard_vx = 0.0
        self._keyboard_vy = 0.0
        self._keyboard_wz = 0.0

    def map(self, state: RawControllerState) -> LocomotionCommand:

        if state.axes:
            return self._map_axes(state)
        return self._map_keyboard(state)

    def _map_axes(self, state: RawControllerState) -> LocomotionCommand:

        left_x = _deadzone(state.axis("left_x"), self.config.deadzone)
        left_y = _deadzone(state.axis("left_y"), self.config.deadzone)
        right_x = _deadzone(state.axis("right_x"), self.config.deadzone)
        if self.config.left_y_mode == "forward":
            vx = max(0.0, -left_y)
        elif self.config.left_y_mode == "bipolar":
            vx = -left_y
        else:
            raise ValueError(f"Unsupported left_y_mode={self.config.left_y_mode!r}")
        vy = -left_x
        wz = -right_x
        if self.config.invert_vx:
            vx = -vx
        if self.config.invert_vy:
            vy = -vy
        if self.config.invert_wz:
            wz = -wz
        return LocomotionCommand(
            vx=_clamp(vx * self.config.max_vx, -self.config.max_vx, self.config.max_vx),
            vy=_clamp(vy * self.config.max_vy, -self.config.max_vy, self.config.max_vy),
            wz=_clamp(wz * self.config.max_wz, -self.config.max_wz, self.config.max_wz),
            source=state.source,
        )

    def _map_keyboard(self, state: RawControllerState) -> LocomotionCommand:

        reset = False
        stop = False
        for key in state.buttons:
            if key == "w":
                self._keyboard_vx = min(self.config.max_vx, self._keyboard_vx + 0.1)
            elif key == "s":
                self._keyboard_vx = max(-self.config.max_vx, self._keyboard_vx - 0.1)
            elif key == "a":
                self._keyboard_vy = min(self.config.max_vy, self._keyboard_vy + 0.1)
            elif key == "d":
                self._keyboard_vy = max(-self.config.max_vy, self._keyboard_vy - 0.1)
            elif key == "q":
                self._keyboard_wz = min(self.config.max_wz, self._keyboard_wz + 0.1)
            elif key == "e":
                self._keyboard_wz = max(-self.config.max_wz, self._keyboard_wz - 0.1)
            elif key == "space":
                self._keyboard_vx = 0.0
                self._keyboard_vy = 0.0
                self._keyboard_wz = 0.0
                stop = True
            elif key == "r":
                reset = True
        return LocomotionCommand(
            vx=self._keyboard_vx,
            vy=self._keyboard_vy,
            wz=self._keyboard_wz,
            reset=reset,
            stop=stop,
            source=state.source,
        )


class ControlInputManager:

    def __init__(
        self,
        *,
        backend: str = "auto",
        mapper: LocomotionMapper | None = None,
    ) -> None:

        self.mapper = mapper or LocomotionMapper()
        self.provider = _make_provider(backend=backend)

    @property
    def source(self) -> str:

        return self.provider.source

    def get_command(self) -> LocomotionCommand:

        return self.mapper.map(self.provider.get_state())

    def close(self) -> None:

        self.provider.close()


def _make_provider(*, backend: str) -> RawInputProvider:

    raise ValueError(
        f"Unsupported control input backend={backend!r}; "
        "this release accepts velocity_command_source=\"stdin\" only"
    )


def _deadzone(value: float, deadzone: float) -> float:

    value = float(value)
    return 0.0 if abs(value) < float(deadzone) else _clamp(value, -1.0, 1.0)


def _clamp(value: float, low: float, high: float) -> float:

    return max(float(low), min(float(high), float(value)))
