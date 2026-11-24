
from __future__ import annotations

from collections.abc import Sequence
from typing import Generic, Protocol, TypeVar

from booster_deploy.utils.isaaclab.configclass import configclass


FrameT = TypeVar("FrameT")


class RtReferenceWindowSource(Protocol[FrameT]):

    def latest_ready_idx(self) -> int | None: ...

    def get_window(self, indices: Sequence[int]) -> list[FrameT] | None: ...


@configclass
class RtReferenceConsumerCfg:

    target_lag_frames: int = 10
    catchup_trigger_lag_frames: int = 12
    catchup_target_lag_frames: int = 10
    max_extra_base_skip_per_inference: int = 1
    missing_behavior: str = "hold"


class RtReferenceConsumer(Generic[FrameT]):

    def __init__(
        self,
        cfg: RtReferenceConsumerCfg,
        *,
        future_count: int,
        stride: int,
    ) -> None:
        self.cfg = cfg
        self.future_count = int(future_count)
        self.stride = int(stride)
        self.future_span_frames = (self.future_count - 1) * self.stride
        self._validate()
        self.consume_base_idx: int | None = None
        self._initial_base_idx: int | None = None
        self._last_valid_window: list[FrameT] | None = None
        self._catchup_active = False
        self._last_was_hold = False

    @property
    def last_was_hold(self) -> bool:

        return self._last_was_hold

    @property
    def minimum_startup_samples(self) -> int:

        return int(self.cfg.target_lag_frames) + 1

    @property
    def catchup_active(self) -> bool:
        return self._catchup_active

    def reset(self) -> None:

        self.consume_base_idx = None
        self._initial_base_idx = None
        self._last_valid_window = None
        self._catchup_active = False
        self._last_was_hold = False

    def start_at(self, index: int) -> None:
        self.reset()
        self._initial_base_idx = int(index)

    def request_rebase_to_latest(self) -> None:

        self._initial_base_idx = None
        self.consume_base_idx = None
        self._catchup_active = False
        self._last_was_hold = False

    def initial_window_available(self, source: RtReferenceWindowSource[FrameT]) -> bool:

        latest = source.latest_ready_idx()
        if latest is None:
            return False
        base_idx = (int(latest) - int(self.cfg.target_lag_frames)
                    if self._initial_base_idx is None else self._initial_base_idx)
        if int(latest) - base_idx < int(self.cfg.target_lag_frames):
            return False
        return source.get_window(self.window_indices(base_idx)) is not None

    def consume(self, source: RtReferenceWindowSource[FrameT]) -> list[FrameT] | None:

        self._last_was_hold = False
        latest = source.latest_ready_idx()
        if latest is None:
            return self._held_window()
        latest = int(latest)

        if self.consume_base_idx is None:
            initial_base_idx = (latest - int(self.cfg.target_lag_frames)
                                if self._initial_base_idx is None else self._initial_base_idx)
            if latest - initial_base_idx < int(self.cfg.target_lag_frames):
                return self._held_window()
            window = self._read_window(source, initial_base_idx)
            if window is None:
                return self._held_window()
            return self._accept(initial_base_idx, window, latest)

        current_lag = latest - self.consume_base_idx
        if self._catchup_active:
            if current_lag <= int(self.cfg.catchup_target_lag_frames):
                self._catchup_active = False
        elif current_lag > int(self.cfg.catchup_trigger_lag_frames):
            self._catchup_active = True

        lag_floor = (
            int(self.cfg.catchup_target_lag_frames)
            if self._catchup_active
            else int(self.cfg.target_lag_frames)
        )
        furthest_base_idx = latest - lag_floor
        sequential_base_idx = self.consume_base_idx + 1
        if sequential_base_idx > furthest_base_idx:
            return self._held_window()

        max_delta = 1
        if self._catchup_active:
            max_delta += int(self.cfg.max_extra_base_skip_per_inference)
        preferred_base_idx = min(
            self.consume_base_idx + max_delta,
            furthest_base_idx,
        )

        candidates = [preferred_base_idx]
        if preferred_base_idx != sequential_base_idx:
            candidates.append(sequential_base_idx)
        for base_idx in candidates:
            window = self._read_window(source, base_idx)
            if window is not None:
                return self._accept(base_idx, window, latest)
        return self._held_window()

    def window_indices(self, base_idx: int) -> list[int]:
        return [
            int(base_idx) + offset * self.stride
            for offset in range(self.future_count)
        ]

    def reference_lag_frames(self, latest_ready_idx: int | None) -> int | None:
        if latest_ready_idx is None or self.consume_base_idx is None:
            return None
        return int(latest_ready_idx) - self.consume_base_idx

    def _read_window(
        self,
        source: RtReferenceWindowSource[FrameT],
        base_idx: int,
    ) -> list[FrameT] | None:
        return source.get_window(self.window_indices(base_idx))

    def _accept(
        self,
        base_idx: int,
        window: Sequence[FrameT],
        latest_ready_idx: int,
    ) -> list[FrameT]:
        self.consume_base_idx = int(base_idx)
        self._last_valid_window = list(window)
        self._last_was_hold = False
        if (
            self._catchup_active
            and self.reference_lag_frames(latest_ready_idx)
            <= int(self.cfg.catchup_target_lag_frames)
        ):
            self._catchup_active = False
        return list(self._last_valid_window)

    def _held_window(self) -> list[FrameT] | None:
        self._last_was_hold = self._last_valid_window is not None
        if self._last_valid_window is None:
            return None
        return list(self._last_valid_window)

    def _validate(self) -> None:
        if self.future_count <= 0:
            raise ValueError("future_count must be positive")
        if self.stride <= 0:
            raise ValueError("stride must be positive")

        target_lag = int(self.cfg.target_lag_frames)
        catchup_trigger = int(self.cfg.catchup_trigger_lag_frames)
        catchup_target = int(self.cfg.catchup_target_lag_frames)
        max_extra_skip = int(self.cfg.max_extra_base_skip_per_inference)
        if target_lag <= self.future_span_frames:
            raise ValueError(
                "target_lag_frames must exceed future_span_frames "
                f"({target_lag} <= {self.future_span_frames})"
            )
        if catchup_target < target_lag:
            raise ValueError(
                "catchup_target_lag_frames must be >= target_lag_frames"
            )
        if catchup_trigger <= catchup_target:
            raise ValueError(
                "catchup_trigger_lag_frames must exceed catchup_target_lag_frames"
            )
        if max_extra_skip < 0:
            raise ValueError(
                "max_extra_base_skip_per_inference must be non-negative"
            )
        if self.cfg.missing_behavior != "hold":
            raise ValueError("RT reference missing_behavior currently only supports 'hold'")


__all__ = [
    "RtReferenceConsumer",
    "RtReferenceConsumerCfg",
    "RtReferenceWindowSource",
]
