
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import logging
import threading
from typing import Mapping, Sequence

import numpy as np


_MONITOR = "monitor"
_BACKOFF_PENDING = "backoff_pending"
_RESTORE_PENDING = "restore_pending"
_COOLDOWN = "cooldown"


@dataclass(frozen=True, slots=True)
class HandStallRecoveryConfig:

    enabled: bool = True
    feedback_stale_s: float = 0.15
    window_frames: int = 5
    stationary_tolerance: float = 1.0
    stationary_duration_s: float = 0.2
    close_error_min: float = 10.0
    backoff_units: float = 1.0
    retry_period_s: float = 0.5
    command_min: float = 0.0
    command_max: float = 1000.0

    def __post_init__(self) -> None:
        if self.window_frames < 2:
            raise ValueError("Hand stall recovery window_frames must be at least 2")
        for name, value, allow_zero in (
            ("feedback_stale_s", self.feedback_stale_s, False),
            ("stationary_tolerance", self.stationary_tolerance, True),
            ("stationary_duration_s", self.stationary_duration_s, False),
            ("close_error_min", self.close_error_min, True),
            ("backoff_units", self.backoff_units, False),
            ("retry_period_s", self.retry_period_s, False),
        ):
            invalid = value < 0.0 if allow_zero else value <= 0.0
            if not np.isfinite(value) or invalid:
                relation = "non-negative" if allow_zero else "positive"
                raise ValueError(f"Hand stall recovery {name} must be finite and {relation}")
        if (
            not np.isfinite([self.command_min, self.command_max]).all()
            or self.command_max <= self.command_min
        ):
            raise ValueError("Hand stall recovery command range must contain finite min < max")


@dataclass(slots=True)
class _FingerState:
    history: deque[tuple[float, float]]
    phase: str = _MONITOR
    feedback: float | None = None
    feedback_recv_s: float = 0.0
    backoff_target: float | None = None
    next_retry_s: float = 0.0


@dataclass(slots=True)
class _SideState:
    joint_names: tuple[str, ...]
    open_positions: np.ndarray
    closed_positions: np.ndarray
    close_directions: np.ndarray
    fingers: list[_FingerState] = field(default_factory=list)
    desired_positions: np.ndarray | None = None


class HandStallRecovery:

    def __init__(
        self,
        config: HandStallRecoveryConfig,
        *,
        joint_names: Mapping[str, Sequence[str]],
        open_positions: Mapping[str, Sequence[float]],
        closed_positions: Mapping[str, Sequence[float]],
        logger: logging.Logger | None = None,
    ) -> None:
        self.config = config
        self._logger = logger or logging.getLogger(__name__)
        self._lock = threading.Lock()
        self._sides: dict[str, _SideState] = {}
        for side in joint_names:
            names = tuple(str(name) for name in joint_names[side])
            opened = self._validated_positions(open_positions[side], len(names), f"{side} open")
            closed = self._validated_positions(closed_positions[side], len(names), f"{side} closed")
            if not names or len(names) != len(set(names)):
                raise ValueError(f"{side} hand joint names must be non-empty and unique")
            self._sides[side] = _SideState(
                joint_names=names,
                open_positions=opened,
                closed_positions=closed,
                close_directions=np.sign(closed - opened),
                fingers=[
                    _FingerState(history=deque())
                    for _ in names
                ],
            )

    def update_feedback(
        self,
        hand_side: str,
        joint_names: Sequence[str],
        positions: Sequence[float],
        *,
        now_s: float,
    ) -> None:

        if not self.config.enabled:
            return
        side = self._side(hand_side)
        now_s = self._validated_time(now_s)
        names = tuple(str(name) for name in joint_names)
        values = np.asarray(positions, dtype=np.float64).reshape(-1)
        if len(names) != values.size or len(names) != len(set(names)):
            raise ValueError(
                f"{hand_side} hand feedback names and positions must have equal unique entries"
            )
        if not np.isfinite(values).all():
            raise ValueError(f"{hand_side} hand feedback contains non-finite positions")
        by_name = dict(zip(names, values, strict=True))
        try:
            ordered = np.asarray([by_name[name] for name in side.joint_names], dtype=np.float64)
        except KeyError as exc:
            raise ValueError(
                f"{hand_side} hand feedback is missing joint {exc.args[0]!r}"
            ) from exc
        if np.any(ordered < self.config.command_min) or np.any(
            ordered > self.config.command_max
        ):
            raise ValueError(
                f"{hand_side} hand feedback must be within "
                f"[{self.config.command_min:g}, {self.config.command_max:g}]"
            )

        stalled_indices: list[int] = []
        with self._lock:
            desired = side.desired_positions
            for index, (state, feedback) in enumerate(zip(side.fingers, ordered, strict=True)):
                state.feedback = float(feedback)
                state.feedback_recv_s = now_s
                direction = float(side.close_directions[index])
                if desired is None or direction == 0.0:
                    self._clear_finger(state, clear_feedback=False)
                    continue
                closing_error = direction * (float(desired[index]) - float(feedback))
                if closing_error <= self.config.close_error_min:
                    self._clear_finger(state, clear_feedback=False)
                    continue

                state.history.append((now_s, float(feedback)))
                cutoff_s = now_s - self.config.stationary_duration_s
                while len(state.history) > 1 and state.history[1][0] <= cutoff_s:
                    state.history.popleft()
                if state.phase not in {_MONITOR, _COOLDOWN}:
                    continue
                if state.phase == _COOLDOWN and now_s < state.next_retry_s:
                    continue
                if len(state.history) < self.config.window_frames:
                    continue
                if (
                    now_s - state.history[0][0]
                    < self.config.stationary_duration_s - 1e-9
                ):
                    continue
                history_positions = [sample[1] for sample in state.history]
                if (
                    max(history_positions) - min(history_positions)
                    > self.config.stationary_tolerance
                ):
                    continue

                backoff = float(feedback) - direction * self.config.backoff_units
                if direction > 0.0:
                    backoff = max(float(side.open_positions[index]), backoff)
                else:
                    backoff = min(float(side.open_positions[index]), backoff)
                backoff = float(
                    np.clip(backoff, self.config.command_min, self.config.command_max)
                )
                if direction * (float(feedback) - backoff) <= 0.0:
                    continue
                state.phase = _BACKOFF_PENDING
                state.backoff_target = backoff
                stalled_indices.append(index)

        if stalled_indices:
            self._logger.info(
                "Hand stall detected: side=%s fingers=%s feedback=%s",
                hand_side,
                stalled_indices,
                ordered[stalled_indices].tolist(),
            )

    def apply(
        self,
        hand_side: str,
        desired_positions: Sequence[float],
        *,
        now_s: float,
    ) -> np.ndarray:

        side = self._side(hand_side)
        desired = self._validated_positions(
            desired_positions,
            len(side.joint_names),
            f"{hand_side} desired",
        )
        if not self.config.enabled:
            return desired
        now_s = self._validated_time(now_s)
        result = desired.copy()
        backoff_indices: list[int] = []
        restore_indices: list[int] = []
        with self._lock:
            side.desired_positions = desired.copy()
            for index, state in enumerate(side.fingers):
                direction = float(side.close_directions[index])
                feedback_is_fresh = (
                    state.feedback is not None
                    and state.feedback_recv_s > 0.0
                    and now_s - state.feedback_recv_s <= self.config.feedback_stale_s
                )
                if not feedback_is_fresh:
                    self._clear_finger(state, clear_feedback=True)
                    continue
                closing_error = direction * (desired[index] - float(state.feedback))
                if direction == 0.0 or closing_error <= self.config.close_error_min:
                    self._clear_finger(state, clear_feedback=False)
                    continue
                if state.phase == _BACKOFF_PENDING:
                    result[index] = float(state.backoff_target)
                    state.phase = _RESTORE_PENDING
                    backoff_indices.append(index)
                elif state.phase == _RESTORE_PENDING:
                    state.phase = _COOLDOWN
                    state.backoff_target = None
                    state.next_retry_s = now_s + self.config.retry_period_s


                    state.history.clear()
                    restore_indices.append(index)

        if backoff_indices:
            self._logger.info(
                "Hand stall BACKOFF: side=%s fingers=%s target=%s",
                hand_side,
                backoff_indices,
                result[backoff_indices].tolist(),
            )
        if restore_indices:
            self._logger.info(
                "Hand stall RETRY_CLOSE: side=%s fingers=%s target=%s cooldown=%.3fs",
                hand_side,
                restore_indices,
                result[restore_indices].tolist(),
                self.config.retry_period_s,
            )
        return result

    def reset(self, hand_side: str | None = None, *, clear_feedback: bool = False) -> None:

        if not self.config.enabled:
            return
        sides = (self._normalize_side(hand_side),) if hand_side is not None else tuple(self._sides)
        with self._lock:
            for side_name in sides:
                side = self._sides[side_name]
                side.desired_positions = None
                for state in side.fingers:
                    self._clear_finger(state, clear_feedback=clear_feedback)

    def _side(self, hand_side: str) -> _SideState:
        return self._sides[self._normalize_side(hand_side)]

    @staticmethod
    def _normalize_side(hand_side: str | None) -> str:
        side = str(hand_side).strip().lower()
        if side not in {"left", "right"}:
            raise ValueError(f"Unsupported hand side {hand_side!r}; expected left or right")
        return side

    def _validated_positions(
        self,
        positions: Sequence[float],
        expected_len: int,
        field: str,
    ) -> np.ndarray:
        values = np.asarray(positions, dtype=np.float64).reshape(-1)
        if values.size != expected_len:
            raise ValueError(f"{field} positions must contain {expected_len} values")
        if not np.isfinite(values).all():
            raise ValueError(f"{field} positions must be finite")
        if np.any(values < self.config.command_min) or np.any(values > self.config.command_max):
            raise ValueError(
                f"{field} positions must be within "
                f"[{self.config.command_min:g}, {self.config.command_max:g}]"
            )
        return values

    @staticmethod
    def _validated_time(now_s: float) -> float:
        value = float(now_s)
        if not np.isfinite(value):
            raise ValueError("Hand stall recovery timestamp must be finite")
        return value

    @staticmethod
    def _clear_finger(state: _FingerState, *, clear_feedback: bool) -> None:
        state.history.clear()
        state.phase = _MONITOR
        state.backoff_target = None
        state.next_retry_s = 0.0
        if clear_feedback:
            state.feedback = None
            state.feedback_recv_s = 0.0


__all__ = ["HandStallRecovery", "HandStallRecoveryConfig"]
