
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import yaml

from booster_deploy.utils.project_path import resolve_project_path


DEFAULT_T2_REVO2_ROS_MAPPING_PATH = (
    Path(__file__).resolve().parent / "resources" / "t2_revo2_ros_mapping.yaml"
)

# The Revo2 ROS position controllers accept radians. The previous JointState
# bridge used the equivalent range 0..1000, while the device-side driver
# applies this scale before writing the SDK command.
REVO2_POSITION_COMMAND_MAX_RAD = 1.7453292519943296
REVO2_POSITION_COMMAND_SCALE = 1000.0 / REVO2_POSITION_COMMAND_MAX_RAD


def _required_string(value: object, field: str) -> str:
    result = str(value).strip() if value is not None else ""
    if not result:
        raise ValueError(f"{field} must be a non-empty string")
    return result


def _required_unique_names(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field} must be a list of joint names")
    names = tuple(_required_string(item, field) for item in value)
    if not names:
        raise ValueError(f"{field} must not be empty")
    if len(names) != len(set(names)):
        raise ValueError(f"{field} contains duplicate joint names: {names}")
    return names


def _required_position_limits(
    value: object,
    field: str,
    *,
    expected_len: int,
) -> tuple[tuple[float, float], ...]:
    if not isinstance(value, (list, tuple)) or len(value) != expected_len:
        raise ValueError(f"{field} must contain {expected_len} [min, max] pairs")
    limits: list[tuple[float, float]] = []
    for index, raw_limit in enumerate(value):
        if not isinstance(raw_limit, (list, tuple)) or len(raw_limit) != 2:
            raise ValueError(f"{field}[{index}] must be a [min, max] pair")
        low, high = (float(item) for item in raw_limit)
        if not np.isfinite([low, high]).all() or high <= low:
            raise ValueError(f"{field}[{index}] must contain finite min < max")
        limits.append((low, high))
    return tuple(limits)


@dataclass(frozen=True, slots=True)
class RealHandSideMapping:

    hand_side: str
    topic: str
    source_joint_names: tuple[str, ...]
    target_joint_names: tuple[str, ...]
    source_position_limits: tuple[tuple[float, float], ...]

    def remap_positions(
        self,
        source_joint_names: Sequence[str],
        source_positions: Sequence[float] | np.ndarray,
    ) -> tuple[tuple[str, ...], np.ndarray]:

        names = tuple(str(name) for name in source_joint_names)
        positions = np.asarray(source_positions, dtype=np.float64).reshape(-1)
        if len(names) != positions.size:
            raise ValueError(
                f"{self.hand_side} hand command has {len(names)} names and "
                f"{positions.size} positions"
            )
        if len(names) != len(set(names)):
            raise ValueError(f"{self.hand_side} hand command contains duplicate source joint names")
        if not np.isfinite(positions).all():
            raise ValueError(f"{self.hand_side} hand command contains non-finite positions")

        position_by_name = dict(zip(names, positions, strict=True))
        missing = [name for name in self.source_joint_names if name not in position_by_name]
        if missing:
            raise ValueError(
                f"{self.hand_side} hand command is missing mapped source joints: {missing}"
            )
        remapped = np.asarray(
            [position_by_name[name] for name in self.source_joint_names],
            dtype=np.float64,
        )
        return self.target_joint_names, remapped

    def to_driver_positions(
        self,
        source_positions: Sequence[float] | np.ndarray,
        *,
        target_min: float = 0.0,
        target_max: float = REVO2_POSITION_COMMAND_MAX_RAD,
    ) -> np.ndarray:

        positions = np.asarray(source_positions, dtype=np.float64).reshape(-1)
        if positions.size != len(self.source_position_limits):
            raise ValueError(
                f"{self.hand_side} hand command has {positions.size} positions but "
                f"{len(self.source_position_limits)} position limits"
            )
        if not np.isfinite(positions).all():
            raise ValueError(f"{self.hand_side} hand command contains non-finite positions")
        target_min = float(target_min)
        target_max = float(target_max)
        if not np.isfinite([target_min, target_max]).all() or target_max <= target_min:
            raise ValueError("driver target range must contain finite min < max")
        limits = np.asarray(self.source_position_limits, dtype=np.float64)
        low = limits[:, 0]
        high = limits[:, 1]
        normalized = (positions - low) / (high - low)
        return np.clip(
            target_min + normalized * (target_max - target_min),
            target_min,
            target_max,
        )


@dataclass(frozen=True, slots=True)
class RealHandCommandMapping:

    path: Path
    left: RealHandSideMapping
    right: RealHandSideMapping

    def for_side(self, hand_side: str) -> RealHandSideMapping:
        side = str(hand_side).strip().lower()
        if side == "left":
            return self.left
        if side == "right":
            return self.right
        raise ValueError(f"Unsupported hand side {hand_side!r}; expected left or right")


def _parse_side(raw_hands: Mapping[str, object], hand_side: str) -> RealHandSideMapping:
    raw_side = raw_hands.get(hand_side)
    if not isinstance(raw_side, Mapping):
        raise ValueError(f"hands.{hand_side} must be a mapping")
    topic = _required_string(raw_side.get("topic"), f"hands.{hand_side}.topic")
    if not topic.startswith("/"):
        raise ValueError(f"hands.{hand_side}.topic must be an absolute ROS topic, got {topic!r}")
    source_joint_names = _required_unique_names(
        raw_side.get("source_joint_names"),
        f"hands.{hand_side}.source_joint_names",
    )
    target_joint_names = _required_unique_names(
        raw_side.get("target_joint_names"),
        f"hands.{hand_side}.target_joint_names",
    )
    source_position_limits = _required_position_limits(
        raw_side.get("source_position_limits"),
        f"hands.{hand_side}.source_position_limits",
        expected_len=len(source_joint_names),
    )
    if len(source_joint_names) != len(target_joint_names):
        raise ValueError(
            f"hands.{hand_side} source/target joint counts differ: "
            f"{len(source_joint_names)} vs {len(target_joint_names)}"
        )
    return RealHandSideMapping(
        hand_side=hand_side,
        topic=topic,
        source_joint_names=source_joint_names,
        target_joint_names=target_joint_names,
        source_position_limits=source_position_limits,
    )


def load_real_hand_command_mapping(
    path: str | Path = DEFAULT_T2_REVO2_ROS_MAPPING_PATH,
) -> RealHandCommandMapping:

    mapping_path = resolve_project_path(path)
    if not mapping_path.is_file():
        raise FileNotFoundError(f"Real-hand mapping config not found: {mapping_path}")
    with mapping_path.open("r", encoding="utf-8") as file_obj:
        raw = yaml.safe_load(file_obj) or {}
    if not isinstance(raw, Mapping):
        raise ValueError(f"Real-hand mapping root must be a mapping: {mapping_path}")
    version = int(raw.get("version", 1))
    if version != 1:
        raise ValueError(f"Unsupported real-hand mapping version {version}: {mapping_path}")
    raw_hands = raw.get("hands")
    if not isinstance(raw_hands, Mapping):
        raise ValueError(f"Real-hand mapping must contain a hands mapping: {mapping_path}")
    return RealHandCommandMapping(
        path=mapping_path,
        left=_parse_side(raw_hands, "left"),
        right=_parse_side(raw_hands, "right"),
    )


__all__ = [
    "DEFAULT_T2_REVO2_ROS_MAPPING_PATH",
    "REVO2_POSITION_COMMAND_MAX_RAD",
    "REVO2_POSITION_COMMAND_SCALE",
    "RealHandCommandMapping",
    "RealHandSideMapping",
    "load_real_hand_command_mapping",
]
