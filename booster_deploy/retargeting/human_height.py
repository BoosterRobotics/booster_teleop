"""Resolve the human height used by GMR retargeting."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math

import numpy as np

from booster_deploy.inputs.pico.coordinate import BODY_JOINT_NAMES, BODY_JOINT_PARENTS
from booster_deploy.retargeting.gmr.params import IK_CONFIG_DICT
from booster_deploy.types import HumanFrame


DEFAULT_HUMAN_HEIGHT_M = 1.75
PICO_HUMAN_FORMATS = frozenset({"xrobot", "pico", "pico_rt"})


@dataclass(frozen=True, slots=True)
class HumanHeightResolution:
    """Resolved GMR height and an operator-readable description of its source."""

    value_m: float
    source: str


def _validate_height(value: float, *, source: str) -> float:
    height = float(value)
    if not math.isfinite(height) or height <= 0.0:
        raise ValueError(f"human height from {source} must be finite and positive, got {value}")
    return height


def _ancestor_indices(index: int, parents: np.ndarray) -> list[int]:
    ancestors: list[int] = []
    seen: set[int] = set()
    while index >= 0:
        if index in seen:
            raise ValueError("Pico skeleton hierarchy contains a cycle")
        seen.add(index)
        ancestors.append(index)
        index = int(parents[index])
    return ancestors


def pico_height_joint_path(start_joint: str, end_joint: str) -> tuple[str, ...]:
    """Return the Pico hierarchy path between two named joints, including both."""

    name_to_index = {name: index for index, name in enumerate(BODY_JOINT_NAMES)}
    missing = [name for name in (start_joint, end_joint) if name not in name_to_index]
    if missing:
        raise KeyError(f"Pico height anchors are absent from the skeleton definition: {missing}")

    start_ancestors = _ancestor_indices(name_to_index[start_joint], BODY_JOINT_PARENTS)
    end_ancestors = _ancestor_indices(name_to_index[end_joint], BODY_JOINT_PARENTS)
    start_set = set(start_ancestors)
    lca = next((index for index in end_ancestors if index in start_set), None)
    if lca is None:
        raise ValueError(f"Pico height anchors have no common ancestor: {start_joint}, {end_joint}")

    start_to_lca = start_ancestors[: start_ancestors.index(lca) + 1]
    end_to_lca = end_ancestors[: end_ancestors.index(lca)]
    path_indices = start_to_lca + list(reversed(end_to_lca))
    return tuple(BODY_JOINT_NAMES[index] for index in path_indices)


def estimate_pico_human_height(
    frame: HumanFrame,
    *,
    start_joint: str = "Head",
    end_joint: str = "Left_Foot",
) -> float:
    """Estimate Pico skeleton height by summing 3D bone lengths between anchors.

    Pico exposes global joint positions rather than static BVH OFFSET values. The
    hierarchy path length is the pose-invariant equivalent of the BVH estimator;
    a direct endpoint distance would shrink when the user bends a joint.
    """

    path = pico_height_joint_path(start_joint, end_joint)
    missing = [name for name in path if name not in frame]
    if missing:
        raise KeyError(f"Pico HumanFrame is missing height-path joints: {missing}")

    positions: list[np.ndarray] = []
    for name in path:
        position = np.asarray(frame[name][0], dtype=np.float64)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError(f"Pico joint {name!r} has invalid position {position}")
        positions.append(position)

    segment_lengths = np.asarray(
        [np.linalg.norm(child - parent) for parent, child in zip(positions, positions[1:])],
        dtype=np.float64,
    )
    if segment_lengths.size == 0 or np.any(segment_lengths <= 1.0e-6):
        raise ValueError(
            f"Pico height path {start_joint}->{end_joint} contains a degenerate bone: "
            f"{segment_lengths.tolist()}"
        )
    return _validate_height(float(np.sum(segment_lengths)), source="Pico skeleton")


def resolve_actual_human_height(
    *,
    human_format: str,
    robot_name: str,
    frame: HumanFrame,
    configured_height: float | None,
    default_height: float = DEFAULT_HUMAN_HEIGHT_M,
) -> HumanHeightResolution:
    """Resolve explicit, config-driven automatic, or fallback GMR human height."""

    if configured_height is not None:
        return HumanHeightResolution(
            _validate_height(configured_height, source="explicit configuration"),
            "explicit configuration",
        )

    format_key = str(human_format).lower()
    ik_config_path = IK_CONFIG_DICT.get(human_format, {}).get(robot_name)
    ik_config: dict = {}
    if ik_config_path is not None:
        with open(ik_config_path, "r", encoding="utf-8") as stream:
            ik_config = json.load(stream)

    if format_key in PICO_HUMAN_FORMATS and bool(ik_config.get("use_auto_scale", True)):
        start_joint = str(ik_config.get("human_height_cal_start", "Head"))
        end_joint = str(ik_config.get("human_height_cal_end", "Left_Foot"))
        height = estimate_pico_human_height(
            frame,
            start_joint=start_joint,
            end_joint=end_joint,
        )
        return HumanHeightResolution(
            height,
            f"Pico skeleton path {start_joint}->{end_joint}",
        )

    fallback = ik_config.get("human_height", default_height)
    return HumanHeightResolution(
        _validate_height(fallback, source="IK/default configuration"),
        "IK/default configuration",
    )


__all__ = [
    "DEFAULT_HUMAN_HEIGHT_M",
    "HumanHeightResolution",
    "estimate_pico_human_height",
    "pico_height_joint_path",
    "resolve_actual_human_height",
]
