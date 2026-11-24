
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import mujoco
import numpy as np


_DEFAULT_PASSIVE_TUNING: dict[str, float | np.ndarray] = {
    "actuated_damping": 0.75,
    "mimic_damping": 0.25,
    "mimic_armature": 0.0,
    "mimic_frictionloss": 0.0,
    "actuator_kp_cap": 5.0,
    "mimic_equality_solref": np.array([0.005, 1.5], dtype=np.float64),
    "mimic_equality_solimp": np.array([0.95, 0.99, 0.0005, 0.5, 2.0], dtype=np.float64),
}

_PASSIVE_TUNING_OVERRIDES: dict[str, dict[str, float]] = {
    "linkerhand_l10": {
        "mimic_damping": 0.35,
        "mimic_armature": 0.01,
        "mimic_frictionloss": 0.01,
        "actuator_kp_cap": 4.0,
    },
    "linkerhand_l20pro": {
        "mimic_damping": 0.35,
        "mimic_armature": 0.01,
        "mimic_frictionloss": 0.01,
        "actuator_kp_cap": 4.0,
    },
    "linkerhand_l6": {
        "mimic_damping": 0.35,
        "mimic_armature": 0.01,
        "mimic_frictionloss": 0.01,
        "actuator_kp_cap": 4.0,
    },
    "linkerhand_o6": {
        "mimic_damping": 0.35,
        "mimic_armature": 0.01,
        "mimic_frictionloss": 0.01,
        "actuator_kp_cap": 4.0,
    },
    "linkerhand_o7": {
        "mimic_damping": 0.35,
        "mimic_armature": 0.01,
        "mimic_frictionloss": 0.01,
        "actuator_kp_cap": 4.0,
    },
    "revo2": {
        "mimic_damping": 0.30,
        "mimic_armature": 0.005,
        "mimic_frictionloss": 0.005,
        "actuator_kp_cap": 4.5,
    },
    "linkerhand_t12": {
        "actuator_kp_cap": 4.0,
    },
}


def _normalize_model_family_key(mjcf_path: str) -> str:
    stem = Path(mjcf_path).resolve().parent.name
    for suffix in ("_left", "_right"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def resolve_hand_passive_tuning(
    *,
    mjcf_path: str | None = None,
    model_family: str | None = None,
) -> dict[str, float | np.ndarray]:

    family = str(model_family).strip().lower() if model_family is not None else ""
    if not family and mjcf_path is not None:
        family = _normalize_model_family_key(mjcf_path)
    tuning = {
        key: value.copy() if isinstance(value, np.ndarray) else value
        for key, value in _DEFAULT_PASSIVE_TUNING.items()
    }
    tuning.update(_PASSIVE_TUNING_OVERRIDES.get(family, {}))
    return tuning


def _validated_unique_indices(name: str, values: Sequence[int], upper_bound: int) -> np.ndarray:
    indices = np.asarray(values, dtype=np.int32).reshape(-1)
    if indices.size != np.unique(indices).size:
        raise ValueError(f"{name} contains duplicate indices: {indices.tolist()}")
    if np.any(indices < 0) or np.any(indices >= int(upper_bound)):
        raise ValueError(f"{name} must be within [0, {int(upper_bound)}), got {indices.tolist()}")
    return indices


def validate_disjoint_actuator_groups(
    body_actuator_ids: Sequence[int],
    hand_actuator_ids: Sequence[int],
    *,
    actuator_count: int,
) -> tuple[np.ndarray, np.ndarray]:

    body_ids = _validated_unique_indices("body_actuator_ids", body_actuator_ids, actuator_count)
    hand_ids = _validated_unique_indices("hand_actuator_ids", hand_actuator_ids, actuator_count)
    overlap = np.intersect1d(body_ids, hand_ids)
    if overlap.size:
        raise ValueError(f"Body and hand actuator groups overlap at ids {overlap.tolist()}")
    return body_ids, hand_ids


def apply_hand_mujoco_passive_tuning(
    model: mujoco.MjModel,
    *,
    actuator_ids: Sequence[int],
    mimic_dof_ids: Sequence[int],
    equality_ids: Sequence[int],
    mjcf_path: str | None = None,
    model_family: str | None = None,
) -> dict[str, float | np.ndarray]:

    tuning = resolve_hand_passive_tuning(mjcf_path=mjcf_path, model_family=model_family)
    actuator_indices = _validated_unique_indices("hand actuator_ids", actuator_ids, int(model.nu))
    mimic_dof_indices = _validated_unique_indices("hand mimic_dof_ids", mimic_dof_ids, int(model.nv))
    equality_indices = _validated_unique_indices("hand equality_ids", equality_ids, int(model.neq))

    actuated_dof_indices: list[int] = []
    for actuator_id in actuator_indices:
        joint_id = int(model.actuator_trnid[int(actuator_id), 0])
        if not 0 <= joint_id < int(model.njnt):
            raise ValueError(f"Hand actuator {int(actuator_id)} is not attached to a joint")
        actuated_dof_indices.append(int(model.jnt_dofadr[joint_id]))
    if actuated_dof_indices:
        actuated_dofs = np.unique(np.asarray(actuated_dof_indices, dtype=np.int32))
        model.dof_damping[actuated_dofs] = np.maximum(
            model.dof_damping[actuated_dofs],
            float(tuning["actuated_damping"]),
        )

    if mimic_dof_indices.size:
        model.dof_damping[mimic_dof_indices] = np.maximum(
            model.dof_damping[mimic_dof_indices],
            float(tuning["mimic_damping"]),
        )
        model.dof_armature[mimic_dof_indices] = np.maximum(
            model.dof_armature[mimic_dof_indices],
            float(tuning["mimic_armature"]),
        )
        model.dof_frictionloss[mimic_dof_indices] = np.maximum(
            model.dof_frictionloss[mimic_dof_indices],
            float(tuning["mimic_frictionloss"]),
        )

    if actuator_indices.size:
        softened_kp = np.minimum(
            model.actuator_gainprm[actuator_indices, 0],
            float(tuning["actuator_kp_cap"]),
        )
        model.actuator_gainprm[actuator_indices, 0] = softened_kp
        model.actuator_biasprm[actuator_indices, 1] = -softened_kp

    if equality_indices.size:
        model.eq_solref[equality_indices, :] = np.asarray(
            tuning["mimic_equality_solref"], dtype=np.float64
        )
        model.eq_solimp[equality_indices, :] = np.asarray(
            tuning["mimic_equality_solimp"], dtype=np.float64
        )
    return tuning


class HandTargetConditioner:

    def __init__(
        self,
        actuator_ids: Sequence[int],
        ctrl_ranges: np.ndarray,
        *,
        max_velocity: float,
        filter_alpha: float = 1.0,
    ) -> None:
        actuator_count = int(np.asarray(ctrl_ranges).shape[0])
        self.actuator_ids = _validated_unique_indices("hand actuator_ids", actuator_ids, actuator_count)
        ranges = np.asarray(ctrl_ranges, dtype=np.float64)
        if ranges.ndim != 2 or ranges.shape[1] != 2:
            raise ValueError(f"ctrl_ranges must have shape (nu, 2), got {ranges.shape}")
        if not np.isfinite(max_velocity) or float(max_velocity) <= 0.0:
            raise ValueError(f"max_velocity must be finite and > 0, got {max_velocity}")
        if not np.isfinite(filter_alpha) or not 0.0 < float(filter_alpha) <= 1.0:
            raise ValueError(f"filter_alpha must be within (0, 1], got {filter_alpha}")
        self._ctrl_ranges = ranges[self.actuator_ids].copy()
        self.max_velocity = float(max_velocity)
        self.filter_alpha = float(filter_alpha)
        self._slot_by_actuator = {int(actuator_id): slot for slot, actuator_id in enumerate(self.actuator_ids)}
        self._requested = np.zeros(self.actuator_ids.size, dtype=np.float64)
        self._applied = np.zeros(self.actuator_ids.size, dtype=np.float64)
        self._initialized = np.zeros(self.actuator_ids.size, dtype=bool)

    def reset(self) -> None:
        self._requested.fill(0.0)
        self._applied.fill(0.0)
        self._initialized.fill(False)

    def update_target(self, actuator_id: int, value: float, *, current_position: float) -> float:

        if not np.isfinite(value):
            raise ValueError(f"Hand target must be finite, got {value}")
        if not np.isfinite(current_position):
            raise ValueError(f"Current hand position must be finite, got {current_position}")
        try:
            slot = self._slot_by_actuator[int(actuator_id)]
        except KeyError as exc:
            raise ValueError(f"Unknown hand actuator id {int(actuator_id)}") from exc
        low, high = self._ctrl_ranges[slot]
        clamped = float(np.clip(float(value), low, high))
        if not self._initialized[slot]:
            self._requested[slot] = clamped
            self._applied[slot] = float(np.clip(float(current_position), low, high))
            self._initialized[slot] = True
        else:
            alpha = self.filter_alpha
            self._requested[slot] = alpha * clamped + (1.0 - alpha) * self._requested[slot]
        return float(self._requested[slot])

    def step(self, dt: float) -> tuple[np.ndarray, np.ndarray]:

        if not np.isfinite(dt) or float(dt) <= 0.0:
            raise ValueError(f"dt must be finite and > 0, got {dt}")
        active_slots = np.flatnonzero(self._initialized)
        if active_slots.size == 0:
            return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float64)
        max_delta = self.max_velocity * float(dt)
        delta = np.clip(
            self._requested[active_slots] - self._applied[active_slots],
            -max_delta,
            max_delta,
        )
        self._applied[active_slots] += delta
        return self.actuator_ids[active_slots], self._applied[active_slots].copy()


__all__ = [
    "HandTargetConditioner",
    "apply_hand_mujoco_passive_tuning",
    "resolve_hand_passive_tuning",
    "validate_disjoint_actuator_groups",
]
