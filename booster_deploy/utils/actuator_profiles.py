
from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Real
from pathlib import Path
from typing import Any

import yaml


DEFAULT_T2_ACTUATOR_PROFILE_PATH = Path(__file__).parent.parent / "robots" / "t2_actuator_profiles.yaml"

_SEQUENCE_FIELDS = ("joint_stiffness", "joint_damping", "effort_limit")
_T2_SECTION_SPECS = (("head", 2), ("arms", 14), ("waist", 3), ("legs", 12))
_T2_SECTION_NAMES = tuple(name for name, _ in _T2_SECTION_SPECS)
_T2_JOINT_COUNT = sum(length for _, length in _T2_SECTION_SPECS)
T2_PROFILE_SECTION_SPECS = _T2_SECTION_SPECS
T2_PROFILE_SECTION_NAMES = _T2_SECTION_NAMES
T2_PROFILE_JOINT_COUNT = _T2_JOINT_COUNT
DEFAULT_ACTION_SCALE_FACTOR = 0.25


def load_actuator_profiles(profile_path: str | Path | None = None) -> Mapping[str, Mapping[str, Any]]:

    path = Path(profile_path) if profile_path is not None else DEFAULT_T2_ACTUATOR_PROFILE_PATH
    with path.open("r", encoding="utf-8") as file_obj:
        data = yaml.safe_load(file_obj) or {}

    if not isinstance(data, Mapping):
        raise TypeError(f"Actuator profile file must contain a mapping: {path}")

    profiles = data.get("profiles")
    if not isinstance(profiles, Mapping):
        raise KeyError(f"Actuator profile file must contain a 'profiles' mapping: {path}")

    return profiles


def available_actuator_profiles(profile_path: str | Path | None = None) -> tuple[str, ...]:

    return tuple(str(name) for name in load_actuator_profiles(profile_path))


def apply_t2_actuator_profile(
    robot_cfg: Any,
    profile_name: str,
    profile_path: str | Path | None = None,
) -> Any:

    profiles = load_actuator_profiles(profile_path)
    if profile_name not in profiles:
        available = ", ".join(str(name) for name in profiles) or "<none>"
        raise KeyError(f"Unknown actuator profile '{profile_name}'. Available profiles: {available}")

    profile = profiles[profile_name]
    if not isinstance(profile, Mapping):
        raise TypeError(f"Actuator profile '{profile_name}' must be a mapping")

    expected_len = len(robot_cfg.joint_names)
    robot_cfg.joint_stiffness = _normalize_profile_field(
        profile_name,
        "joint_stiffness",
        _resolve_t2_profile_field(profile_name, profile, "joint_stiffness", expected_len),
        expected_len,
    )
    robot_cfg.joint_damping = _normalize_profile_field(
        profile_name,
        "joint_damping",
        _resolve_t2_profile_field(profile_name, profile, "joint_damping", expected_len),
        expected_len,
    )
    robot_cfg.effort_limit = _normalize_profile_field(
        profile_name,
        "effort_limit",
        _resolve_t2_profile_field(profile_name, profile, "effort_limit", expected_len),
        expected_len,
    )
    robot_cfg.action_scale_factor = _coerce_numeric_scalar(
        profile_name,
        "action_scale_factor",
        profile.get("action_scale_factor", DEFAULT_ACTION_SCALE_FACTOR),
    )
    return robot_cfg


def compact_t2_profile(
    profile: Mapping[str, Any],
    profile_name: str = "<profile>",
) -> dict[str, Any]:

    field_sections: dict[str, dict[str, list[float]]] = {}
    for field_name in _SEQUENCE_FIELDS:
        flat = _resolve_t2_profile_field(profile_name, profile, field_name, _T2_JOINT_COUNT)
        field_sections[field_name] = split_t2_profile_sections(flat, profile_name, field_name)

    compact: dict[str, Any] = {}
    for section_name, _ in _T2_SECTION_SPECS:
        compact[section_name] = {
            field_name: field_sections[field_name][section_name]
            for field_name in _SEQUENCE_FIELDS
        }

    action_scale_factor = _coerce_numeric_scalar(
        profile_name,
        "action_scale_factor",
        profile.get("action_scale_factor", DEFAULT_ACTION_SCALE_FACTOR),
    )
    if action_scale_factor != DEFAULT_ACTION_SCALE_FACTOR:
        compact["action_scale_factor"] = action_scale_factor
    return compact


def split_t2_profile_sections(
    values: Sequence[Any],
    profile_name: str = "<profile>",
    field_name: str = "<field>",
) -> dict[str, list[float]]:

    flat = _coerce_numeric_sequence(profile_name, field_name, values, _T2_JOINT_COUNT)
    sections: dict[str, list[float]] = {}
    cursor = 0
    for section_name, section_len in _T2_SECTION_SPECS:
        sections[section_name] = flat[cursor : cursor + section_len]
        cursor += section_len
    return sections


def _coerce_numeric_sequence(
    profile_name: str,
    field_name: str,
    value: Any,
    expected_len: int,
) -> list[float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"Actuator profile '{profile_name}.{field_name}' must be a sequence")
    if len(value) != expected_len:
        raise ValueError(
            f"Actuator profile '{profile_name}.{field_name}' has length {len(value)}, "
            f"expected {expected_len}"
        )

    output: list[float] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, Real):
            raise TypeError(
                f"Actuator profile '{profile_name}.{field_name}[{index}]' must be numeric, "
                f"got {type(item).__name__}"
            )
        output.append(float(item))
    return output


def _compact_t2_field(profile_name: str, field_name: str, value: Any) -> dict[str, list[float]]:
    flat = _normalize_profile_field(profile_name, field_name, value, _T2_JOINT_COUNT)
    return split_t2_profile_sections(flat, profile_name, field_name)


def _resolve_t2_profile_field(
    profile_name: str,
    profile: Mapping[str, Any],
    field_name: str,
    expected_len: int,
) -> list[float]:
    if field_name in profile:
        return _normalize_profile_field(profile_name, field_name, profile[field_name], expected_len)

    if all(section_name in profile for section_name in _T2_SECTION_NAMES):
        output: list[float] = []
        for section_name, section_len in _T2_SECTION_SPECS:
            section = profile[section_name]
            if not isinstance(section, Mapping):
                raise TypeError(
                    f"Actuator profile '{profile_name}.{section_name}' must be a mapping"
                )
            if field_name not in section:
                raise KeyError(f"Actuator profile '{profile_name}.{section_name}' is missing '{field_name}'")
            output.extend(
                _coerce_numeric_sequence(
                    profile_name,
                    f"{section_name}.{field_name}",
                    section[field_name],
                    section_len,
                )
            )
        return output

    missing = [field for field in _SEQUENCE_FIELDS if field not in profile]
    if missing:
        raise KeyError(f"Actuator profile '{profile_name}' is missing fields: {missing}")
    raise KeyError(
        f"Actuator profile '{profile_name}' must provide either flat fields or "
        f"section fields {list(_T2_SECTION_NAMES)}"
    )


def _normalize_profile_field(
    profile_name: str,
    field_name: str,
    value: Any,
    expected_len: int,
) -> list[float]:
    if isinstance(value, Mapping):
        missing = [section_name for section_name in _T2_SECTION_NAMES if section_name not in value]
        extra = [section_name for section_name in value if section_name not in _T2_SECTION_NAMES]
        if missing:
            raise KeyError(f"Actuator profile '{profile_name}.{field_name}' is missing sections: {missing}")
        if extra:
            raise KeyError(f"Actuator profile '{profile_name}.{field_name}' has unknown sections: {extra}")

        output: list[float] = []
        for section_name, section_len in _T2_SECTION_SPECS:
            output.extend(
                _coerce_numeric_sequence(
                    profile_name,
                    f"{field_name}.{section_name}",
                    value[section_name],
                    section_len,
                )
            )
        return output

    return _coerce_numeric_sequence(profile_name, field_name, value, expected_len)


def _coerce_numeric_scalar(profile_name: str, field_name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(
            f"Actuator profile '{profile_name}.{field_name}' must be numeric, "
            f"got {type(value).__name__}"
        )
    return float(value)
