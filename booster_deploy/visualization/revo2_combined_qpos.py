
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import mujoco
import numpy as np

from booster_deploy.hand_retargeting.somehand_adapter import REVO2_COMMAND_JOINT_NAMES
from booster_deploy.types.hand_retarget_types import HandRetargetFrameNs
from booster_deploy.utils.project_path import resolve_project_path


DEFAULT_T2_REVO2_XML_PATH = (
    Path(__file__).resolve().parents[1]
    / "assets"
    / "t2"
    / "t2_31dof_revo2hand.xml"
)

_REVO2_HAND_XML_PATHS = {
    "left": Path(__file__).resolve().parents[1]
    / "hand_retargeting"
    / "resources"
    / "mjcf"
    / "revo2_left"
    / "model.xml",
    "right": Path(__file__).resolve().parents[1]
    / "hand_retargeting"
    / "resources"
    / "mjcf"
    / "revo2_right"
    / "model.xml",
}


def _named_qpos_slices(model: mujoco.MjModel) -> dict[str, slice]:
    addresses = [int(model.jnt_qposadr[joint_id]) for joint_id in range(model.njnt)]
    result: dict[str, slice] = {}
    for joint_id, start in enumerate(addresses):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if not name:
            continue
        stop = addresses[joint_id + 1] if joint_id + 1 < len(addresses) else int(model.nq)
        result[str(name)] = slice(start, stop)
    return result


class LatestHandFrameSelector:

    def __init__(self, *, stale_s: float = 0.25) -> None:
        self.stale_s = float(stale_s)
        if not np.isfinite(self.stale_s) or self.stale_s < 0.0:
            raise ValueError("Hand frame stale_s must be finite and non-negative")
        self._independent_frame: HandRetargetFrameNs | None = None
        self._independent_recv_s = 0.0

    def update_independent(self, frame: HandRetargetFrameNs, *, now_s: float) -> None:
        now_s = self._validated_time(now_s)
        self._independent_frame = frame
        self._independent_recv_s = now_s

    def select(
        self,
        embedded_frame: HandRetargetFrameNs | None,
        *,
        now_s: float,
    ) -> tuple[HandRetargetFrameNs | None, str]:
        now_s = self._validated_time(now_s)
        if embedded_frame is not None and embedded_frame.commands:
            return embedded_frame, "embedded"
        if (
            self._independent_frame is not None
            and self._independent_frame.commands
            and self._independent_recv_s > 0.0
            and now_s - self._independent_recv_s <= self.stale_s
        ):
            return self._independent_frame, "independent"
        return None, "fist"

    @staticmethod
    def _validated_time(now_s: float) -> float:
        value = float(now_s)
        if not np.isfinite(value):
            raise ValueError("Hand frame timestamp must be finite")
        return value


class Revo2CombinedQposMapper:

    def __init__(self, *, body_xml_path: str | Path, combined_xml_path: str | Path) -> None:
        self.body_xml_path = resolve_project_path(body_xml_path)
        self.combined_xml_path = resolve_project_path(combined_xml_path)
        self.body_model = mujoco.MjModel.from_xml_path(str(self.body_xml_path))
        self.combined_model = mujoco.MjModel.from_xml_path(str(self.combined_xml_path))
        self._body_slices = _named_qpos_slices(self.body_model)
        self._combined_slices = _named_qpos_slices(self.combined_model)
        self._body_copy_slices: list[tuple[str, slice, slice]] = []
        for joint_name, source_slice in self._body_slices.items():
            target_slice = self._combined_slices.get(joint_name)
            if target_slice is None:
                raise ValueError(
                    f"Combined XML {self.combined_xml_path} is missing body joint {joint_name!r}"
                )
            if self._slice_width(source_slice) != self._slice_width(target_slice):
                raise ValueError(f"qpos width mismatch for body joint {joint_name!r}")
            self._body_copy_slices.append((joint_name, source_slice, target_slice))

        self._allowed_hand_joints = {
            side: frozenset(joint_names)
            for side, joint_names in REVO2_COMMAND_JOINT_NAMES.items()
        }
        self._hand_source_slices: dict[str, dict[str, slice]] = {}
        self._hand_source_nq: dict[str, int] = {}
        for side, xml_path in _REVO2_HAND_XML_PATHS.items():
            hand_model = mujoco.MjModel.from_xml_path(str(xml_path))
            self._hand_source_slices[side] = _named_qpos_slices(hand_model)
            self._hand_source_nq[side] = int(hand_model.nq)
        for side, joint_names in self._allowed_hand_joints.items():
            missing = sorted(name for name in joint_names if name not in self._combined_slices)
            if missing:
                raise ValueError(
                    f"Combined XML {self.combined_xml_path} is missing {side} Revo2 joints: {missing}"
                )
        self._closed_qpos = np.asarray(
            self.combined_model.qpos0,
            dtype=np.float64,
        ).copy()
        self._set_closed_hands(self._closed_qpos)

    def compose(
        self,
        body_qpos: Sequence[float] | np.ndarray,
        hand_frame: HandRetargetFrameNs | None,
    ) -> np.ndarray:

        source = np.asarray(body_qpos, dtype=np.float64).reshape(-1)
        if source.shape != (int(self.body_model.nq),):
            raise ValueError(
                f"Body qpos shape {source.shape} does not match "
                f"{self.body_xml_path} nq={int(self.body_model.nq)}"
            )
        if not np.isfinite(source).all():
            raise ValueError("Body qpos contains non-finite values")

        combined = self._closed_qpos.copy()
        for _, source_slice, target_slice in self._body_copy_slices:
            combined[target_slice] = source[source_slice]

        seen_sides: set[str] = set()
        commands = () if hand_frame is None else hand_frame.commands
        for command in commands:
            side = str(command.hand_side).strip().lower()
            if side not in self._allowed_hand_joints:
                raise ValueError(
                    f"Unsupported Revo2 hand side {command.hand_side!r}; expected left or right"
                )
            if side in seen_sides:
                raise ValueError(f"Hand frame contains duplicate {side} commands")
            seen_sides.add(side)
            names = tuple(str(name) for name in command.joint_names)
            positions = np.asarray(command.joint_pos, dtype=np.float64).reshape(-1)
            if len(names) != positions.size or len(names) != len(set(names)):
                raise ValueError(
                    f"{side} hand command names and positions must have equal unique entries"
                )
            if not np.isfinite(positions).all():
                raise ValueError(f"{side} hand command contains non-finite positions")
            unknown = sorted(set(names) - self._allowed_hand_joints[side])
            if unknown:
                raise ValueError(f"{side} hand command contains unknown Revo2 joints: {unknown}")
            for joint_name, value in zip(names, positions, strict=True):
                clipped = self._clip_hinge_value(joint_name, float(value))
                self._set_hinge_value(combined, joint_name, clipped)
                self._set_coupled_distal(combined, joint_name, clipped)




            hand_qpos = np.asarray(command.qpos, dtype=np.float64).reshape(-1)
            source_slices = self._hand_source_slices[side]
            expected_nq = self._hand_source_nq[side]
            if hand_qpos.size not in (0, expected_nq):
                raise ValueError(
                    f"{side} hand command qpos must be empty or have {expected_nq} values, "
                    f"got {hand_qpos.size}"
                )
            if hand_qpos.size == expected_nq:
                if not np.isfinite(hand_qpos).all():
                    raise ValueError(f"{side} hand command qpos contains non-finite values")
                for joint_name, source_slice in source_slices.items():
                    target_slice = self._combined_slices.get(joint_name)
                    if target_slice is None:
                        continue
                    width = self._slice_width(source_slice)
                    if width != self._slice_width(target_slice):
                        raise ValueError(f"qpos width mismatch for hand joint {joint_name!r}")
                    values = hand_qpos[source_slice]
                    if width == 1:
                        combined[target_slice.start] = self._clip_hinge_value(
                            joint_name, float(values[0])
                        )
                    else:
                        combined[target_slice] = values
        return combined

    @staticmethod
    def available_hand_sides(hand_frame: HandRetargetFrameNs | None) -> tuple[str, ...]:
        if hand_frame is None:
            return ()
        return tuple(
            str(command.hand_side).strip().lower()
            for command in hand_frame.commands
            if str(command.hand_side).strip().lower() in {"left", "right"}
        )

    def _set_closed_hands(self, qpos: np.ndarray) -> None:
        for joint_names in self._allowed_hand_joints.values():
            for joint_name in joint_names:
                joint_id = self._joint_id(joint_name)
                closed_value = float(self.combined_model.jnt_range[joint_id, 1])
                self._set_hinge_value(qpos, joint_name, closed_value)
                self._set_coupled_distal(qpos, joint_name, closed_value)

    def _set_coupled_distal(
        self,
        qpos: np.ndarray,
        joint_name: str,
        value: float,
    ) -> None:
        if not joint_name.endswith("_proximal_joint"):
            return
        distal_name = joint_name.replace("_proximal_joint", "_distal_joint")
        if "_thumb_" in joint_name:
            distal_value = value
        elif any(
            f"_{finger}_" in joint_name
            for finger in ("index", "middle", "ring", "pinky")
        ):
            distal_value = 1.155 * value
        else:
            return
        self._set_hinge_value(
            qpos,
            distal_name,
            self._clip_hinge_value(distal_name, distal_value),
        )

    def _clip_hinge_value(self, joint_name: str, value: float) -> float:
        joint_id = self._joint_id(joint_name)
        if not bool(self.combined_model.jnt_limited[joint_id]):
            return value
        low, high = self.combined_model.jnt_range[joint_id]
        return float(np.clip(value, float(low), float(high)))

    def _set_hinge_value(self, qpos: np.ndarray, joint_name: str, value: float) -> None:
        target_slice = self._combined_slices.get(joint_name)
        if target_slice is None:
            raise ValueError(
                f"Combined XML {self.combined_xml_path} is missing hand joint {joint_name!r}"
            )
        if self._slice_width(target_slice) != 1:
            raise ValueError(f"Expected scalar qpos for hand joint {joint_name!r}")
        qpos[target_slice.start] = value

    def _joint_id(self, joint_name: str) -> int:
        joint_id = int(
            mujoco.mj_name2id(
                self.combined_model,
                mujoco.mjtObj.mjOBJ_JOINT,
                joint_name,
            )
        )
        if joint_id < 0:
            raise ValueError(
                f"Combined XML {self.combined_xml_path} is missing hand joint {joint_name!r}"
            )
        return joint_id

    @staticmethod
    def _slice_width(value: slice) -> int:
        return int(value.stop) - int(value.start)


__all__ = [
    "DEFAULT_T2_REVO2_XML_PATH",
    "LatestHandFrameSelector",
    "Revo2CombinedQposMapper",
]
