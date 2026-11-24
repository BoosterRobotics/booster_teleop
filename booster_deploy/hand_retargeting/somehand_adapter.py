
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from booster_deploy.types import HumanFrame
from booster_deploy.types.hand_retarget_types import HandRetargetCommandNs

from .landmarks import human_frame_to_landmarks, normalize_hand_side

RESOURCE_ROOT = Path(__file__).resolve().parent / "resources"
DEFAULT_REVO2_RIGHT_CONFIG = RESOURCE_ROOT / "revo2_right.yaml"
DEFAULT_REVO2_LEFT_CONFIG = RESOURCE_ROOT / "revo2_left.yaml"

REVO2_COMMAND_JOINT_NAMES: dict[str, tuple[str, ...]] = {
    "right": (
        "right_thumb_metacarpal_joint",
        "right_thumb_proximal_joint",
        "right_index_proximal_joint",
        "right_middle_proximal_joint",
        "right_ring_proximal_joint",
        "right_pinky_proximal_joint",
    ),
    "left": (
        "left_thumb_metacarpal_joint",
        "left_thumb_proximal_joint",
        "left_index_proximal_joint",
        "left_middle_proximal_joint",
        "left_ring_proximal_joint",
        "left_pinky_proximal_joint",
    ),
}

def ensure_somehand_importable(repo_path: str | Path | None = None) -> Path:

    if repo_path is not None:
        candidate = Path(repo_path).expanduser().resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"Configured somehand resource root does not exist: {candidate}")
        return candidate
    return RESOURCE_ROOT

class Revo2SomehandRetargeter:

    def __init__(
        self,
        hand_side: str,
        *,
        config_path: str | Path | None = None,
        somehand_repo: str | Path | None = None,
        tip_mode: str = "extrapolate",
    ) -> None:
        self.hand_side = normalize_hand_side(hand_side)
        self.tip_mode = tip_mode
        ensure_somehand_importable(somehand_repo)
        from booster_deploy.hand_retargeting.somehand_core.application.engine import RetargetingEngine
        from booster_deploy.hand_retargeting.somehand_core.domain.models import HandFrame
        from booster_deploy.hand_retargeting.somehand_core.paths import resolve_config_path

        self._hand_frame_cls = HandFrame
        self._resolve_config_path = resolve_config_path
        self._engine_cls = RetargetingEngine
        resolved_config = self._resolve_config_path(
            str(
                config_path
                or (
                    DEFAULT_REVO2_LEFT_CONFIG
                    if self.hand_side == "left"
                    else DEFAULT_REVO2_RIGHT_CONFIG
                )
            )
        )
        self.config_path = str(resolved_config)
        self._engine = self._engine_cls.from_config_path(self.config_path)
        config_side = normalize_hand_side(getattr(self._engine.config.hand, "side", self.hand_side))
        if config_side != self.hand_side:
            raise ValueError(
                f"somehand config side {config_side!r} does not match requested hand side {self.hand_side!r}"
            )
        self._joint_name_to_qpos_index = self._engine.hand_model.get_joint_name_to_qpos_index()
        self._command_joint_names = tuple(
            name
            for name in self._engine.hand_model.get_joint_names()
            if not name.endswith("_distal_joint")
        )
        self._last_result = None

    @property
    def command_joint_names(self) -> tuple[str, ...]:
        return self._command_joint_names

    @property
    def joint_name_to_qpos_index(self) -> dict[str, int]:
        return dict(self._joint_name_to_qpos_index)

    @property
    def hand_model(self) -> Any:
        return self._engine.hand_model

    @property
    def config(self) -> Any:
        return self._engine.config

    @property
    def last_result(self) -> Any:
        return self._last_result

    def retarget(self, frame: HumanFrame) -> tuple[HandRetargetCommandNs, float | None]:

        landmarks = human_frame_to_landmarks(frame, self.hand_side, tip_mode=self.tip_mode)
        hand_frame = self._hand_frame_cls(landmarks_3d=landmarks, landmarks_2d=None, hand_side=self.hand_side)
        result = self._engine.process(hand_frame)
        self._last_result = result
        qpos = np.asarray(result.qpos, dtype=np.float64).copy()
        joint_pos = np.asarray(
            [qpos[self._joint_name_to_qpos_index[name]] for name in self._command_joint_names],
            dtype=np.float64,
        )
        loss = None
        try:
            loss = float(self._engine.retargeter.compute_error())
        except Exception:
            loss = None
        command = HandRetargetCommandNs(
            hand_side=self.hand_side,
            robot_name=str(self._engine.config.hand.name),
            joint_names=self._command_joint_names,
            joint_pos=joint_pos,
            qpos=qpos,
            loss=loss,
        )
        return command, loss


__all__ = [
    "DEFAULT_REVO2_LEFT_CONFIG",
    "DEFAULT_REVO2_RIGHT_CONFIG",
    "RESOURCE_ROOT",
    "REVO2_COMMAND_JOINT_NAMES",
    "Revo2SomehandRetargeter",
    "ensure_somehand_importable",
]
