
from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray

from booster_deploy.types import HumanFrame, RetargetFrame, clone_human_frame

from .gmr import GeneralMotionRetargeting


PICO_HUMAN_FORMATS = frozenset({"xrobot", "pico", "pico_rt"})
PICO_FOOT_MOD_SOURCES = {
    "Left_Foot_Mod": ("Left_Ankle", "Left_Foot"),
    "Right_Foot_Mod": ("Right_Ankle", "Right_Foot"),
}


def prepare_human_frame_for_gmr(frame: HumanFrame, human_format: str) -> HumanFrame:
    """Clone the input and compose PICO ankle-position/foot-orientation targets."""

    prepared = clone_human_frame(frame)
    if str(human_format).lower() not in PICO_HUMAN_FORMATS:
        return prepared

    required = {
        source_name
        for source_names in PICO_FOOT_MOD_SOURCES.values()
        for source_name in source_names
    }
    missing = sorted(required.difference(prepared))
    if missing:
        raise KeyError(f"Pico HumanFrame is missing foot-mod source joints: {missing}")

    for virtual_name, (position_source, rotation_source) in PICO_FOOT_MOD_SOURCES.items():
        prepared[virtual_name] = (
            prepared[position_source][0].copy(),
            prepared[rotation_source][1].copy(),
        )
    return prepared


class GMRRetargeter:

    def __init__(
        self,
        *,
        human_format: str,
        robot_name: str = "g1",
        actual_human_height: float = 1.75,
        solver: str = "daqp",
        damping: float = 1e-1,
        verbose: bool = False,
        ground_detection_mode: str = "mesh",
        use_velocity_limit: bool = False,
        source: str = "",
    ) -> None:

        self.human_format = human_format
        self.robot_name = robot_name
        self.source = source or human_format
        self._gmr = GeneralMotionRetargeting(
            src_human=human_format,
            tgt_robot=robot_name,
            actual_human_height=actual_human_height,
            solver=solver,
            damping=damping,
            verbose=verbose,
            ground_detection_mode=ground_detection_mode,
            use_velocity_limit=use_velocity_limit,
        )

    def retarget(
        self,
        frame: HumanFrame,
        *,
        timestamp_s: Optional[float] = None,
        seq: Optional[int] = None,
    ) -> RetargetFrame:

        prepared = prepare_human_frame_for_gmr(frame, self.human_format)
        qpos = self._gmr.retarget(prepared)
        return RetargetFrame.from_qpos(
            np.asarray(qpos, dtype=np.float64),
            robot_name=self.robot_name,
            source=self.source,
            timestamp_s=timestamp_s,
            seq=seq,
        )

    def retarget_sequence(self, frames: list[HumanFrame]) -> NDArray[np.float64]:

        prepared = [prepare_human_frame_for_gmr(frame, self.human_format) for frame in frames]
        return np.asarray(self._gmr.retarget_sequence(prepared), dtype=np.float64)

    @property
    def last_scaled_human_frame(self) -> HumanFrame | None:
        """Return display targets in the same ground-corrected world as output qpos."""

        scaled = getattr(self._gmr, "scaled_human_data", None)
        if scaled is None:
            return None
        frame = clone_human_frame(scaled)
        ground_z_offset = float(getattr(self._gmr, "ground_qpos_z_offset", 0.0))
        offset = np.asarray([0.0, 0.0, ground_z_offset], dtype=np.float64)
        return {name: (pos + offset, quat) for name, (pos, quat) in frame.items()}

    @property
    def robot_xml_path(self) -> str:

        return str(self._gmr.xml_file)

    @property
    def ik_joint_mappings(self) -> dict:

        return dict(getattr(self._gmr, "ik_joint_mappings", {}))
