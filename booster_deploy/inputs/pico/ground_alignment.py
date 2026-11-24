
from __future__ import annotations

from collections import deque
from collections.abc import Iterable

import numpy as np

from booster_deploy.types import HumanFrame


DEFAULT_GROUND_JOINTS = ("Left_Foot", "Right_Foot", "Left_Ankle", "Right_Ankle")


class PicoGroundAligner:

    def __init__(
        self,
        *,
        calibration_frames: int = 30,
        target_ground_z: float = 0.0,
        ground_z: float | None = None,
        joint_names: Iterable[str] = DEFAULT_GROUND_JOINTS,
    ) -> None:

        self.calibration_frames = max(int(calibration_frames), 1)
        self.target_ground_z = float(target_ground_z)
        self.ground_z = None if ground_z is None else float(ground_z)
        self.joint_names = tuple(joint_names)
        self._samples: deque[float] = deque(maxlen=self.calibration_frames)

    @property
    def is_calibrated(self) -> bool:

        return self.ground_z is not None

    @property
    def sample_count(self) -> int:

        return len(self._samples)

    @property
    def z_offset(self) -> float | None:

        if self.ground_z is None:
            return None
        return self.target_ground_z - self.ground_z

    def align(self, frame: HumanFrame) -> HumanFrame:

        ground_sample = self._lowest_joint_z(frame)
        if ground_sample is not None and self.ground_z is None:
            self._samples.append(ground_sample)
            if len(self._samples) >= self.calibration_frames:
                self.ground_z = float(np.median(np.asarray(self._samples, dtype=np.float64)))

        if self.ground_z is None:
            return self._translate_z(frame, self.target_ground_z - ground_sample) if ground_sample is not None else self._copy_frame(frame)
        return self._translate_z(frame, self.target_ground_z - self.ground_z)

    def _lowest_joint_z(self, frame: HumanFrame) -> float | None:

        z_values = [
            float(np.asarray(frame[name][0], dtype=np.float64)[2])
            for name in self.joint_names
            if name in frame
        ]
        if not z_values:
            return None
        return min(z_values)

    @staticmethod
    def _copy_frame(frame: HumanFrame) -> HumanFrame:

        return {
            name: (
                np.asarray(pos, dtype=np.float64).copy(),
                np.asarray(quat, dtype=np.float64).copy(),
            )
            for name, (pos, quat) in frame.items()
        }

    @staticmethod
    def _translate_z(frame: HumanFrame, dz: float) -> HumanFrame:

        offset = np.asarray([0.0, 0.0, float(dz)], dtype=np.float64)
        return {
            name: (
                np.asarray(pos, dtype=np.float64) + offset,
                np.asarray(quat, dtype=np.float64).copy(),
            )
            for name, (pos, quat) in frame.items()
        }
