
from __future__ import annotations

import time
from collections.abc import Sequence

import mujoco
import mujoco.viewer
import numpy as np

from booster_deploy.types import HumanFrame
from booster_deploy.visualization.drawers import MujocoSceneDrawer
from booster_deploy.visualization.layers import (
    DEFAULT_HAND_POINT_SIZE,
    DEFAULT_HAND_LINE_WIDTH,
    DEFAULT_HAND_TRANSITION_LINE_WIDTH,
    DEFAULT_HAND_WRIST_LINE_WIDTH,
    DEFAULT_HUMAN_POINT_SIZE,
    HumanSkeletonLayer,
    VisualizationLayer,
)


def create_overlay_world_model(title: str = "HumanFrame Viewer") -> mujoco.MjModel:

    xml = f"""
<mujoco model="{title}">
  <visual>
    <global offwidth="1280" offheight="720"/>
    <map znear="0.01" zfar="50"/>
    <headlight ambient="0.15 0.15 0.15" diffuse="0.2 0.2 0.2" specular="0 0 0"/>
  </visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.86 0.86 0.84" rgb2="0.68 0.68 0.66" width="512" height="512"/>
    <material name="ground_mat" texture="grid" texrepeat="8 8" reflectance="0.12"/>
  </asset>
  <worldbody>
    <light pos="2 -2 5" dir="-0.3 0.3 -1" diffuse="0.7 0.7 0.7" specular="0.3 0.3 0.3"/>
    <light pos="-2 2 4" dir="0.3 -0.3 -1" diffuse="0.4 0.4 0.4" specular="0.1 0.1 0.1"/>
    <geom name="ground" type="plane" size="8 8 0.1" material="ground_mat"/>
  </worldbody>
</mujoco>
"""
    return mujoco.MjModel.from_xml_string(xml)


def human_frame_positions(frame: HumanFrame) -> np.ndarray:

    if not frame:
        return np.zeros((0, 3), dtype=np.float64)
    return np.stack([np.asarray(pos, dtype=np.float64).reshape(3) for pos, _quat in frame.values()], axis=0)


class HumanFrameMujocoViewer:

    def __init__(
        self,
        *,
        title: str = "HumanFrame Viewer",
        skeleton_connections: Sequence[tuple[str, str]] | None = None,
        fit_camera: bool = True,
        min_camera_distance: float = 2.6,
        key_callback=None,
        release_camera_after_first_frame: bool = False,
    ) -> None:

        self.model = create_overlay_world_model(title)
        self.data = mujoco.MjData(self.model)
        self.skeleton_connections = skeleton_connections
        self.fit_camera = fit_camera
        self._release_camera_after_first_frame = bool(release_camera_after_first_frame)
        self.min_camera_distance = float(min_camera_distance)
        self._closed = False
        self.viewer = mujoco.viewer.launch_passive(
            model=self.model,
            data=self.data,
            key_callback=key_callback,
            show_left_ui=False,
            show_right_ui=False,
        )
        self.viewer.cam.distance = self.min_camera_distance
        self.viewer.cam.azimuth = 135.0
        self.viewer.cam.elevation = -15.0
        self._drawer = MujocoSceneDrawer(self.viewer.user_scn)

    def is_running(self) -> bool:

        if self._closed:
            return False
        return self.viewer.is_running()

    def close(self) -> None:

        if self._closed:
            return
        self._closed = True
        self.viewer.close()

    def release_camera_control(self) -> None:

        self.fit_camera = False

    def step(
        self,
        human_frame: HumanFrame,
        *,
        layers: Sequence[VisualizationLayer] | None = None,
        rate_limit_s: float | None = None,
    ) -> None:

        mujoco.mj_forward(self.model, self.data)
        if self.fit_camera:
            self._fit_camera(human_frame_positions(human_frame))
        self._drawer.clear()
        HumanSkeletonLayer(
            human_frame,
            connections=self.skeleton_connections,
            point_rgba=(1.0, 0.95, 0.82, 1.0),
            line_rgba=(1.0, 0.55, 0.12, 0.9),
            point_size=0.025,
            hand_point_size=DEFAULT_HAND_POINT_SIZE,
            hand_line_width=DEFAULT_HAND_LINE_WIDTH,
            hand_transition_line_width=DEFAULT_HAND_TRANSITION_LINE_WIDTH,
            hand_wrist_line_width=DEFAULT_HAND_WRIST_LINE_WIDTH,
            line_width=0.014,
            bone_style="capsule",
        ).draw(self._drawer)
        if layers is not None:
            for layer in layers:
                layer.draw(self._drawer)
        self.viewer.sync()
        if self._release_camera_after_first_frame:
            self.release_camera_control()
            self._release_camera_after_first_frame = False
        if rate_limit_s is not None and rate_limit_s > 0.0:
            time.sleep(rate_limit_s)

    def _fit_camera(self, positions: np.ndarray) -> None:

        pos = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
        if pos.size == 0:
            self.viewer.cam.lookat[:] = [0.0, 0.0, 0.8]
            self.viewer.cam.distance = self.min_camera_distance
            return
        valid = pos[np.all(np.isfinite(pos), axis=1)]
        if valid.shape[0] == 0:
            self.viewer.cam.lookat[:] = [0.0, 0.0, 0.8]
            self.viewer.cam.distance = self.min_camera_distance
            return
        mins = np.min(valid, axis=0)
        maxs = np.max(valid, axis=0)
        center = 0.5 * (mins + maxs)
        extent = float(np.max(maxs - mins))
        self.viewer.cam.lookat[:] = center
        self.viewer.cam.distance = max(self.min_camera_distance, extent * 2.2)
        self.viewer.cam.azimuth = 135.0
        self.viewer.cam.elevation = -15.0
