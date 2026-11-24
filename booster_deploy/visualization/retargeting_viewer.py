
from __future__ import annotations

import time
from collections.abc import Sequence

import mujoco
import mujoco.viewer
import numpy as np

from booster_deploy.utils.project_path import resolve_project_path

from booster_deploy.types import HumanFrame, RetargetFrame
from booster_deploy.visualization.drawers import MujocoSceneDrawer
from booster_deploy.visualization.layers import HumanFrameAxesLayer, HumanSkeletonLayer, VisualizationLayer
from booster_deploy.visualization.robot_support import RobotSupportBaseLayer, compute_robot_support_state


class RetargetingMujocoViewer:

    def __init__(
        self,
        xml_path: str,
        *,
        robot_root_body: str = "pelvis",
        show_human: bool = True,
        show_scaled_human: bool = True,
        follow_robot: bool = True,
        camera_distance: float | None = None,
        camera_azimuth: float | None = None,
        camera_elevation: float | None = None,
        show_robot_support_base: bool = False,
        support_ground_geom_names: Sequence[str] | None = None,
        key_callback=None,
    ) -> None:

        self.model = mujoco.MjModel.from_xml_path(str(resolve_project_path(xml_path)))
        self.data = mujoco.MjData(self.model)
        self.robot_root_body = robot_root_body
        self.show_human = show_human
        self.show_scaled_human = show_scaled_human
        self.follow_robot = bool(follow_robot)
        self.show_robot_support_base = bool(show_robot_support_base)
        self.support_ground_geom_names = tuple(support_ground_geom_names) if support_ground_geom_names is not None else None
        self._closed = False
        self.viewer = mujoco.viewer.launch_passive(
            model=self.model,
            data=self.data,
            key_callback=key_callback,
            show_left_ui=False,
            show_right_ui=False,
        )
        self.viewer.cam.distance = 2.2 if camera_distance is None else float(camera_distance)
        self.viewer.cam.azimuth = 135.0 if camera_azimuth is None else float(camera_azimuth)
        self.viewer.cam.elevation = -15.0 if camera_elevation is None else float(camera_elevation)
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

    def sync(self) -> None:

        if self._closed:
            return
        self.viewer.sync()

    def step(
        self,
        retarget_frame: RetargetFrame,
        *,
        human_frame: HumanFrame | None = None,
        scaled_human_frame: HumanFrame | None = None,
        skeleton_connections: Sequence[tuple[str, str]] | None = None,
        layers: Sequence[VisualizationLayer] | None = None,
        rate_limit_s: float | None = None,
    ) -> None:

        qpos = np.asarray(retarget_frame.qpos, dtype=np.float64)
        self.data.qpos[: qpos.shape[0]] = qpos
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        if self.follow_robot:
            self._follow_robot()
        self._drawer.clear()
        active_layers = list(
            self._build_layers(
                human_frame=human_frame,
                scaled_human_frame=scaled_human_frame,
                skeleton_connections=skeleton_connections,
                extra_layers=layers,
            )
        )
        if self.show_robot_support_base:
            support_state = compute_robot_support_state(
                self.model,
                self.data,
                root_body_name=self.robot_root_body,
                ground_geom_names=self.support_ground_geom_names,
            )
            active_layers.append(RobotSupportBaseLayer(support_state))
        for layer in active_layers:
            layer.draw(self._drawer)

        self.viewer.sync()
        if rate_limit_s is not None and rate_limit_s > 0.0:
            time.sleep(rate_limit_s)

    def _follow_robot(self) -> None:

        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.robot_root_body)
        if body_id >= 0:
            self.viewer.cam.lookat[:] = self.data.xpos[body_id]
        else:
            self.viewer.cam.lookat[:] = [self.data.qpos[0], self.data.qpos[1], 0.8]

    def _build_layers(
        self,
        *,
        human_frame: HumanFrame | None,
        scaled_human_frame: HumanFrame | None,
        skeleton_connections: Sequence[tuple[str, str]] | None,
        extra_layers: Sequence[VisualizationLayer] | None,
    ) -> tuple[VisualizationLayer, ...]:

        layers: list[VisualizationLayer] = []
        if self.show_human and human_frame is not None:
            layers.append(HumanSkeletonLayer(human_frame, connections=skeleton_connections))
        if self.show_scaled_human and scaled_human_frame is not None:
            layers.append(HumanFrameAxesLayer(scaled_human_frame))
        if extra_layers is not None:
            layers.extend(extra_layers)
        return tuple(layers)

    def _draw_human_points(
        self,
        human_frame: HumanFrame,
        *,
        skeleton_connections: Sequence[tuple[str, str]] | None,
        point_rgba: tuple[float, float, float, float],
        line_rgba: tuple[float, float, float, float],
        point_size: float,
    ) -> None:

        self._drawer.draw_human_skeleton(
            human_frame,
            connections=skeleton_connections,
            point_rgba=point_rgba,
            line_rgba=line_rgba,
            point_size=point_size,
            line_width=0.004,
        )

    def _draw_human_frames(self, human_frame: HumanFrame, *, axis_size: float) -> None:

        self._drawer.draw_human_frame_axes(human_frame, axis_size=axis_size, width=0.006)

    def _draw_sphere(
        self,
        pos: np.ndarray,
        *,
        size: float,
        rgba: tuple[float, float, float, float],
    ) -> None:

        self._drawer.draw_sphere(pos, size=size, rgba=rgba)

    def _draw_line(
        self,
        start: np.ndarray,
        end: np.ndarray,
        *,
        rgba: tuple[float, float, float, float],
        width: float,
    ) -> None:

        self._drawer.draw_line(start, end, rgba=rgba, width=width)
