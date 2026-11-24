
from __future__ import annotations

import time
import warnings
from collections.abc import Sequence

import mujoco
import numpy as np

from booster_deploy.types import HumanFrame, RetargetFrame
from booster_deploy.utils.project_path import resolve_project_path
from booster_deploy.visualization.viser_camera import ViserCameraController, fit_camera_to_points
from booster_deploy.visualization.layers import (
    DEFAULT_HAND_POINT_SIZE,
    DEFAULT_HAND_LINE_WIDTH,
    DEFAULT_HAND_TRANSITION_LINE_WIDTH,
    DEFAULT_HAND_WRIST_LINE_WIDTH,
    DEFAULT_HUMAN_POINT_SIZE,
    VisualizationLayer,
)
from booster_deploy.visualization.drawers import (
    is_hand_bone,
    is_hand_transition_bone,
    is_hand_wrist_bone,
    resolve_joint_point_size,
)
from booster_deploy.visualization.robot_support import compute_robot_support_state


def _view_direction_from_mujoco_angles(
    azimuth_deg: float,
    elevation_deg: float,
) -> np.ndarray:

    azimuth = np.deg2rad(float(azimuth_deg) - 180.0)
    elevation = np.deg2rad(-float(elevation_deg))
    horizontal = np.cos(elevation)
    return np.array(
        [
            horizontal * np.cos(azimuth),
            horizontal * np.sin(azimuth),
            np.sin(elevation),
        ],
        dtype=np.float64,
    )


def _rgb(rgba: tuple[float, float, float, float]) -> tuple[int, int, int]:
    return tuple(int(np.clip(channel, 0.0, 1.0) * 255.0) for channel in rgba[:3])


def _safe_name(name: str) -> str:
    return str(name).replace("/", "_").replace(" ", "_")


class _HumanFrameViserOverlay:

    def __init__(
        self,
        server,
        *,
        prefix: str,
        point_rgba: tuple[float, float, float, float],
        line_rgba: tuple[float, float, float, float],
        point_size: float,
        line_width: float,
        hand_point_size: float | None = None,
        hand_line_width: float | None = None,
        hand_transition_line_width: float | None = None,
        hand_wrist_line_width: float | None = None,
        show_axes: bool = False,
    ) -> None:
        self.server = server
        self.prefix = prefix.rstrip("/")
        self.point_rgba = point_rgba
        self.line_rgba = line_rgba
        self.point_size = float(point_size)
        self.line_width = float(line_width)
        self.hand_point_size = None if hand_point_size is None else float(hand_point_size)
        self.hand_line_width = None if hand_line_width is None else float(hand_line_width)
        self.hand_transition_line_width = (
            None if hand_transition_line_width is None else float(hand_transition_line_width)
        )
        self.hand_wrist_line_width = None if hand_wrist_line_width is None else float(hand_wrist_line_width)
        self.show_axes = bool(show_axes)
        self._point_handles: dict[str, object] = {}
        self._axis_handles: dict[str, object] = {}
        self._body_line_handle = None
        self._hand_line_handle = None

    def update(
        self,
        frame: HumanFrame | None,
        *,
        connections: Sequence[tuple[str, str]] | None,
    ) -> None:
        active_names = set(frame or {})
        for name, handle in self._point_handles.items():
            handle.visible = name in active_names
        for name, handle in self._axis_handles.items():
            handle.visible = self.show_axes and name in active_names

        if frame is None:
            for kind in ("body", "hand"):
                handle = getattr(self, f"_{kind}_line_handle", None)
                if handle is not None:
                    handle.visible = False
            return

        for name, (position, quaternion) in frame.items():
            point = self._point_handles.get(name)
            if point is None:
                point = self.server.scene.add_icosphere(
                    f"{self.prefix}/joints/{_safe_name(name)}",
                    radius=resolve_joint_point_size(name, self.point_size, self.hand_point_size),
                    color=_rgb(self.point_rgba),
                    opacity=float(self.point_rgba[3]),
                )
                self._point_handles[name] = point
            point.position = np.asarray(position, dtype=np.float64)
            point.visible = True
            if self.show_axes:
                axis = self._axis_handles.get(name)
                if axis is None:
                    axis = self.server.scene.add_frame(
                        f"{self.prefix}/axes/{_safe_name(name)}",
                        axes_length=0.08,
                        axes_radius=0.003,
                    )
                    self._axis_handles[name] = axis
                axis.position = np.asarray(position, dtype=np.float64)
                axis.wxyz = np.asarray(quaternion, dtype=np.float64)
                axis.visible = True

        segments_by_kind: dict[str, list[list[np.ndarray]]] = {
            "body": [],
            "hand_wrist": [],
            "hand_transition": [],
            "hand": [],
        }
        if connections is not None:
            for parent, child in connections:
                if parent in frame and child in frame:
                    if self.hand_line_width is None:
                        kind = "body"
                    elif self.hand_wrist_line_width is not None and is_hand_wrist_bone(parent, child):
                        kind = "hand_wrist"
                    elif self.hand_transition_line_width is not None and is_hand_transition_bone(parent, child):
                        kind = "hand_transition"
                    elif not is_hand_bone(parent, child):
                        kind = "body"
                    else:
                        kind = "hand"
                    segments_by_kind[kind].append(
                        [
                            np.asarray(frame[parent][0], dtype=np.float64),
                            np.asarray(frame[child][0], dtype=np.float64),
                        ]
                    )
        for kind, segments in segments_by_kind.items():
            handle_name = f"{self.prefix}/bones/{kind}"
            if kind == "body" or self.hand_line_width is None:
                line_width = self.line_width
            elif kind in {"hand_wrist", "hand_transition"}:
                line_width = (
                    self.hand_wrist_line_width if kind == "hand_wrist" else self.hand_transition_line_width
                ) or self.line_width
            else:
                line_width = self.hand_line_width
            handle = getattr(self, f"_{kind}_line_handle", None)
            if not segments:
                if handle is not None:
                    handle.visible = False
                continue
            points = np.asarray(segments, dtype=np.float32)
            if handle is None:
                handle = self.server.scene.add_line_segments(
                    handle_name,
                    points=points,
                    colors=_rgb(self.line_rgba),
                    line_width=max(line_width * 500.0, 1.0),
                )
                setattr(self, f"_{kind}_line_handle", handle)
            else:
                handle.points = points
            handle.visible = True

    def close(self) -> None:
        handles = list(self._point_handles.values()) + list(self._axis_handles.values())
        for kind in ("body", "hand"):
            line_handle = getattr(self, f"_{kind}_line_handle", None)
            if line_handle is not None:
                handles.append(line_handle)
        for handle in handles:
            try:
                handle.remove()
            except Exception:
                pass


class HumanFrameViserViewer:

    def __init__(
        self,
        *,
        title: str = "HumanFrame Viewer",
        skeleton_connections: Sequence[tuple[str, str]] | None = None,
        fit_camera: bool = True,
        min_camera_distance: float = 2.6,
        key_callback=None,
        host: str = "0.0.0.0",
        port: int = 8080,
        release_camera_after_first_frame: bool = False,
        camera_azimuth: float | None = None,
        camera_elevation: float | None = None,
    ) -> None:
        import viser

        self.server = viser.ViserServer(host=host, port=int(port), label=title)
        # Collapsible panel (starts expanded in viser); the dashboard clicks the
        # "Collapse sidebar" chevron inside the iframe.  Keep the label: that title
        # is how the dashboard locates the chevron.
        self.server.gui.configure_theme(control_layout="collapsible")
        # Infinite grid: the operator walks around, the floor must not end.
        self.server.scene.add_grid(
            "/ground", width=80.0, height=80.0, cell_size=1.0, infinite_grid=True
        )
        self.skeleton_connections = skeleton_connections
        self._camera_angles = (
            None if camera_azimuth is None
            else (float(camera_azimuth), float(camera_elevation if camera_elevation is not None else 0.0))
        )
        self._camera_angles_applied = False
        self.fit_camera = bool(fit_camera)
        self._release_camera_after_first_frame = bool(release_camera_after_first_frame)
        self.min_camera_distance = float(min_camera_distance)
        self._last_camera_target = np.array([0.0, 0.0, 0.8], dtype=np.float64)
        self._last_camera_distance = self.min_camera_distance
        self._closed = False
        self._warned_custom_layers = False
        self._camera = ViserCameraController(
            self.server,
            enabled=self.fit_camera,
            fallback_distance=self.min_camera_distance,
        )
        self._camera.update_target(
            self._last_camera_target,
            distance=self._last_camera_distance,
        )
        self._overlay = _HumanFrameViserOverlay(
            self.server,
            prefix="/human",
            point_rgba=(1.0, 0.95, 0.82, 1.0),
            line_rgba=(1.0, 0.55, 0.12, 0.9),
            point_size=0.025,
            hand_point_size=DEFAULT_HAND_POINT_SIZE,
            hand_line_width=DEFAULT_HAND_LINE_WIDTH,
            hand_transition_line_width=DEFAULT_HAND_TRANSITION_LINE_WIDTH,
            hand_wrist_line_width=DEFAULT_HAND_WRIST_LINE_WIDTH,
            line_width=0.014,
        )
        if key_callback is not None:
            warnings.warn("Viser does not forward native MuJoCo key callbacks; use the source's other controls.")
        print(f"Viser HumanFrame viewer: http://localhost:{int(port)}", flush=True)

    def is_running(self) -> bool:
        return not self._closed

    def step(
        self,
        human_frame: HumanFrame,
        *,
        layers: Sequence[VisualizationLayer] | None = None,
        rate_limit_s: float | None = None,
    ) -> None:
        if layers:
            if not self._warned_custom_layers:
                warnings.warn("Custom MuJoCo visualization layers are not rendered by HumanFrameViserViewer.")
                self._warned_custom_layers = True
        self._overlay.update(human_frame, connections=self.skeleton_connections)
        if self.fit_camera:
            positions = np.asarray(
                [pose[0] for pose in (human_frame or {}).values()],
                dtype=np.float64,
            )
            target, distance = fit_camera_to_points(
                positions,
                min_distance=self.min_camera_distance,
            )
            self._last_camera_target = target.copy()
            self._last_camera_distance = float(distance)
            if self._camera_angles is not None and not self._camera_angles_applied:
                # Apply the configured direction once, on the first real frame (the
                # ground-alignment offset is only known by then); fitting keeps
                # running afterwards, so the direction sticks but the body is followed.
                self._camera.set_initial_view(
                    target,
                    distance=float(distance),
                    view_direction=_view_direction_from_mujoco_angles(*self._camera_angles),
                )
                self._camera_angles_applied = True
            self._camera.update_target(target, distance=distance)
        if self._release_camera_after_first_frame:
            self.release_camera_control()
            self._release_camera_after_first_frame = False
        if rate_limit_s is not None and rate_limit_s > 0.0:
            time.sleep(rate_limit_s)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._camera.set_enabled(False)
        self._overlay.close()
        self.server.stop()

    def release_camera_control(self) -> None:

        self.fit_camera = False
        self._camera.set_initial_view(
            self._last_camera_target,
            distance=self._last_camera_distance,
            view_direction=_view_direction_from_mujoco_angles(135.0, -15.0),
        )
        self._camera.set_enabled(False)


class _SupportViserOverlay:
    def __init__(self, server) -> None:
        self.server = server
        self._projection = None
        self._support_points: list[object] = []
        self._lines = None

    def update(self, state) -> None:
        if self._projection is None:
            self._projection = self.server.scene.add_icosphere(
                "/support/com_projection", radius=0.025, color=(255, 31, 20)
            )
        self._projection.position = state.com_projection
        while len(self._support_points) < len(state.support_points):
            idx = len(self._support_points)
            self._support_points.append(
                self.server.scene.add_icosphere(
                    f"/support/point_{idx}", radius=0.014, color=(13, 255, 64)
                )
            )
        for idx, handle in enumerate(self._support_points):
            handle.visible = idx < len(state.support_points)
            if handle.visible:
                handle.position = state.support_points[idx]

        lines = [[state.com, state.com_projection]]
        hull = state.support_hull
        if hull.shape[0] == 2:
            lines.append([hull[0], hull[1]])
        elif hull.shape[0] >= 3:
            lines.extend([hull[idx], hull[(idx + 1) % hull.shape[0]]] for idx in range(hull.shape[0]))
        points = np.asarray(lines, dtype=np.float32)
        colors = np.zeros((len(lines), 2, 3), dtype=np.uint8)
        colors[0, :, :] = (255, 31, 20)
        if len(lines) > 1:
            colors[1:, :, :] = (13, 255, 64)
        if self._lines is None:
            self._lines = self.server.scene.add_line_segments(
                "/support/lines", points=points, colors=colors, line_width=3.0
            )
        else:
            self._lines.points = points
            self._lines.colors = colors

    def close(self) -> None:
        handles = list(self._support_points)
        if self._projection is not None:
            handles.append(self._projection)
        if self._lines is not None:
            handles.append(self._lines)
        for handle in handles:
            try:
                handle.remove()
            except Exception:
                pass


class RetargetingViserViewer:

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
        host: str = "0.0.0.0",
        port: int = 8080,
        show_joint_axes: bool = False,
    ) -> None:
        from booster_deploy.visualization.viser import ViserViewerAdapter

        self.model = mujoco.MjModel.from_xml_path(str(resolve_project_path(xml_path)))
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.robot_root_body = robot_root_body
        self._robot_root_body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            self.robot_root_body,
        )
        if self._robot_root_body_id < 0:
            warnings.warn(
                f"Robot root body {self.robot_root_body!r} was not found; "
                "Viser camera follow will use the free-root position."
            )
        self.show_human = bool(show_human)
        self.show_scaled_human = bool(show_scaled_human)
        self.follow_robot = bool(follow_robot)
        self.show_robot_support_base = bool(show_robot_support_base)
        self.support_ground_geom_names = tuple(support_ground_geom_names) if support_ground_geom_names else None
        self.viewer = ViserViewerAdapter(
            self.model,
            self.data,
            host=host,
            port=int(port),
            label="Booster Retargeting Viewer",
            show_joint_axes=show_joint_axes,
        )
        self.viewer.set_camera_follow(self.follow_robot)
        if camera_distance is not None:
            self.viewer.set_initial_camera_view(
                self._camera_target(),
                distance=float(camera_distance),
                view_direction=_view_direction_from_mujoco_angles(
                    135.0 if camera_azimuth is None else float(camera_azimuth),
                    -15.0 if camera_elevation is None else float(camera_elevation),
                ),
            )
        self._closed = False
        self._warned_custom_layers = False
        self._human = _HumanFrameViserOverlay(
            self.viewer.server,
            prefix="/human/raw",
            point_rgba=(0.0, 0.85, 1.0, 0.85),
            line_rgba=(0.0, 0.75, 1.0, 0.45),
            point_size=DEFAULT_HUMAN_POINT_SIZE,
            hand_point_size=DEFAULT_HAND_POINT_SIZE,
            line_width=0.004,
        )
        self._scaled_human = _HumanFrameViserOverlay(
            self.viewer.server,
            prefix="/human/scaled",
            point_rgba=(1.0, 0.72, 0.12, 0.8),
            line_rgba=(1.0, 0.48, 0.08, 0.55),
            point_size=0.014,
            hand_point_size=DEFAULT_HAND_POINT_SIZE,
            hand_line_width=DEFAULT_HAND_LINE_WIDTH,
            hand_transition_line_width=DEFAULT_HAND_TRANSITION_LINE_WIDTH,
            hand_wrist_line_width=DEFAULT_HAND_WRIST_LINE_WIDTH,
            line_width=0.003,
            show_axes=True,
        )
        self._support = _SupportViserOverlay(self.viewer.server) if self.show_robot_support_base else None
        if key_callback is not None:
            warnings.warn("Viser does not forward native MuJoCo key callbacks; use the source's other controls.")

    def is_running(self) -> bool:
        return not self._closed and self.viewer.is_running()

    def sync(self) -> None:
        if not self._closed:
            self.viewer.sync(camera_target=self._camera_target())

    def _camera_target(self) -> np.ndarray:
        if self._robot_root_body_id >= 0:
            return np.asarray(
                self.data.xpos[self._robot_root_body_id],
                dtype=np.float64,
            )
        return np.array(
            [float(self.data.qpos[0]), float(self.data.qpos[1]), 0.8],
            dtype=np.float64,
        )

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
        self._human.update(human_frame if self.show_human else None, connections=skeleton_connections)
        self._scaled_human.update(
            scaled_human_frame if self.show_scaled_human else None,
            connections=skeleton_connections,
        )
        if self._support is not None:
            self._support.update(
                compute_robot_support_state(
                    self.model,
                    self.data,
                    root_body_name=self.robot_root_body,
                    ground_geom_names=self.support_ground_geom_names,
                )
            )
        if layers:
            if not self._warned_custom_layers:
                warnings.warn("Custom MuJoCo visualization layers are not rendered by RetargetingViserViewer.")
                self._warned_custom_layers = True
        self.viewer.sync(camera_target=self._camera_target())
        if rate_limit_s is not None and rate_limit_s > 0.0:
            time.sleep(rate_limit_s)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._human.close()
        self._scaled_human.close()
        if self._support is not None:
            self._support.close()
        self.viewer.close()


__all__ = ["HumanFrameViserViewer", "RetargetingViserViewer"]
