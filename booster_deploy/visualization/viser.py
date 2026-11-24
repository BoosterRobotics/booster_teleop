
from __future__ import annotations

import mujoco
import numpy as np
import trimesh
import viser


def _view_direction_from_angles(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    """MuJoCo-style azimuth/elevation -> unit view direction (same as viser_viewer)."""
    azimuth = np.deg2rad(float(azimuth_deg) - 180.0)
    elevation = np.deg2rad(-float(elevation_deg))
    horizontal = np.cos(elevation)
    direction = np.array(
        [horizontal * np.cos(azimuth), horizontal * np.sin(azimuth), np.sin(elevation)],
        dtype=np.float64,
    )
    return direction / max(float(np.linalg.norm(direction)), 1e-9)
from scipy.spatial.transform import Rotation as R
from dataclasses import dataclass
from typing import Literal, Optional

from booster_deploy.visualization.viser_camera import ViserCameraController


@dataclass
class _MeshHandle:
    handle: object
    geom_id: int


class ViserViewerAdapter:

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        host: str = "0.0.0.0",
        port: int = 8080,
        label: str = "Booster Deploy - Viser Viewer",
        show_joint_axes: bool = False,
        show_pause_control: bool = True,
        control_layout: Literal["floating", "collapsible", "fixed"] = "collapsible",
        control_width: Literal["small", "medium", "large"] = "medium",
        camera_azimuth: float | None = None,
        camera_elevation: float | None = None,
        camera_distance: float | None = None,
        camera_follow: bool = False,
    ):
        self.model = model
        self.data = data
        self._running = True
        self._show_joint_axes = show_joint_axes
        self._show_pause_control = bool(show_pause_control)


        self.server = viser.ViserServer(host=host, port=port, label=label)
        self.server.gui.configure_theme(
            control_layout=control_layout,
            control_width=control_width,
        )
        print(f"🌐 Viser viewer started at http://localhost:{port}")


        self._setup_scene()



        self.cam = _CameraControl()
        self._camera_controller = ViserCameraController(self.server)


        self._robot_meshes: list[_MeshHandle] = []
        self._create_robot_meshes()


        self._ghost_meshes: dict[str, list[_MeshHandle]] = {}
        self._ghost_data: dict[str, mujoco.MjData] = {}


        self._joint_frames: dict[int, object] = {}
        if self._show_joint_axes:
            self._create_joint_frames()


        self._setup_gui_controls()

        # Camera: applied last, because follow needs the GUI checkbox and both need
        # the camera controller that was created earlier.
        if camera_follow:
            self.set_camera_follow(True)
        if camera_azimuth is not None:
            self.set_initial_camera_view(
                np.array([0.0, 0.0, 0.8], dtype=np.float64),
                distance=float(camera_distance if camera_distance is not None else 2.4),
                view_direction=_view_direction_from_angles(
                    float(camera_azimuth),
                    float(camera_elevation if camera_elevation is not None else -10.0),
                ),
            )

    def _setup_scene(self):
        self.server.scene.add_grid(
            name="/ground",
            # Infinite grid so a walking robot cannot step off the visible floor.
            width=80.0,
            height=80.0,
            cell_size=1.0,
            cell_thickness=1.0,
            infinite_grid=True,
        )
        self.server.scene.add_frame(
            name="/origin",
            axes_length=0.5,
            axes_radius=0.015,
        )

    def _setup_gui_controls(self):
        # Collapsed by default: the operator wants the scene, not the panel.  The
        # header stays so Pause / Camera Follow are one click away.
        with self.server.gui.add_folder("Controls", expand_by_default=False):
            self._pause_checkbox = self.server.gui.add_checkbox(
                "Pause",
                initial_value=False,
                visible=self._show_pause_control,
            )
            self._camera_follow_checkbox = self.server.gui.add_checkbox(
                "Camera Follow", initial_value=False
            )

    def _create_robot_meshes(self):
        for i in range(self.model.ngeom):
            geom_type = int(self.model.geom_type[i])
            geom_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i)


            name_lower = (geom_name or "").lower()
            if "floor" in name_lower or "ground" in name_lower:
                continue
            if geom_type == int(mujoco.mjtGeom.mjGEOM_PLANE):
                continue



            is_noncolliding_mesh = (
                geom_type == int(mujoco.mjtGeom.mjGEOM_MESH)
                and int(self.model.geom_contype[i]) == 0
                and int(self.model.geom_conaffinity[i]) == 0
            )
            if self.model.geom_group[i] != 1 and (
                not geom_name or "visual" not in name_lower
            ) and not is_noncolliding_mesh:
                continue

            mesh = self._create_mesh_from_geom(i)
            if mesh is None:
                continue


            rgba = np.array(self.model.geom_rgba[i], dtype=float)
            rgba = np.clip(rgba, 0.0, 1.0)
            color = tuple(int(c * 255) for c in rgba)
            mesh.visual = trimesh.visual.ColorVisuals(
                vertex_colors=[color] * len(mesh.vertices)
            )

            handle = self.server.scene.add_mesh_trimesh(
                name=f"/robot/geom_{i}",
                mesh=mesh,
                wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
                position=np.zeros(3),
            )
            self._robot_meshes.append(_MeshHandle(handle=handle, geom_id=i))

    def _create_mesh_from_geom(self, geom_id: int) -> Optional[trimesh.Trimesh]:
        geom_type = int(self.model.geom_type[geom_id])
        size = np.array(self.model.geom_size[geom_id], dtype=float)

        if geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
            return trimesh.creation.box(extents=size * 2)
        elif geom_type == int(mujoco.mjtGeom.mjGEOM_SPHERE):
            return trimesh.creation.icosphere(radius=float(size[0]), subdivisions=2)
        elif geom_type == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
            radius = float(size[0])
            height = float(size[1]) * 2
            return trimesh.creation.cylinder(radius=radius, height=height)
        elif geom_type == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
            radius = float(size[0])
            height = float(size[1]) * 2
            return trimesh.creation.capsule(radius=radius, height=height)
        elif geom_type == int(mujoco.mjtGeom.mjGEOM_MESH):
            mesh_id = int(self.model.geom_dataid[geom_id])
            if mesh_id < 0:
                return None
            vert_start = int(self.model.mesh_vertadr[mesh_id])
            vert_num = int(self.model.mesh_vertnum[mesh_id])
            face_start = int(self.model.mesh_faceadr[mesh_id])
            face_num = int(self.model.mesh_facenum[mesh_id])
            vertices = self.model.mesh_vert[vert_start : vert_start + vert_num]
            faces = self.model.mesh_face[face_start : face_start + face_num]
            return trimesh.Trimesh(vertices=vertices, faces=faces)

        return None

    def _create_joint_frames(self):
        for j in range(self.model.njnt):
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)
            frame_name = joint_name or f"joint_{j}"
            handle = self.server.scene.add_frame(
                name=f"/joints/{frame_name}",
                axes_length=0.15,
                axes_radius=0.004,
            )
            self._joint_frames[j] = handle

    def _update_robot_meshes(self):
        for mh in self._robot_meshes:
            geom_pos = np.array(self.data.geom_xpos[mh.geom_id], dtype=float)
            geom_mat = np.array(self.data.geom_xmat[mh.geom_id], dtype=float).reshape(
                3, 3
            )
            quat_xyzw = R.from_matrix(geom_mat).as_quat()
            quat_wxyz = np.array(
                [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=float
            )
            mh.handle.wxyz = quat_wxyz
            mh.handle.position = geom_pos

    def _update_joint_frames(self):
        for j, handle in self._joint_frames.items():
            body_id = int(self.model.jnt_bodyid[j])
            pos = np.array(self.data.xpos[body_id], dtype=float)
            mat = np.array(self.data.xmat[body_id], dtype=float).reshape(3, 3)
            quat_xyzw = R.from_matrix(mat).as_quat()
            quat_wxyz = np.array(
                [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=float
            )
            handle.position = pos
            handle.wxyz = quat_wxyz

    def add_ghost_robot(self, ghost_id: str, ghost_data: mujoco.MjData, rgba: np.ndarray):
        if ghost_id in self._ghost_meshes:

            self._ghost_data[ghost_id] = ghost_data
            self._update_ghost_meshes(ghost_id, rgba)
            return


        ghost_meshes = []
        for i in range(self.model.ngeom):
            geom_type = int(self.model.geom_type[i])
            geom_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i)


            name_lower = (geom_name or "").lower()
            if "floor" in name_lower or "ground" in name_lower:
                continue
            if geom_type == int(mujoco.mjtGeom.mjGEOM_PLANE):
                continue

            is_noncolliding_mesh = (
                geom_type == int(mujoco.mjtGeom.mjGEOM_MESH)
                and int(self.model.geom_contype[i]) == 0
                and int(self.model.geom_conaffinity[i]) == 0
            )
            if self.model.geom_group[i] != 1 and (
                not geom_name or "visual" not in name_lower
            ) and not is_noncolliding_mesh:
                continue

            mesh = self._create_mesh_from_geom(i)
            if mesh is None:
                continue


            color = tuple(int(c * 255) for c in rgba)
            mesh.visual = trimesh.visual.ColorVisuals(
                vertex_colors=[color] * len(mesh.vertices)
            )

            handle = self.server.scene.add_mesh_trimesh(
                name=f"/ghost_{ghost_id}/geom_{i}",
                mesh=mesh,
                wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
                position=np.zeros(3),
            )
            ghost_meshes.append(_MeshHandle(handle=handle, geom_id=i))

        self._ghost_meshes[ghost_id] = ghost_meshes
        self._ghost_data[ghost_id] = ghost_data

    def _update_ghost_meshes(self, ghost_id: str, rgba: Optional[np.ndarray] = None):
        if ghost_id not in self._ghost_meshes:
            return

        ghost_data = self._ghost_data[ghost_id]
        ghost_meshes = self._ghost_meshes[ghost_id]

        for mh in ghost_meshes:
            geom_pos = np.array(ghost_data.geom_xpos[mh.geom_id], dtype=float)
            geom_mat = np.array(ghost_data.geom_xmat[mh.geom_id], dtype=float).reshape(
                3, 3
            )
            quat_xyzw = R.from_matrix(geom_mat).as_quat()
            quat_wxyz = np.array(
                [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=float
            )
            mh.handle.wxyz = quat_wxyz
            mh.handle.position = geom_pos

    def remove_ghost_robot(self, ghost_id: str):
        if ghost_id not in self._ghost_meshes:
            return

        for mh in self._ghost_meshes[ghost_id]:
            try:
                mh.handle.remove()
            except Exception:
                pass

        del self._ghost_meshes[ghost_id]
        del self._ghost_data[ghost_id]

    def sync(
        self,
        *,
        camera_target: np.ndarray | None = None,
        camera_distance: float | None = None,
        update_scene: bool = True,
    ):
        if update_scene:
            self._update_robot_meshes()
            if self._show_joint_axes:
                self._update_joint_frames()
            for ghost_id in self._ghost_meshes:
                self._update_ghost_meshes(ghost_id)

        follow_enabled = bool(self._camera_follow_checkbox.value)
        self._camera_controller.set_enabled(follow_enabled)
        if follow_enabled:
            target = (
                np.asarray(self.data.qpos[:3], dtype=np.float64)
                if camera_target is None
                else np.asarray(camera_target, dtype=np.float64)
            )
            self.cam.lookat = target.copy()
            self._camera_controller.update_target(
                target,
                distance=camera_distance,
            )

    def set_camera_follow(self, enabled: bool) -> None:

        self._camera_follow_checkbox.value = bool(enabled)
        self._camera_controller.set_enabled(enabled)

    def set_camera_target(
        self,
        target: np.ndarray,
        *,
        distance: float | None = None,
    ) -> None:

        target_array = np.asarray(target, dtype=np.float64)
        self.cam.lookat = target_array.copy()
        self._camera_controller.update_target(target_array, distance=distance)

    def set_initial_camera_view(
        self,
        target: np.ndarray,
        *,
        distance: float,
        view_direction: np.ndarray,
    ) -> None:

        self._camera_controller.set_initial_view(
            target,
            distance=distance,
            view_direction=view_direction,
        )

    def is_running(self) -> bool:
        return self._running

    def close(self):
        if not self._running:
            return

        self._running = False
        self._camera_controller.set_enabled(False)


        for mh in self._robot_meshes:
            try:
                mh.handle.remove()
            except Exception:
                pass
        self._robot_meshes.clear()


        for ghost_id in list(self._ghost_meshes.keys()):
            self.remove_ghost_robot(ghost_id)


        for handle in self._joint_frames.values():
            try:
                handle.remove()
            except Exception:
                pass
        self._joint_frames.clear()

        stop_server = getattr(self.server, "stop", None)
        if callable(stop_server):
            stop_server()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class _CameraControl:

    def __init__(self):

        self.lookat = np.array([0.0, 0.0, 0.0])
        self.elevation = -20.0
        self.azimuth = 90.0
        self.distance = 3.0
