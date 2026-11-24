
from __future__ import annotations

from collections.abc import Sequence

import mujoco
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as R

from booster_deploy.types import HumanFrame


Rgba = tuple[float, float, float, float]


def is_hand_joint_name(name: str) -> bool:

    normalized = str(name).lower()
    return any(
        token in normalized
        for token in ("hand", "wrist", "thumb", "index", "middle", "ring", "pinky")
    )


def resolve_joint_point_size(name: str, point_size: float, hand_point_size: float | None) -> float:

    if hand_point_size is not None and is_hand_joint_name(name):
        return float(hand_point_size)
    return float(point_size)


def is_hand_bone(parent_name: str, child_name: str) -> bool:

    return is_hand_joint_name(parent_name) and is_hand_joint_name(child_name)


def is_hand_transition_bone(parent_name: str, child_name: str) -> bool:

    parent_hand = is_hand_joint_name(parent_name)
    child_hand = is_hand_joint_name(child_name)
    if parent_hand != child_hand:
        return True
    if not parent_hand:
        return False
    parent = str(parent_name).lower()
    child = str(child_name).lower()
    return parent.endswith("hand") or child.endswith("hand") or "inhand" in parent or "inhand" in child


def is_hand_wrist_bone(parent_name: str, child_name: str) -> bool:

    return not is_hand_joint_name(parent_name) and is_hand_joint_name(child_name)


def resolve_bone_line_width(
    parent_name: str,
    child_name: str,
    body_line_width: float,
    hand_line_width: float | None,
    hand_transition_line_width: float | None,
    hand_wrist_line_width: float | None = None,
) -> float:

    if hand_line_width is None:
        return float(body_line_width)
    if hand_wrist_line_width is not None and is_hand_wrist_bone(parent_name, child_name):
        return float(hand_wrist_line_width)
    if hand_transition_line_width is not None and is_hand_transition_bone(parent_name, child_name):
        return float(hand_transition_line_width)
    if is_hand_bone(parent_name, child_name):
        return float(hand_line_width)
    return float(body_line_width)


def resolve_joint_line_radius(
    name: str,
    body_line_width: float,
    hand_line_width: float | None,
    hand_transition_line_width: float | None,
    hand_wrist_line_width: float | None,
) -> float:

    if hand_line_width is None or not is_hand_joint_name(name):
        return float(body_line_width)
    normalized = str(name).lower()
    if normalized.endswith("hand"):
        return float(hand_wrist_line_width or body_line_width)
    if "inhand" in normalized:
        return float(hand_transition_line_width or hand_line_width)
    return float(hand_line_width)


class MujocoSceneDrawer:

    def __init__(self, scene: mujoco.MjvScene) -> None:

        self._scene = scene

    @property
    def ngeom(self) -> int:

        return int(self._scene.ngeom)

    def clear(self) -> None:

        self._scene.ngeom = 0

    def draw_sphere(self, pos: NDArray[np.float64], *, size: float, rgba: Rgba) -> bool:

        geom = self._next_geom()
        if geom is None:
            return False
        mujoco.mjv_initGeom(
            geom,
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[size, size, size],
            pos=np.asarray(pos, dtype=np.float64),
            mat=np.eye(3).reshape(-1),
            rgba=rgba,
        )
        self._scene.ngeom += 1
        return True

    def draw_line(
        self,
        start: NDArray[np.float64],
        end: NDArray[np.float64],
        *,
        rgba: Rgba,
        width: float,
    ) -> bool:

        geom = self._next_geom()
        if geom is None:
            return False
        start_np = np.asarray(start, dtype=np.float64)
        end_np = np.asarray(end, dtype=np.float64)
        mujoco.mjv_initGeom(
            geom,
            type=mujoco.mjtGeom.mjGEOM_LINE,
            size=[width, width, width],
            pos=start_np,
            mat=np.eye(3).reshape(-1),
            rgba=rgba,
        )
        mujoco.mjv_connector(
            geom,
            type=mujoco.mjtGeom.mjGEOM_LINE,
            width=width,
            from_=start_np,
            to=end_np,
        )
        self._scene.ngeom += 1
        return True

    def draw_capsule(
        self,
        start: NDArray[np.float64],
        end: NDArray[np.float64],
        *,
        rgba: Rgba,
        radius: float,
    ) -> bool:

        geom = self._next_geom()
        if geom is None:
            return False
        start_np = np.asarray(start, dtype=np.float64)
        end_np = np.asarray(end, dtype=np.float64)
        mujoco.mjv_initGeom(
            geom,
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            size=[radius, 0.0, 0.0],
            pos=np.zeros(3, dtype=np.float64),
            mat=np.eye(3).reshape(-1),
            rgba=rgba,
        )
        mujoco.mjv_connector(
            geom,
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            width=radius,
            from_=start_np,
            to=end_np,
        )
        self._scene.ngeom += 1
        return True

    def draw_tapered_capsule(
        self,
        start: NDArray[np.float64],
        end: NDArray[np.float64],
        *,
        rgba: Rgba,
        start_radius: float,
        end_radius: float,
        segments: int = 4,
    ) -> bool:

        start_np = np.asarray(start, dtype=np.float64)
        end_np = np.asarray(end, dtype=np.float64)
        count = max(int(segments), 1)
        direction = end_np - start_np
        ok = True
        for index in range(count):
            t0 = index / count
            t1 = (index + 1) / count
            radius = float(start_radius) + (float(end_radius) - float(start_radius)) * (0.5 * (t0 + t1))
            ok = self.draw_capsule(
                start_np + direction * t0,
                start_np + direction * t1,
                rgba=rgba,
                radius=radius,
            ) and ok
        return ok

    def draw_arrow(
        self,
        start: NDArray[np.float64],
        end: NDArray[np.float64],
        *,
        rgba: Rgba,
        width: float,
    ) -> bool:

        geom = self._next_geom()
        if geom is None:
            return False
        start_np = np.asarray(start, dtype=np.float64)
        end_np = np.asarray(end, dtype=np.float64)
        mujoco.mjv_initGeom(
            geom,
            type=mujoco.mjtGeom.mjGEOM_ARROW,
            size=[width, width, width],
            pos=start_np,
            mat=np.eye(3).reshape(-1),
            rgba=rgba,
        )
        mujoco.mjv_connector(
            geom,
            type=mujoco.mjtGeom.mjGEOM_ARROW,
            width=width,
            from_=start_np,
            to=end_np,
        )
        self._scene.ngeom += 1
        return True

    def draw_frame_axes(
        self,
        pos: NDArray[np.float64],
        quat_wxyz: NDArray[np.float64],
        *,
        axis_size: float,
        width: float,
    ) -> None:

        rot = R.from_quat(np.asarray(quat_wxyz, dtype=np.float64), scalar_first=True).as_matrix()
        origin = np.asarray(pos, dtype=np.float64)
        axes: tuple[tuple[NDArray[np.float64], Rgba], ...] = (
            (rot[:, 0], (1.0, 0.0, 0.0, 0.9)),
            (rot[:, 1], (0.0, 1.0, 0.0, 0.9)),
            (rot[:, 2], (0.0, 0.0, 1.0, 0.9)),
        )
        for axis, rgba in axes:
            self.draw_line(origin, origin + axis * axis_size, rgba=rgba, width=width)

    def draw_human_skeleton(
        self,
        frame: HumanFrame,
        *,
        connections: Sequence[tuple[str, str]] | None,
        point_rgba: Rgba,
        line_rgba: Rgba,
        point_size: float,
        line_width: float,
        hand_point_size: float | None = None,
        hand_line_width: float | None = None,
        hand_transition_line_width: float | None = None,
        hand_wrist_line_width: float | None = None,
        bone_style: str = "line",
    ) -> None:

        if connections is not None:
            for parent_name, child_name in connections:
                if parent_name not in frame or child_name not in frame:
                    continue
                parent = np.asarray(frame[parent_name][0], dtype=np.float64)
                child = np.asarray(frame[child_name][0], dtype=np.float64)
                edge_width = resolve_bone_line_width(
                    parent_name,
                    child_name,
                    line_width,
                    hand_line_width,
                    hand_transition_line_width,
                    hand_wrist_line_width,
                )
                if bone_style == "capsule":
                    self.draw_tapered_capsule(
                        parent,
                        child,
                        rgba=line_rgba,
                        start_radius=resolve_joint_line_radius(
                            parent_name,
                            line_width,
                            hand_line_width,
                            hand_transition_line_width,
                            hand_wrist_line_width,
                        ),
                        end_radius=resolve_joint_line_radius(
                            child_name,
                            line_width,
                            hand_line_width,
                            hand_transition_line_width,
                            hand_wrist_line_width,
                        ),
                    )
                else:
                    self.draw_line(parent, child, rgba=line_rgba, width=edge_width)

        for joint_name, (pos, _quat) in frame.items():
            self.draw_sphere(
                np.asarray(pos, dtype=np.float64),
                size=resolve_joint_point_size(joint_name, point_size, hand_point_size),
                rgba=point_rgba,
            )

    def draw_human_frame_axes(self, frame: HumanFrame, *, axis_size: float, width: float) -> None:

        for pos, quat in frame.values():
            self.draw_frame_axes(
                np.asarray(pos, dtype=np.float64),
                np.asarray(quat, dtype=np.float64),
                axis_size=axis_size,
                width=width,
            )

    def _next_geom(self) -> mujoco.MjvGeom | None:

        if self._scene.ngeom >= self._scene.maxgeom:
            return None
        return self._scene.geoms[self._scene.ngeom]
