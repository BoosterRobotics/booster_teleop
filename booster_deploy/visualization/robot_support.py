
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import mujoco
import numpy as np
from numpy.typing import NDArray

from booster_deploy.visualization.drawers import MujocoSceneDrawer, Rgba


@dataclass(frozen=True)
class RobotSupportState:

    com: NDArray[np.float64]
    com_projection: NDArray[np.float64]
    contact_points: NDArray[np.float64]
    support_points: NDArray[np.float64]
    support_hull: NDArray[np.float64]


@dataclass(frozen=True)
class RobotSupportBaseLayer:

    state: RobotSupportState
    projection_rgba: Rgba = (1.0, 0.12, 0.08, 0.95)
    projection_size: float = 0.025
    vertical_rgba: Rgba = (1.0, 0.12, 0.08, 0.45)
    vertical_width: float = 0.006
    contact_rgba: Rgba = (0.05, 1.0, 0.25, 0.95)
    contact_size: float = 0.014
    hull_rgba: Rgba = (0.05, 1.0, 0.25, 0.85)
    hull_width: float = 0.01

    def draw(self, drawer: MujocoSceneDrawer) -> None:

        drawer.draw_line(
            self.state.com,
            self.state.com_projection,
            rgba=self.vertical_rgba,
            width=float(self.vertical_width),
        )
        drawer.draw_sphere(
            self.state.com_projection,
            size=float(self.projection_size),
            rgba=self.projection_rgba,
        )
        for point in self.state.support_points:
            drawer.draw_sphere(point, size=float(self.contact_size), rgba=self.contact_rgba)

        hull = self.state.support_hull
        if hull.shape[0] == 2:
            drawer.draw_line(hull[0], hull[1], rgba=self.hull_rgba, width=float(self.hull_width))
        elif hull.shape[0] >= 3:
            for idx in range(hull.shape[0]):
                drawer.draw_line(
                    hull[idx],
                    hull[(idx + 1) % hull.shape[0]],
                    rgba=self.hull_rgba,
                    width=float(self.hull_width),
                )


def compute_robot_support_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    root_body_name: str | None = None,
    ground_geom_names: Sequence[str] | None = None,
    unique_xy_tol: float = 1e-5,
    fallback_height_threshold: float = 0.06,
    fallback_height_band: float = 0.03,
    min_support_points: int = 3,
    fallback_polygon_radius: float = 0.06,
) -> RobotSupportState:

    ground_geom_ids = _resolve_ground_geom_ids(model, ground_geom_names)
    ground_z = _ground_projection_z(data, ground_geom_ids)
    com = _robot_com(model, data, root_body_name=root_body_name)
    com_projection = np.array([com[0], com[1], ground_z], dtype=np.float64)
    contact_points = _ground_contact_points(model, data, ground_geom_ids, unique_xy_tol=unique_xy_tol)
    required_points = max(int(min_support_points), 3)
    support_points = contact_points
    support_hull = _convex_hull_xy(support_points)
    if support_hull.shape[0] < required_points:
        fallback_points = _near_ground_fallback_points(
            model,
            data,
            ground_geom_ids,
            ground_z=ground_z,
            height_threshold=float(fallback_height_threshold),
            height_band=float(fallback_height_band),
            min_points=required_points,
            unique_xy_tol=float(unique_xy_tol),
        )
        support_points = _merge_unique_xy((contact_points, fallback_points), unique_xy_tol=float(unique_xy_tol))
        support_points = _ensure_min_support_points(
            support_points,
            center=com_projection,
            ground_z=ground_z,
            min_points=required_points,
            radius=float(fallback_polygon_radius),
            unique_xy_tol=float(unique_xy_tol),
        )
        support_hull = _convex_hull_xy(support_points)
        if support_hull.shape[0] < required_points:
            regular_hull = _regular_support_polygon(
                _support_center(support_points, com_projection),
                ground_z=ground_z,
                radius=float(fallback_polygon_radius),
                count=required_points,
            )
            support_points = _merge_unique_xy((support_points, regular_hull), unique_xy_tol=float(unique_xy_tol))
            support_hull = _convex_hull_xy(support_points)
            if support_hull.shape[0] < required_points:
                support_hull = regular_hull
    return RobotSupportState(
        com=com,
        com_projection=com_projection,
        contact_points=contact_points,
        support_points=support_points,
        support_hull=support_hull,
    )


def _robot_com(model: mujoco.MjModel, data: mujoco.MjData, *, root_body_name: str | None) -> NDArray[np.float64]:

    if root_body_name:
        root_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root_body_name)
        if root_body_id >= 0:
            return np.asarray(data.subtree_com[root_body_id], dtype=np.float64).copy()

    body_mass = np.asarray(model.body_mass, dtype=np.float64)
    moving_body_ids = np.flatnonzero(body_mass > 0.0)
    if moving_body_ids.size == 0:
        return np.zeros(3, dtype=np.float64)
    masses = body_mass[moving_body_ids]
    weighted_pos = np.asarray(data.xipos[moving_body_ids], dtype=np.float64) * masses[:, None]
    return weighted_pos.sum(axis=0) / float(masses.sum())


def _resolve_ground_geom_ids(model: mujoco.MjModel, ground_geom_names: Sequence[str] | None) -> tuple[int, ...]:

    if ground_geom_names:
        ids: list[int] = []
        for name in ground_geom_names:
            geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, str(name))
            if geom_id >= 0:
                ids.append(int(geom_id))
        return tuple(ids)

    return tuple(
        int(geom_id)
        for geom_id in range(model.ngeom)
        if model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_PLANE
    )


def _ground_projection_z(data: mujoco.MjData, ground_geom_ids: Sequence[int]) -> float:

    if not ground_geom_ids:
        return 0.0
    return float(data.geom_xpos[int(ground_geom_ids[0])][2])


def _ground_contact_points(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ground_geom_ids: Sequence[int],
    *,
    unique_xy_tol: float,
) -> NDArray[np.float64]:

    ground_ids = {int(geom_id) for geom_id in ground_geom_ids}
    if not ground_ids or data.ncon <= 0:
        return np.empty((0, 3), dtype=np.float64)

    points: list[NDArray[np.float64]] = []
    for contact_idx in range(int(data.ncon)):
        contact = data.contact[contact_idx]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        if geom1 in ground_ids:
            ground_geom_id = geom1
            other_geom_id = geom2
        elif geom2 in ground_ids:
            ground_geom_id = geom2
            other_geom_id = geom1
        else:
            continue
        if other_geom_id < 0 or model.geom_type[other_geom_id] == mujoco.mjtGeom.mjGEOM_PLANE:
            continue

        point = np.asarray(contact.pos, dtype=np.float64).copy()
        point[2] = float(data.geom_xpos[ground_geom_id][2])
        if not _has_near_xy(points, point, tol=float(unique_xy_tol)):
            points.append(point)

    if not points:
        return np.empty((0, 3), dtype=np.float64)
    return np.stack(points, axis=0).astype(np.float64, copy=False)


def _near_ground_fallback_points(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ground_geom_ids: Sequence[int],
    *,
    ground_z: float,
    height_threshold: float,
    height_band: float,
    min_points: int,
    unique_xy_tol: float,
) -> NDArray[np.float64]:

    ground_ids = {int(geom_id) for geom_id in ground_geom_ids}
    candidates: list[tuple[float, NDArray[np.float64]]] = []
    for geom_id in range(model.ngeom):
        if geom_id in ground_ids or model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_PLANE:
            continue
        if int(model.geom_bodyid[geom_id]) == 0:
            continue
        if int(model.geom_contype[geom_id]) == 0 and int(model.geom_conaffinity[geom_id]) == 0:
            continue

        point = np.asarray(data.geom_xpos[geom_id], dtype=np.float64).copy()
        clearance = _geom_lowest_z(model, data, geom_id) - float(ground_z)
        point[2] = float(ground_z)
        candidates.append((clearance, point))

    if not candidates:
        return np.empty((0, 3), dtype=np.float64)

    candidates.sort(key=lambda item: item[0])
    lowest_clearance = candidates[0][0]
    height_limit = max(float(height_threshold), lowest_clearance + max(float(height_band), 0.0))
    selected: list[NDArray[np.float64]] = []
    for clearance, point in candidates:
        if clearance > height_limit and len(selected) >= int(min_points):
            break
        if clearance <= height_limit and not _has_near_xy(selected, point, tol=unique_xy_tol):
            selected.append(point)

    if len(selected) < int(min_points):
        for _, point in candidates:
            if not _has_near_xy(selected, point, tol=unique_xy_tol):
                selected.append(point)
                if len(selected) >= int(min_points):
                    break

    if not selected:
        return np.empty((0, 3), dtype=np.float64)
    return np.stack(selected, axis=0).astype(np.float64, copy=False)


def _geom_lowest_z(model: mujoco.MjModel, data: mujoco.MjData, geom_id: int) -> float:

    geom_type = model.geom_type[geom_id]
    geom_pos = np.asarray(data.geom_xpos[geom_id], dtype=np.float64)
    geom_size = np.asarray(model.geom_size[geom_id], dtype=np.float64)
    geom_mat = np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
    if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
        return float(geom_pos[2] - geom_size[0])
    if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
        return float(geom_pos[2] - np.abs(geom_mat[2, :]) @ geom_size[:3])
    if geom_type in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER):
        axis_z = abs(float(geom_mat[2, 2]))
        radial_z = float(geom_size[0]) * float(np.sqrt(max(0.0, 1.0 - axis_z * axis_z)))
        return float(geom_pos[2] - axis_z * float(geom_size[1]) - radial_z)
    return float(geom_pos[2])


def _merge_unique_xy(point_sets: Sequence[NDArray[np.float64]], *, unique_xy_tol: float) -> NDArray[np.float64]:

    merged: list[NDArray[np.float64]] = []
    for points in point_sets:
        for point in np.asarray(points, dtype=np.float64).reshape(-1, 3):
            point_copy = point.copy()
            if not _has_near_xy(merged, point_copy, tol=unique_xy_tol):
                merged.append(point_copy)
    if not merged:
        return np.empty((0, 3), dtype=np.float64)
    return np.stack(merged, axis=0).astype(np.float64, copy=False)


def _ensure_min_support_points(
    points: NDArray[np.float64],
    *,
    center: NDArray[np.float64],
    ground_z: float,
    min_points: int,
    radius: float,
    unique_xy_tol: float,
) -> NDArray[np.float64]:

    if points.shape[0] >= int(min_points):
        return points
    supplement = _regular_support_polygon(
        _support_center(points, center),
        ground_z=ground_z,
        radius=radius,
        count=int(min_points),
    )
    return _merge_unique_xy((points, supplement), unique_xy_tol=unique_xy_tol)


def _support_center(points: NDArray[np.float64], fallback_center: NDArray[np.float64]) -> NDArray[np.float64]:

    if points.shape[0] == 0:
        return np.asarray(fallback_center, dtype=np.float64).copy()
    return np.mean(np.asarray(points, dtype=np.float64), axis=0)


def _regular_support_polygon(
    center: NDArray[np.float64],
    *,
    ground_z: float,
    radius: float,
    count: int,
) -> NDArray[np.float64]:

    safe_count = max(int(count), 3)
    safe_radius = max(float(radius), 1e-3)
    center_np = np.asarray(center, dtype=np.float64)
    points = np.empty((safe_count, 3), dtype=np.float64)
    for idx in range(safe_count):
        angle = 2.0 * np.pi * float(idx) / float(safe_count)
        points[idx] = [
            float(center_np[0]) + safe_radius * float(np.cos(angle)),
            float(center_np[1]) + safe_radius * float(np.sin(angle)),
            float(ground_z),
        ]
    return points


def _has_near_xy(points: Sequence[NDArray[np.float64]], point: NDArray[np.float64], *, tol: float) -> bool:

    tol_sq = tol * tol
    for existing in points:
        delta = existing[:2] - point[:2]
        if float(delta @ delta) <= tol_sq:
            return True
    return False


def _convex_hull_xy(points: NDArray[np.float64]) -> NDArray[np.float64]:

    if points.shape[0] <= 1:
        return points.copy()

    sorted_points = sorted((np.asarray(point, dtype=np.float64).copy() for point in points), key=lambda p: (p[0], p[1]))
    unique_points: list[NDArray[np.float64]] = []
    for point in sorted_points:
        if not unique_points or point[0] != unique_points[-1][0] or point[1] != unique_points[-1][1]:
            unique_points.append(point)
    if len(unique_points) <= 1:
        return np.asarray(unique_points, dtype=np.float64).reshape(-1, 3)

    lower: list[NDArray[np.float64]] = []
    for point in unique_points:
        while len(lower) >= 2 and _cross_xy(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)

    upper: list[NDArray[np.float64]] = []
    for point in reversed(unique_points):
        while len(upper) >= 2 and _cross_xy(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)

    hull = lower[:-1] + upper[:-1]
    return np.asarray(hull, dtype=np.float64).reshape(-1, 3)


def _cross_xy(origin: NDArray[np.float64], a: NDArray[np.float64], b: NDArray[np.float64]) -> float:

    return float((a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (b[0] - origin[0]))
