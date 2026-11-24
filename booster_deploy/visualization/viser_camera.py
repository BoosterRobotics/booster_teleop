
from __future__ import annotations

import threading
from contextlib import nullcontext
from collections.abc import Sequence

import numpy as np


_FALLBACK_TARGET = np.array([0.0, 0.0, 0.8], dtype=np.float64)
_FALLBACK_VIEW_DIRECTION = np.array([1.0, -1.0, 0.6], dtype=np.float64)


def fit_camera_to_points(
    positions: np.ndarray | Sequence[Sequence[float]],
    *,
    min_distance: float,
    distance_scale: float = 2.2,
    fallback_target: Sequence[float] = _FALLBACK_TARGET,
) -> tuple[np.ndarray, float]:

    minimum = float(min_distance)
    scale = float(distance_scale)
    if not np.isfinite(minimum) or minimum <= 0.0:
        raise ValueError("min_distance must be positive and finite")
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("distance_scale must be positive and finite")

    fallback = np.asarray(fallback_target, dtype=np.float64)
    if fallback.shape != (3,) or not np.isfinite(fallback).all():
        raise ValueError("fallback_target must contain three finite values")

    points = np.asarray(positions, dtype=np.float64)
    if points.size == 0:
        return fallback.copy(), minimum
    points = points.reshape(-1, 3)
    valid = points[np.all(np.isfinite(points), axis=1)]
    if valid.shape[0] == 0:
        return fallback.copy(), minimum

    mins = np.min(valid, axis=0)
    maxs = np.max(valid, axis=0)
    center = 0.5 * (mins + maxs)
    extent = float(np.max(maxs - mins))
    return center, max(minimum, extent * scale)


class ViserCameraController:

    def __init__(
        self,
        server,
        *,
        enabled: bool = False,
        fallback_distance: float = 3.0,
    ) -> None:
        distance = float(fallback_distance)
        if not np.isfinite(distance) or distance <= 0.0:
            raise ValueError("fallback_distance must be positive and finite")
        self._server = server
        self._enabled = bool(enabled)
        self._fallback_distance = distance
        self._target: np.ndarray | None = None
        self._distance: float | None = None
        self._initial_target: np.ndarray | None = None
        self._initial_distance: float | None = None
        self._initial_view_direction: np.ndarray | None = None
        self._lock = threading.RLock()
        on_client_connect = getattr(self._server, "on_client_connect", None)
        if callable(on_client_connect):
            on_client_connect(self._on_client_connect)

    @property
    def enabled(self) -> bool:
        with self._lock:
            return self._enabled

    def set_enabled(self, enabled: bool) -> None:

        with self._lock:
            became_enabled = bool(enabled) and not self._enabled
            self._enabled = bool(enabled)
            target = None if self._target is None else self._target.copy()
            distance = self._distance
        if became_enabled and target is not None:
            self._apply_to_connected_clients(target, distance)

    def update_target(
        self,
        target: np.ndarray | Sequence[float],
        *,
        distance: float | None = None,
    ) -> None:

        target_array = np.asarray(target, dtype=np.float64)
        if target_array.shape != (3,) or not np.isfinite(target_array).all():
            raise ValueError("Viser camera target must contain three finite values")
        if distance is not None:
            distance = float(distance)
            if not np.isfinite(distance) or distance <= 0.0:
                raise ValueError("Viser camera distance must be positive and finite")

        with self._lock:
            self._target = target_array.copy()
            self._distance = distance
            enabled = self._enabled
        if enabled:
            self._apply_to_connected_clients(target_array, distance)

    def set_initial_view(
        self,
        target: np.ndarray | Sequence[float],
        *,
        distance: float,
        view_direction: np.ndarray | Sequence[float],
    ) -> None:

        target_array = np.asarray(target, dtype=np.float64)
        direction_array = np.asarray(view_direction, dtype=np.float64)
        distance = float(distance)
        if target_array.shape != (3,) or not np.isfinite(target_array).all():
            raise ValueError("Initial Viser camera target must contain three finite values")
        if direction_array.shape != (3,) or not np.isfinite(direction_array).all():
            raise ValueError("Initial Viser camera direction must contain three finite values")
        if float(np.linalg.norm(direction_array)) <= 1e-9:
            raise ValueError("Initial Viser camera direction must be non-zero")
        if not np.isfinite(distance) or distance <= 0.0:
            raise ValueError("Initial Viser camera distance must be positive and finite")

        with self._lock:
            self._initial_target = target_array.copy()
            self._initial_distance = distance
            self._initial_view_direction = direction_array.copy()
        for client in tuple(self._server.get_clients().values()):
            self._apply_to_client(
                client,
                target_array,
                distance,
                view_direction=direction_array,
            )

    def _on_client_connect(self, client) -> None:
        with self._lock:
            if self._enabled and self._target is not None:
                target = self._target.copy()
                distance = self._distance
                # Keep following the target, but start at the configured direction:
                # otherwise every new client inherits viser's default diagonal angle
                # and "front view" silently does not apply when follow is on.
                view_direction = (
                    None
                    if self._initial_view_direction is None
                    else self._initial_view_direction.copy()
                )
            elif self._initial_target is not None:
                target = self._initial_target.copy()
                distance = self._initial_distance
                view_direction = self._initial_view_direction.copy()
            else:
                return
        self._apply_to_client(
            client,
            target,
            distance,
            view_direction=view_direction,
        )

    def _apply_to_connected_clients(
        self,
        target: np.ndarray,
        distance: float | None,
    ) -> None:
        for client in tuple(self._server.get_clients().values()):
            self._apply_to_client(client, target, distance)

    def _apply_to_client(
        self,
        client,
        target: np.ndarray,
        distance: float | None,
        *,
        view_direction: np.ndarray | None = None,
    ) -> None:
        camera = client.camera
        position = np.asarray(camera.position, dtype=np.float64)
        look_at = np.asarray(camera.look_at, dtype=np.float64)
        valid_camera = (
            position.shape == (3,)
            and look_at.shape == (3,)
            and np.isfinite(position).all()
            and np.isfinite(look_at).all()
        )
        if view_direction is not None:
            offset = np.asarray(view_direction, dtype=np.float64).copy()
        elif valid_camera:
            offset = position - look_at
        else:
            offset = _FALLBACK_VIEW_DIRECTION.copy()
        offset_norm = float(np.linalg.norm(offset))
        if offset_norm <= 1e-9:
            offset = _FALLBACK_VIEW_DIRECTION.copy()
            offset_norm = float(np.linalg.norm(offset))
        desired_distance = offset_norm if distance is None else distance
        if distance is None and not valid_camera and view_direction is None:
            desired_distance = self._fallback_distance

        desired_position = target + offset * (float(desired_distance) / offset_norm)
        if valid_camera and np.allclose(
            position,
            desired_position,
            rtol=0.0,
            atol=1e-9,
        ) and np.allclose(
            look_at,
            target,
            rtol=0.0,
            atol=1e-9,
        ):
            return

        atomic = client.atomic() if hasattr(client, "atomic") else nullcontext()
        with atomic:


            camera.position = desired_position
            camera.look_at = target


__all__ = ["ViserCameraController", "fit_camera_to_points"]
