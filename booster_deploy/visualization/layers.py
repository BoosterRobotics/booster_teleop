
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from scipy.spatial.transform import Rotation as R

from booster_deploy.types import HumanFrame
from booster_deploy.visualization.drawers import MujocoSceneDrawer, Rgba


DEFAULT_HUMAN_POINT_SIZE = 0.018
DEFAULT_HAND_POINT_SIZE = 0.008
DEFAULT_HUMAN_LINE_WIDTH = 0.004
DEFAULT_HAND_LINE_WIDTH = 0.0025
DEFAULT_HAND_TRANSITION_LINE_WIDTH = 0.008
DEFAULT_HAND_WRIST_LINE_WIDTH = 0.012


class VisualizationLayer(Protocol):

    def draw(self, drawer: MujocoSceneDrawer) -> None:

        ...


@dataclass(frozen=True)
class HumanSkeletonLayer:

    frame: HumanFrame
    connections: Sequence[tuple[str, str]] | None = None
    point_rgba: Rgba = (0.0, 0.85, 1.0, 0.85)
    line_rgba: Rgba = (0.0, 0.75, 1.0, 0.45)
    point_size: float = DEFAULT_HUMAN_POINT_SIZE
    hand_point_size: float | None = DEFAULT_HAND_POINT_SIZE
    line_width: float = DEFAULT_HUMAN_LINE_WIDTH
    hand_line_width: float | None = DEFAULT_HAND_LINE_WIDTH
    hand_transition_line_width: float | None = DEFAULT_HAND_TRANSITION_LINE_WIDTH
    hand_wrist_line_width: float | None = DEFAULT_HAND_WRIST_LINE_WIDTH
    bone_style: str = "line"

    def draw(self, drawer: MujocoSceneDrawer) -> None:

        drawer.draw_human_skeleton(
            self.frame,
            connections=self.connections,
            point_rgba=self.point_rgba,
            line_rgba=self.line_rgba,
            point_size=self.point_size,
            hand_point_size=self.hand_point_size,
            line_width=self.line_width,
            hand_line_width=self.hand_line_width,
            hand_transition_line_width=self.hand_transition_line_width,
            hand_wrist_line_width=self.hand_wrist_line_width,
            bone_style=self.bone_style,
        )


@dataclass(frozen=True)
class HumanFrameAxesLayer:

    frame: HumanFrame
    axis_size: float = 0.08
    line_width: float = 0.006

    def draw(self, drawer: MujocoSceneDrawer) -> None:

        drawer.draw_human_frame_axes(self.frame, axis_size=self.axis_size, width=self.line_width)


@dataclass(frozen=True)
class CommandVelocityArrowLayer:

    root_pos: np.ndarray
    root_quat: np.ndarray
    vx: float
    vy: float
    z_offset: float = 0.55
    length_scale: float = 0.55
    origin_rgba: Rgba = (1.0, 0.95, 0.15, 0.9)
    arrow_rgba: Rgba = (1.0, 0.92, 0.05, 0.95)
    origin_size: float = 0.025
    arrow_width: float = 0.035

    def draw(self, drawer: MujocoSceneDrawer) -> None:

        speed = float(np.hypot(self.vx, self.vy))
        start = np.asarray(self.root_pos, dtype=np.float64).copy()
        start[2] += float(self.z_offset)
        drawer.draw_sphere(start, size=float(self.origin_size), rgba=self.origin_rgba)
        if speed <= 1e-3:
            return

        body_dir = np.array([float(self.vx), float(self.vy), 0.0], dtype=np.float64) / speed
        yaw = float(R.from_quat(np.asarray(self.root_quat, dtype=np.float64), scalar_first=True).as_euler("xyz")[2])
        direction = _rotate_xy3(body_dir, yaw)
        drawer.draw_arrow(
            start,
            start + direction * speed * float(self.length_scale),
            rgba=self.arrow_rgba,
            width=float(self.arrow_width),
        )


def _rotate_xy3(value: np.ndarray, yaw: float) -> np.ndarray:

    out = np.asarray(value, dtype=np.float64).copy()
    c = float(np.cos(yaw))
    s = float(np.sin(yaw))
    x, y = out[:2]
    out[:2] = [c * x - s * y, s * x + c * y]
    return out
