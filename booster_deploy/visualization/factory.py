
from __future__ import annotations

import argparse
from typing import Any

from .human_frame_viewer import HumanFrameMujocoViewer
from .retargeting_viewer import RetargetingMujocoViewer
from .viser_viewer import HumanFrameViserViewer, RetargetingViserViewer


def _add_once(parser: argparse.ArgumentParser, *flags: str, **kwargs: Any) -> None:
    """Register an option unless the caller's script already declares it.

    The viewer scripts define some of these themselves (e.g. the robot viewer has
    its own ``--camera-azimuth``), and ``argparse`` raises
    "conflicting option string" when the same flag is added twice — which killed
    the :8081 pane at startup.
    """
    if any(flag in parser._option_string_actions for flag in flags):
        return
    parser.add_argument(*flags, **kwargs)


def add_viewer_backend_args(parser: argparse.ArgumentParser) -> None:
    _add_once(
        parser,
        "--viewer-backend",
        choices=("mujoco", "viser"),
        default="mujoco",
        help="Visualization backend.",
    )
    _add_once(parser, "--viser-host", default="0.0.0.0", help="Viser bind host.")
    _add_once(parser, "--viser-port", type=int, default=8080, help="Viser HTTP/WebSocket port.")
    _add_once(parser, "--viser-show-joint-axes", action="store_true", help="Show robot joint axes in Viser.")
    # Use the same robot-front convention in every operator view.
    _add_once(parser, "--camera-azimuth", type=float, default=180.0,
              help="Initial camera azimuth in degrees (180 = robot front).")
    _add_once(parser, "--camera-elevation", type=float, default=-10.0,
              help="Initial camera elevation in degrees.")


def create_human_frame_viewer(backend: str, **kwargs: Any):
    if backend == "mujoco":
        for key in ("host", "port", "camera_azimuth", "camera_elevation"):
            kwargs.pop(key, None)
        return HumanFrameMujocoViewer(**kwargs)
    if backend == "viser":
        return HumanFrameViserViewer(**kwargs)
    raise ValueError(f"Unknown viewer backend: {backend!r}")


def create_retargeting_viewer(backend: str, xml_path: str, **kwargs: Any):
    if backend == "mujoco":
        for key in ("host", "port", "show_joint_axes", "camera_azimuth", "camera_elevation",
                    "camera_distance", "camera_follow"):
            kwargs.pop(key, None)
        return RetargetingMujocoViewer(xml_path, **kwargs)
    if backend == "viser":
        return RetargetingViserViewer(xml_path, **kwargs)
    raise ValueError(f"Unknown viewer backend: {backend!r}")


__all__ = ["add_viewer_backend_args", "create_human_frame_viewer", "create_retargeting_viewer"]
