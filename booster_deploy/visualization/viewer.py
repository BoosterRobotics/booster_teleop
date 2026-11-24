
from __future__ import annotations
import logging
from contextlib import contextmanager
from time import perf_counter, sleep
from typing import ContextManager, Iterator, Literal

import mujoco
import mujoco.viewer

ViewerBackend = Literal["mujoco", "viser"]
logger = logging.getLogger("booster_deploy")


@contextmanager
def _managed_mujoco_passive_viewer(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    show_left_ui: bool,
    show_right_ui: bool,
    key_callback,
) -> Iterator:

    viewer = mujoco.viewer.launch_passive(
        model,
        data,
        show_left_ui=show_left_ui,
        show_right_ui=show_right_ui,
        key_callback=key_callback,
    )
    sim_ref = getattr(viewer, "_sim", None)
    try:
        yield viewer
    finally:
        viewer.close()
        if callable(sim_ref):
            deadline_s = perf_counter() + 5.0
            while sim_ref() is not None and perf_counter() < deadline_s:
                sleep(0.001)
            if sim_ref() is not None:
                logger.warning("MuJoCo viewer thread did not stop within 5.0s")


def create_viewer(
    backend: ViewerBackend,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    **kwargs,
) -> ContextManager:
    if backend == "mujoco":
        show_left_ui = kwargs.get("show_left_ui", True)
        show_right_ui = kwargs.get("show_right_ui", True)
        key_callback = kwargs.get("key_callback", None)
        return _managed_mujoco_passive_viewer(
            model,
            data,
            show_left_ui=show_left_ui,
            show_right_ui=show_right_ui,
            key_callback=key_callback,
        )
    elif backend == "viser":
        try:
            from .viser import ViserViewerAdapter
        except ImportError as e:
            raise ImportError(
                f"Viser backend requires 'viser' and 'trimesh' packages. "
                f"Install with: pip install viser trimesh scipy\n"
                f"Original error: {e}"
            )
        

        host = kwargs.get("host", "0.0.0.0")
        port = kwargs.get("port", 8080)
        label = kwargs.get("label", "Booster Deploy - Viser Viewer")
        show_joint_axes = kwargs.get("show_joint_axes", False)
        
        return ViserViewerAdapter(
            model=model,
            data=data,
            host=host,
            port=port,
            label=label,
            show_joint_axes=show_joint_axes,
            camera_azimuth=kwargs.get("camera_azimuth"),
            camera_elevation=kwargs.get("camera_elevation"),
            camera_distance=kwargs.get("camera_distance"),
            camera_follow=bool(kwargs.get("camera_follow", False)),
        )
    else:
        raise ValueError(
            f"Unknown viewer backend: {backend}. "
            f"Supported backends: 'mujoco', 'viser'"
        )


def is_viewer_backend_available(backend: ViewerBackend) -> bool:
    if backend == "mujoco":

        return True
    elif backend == "viser":
        try:
            import viser
            import trimesh
            from scipy.spatial.transform import Rotation
            return True
        except ImportError:
            return False
    else:
        return False
