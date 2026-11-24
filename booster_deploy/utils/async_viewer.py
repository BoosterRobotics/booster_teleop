
from __future__ import annotations

import copy
import multiprocessing
import queue
import threading
import traceback
from contextlib import nullcontext
from dataclasses import dataclass
from time import perf_counter, sleep
from typing import Callable, ContextManager

import mujoco
import numpy as np

from booster_deploy.types import HumanFrame, clone_human_frame
from booster_deploy.visualization.drawers import MujocoSceneDrawer
from booster_deploy.visualization.layers import CommandVelocityArrowLayer, HumanSkeletonLayer


@dataclass(frozen=True, slots=True)
class ViewerSnapshot:

    qpos: np.ndarray
    reference_qpos: np.ndarray | None = None
    human_frame: HumanFrame | None = None
    skeleton_connections: tuple[tuple[str, str], ...] | None = None
    velocity_command_xy: tuple[float, float] | None = None

    @classmethod
    def create(
        cls,
        *,
        qpos: np.ndarray,
        reference_qpos: np.ndarray | None = None,
        human_frame: HumanFrame | None = None,
        skeleton_connections: tuple[tuple[str, str], ...] | None = None,
        velocity_command_xy: tuple[float, float] | None = None,
    ) -> "ViewerSnapshot":
        return cls(
            qpos=np.asarray(qpos, dtype=np.float64).reshape(-1).copy(),
            reference_qpos=(
                None
                if reference_qpos is None
                else np.asarray(reference_qpos, dtype=np.float64).reshape(-1).copy()
            ),
            human_frame=None if human_frame is None else clone_human_frame(human_frame),
            skeleton_connections=skeleton_connections,
            velocity_command_xy=velocity_command_xy,
        )


ViewerFactory = Callable[
    [mujoco.MjModel, mujoco.MjData, Callable[[int], None]],
    ContextManager,
]


class AsyncViewerRuntime:

    def __init__(
        self,
        *,
        model: mujoco.MjModel,
        viewer_factory: ViewerFactory,
        render_hz: float,
        visualize_reference_ghost: bool,
        visualize_realtime_human: bool,
        visualize_command_velocity_arrow: bool,
        ghost_rgba: np.ndarray,
        command_velocity_arrow_z: float,
        command_velocity_arrow_scale: float,
    ) -> None:
        render_hz = float(render_hz)
        if render_hz <= 0.0:
            raise ValueError(f"render_hz must be positive, got {render_hz}")
        self._model = copy.copy(model)
        self._viewer_factory = viewer_factory
        self._render_period_s = 1.0 / render_hz
        self._visualize_reference_ghost = bool(visualize_reference_ghost)
        self._visualize_realtime_human = bool(visualize_realtime_human)
        self._visualize_command_velocity_arrow = bool(visualize_command_velocity_arrow)
        self._ghost_rgba = np.asarray(ghost_rgba, dtype=np.float32).reshape(4).copy()
        self._command_velocity_arrow_z = float(command_velocity_arrow_z)
        self._command_velocity_arrow_scale = float(command_velocity_arrow_scale)

        self._snapshot_lock = threading.Lock()
        self._snapshot: ViewerSnapshot | None = None
        self._snapshot_version = 0
        self._force_lock = threading.Lock()
        self._viewer_xfrc_applied = np.zeros((int(model.nbody), 6), dtype=np.float64)
        self._key_events: queue.SimpleQueue[int] = queue.SimpleQueue()
        self._stop_event = threading.Event()
        self._running_event = threading.Event()

    def run(self, *, on_started: Callable[[], None] | None = None) -> None:

        if self._running_event.is_set():
            raise RuntimeError("Async viewer runtime is already running")
        if self._stop_event.is_set():
            raise RuntimeError("Async viewer runtime cannot be restarted")

        render_data = mujoco.MjData(self._model)
        ghost_data = mujoco.MjData(self._model)
        ghost_option = mujoco.MjvOption()
        mujoco.mj_forward(self._model, render_data)
        mujoco.mj_forward(self._model, ghost_data)
        try:
            with self._viewer_factory(self._model, render_data, self._enqueue_key) as viewer:
                viewer.cam.elevation = -20
                set_camera_follow = getattr(viewer, "set_camera_follow", None)
                if callable(set_camera_follow):
                    set_camera_follow(True)
                self._running_event.set()
                if on_started is not None:
                    on_started()
                try:
                    self._render_loop(viewer, render_data, ghost_data, ghost_option)
                finally:


                    self._stop_event.set()
        finally:
            self._stop_event.set()
            self._running_event.clear()

    def publish(self, snapshot: ViewerSnapshot) -> None:

        with self._snapshot_lock:
            self._snapshot = snapshot
            self._snapshot_version += 1

    def is_running(self) -> bool:
        return self._running_event.is_set() and not self._stop_event.is_set()

    def drain_key_events(self, handler: Callable[[int], None]) -> None:
        while True:
            try:
                keycode = self._key_events.get_nowait()
            except queue.Empty:
                return
            handler(keycode)

    def latest_xfrc_applied(self) -> np.ndarray:

        with self._force_lock:
            return self._viewer_xfrc_applied.copy()

    def close(self) -> None:

        self._stop_event.set()

    def _enqueue_key(self, keycode: int) -> None:
        self._key_events.put(int(keycode))

    def _latest_snapshot(self) -> tuple[ViewerSnapshot | None, int]:
        with self._snapshot_lock:
            return self._snapshot, self._snapshot_version

    def _render_loop(
        self,
        viewer,
        render_data: mujoco.MjData,
        ghost_data: mujoco.MjData,
        ghost_option: mujoco.MjvOption,
    ) -> None:
        next_render_s = perf_counter()
        rendered_version = -1
        while not self._stop_event.is_set() and viewer.is_running():
            now_s = perf_counter()
            if now_s < next_render_s:
                self._stop_event.wait(next_render_s - now_s)
                continue
            next_render_s += self._render_period_s
            if now_s > next_render_s:
                next_render_s = now_s + self._render_period_s

            snapshot, version = self._latest_snapshot()
            if snapshot is not None and version != rendered_version:
                rendered_version = version
                viewer_lock = (
                    viewer.lock() if hasattr(viewer, "lock") else nullcontext()
                )
                with viewer_lock:
                    self._apply_snapshot(
                        viewer,
                        render_data,
                        ghost_data,
                        ghost_option,
                        snapshot,
                    )
            viewer.sync()
            with self._force_lock:
                self._viewer_xfrc_applied[:] = render_data.xfrc_applied

    def _apply_snapshot(
        self,
        viewer,
        render_data: mujoco.MjData,
        ghost_data: mujoco.MjData,
        ghost_option: mujoco.MjvOption,
        snapshot: ViewerSnapshot,
    ) -> None:
        if snapshot.qpos.shape != (int(self._model.nq),):
            raise ValueError(
                f"viewer qpos shape {snapshot.qpos.shape} != model nq {int(self._model.nq)}"
            )
        render_data.qpos[:] = snapshot.qpos
        render_data.qvel[:] = 0.0
        mujoco.mj_forward(self._model, render_data)
        set_camera_target = getattr(viewer, "set_camera_target", None)
        if callable(set_camera_target):
            set_camera_target(snapshot.qpos[:3])
        else:
            viewer.cam.lookat[:] = snapshot.qpos[:3]

        has_user_scene = hasattr(viewer, "user_scn")
        if has_user_scene:
            viewer.user_scn.ngeom = 0

        if self._visualize_reference_ghost and snapshot.reference_qpos is not None:
            ghost_data.qpos[:] = snapshot.reference_qpos
            ghost_data.qvel[:] = 0.0
            mujoco.mj_forward(self._model, ghost_data)
            if hasattr(viewer, "add_ghost_robot"):
                viewer.add_ghost_robot("reference", ghost_data, self._ghost_rgba)
            elif has_user_scene:
                mujoco.mjv_updateScene(
                    self._model,
                    ghost_data,
                    ghost_option,
                    None,
                    viewer.cam,
                    int(mujoco.mjtCatBit.mjCAT_DYNAMIC),
                    viewer.user_scn,
                )
                for geom_id in range(viewer.user_scn.ngeom):
                    viewer.user_scn.geoms[geom_id].rgba[:] = self._ghost_rgba
        elif hasattr(viewer, "remove_ghost_robot"):
            viewer.remove_ghost_robot("reference")

        if not has_user_scene:
            return
        drawer = MujocoSceneDrawer(viewer.user_scn)
        if self._visualize_realtime_human and snapshot.human_frame is not None:
            HumanSkeletonLayer(
                snapshot.human_frame,
                connections=snapshot.skeleton_connections,
                point_rgba=(0.0, 0.85, 1.0, 0.85),
                line_rgba=(0.0, 0.75, 1.0, 0.45),
                point_size=0.018,
                line_width=0.004,
                bone_style="line",
            ).draw(drawer)
        if (
            self._visualize_command_velocity_arrow
            and snapshot.velocity_command_xy is not None
        ):
            CommandVelocityArrowLayer(
                root_pos=snapshot.qpos[:3],
                root_quat=snapshot.qpos[3:7],
                vx=snapshot.velocity_command_xy[0],
                vy=snapshot.velocity_command_xy[1],
                z_offset=self._command_velocity_arrow_z,
                length_scale=self._command_velocity_arrow_scale,
            ).draw(drawer)


def _viser_process_entry(
    model: mujoco.MjModel,
    snapshot_queue,
    stop_event,
    ready_event,
    error_queue,
    *,
    render_hz: float,
    host: str,
    port: int,
    show_joint_axes: bool,
    visualize_reference_ghost: bool,
    visualize_realtime_human: bool,
    visualize_command_velocity_arrow: bool,
    ghost_rgba: np.ndarray,
    command_velocity_arrow_z: float,
    command_velocity_arrow_scale: float,
) -> None:

    runtime: AsyncViewerRuntime | None = None
    transfer_thread: threading.Thread | None = None
    transfer_errors: list[BaseException] = []
    try:
        from booster_deploy.visualization.viewer import create_viewer

        def viewer_factory(viewer_model, viewer_data, key_callback):
            return create_viewer(
                "viser",
                viewer_model,
                viewer_data,
                key_callback=key_callback,
                host=host,
                port=port,
                show_joint_axes=show_joint_axes,
            )

        runtime = AsyncViewerRuntime(
            model=model,
            viewer_factory=viewer_factory,
            render_hz=render_hz,
            visualize_reference_ghost=visualize_reference_ghost,
            visualize_realtime_human=visualize_realtime_human,
            visualize_command_velocity_arrow=visualize_command_velocity_arrow,
            ghost_rgba=ghost_rgba,
            command_velocity_arrow_z=command_velocity_arrow_z,
            command_velocity_arrow_scale=command_velocity_arrow_scale,
        )

        def transfer_snapshots() -> None:
            try:
                while not stop_event.is_set():
                    try:
                        snapshot = snapshot_queue.get(timeout=0.05)
                    except queue.Empty:
                        continue
                    while True:
                        try:
                            snapshot = snapshot_queue.get_nowait()
                        except queue.Empty:
                            break
                    runtime.publish(snapshot)
            except BaseException as exc:
                transfer_errors.append(exc)
            finally:
                runtime.close()

        def viewer_started() -> None:
            nonlocal transfer_thread
            transfer_thread = threading.Thread(
                target=transfer_snapshots,
                name="viser-snapshot-receiver",
                daemon=False,
            )
            transfer_thread.start()
            ready_event.set()

        runtime.run(on_started=viewer_started)
        if transfer_thread is not None:
            transfer_thread.join()
        if transfer_errors:
            raise transfer_errors[0]
    except BaseException:
        try:
            error_queue.put_nowait(traceback.format_exc())
        except queue.Full:
            pass
        ready_event.set()
    finally:
        stop_event.set()
        if runtime is not None:
            runtime.close()
        if transfer_thread is not None and transfer_thread.is_alive():
            transfer_thread.join(timeout=1.0)


class AsyncViserViewerRuntime:

    def __init__(
        self,
        *,
        model: mujoco.MjModel,
        render_hz: float,
        host: str,
        port: int,
        show_joint_axes: bool,
        visualize_reference_ghost: bool,
        visualize_realtime_human: bool,
        visualize_command_velocity_arrow: bool,
        ghost_rgba: np.ndarray,
        command_velocity_arrow_z: float,
        command_velocity_arrow_scale: float,
    ) -> None:
        render_hz = float(render_hz)
        if render_hz <= 0.0:
            raise ValueError(f"render_hz must be positive, got {render_hz}")
        self._model = copy.copy(model)
        self._render_hz = render_hz
        self._host = str(host)
        self._port = int(port)
        self._show_joint_axes = bool(show_joint_axes)
        self._visualize_reference_ghost = bool(visualize_reference_ghost)
        self._visualize_realtime_human = bool(visualize_realtime_human)
        self._visualize_command_velocity_arrow = bool(
            visualize_command_velocity_arrow
        )
        self._ghost_rgba = np.asarray(ghost_rgba, dtype=np.float32).reshape(4).copy()
        self._command_velocity_arrow_z = float(command_velocity_arrow_z)
        self._command_velocity_arrow_scale = float(command_velocity_arrow_scale)

        self._context = multiprocessing.get_context("spawn")
        self._snapshot_queue = self._context.Queue(maxsize=1)
        self._stop_event = self._context.Event()
        self._ready_event = self._context.Event()
        self._error_queue = self._context.Queue(maxsize=1)
        self._local_stop_event = threading.Event()
        self._running_event = threading.Event()
        self._snapshot_lock = threading.Lock()
        self._snapshot: ViewerSnapshot | None = None
        self._snapshot_version = 0
        self._snapshot_event = threading.Event()
        self._sender_thread: threading.Thread | None = None
        self._key_events: queue.SimpleQueue[int] = queue.SimpleQueue()
        self._zero_xfrc_applied = np.zeros((int(model.nbody), 6), dtype=np.float64)
        self._process = None

    def run(self, *, on_started: Callable[[], None] | None = None) -> None:
        if self._process is not None:
            raise RuntimeError("Async Viser runtime cannot be restarted")

        process = self._context.Process(
            target=_viser_process_entry,
            kwargs={
                "model": self._model,
                "snapshot_queue": self._snapshot_queue,
                "stop_event": self._stop_event,
                "ready_event": self._ready_event,
                "error_queue": self._error_queue,
                "render_hz": self._render_hz,
                "host": self._host,
                "port": self._port,
                "show_joint_axes": self._show_joint_axes,
                "visualize_reference_ghost": self._visualize_reference_ghost,
                "visualize_realtime_human": self._visualize_realtime_human,
                "visualize_command_velocity_arrow": self._visualize_command_velocity_arrow,
                "ghost_rgba": self._ghost_rgba,
                "command_velocity_arrow_z": self._command_velocity_arrow_z,
                "command_velocity_arrow_scale": self._command_velocity_arrow_scale,
            },
            name="viser-render-process",
            daemon=True,
        )
        self._process = process
        process.start()
        startup_deadline_s = perf_counter() + 30.0
        while not self._ready_event.wait(timeout=0.05):
            if not process.is_alive():
                break
            if perf_counter() >= startup_deadline_s:
                self.close()
                process.join(timeout=2.0)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=1.0)
                raise TimeoutError("Viser render process did not start within 30.0s")

        error = self._pop_error()
        if error is not None:
            process.join(timeout=2.0)
            raise RuntimeError(f"Viser render process failed:\n{error}")
        if not process.is_alive():
            raise RuntimeError(
                f"Viser render process exited during startup (exitcode={process.exitcode})"
            )

        self._running_event.set()
        self._sender_thread = threading.Thread(
            target=self._send_snapshots,
            name="viser-snapshot-sender",
            daemon=False,
        )
        self._sender_thread.start()
        if on_started is not None:
            on_started()
        try:
            while not self._local_stop_event.is_set() and process.is_alive():
                process.join(timeout=0.05)
        finally:
            self.close()
            if self._sender_thread is not None:
                self._sender_thread.join(timeout=1.0)
            process.join(timeout=5.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)
            self._running_event.clear()

        error = self._pop_error()
        if error is not None:
            raise RuntimeError(f"Viser render process failed:\n{error}")
        if process.exitcode not in (0, None):
            raise RuntimeError(
                f"Viser render process exited with code {process.exitcode}"
            )

    def publish(self, snapshot: ViewerSnapshot) -> None:
        with self._snapshot_lock:
            self._snapshot = snapshot
            self._snapshot_version += 1
        self._snapshot_event.set()

    def is_running(self) -> bool:
        process = self._process
        return (
            self._running_event.is_set()
            and not self._local_stop_event.is_set()
            and process is not None
            and process.is_alive()
        )

    def drain_key_events(self, handler: Callable[[int], None]) -> None:
        while True:
            try:
                keycode = self._key_events.get_nowait()
            except queue.Empty:
                return
            handler(keycode)

    def latest_xfrc_applied(self) -> np.ndarray:
        return self._zero_xfrc_applied.copy()

    def close(self) -> None:
        self._local_stop_event.set()
        self._stop_event.set()
        self._snapshot_event.set()

    def _pop_error(self) -> str | None:
        try:
            return str(self._error_queue.get_nowait())
        except queue.Empty:
            return None

    def _send_snapshots(self) -> None:
        sent_version = -1
        while not self._local_stop_event.is_set():
            self._snapshot_event.wait(timeout=0.05)
            self._snapshot_event.clear()
            with self._snapshot_lock:
                snapshot = self._snapshot
                version = self._snapshot_version
            if snapshot is None or version == sent_version:
                continue

            while not self._local_stop_event.is_set():
                try:
                    self._snapshot_queue.put_nowait(snapshot)
                    sent_version = version
                    break
                except queue.Full:
                    try:
                        self._snapshot_queue.get_nowait()
                    except queue.Empty:
                        sleep(0.001)
