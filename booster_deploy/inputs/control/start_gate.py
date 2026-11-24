"""Robot remote-controller buttons as the offline-replay start/stop gate.

Offline replay is driven by the robot's own controller: **A** starts one replay
cycle, **B** aborts the cycle in progress. The deploy side keeps its normal
button mapping (A enters Tracking, B returns to Walking); the replay publisher
only needs to know *when* A and B were pressed, so this module subscribes to
``/remote_controller_state`` in a background thread and exposes rising edges.

The edge logic is kept in :class:`ButtonEdgeDetector` so it can be tested without
ROS or hardware.
"""

from __future__ import annotations

import os
import queue
import threading
from collections.abc import Callable

from booster_deploy.utils.isaaclab.configclass import configclass


BUTTONS = ("a", "b", "x", "y", "lt", "rt")


class ButtonEdgeDetector:
    """Track rising edges of boolean buttons across successive states."""

    def __init__(self, buttons: tuple[str, ...] = BUTTONS) -> None:
        if not buttons:
            raise ValueError("at least one button must be tracked")
        self.buttons = tuple(buttons)
        self._latched = {name: False for name in self.buttons}

    def update(self, pressed: dict[str, bool]) -> tuple[str, ...]:
        """Return the names that went from released to pressed."""

        rising: list[str] = []
        for name in self.buttons:
            is_pressed = bool(pressed.get(name, False))
            if is_pressed and not self._latched[name]:
                rising.append(name)
            self._latched[name] = is_pressed
        return tuple(rising)

    def reset(self) -> None:
        for name in self.buttons:
            self._latched[name] = False


@configclass
class RobotButtonMonitorCfg:
    topic: str = "/remote_controller_state"
    node_name: str = "offline_replay_buttons"
    spin_timeout_s: float = 0.05


class RobotButtonMonitor:
    """Background ROS subscriber for replay edges or a raw-state callback.

    The topic is published by the robot itself, so this only works on a machine
    with ``rclpy`` + ``booster_interface`` (the onboard computer).
    """

    def __init__(
        self,
        cfg: RobotButtonMonitorCfg | None = None,
        *,
        on_state: Callable[[dict[str, bool]], None] | None = None,
    ) -> None:
        self._on_state = on_state
        self.cfg = cfg or RobotButtonMonitorCfg()
        self.cfg.topic = os.getenv("BOOSTER_ROBOT_JOYSTICK_TOPIC", self.cfg.topic)
        self._edges: queue.Queue[str] = queue.Queue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        #: set when the ROS subscriber could not start (rclpy missing, bad topic, ...)
        self.error: str | None = None

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="robot-button-monitor", daemon=True)
        self._thread.start()

    def wait_until_ready(self, timeout_s: float = 2.0) -> None:
        if not self._ready.wait(timeout=timeout_s):
            raise RuntimeError(f"Timed out starting robot button subscriber on {self.cfg.topic}")
        if self.error is not None:
            raise RuntimeError(f"Robot button subscriber unavailable: {self.error}")

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    # ------------------------------------------------------------------ queries
    def wait_for_start(self, *, timeout_s: float | None = None) -> bool:
        """Block until A is pressed; returns False on timeout."""

        import time

        self._ready.wait(timeout=2.0)
        deadline = None if timeout_s is None else time.monotonic() + float(timeout_s)
        while not self._stop.is_set():
            remaining = 0.1 if deadline is None else min(0.1, deadline - time.monotonic())
            if deadline is not None and remaining <= 0.0:
                return False
            try:
                edge = self._edges.get(timeout=max(remaining, 0.01))
            except queue.Empty:
                continue
            if edge == "a":
                self.clear_edges()
                return True
        return False

    def abort_requested(self) -> bool:
        """True when B was pressed since the last check (consumes the edge)."""

        abort = False
        while True:
            try:
                edge = self._edges.get_nowait()
            except queue.Empty:
                return abort
            if edge == "b":
                abort = True

    def clear_edges(self) -> None:
        while True:
            try:
                self._edges.get_nowait()
            except queue.Empty:
                return

    def failed(self) -> bool:
        """True when the subscriber died: callers must not wait forever."""

        return self.error is not None

    def is_stopped(self) -> bool:
        """True once ``stop()`` was called (lets timed waits tell stop from timeout)."""

        return self._stop.is_set()

    # ------------------------------------------------------------------ internals
    def _run(self) -> None:
        try:
            self._run_inner()
        except BaseException as exc:  # noqa: BLE001 - a dead monitor must be visible
            self.error = f"{type(exc).__name__}: {exc}"
            print(f"[start-gate] button subscriber unavailable: {self.error}", flush=True)
            self._ready.set()

    def _run_inner(self) -> None:
        import rclpy
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

        from booster_interface.msg import RemoteControllerState

        initialized_here = not rclpy.ok()
        if initialized_here:
            if self._on_state is None:
                rclpy.init(args=None)
            else:
                from rclpy.signals import SignalHandlerOptions

                rclpy.init(args=None, signal_handler_options=SignalHandlerOptions.NO)
        node = rclpy.create_node(self.cfg.node_name)
        detector = ButtonEdgeDetector()

        def on_state(state: RemoteControllerState) -> None:
            pressed = {
                "a": bool(state.a), "b": bool(state.b),
                "x": bool(state.x), "y": bool(state.y),
                "lt": bool(state.lt), "rt": bool(state.rt),
            }
            if self._on_state is not None:
                self._on_state(pressed)
                return
            for edge in detector.update(pressed):
                self._edges.put(edge)

        node.create_subscription(
            RemoteControllerState,
            self.cfg.topic,
            on_state,
            QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                history=HistoryPolicy.KEEP_LAST,
                durability=DurabilityPolicy.VOLATILE,
            ),
        )
        if self._on_state is None:
            print(f"[replay] listening for A (start) / B (abort) on {self.cfg.topic}", flush=True)
        else:
            print(f"[mocap-record] listening for LT+RT+X / LT+RT+Y on {self.cfg.topic}", flush=True)
        self._ready.set()
        try:
            while not self._stop.is_set() and rclpy.ok():
                rclpy.spin_once(node, timeout_sec=float(self.cfg.spin_timeout_s))
        finally:
            node.destroy_node()
            if initialized_here and rclpy.ok():
                rclpy.shutdown()


__all__ = ["BUTTONS", "ButtonEdgeDetector", "RobotButtonMonitor", "RobotButtonMonitorCfg"]
