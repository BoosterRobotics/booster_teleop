
from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass
import logging
import queue
import threading
import time
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .coordinate import BODY_JOINT_NAMES, body_poses_to_human_frame
from booster_deploy.types.pico_types import PicoBodyRawFrameNs, PicoButtonBits, PicoControlFrameNs, PicoRetargetInputFrameNs

logger = logging.getLogger(__name__)


SNAPSHOT_BUTTON_ALIASES: dict[PicoButtonBits, tuple[str, str]] = {
    PicoButtonBits.X: ("left", "primary_button"),
    PicoButtonBits.Y: ("left", "secondary_button"),
    PicoButtonBits.A: ("right", "primary_button"),
    PicoButtonBits.B: ("right", "secondary_button"),
    PicoButtonBits.LEFT_AXIS_CLICK: ("left", "axis_click"),
    PicoButtonBits.RIGHT_AXIS_CLICK: ("right", "axis_click"),
    PicoButtonBits.LEFT_MENU: ("left", "menu_button"),
    PicoButtonBits.RIGHT_MENU: ("right", "menu_button"),
}


@dataclass(frozen=True)
class PicoRtCaptureConfig:

    human_format: str = "xrobot"
    timeout_s: float = 30.0
    wait_body_available: bool = True
    dedupe_source_ns: bool = True
    max_cached_frames: int = 4
    input_mode: str = "auto"
    poll_hz: float = 50.0
    raw_recording: bool = False


class PicoRtCapture:

    def __init__(self, config: PicoRtCaptureConfig | None = None) -> None:

        try:
            import xrobotoolkit_sdk as xrt
        except ImportError as exc:
            raise ImportError(
                "xrobotoolkit_sdk is required for Pico RT input. Install a callback- or polling-capable "
                "XRoboToolkit SDK in the current Python environment."
            ) from exc

        self.config = config or PicoRtCaptureConfig()
        import uuid
        from booster_deploy.utils.diag_report import reporter
        self.stream_id = uuid.uuid4().hex
        self._diag = reporter("capture")
        self.raw_frames: queue.Queue[PicoBodyRawFrameNs] | None = (
            queue.Queue(maxsize=2048) if self.config.raw_recording else None
        )
        self.raw_snapshots: queue.Queue[dict[str, Any]] | None = (
            queue.Queue(maxsize=2048) if self.config.raw_recording else None
        )
        self.raw_dropped_frames = 0
        self.raw_activity_event = threading.Event()
        requested_mode = str(self.config.input_mode).strip().lower()
        if requested_mode not in {"auto", "callback", "polling"}:
            raise ValueError(
                f"Unsupported Pico input_mode={self.config.input_mode!r}; expected auto, callback, or polling"
            )
        self._callback_available = all(
            callable(getattr(xrt, name, None))
            for name in ("register_frame_callback", "clear_frame_callback")
        )
        self._polling_available = all(
            callable(getattr(xrt, name, None))
            for name in (
                "is_body_data_available",
                "get_body_joints_pose",
                "get_A_button",
                "get_B_button",
                "get_X_button",
                "get_Y_button",
            )
        )
        if requested_mode == "callback" and not self._callback_available:
            raise RuntimeError("Pico input_mode=callback requested, but callback APIs are unavailable")
        if requested_mode == "polling" and not self._polling_available:
            raise RuntimeError("Pico input_mode=polling requested, but polling APIs are unavailable")
        if requested_mode == "auto" and not (self._callback_available or self._polling_available):
            raise RuntimeError(
                "The installed xrobotoolkit_sdk exposes neither callback APIs nor the required polling APIs"
            )

        self._xrt = xrt
        self._requested_input_mode = requested_mode
        self._input_mode = "uninitialized"
        self._closed = False
        self._started = False
        self._seq = 0
        self._pico_controller_seq = 0
        self._last_source_ns: int | None = None
        self._latest_body: PicoBodyRawFrameNs | None = None
        self._latest_retarget_input: PicoRetargetInputFrameNs | None = None
        self._latest_pico_controller: PicoControlFrameNs | None = None
        self._body_event = threading.Event()
        self._input_event = threading.Event()
        self._pico_controller_event = threading.Event()
        self._lock = threading.Lock()
        self._callback_lock = threading.Lock()
        # Consumers serialize conversion, never holding the callback's state lock
        # across coordinate work. A seq is converted at most once.
        self._convert_lock = threading.Lock()
        self._poll_stop_event = threading.Event()
        self._poll_thread: threading.Thread | None = None
        self._cached_bodies: deque[PicoBodyRawFrameNs] = deque(maxlen=max(int(self.config.max_cached_frames), 0))
        self._callback_count = 0
        self._last_body_poses: NDArray[np.float64] | None = None

    @property
    def input_mode(self) -> str:

        return self._input_mode

    @property
    def callback_count(self) -> int:

        with self._lock:
            return self._callback_count

    @property
    def body_seq(self) -> int:

        with self._lock:
            return self._seq

    @property
    def pico_controller_seq(self) -> int:

        with self._lock:
            return self._pico_controller_seq

    @property
    def sdk_body_available(self) -> bool:

        try:
            return bool(self._xrt.is_body_data_available())
        except Exception:
            logger.exception("Failed to query Pico body availability")
            return False

    def start(self) -> None:

        if self._started:
            return
        if self._closed:
            raise RuntimeError("Cannot restart a closed PicoRtCapture")
        sdk_initialized = False
        try:
            self._xrt.init()
            sdk_initialized = True
            if self._requested_input_mode == "callback" or (
                self._requested_input_mode == "auto" and self._callback_available
            ):
                self._xrt.register_frame_callback(self._on_frame)
                self._input_mode = "callback"
            else:
                self._input_mode = "polling"
                self._poll_thread = threading.Thread(
                    target=self._poll_loop,
                    name="pico-xrobot-poll",
                    daemon=True,
                )
                self._poll_thread.start()
        except Exception:
            self._input_mode = "uninitialized"
            if sdk_initialized:
                close = getattr(self._xrt, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception:
                        logger.exception("Failed to close Pico SDK after startup failure")
            raise
        self._started = True
        if self.config.wait_body_available and not self.sdk_body_available:
            logger.info("Pico SDK started in %s mode; body tracking is not available yet.", self._input_mode)

    def close(self, *, close_sdk: bool = True, clear_callback: bool = True) -> None:

        self._closed = True
        self._poll_stop_event.set()
        self._body_event.set()
        self._input_event.set()
        self._pico_controller_event.set()
        if self._poll_thread is not None and self._poll_thread is not threading.current_thread():
            self._poll_thread.join(timeout=1.0)
            if self._poll_thread.is_alive():
                logger.warning("Pico polling thread did not stop within 1.0s")
        if clear_callback and self._input_mode == "callback":
            try:
                self._xrt.clear_frame_callback()
            except Exception:
                logger.exception("Failed to clear Pico frame callback")
        if not close_sdk:
            return
        close = getattr(self._xrt, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                logger.exception("Failed to close Pico SDK")

    def _safe_xrt_call(self, name: str, default: Any = None) -> Any:

        fn = getattr(self._xrt, name, None)
        if not callable(fn):
            return default
        try:
            return fn()
        except Exception:
            return default

    def _build_polling_snapshot(self) -> dict[str, Any]:

        body_available = bool(self._safe_xrt_call("is_body_data_available", False))
        body_source_ns = int(self._safe_xrt_call("get_body_timestamp_ns", 0) or 0)
        source_ns = int(self._safe_xrt_call("get_time_stamp_ns", body_source_ns) or body_source_ns)
        poses = self._safe_xrt_call("get_body_joints_pose", None) if body_available else None
        return {
            "timestamp_ns": source_ns,
            "controllers": {
                "left": {
                    "primary_button": bool(self._safe_xrt_call("get_X_button", False)),
                    "secondary_button": bool(self._safe_xrt_call("get_Y_button", False)),
                    "axis_click": bool(self._safe_xrt_call("get_left_axis_click", False)),
                    "menu_button": bool(self._safe_xrt_call("get_left_menu_button", False)),
                    "trigger": self._safe_xrt_call("get_left_trigger", 0.0),
                    "grip": self._safe_xrt_call("get_left_grip", 0.0),
                    "axis": self._safe_xrt_call("get_left_axis", (0.0, 0.0)),
                },
                "right": {
                    "primary_button": bool(self._safe_xrt_call("get_A_button", False)),
                    "secondary_button": bool(self._safe_xrt_call("get_B_button", False)),
                    "axis_click": bool(self._safe_xrt_call("get_right_axis_click", False)),
                    "menu_button": bool(self._safe_xrt_call("get_right_menu_button", False)),
                    "trigger": self._safe_xrt_call("get_right_trigger", 0.0),
                    "grip": self._safe_xrt_call("get_right_grip", 0.0),
                    "axis": self._safe_xrt_call("get_right_axis", (0.0, 0.0)),
                },
            },
            "body": {
                "available": body_available,
                "timestamp_ns": body_source_ns,
                "poses": poses,
            },
        }

    def _poll_loop(self) -> None:

        period_s = 1.0 / max(float(self.config.poll_hz), 1e-6)
        while not self._closed and not self._poll_stop_event.is_set():
            self._on_frame(self._build_polling_snapshot())
            self._poll_stop_event.wait(period_s)

    def latest_body(self) -> PicoBodyRawFrameNs | None:

        with self._lock:
            return self._latest_body

    def wait_input_activity(self, body_seq: int | None, control_seq: int | None, timeout_s: float) -> None:
        """Wake for either channel, with no lost notification between check/wait."""
        with self._lock:
            if self._closed or (
                self._latest_body is not None and self._latest_body.seq != body_seq
            ) or (
                self._latest_pico_controller is not None and self._latest_pico_controller.seq != control_seq
            ):
                return
            self._input_event.clear()
        self._input_event.wait(max(float(timeout_s), 0.0))

    def latest_retarget_input(self) -> PicoRetargetInputFrameNs | None:

        with self._convert_lock:
            with self._lock:
                raw = self._latest_body
                cached = self._latest_retarget_input
            if raw is None:
                return None
            if cached is not None and cached.body_seq == raw.seq:
                return cached
            human_frame = body_poses_to_human_frame(raw.poses)
            frame = PicoRetargetInputFrameNs(
                body_seq=raw.seq, body_recv_ns=raw.body_recv_ns,
                convert_done_ns=time.monotonic_ns(),
                human_format=self.config.human_format, human_frame=human_frame,
                stream_id=raw.stream_id, clock_domain=raw.clock_domain,
            )
            with self._lock:
                self._latest_retarget_input = frame
            return frame

    def latest_pico_controller(self) -> PicoControlFrameNs | None:

        with self._lock:
            return self._latest_pico_controller

    def wait_next_retarget_input(
        self,
        last_seq: int | None,
        timeout_s: float | None = None,
    ) -> PicoRetargetInputFrameNs:

        deadline = None if timeout_s is None else time.monotonic() + max(float(timeout_s), 0.0)
        while not self._closed:
            with self._lock:
                raw = self._latest_body
                available = raw is not None and (last_seq is None or raw.seq != last_seq)
                if not available:
                    self._body_event.clear()
            if available:
                return self.latest_retarget_input()
            wait_s = self._remaining_timeout(deadline)
            if wait_s is not None and wait_s <= 0.0:
                break
            self._body_event.wait(wait_s)
        raise TimeoutError(f"No Pico RT body frame received within {timeout_s or self.config.timeout_s:.1f}s")

    def wait_next_pico_controller(
        self,
        last_seq: int | None,
        timeout_s: float | None = None,
    ) -> PicoControlFrameNs:

        deadline = None if timeout_s is None else time.monotonic() + max(float(timeout_s), 0.0)
        while not self._closed:
            with self._lock:
                frame = self._latest_pico_controller
                if frame is not None and (last_seq is None or frame.seq != last_seq):
                    return frame
                self._pico_controller_event.clear()
            wait_s = self._remaining_timeout(deadline)
            if wait_s is not None and wait_s <= 0.0:
                break
            self._pico_controller_event.wait(wait_s)
        raise TimeoutError(f"No Pico RT control frame received within {timeout_s or self.config.timeout_s:.1f}s")

    @staticmethod
    def _remaining_timeout(deadline: float | None) -> float | None:

        if deadline is None:
            return None
        return max(deadline - time.monotonic(), 0.0)

    def _on_frame(self, snapshot: dict[str, Any]) -> None:

        if self._closed:
            return
        with self._callback_lock:
            if self._closed:
                return
            recv_ns = time.monotonic_ns()
            if self.raw_snapshots is not None:
                try:
                    self.raw_snapshots.put_nowait({
                        "sdk_seq": self._callback_count + 1, "recv_ns": recv_ns,
                        # Bundled bindings build value snapshots, but the installed
                        # x86_64 extension's ownership is not proven by that source.
                        # Keep an owned copy for asynchronous raw serialization.
                        "snapshot": deepcopy(snapshot),
                    })
                    self.raw_activity_event.set()
                except queue.Full:
                    self.raw_dropped_frames += 1
            try:
                body_poses, source_ns = self._extract_body_poses(snapshot)
                pico_controller = self._extract_pico_controller(snapshot, recv_ns=recv_ns, source_ns=source_ns)
            except Exception:
                logger.exception("Failed to parse Pico RT SDK snapshot")
                return

            with self._lock:
                self._callback_count += 1
                self._pico_controller_seq += 1
                self._latest_pico_controller = pico_controller
                self._pico_controller_event.set()
                self._input_event.set()

            if body_poses is None:
                return

            with self._lock:
                if (
                    self.config.dedupe_source_ns
                    and source_ns is not None
                    and self._last_source_ns == source_ns
                ):
                    return
                if source_ns is None and self._last_body_poses is not None and np.array_equal(
                    body_poses, self._last_body_poses
                ):
                    return
                self._last_source_ns = source_ns
                self._last_body_poses = body_poses.copy()
                self._seq += 1
                seq = self._seq

            raw = PicoBodyRawFrameNs(seq=seq, body_recv_ns=recv_ns, source_ns=source_ns, poses=body_poses,
                                     stream_id=self.stream_id, clock_domain=self._diag.clock_domain)
            if self.raw_frames is not None:
                try:
                    self.raw_frames.put_nowait(raw)
                    self.raw_activity_event.set()
                except queue.Full:
                    self.raw_dropped_frames += 1
            with self._lock:
                self._latest_body = raw
                if self._cached_bodies.maxlen != 0:
                    self._cached_bodies.append(raw)
                self._body_event.set()
                self._input_event.set()

    def _extract_body_poses(self, snapshot: dict[str, Any]) -> tuple[NDArray[np.float64] | None, int | None]:

        if not isinstance(snapshot, dict):
            return None, None
        body = snapshot.get("body", {})
        body = body if isinstance(body, dict) else {}
        source_ns = self._extract_source_ns(snapshot, body)
        if not bool(body.get("available", False)):
            return None, source_ns
        raw_poses = body.get("poses")
        if raw_poses is None:
            return None, source_ns
        poses = np.asarray(raw_poses, dtype=np.float64)
        if poses.ndim != 2 or poses.shape[0] < len(BODY_JOINT_NAMES) or poses.shape[1] < 7:
            logger.warning("Ignoring malformed Pico body poses with shape=%s", poses.shape)
            return None, source_ns
        return poses[: len(BODY_JOINT_NAMES), :7].copy(), source_ns

    @staticmethod
    def _extract_source_ns(snapshot: dict[str, Any], body: dict[str, Any]) -> int | None:

        for container in (body, snapshot):
            try:
                value = int(container.get("timestamp_ns", 0))
            except Exception:
                value = 0
            if value > 0:
                return value
        return None

    def _extract_pico_controller(self, snapshot: dict[str, Any], *, recv_ns: int, source_ns: int | None) -> PicoControlFrameNs:

        controllers = snapshot.get("controllers", {}) if isinstance(snapshot, dict) else {}
        controllers = controllers if isinstance(controllers, dict) else {}
        left = controllers.get("left", {})
        right = controllers.get("right", {})
        left = left if isinstance(left, dict) else {}
        right = right if isinstance(right, dict) else {}
        sides = {"left": left, "right": right}

        buttons = PicoButtonBits.NONE
        for bit, (side_name, field_name) in SNAPSHOT_BUTTON_ALIASES.items():
            if self._button_pressed(sides.get(side_name, {}).get(field_name, False)):
                buttons |= bit

        with self._lock:
            seq = self._pico_controller_seq + 1
        return PicoControlFrameNs(
            seq=seq,
            recv_ns=recv_ns,
            source_ns=source_ns,
            buttons=buttons,
            left_axis=self._axis(left.get("axis", (0.0, 0.0))),
            right_axis=self._axis(right.get("axis", (0.0, 0.0))),
            left_trigger=self._scalar(left.get("trigger", left.get("index_trigger", 0.0))),
            right_trigger=self._scalar(right.get("trigger", right.get("index_trigger", 0.0))),
            left_grip=self._scalar(left.get("grip", left.get("grip_trigger", 0.0))),
            right_grip=self._scalar(right.get("grip", right.get("grip_trigger", 0.0))),
        )

    @staticmethod
    def _axis(raw_axis: object) -> tuple[float, float]:

        try:
            values = np.asarray(raw_axis, dtype=np.float64).reshape(-1)
        except Exception:
            return (0.0, 0.0)
        if values.shape[0] < 2:
            return (0.0, 0.0)
        return (float(values[0]), float(values[1]))

    @staticmethod
    def _scalar(raw_value: object) -> float:

        try:
            values = np.asarray(raw_value, dtype=np.float64).reshape(-1)
        except Exception:
            return 0.0
        if values.size <= 0:
            return 0.0
        value = float(values[0])
        return value if np.isfinite(value) else 0.0

    @staticmethod
    def _button_pressed(raw_value: object) -> bool:

        if isinstance(raw_value, np.ndarray):
            if raw_value.size <= 0:
                return False
            raw_value = raw_value.reshape(-1)[0]
        if isinstance(raw_value, (tuple, list)):
            if not raw_value:
                return False
            raw_value = raw_value[0]
        if isinstance(raw_value, (bool, np.bool_)):
            return bool(raw_value)
        if isinstance(raw_value, (int, float, np.integer, np.floating)):
            return float(raw_value) > 0.5
        return bool(raw_value)
