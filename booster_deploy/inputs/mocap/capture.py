
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import logging
import threading
import time

from .adapter import PNLinkMocapAdapter, add_synthetic_tip_joints
from .sdk import load_mocap_api
from booster_deploy.types.mocap_types import MocapBodyRawFrameNs, MocapJoints, MocapRetargetInputFrameNs

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MocapRtCaptureConfig:

    human_format: str = "bvh"
    local_ip: str = "10.42.0.1"
    local_port: int = 8002
    server_ip: str = "10.42.0.202"
    server_port: int = 8080
    poll_hz: float = 500.0
    timeout_s: float = 30.0
    dedupe_source_ns: bool = False
    max_cached_frames: int = 4
    synthetic_tip_quat_mode: str = "wxyz_identity"


class MocapRtCapture:

    def __init__(self, config: MocapRtCaptureConfig | None = None) -> None:

        self.config = config or MocapRtCaptureConfig()
        self._init_source_telemetry()
        self._mocap_api = load_mocap_api()
        self._adapter = PNLinkMocapAdapter()
        self._poll_period_s = 1.0 / max(float(self.config.poll_hz), 1.0)

        self._closed = False
        self._started = False
        self._seq = 0
        self._callback_count = 0
        self._last_source_ns: int | None = None
        self._latest_body: MocapBodyRawFrameNs | None = None
        self._latest_retarget_input: MocapRetargetInputFrameNs | None = None
        self._cached_bodies: deque[MocapBodyRawFrameNs] = deque(
            maxlen=max(int(self.config.max_cached_frames), 0)
        )
        self._body_event = threading.Event()
        self._lock = threading.Lock()
        self._poll_lock = threading.Lock()
        self._thread: threading.Thread | None = None

        self._app = self._mocap_api.MCPApplication()
        settings = self._mocap_api.MCPSettings()
        settings.set_bvh_data(self._mocap_api.MCPBvhData.Binary)
        settings.set_bvh_transformation(self._mocap_api.MCPBvhDisplacement.Enable)
        settings.set_bvh_rotation(self._mocap_api.MCPBvhRotation.ZYX)
        settings.SetSettingsUDPEx(self.config.local_ip, int(self.config.local_port))
        settings.SetSettingsUDPServer(self.config.server_ip, int(self.config.server_port))
        self._settings = settings
        self._app.set_settings(settings)
        opened, message = self._app.open()
        if not opened:
            raise RuntimeError(f"Failed to open PN-Link MocapApi application: {message}")

    def _init_source_telemetry(self):
        import uuid
        from booster_deploy.utils.diag_report import reporter
        self._stream_id = uuid.uuid4().hex
        self._source_epoch = 0
        self._diag = reporter('mocap_capture')

    @property
    def callback_count(self) -> int:

        with self._lock:
            return self._callback_count

    @property
    def body_seq(self) -> int:

        with self._lock:
            return self._seq

    def start(self) -> None:

        if self._started:
            return
        self._started = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True, name="mocap_rt_capture")
        self._thread.start()

    def close(self, *, close_sdk: bool = True) -> None:

        self._closed = True
        self._body_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._diag.close()
        if not close_sdk:
            return
        try:
            self._app.close()
        except Exception:
            logger.exception("Failed to close PN-Link MocapApi application")

    def latest_body(self) -> MocapBodyRawFrameNs | None:

        with self._lock:
            return self._latest_body

    def latest_retarget_input(self) -> MocapRetargetInputFrameNs | None:

        with self._lock:
            return self._latest_retarget_input

    def wait_next_body(
        self,
        last_seq: int | None,
        timeout_s: float | None = None,
    ) -> MocapBodyRawFrameNs:

        effective_timeout = self.config.timeout_s if timeout_s is None else float(timeout_s)
        deadline = None if effective_timeout is None else time.monotonic() + max(effective_timeout, 0.0)
        while not self._closed:
            with self._lock:
                frame = self._latest_body
                if frame is not None and (last_seq is None or frame.seq != last_seq):
                    return frame
                self._body_event.clear()
            wait_s = self._remaining_timeout(deadline)
            if wait_s is not None and wait_s <= 0.0:
                break
            self._body_event.wait(wait_s)
        raise TimeoutError(
            f"No PN-Link mocap RT raw body frame received within {effective_timeout:.1f}s"
        )

    def wait_next_retarget_input(
        self,
        last_seq: int | None,
        timeout_s: float | None = None,
    ) -> MocapRetargetInputFrameNs:

        effective_timeout = self.config.timeout_s if timeout_s is None else float(timeout_s)
        deadline = None if effective_timeout is None else time.monotonic() + max(effective_timeout, 0.0)
        while not self._closed:
            with self._lock:
                frame = self._latest_retarget_input
                if frame is not None and (last_seq is None or frame.body_seq != last_seq):
                    return frame
                self._body_event.clear()
            wait_s = self._remaining_timeout(deadline)
            if wait_s is not None and wait_s <= 0.0:
                break
            self._body_event.wait(wait_s)
        raise TimeoutError(
            f"No PN-Link mocap RT body frame received within {effective_timeout:.1f}s"
        )

    @staticmethod
    def _remaining_timeout(deadline: float | None) -> float | None:

        if deadline is None:
            return None
        return max(deadline - time.monotonic(), 0.0)

    def _poll_loop(self) -> None:

        while not self._closed:
            try:
                with self._poll_lock:
                    events = self._app.poll_next_event() if not self._closed else []
                for event in events:
                    if self._closed:
                        break
                    recv_ns = time.monotonic_ns()
                    self._handle_event(event, recv_ns=recv_ns)
            except Exception:
                logger.exception("PN-Link mocap RT polling failed")
                time.sleep(0.05)
            time.sleep(self._poll_period_s)

    def _handle_event(self, event, *, recv_ns: int) -> None:

        api = self._mocap_api
        if event.event_type != api.MCPEventType.AvatarUpdated:
            return
        avatar = api.MCPAvatar(event.event_data.avatar_handle)
        self._accept_avatar(avatar, recv_ns=recv_ns)

    def _accept_avatar(self, avatar, *, recv_ns: int) -> None:

        joints: MocapJoints = {}
        for joint in avatar.get_joints():
            position = joint.get_local_position()
            if position is None:
                continue
            joints[joint.get_name()] = {
                "local_pos": list(position),
                "local_rot": list(joint.get_local_rotation()),
            }
        if not joints:
            return

        joints = add_synthetic_tip_joints(joints, quat_mode=self.config.synthetic_tip_quat_mode)
        human_frame = self._adapter.joints_to_human_frame(joints)
        convert_done_ns = time.monotonic_ns()

        with self._lock:
            if (
                self.config.dedupe_source_ns
                and self._last_source_ns is not None
                and self._last_source_ns == recv_ns
            ):

                return
            self._seq += 1
            seq = self._seq
            self._callback_count += 1

            raw = MocapBodyRawFrameNs(
                seq=seq,
                body_recv_ns=recv_ns,
                source_ns=None,
                raw_joints=joints, stream_id=self._stream_id,
                source_epoch=self._source_epoch, clock_domain=self._diag.clock_domain,
            )
            retarget_input = MocapRetargetInputFrameNs(
                body_seq=seq,
                body_recv_ns=recv_ns,
                convert_done_ns=convert_done_ns,
                human_format=self.config.human_format,
                human_frame=human_frame, stream_id=self._stream_id,
                source_epoch=self._source_epoch, clock_domain=self._diag.clock_domain,
            )
            self._latest_body = raw
            self._latest_retarget_input = retarget_input
            if self._cached_bodies.maxlen != 0:
                self._cached_bodies.append(raw)
            self._body_event.set()



__all__ = [
    "MocapRtCapture",
    "MocapRtCaptureConfig",
]
