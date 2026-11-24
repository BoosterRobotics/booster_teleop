
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import logging
import threading
import time
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from booster_deploy.types import HumanFrame

from .adapter import MOCAP_HIERARCHY, MOCAP_JOINT_NAMES, PNLinkMocapAdapter, add_synthetic_tip_joints
from .calibration_sdk import load_mocap_api
from booster_deploy.types.mocap_types import MocapJoints

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CalibrationProgress:

    seq: int
    support_poses: tuple[str, ...]
    pose_name: str
    step: str
    countdown: int | None
    progress: int | None
    text: str


@dataclass(frozen=True)
class CommandResultDetail:

    command: int | None
    result_code: int
    result_message: str


class PNLinkMocapProvider:

    def __init__(
        self,
        *,
        human_format: str = "bvh",
        human_height: float = 1.75,
        local_ip: str = "10.42.0.1",
        local_port: int = 8002,
        server_ip: str = "10.42.0.202",
        server_port: int = 8080,
        auto_start_capture: bool = False,
        skip_calibration: bool = False,
        buffer_size: int = 60,
        timeout: float = 10.0,
        poll_hz: float = 100.0,
        synthetic_tip_quat_mode: str = "wxyz_identity",
    ) -> None:

        self._mocap_api = load_mocap_api()
        self._human_format = human_format
        self._human_height = float(human_height)
        self._timeout = float(timeout)
        self._poll_period_s = 1.0 / max(float(poll_hz), 1.0)
        self._adapter = PNLinkMocapAdapter(MOCAP_HIERARCHY)

        self._frames: deque[tuple[HumanFrame, float, int]] = deque(maxlen=max(int(buffer_size), 1))
        self._fps_window: deque[float] = deque(maxlen=30)

        self._lock = threading.Lock()
        self._frame_ready = threading.Event()
        self._running = True
        self._seq = 0
        self._capture_started = False
        self._calibration_done = bool(skip_calibration)
        self._avatar_seen = False
        self._last_command_result: tuple[int | None, int | None] | None = None
        self._last_command_result_detail: CommandResultDetail | None = None
        self._last_calibration_progress: CalibrationProgress | None = None
        self._calibration_progress_history: deque[CalibrationProgress] = deque(maxlen=256)
        self._progress_seq = 0
        self._current_command: int | None = None
        self._synthetic_tip_quat_mode = synthetic_tip_quat_mode

        self._app = self._mocap_api.MCPApplication()
        settings = self._mocap_api.MCPSettings()
        settings.set_bvh_data(self._mocap_api.MCPBvhData.Binary)
        settings.set_bvh_transformation(self._mocap_api.MCPBvhDisplacement.Enable)
        settings.set_bvh_rotation(self._mocap_api.MCPBvhRotation.ZYX)
        settings.SetSettingsUDPEx(local_ip, int(local_port))
        settings.SetSettingsUDPServer(server_ip, int(server_port))
        self._settings = settings
        self._app.set_settings(settings)
        opened, message = self._app.open()
        if not opened:
            raise RuntimeError(f"Failed to open PN-Link MocapApi application: {message}")

        self._thread = threading.Thread(target=self._poll_loop, daemon=True, name="pnlink_mocap")
        self._thread.start()
        if auto_start_capture:
            self.start_capture()


    @property
    def fps(self) -> float:

        with self._lock:
            window = list(self._fps_window)
        if len(window) < 2:
            return 0.0
        elapsed = window[-1] - window[0]
        if elapsed <= 0:
            return 0.0
        return (len(window) - 1) / elapsed

    @property
    def human_format(self) -> str:

        return self._human_format

    @property
    def human_height(self) -> float:

        return self._human_height

    @property
    def bone_names(self) -> list[str]:

        return list(MOCAP_JOINT_NAMES)

    @property
    def bone_parents(self) -> NDArray[np.int32]:

        name_to_idx = {name: idx for idx, name in enumerate(MOCAP_JOINT_NAMES)}
        parents: list[int] = []
        for joint_name in MOCAP_JOINT_NAMES:
            parent_name = MOCAP_HIERARCHY.get(joint_name, "world")
            parents.append(-1 if parent_name == "world" else name_to_idx[parent_name])
        return np.asarray(parents, dtype=np.int32)

    def is_available(self) -> bool:

        return self._running and self._thread.is_alive()

    @property
    def capture_started(self) -> bool:

        return self._capture_started

    @property
    def calibration_done(self) -> bool:

        return self._calibration_done

    @property
    def avatar_seen(self) -> bool:

        return self._avatar_seen

    @property
    def last_command_result(self) -> tuple[int | None, int | None] | None:

        return self._last_command_result

    @property
    def last_command_result_detail(self) -> CommandResultDetail | None:

        return self._last_command_result_detail

    @property
    def last_calibration_progress(self) -> CalibrationProgress | None:

        return self._last_calibration_progress

    def drain_calibration_progress(self, last_seq: int = -1) -> list[CalibrationProgress]:

        with self._lock:
            snapshot = list(self._calibration_progress_history)
        return [p for p in snapshot if p.seq > last_seq]

    @property
    def pending_command(self) -> int | None:

        return self._current_command

    def command_title(self, command: int | None = None) -> str:

        api = self._mocap_api
        target = command if command is not None else self._current_command
        if target is None:
            return "None"
        try:
            if target == api.EMCPCommand.CommandStartCapture:
                return "Start Capture"
            if target == api.EMCPCommand.CommandStopCapture:
                return "Stop Capture"
            if target == api.EMCPCommand.CommandCalibrateMotion:
                return "Calibrate Motion"
            if target == api.EMCPCommand.CommandResumeOriginalHandsPosture:
                return "Resume Hands"
            if target == api.EMCPCommand.CommandClearZeroMotionDrift:
                return "Reset 0 Motion Drift"
            if target == api.EMCPCommand.CommandResumeOriginalPosture:
                return "Resume Body"
            if target == api.EMCPCommand.CommandZeroPosition:
                return "Zero Position"
        except AttributeError:
            pass
        return f"Command<{int(target)}>"


    def start_capture(self) -> None:

        self._queue_command(self._mocap_api.EMCPCommand.CommandStartCapture)

    def stop_capture(self) -> None:

        self._queue_command(self._mocap_api.EMCPCommand.CommandStopCapture)

    def calibrate(self) -> None:

        self._queue_command(self._mocap_api.EMCPCommand.CommandCalibrateMotion)


    def get_frame(self) -> HumanFrame:

        return self.get_frame_packet()[0]

    def get_frame_packet(self) -> tuple[HumanFrame, float, int]:

        with self._lock:
            if self._frames:
                return self._frames[-1]
        if not self._frame_ready.wait(timeout=self._timeout):
            raise TimeoutError(f"No PN-Link mocap frame received within {self._timeout:.1f}s")
        with self._lock:
            if not self._frames:
                raise RuntimeError("PN-Link frame buffer signaled ready without a latest frame")
            return self._frames[-1]

    def sample_frame(self, query_time_s: float, delay_s: float) -> HumanFrame:

        if not self._frame_ready.wait(timeout=self._timeout):
            raise TimeoutError(f"No PN-Link mocap frame received within {self._timeout:.1f}s")
        with self._lock:
            buf = [(frame, ts) for frame, ts, _seq in self._frames]
        return _sample_buffer(buf, query_time_s=query_time_s, delay_s=delay_s)

    def close(self) -> None:

        self._running = False
        self._frame_ready.set()
        self._thread.join(timeout=2.0)
        try:
            self._app.close()
        except Exception:
            logger.exception("Failed to close PN-Link MocapApi application")


    def _queue_command(self, command: int) -> None:

        self._current_command = command
        self._app.queue_command(command)

    def _poll_loop(self) -> None:

        while self._running:
            try:
                events = self._app.poll_next_event()
                for event in events:
                    self._handle_event(event)
            except Exception:
                logger.exception("PN-Link mocap polling failed")
                time.sleep(0.05)
            time.sleep(self._poll_period_s)

    def _handle_event(self, event) -> None:

        api = self._mocap_api
        if event.event_type == api.MCPEventType.AvatarUpdated:
            self._avatar_seen = True

            if self._current_command == api.EMCPCommand.CommandCalibrateMotion:
                return
            if self._calibration_done:
                avatar = api.MCPAvatar(event.event_data.avatar_handle)
                self._accept_avatar(avatar)
        elif event.event_type == api.MCPEventType.CommandReply:
            reply = event.event_data.commandRespond
            if reply._replay == api.MCPReplay.MCPReplay_Running:
                self._handle_command_running(reply)
            elif reply._replay == api.MCPReplay.MCPReplay_Result:
                self._handle_command_reply(reply)
        elif event.event_type == api.MCPEventType.Notify:
            logger.info("PN-Link notify event received")

    def _handle_command_running(self, command_respond) -> None:

        api = self._mocap_api
        progress_handle = api.MCPCommand().get_progress(command_respond._commandHandle)
        progress = api.MCPCalibrateMotionProgress(progress_handle)

        support_count = progress.get_count_of_support_poses()
        poses: list[str] = []
        for i in range(support_count):
            poses.append(progress.get_name_of_support_poses(i))

        current_step, pose_name = progress.get_step_current_pose()
        step_label = "Unknown"
        countdown: int | None = None
        progress_val: int | None = None

        try:
            steps = api.MCPCalibrateMotionProgressStep
            if current_step == steps.CalibrateMotionProgressStep_Countdown:
                step_label = "Countdown"
                countdown, pose_name = progress.get_countdown_current_pose()
            elif current_step == steps.CalibrateMotionProgressStep_Progress:
                step_label = "Progress"
                progress_val, pose_name = progress.get_progress_current_pose()
            elif current_step == steps.CalibrateMotionProgressStep_Prepare:
                step_label = "Prepare"
            else:
                step_label = f"Unknown({current_step})"
        except AttributeError:
            step_label = f"Unknown({current_step})"

        if step_label == "Countdown":
            detail = f"Calibration-({pose_name})-Countdown {countdown}"
        elif step_label == "Progress":
            detail = f"Calibration-({pose_name})-Progress {progress_val}"
        elif step_label == "Prepare":
            detail = f"Calibration-({pose_name}) is preparing"
        else:
            detail = f"Calibration-{step_label} is running"
        text = f"Support poses: {', '.join(poses)} : {detail}."

        with self._lock:
            self._progress_seq += 1
            entry = CalibrationProgress(
                seq=self._progress_seq,
                support_poses=tuple(poses),
                pose_name=str(pose_name) if pose_name is not None else "",
                step=step_label,
                countdown=int(countdown) if countdown is not None else None,
                progress=int(progress_val) if progress_val is not None else None,
                text=text,
            )
            self._last_calibration_progress = entry
            self._calibration_progress_history.append(entry)

    def _handle_command_reply(self, command_respond) -> None:

        api = self._mocap_api
        command = api.MCPCommand()
        finished_command = self._current_command
        try:
            result_code = command.get_result_code(command_respond._commandHandle)
            try:
                result_message = command.get_result_message(command_respond._commandHandle)
            except AttributeError:
                result_message = ""
            self._last_command_result = (finished_command, result_code)
            self._last_command_result_detail = CommandResultDetail(
                command=finished_command,
                result_code=int(result_code),
                result_message=str(result_message or ""),
            )
            if result_code == api.MCPError.NoError:
                if finished_command == api.EMCPCommand.CommandStartCapture:
                    self._capture_started = True
                elif finished_command == api.EMCPCommand.CommandStopCapture:
                    self._capture_started = False
                elif finished_command == api.EMCPCommand.CommandCalibrateMotion:
                    self._calibration_done = True
            else:
                logger.info(
                    "PN-Link command=%s failed: code=%s message=%r",
                    finished_command,
                    result_code,
                    result_message,
                )
        finally:
            command.destroy_command(command_respond._commandHandle)
            self._current_command = None

    def _accept_avatar(self, avatar) -> None:

        joints: MocapJoints = {}
        for joint in avatar.get_joints():
            position = joint.get_local_position()
            if position is None:
                continue
            joints[joint.get_name()] = {
                "local_pos": list(position),
                "local_rot": list(joint.get_local_rotation()),
            }
        joints = add_synthetic_tip_joints(joints, quat_mode=self._synthetic_tip_quat_mode)
        frame = self._adapter.joints_to_human_frame(joints)
        timestamp = time.monotonic()
        with self._lock:
            seq = self._seq
            self._seq += 1
            self._frames.append((frame, timestamp, seq))
            self._fps_window.append(timestamp)
        self._frame_ready.set()


def _sample_buffer(buf: list[tuple[HumanFrame, float]], *, query_time_s: float, delay_s: float) -> HumanFrame:

    if not buf:
        raise RuntimeError("Realtime frame buffer is empty")
    if len(buf) == 1:
        return buf[0][0]
    target_time = float(query_time_s - max(delay_s, 0.0))
    if target_time <= buf[0][1]:
        return buf[0][0]
    if target_time >= buf[-1][1]:
        return buf[-1][0]
    for i in range(1, len(buf)):
        older_frame, _older_ts = buf[i - 1]
        _newer_frame, newer_ts = buf[i]
        if target_time <= newer_ts:
            return older_frame
    return buf[-1][0]


__all__ = [
    "CalibrationProgress",
    "CommandResultDetail",
    "PNLinkMocapProvider",
]
