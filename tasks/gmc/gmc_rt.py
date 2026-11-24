
from __future__ import annotations

import os
import threading
import time
from dataclasses import MISSING, replace
from typing import Callable, Sequence

import numpy as np
import torch

try:
    import onnxruntime as ort
except ImportError as e:
    ort = None
    _ORT_IMPORT_ERROR = e
else:
    _ORT_IMPORT_ERROR = None

from booster_deploy.controllers.base_controller import BaseController, Policy
from booster_deploy.controllers.controller_cfg import PolicyCfg
from booster_deploy.retargeting.rt_reference_consumer import (
    RtReferenceConsumer,
    RtReferenceConsumerCfg,
)
from booster_deploy.retargeting.rt_reference_transition import blend_reference_windows
from booster_deploy.types.retarget_types import ReferenceFrame50HzNs
from booster_deploy.retargeting.rt_zmq import Reference50HzSubscriber
from booster_deploy.utils.inference_diagnostics import InferenceDiagnostics
from booster_deploy.utils.isaaclab import math as lab_math
from booster_deploy.utils.diag_report import below_floor, format_gap, format_lat, reporter
from booster_deploy.utils.isaaclab.configclass import configclass
from booster_deploy.utils.offline_replay import offline_replay_enabled, offline_replay_end_timeout_s
from booster_deploy.utils.project_path import resolve_project_path
from .observations import flatten_history, state_row, state_term_widths


_RT_REFERENCE_WAIT_STATUS_INTERVAL_S = 5.0


def _make_onnx_session(model_path: str) -> ort.InferenceSession:
    if ort is None:
        raise ImportError("onnxruntime is required for M0-1010") from _ORT_IMPORT_ERROR

    so = ort.SessionOptions()
    providers = ["CPUExecutionProvider"]
    if "CUDAExecutionProvider" in ort.get_available_providers():
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ort.InferenceSession(model_path, so, providers=providers)


def _input_last_dim(inp) -> int | None:
    if not inp.shape:
        return None
    last = inp.shape[-1]
    return last if isinstance(last, int) and last > 0 else None


def _resolve_onnx_inputs(inputs, motion_dim: int, state_dim: int) -> tuple[str, str]:
    by_name: dict[str, str] = {}
    for inp in inputs:
        last_dim = _input_last_dim(inp)
        if last_dim == motion_dim and last_dim != state_dim:
            by_name["motion"] = inp.name
        elif last_dim == state_dim and last_dim != motion_dim:
            by_name["state"] = inp.name

    if "motion" not in by_name or "state" not in by_name:
        for inp in inputs:
            name = inp.name.lower()
            if "state" in name and "state" not in by_name:
                by_name["state"] = inp.name
            elif any(token in name for token in ("motion", "command", "encoder", "teleop")) and "motion" not in by_name:
                by_name["motion"] = inp.name

    if "motion" in by_name and "state" not in by_name:
        by_name["state"] = next(inp.name for inp in inputs if inp.name != by_name["motion"])
    elif "state" in by_name and "motion" not in by_name:
        by_name["motion"] = next(inp.name for inp in inputs if inp.name != by_name["state"])

    if "motion" not in by_name or "state" not in by_name:
        by_name["motion"] = inputs[0].name
        by_name["state"] = inputs[1].name

    motion_name, state_name = by_name["motion"], by_name["state"]
    expected = {motion_name: motion_dim, state_name: state_dim}
    for inp in inputs:
        last = _input_last_dim(inp)
        if last is not None and last != expected[inp.name]:
            raise ValueError(
                f"ONNX input '{inp.name}' last dim is {last}, "
                f"but config-derived dim is {expected[inp.name]}."
            )
    return motion_name, state_name


class ReferenceFramePool:

    def __init__(self, capacity: int, num_joints: int) -> None:

        self.capacity = max(int(capacity), 1)
        self.num_joints = int(num_joints)
        self._frames: dict[int, ReferenceFrame50HzNs] = {}
        self._latest_ready_idx: int | None = None
        self._lock = threading.Lock()

    def append(self, frame: ReferenceFrame50HzNs) -> None:

        if frame.joint_pos.shape[0] != self.num_joints:
            raise ValueError(f"reference joint_pos dim {frame.joint_pos.shape[0]} != expected {self.num_joints}")
        with self._lock:
            self._frames[int(frame.sample_idx)] = frame
            self._latest_ready_idx = int(frame.sample_idx)
            while len(self._frames) > self.capacity:
                oldest = min(self._frames)
                self._frames.pop(oldest, None)

    def latest_ready_idx(self) -> int | None:

        with self._lock:
            return self._latest_ready_idx

    def size(self) -> int:

        with self._lock:
            return len(self._frames)

    def get(self, sample_idx: int) -> ReferenceFrame50HzNs | None:

        with self._lock:
            return self._frames.get(int(sample_idx))

    def get_window(self, indices: Sequence[int]) -> list[ReferenceFrame50HzNs] | None:

        with self._lock:
            frames: list[ReferenceFrame50HzNs] = []
            for idx in indices:
                frame = self._frames.get(int(idx))
                if frame is None:
                    return None
                frames.append(frame)
            return frames


class RtReferenceReceiver:

    def __init__(self, cfg: "GmcRtPolicyCfg") -> None:

        self.cfg = cfg
        self.pool = ReferenceFramePool(int(cfg.rt_reference_capacity), int(cfg.num_joints))
        self._subscriber: Reference50HzSubscriber | None = None
        self._thread: threading.Thread | None = None
        self._closed = threading.Event()
        self._started = False
        self._last_sample_idx: int | None = None
        self.first_sample_idx = None
        self.offline_cycle = 0
        self._last_error: Exception | None = None

    def start(self) -> None:

        if self._started:
            return
        self._started = True
        self._subscriber = Reference50HzSubscriber(
            host=self.cfg.rt_reference_host,
            port=int(self.cfg.rt_reference_port),
            topic=self.cfg.rt_reference_topic,
            recv_hwm=int(self.cfg.rt_reference_capacity),
            recv_timeout_ms=50,
        )
        self._thread = threading.Thread(target=self._run, name="rt-reference-receiver", daemon=True)
        self._thread.start()

    def close(self) -> None:

        self._closed.set()


        if self._thread is not None:
            self._thread.join(timeout=1.0)
            if self._thread.is_alive():
                raise RuntimeError("RT reference receiver did not stop within 1 second")
            self._thread = None
        if self._subscriber is not None:
            self._subscriber.close()
            self._subscriber = None

    def latest_ready_idx(self) -> int | None:

        return self.pool.latest_ready_idx()

    def get_window(self, indices: Sequence[int]) -> list[ReferenceFrame50HzNs] | None:

        return self.pool.get_window(indices)

    def wait_until_ready(
        self,
        min_samples: int,
        timeout_s: float,
        *,
        is_ready: Callable[[], bool] | None = None,
    ) -> None:

        min_samples = max(int(min_samples), 0)
        timeout_s = max(float(timeout_s), 0.0)
        started_s = time.monotonic()
        deadline = started_s + timeout_s
        next_status_s = started_s
        endpoint = (
            f"tcp://{self.cfg.rt_reference_host}:"
            f"{int(self.cfg.rt_reference_port)}/{self.cfg.rt_reference_topic}"
        )
        latest: int | None = None
        sample_count = 0
        while True:
            now_s = time.monotonic()
            latest = self.latest_ready_idx()
            sample_count = self.pool.size()
            enough_samples = latest is not None and sample_count >= min_samples
            complete_window = enough_samples and (
                is_ready is None or is_ready()
            )
            if complete_window:
                print(
                    "[RT reference] Ready: "
                    f"endpoint={endpoint} received={sample_count}/{min_samples} "
                    f"latest_idx={latest}",
                    flush=True,
                )
                return
            if self._last_error is not None:
                raise RuntimeError(
                    f"RT reference receiver failed for endpoint={endpoint}"
                ) from self._last_error
            if now_s >= next_status_s:
                print(
                    "[RT reference] Waiting for startup window: "
                    f"endpoint={endpoint} received={sample_count}/{min_samples} "
                    f"latest_idx={latest} complete_window={complete_window} "
                    f"elapsed={now_s - started_s:.1f}/{timeout_s:.1f}s. "
                    "If this is not the current upper-computer IP, update "
                    "BOOSTER_RT_REFERENCE_HOST or re-source scripts/env_set.sh.",
                    flush=True,
                )
                next_status_s = now_s + _RT_REFERENCE_WAIT_STATUS_INTERVAL_S
            if now_s >= deadline:
                break
            time.sleep(min(0.005, max(deadline - now_s, 0.0)))

        message = (
            "RT reference startup timed out: "
            f"endpoint={endpoint} received={sample_count}/{min_samples} "
            f"latest_idx={latest} complete_window=False after {timeout_s:.1f}s. "
            "Verify that BOOSTER_RT_REFERENCE_HOST is the current "
            "upper-computer IP and that the retarget worker publishes this topic."
        )
        print(f"[RT reference] ERROR: {message}", flush=True)
        raise TimeoutError(message)

    def _run(self) -> None:

        assert self._subscriber is not None
        while not self._closed.is_set():
            try:
                frames = self._subscriber.recv_newer_than(self._last_sample_idx, timeout_s=0.2)
            except TimeoutError:
                continue
            except Exception as exc:
                self._last_error = exc
                time.sleep(0.1)
                continue
            for frame in frames:
                if self.offline_cycle and getattr(frame,"offline_cycle",0) != self.offline_cycle:
                    continue
                if self.first_sample_idx is None:
                    self.first_sample_idx = int(frame.sample_idx)
                    if self.offline_cycle:
                        import os
                        from booster_deploy.utils.replay_control import _write_token
                        _write_token(os.getenv('BOOSTER_DIAG_DIR'), 'replay.first', self.offline_cycle)
                self.pool.append(frame)
                self._last_sample_idx = frame.sample_idx


class RtReferenceAlignment:

    def __init__(
        self,
        *,
        enabled: bool,
        align_xy: bool,
        align_yaw: bool,
        device: torch.device | str,
    ) -> None:
        self.enabled = bool(enabled)
        self.align_xy = bool(align_xy)
        self.align_yaw = bool(align_yaw)
        self.device = device
        self._base_root_pos: torch.Tensor | None = None
        self._target_root_pos: torch.Tensor | None = None
        self._yaw_offset_quat: torch.Tensor | None = None
        self._pending_reanchor = bool(enabled)

    @property
    def needs_anchor(self) -> bool:

        return self.enabled and self._pending_reanchor

    def reset(self) -> None:

        self._base_root_pos = None
        self._target_root_pos = None
        self._yaw_offset_quat = None
        self._pending_reanchor = bool(self.enabled)

    def request_reanchor(self) -> None:

        if not self.enabled:
            return
        self._base_root_pos = None
        self._target_root_pos = None
        self._yaw_offset_quat = None
        self._pending_reanchor = True

    def align_to_robot(
        self,
        *,
        robot_root_pos_w: torch.Tensor,
        robot_root_quat_w: torch.Tensor,
        reference_window: Sequence[ReferenceFrame50HzNs],
    ) -> None:

        self.reset()
        if not self.enabled or not reference_window:
            return

        base = reference_window[0]
        base_root_pos = torch.as_tensor(base.root_pos, dtype=torch.float32, device=self.device).reshape(3)
        base_root_quat = torch.as_tensor(base.root_quat, dtype=torch.float32, device=self.device).reshape(4)
        robot_root_pos = robot_root_pos_w.to(device=self.device, dtype=torch.float32).reshape(3)
        robot_root_quat = robot_root_quat_w.to(device=self.device, dtype=torch.float32).reshape(4)

        if self.align_yaw:
            robot_yaw = lab_math.yaw_quat(robot_root_quat)
            ref_yaw = lab_math.yaw_quat(base_root_quat)
            yaw_offset = lab_math.quat_mul(robot_yaw, lab_math.quat_inv(ref_yaw))
        else:
            yaw_offset = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=self.device)

        self._base_root_pos = base_root_pos
        self._target_root_pos = robot_root_pos
        self._yaw_offset_quat = yaw_offset
        self._pending_reanchor = False

    def apply(self, reference_window: Sequence[ReferenceFrame50HzNs]) -> list[ReferenceFrame50HzNs]:

        if not reference_window:
            return []
        if not self.enabled or self._yaw_offset_quat is None:
            return list(reference_window)

        root_pos = torch.stack(
            [
                torch.as_tensor(frame.root_pos, dtype=torch.float32, device=self.device)
                for frame in reference_window
            ],
            dim=0,
        )
        root_quat = torch.stack(
            [
                torch.as_tensor(frame.root_quat, dtype=torch.float32, device=self.device)
                for frame in reference_window
            ],
            dim=0,
        )
        yaw_offset = self._yaw_offset_quat.reshape(1, 4).repeat(root_pos.shape[0], 1)

        if self.align_xy and self._base_root_pos is not None and self._target_root_pos is not None:
            rel_pos = root_pos - self._base_root_pos.reshape(1, 3)
            aligned_pos = lab_math.quat_apply(yaw_offset, rel_pos)
            aligned_pos = aligned_pos + self._target_root_pos.reshape(1, 3)
            aligned_pos[:, 2] = root_pos[:, 2]
        elif self.align_yaw:
            aligned_pos = lab_math.quat_apply(yaw_offset, root_pos)
        else:
            aligned_pos = root_pos

        if self.align_yaw:
            aligned_quat = lab_math.quat_mul(yaw_offset, root_quat)
        else:
            aligned_quat = root_quat

        aligned_pos_np = aligned_pos.detach().cpu().numpy().astype(np.float64, copy=False)
        aligned_quat_np = aligned_quat.detach().cpu().numpy().astype(np.float64, copy=False)
        return [
            ReferenceFrame50HzNs(
                sample_idx=frame.sample_idx,
                sample_ns=frame.sample_ns,
                body_seq=frame.body_seq,
                source_ids=frame.source_ids, clock_domain=frame.clock_domain, identity=frame.identity,
                body_recv_ns=frame.body_recv_ns,
                retarget_done_ns=frame.retarget_done_ns,
                robot_name=frame.robot_name,
                joint_pos=frame.joint_pos.copy(),
                joint_vel=frame.joint_vel.copy(),
                root_pos=aligned_pos_np[i].copy(),
                root_quat=aligned_quat_np[i].copy(),
                human_format=frame.human_format,
                human_frame=frame.human_frame,
                hand_frame=frame.hand_frame,
            )
            for i, frame in enumerate(reference_window)
        ]


@configclass
class GmcRtPolicyCfg(PolicyCfg):

    constructor = None
    checkpoint_path: str = ""
    onnx_model_path: str = MISSING

    traj_anchor_name: str = "pelvis"
    obs_anchor_name: str = "torso_link"
    obs_anchor_fk_chain: list[tuple[int, list[float]]] | None = None

    num_joints: int = 29
    num_actions: int = 29

    history_length: int = 10
    command_history_length: int | None = None
    state_history_length: int | None = None
    command_input_mode: str = "motion"
    command_window_mode: str = "future"
    command_num_future_frames: int = 1
    command_frame_interval: int = 0

    include_future_anchor_ori_in_command: bool = True
    include_projected_gravity_in_state: bool = True
    include_root_ang_vel_in_state: bool = True
    include_joint_pos_in_state: bool = True
    include_joint_vel_in_state: bool = True
    include_last_action_in_state: bool = True
    action_offset_source: str = "command"
    clip_actions: float | None = 10.0

    projected_gravity_dim: int = 3
    anchor_ori_dim: int = 6
    root_ang_vel_dim: int = 3

    input_path: str = "retarget"
    rt_reference_host: str = "127.0.0.1"
    rt_reference_port: int = 5556
    rt_reference_topic: str = "retarget_rt.reference_50hz"
    rt_reference_timeout_s: float = 30.0
    rt_reference_capacity: int = 128
    rt_startup_min_samples: int | None = None
    rt_reference_consumer: RtReferenceConsumerCfg = RtReferenceConsumerCfg()
    rt_align_reference_on_resume: bool = True
    rt_align_reference_xy: bool = True
    rt_align_reference_yaw: bool = True
    rt_reference_transition_s: float = 1.0


class GmcRtPolicy(Policy):

    def __init__(self, cfg: GmcRtPolicyCfg, controller: BaseController):

        super().__init__(cfg, controller)
        if not np.isfinite(cfg.rt_reference_transition_s) or cfg.rt_reference_transition_s < 0.0:
            raise ValueError("rt_reference_transition_s must be finite and >= 0")
        self.robot = controller.robot
        self.robot.data.to(self.cfg.device)
        self.default_joint_pos = self.robot.default_joint_pos.to(self.cfg.device)
        self.action_scale = self.robot.action_scale.to(self.cfg.device)
        self._gravity = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32, device=self.cfg.device)
        self._real2sim_joint_indexes = torch.as_tensor(
            self.robot.data.real2sim_joint_indexes,
            dtype=torch.long,
            device=self.cfg.device,
        )
        self._sim2real_joint_indexes = torch.as_tensor(
            self.robot.data.sim2real_joint_indexes,
            dtype=torch.long,
            device=self.cfg.device,
        )

        onnx_path = resolve_project_path(self.cfg.onnx_model_path or self.cfg.checkpoint_path)
        self._session = _make_onnx_session(str(onnx_path))
        inputs = self._session.get_inputs()
        if len(inputs) != 2:
            raise ValueError(
                "EQD ONNX policy expects exactly two inputs: motion/encoder command and robot_state. "
                f"Got inputs: {[inp.name for inp in inputs]}"
            )
        self._output_name = self._session.get_outputs()[0].name
        self._output_names = [self._output_name]

        self._traj_body_index = 0
        self._obs_body_index = 1
        self._fk_axes: list[torch.Tensor] = []
        self._fk_joint_ids: list[int] = []
        if self.cfg.obs_anchor_name != self.cfg.traj_anchor_name and not self.cfg.obs_anchor_fk_chain:
            raise ValueError(
                "obs_anchor_fk_chain is required when obs_anchor_name differs from traj_anchor_name."
            )
        if self.cfg.obs_anchor_fk_chain:
            for joint_idx, axis in self.cfg.obs_anchor_fk_chain:
                self._fk_joint_ids.append(joint_idx)
                self._fk_axes.append(
                    torch.tensor(axis, dtype=torch.float32, device=self.cfg.device).unsqueeze(0)
                )

        self._command_window_mode = str(self.cfg.command_window_mode)
        self._step_future_count = int(self.cfg.command_num_future_frames)
        self._step_stride = int(self.cfg.command_frame_interval) + 1
        if self.cfg.command_input_mode != "motion" or self._command_window_mode != "future":
            raise ValueError("GmcRtPolicy first phase only supports motion/future command input.")
        if self._step_future_count < 1:
            raise ValueError("command_num_future_frames must be at least 1.")
        num_joints = self.cfg.num_joints or self.robot.num_joints
        command_width = 2 * num_joints
        if self.cfg.include_future_anchor_ori_in_command:
            command_width += self.cfg.anchor_ori_dim
        self._command_step_dim = self._step_future_count * command_width
        self._state_term_widths = state_term_widths(self.cfg, self.robot.num_joints)
        self._state_step_dim = sum(self._state_term_widths)
        self._command_history_length = int(self.cfg.command_history_length or self.cfg.history_length)
        self._state_history_length = int(self.cfg.state_history_length or self.cfg.history_length)
        self._history_length = max(self._command_history_length, self._state_history_length)
        self._motion_cmd_dim = self._command_step_dim * self._command_history_length
        self._robot_state_dim = self._state_step_dim * self._state_history_length
        self._motion_input_name, self._robot_state_input_name = _resolve_onnx_inputs(
            inputs, self._motion_cmd_dim, self._robot_state_dim
        )

        self._reference_receiver = RtReferenceReceiver(self.cfg)
        self._reference_consumer = RtReferenceConsumer[ReferenceFrame50HzNs](
            self.cfg.rt_reference_consumer,
            future_count=self._step_future_count,
            stride=self._step_stride,
        )
        self._reference_alignment = RtReferenceAlignment(
            enabled=bool(self.cfg.rt_align_reference_on_resume),
            align_xy=bool(self.cfg.rt_align_reference_xy),
            align_yaw=bool(self.cfg.rt_align_reference_yaw),
            device=self.cfg.device,
        )
        self._current_window: list[ReferenceFrame50HzNs] | None = None
        self._last_raw_reference_window: list[ReferenceFrame50HzNs] | None = None
        self._last_reference_was_held = False
        self._last_consumed_base_idx: int | None = None
        self._last_reference_advance_s: float | None = None
        self._standing_window: list[ReferenceFrame50HzNs] | None = None
        self._paused_window: list[ReferenceFrame50HzNs] | None = None
        self._reference_paused = False
        self._reference_transition_start: list[ReferenceFrame50HzNs] | None = None
        self._reference_transition_elapsed_s = 0.0
        self._reference_transition_waiting_for_activation = True
        self._inference_diagnostics = InferenceDiagnostics.from_environment()
        from booster_deploy.utils.input_health import InputHealth
        self._input_health = InputHealth()
        self._diag = reporter("deploy")
        self._diag.note(mode="zmq_raw", task="retarget")
        self._diag_holds = 0

    def reset(self) -> None:
        try:
            self._reset_reference()
        except BaseException as exc:
            self._telemetry_session_error=repr(exc)
            self._end_telemetry_session(repr(exc))
            raise

    def _reset_reference(self) -> None:

        from booster_deploy.utils.input_health import InputHealth
        self._input_health = InputHealth()
        if getattr(self,'_telemetry_session_open',False):
            self._end_telemetry_session('reset')
        self._diag.start_policy_session()
        self._telemetry_session_open=True
        self._telemetry_session_error=None
        self.init_root_yaw_quat_w_inv = lab_math.quat_inv(
            lab_math.yaw_quat(self.robot.data.root_quat_w)
        )
        self.current_frame = 0
        self.last_action = torch.zeros(
            self.cfg.num_actions or self.robot.num_joints,
            dtype=torch.float32,
            device=self.cfg.device,
        )
        self._hist_buf = torch.zeros(
            self._history_length,
            self._command_step_dim + self._state_step_dim,
            dtype=torch.float32,
            device=self.cfg.device,
        )
        self._hist_buf_filled = False
        self._exit_reference_token = None
        self._exit_reference_input_ready = False
        self._current_window = None
        self._last_raw_reference_window = None
        self._last_reference_was_held = False
        self._last_consumed_base_idx = None
        self._last_reference_advance_s = None
        self._standing_window = None
        self._paused_window = None
        self._reference_paused = False
        self._reference_transition_start = None
        self._reference_transition_elapsed_s = 0.0
        self._reference_transition_waiting_for_activation = True
        self._inference_diagnostics.reset_continuity()
        self._reference_consumer.reset()
        self._reference_alignment.reset()
        from booster_deploy.utils.offline_entry import EntryHandshake
        from booster_deploy.utils.replay_control import _write_token
        import os
        cycle = int(getattr(getattr(self.controller, 'portal', None), 'offline_entry_cycle', 0))
        self._entry = EntryHandshake(cycle) if cycle else None
        from booster_deploy.utils.offline_entry import record_startup
        record_startup(self,cycle)
        self._entry_final_input = False
        if cycle:
            self._reference_receiver.offline_cycle = cycle
            if not _write_token(os.getenv('BOOSTER_DIAG_DIR'), 'replay.ready', cycle):
                raise RuntimeError('offline consumer readiness could not be published')
        self._reference_receiver.start()
        def initial_ready():
            if cycle:
                first = getattr(self._reference_receiver, 'first_sample_idx', None)
                if first is None:
                    return False
                self._reference_consumer.start_at(first)
            return self._reference_consumer.initial_window_available(self._reference_receiver)
        min_samples = self._startup_min_samples()
        try:
            self._reference_receiver.wait_until_ready(
                min_samples,
                float(self.cfg.rt_reference_timeout_s),
                is_ready=initial_ready,
            )
        except BaseException as exc:
            self._telemetry_session_error=repr(exc)
            self._end_telemetry_session(repr(exc))
            if isinstance(exc,KeyboardInterrupt):
                self._reference_receiver.close()
            raise
        if cycle:
            self._entry.prepared(self._reference_consumer.consume(self._reference_receiver))
            self._reference_consumer.start_at(self._reference_receiver.first_sample_idx)
            _write_token(os.getenv("BOOSTER_DIAG_DIR"), "replay.prepared", cycle)
            self._diag.event('offline_entry_prepared',dict(cycle=cycle,source='received_reference_prefix',
                input_transition_s=float(self.cfg.rt_reference_transition_s)))
        self._diag.event('policy_seed', dict(tick=0,history='zero_unfilled',last_action='zero_initialized_by_existing_reset'))

    # End each real session once, including reference transport failures.
    def _end_telemetry_session(self,reason):
        if getattr(self,'_telemetry_session_open',False):
            self._telemetry_session_open=False
            self._diag.event('policy_session_end',dict(reason=reason))

    def close(self) -> None:

        try:
            self._reference_receiver.close()
        except BaseException as exc:
            self._telemetry_session_error=repr(exc)
            raise
        finally:
            self._end_telemetry_session(getattr(self,'_telemetry_session_error',None) or 'closed')
            self._diag.emit(
                "closed",
                extra={"state": {"exit": "closed"}},
                force=True,
            )
            self._inference_diagnostics.close()
            self._diag.close()


    def handle_control_events(self) -> None:

        return

    def toggle_reference_pause(self) -> bool:

        if self._standing_window is not None:
            return False
        self._reference_paused = not self._reference_paused
        if self._reference_paused:
            self._paused_window = list(self._current_window or self._current_reference_window())
            self._reference_transition_start = None
            self._reference_transition_elapsed_s = 0.0
            print("Realtime RT reference paused")
        else:
            self._reference_consumer.request_rebase_to_latest()
            self._inference_diagnostics.reset_continuity()
            self._reference_alignment.request_reanchor()
            self._reference_transition_start = list(self._paused_window or self._current_window or []) or None
            self._reference_transition_elapsed_s = 0.0
            self._paused_window = None
            print("Realtime RT reference resumed")
        if hasattr(self, '_diag'):
            self._diag.event('reference_pause_control', dict(paused=self._reference_paused))
        return self._reference_paused

    def begin_reference_exit(self, token: str) -> None:
        """Transition the last actual model input to default; leave output live."""
        if not self._current_window:
            raise RuntimeError('RT exit has no actual model window')
        self._exit_reference_start = list(self._current_window)
        self._exit_reference_target = self._make_constant_reference_window(
            root_pos=self.robot.data.root_pos_w.to(self.cfg.device, dtype=torch.float32),
            root_quat=lab_math.yaw_quat(self._robot_motion_frame_quat()),
            joint_pos=self.default_joint_pos.to(self.cfg.device, dtype=torch.float32))
        self._exit_reference_token = token
        self._exit_reference_elapsed_s = 0.0
        self._exit_reference_input_ready = False

    def enter_standing_reference(self) -> None:

        self._reference_alignment.reset()
        self._reference_paused = False
        self._paused_window = None
        self._reference_transition_start = None
        self._reference_transition_elapsed_s = 0.0
        self._reference_transition_waiting_for_activation = False
        self._standing_window = self._make_constant_reference_window(
            root_pos=self.robot.data.root_pos_w.to(self.cfg.device, dtype=torch.float32),
            root_quat=lab_math.yaw_quat(self._robot_motion_frame_quat()),
            joint_pos=self.default_joint_pos.to(self.cfg.device, dtype=torch.float32),
        )
        self._current_window = None

    def enter_tracking_reference(self) -> None:

        self._standing_window = None
        self._paused_window = None
        self._reference_paused = False
        self.reset()
        self._reference_alignment.request_reanchor()

    def _make_constant_reference_window(
        self,
        *,
        root_pos: torch.Tensor,
        root_quat: torch.Tensor,
        joint_pos: torch.Tensor,
    ) -> list[ReferenceFrame50HzNs]:

        now_ns = time.monotonic_ns()
        latest = self._reference_receiver.latest_ready_idx()
        base_idx = int(latest) if latest is not None else 0
        sample_dt_ns = int(round(float(getattr(self.controller.cfg, "policy_dt", 0.02)) * 1_000_000_000))
        root_pos_np = root_pos.detach().cpu().numpy().astype(np.float64, copy=True)
        root_quat_np = root_quat.detach().cpu().numpy().astype(np.float64, copy=True)
        joint_pos_np = joint_pos.detach().cpu().numpy().astype(np.float64, copy=True)
        joint_vel_np = np.zeros_like(joint_pos_np, dtype=np.float64)
        robot_name = str(getattr(self.robot.cfg, "name", ""))
        result = [
            ReferenceFrame50HzNs(
                sample_idx=base_idx + i,
                sample_ns=now_ns + i * sample_dt_ns,
                body_seq=-1,
                body_recv_ns=now_ns,
                retarget_done_ns=now_ns,
                robot_name=robot_name,
                joint_pos=joint_pos_np.copy(),
                joint_vel=joint_vel_np.copy(),
                root_pos=root_pos_np.copy(),
                root_quat=root_quat_np.copy(),
            )
            for i in range(self._step_future_count)
        ]
        from booster_deploy.telemetry.postprocess import reference_window
        return reference_window(result,'standing_config',dict(recv_id=getattr(self.controller,'state_recv_id',None),
            recv_ns=getattr(self.controller,'state_recv_ns',None)))

    def _startup_min_samples(self) -> int:

        if self.cfg.rt_startup_min_samples is not None:
            return int(self.cfg.rt_startup_min_samples)
        return self._reference_consumer.minimum_startup_samples

    def _rt_joint_to_policy_order(self, joint: torch.Tensor) -> torch.Tensor:

        return joint[..., self._real2sim_joint_indexes]

    def _robot_motion_frame_quat(self) -> torch.Tensor:

        return lab_math.quat_mul(
            self.init_root_yaw_quat_w_inv.to(self.cfg.device, dtype=torch.float32),
            self.robot.data.root_quat_w.to(self.cfg.device, dtype=torch.float32),
        )

    def _set_command(self) -> None:

        frames = self._current_reference_window()
        base = frames[0]
        set_hand_frame = getattr(self.controller, "set_hand_retarget_frame", None)
        if callable(set_hand_frame):
            set_hand_frame(base.hand_frame)
        joint_pos_qpos = torch.as_tensor(base.joint_pos, dtype=torch.float32, device=self.cfg.device)
        joint_vel_qpos = torch.as_tensor(base.joint_vel, dtype=torch.float32, device=self.cfg.device)
        self.cmd_dof_pos = self._rt_joint_to_policy_order(joint_pos_qpos)
        self.cmd_dof_vel = self._rt_joint_to_policy_order(joint_vel_qpos)
        self.cmd_traj_root_pos_w = torch.as_tensor(base.root_pos, dtype=torch.float32, device=self.cfg.device)
        self.cmd_traj_root_quat_w = torch.as_tensor(base.root_quat, dtype=torch.float32, device=self.cfg.device)
        self.cmd_obs_root_pos_w = self.cmd_traj_root_pos_w
        self.cmd_obs_root_quat_w = self._compute_reference_obs_anchor_quat(
            self.cmd_traj_root_quat_w,
            self.cmd_dof_pos,
        )
        self._current_window = frames

    def _current_reference_window(self) -> list[ReferenceFrame50HzNs]:

        if getattr(self, '_exit_reference_token', None):
            frames, finished = blend_reference_windows(self._exit_reference_start, self._exit_reference_target,
                elapsed_s=self._exit_reference_elapsed_s, duration_s=float(self.cfg.rt_reference_transition_s))
            self._exit_reference_elapsed_s += float(getattr(self.controller.cfg, 'policy_dt', .02))
            self._exit_reference_input_ready = finished
            self._last_raw_reference_window = list(frames)
            self._last_reference_was_held = False
            return frames
        if self._standing_window is not None:
            self._last_raw_reference_window = list(self._standing_window)
            self._last_reference_was_held = False
            return self._standing_window
        if self._reference_paused and self._paused_window is not None:
            self._last_raw_reference_window = list(self._paused_window)
            self._last_reference_was_held = True
            return self._paused_window

        raw_frames = self._raw_reference_window()
        if raw_frames is None:
            raise RuntimeError("RT reference pool is empty")
        if self._reference_alignment.needs_anchor:
            self._reference_alignment.align_to_robot(
                robot_root_pos_w=self.robot.data.root_pos_w,
                robot_root_quat_w=self._robot_motion_frame_quat(),
                reference_window=raw_frames,
            )
        frames = self._reference_alignment.apply(raw_frames)
        frames = self._apply_reference_transition(frames)
        self._last_raw_reference_window = list(raw_frames)
        return frames

    @property
    def entry_prepared(self):
        return getattr(self, '_entry', None) is not None and self._entry.window is not None

    @property
    def entry_complete(self):
        return self.entry_prepared and self._tracking_transition_is_active() and self._entry_final_input

    def _entry_playing(self):
        event = getattr(getattr(self.controller, 'portal', None), 'offline_playback_active', None)
        return event is None or event.is_set()

    def _tracking_transition_is_active(self) -> bool:
        portal = getattr(self.controller, "portal", None)
        event = getattr(portal, "reference_tracking_active", None)
        return True if event is None else bool(event.is_set())

    def _make_activation_hold(
        self, frames: list[ReferenceFrame50HzNs]
    ) -> list[ReferenceFrame50HzNs]:
        joint_pos = np.asarray(self.robot.data.joint_pos.detach().cpu().numpy(), dtype=np.float64).copy()
        joint_vel = np.zeros_like(joint_pos)
        root_pos = np.asarray(frames[0].root_pos, dtype=np.float64).copy()
        root_pos[:2] = self.robot.data.root_pos_w.detach().cpu().numpy()[:2]
        root_quat = np.asarray(self._robot_motion_frame_quat().detach().cpu().numpy(), dtype=np.float64).copy()
        result = [
            replace(
                frame,
                joint_pos=joint_pos.copy(),
                joint_vel=joint_vel.copy(),
                root_pos=root_pos.copy(),
                root_quat=root_quat.copy(),
            )
            for frame in frames
        ]
        from booster_deploy.telemetry.postprocess import reference_window
        return reference_window(result,'activation_hold',dict(recv_id=getattr(self.controller,'state_recv_id',None),
            recv_ns=getattr(self.controller,'state_recv_ns',None)),live_parent=frames[0])

    def _apply_reference_transition(
        self, frames: list[ReferenceFrame50HzNs]
    ) -> list[ReferenceFrame50HzNs]:
        duration = float(self.cfg.rt_reference_transition_s)
        self._entry_final_input = False
        if self._reference_transition_waiting_for_activation:
            if not self._tracking_transition_is_active():
                held = self._make_activation_hold(frames)
                self._reference_transition_start = held
                return held
            self._reference_transition_waiting_for_activation = False
            self._reference_transition_start = self._make_activation_hold(frames)
            self._reference_transition_elapsed_s = 0.0
        if self._reference_transition_start is None:
            self._entry_final_input = True
            return frames
        if duration <= 0.0:
            self._reference_transition_start = None
            self._entry_final_input = True
            return frames
        blended, finished = blend_reference_windows(
            self._reference_transition_start,
            frames,
            elapsed_s=self._reference_transition_elapsed_s,
            duration_s=duration,
        )
        self._reference_transition_elapsed_s += float(getattr(self.controller.cfg, "policy_dt", 0.02))
        if finished:
            self._reference_transition_start = None
            self._entry_final_input = True
        return blended



    def _raw_reference_window(self) -> list[ReferenceFrame50HzNs] | None:

        if self.entry_prepared and not self._entry_playing():
            self._last_reference_was_held = True
            return list(self._entry.window)
        previous_base_idx = getattr(self._reference_consumer, "consume_base_idx", None)
        frames = self._reference_consumer.consume(self._reference_receiver)
        if frames is None and self.entry_prepared:
            self._last_reference_was_held = True
            return list(self._entry.window)
        hold_state = getattr(self._reference_consumer, "last_was_hold", None)
        current_base_idx = getattr(self._reference_consumer, "consume_base_idx", None)
        if hold_state is None:
            hold_state = (
                previous_base_idx is not None
                and current_base_idx == previous_base_idx
            )
        self._last_reference_was_held = bool(hold_state)
        if current_base_idx is not None and current_base_idx != self._last_consumed_base_idx:
            self._last_consumed_base_idx = int(current_base_idx)
            self._last_reference_advance_s = time.monotonic()
        return frames

    def reference_finished(self) -> bool:
        """Offline replay: no new reference window arrived because the clip ended."""

        if getattr(self, '_exit_reference_token', None):
            return False
        if not offline_replay_enabled():
            return False
        if self._reference_paused:
            return False
        last_s = self._last_reference_advance_s
        if last_s is None:
            return False
        return (time.monotonic() - last_s) > offline_replay_end_timeout_s()

    def _record_diag(self, onnx_elapsed_ns: int, window: list[ReferenceFrame50HzNs] | None, *, target=None) -> None:
        """Human line: inference rate + end-to-end age; health alerts only."""

        diag = self._diag
        now_ns = time.monotonic_ns()
        sources = tuple(dict.fromkeys(s for frame in (window or []) for s in frame.source_ids))
        health_state = ("walking" if not self._tracking_transition_is_active() else
                        "paused" if self._reference_paused else
                        "held" if self._last_reference_was_held else
                        "warmup" if self._standing_window is not None else "active")
        progress = self._input_health.observe(sources, now_ns, state=health_state)

        # Sampling must happen on every tick; only the record is throttled.
        diag.mark("tick", now_ns)
        diag.count("infer", now_ns)
        diag.sample("infer", onnx_elapsed_ns / 1e6, now_ns)
        if self._last_reference_was_held:
            self._diag_holds += 1
        if window and window[-1].clock_domain == diag.clock_domain:
            newest = window[-1]
            e2e_ms = max((now_ns - int(newest.body_recv_ns)) / 1e6, 0.0)
            diag.sample("e2e", e2e_ms, now_ns)
            sample_ns = getattr(newest, "sample_ns", None)
            if sample_ns:
                grid_ms = max((now_ns - int(sample_ns)) / 1e6, 0.0)
                diag.sample("grid_lag", max(grid_ms - e2e_ms, 0.0), now_ns)

        if not diag.due(now_ns):
            return

        tick_window = diag.interval("tick")
        gap = tick_window.gap_stats(now_ns)
        stall = tick_window.stall_ms(now_ns)
        infer_hz = diag.rate("infer").hz(now_ns)
        e2e = diag.latency("e2e", now_ns)
        infer = diag.latency("infer", now_ns)
        alerts: list[str] = []
        if below_floor(infer_hz):
            alerts.append(f"infer {infer_hz:.1f} Hz < 50")
        if stall is not None and stall > 200.0:
            alerts.append(f"stall {stall:.0f} ms")
        if infer.count and infer.p95 > 15.0:
            alerts.append(f"infer_p95 {infer.p95:.1f} ms")
        line = diag.emit(
            f"infer {infer_hz:5.1f} Hz   e2e {format_lat(e2e)} ms",
            alerts=alerts,
            extra={"state": {
                "infer_p50_ms": round(infer.p50, 2),
                "infer_p95_ms": round(infer.p95, 2),
                "grid_lag_p95_ms": round(diag.latency("grid_lag", now_ns).p95, 1),
                "gap_p95_ms": round(gap.p95, 1),
                "held": bool(self._last_reference_was_held),
                "input_progress": self._input_health.facts(),
            }},
            now_ns=now_ns,
        )
        if line is not None:
            self._diag_holds = 0

    def _compute_obs_anchor_quat(
        self, traj_anchor_q: torch.Tensor, joint_pos_sim: torch.Tensor
    ) -> torch.Tensor:
        q = traj_anchor_q.unsqueeze(0)
        for idx, axis in zip(self._fk_joint_ids, self._fk_axes):
            angle = joint_pos_sim[idx].reshape(1)
            q_joint = lab_math.quat_from_angle_axis(angle, axis)
            q = lab_math.quat_mul(q, q_joint)
        return q.squeeze(0)

    def _compute_reference_obs_anchor_quat(
        self,
        traj_anchor_q: torch.Tensor,
        joint_pos_sim: torch.Tensor,
    ) -> torch.Tensor:

        if not self._fk_axes:
            return traj_anchor_q
        return self._compute_obs_anchor_quat(traj_anchor_q, joint_pos_sim)

    def _compute_future_command_row(self, robot_obs_quat_w: torch.Tensor) -> torch.Tensor:

        frames = getattr(self, "_current_window", None)
        if frames is None:
            frames = self._current_reference_window()
        joint_pos = torch.stack(
            [torch.as_tensor(frame.joint_pos, dtype=torch.float32, device=self.cfg.device) for frame in frames],
            dim=0,
        )
        joint_vel = torch.stack(
            [torch.as_tensor(frame.joint_vel, dtype=torch.float32, device=self.cfg.device) for frame in frames],
            dim=0,
        )
        joint_pos = self._rt_joint_to_policy_order(joint_pos)
        joint_vel = self._rt_joint_to_policy_order(joint_vel)
        command = torch.cat([joint_pos, joint_vel], dim=-1).reshape(-1)
        if self.cfg.include_future_anchor_ori_in_command:
            ref_pos = torch.stack(
                [torch.as_tensor(frame.root_pos, dtype=torch.float32, device=self.cfg.device) for frame in frames],
                dim=0,
            )
            ref_quat = torch.stack(
                [
                    self._compute_reference_obs_anchor_quat(
                        torch.as_tensor(frame.root_quat, dtype=torch.float32, device=self.cfg.device),
                        joint_pos[i],
                    )
                    for i, frame in enumerate(frames)
                ],
                dim=0,
            )
            _, ori = lab_math.subtract_frame_transforms(
                self.robot.data.root_pos_w.unsqueeze(0).repeat(self._step_future_count, 1),
                robot_obs_quat_w.unsqueeze(0).repeat(self._step_future_count, 1),
                ref_pos,
                ref_quat,
            )
            ori_mf = lab_math.matrix_from_quat(ori)[..., :2].reshape(-1)
            return torch.cat([command, ori_mf], dim=-1)
        return command

    def compute_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._set_command()
        real2sim_map = self.robot.data.real2sim_joint_indexes
        dof_pos = self.robot.data.joint_pos[real2sim_map]
        joint_pos = dof_pos - self.default_joint_pos[real2sim_map]
        joint_vel = self.robot.data.joint_vel[real2sim_map]

        cur_traj_root_quat_w = lab_math.quat_mul(
            self.init_root_yaw_quat_w_inv, self.robot.data.root_quat_w
        )
        robot_obs_quat_w = self._compute_reference_obs_anchor_quat(cur_traj_root_quat_w, dof_pos)
        projected_gravity = lab_math.quat_apply_inverse(self.robot.data.root_quat_w, self._gravity)
        command = self._compute_future_command_row(robot_obs_quat_w)
        state = state_row(
            self.cfg, projected_gravity, self.robot.data.root_ang_vel_b,
            joint_pos, joint_vel, self.last_action,
        )
        history_row = torch.cat((command, state), dim=-1)
        if not self._hist_buf_filled:
            self._hist_buf[:] = history_row.unsqueeze(0)
            self._hist_buf_filled = True
        else:
            self._hist_buf[:-1] = self._hist_buf[1:].clone()
            self._hist_buf[-1] = history_row
        return flatten_history(
            self._hist_buf, self._command_step_dim, self._command_history_length,
            self._state_history_length, self._state_term_widths,
        )

    def inference(self) -> torch.Tensor:

        motion_cmd, robot_state = self.compute_inputs()
        feeds = {
            self._motion_input_name: motion_cmd.detach().cpu().numpy().astype(np.float32, copy=False),
            self._robot_state_input_name: robot_state.detach().cpu().numpy().astype(np.float32, copy=False),
        }
        from booster_deploy.telemetry.encoder import Encoder
        if not hasattr(self,'_encoder_telemetry'): self._encoder_telemetry=Encoder()
        window=self._current_window or []
        sources=tuple(dict.fromkeys(s for frame in window for s in frame.source_ids))
        tick=self._encoder_telemetry.begin(self.controller,feeds,sources,
            window=[{'sample_idx':f.sample_idx,'identity':f.identity} for f in window],
            source_channel='retarget_input',layout={'motion_cmd':'batch,future,joint_pos_joint_vel_anchor','robot_state':'batch,history,proprioception','joint_order':'policy_sim'})
        onnx_start_ns = time.monotonic_ns()
        out = self._session.run(self._output_names, feeds)
        onnx_elapsed_ns = time.monotonic_ns() - onnx_start_ns
        current_window = self._current_window
        consumed_base_idx = (
            int(current_window[0].sample_idx) if current_window else None
        )
        self._inference_diagnostics.record_success(
            consumed_base_idx,
            track_continuity=(
                self._standing_window is None and not self._reference_paused
            ),
        )
        action_raw_np = np.asarray(out[0], dtype=np.float32).reshape(-1)
        action = torch.as_tensor(action_raw_np, dtype=torch.float32, device=self.cfg.device)
        if self.cfg.clip_actions is not None:
            action = action.clip(-self.cfg.clip_actions, self.cfg.clip_actions)

        if hasattr(self.controller, "set_reference_qpos"):
            base = getattr(self, "_current_window", None)
            if base:
                frame = base[0]
                joint_pos = torch.as_tensor(frame.joint_pos, dtype=torch.float32, device=self.cfg.device)
                ref_qpos = torch.cat(
                    [
                        torch.as_tensor(frame.root_pos, dtype=torch.float32, device=self.cfg.device),
                        torch.as_tensor(frame.root_quat, dtype=torch.float32, device=self.cfg.device),
                        joint_pos,
                    ],
                    dim=0,
                )
                self.controller.set_reference_qpos(ref_qpos)

        self.current_frame += 1
        self.last_action = action

        action_real = action[self._sim2real_joint_indexes]
        offset = self.cmd_dof_pos[self._sim2real_joint_indexes]
        target_joint_pos = action_real * self.action_scale + offset
        self._encoder_telemetry.result(tick,out[0],action,target_joint_pos)
        self._record_diag(onnx_elapsed_ns, current_window, target=target_joint_pos)
        return target_joint_pos



GmcRtPolicyCfg.constructor = GmcRtPolicy
