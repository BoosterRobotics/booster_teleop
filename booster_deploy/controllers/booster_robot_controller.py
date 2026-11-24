
from __future__ import annotations
from dataclasses import MISSING
import json
import logging
import math
import os
import signal
import time
import threading
import multiprocessing as mp
from multiprocessing import synchronize
from enum import Enum
from pathlib import Path

import numpy as np
import torch

import rclpy
from rclpy.executors import SingleThreadedExecutor, ExternalShutdownException
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from rclpy.signals import SignalHandlerOptions
from booster_interface.msg import (
    BoosterApiReqMsg,
    LowState,
    LowCmd,
    MotorCmd,
    RemoteControllerState,
)
from booster_interface.srv import RpcService
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray


from .controller_cfg import ControllerCfg
from .base_controller import BaseController, BoosterRobot
from ..utils.diag_report import below_floor, format_gap, format_lat, reporter

#: A ctrl tick later than ``policy_dt + LATE_TICK_MS`` counts as late, and only
#: ``LATE_TICKS_PER_S`` such ticks in one second raise an alert.
LATE_TICK_MS = 8.0
LATE_TICKS_PER_S = 5
from .hand_control import hand_control_active_for_state
from ..hand_retargeting.real_hand_mapping import (
    DEFAULT_T2_REVO2_ROS_MAPPING_PATH,
    REVO2_POSITION_COMMAND_MAX_RAD,
    REVO2_POSITION_COMMAND_SCALE,
    RealHandCommandMapping,
    load_real_hand_command_mapping,
)
from ..hand_retargeting.stall_recovery import (
    HandStallRecovery,
    HandStallRecoveryConfig,
)
from ..utils.synced_array import SyncedArray
from ..telemetry.state_ipc import AtomicLowStateArray, PAYLOAD_CAPACITY, pack_message, message_fields
from ..utils.metrics import SyncedMetrics
from ..utils.periodic_deadline import PeriodicDeadline
from ..utils.isaaclab import math as lab_math
from ..utils.remote_control_service import (
    ControlEventType,
    PicoWalkingVelocityFilter,
    RemoteControlService,
)

_LOC_API_CHANGE_MODE = 2000
_LOC_API_MOVE = 2001
_LOC_API_GET_STATUS = 2018

class _RobotModeInt:
    kWalking = 2
    kCustom = 3


class BoosterControlState(str, Enum):
    WALKING = "walking"
    ARMING_PVT = "arming_pvt"
    PVT_HOLD = "pvt_hold"
    ARMING_TRACKING = "arming_tracking"
    TRACKING = "tracking"
    TRACKING_PAUSED = "tracking_paused"
    EXITING_TO_WALK = "exiting_to_walk"
    FAULT = "fault"


logger = logging.getLogger("booster_deploy")
logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")


_HAND_JOINT_SUFFIXES = (
    "thumb_proximal_joint",
    "thumb_metacarpal_joint",
    "index_proximal_joint",
    "middle_proximal_joint",
    "ring_proximal_joint",
    "pinky_proximal_joint",
)
_HAND_COMMAND_MIN = 0.0
_HAND_COMMAND_MAX = REVO2_POSITION_COMMAND_MAX_RAD
_WALKING_HAND_CLOSE_DURATION_S = 1.0


def _env_bool(name: str, default: bool = False) -> bool:

    value = os.getenv(name)
    if value is None or value == "":
        return bool(default)
    return value.strip().lower() in ("1", "true", "yes", "on")


def _env_float(name: str, default: float) -> float:

    value = os.getenv(name)
    if value is None or value == "":
        return float(default)
    return float(value)


def _env_float_alias(name: str, legacy_name: str, default: float) -> float:

    if os.getenv(name) not in (None, ""):
        return _env_float(name, default)
    return _env_float(legacy_name, default)


def _limit_joint_position_by_effort(
    requested_pos: np.ndarray | list[float],
    measured_pos: np.ndarray | list[float],
    stiffness: np.ndarray | list[float],
    effort_limit: np.ndarray | list[float],
) -> np.ndarray:

    requested = np.asarray(requested_pos, dtype=np.float64)
    measured = np.asarray(measured_pos, dtype=np.float64)
    kp = np.asarray(stiffness, dtype=np.float64)
    limit = np.asarray(effort_limit, dtype=np.float64)
    if not (requested.shape == measured.shape == kp.shape == limit.shape):
        raise ValueError(
            "Booster joint command arrays must have identical shapes, got "
            f"requested={requested.shape}, measured={measured.shape}, "
            f"stiffness={kp.shape}, effort_limit={limit.shape}"
        )
    if requested.ndim != 1:
        raise ValueError(
            f"Booster joint command arrays must be one-dimensional, got {requested.shape}"
        )
    if not (
        np.isfinite(requested).all()
        and np.isfinite(measured).all()
        and np.isfinite(kp).all()
        and np.isfinite(limit).all()
    ):
        raise ValueError("Booster joint command arrays must contain only finite values")
    if np.any(kp <= 0.0):
        raise ValueError("Booster joint stiffness must be positive")
    if np.any(limit < 0.0):
        raise ValueError("Booster joint effort limits must be non-negative")

    position_range = limit / kp
    return np.clip(
        requested,
        measured - position_range,
        measured + position_range,
    )


def _parse_float_list_env(
    name: str,
    default: list[float] | tuple[float, ...],
    *,
    expected_len: int | None = None,
) -> list[float]:

    value = os.getenv(name)
    if value is None or value.strip() == "":
        result = [float(item) for item in default]
    else:
        result = [float(item.strip()) for item in value.split(",") if item.strip()]
    if expected_len is not None and len(result) != int(expected_len):
        raise ValueError(f"{name} must contain {expected_len} comma-separated values, got {len(result)}")
    return result


def _parse_string_list_env(
    name: str,
    default: list[str] | tuple[str, ...],
    *,
    expected_len: int | None = None,
) -> list[str]:

    value = os.getenv(name)
    if value is None or value.strip() == "":
        result = [str(item) for item in default]
    else:
        result = [item.strip() for item in value.split(",") if item.strip()]
    if expected_len is not None and len(result) != int(expected_len):
        raise ValueError(f"{name} must contain {expected_len} comma-separated values, got {len(result)}")
    return result


def _validate_hand_command_values(
    values: list[float] | tuple[float, ...] | np.ndarray,
    field: str,
) -> list[float]:

    result = [float(item) for item in values]
    if not np.isfinite(result).all():
        raise ValueError(f"{field} must contain only finite values")
    if any(value < _HAND_COMMAND_MIN or value > _HAND_COMMAND_MAX for value in result):
        raise ValueError(
            f"{field} values must be in [{_HAND_COMMAND_MIN:g}, {_HAND_COMMAND_MAX:g}], got {result}"
        )
    return result


def _make_hand_joint_names(prefix: str) -> list[str]:

    return [f"{prefix}_{suffix}" for suffix in _HAND_JOINT_SUFFIXES]


def _hand_position_from_trigger(
    trigger: float,
    open_position: list[float] | tuple[float, ...],
    closed_position: list[float] | tuple[float, ...],
) -> list[float]:

    ratio = min(max(float(trigger), 0.0), 1.0)
    return [
        float(open_i) + ratio * (float(closed_i) - float(open_i))
        for open_i, closed_i in zip(open_position, closed_position)
    ]


def _hand_four_finger_closed_position(
    names: list[str] | tuple[str, ...],
    open_position: list[float] | tuple[float, ...],
    closed_position: list[float] | tuple[float, ...],
) -> list[float]:

    return [
        float(open_i) if "thumb" in str(name).lower() else float(closed_i)
        for name, open_i, closed_i in zip(names, open_position, closed_position)
    ]


class CountTimer:

    def __init__(self, dt: float = 0.002, use_sim_time: bool = False):

        self.dt = dt

        self.counter = mp.Value('L', 0)
        self.use_sim_time = use_sim_time

    def tick_timer_if_sim(self):

        if self.use_sim_time:
            with self.counter.get_lock():
                self.counter.value += 1

    def get_time(self):

        if self.use_sim_time:
            with self.counter.get_lock():
                return self.counter.value * self.dt
        else:
            return time.perf_counter()


class BoosterRobotPortal:

    synced_state: SyncedArray
    synced_command: SyncedArray
    synced_action: SyncedArray
    exit_event: synchronize.Event

    def __init__(self, cfg: ControllerCfg, use_sim_time: bool = False) -> None:

        self.cfg = cfg

        self.robot = BoosterRobot(cfg.robot)
        self._joint_effort_limit = np.asarray(
            cfg.robot.effort_limit, dtype=np.float64
        ).copy()

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        self.remoteControlService = RemoteControlService()
        # Build gate: compare the tree that is running with what was last synced.
        # "Local fixed, board not synced" has already shipped a broken deploy once.
        self._pico_walking_velocity_filter = PicoWalkingVelocityFilter(
            self.remoteControlService.pico_walking_config
        )
        self._next_pico_walking_update_s = 0.0
        self._pico_walking_zero_pending = False
        self._pico_walking_suppressed_by_robot_joystick = False


        self.exit_event = mp.Event()
        self.policy_stop_event = mp.Event()
        self.reference_tracking_active = mp.Event()
        # Set by the inference process when a replayed clip runs out (offline only).
        self.offline_replay_finished = mp.Event()
        # Robot-side control diagnostics: targets handed to the robot per second.
        self._diag_ctrl = reporter("ctrl")
        self._diag_last_publish_ns: int | None = None
        self._diag_late = 0
        self.remoteControlService.shutdown_callback = self._request_normal_shutdown
        self.policy_command_ready = mp.Event()
        self.policy_action_updated = mp.Event()
        self.policy_output_published = threading.Event()
        self.is_running = True
        self._output_lock = threading.Lock()
        self._output_mode = "none"
        self._hold_joint_pos = np.asarray(
            self.robot.cfg.default_joint_pos, dtype=np.float64
        ).copy()
        self._hold_stiffness = np.asarray(
            self.robot.cfg.joint_stiffness, dtype=np.float64
        ).copy()
        self._hold_damping = np.asarray(
            self.robot.cfg.joint_damping, dtype=np.float64
        ).copy()
        self._last_low_state_time = 0.0
        self.control_state = BoosterControlState.WALKING
        self._init_hand_trigger_config()
        self.timer = CountTimer(
            self.cfg.booster.low_state_dt, use_sim_time=use_sim_time)

        def signal_handler(sig, frame):

            if mp.current_process().name == "MainProcess":
                print("\nKeyboard interrupt: requesting fresh actual mode 0/1/2 shutdown (repeat to force)")
            if getattr(self, "_shutdown_requested", False):
                self.logger.error("Forced shutdown; actual safe mode may NOT be confirmed")
                self.exit_event.set()
            else:
                self._request_normal_shutdown()

        self._init_synced_buffer()
        self._init_metrics()

        self._cleanup_done = False
        self.inference_process = None
        self.low_cmd_publisher: rclpy.publisher.Publisher = None
        self.left_hand_cmd_publisher: rclpy.publisher.Publisher | None = None
        self.right_hand_cmd_publisher: rclpy.publisher.Publisher | None = None
        self.low_state_thread = None
        self.policy_output_thread = None
        self.low_cmd_process: mp.Process | None = None
        self.rpc_service_client: rclpy.client.Client | None = None

        rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
        signal.signal(signal.SIGINT, signal_handler)

        self._init_communication()

    def _init_synced_buffer(self):

        action_dtype = np.dtype(
            [
                ("policy_session", "S32"),
                ("policy_tick", np.int64),
                ("reference_exit_token", "S32"),
                ("reference_entry_cycle", np.int64),
                ("reference_entry_ready", np.bool_),
                ("telemetry_tick", "S96"),
                ("consumed_state_recv_id", np.int64),
                ("dof_target", float, (self.robot.num_joints,)),
                ("stiffness", float, (self.robot.num_joints,)),
                ("damping", float, (self.robot.num_joints,)),
            ]
        )
        self.synced_action = SyncedArray(
            "action",
            shape=(1,),
            dtype=action_dtype,
        )
        self._action_buf = np.zeros((1,), dtype=action_dtype)

        state_dtype = np.dtype(
            [
                ("state_recv_id", np.int64),
                ("state_recv_ns", np.int64),
                ("low_state_size", np.int64),
                ("low_state_payload", f"V{PAYLOAD_CAPACITY}"),
                ("root_rpy_w", float, (3,)),
                ("root_ang_vel_b", float, (3,)),
                ("root_pos_w", float, (3,)),
                ("root_lin_vel_w", float, (3,)),
                ("joint_pos", float, (self.robot.num_joints,)),
                ("joint_vel", float, (self.robot.num_joints,)),
                ("feedback_torque", float, (self.robot.num_joints,)),
            ]
        )
        self.synced_state = AtomicLowStateArray(
            "state",
            shape=(1,),
            dtype=state_dtype
        )
        self._state_buf = np.zeros((1,), dtype=state_dtype)

        command_dtype = np.dtype(
            [
                ("vx", float),
                ("vy", float),
                ("vyaw", float),


                ("pause_toggle", int),
                ("reference_exit_token", "S32"),
                ("reference_exit_session", "S32"),
            ]
        )
        self.synced_command = SyncedArray(
            "command",
            shape=(1,),
            dtype=command_dtype,
        )
        self._pause_toggle_seq = 0

    def _init_metrics(self):

        max_events = self.cfg.booster.metrics_max_events
        self.metrics = {
            "low_state_handler": SyncedMetrics(
                "low_state_handler", max_events=max_events
            ),
            "policy_step": SyncedMetrics(
                "policy_step", max_events=max_events
            ),
        }

    def _init_hand_trigger_config(self) -> None:

        hand_control_enabled = _env_bool("BOOSTER_HAND_CONTROL_ENABLED", True)
        self.hand_control_enabled = hand_control_enabled
        configured_hand_control_mode = os.getenv("BOOSTER_HAND_CONTROL_MODE", "").strip().lower()
        self.hand_control_mode = configured_hand_control_mode or str(
            getattr(self.cfg.booster, "hand_control_mode", "closed")
        ).strip().lower()
        if self.hand_control_mode not in {"environment", "closed"}:
            raise ValueError(
                "Unsupported hand control mode="
                f"{self.hand_control_mode!r}; expected 'environment' or 'closed'"
            )
        pico_hand_control = self.remoteControlService.pico_walking_control_enabled
        trigger_control = hand_control_enabled and (
            pico_hand_control or _env_bool("BOOSTER_HAND_TRIGGER_CONTROL", False)
        )
        retarget_control = (
            hand_control_enabled
            and not pico_hand_control
            and _env_bool("BOOSTER_HAND_RETARGET_CONTROL", False)
        )
        # On Pico, trigger control owns Custom hands regardless of the task's
        # default fixed-hand configuration. Walking applies the closed pose below.
        if hand_control_enabled and (
            pico_hand_control
            or (self.hand_control_mode == "closed" and (trigger_control or retarget_control))
        ):
            self.hand_control_mode = "environment"
        self.hand_hold_closed = hand_control_enabled and self.hand_control_mode == "closed"
        self.hand_trigger_enabled = not self.hand_hold_closed and trigger_control
        self.hand_retarget_enabled = not self.hand_hold_closed and retarget_control
        self.hand_publish_rate_hz = max(_env_float("BOOSTER_HAND_PUBLISH_RATE_HZ", 10.0), 1e-6)
        self.hand_trigger_eps = max(_env_float("BOOSTER_HAND_TRIGGER_EPS", 0.01), 0.0)

        mapping_config = os.getenv("BOOSTER_HAND_MAPPING_CONFIG", "").strip()
        self.hand_command_mapping: RealHandCommandMapping | None = None
        if hand_control_enabled and (self.hand_retarget_enabled or (mapping_config and not self.hand_hold_closed)):
            self.hand_command_mapping = load_real_hand_command_mapping(
                mapping_config or DEFAULT_T2_REVO2_ROS_MAPPING_PATH
            )

        self.revo2_hand_sides = ("left", "right")

        default_open = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        default_closed = (
            750.0 / REVO2_POSITION_COMMAND_SCALE,
            60.0 / REVO2_POSITION_COMMAND_SCALE,
            REVO2_POSITION_COMMAND_MAX_RAD,
            REVO2_POSITION_COMMAND_MAX_RAD,
            REVO2_POSITION_COMMAND_MAX_RAD,
            REVO2_POSITION_COMMAND_MAX_RAD,
        )
        if (
            self.hand_trigger_enabled
            and not self.hand_retarget_enabled
            and self.remoteControlService.pico_walking_control_enabled
        ):
            default_open = (0.0, 750.0 / REVO2_POSITION_COMMAND_SCALE, 0.0, 0.0, 0.0, 0.0)
            default_closed = (
                850.0 / REVO2_POSITION_COMMAND_SCALE,
                750.0 / REVO2_POSITION_COMMAND_SCALE,
                850.0 / REVO2_POSITION_COMMAND_SCALE,
                REVO2_POSITION_COMMAND_MAX_RAD,
                REVO2_POSITION_COMMAND_MAX_RAD,
                REVO2_POSITION_COMMAND_MAX_RAD,
            )
        default_open = _parse_float_list_env("BOOSTER_HAND_OPEN_POSITION", default_open, expected_len=6)
        default_closed = _parse_float_list_env("BOOSTER_HAND_CLOSED_POSITION", default_closed, expected_len=6)
        _validate_hand_command_values(default_open, "BOOSTER_HAND_OPEN_POSITION")
        _validate_hand_command_values(default_closed, "BOOSTER_HAND_CLOSED_POSITION")
        for side in self.revo2_hand_sides:
            prefix = f"BOOSTER_{side.upper()}_HAND"
            if self.hand_command_mapping is not None:
                mapping = self.hand_command_mapping.for_side(side)
                topic = os.getenv(f"{prefix}_TOPIC", mapping.topic)
                names = list(mapping.target_joint_names)
            else:
                topic = os.getenv(f"{prefix}_TOPIC") or f"/hand_{side}/{side}_revo2_hand_position_controller/commands"
                names = _parse_string_list_env(
                    f"{prefix}_JOINT_NAMES",
                    _make_hand_joint_names(os.getenv(f"{prefix}_JOINT_PREFIX", side)),
                    expected_len=6,
                )
            opened = _parse_float_list_env(f"{prefix}_OPEN_POSITION", default_open, expected_len=6)
            closed = _parse_float_list_env(f"{prefix}_CLOSED_POSITION", default_closed, expected_len=6)
            _validate_hand_command_values(opened, f"{prefix}_OPEN_POSITION")
            _validate_hand_command_values(closed, f"{prefix}_CLOSED_POSITION")
            setattr(self, f"{side}_hand_topic", topic)
            setattr(self, f"{side}_hand_joint_names", names)
            setattr(self, f"{side}_hand_open_position", opened)
            setattr(self, f"{side}_hand_closed_position", closed)
        if self.left_hand_topic == self.right_hand_topic:
            raise ValueError("Left and right hand command topics must differ")
        self.hand_stall_recovery_enabled = (
            _env_bool(
                "BOOSTER_HAND_TURBO_ENABLED",
                _env_bool("BOOSTER_HAND_SOFT_TURBO", True),
            )
            and (self.hand_trigger_enabled or self.hand_retarget_enabled)
        )
        self.left_hand_feedback_topic = os.getenv(
            "BOOSTER_LEFT_HAND_FEEDBACK_TOPIC",
            "/hand_left/joint_states",
        )
        self.right_hand_feedback_topic = os.getenv(
            "BOOSTER_RIGHT_HAND_FEEDBACK_TOPIC",
            "/hand_right/joint_states",
        )
        recovery_config = HandStallRecoveryConfig(
            enabled=self.hand_stall_recovery_enabled,
            feedback_stale_s=_env_float_alias(
                "BOOSTER_HAND_TURBO_FEEDBACK_STALE_S",
                "BOOSTER_HAND_SOFT_TURBO_FEEDBACK_STALE_S",
                0.15,
            ),
            window_frames=int(
                _env_float_alias(
                    "BOOSTER_HAND_TURBO_WINDOW_FRAMES",
                    "BOOSTER_HAND_SOFT_TURBO_WINDOW_FRAMES",
                    5.0,
                )
            ),
            stationary_tolerance=_env_float_alias(
                "BOOSTER_HAND_TURBO_STATIONARY_TOLERANCE",
                "BOOSTER_HAND_SOFT_TURBO_STATIONARY_TOLERANCE",
                1.0 / REVO2_POSITION_COMMAND_SCALE,
            ),
            stationary_duration_s=_env_float_alias(
                "BOOSTER_HAND_TURBO_STATIONARY_DURATION_S",
                "BOOSTER_HAND_SOFT_TURBO_STATIONARY_DURATION_S",
                0.2,
            ),
            close_error_min=_env_float_alias(
                "BOOSTER_HAND_TURBO_CLOSE_ERROR_MIN",
                "BOOSTER_HAND_SOFT_TURBO_CLOSE_ERROR_MIN",
                10.0 / REVO2_POSITION_COMMAND_SCALE,
            ),
            backoff_units=_env_float_alias(
                "BOOSTER_HAND_TURBO_BACKOFF_UNITS",
                "BOOSTER_HAND_SOFT_TURBO_BACKOFF_UNITS",
                1.0 / REVO2_POSITION_COMMAND_SCALE,
            ),
            retry_period_s=_env_float_alias(
                "BOOSTER_HAND_TURBO_RETRY_PERIOD_S",
                "BOOSTER_HAND_SOFT_TURBO_RETRY_PERIOD_S",
                0.5,
            ),
            command_min=_HAND_COMMAND_MIN,
            command_max=_HAND_COMMAND_MAX,
        )
        self.hand_stall_recovery = HandStallRecovery(
            recovery_config,
            joint_names={side: getattr(self, f"{side}_hand_joint_names") for side in self.revo2_hand_sides},
            open_positions={side: getattr(self, f"{side}_hand_open_position") for side in self.revo2_hand_sides},
            closed_positions={side: getattr(self, f"{side}_hand_closed_position") for side in self.revo2_hand_sides},
            logger=self.logger,
        )
        self.hand_close_thumb_delay_s = max(
            _env_float("BOOSTER_HAND_CLOSE_THUMB_DELAY_S", 0.15),
            0.0,
        )
        self._last_hand_publish_s = 0.0
        self._last_hand_commands: dict[str, Float64MultiArray] = {}
        self._hand_feedback_positions: dict[str, tuple[float, list[float]]] = {}
        self._walking_hand_close_starts: dict[str, tuple[float, list[float]]] = {}
        self._next_walking_hand_status_s = 0.0
        self._last_left_trigger: float | None = None
        self._last_right_trigger: float | None = None
        self._hand_input_status = None
        self._next_hand_status_s = 0.0
        self.hand_retarget_host = os.getenv("BOOSTER_HAND_RETARGET_HOST", "127.0.0.1")
        self.hand_retarget_port = int(_env_float("BOOSTER_HAND_RETARGET_PORT", 5558))
        self.hand_retarget_topic = os.getenv("BOOSTER_HAND_RETARGET_TOPIC", "hand_retarget.revo2_commands")
        self.hand_retarget_stale_s = max(_env_float("BOOSTER_HAND_RETARGET_STALE_S", 0.25), 0.0)
        self.hand_retarget_subscriber = None
        self._last_hand_retarget_frame = None
        self._last_hand_retarget_recv_ns = 0
        if self.hand_retarget_enabled:
            from booster_deploy.hand_retargeting.zmq import HandRetargetFrameSubscriber

            self.hand_retarget_subscriber = HandRetargetFrameSubscriber(
                self.hand_retarget_host,
                self.hand_retarget_port,
                self.hand_retarget_topic,
                recv_timeout_ms=0,
            )

        self.logger.info(
            "[hand] config: backend=%s mode=%s closed=%s trigger=%s retarget=%s "
            "topics=(%s, %s) rate=%.1fHz",
            getattr(self.remoteControlService, "_backend", "unknown"),
            self.hand_control_mode,
            self.hand_hold_closed,
            self.hand_trigger_enabled,
            self.hand_retarget_enabled,
            self.left_hand_topic,
            self.right_hand_topic,
            self.hand_publish_rate_hz,
        )

    def _make_hand_command(self, names: list[str] | tuple[str, ...], position) -> Float64MultiArray:

        positions = [float(item) for item in position]
        if len(names) != len(positions):
            raise ValueError(f"Hand command has {len(names)} names and {len(positions)} positions")
        positions = _validate_hand_command_values(positions, "Float64MultiArray.data")
        msg = Float64MultiArray()
        msg.data = positions
        return msg

    def _publish_hand_command(self, hand_side: str, command: Float64MultiArray) -> None:
        publisher = getattr(self, f"{hand_side}_hand_cmd_publisher")
        if publisher is None:
            return
        if self.control_state != BoosterControlState.WALKING:
            self._walking_hand_close_starts = {}
        publisher.publish(command)
        self._last_hand_commands[hand_side] = command

    def _publish_paused_hand_commands(self) -> None:
        if (
            self.control_state != BoosterControlState.TRACKING_PAUSED
            or self._output_mode != "policy"
        ):
            return
        now_s = time.monotonic()
        if now_s - self._last_hand_publish_s < 1.0 / self.hand_publish_rate_hz:
            return
        # Replay the last published targets, including any stall-recovery offset.
        # Do not derive a new pose from triggers or feedback while paused.
        for hand_side, command in self._last_hand_commands.items():
            self._publish_hand_command(hand_side, command)
        if self._last_hand_commands:
            self._last_hand_publish_s = now_s

    def _hand_feedback_handler(self, hand_side: str, msg: JointState) -> None:
        if hand_side not in self.revo2_hand_sides:
            return
        recovery = getattr(self, "hand_stall_recovery", None)
        try:
            names = list(getattr(msg, "name", []))
            positions = list(getattr(msg, "position", []))
            if len(names) != len(positions) or len(names) != len(set(names)):
                raise ValueError("Hand feedback names and positions must have equal unique entries")
            by_name = dict(zip(names, positions, strict=True))
            ordered = _validate_hand_command_values(
                [by_name[name] for name in getattr(self, f"{hand_side}_hand_joint_names")],
                f"{hand_side} hand feedback",
            )
            now_s = time.monotonic()
            self._hand_feedback_positions[hand_side] = (now_s, ordered)
            if recovery is not None:
                recovery.update_feedback(hand_side, names, positions, now_s=now_s)
        except (KeyError, TypeError, ValueError) as exc:
            self.logger.warning("Ignoring invalid %s hand feedback: %s", hand_side, exc)
            self._hand_feedback_positions.pop(hand_side, None)
            if recovery is not None:
                recovery.reset(hand_side, clear_feedback=True)

    def _reset_hand_stall_recovery(self) -> None:
        recovery = getattr(self, "hand_stall_recovery", None)
        if recovery is not None:
            recovery.reset()

    def _apply_hand_stall_recovery(
        self,
        hand_side: str,
        desired_positions,
        *,
        now_s: float,
    ) -> np.ndarray:
        recovery = getattr(self, "hand_stall_recovery", None)
        if recovery is None or hand_side not in self.revo2_hand_sides:
            return np.asarray(desired_positions, dtype=np.float64)
        return recovery.apply(hand_side, desired_positions, now_s=now_s)

    def _hand_control_active(self) -> bool:

        return hand_control_active_for_state(
            self.control_state,
            getattr(self, "_output_mode", "none"),
        )

    def _publish_closed_hand_commands(self) -> None:

        if self.control_state == BoosterControlState.WALKING:
            # Starting/re-entering Walking must not bypass the one-second ramp.
            self._walking_hand_close_starts.clear()
            self._next_walking_hand_status_s = 0.0
            self._publish_walking_hand_commands(force=True)
            return
        if not (
            self.hand_trigger_enabled
            or self.hand_retarget_enabled
            or getattr(self, "hand_hold_closed", False)
        ):
            return
        if self.left_hand_cmd_publisher is None and self.right_hand_cmd_publisher is None:
            return
        if getattr(self, "hand_hold_closed", False):
            self._publish_fixed_closed_hand_commands(force=True)
            return
        for side in self.revo2_hand_sides:
            names = getattr(self, f"{side}_hand_joint_names")
            position = _hand_four_finger_closed_position(
                names,
                getattr(self, f"{side}_hand_open_position"),
                getattr(self, f"{side}_hand_closed_position"),
            )
            self._publish_hand_command(side, self._make_hand_command(names, position))
        delay_s = float(getattr(self, "hand_close_thumb_delay_s", 0.15))
        if delay_s > 0.0:
            time.sleep(delay_s)
        self._publish_hand_closed_targets()
        self._last_hand_publish_s = time.monotonic()

    def _publish_fixed_closed_hand_commands(self, *, force: bool = False) -> None:

        if self.control_state == BoosterControlState.WALKING:
            self._publish_walking_hand_commands(force=force)
            return
        if not getattr(self, "hand_hold_closed", False):
            return
        if self.left_hand_cmd_publisher is None and self.right_hand_cmd_publisher is None:
            return
        now_s = time.monotonic()
        if (
            not force
            and now_s - self._last_hand_publish_s < 1.0 / self.hand_publish_rate_hz
        ):
            return
        self._publish_hand_closed_targets()
        self._last_hand_publish_s = now_s

    def _publish_hand_closed_targets(self) -> None:
        for side in self.revo2_hand_sides:
            command = self._make_hand_command(
                getattr(self, f"{side}_hand_joint_names"),
                getattr(self, f"{side}_hand_closed_position"),
            )
            self._publish_hand_command(side, command)

    def _log_walking_hand_status(self, *, now_s: float | None = None) -> None:
        now = time.monotonic() if now_s is None else now_s
        if now < self._next_walking_hand_status_s:
            return
        self._next_walking_hand_status_s = now + 2.0
        self.logger.info(
            "[hand] walking closed target: left=%s right=%s ramp=%.1fs "
            "current=(%s, %s) topics=(%s, %s) "
            "ROS subscribers=(%s, %s)",
            self.left_hand_closed_position,
            self.right_hand_closed_position,
            _WALKING_HAND_CLOSE_DURATION_S,
            getattr(self._last_hand_commands.get("left"), "data", []),
            getattr(self._last_hand_commands.get("right"), "data", []),
            self.left_hand_topic,
            self.right_hand_topic,
            self._hand_subscription_count("left"),
            self._hand_subscription_count("right"),
        )

    def _hand_subscription_count(self, side: str) -> int:
        publisher = getattr(self, f"{side}_hand_cmd_publisher")
        return 0 if publisher is None else publisher.get_subscription_count()

    def _walking_hand_start_position(self, hand_side: str, *, now_s: float) -> list[float]:
        feedback = self._hand_feedback_positions.get(hand_side)
        if (
            feedback is not None
            and 0.0 <= now_s - feedback[0] <= self.hand_stall_recovery.config.feedback_stale_s
        ):
            return list(feedback[1])
        previous = self._last_hand_commands.get(hand_side)
        if previous is not None:
            return list(previous.data)
        return list(getattr(self, f"{hand_side}_hand_open_position"))

    def _publish_walking_hand_commands(self, *, force: bool = False) -> None:
        if (
            not getattr(self, "hand_control_enabled", True)
            or not self.is_running
            or self.exit_event.is_set()
            or self.control_state != BoosterControlState.WALKING
        ):
            self._walking_hand_close_starts.clear()
            return
        if self.left_hand_cmd_publisher is None and self.right_hand_cmd_publisher is None:
            return
        now_s = time.monotonic()
        if not force and now_s - self._last_hand_publish_s < 1.0 / self.hand_publish_rate_hz:
            return
        self._reset_hand_stall_recovery()
        for hand_side in ("left", "right"):
            publisher = getattr(self, f"{hand_side}_hand_cmd_publisher")
            if publisher is None:
                continue
            subscribers = publisher.get_subscription_count()
            start = self._walking_hand_close_starts.get(hand_side)
            if subscribers == 0:
                # Do not let the ramp finish before the driver can receive it.
                self._walking_hand_close_starts.pop(hand_side, None)
                start = None
            if start is None:
                start = (now_s, self._walking_hand_start_position(hand_side, now_s=now_s))
                if subscribers > 0:
                    self._walking_hand_close_starts[hand_side] = start
            progress = min(max((now_s - start[0]) / _WALKING_HAND_CLOSE_DURATION_S, 0.0), 1.0)
            closed = getattr(self, f"{hand_side}_hand_closed_position")
            positions = closed if progress >= 1.0 else [
                opened + progress * (target - opened)
                for opened, target in zip(start[1], closed, strict=True)
            ]
            if self.control_state != BoosterControlState.WALKING:
                self._walking_hand_close_starts.clear()
                return
            self._publish_hand_command(
                hand_side,
                self._make_hand_command(getattr(self, f"{hand_side}_hand_joint_names"), positions),
            )
        self._last_hand_publish_s = now_s
        self._log_walking_hand_status(now_s=now_s)

    def _publish_hand_retarget_commands(self) -> None:

        if not self.hand_retarget_enabled:
            return
        subscriber = self.hand_retarget_subscriber
        if subscriber is None:
            return
        frame = subscriber.try_recv_latest()
        if frame is not None:
            self._last_hand_retarget_frame = frame
            self._last_hand_retarget_recv_ns = time.monotonic_ns()
        if not self._hand_control_active():
            self._reset_hand_stall_recovery()
            return
        frame = self._last_hand_retarget_frame
        if frame is None:
            return
        now_ns = time.monotonic_ns()
        if self._last_hand_retarget_recv_ns > 0 and self.hand_retarget_stale_s > 0.0:
            age_s = (now_ns - self._last_hand_retarget_recv_ns) / 1e9
            if age_s > self.hand_retarget_stale_s:
                self._reset_hand_stall_recovery()
                return
        now_s = time.monotonic()
        if now_s - self._last_hand_publish_s < 1.0 / self.hand_publish_rate_hz:
            return

        published = False
        for command in frame.commands:
            hand_side = str(command.hand_side).strip().lower()
            if hand_side not in self.revo2_hand_sides or getattr(self, f"{hand_side}_hand_cmd_publisher") is None:
                continue
            try:
                if self.hand_command_mapping is not None:
                    side_mapping = self.hand_command_mapping.for_side(hand_side)
                    names, positions = side_mapping.remap_positions(
                        command.joint_names,
                        command.joint_pos,
                    )
                    positions = side_mapping.to_driver_positions(positions)
                    positions = self._apply_hand_stall_recovery(
                        hand_side,
                        positions,
                        now_s=now_s,
                    )
                else:
                    raise ValueError("hand retarget mapping is required for the Revo2 ROS radian command contract")
            except ValueError as exc:
                self.logger.warning(
                    "Skipping invalid %s hand retarget command: %s",
                    hand_side,
                    exc,
                )
                continue
            self._publish_hand_command(hand_side, self._make_hand_command(names, positions))
            published = True
        if published:
            self._last_hand_publish_s = now_s

    def _publish_hand_trigger_commands(self) -> None:

        if not self.hand_trigger_enabled:
            return
        now_s = time.monotonic()
        if self.left_hand_cmd_publisher is None and self.right_hand_cmd_publisher is None:
            return
        if not self._hand_control_active():
            self._reset_hand_stall_recovery()
            self._publish_paused_hand_commands()
            status = (
                f"gated:{getattr(self.control_state, 'value', self.control_state)}:"
                f"{getattr(self, '_output_mode', 'none')}"
            )
            if self._hand_input_status != status or now_s >= self._next_hand_status_s:
                self.logger.info(
                    "[hand] trigger gated: state=%s output=%s (Walking closes hands; pause holds last command)",
                    getattr(self.control_state, "value", self.control_state),
                    getattr(self, "_output_mode", "none"),
                )
                self._hand_input_status = status
                self._next_hand_status_s = now_s + 1.0
            return
        fresh = getattr(self.remoteControlService, "pico_control_fresh", None)
        if not callable(fresh) or not fresh():
            self._reset_hand_stall_recovery()
            if self._hand_input_status != "stale" or now_s >= self._next_hand_status_s:
                pico_config = getattr(self.remoteControlService, "pico_config", None)
                self.logger.warning(
                    "[hand] no command: PICO input missing/stale; backend=%s endpoint=tcp://%s:%s/%s "
                    "state=%s output=%s",
                    getattr(self.remoteControlService, "_backend", "unknown"),
                    getattr(pico_config, "host", "?"),
                    getattr(pico_config, "port", "?"),
                    getattr(pico_config, "topic", "?"),
                    getattr(self.control_state, "value", self.control_state),
                    getattr(self, "_output_mode", "none"),
                )
                self._hand_input_status = "stale"
                self._next_hand_status_s = now_s + 1.0
            return
        if now_s - self._last_hand_publish_s < 1.0 / self.hand_publish_rate_hz:
            return

        left_trigger = self.remoteControlService.get_left_trigger_cmd()
        right_trigger = self.remoteControlService.get_right_trigger_cmd()
        left_position = _hand_position_from_trigger(
            left_trigger,
            self.left_hand_open_position,
            self.left_hand_closed_position,
        )
        right_position = _hand_position_from_trigger(
            right_trigger,
            self.right_hand_open_position,
            self.right_hand_closed_position,
        )
        left_position = self._apply_hand_stall_recovery(
            "left",
            left_position,
            now_s=now_s,
        )
        right_position = self._apply_hand_stall_recovery(
            "right",
            right_position,
            now_s=now_s,
        )
        left_msg = self._make_hand_command(self.left_hand_joint_names, left_position)
        right_msg = self._make_hand_command(self.right_hand_joint_names, right_position)
        self._publish_hand_command("left", left_msg)
        self._publish_hand_command("right", right_msg)
        self._last_left_trigger = left_trigger
        self._last_right_trigger = right_trigger
        self._last_hand_publish_s = now_s
        if self._hand_input_status != "sending" or now_s >= self._next_hand_status_s:
            left_subscribers = self._hand_subscription_count("left")
            right_subscribers = self._hand_subscription_count("right")
            self.logger.info(
                "[hand] sending: backend=%s state=%s output=%s trigger=(%.3f, %.3f) "
                "left=%s right=%s topics=(%s, %s) ROS subscribers=(%s, %s)",
                getattr(self.remoteControlService, "_backend", "unknown"),
                getattr(self.control_state, "value", self.control_state),
                getattr(self, "_output_mode", "none"),
                left_trigger, right_trigger, left_msg.data, right_msg.data,
                self.left_hand_topic, self.right_hand_topic,
                left_subscribers, right_subscribers,
            )
            self._hand_input_status = "sending"
            self._next_hand_status_s = now_s + 1.0

    def _init_communication(self) -> None:

        try:
            self.create_low_cmd_publisher("booster_deploy_low_cmd_pub")
            self._start_low_state_subscription()
            self._start_policy_output_thread()
        except Exception as e:
            self.logger.error(f"Failed to initialize communication: {e}")
            raise

    def _start_low_state_subscription(self) -> None:

        def low_state_service_executor():

            self.logger.info("Low state subscription started")
            low_state_node = rclpy.create_node("booster_deploy_low_state_sub")
            low_state_node.create_subscription(
                LowState,
                "/low_state",
                self._low_state_handler,
                QoSProfile(
                    depth=1,
                    reliability=ReliabilityPolicy.BEST_EFFORT,
                    history=HistoryPolicy.KEEP_LAST,
                ),
            )
            if self.remoteControlService.robot_joystick_enabled:
                low_state_node.create_subscription(
                    RemoteControllerState,
                    self.remoteControlService.robot_joystick_topic,
                    self._robot_joystick_state_handler,
                    QoSProfile(
                        depth=1,
                        reliability=ReliabilityPolicy.RELIABLE,
                        history=HistoryPolicy.KEEP_LAST,
                        durability=DurabilityPolicy.VOLATILE,
                    ),
                )
                self.logger.info(
                    "Robot joystick subscription started: topic=%s",
                    self.remoteControlService.robot_joystick_topic,
                )
            # Walking also needs measured positions when stall recovery is off.
            feedback_qos = QoSProfile(
                depth=max(self.hand_stall_recovery.config.window_frames, 5),
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
            )
            for side in self.revo2_hand_sides:
                topic = getattr(self, f"{side}_hand_feedback_topic")
                low_state_node.create_subscription(
                    JointState,
                    topic,
                    lambda msg, hand_side=side: self._hand_feedback_handler(hand_side, msg),
                    feedback_qos,
                )
                self.logger.info("%s hand feedback subscription started: %s", side, topic)

            # Run even before body feedback or PICO packets arrive, and retry as
            # the hand drivers finish ROS discovery. The same executor handles
            # low-state callbacks, so both paths share the publish rate limit.
            low_state_node.create_timer(
                1.0 / self.hand_publish_rate_hz,
                self._publish_walking_hand_commands,
            )

            executor = SingleThreadedExecutor()
            executor.add_node(low_state_node)
            # RPC client belongs to publish_node, not low_state_node. Never spin
            # it from the main control loop or attach it to a second executor.
            if getattr(self, "publish_node", None) is not None:
                executor.add_node(self.publish_node)

            try:

                while rclpy.ok() and not self.exit_event.is_set():
                    executor.spin_once(timeout_sec=0.1)
            except ExternalShutdownException:
                pass
            except Exception as exc:

                is_rcl_error = "RCLError" in type(exc).__name__
                is_shutting_down = self.exit_event.is_set() or not rclpy.ok()

                if is_rcl_error and is_shutting_down:
                    pass
                else:
                    self.logger.error(
                        "Low state subscription executor stopped: %s",
                        exc,
                        exc_info=True
                    )
            finally:
                executor.shutdown()
                low_state_node.destroy_node()
            self.logger.info("Low state subscription stopped")

        self.low_state_thread = threading.Thread(
            target=low_state_service_executor,
            name="low_state_executor",
            daemon=True,
        )
        self.low_state_thread.start()

    def _start_policy_output_thread(self) -> None:

        def policy_output_loop() -> None:
            self.logger.info("Policy action publisher started")
            while not self.exit_event.is_set():
                updated = self.policy_action_updated.wait(timeout=0.02)
                if not updated:
                    continue
                self.policy_action_updated.clear()
                try:
                    self._publish_latest_policy_action()
                except Exception as exc:
                    self.logger.error(
                        "Policy action publisher stopped: %s",
                        exc,
                        exc_info=True,
                    )
                    self.is_running = False
                    self.exit_event.set()
                    break
            self.logger.info("Policy action publisher stopped")

        self.policy_output_thread = threading.Thread(
            target=policy_output_loop,
            name="policy_action_publisher",
            daemon=True,
        )
        self.policy_output_thread.start()

    def _robot_joystick_state_handler(
        self,
        remote_state_msg: RemoteControllerState,
    ) -> None:

        try:
            self.remoteControlService.handle_robot_joystick_state(
                remote_state_msg,
                local_recv_s=time.monotonic(),
            )
        except Exception as exc:
            self.logger.error(
                "Error in robot joystick callback: %s",
                exc,
                exc_info=True,
            )

    def _low_state_handler(self, low_state_msg: LowState):

        self.metrics["low_state_handler"].mark()
        try:
            if not self.is_running or self.exit_event.is_set():
                return

            self._last_low_state_time = time.monotonic()
            recv_ns = time.monotonic_ns()
            self.timer.tick_timer_if_sim()

            rpy = np.array(low_state_msg.imu_state.rpy, dtype=np.float32)
            gyro = np.array(low_state_msg.imu_state.gyro, dtype=np.float32)
            dof_pos = np.zeros(self.robot.num_joints, dtype=np.float32)
            dof_vel = np.zeros(self.robot.num_joints, dtype=np.float32)
            fb_torque = np.zeros(self.robot.num_joints, dtype=np.float32)

            motor_states = low_state_msg.motor_state_serial
            if len(motor_states) != self.robot.num_joints:
                raise ValueError(
                    "Booster low state joint count does not match the configured robot: "
                    f"received={len(motor_states)}, expected={self.robot.num_joints}"
                )
            for i, motor in enumerate(motor_states):
                dof_pos[i] = motor.q
                dof_vel[i] = motor.dq
                fb_torque[i] = motor.tau_est

            self._state_buf[0]["root_rpy_w"][:] = rpy
            self._state_buf[0]["root_ang_vel_b"][:] = gyro
            self._state_buf[0]["root_pos_w"][:] = np.zeros(
                3, dtype=np.float32
            )
            self._state_buf[0]["root_lin_vel_w"][:] = np.zeros(
                3, dtype=np.float32
            )
            self._state_buf[0]["joint_pos"][:] = dof_pos
            self._state_buf[0]["joint_vel"][:] = dof_vel
            self._state_buf[0]["feedback_torque"][:] = fb_torque
            has_recv_id = isinstance(self._state_buf, np.ndarray) and "state_recv_id" in (self._state_buf.dtype.names or ())
            if has_recv_id:
                self._state_buf[0]["state_recv_id"] += 1
                self._state_buf[0]["state_recv_ns"] = recv_ns
                if "low_state_size" in self._state_buf.dtype.names:
                    # Own nested arrays before callback return, never JSON under lock.
                    try:
                        payload = pack_message(low_state_msg)
                        self._state_buf[0]["low_state_payload"] = np.void(
                            payload.ljust(PAYLOAD_CAPACITY, b"\0"))
                        self._state_buf[0]["low_state_size"] = len(payload)
                    except Exception as exc:
                        self._state_buf[0]["low_state_size"] = 0
                        if not getattr(self, "_snapshot_error_reported", False):
                            self.logger.error("Full low_state snapshot unavailable: %s", exc)
                            self._snapshot_error_reported = True
            self.synced_state.write(self._state_buf)
            if self._output_mode == "hold":
                self._publish_hold_target()

            cmd = np.zeros((1,), dtype=self.synced_command.dtype)
            cmd[0]["vx"] = self.remoteControlService.get_vx_cmd()
            cmd[0]["vy"] = self.remoteControlService.get_vy_cmd()
            cmd[0]["vyaw"] = self.remoteControlService.get_vyaw_cmd()
            cmd[0]["pause_toggle"] = self._pause_toggle_seq
            exiting = getattr(self, '_reference_exit', None)
            if exiting:
                cmd[0]['reference_exit_token'] = exiting['token'].encode()
                cmd[0]['reference_exit_session'] = exiting['session'].encode()
            self.synced_command.write(cmd)
            if self.control_state == BoosterControlState.WALKING:
                self._publish_walking_hand_commands()
            elif self.hand_hold_closed:
                self._reset_hand_stall_recovery()
                self._publish_fixed_closed_hand_commands()
            elif self.hand_retarget_enabled:
                self._publish_hand_retarget_commands()
            elif self.hand_trigger_enabled:
                self._publish_hand_trigger_commands()
            else:
                self._reset_hand_stall_recovery()

        except Exception as e:
            self.logger.error(f"Error in _low_state_handler: {e}")
            self.is_running = False
            self.exit_event.set()

    def _set_output_mode(self, mode: str) -> None:

        if mode not in {"none", "hold", "policy"}:
            raise ValueError(f"Unsupported Booster output mode: {mode!r}")
        with self._output_lock:
            self._output_mode = mode

    def _set_hold_target(
        self,
        joint_pos: np.ndarray | list[float],
        stiffness: np.ndarray | list[float] | None = None,
        damping: np.ndarray | list[float] | None = None,
    ) -> None:

        joint_pos = np.asarray(joint_pos, dtype=np.float64)
        if joint_pos.shape != (self.robot.num_joints,):
            raise ValueError(
                f"Expected {self.robot.num_joints} hold joints, got {joint_pos.shape}"
            )
        with self._output_lock:
            self._hold_joint_pos = joint_pos.copy()
            if stiffness is not None:
                self._hold_stiffness = np.asarray(stiffness, dtype=np.float64).copy()
            if damping is not None:
                self._hold_damping = np.asarray(damping, dtype=np.float64).copy()

    def _set_hold_from_current_state(self) -> np.ndarray:

        state = self.synced_state.read()[0]
        joint_pos = np.asarray(state["joint_pos"], dtype=np.float64)
        self._set_hold_target(joint_pos)
        return joint_pos.copy()

    def _wait_for_fresh_low_state(
        self, *, timeout_s: float = 0.0, stale_after_s: float = 0.5,
    ) -> bool:
        # Compatibility name; readiness is now always a non-blocking snapshot.
        return (not self.exit_event.is_set() and self._last_low_state_time > 0.0
                and time.monotonic() - self._last_low_state_time <= stale_after_s)

    def _prepare_target(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

        rcfg = self.robot.cfg
        if isinstance(rcfg.prepare_state, type(MISSING)):
            stiffness = np.asarray(rcfg.joint_stiffness, dtype=np.float64)
            damping = np.asarray(rcfg.joint_damping, dtype=np.float64)
            joint_pos = np.asarray(rcfg.default_joint_pos, dtype=np.float64)
        else:
            stiffness = np.asarray(rcfg.prepare_state.stiffness, dtype=np.float64)
            damping = np.asarray(rcfg.prepare_state.damping, dtype=np.float64)
            joint_pos = np.asarray(rcfg.prepare_state.joint_pos, dtype=np.float64)
        return joint_pos.copy(), stiffness.copy(), damping.copy()

    def _wait_for_joint_ctrl_subscriber(self, timeout_s: float = 0.0) -> bool:
        publisher = self.low_cmd_publisher
        return bool(publisher is not None and rclpy.ok()
                    and publisher.get_subscription_count() > 0)

    def _publish_joint_target(
        self,
        joint_pos: np.ndarray,
        stiffness: np.ndarray,
        damping: np.ndarray,
        measured_joint_pos: np.ndarray | None = None,
        *,policy_tick=None,
    ) -> None:

        if self.low_cmd_publisher is None:
            return
        joint_pos = np.asarray(joint_pos, dtype=np.float64)
        stiffness = np.asarray(stiffness, dtype=np.float64)
        damping = np.asarray(damping, dtype=np.float64)
        expected_shape = (self.robot.num_joints,)
        if not (
            joint_pos.shape
            == stiffness.shape
            == damping.shape
            == expected_shape
        ):
            raise ValueError(
                "Booster joint command arrays must match the robot joint count, got "
                f"joint_pos={joint_pos.shape}, stiffness={stiffness.shape}, "
                f"damping={damping.shape}, expected={expected_shape}"
            )
        if not np.isfinite(damping).all() or np.any(damping < 0.0):
            raise ValueError("Booster joint damping must be finite and non-negative")
        joint_pos = _limit_joint_position_by_effort(
            joint_pos,
            measured_joint_pos,
            stiffness,
            self._joint_effort_limit,
        )
        with self._output_lock:
            for i in range(self.robot.num_joints):
                self.motor_cmd[i].q = float(joint_pos[i])
                self.motor_cmd[i].kp = float(stiffness[i])
                self.motor_cmd[i].kd = float(damping[i])
            self.low_cmd_publisher.publish(self.low_cmd)
            published_ns=time.monotonic_ns()
            owned_command=message_fields(self.low_cmd)
        from booster_deploy.telemetry.encoder import target as record_target
        record_target(self,joint_pos,policy_tick=policy_tick,backend='real',low_cmd=owned_command,
            _event_mono_ns=published_ns,published_mono_ns=published_ns,
            target_kind='policy' if policy_tick else 'local_hold_or_transition',
            stiffness=stiffness,damping=damping,output_mode=getattr(self,'_output_mode','unknown'),
            consumed_state_recv_id=getattr(self,'_telemetry_state_recv_id',None),
            semantics='local_publish_not_execution_ack')
        return True

    def _publish_hold_target(self) -> None:

        with self._output_lock:
            hold_joint_pos = self._hold_joint_pos.copy()
            hold_stiffness = self._hold_stiffness.copy()
            hold_damping = self._hold_damping.copy()
        measured_joint_pos = np.asarray(
            self.synced_state.read()[0]["joint_pos"], dtype=np.float64
        )
        self._publish_joint_target(
            hold_joint_pos,
            hold_stiffness,
            hold_damping,
            measured_joint_pos,
        )

    def _publish_latest_policy_action(self) -> bool:

        with self._output_lock:
            if self._output_mode != "policy":
                return False
        action = self.synced_action.read()[0]
        if "policy_session" in action.dtype.names:
            self._diag_ctrl.policy_session = bytes(action["policy_session"]).decode() or None
            self._telemetry_policy_tick = bytes(action['telemetry_tick']).decode() if 'telemetry_tick' in action.dtype.names else None
            self._telemetry_state_recv_id = int(action["consumed_state_recv_id"])
        measured_joint_pos = np.asarray(
            self.synced_state.read()[0]["joint_pos"], dtype=np.float64
        )
        published = self._publish_joint_target(
            np.asarray(action["dof_target"], dtype=np.float64),
            np.asarray(action["stiffness"], dtype=np.float64),
            np.asarray(action["damping"], dtype=np.float64),
            measured_joint_pos,
            policy_tick=getattr(self,'_telemetry_policy_tick',None),
        )
        if published is True and 'reference_exit_token' in (action.dtype.names or ()):
            token = bytes(action['reference_exit_token']).decode()
            session = bytes(action['policy_session']).decode()
            with self._output_lock:
                if token and self._output_mode == 'policy':
                    self._reference_exit_published = (token, session)
        if published is True and 'reference_entry_cycle' in (action.dtype.names or ()):
            cycle = int(action['reference_entry_cycle'])
            session = bytes(action['policy_session']).decode()
            with self._output_lock:
                if cycle and self._output_mode == 'policy':
                    self._reference_entry_published = (cycle, session, int(action['policy_tick']))
        self.policy_output_published.set()
        self._record_ctrl_diag()
        return True

    def _publish_tracking_preload(self, transition: dict) -> bool:
        """Pre-send this startup's target; local publish is not a vendor ACK.

        Keep normal policy output/reference gates closed until Custom confirms.
        """
        if (transition is not getattr(self, '_tracking_transition', None)
                or time.monotonic() > transition['deadline_s']
                or getattr(self, '_shutdown_requested', False)
                or self.exit_event.is_set() or self.policy_stop_event.is_set()
                or self.inference_process is None or not self.inference_process.is_alive()
                or not self._wait_for_fresh_low_state(timeout_s=0.0)
                or not self._wait_for_joint_ctrl_subscriber(timeout_s=0.0)):
            return False
        if not self.policy_command_ready.is_set():
            return False
        with self._output_lock:
            if self._output_mode != 'none':
                return False
        action = self.synced_action.read()[0]
        session = bytes(action['policy_session']).decode()
        if session != transition.get('preload_session', transition.get('session', session)):
            return False
        if transition.get('cycle') and not bool(action['reference_entry_ready']):
            return False
        target, stiffness, damping = (
            np.asarray(action[name], dtype=np.float64)
            for name in ('dof_target', 'stiffness', 'damping'))
        if (any(value.shape != (self.robot.num_joints,) or not np.isfinite(value).all()
                for value in (target, stiffness, damping))
                or np.any(stiffness <= 0.0) or np.any(damping < 0.0)):
            return False
        self._telemetry_policy_tick = bytes(action['telemetry_tick']).decode() or None
        self._telemetry_state_recv_id = int(action['consumed_state_recv_id'])
        measured = np.asarray(self.synced_state.read()[0]['joint_pos'], dtype=np.float64)
        try:
            published = self._publish_joint_target(
                target, stiffness, damping, measured, policy_tick=self._telemetry_policy_tick)
        except Exception as exc:
            self._abort_tracking_transition('Custom preload publish failed: ' + str(exc))
            return False
        if published is not True:
            return False
        if 'preload_session' not in transition:
            transition['preload_session'] = session
            self._diag_ctrl.event('custom_target_preloaded', dict(
                cycle=transition.get('cycle', 0), policy_session_id=session,
                policy_tick=self._telemetry_policy_tick, semantics='local_publish_not_execution_ack'))
        return True

    def _record_ctrl_diag(self) -> None:
        """Count one target handed to the robot and the interval it arrived with."""

        diag = getattr(self, "_diag_ctrl", None)
        if diag is None:
            return
        now_ns = time.monotonic_ns()
        target_ms = 1000.0 * float(getattr(self.cfg, "policy_dt", 0.02))
        last_ns = getattr(self, "_diag_last_publish_ns", None)
        if last_ns is not None:
            dt_ms = (now_ns - last_ns) / 1e6
            diag.sample("dt", dt_ms, now_ns)
            # A 5 ms band fires on healthy hardware; 8 ms + "at least 5 per
            # second" separates real stalls from ordinary scheduling noise.
            if dt_ms > target_ms + LATE_TICK_MS:
                self._diag_late += 1
        self._diag_last_publish_ns = now_ns
        diag.mark("tick", now_ns)

    def _emit_ctrl_diag(self) -> None:
        """Human line: targets per second, their period tail, and late ticks."""

        diag = getattr(self, "_diag_ctrl", None)
        if diag is None or not diag.due():
            return
        now_ns = time.monotonic_ns()
        window = diag.interval("tick")
        hz = window.hz(now_ns)
        gap = window.gap_stats(now_ns)
        stall = window.stall_ms(now_ns)
        dt = diag.latency("dt", now_ns)
        late = getattr(self, "_diag_late", 0)
        alerts: list[str] = []
        tracking = self.control_state in {
            BoosterControlState.TRACKING,
            BoosterControlState.TRACKING_PAUSED,
        }
        if tracking and below_floor(hz):
            alerts.append(f"{hz:.1f} Hz < 50")
        if late >= LATE_TICKS_PER_S:
            alerts.append(f"late>+{LATE_TICK_MS:.0f}ms {late}/s")
        if tracking and stall is not None and stall > 200.0:
            alerts.append(f"stall {stall:.0f} ms")
        diag.emit(
            f"{hz:5.1f} Hz   dt {format_lat(dt)} ms",
            alerts=alerts,
            extra={"state": {"mode": self._output_mode,
                             "state": self.control_state.value,
                             **self._exit_safety_state(),
                             "gap_p95_ms": round(gap.p95, 1),
                             "late_ticks": late}},
            now_ns=now_ns,
        )
        self._diag_late = 0

    def _write_policy_action(self, dof_targets: torch.Tensor) -> None:

        dof_targets = torch.clamp(
            dof_targets,
            min=self.robot.joint_pos_limit_low.to(dof_targets.device),
            max=self.robot.joint_pos_limit_high.to(dof_targets.device),
        )
        for i in range(self.robot.num_joints):
            self._action_buf[0]["dof_target"][i] = float(dof_targets[i].item())
            self._action_buf[0]["stiffness"][i] = float(self.robot.joint_stiffness[i].item())
            self._action_buf[0]["damping"][i] = float(self.robot.joint_damping[i].item())
        if "policy_session" in self._action_buf.dtype.names:
            self._action_buf[0]["policy_session"] = getattr(self, "_telemetry_policy_session", "").encode()
            self._action_buf[0]["policy_tick"] = getattr(self, "_telemetry_policy_tick", 0)
            self._action_buf[0]["consumed_state_recv_id"] = getattr(self, "_telemetry_state_recv_id", 0) or 0
            if 'telemetry_tick' in self._action_buf.dtype.names:
                self._action_buf[0]['telemetry_tick']=getattr(self,'_telemetry_tick','').encode()
        if 'reference_exit_token' in self._action_buf.dtype.names:
            self._action_buf[0]['reference_exit_token'] = getattr(self, '_reference_exit_ack', '').encode()
        if "reference_entry_cycle" in self._action_buf.dtype.names:
            self._action_buf[0]["reference_entry_cycle"] = getattr(self, "_reference_entry_ack", 0)
            self._action_buf[0]["reference_entry_ready"] = getattr(self, "_reference_entry_ready", False)
        self.synced_action.write(self._action_buf)
        self.policy_command_ready.set()
        self.policy_action_updated.set()

    def create_low_cmd_publisher(self, name):

        self.publish_node = rclpy.create_node(name)
        publisher = self.publish_node.create_publisher(
            LowCmd,
            "joint_ctrl",
            QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                history=HistoryPolicy.KEEP_LAST
            )
        )
        self.low_cmd_publisher = publisher

        hand_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST)
        for side in self.revo2_hand_sides:
            if not self.hand_control_enabled:
                hand_publisher = None
            else:
                hand_publisher = self.publish_node.create_publisher(
                    Float64MultiArray, getattr(self, f"{side}_hand_topic"), hand_qos,
                )
            setattr(self, f"{side}_hand_cmd_publisher", hand_publisher)
        if self.hand_control_enabled:
            self.logger.info(
                "Walking closed-hand control enabled: left=%s right=%s rate=%.1fHz",
                self.left_hand_topic,
                self.right_hand_topic,
                self.hand_publish_rate_hz,
            )
        if self.hand_trigger_enabled or self.hand_retarget_enabled:
            self.logger.info(
                "Hand control: fixed closed hands in Walking; "
                "PICO/retarget targets in Custom/PVT hold and tracking; "
                "live hand commands disabled while tracking is paused; "
                "stall recovery=%s",
                self.hand_stall_recovery_enabled,
            )
        if self.hand_hold_closed:
            self.logger.info(
                "Fixed closed-hand control enabled: left=%s right=%s",
                self.left_hand_topic,
                self.right_hand_topic,
            )
        elif self.hand_retarget_enabled:
            self.logger.info(
                "Hand retarget control enabled: input=tcp://%s:%d/%s "
                "mapping=%s left=%s right=%s rate=%.1fHz interface=Float64MultiArray(rad)",
                self.hand_retarget_host,
                self.hand_retarget_port,
                self.hand_retarget_topic,
                self.hand_command_mapping.path if self.hand_command_mapping is not None else "source-names",
                self.left_hand_topic,
                self.right_hand_topic,
                self.hand_publish_rate_hz,
            )
            if self.hand_trigger_enabled:
                self.logger.warning("BOOSTER_HAND_RETARGET_CONTROL overrides PICO hand trigger publishing")
        elif self.hand_trigger_enabled:
            self.logger.info(
                "PICO hand trigger control enabled: left=%s right=%s rate=%.1fHz",
                self.left_hand_topic,
                self.right_hand_topic,
                self.hand_publish_rate_hz,
            )

        self.low_cmd = LowCmd()
        self.low_cmd.cmd_type = LowCmd.CMD_TYPE_SERIAL
        motor_cmd_buf = [
            MotorCmd() for _ in range(self.robot.num_joints)
        ]
        for i in range(self.robot.num_joints):
            motor_cmd_buf[i].q = 0.0
            motor_cmd_buf[i].dq = 0.0
            motor_cmd_buf[i].tau = 0.0
            motor_cmd_buf[i].kp = 0.0
            motor_cmd_buf[i].kd = 0.0
            motor_cmd_buf[i].weight = 0.0
        self.low_cmd.motor_cmd.extend(motor_cmd_buf)
        self.motor_cmd = self.low_cmd.motor_cmd

        self.rpc_service_client = self.publish_node.create_client(
            RpcService, "booster_rpc_service"
        )

        return publisher

    def _call_booster_rpc(
        self, api_id: int, body: dict[str, object] | None = None,
        *, timeout_sec: float | None = None,
    ) -> tuple[bool, dict | None]:
        # MOVE compatibility seam: one in-flight call, advanced by the executor.
        # False means not yet acknowledged (or failed), never a mode verdict.
        client = self.rpc_service_client
        if client is None or not rclpy.ok():
            return False, None
        key = (api_id, json.dumps(body) if body is not None else "")
        pending = getattr(self, "_velocity_rpc", None)
        now = time.monotonic()
        if pending:
            future = pending["future"]
            if not future.done() and now < pending["deadline"]:
                return False, None
            client.remove_pending_request(future)
            self._velocity_rpc = None
            if not future.done():
                future.cancel()
                return False, None
            try:
                response = future.result() if future.exception() is None else None
                if response is not None:
                    parsed = json.loads(response.msg.body) if response.msg.body else None
                    if pending["key"] == key:
                        return response.msg.status == 0, parsed
            except Exception as exc:
                self.logger.warning("MOVE response unknown: %s", exc)
        if not client.service_is_ready():
            return False, None
        try:
            request = RpcService.Request()
            request.msg = BoosterApiReqMsg()
            request.msg.api_id, request.msg.body = key
            self._velocity_rpc = dict(key=key, future=client.call_async(request),
                                      deadline=now + (3.0 if timeout_sec is None else timeout_sec))
        except Exception as exc:
            self.logger.warning("MOVE RPC unavailable: %s", exc)
        return False, None

    def _send_walking_velocity_rpc(
        self,
        command: tuple[float, float, float],
    ) -> bool:

        if getattr(self, "_shutdown_requested", False):
            # Shutdown owns MOVE until its zero ACK; joystick ticks cannot consume
            # that ACK or replace the stop with a nonzero command.
            return False
        if not self.remoteControlService.pico_walking_control_enabled:
            return False
        if self.control_state != BoosterControlState.WALKING and not (
            self.control_state in {BoosterControlState.ARMING_TRACKING, BoosterControlState.ARMING_PVT}
            and tuple(command) == (0.0, 0.0, 0.0)
        ):
            self.logger.error(
                "Refusing PICO Walking velocity while state=%s",
                self.control_state.value,
            )
            return False
        values = tuple(float(value) for value in command)
        if len(values) != 3 or not np.isfinite(values).all():
            self.logger.error(
                "Refusing invalid PICO Walking velocity: %s", command
            )
            values = (0.0, 0.0, 0.0)
        ok, _ = self._call_booster_rpc(
            _LOC_API_MOVE,
            {"vx": values[0], "vy": values[1], "vyaw": values[2]},
            timeout_sec=float(
                self.remoteControlService.pico_walking_config.rpc_timeout_s
            ),
        )
        return ok

    def _reset_pico_walking_velocity(self, *, send_stop: bool) -> bool:

        self._pico_walking_velocity_filter.reset()
        self._next_pico_walking_update_s = 0.0
        self._pico_walking_zero_pending = False
        self._pico_walking_suppressed_by_robot_joystick = False
        if (
            not send_stop
            or not self.remoteControlService.pico_walking_control_enabled
        ):
            return True
        stopped = self._send_walking_velocity_rpc((0.0, 0.0, 0.0))
        self._pico_walking_zero_pending = not stopped
        return stopped

    def _tick_pico_walking_velocity(
        self,
        *,
        now_s: float | None = None,
    ) -> bool:

        if (
            self.control_state != BoosterControlState.WALKING
            or not self.remoteControlService.pico_walking_control_enabled
            or getattr(self, "_shutdown_requested", False)
            or self._exit_safety_state()["robot_mode"] in {0, 1}
        ):
            return False
        now = time.monotonic() if now_s is None else float(now_s)
        if now < self._next_pico_walking_update_s:
            return False
        self._next_pico_walking_update_s = now + 1.0 / float(
            self.remoteControlService.pico_walking_config.update_rate_hz
        )
        robot_joystick_active = (
            self.remoteControlService.robot_joystick_axes_active()
        )
        if robot_joystick_active:
            if not self._pico_walking_suppressed_by_robot_joystick:
                had_pico_command = (
                    self._pico_walking_velocity_filter.value
                    != (0.0, 0.0, 0.0)
                    or self._pico_walking_zero_pending
                )
                stopped = self._reset_pico_walking_velocity(
                    send_stop=had_pico_command
                )
                self._pico_walking_suppressed_by_robot_joystick = True
                return had_pico_command and stopped
            elif self._pico_walking_zero_pending:
                if self._send_walking_velocity_rpc((0.0, 0.0, 0.0)):
                    self._pico_walking_zero_pending = False
                    return True
            return False
        if self._pico_walking_suppressed_by_robot_joystick:
            self._pico_walking_suppressed_by_robot_joystick = False
            self._pico_walking_velocity_filter.reset()
        epsilon = float(
            self.remoteControlService.pico_walking_config.stop_epsilon
        )
        if (
            self._pico_walking_zero_pending
            and self._pico_walking_velocity_filter.value == (0.0, 0.0, 0.0)
        ):
            if self._send_walking_velocity_rpc((0.0, 0.0, 0.0)):
                self._pico_walking_zero_pending = False
                return True
            return False
        target = self.remoteControlService.get_pico_walking_velocity_target(
            now_s=now
        )
        command = self._pico_walking_velocity_filter.step(target, now_s=now)
        is_zero = all(abs(value) <= epsilon for value in command)
        if not is_zero:
            self._pico_walking_zero_pending = True
            return self._send_walking_velocity_rpc(command)
        if not self._pico_walking_zero_pending:
            return False
        if self._send_walking_velocity_rpc((0.0, 0.0, 0.0)):
            self._pico_walking_zero_pending = False
            return True
        return False

    def _discard_mode_future(self, intent: dict, key: str) -> None:
        future = intent.pop(key, None)
        if future is None:
            return
        # Local cancellation only; the vendor may still execute a sent request.
        self.rpc_service_client.remove_pending_request(future)
        if not future.done():
            future.cancel()

    def _request_robot_mode(self, target_mode: int) -> dict:
        previous = getattr(self, "_mode_request", None)
        if previous and previous["target"] == target_mode and previous["status"] == "pending":
            return previous
        force_change = bool(previous and previous["sent"] and previous["status"] == "pending")
        if previous:
            for key in ("query", "change"):
                self._discard_mode_future(previous, key)
        self._mode_intent_seq = getattr(self, "_mode_intent_seq", 0) + 1
        intent = dict(token=self._mode_intent_seq, target=int(target_mode),
                      status="pending", initial=True, sent=False,
                      force_change=force_change, next_query=0.0)
        self._mode_request = intent
        return intent

    def _service_mode_request(self) -> None:
        # Bounded poll; the existing executor advances futures. Acceptance is not
        # confirmation: only successful GET_STATUS/current_mode grants ownership.
        t = getattr(self, "_mode_request", None)
        if not t or t["status"] != "pending":
            return
        chord = getattr(self, "_shutdown_source", None) == "chord"
        if chord and getattr(self, "_shutdown_complete", False):
            return
        if (getattr(self, "_shutdown_requested", False) and not t["sent"]
                and not (chord and t["target"] == _RobotModeInt.kWalking)):
            return  # Only B+X may submit its one Walking intent during shutdown.
        if not rclpy.ok():
            t["status"] = "ros_shutdown"
            for key in ("query", "change"):
                self._discard_mode_future(t, key)
            self.logger.error("ROS stopped; mode %s NOT confirmed", t["target"])
            return
        client = self.rpc_service_client
        if client is None:
            return
        now = time.monotonic()
        for key in ("change", "query"):
            future = t.get(key)
            if future is None:
                continue
            if not future.done() and now < t[key + "_deadline"]:
                continue
            done = future.done()
            self._discard_mode_future(t, key)
            if not done:
                self.logger.warning("Mode %s RPC timed out; result unknown", key)
                if key == "query":
                    t["initial"] = False
                continue
            try:
                if future.exception() is not None:
                    raise future.exception()
                response = future.result()
                if response is None:
                    raise ValueError("RPC returned None")
                # Same contract as _call_booster_rpc: msg.status == 0.
                if response.msg.status != 0:
                    if key == "change":
                        t["status"] = "rejected"
                        self.logger.error("CHANGE_MODE rejected: status=%s body=%s",
                                          response.msg.status, response.msg.body)
                        self._discard_mode_future(t, "query")
                        return
                    t["initial"] = False
                    continue
                if key == "query":
                    body = json.loads(response.msg.body)
                    if chord:
                        mode = body["current_mode"]
                        if type(mode) is not int:
                            raise ValueError("current_mode is not an integer")
                        self._service_chord_shutdown(mode)
                        if getattr(self, "_shutdown_complete", False):
                            return
                    mode = int(body["current_mode"])
                    if mode == t["target"] and not (t["initial"] and t["force_change"]):
                        t["status"] = "confirmed"
                        self._discard_mode_future(t, "change")
                        self.logger.info("Robot mode %s confirmed (intent %s)", mode, t["token"])
                        return
                    t["initial"] = False
            except Exception as exc:
                self.logger.warning("Mode %s response unknown: %s", key, exc)
                if key == "query":
                    t["initial"] = False
        if chord and t["target"] != _RobotModeInt.kWalking:
            # Drain a submitted Custom CHANGE, but never resume its transition.
            # Local cancellation cannot recall a vendor request already sent.
            return
        if not client.service_is_ready():
            return
        if not t["initial"] and not t["sent"]:
            entry = getattr(self, "_tracking_transition", None)
            if t["target"] == _RobotModeInt.kCustom and entry:
                if not self._wait_for_fresh_low_state(timeout_s=0.0):
                    return
                if not entry.get("pvt_only") and not self.policy_command_ready.is_set():
                    return
                if entry.get("preload_required") and not self._publish_tracking_preload(entry):
                    return
            # Mark before send: exceptions cannot establish remote non-delivery.
            # A pre-send status snapshot cannot confirm a submitted CHANGE.
            self._discard_mode_future(t, "query")
            t["next_query"] = 0.0
            t["sent"] = True
            try:
                request = RpcService.Request()
                request.msg = BoosterApiReqMsg()
                request.msg.api_id = _LOC_API_CHANGE_MODE
                request.msg.body = json.dumps({"mode": t["target"]})
                t["change"] = client.call_async(request)
                t["change_deadline"] = now + 3.0
            except Exception as exc:
                self.logger.warning("CHANGE_MODE delivery unknown, only querying: %s", exc)
        # Wait for acceptance (or the one RPC deadline) before a fresh query.
        # Never confirm/cancel an outstanding CHANGE using an earlier snapshot.
        if "change" not in t and "query" not in t and now >= t["next_query"]:
            t["next_query"] = now + 0.25
            try:
                request = RpcService.Request()
                request.msg = BoosterApiReqMsg()
                request.msg.api_id = _LOC_API_GET_STATUS
                request.msg.body = ""
                t["query"] = client.call_async(request)
                t["query_deadline"] = now + 3.0
            except Exception as exc:
                self.logger.warning("GET_STATUS unavailable: %s", exc)
                t["initial"] = False

    def start_custom_mode_conditionally(self) -> tuple[bool, bool]:

        return self._enter_pvt_mode(), False

    def _start_policy_session(self) -> bool:

        if self.inference_process is not None:
            self._stop_inference_process()
        self.policy_stop_event = mp.Event()
        self.reference_tracking_active = mp.Event()
        self.offline_playback_active = mp.Event()
        self.offline_entry_cycle = int((getattr(self, "_tracking_transition", None) or {}).get("cycle", 0))
        self._reference_entry_published = None
        self.offline_replay_finished = mp.Event()
        self.policy_command_ready = mp.Event()
        self.policy_action_updated.clear()
        self.policy_output_published.clear()
        self.remoteControlService.clear_exit_rl_gait()
        self.inference_process = mp.Process(
            target=BoosterRobotPortal.inference_process_func,
            args=(self.cfg, self),
            daemon=True,
        )
        self.inference_process.start()
        self.logger.info("Inference policy session started")
        return True

    def _enter_pvt_mode(self) -> bool:
        if getattr(self, "_walking_transition", None) or getattr(self, "_shutdown_requested", False):
            return False
        if not getattr(self, "_tracking_transition", None):
            self._begin_tracking_transition(False, pvt_only=True)
        return self.control_state == BoosterControlState.PVT_HOLD

    _TRANSITION_TIMEOUT_S = 45.0

    def _begin_tracking_transition(self, was_pvt: bool, *, pvt_only: bool = False) -> None:
        if (getattr(self, "_tracking_transition", None)
                or getattr(self, "_walking_transition", None)
                or getattr(self, "_shutdown_requested", False)):
            return
        if not pvt_only and self.control_state not in {BoosterControlState.WALKING, BoosterControlState.PVT_HOLD}:
            return
        cycle = 0
        if not pvt_only:
            from booster_deploy.telemetry.run_config import reserve_offline_playback_start
            if not reserve_offline_playback_start(os.getenv("BOOSTER_TELEMETRY_DIR") or os.getenv("BOOSTER_DIAG_DIR", "")):
                self._diag_ctrl.event('tracking_rejected',dict(reason='telemetry_trajectory_switch_in_progress'))
                return
            self._diag_ctrl.event('tracking_requested',dict(reason='accepted_A'))
            from booster_deploy.utils.offline_replay import offline_replay_enabled
            if offline_replay_enabled():
                from booster_deploy.utils.replay_control import write_prepare
                cycle = write_prepare(os.getenv("BOOSTER_DIAG_DIR"))
                if not cycle:
                    self._diag_ctrl.event('tracking_rejected', dict(reason='offline_prepare_write_failed'))
                    from booster_deploy.telemetry.run_config import clear_offline_playback_start_pending
                    clear_offline_playback_start_pending(os.getenv('BOOSTER_TELEMETRY_DIR') or os.getenv('BOOSTER_DIAG_DIR', ''))
                    return
                self._diag_ctrl.event('offline_prepare', dict(cycle=cycle))
            else:
                from booster_deploy.utils.replay_control import write_start
                write_start(os.getenv("BOOSTER_DIAG_DIR"))
        self._tracking_transition = dict(
            cycle=cycle, phase="low_state", deadline_s=time.monotonic() + self._TRANSITION_TIMEOUT_S,
            was_walking=self.control_state == BoosterControlState.WALKING,
            was_pvt=bool(was_pvt), pvt_only=pvt_only)
        self.control_state = (BoosterControlState.ARMING_PVT if pvt_only
                              else BoosterControlState.ARMING_TRACKING)

    def _abort_tracking_transition(self, reason: str) -> None:
        from booster_deploy.telemetry.run_config import clear_offline_playback_start_pending
        clear_offline_playback_start_pending(os.getenv("BOOSTER_TELEMETRY_DIR") or os.getenv("BOOSTER_DIAG_DIR", ""))
        transition = getattr(self, '_tracking_transition', None) or {}
        # A preload wait may have queued an intent without submitting CHANGE.
        # Do not let a later mode-service tick send it after startup aborts.
        intent = transition.get('intent')
        if (transition.get('preload_required') and intent
                and intent is getattr(self, '_mode_request', None)
                and intent['status'] == 'pending' and not intent.get('sent')):
            intent['status'] = 'cancelled'
            for key in ('query', 'change'):
                self._discard_mode_future(intent, key)
        cycle = int(transition.get('cycle', 0))
        if cycle:
            self._diag_ctrl.event('offline_prepare_cancelled',dict(cycle=cycle,reason=reason))
            active=getattr(self,'reference_tracking_active',None)
            if active is not None: active.clear()
            self.offline_playback_active.clear()
            self._set_hold_from_current_state()
            self._set_output_mode('hold')
            self.policy_stop_event.set()
        self._diag_ctrl.event('tracking_failed',dict(reason=reason))
        from booster_deploy.utils.replay_control import write_abort
        write_abort(os.getenv("BOOSTER_DIAG_DIR"))
        self._tracking_transition = None
        self.logger.error("Tracking transition failed: %s; current output retained", reason)
        # No invented Walking state or implicit fallback mode request.
        self.control_state = BoosterControlState.FAULT

    def _service_tracking_transition(self) -> None:
        t = getattr(self, "_tracking_transition", None)
        if not t:
            return
        if getattr(self,'_shutdown_requested',False):
            self._abort_tracking_transition('shutdown cancelled offline startup')
            return
        now = time.monotonic()
        phase = t["phase"]
        if now > t["deadline_s"] and (t.get('cycle') or t.get('preload_required') or phase in {"low_state", "playback_lock", "policy_ready", "entry_transition"}):
            self._abort_tracking_transition("preparation timeout in " + phase)
            return
        if phase == "entry_transition":
            if not self._wait_for_fresh_low_state(timeout_s=0.0) or self.inference_process is None or not self.inference_process.is_alive():
                self._abort_tracking_transition('offline entry lost fresh robot state or policy worker')
                return
            from booster_deploy.utils.replay_control import read_abort,read_prepare_ns
            if (read_abort(os.getenv('BOOSTER_DIAG_DIR'),t['cycle']) or
                    read_prepare_ns(os.getenv('BOOSTER_DIAG_DIR')) != t['cycle']):
                self._abort_tracking_transition('offline preparation cancelled')
                return
            action = self.synced_action.read()[0]
            cycle = int(action['reference_entry_cycle'])
            session = bytes(action['policy_session']).decode()
            published = getattr(self, '_reference_entry_published', None)
            if (cycle != t['cycle'] or session != t['session']
                    or published is None or published[:2] != (cycle, session)
                    or not np.isfinite(action['dof_target']).all()):
                return
            from booster_deploy.telemetry.run_config import commit_offline_playback_start, offline_playback_provenance
            from booster_deploy.utils.replay_control import read_input, write_start
            run_dir = os.getenv('BOOSTER_DIAG_DIR')
            directory = os.getenv('BOOSTER_TELEMETRY_DIR') or run_dir or ''
            provenance = offline_playback_provenance(directory)
            if getattr(self.cfg.policy, 'pico_input_mode', '') != 'recording':
                selection = read_input(run_dir)
                expected = provenance.get('trajectory_path')
                if expected and (selection is None or selection.get('path') != expected):
                    self._abort_tracking_transition('offline publisher selection does not match the selected trajectory')
                    return
                if selection is not None:
                    provenance['input_selection'] = selection
            if not commit_offline_playback_start(directory, nonblocking=True):
                return
            if not write_start(run_dir, token=cycle):
                self._abort_tracking_transition('offline play commit failed or cancelled')
                return
            self._diag_ctrl.event('offline_playback_started', dict(cycle=cycle, entry_output_tick=published[2],
                policy_session_id=session, **provenance))
            self.offline_playback_active.set()
            self.control_state = BoosterControlState.TRACKING
            self._clear_offline_replay_finished()
            self._tracking_transition = None
            self.logger.info('Offline tracking started after entry output publication')
            return
        if phase == "low_state":
            if t.get("was_walking") and not self._reset_pico_walking_velocity(send_stop=True):
                return
            if self._wait_for_fresh_low_state(timeout_s=0.0):
                if t.get("cycle") and getattr(self.cfg.policy, "pico_input_mode", "") != "recording":
                    from booster_deploy.utils.replay_control import read_input
                    if read_input(os.getenv("BOOSTER_DIAG_DIR")) is None:
                        return
                if t.get("pvt_only"):
                    t["start"] = self._set_hold_from_current_state()
                    t["target"], t["stiffness"], t["damping"] = self._prepare_target()
                    t["phase"] = "robot_mode"
                elif self._start_policy_session():
                    t["phase"] = "playback_lock"
                else:
                    self._abort_tracking_transition("policy session failed to start")
            return
        if phase == "playback_lock":
            if t.get("cycle"):
                t["phase"] = "policy_ready"
                return
            from booster_deploy.telemetry.run_config import commit_offline_playback_start
            if commit_offline_playback_start(
                    os.getenv("BOOSTER_TELEMETRY_DIR") or os.getenv("BOOSTER_DIAG_DIR", ""),
                    nonblocking=True):
                t["phase"] = "policy_ready"
            return
        if phase == "policy_ready":
            if self.policy_command_ready.is_set() and self._wait_for_fresh_low_state(timeout_s=0.0):
                if t.get("cycle"):
                    action = self.synced_action.read()[0]
                    if not bool(action["reference_entry_ready"]):
                        return
                    t["session"] = bytes(action["policy_session"]).decode()
                    if not t["session"]:
                        return
                t["preload_required"] = bool(t.get("was_walking") and not t.get("was_pvt"))
                if t["preload_required"] and not self._publish_tracking_preload(t):
                    return
                t["phase"] = "robot_mode"
            return
        if phase == "robot_mode":
            if (t.get("preload_required")
                    and t.get("intent", {}).get("status", "pending") == "pending"
                    and not self._publish_tracking_preload(t)):
                return
            if "intent" not in t:
                t["intent"] = self._request_robot_mode(_RobotModeInt.kCustom)
            intent = t["intent"]
            if intent["status"] == "confirmed":
                t["phase"] = "joint_ctrl"
            elif intent["status"] != "pending":
                self._abort_tracking_transition("Custom " + intent["status"])
            return
        intent = getattr(self, "_mode_request", None)
        if not intent or intent["target"] != _RobotModeInt.kCustom or intent["status"] != "confirmed":
            return
        if phase == "joint_ctrl":
            if not self._wait_for_joint_ctrl_subscriber(timeout_s=0.0):
                return
            if t.get("pvt_only"):
                self._set_output_mode("hold")
                t["ramp_start"] = now
                t["phase"] = "pvt_ramp"
            elif t["was_pvt"]:
                t["start"] = self._hold_joint_pos.copy()
                t["ramp_start"] = now
                t["phase"] = "handoff"
            else:
                t["phase"] = "policy_output"
            return
        if phase in {"pvt_ramp", "handoff"}:
            duration = 2.0 if phase == "pvt_ramp" else 0.3
            ratio = min(1.0, max(0.0, (now - t["ramp_start"]) / duration))
            if phase == "handoff":
                action = self.synced_action.read()[0]
                target, stiffness, damping = (action["dof_target"], action["stiffness"], action["damping"])
            else:
                target, stiffness, damping = t["target"], t["stiffness"], t["damping"]
            self._set_hold_target((1.0 - ratio) * t["start"] + ratio * target, stiffness, damping)
            self._publish_hold_target()
            if ratio >= 1.0:
                if phase == "pvt_ramp":
                    self.control_state = BoosterControlState.PVT_HOLD
                    self._tracking_transition = None
                else:
                    t["phase"] = "policy_output"
            return
        if phase == "policy_output":
            if not t.get("output_enabled"):
                self.policy_output_published.clear()
                self._set_output_mode("policy")
                self.policy_action_updated.set()
                t["output_enabled"] = True
            if self.policy_output_published.is_set():
                if t.get("cycle"):
                    self.reference_tracking_active.set()
                    t["phase"] = "entry_transition"
                    return
                self.control_state = BoosterControlState.TRACKING
                self._clear_offline_replay_finished()
                self.reference_tracking_active.set()
                self._tracking_transition = None
                self.logger.info("Tracking policy session active")

    def _enter_tracking_mode(self) -> tuple[bool, ControlEventType | None]:
        self._begin_tracking_transition(self.control_state == BoosterControlState.PVT_HOLD)
        return self.control_state == BoosterControlState.TRACKING, None

    def _clear_offline_replay_finished(self) -> None:
        """Clear the replay-end flag, tolerating portals built without __init__."""

        event = getattr(self, "offline_replay_finished", None)
        if event is not None:
            event.clear()

    def _begin_reference_exit(self) -> None:
        if getattr(self, '_reference_exit', None) or getattr(self, '_walking_transition', None):
            return
        if getattr(self, '_shutdown_source', None) == 'chord':
            return
        process = self.inference_process
        if (self.control_state not in {BoosterControlState.TRACKING, BoosterControlState.TRACKING_PAUSED}
                or process is None or not process.is_alive() or self._output_mode != 'policy'):
            self._leave_to_walking()
            return
        import uuid
        action = self.synced_action.read()[0]
        session = bytes(action['policy_session']).decode()
        if not session:
            self.logger.warning('Reference exit unavailable: no active policy session; using Walking flow')
            self._leave_to_walking()
            return
        cfg = self.cfg.policy
        duration = float(getattr(cfg, 'resume_transition_s', 1.) if getattr(cfg, 'input_path', '') == 'smpl'
                         else getattr(cfg, 'rt_reference_transition_s', 1.5))
        token = uuid.uuid4().hex
        self._reference_exit = dict(token=token, session=session, deadline=time.monotonic()+max(3.,duration+2.))
        cmd = self.synced_command.read()
        cmd[0]['reference_exit_token'] = token.encode()
        cmd[0]['reference_exit_session'] = session.encode()
        self.synced_command.write(cmd)
        self.control_state = BoosterControlState.EXITING_TO_WALK
        self.logger.info('Reference input → default (%.2fs); Custom/policy output retained', duration)

    def _service_reference_exit(self) -> None:
        transition = getattr(self, '_reference_exit', None)
        if not transition:
            return
        if getattr(self, '_shutdown_source', None) == 'chord':
            self._reference_exit = None
            return
        action = self.synced_action.read()[0]
        complete = (bytes(action['reference_exit_token']).decode() == transition['token']
                    and bytes(action['policy_session']).decode() == transition['session']
                    and np.isfinite(action['dof_target']).all()
                    and getattr(self, '_reference_exit_published', None) == (transition['token'], transition['session']))
        if complete:
            self._reference_exit = None
            self._leave_to_walking()
        elif (self.inference_process is None or not self.inference_process.is_alive()
              or time.monotonic() >= transition['deadline']):
            self.logger.warning('Reference exit failed/timed out; falling back to existing Walking flow')
            self._reference_exit = None
            self._leave_to_walking()

    def _leave_to_walking(self) -> bool:
        if getattr(self, "_shutdown_source", None) == "chord":
            return False  # B+X owns mode intent; queued B must not replace it.
        from booster_deploy.telemetry.run_config import clear_offline_playback_start_pending
        clear_offline_playback_start_pending(os.getenv("BOOSTER_TELEMETRY_DIR") or os.getenv("BOOSTER_DIAG_DIR", ""))
        self._diag_ctrl.event('return_to_walk_requested')
        pending=getattr(self,'_tracking_transition',None) or {}
        if pending.get('cycle'):
            self._diag_ctrl.event('offline_prepare_cancelled',dict(cycle=pending['cycle'],reason='return_to_walk'))
            from booster_deploy.utils.replay_control import write_abort
            write_abort(os.getenv('BOOSTER_DIAG_DIR'))
        if not getattr(self, "_walking_transition", None):
            self._tracking_transition = None
            self._walking_transition = self._request_robot_mode(_RobotModeInt.kWalking)
            self.control_state = BoosterControlState.EXITING_TO_WALK
        self._service_walking_transition()
        return self.control_state == BoosterControlState.WALKING

    def _service_walking_transition(self) -> None:
        if getattr(self, "_shutdown_source", None) == "chord":
            return  # B+X releases output only on the new SDK response.
        intent = getattr(self, "_walking_transition", None)
        if not intent:
            return
        if intent["status"] == "confirmed":
            self._set_output_mode("none")
            from booster_deploy.telemetry.run_config import release_offline_playback
            directory = os.getenv('BOOSTER_TELEMETRY_DIR') or os.getenv('BOOSTER_DIAG_DIR', '')
            if not release_offline_playback(directory):
                return  # Do not advertise Walking with an unreleased playback reservation.
            self._stop_inference_process()
            self.control_state = BoosterControlState.WALKING
            self._walking_transition = None
            self._publish_closed_hand_commands()
            self._reset_pico_walking_velocity(send_stop=False)
            self.logger.info("Walking confirmed; policy output released")
        elif intent["status"] != "pending":
            self.control_state = BoosterControlState.FAULT
            if not intent.get("failure_reported"):
                self.logger.error("Walking %s; current output/session retained", intent["status"])
                intent["failure_reported"] = True
            # Keep the failed intent to block A; B must not auto-retry it.

    def _handle_control_event(self, event: ControlEventType) -> None:

        state = self.control_state
        if event == ControlEventType.RETURN_TO_WALK:
            if state != BoosterControlState.WALKING:
                from booster_deploy.utils.replay_control import write_abort
                write_abort(os.getenv('BOOSTER_DIAG_DIR'))
                self._begin_reference_exit()
            return
        if event == ControlEventType.ENTER_PVT:
            if state != BoosterControlState.PVT_HOLD:
                self._enter_pvt_mode()
            return
        if event == ControlEventType.ENTER_TRACKING:
            if state in {BoosterControlState.WALKING, BoosterControlState.PVT_HOLD}:
                self._begin_tracking_transition(state == BoosterControlState.PVT_HOLD)
            return
        if event == ControlEventType.TOGGLE_TRACKING_PAUSE:
            if state in {
                BoosterControlState.TRACKING,
                BoosterControlState.TRACKING_PAUSED,
            }:
                self._pause_toggle_seq += 1
                self.control_state = (
                    BoosterControlState.TRACKING_PAUSED
                    if state == BoosterControlState.TRACKING
                    else BoosterControlState.TRACKING
                )

    def start_rl_gait_conditionally(self, *, start_requested: bool = False):
        if start_requested:
            self._handle_control_event(ControlEventType.ENTER_TRACKING)
        return self.control_state == BoosterControlState.TRACKING

    def _activate_direct_rl_control(self) -> bool:
        return self._enter_tracking_mode()[0]

    def _stop_inference_process(self) -> bool:
        process = self.inference_process
        if process is None:
            return True

        self.policy_stop_event.set()
        self.reference_tracking_active.clear()

        if process.is_alive():
            self.logger.info("Waiting for inference process...")
            process.join(timeout=2.0)
        if process.is_alive():
            self.logger.warning(
                "Inference process did not stop, terminating..."
            )
            process.terminate()
            process.join(timeout=1.0)
        if process.is_alive():
            self.logger.error("Inference process is still alive")
            return False

        process.join(timeout=0.0)
        self.inference_process = None
        self.policy_command_ready.clear()
        return True

    def _service_chord_shutdown(self, mode: int) -> None:
        """Consume a successful post-request SDK reply, never a cached mode."""
        if getattr(self, "_shutdown_complete", False):
            return
        if mode in {0, 1, 2}:
            # Damping/PVT are already safe to exit; never force them into Walk.
            self.logger.info("B+X shutdown: SDK current_mode=%s confirmed", mode)
            self._set_output_mode("none")
            self._stop_inference_process()
            self._confirm_shutdown()
            self._request_stack_shutdown("chord")
        elif mode == _RobotModeInt.kCustom:
            changing = getattr(self, "_mode_request", None)
            if (not changing or changing["target"] != _RobotModeInt.kWalking
                    or (not changing["sent"] and changing["status"] != "pending")):
                self.logger.info("B+X shutdown pending: Custom; requesting Walking once")
                self._request_robot_mode(_RobotModeInt.kWalking)
        else:
            self.logger.warning("B+X shutdown pending: SDK current_mode=%s unknown; current control retained", mode)

    def _service_exit_mode_query(self) -> None:
        """Read-only actual mode probe; never infer vendor mode from our FSM."""
        if not rclpy.ok() or self.rpc_service_client is None:
            return
        chord = getattr(self, "_shutdown_source", None) == "chord"
        if chord:
            changing = getattr(self, "_mode_request", None)
            if getattr(self, "_shutdown_complete", False):
                return
            if changing and ("change" in changing or (
                    changing["target"] == _RobotModeInt.kWalking
                    and changing["status"] == "pending")):
                return  # Existing mode intent owns the post-CHANGE query.
        now = time.monotonic()
        probe = getattr(self, "_exit_mode_query", None)
        if probe:
            future = probe["future"]
            if not future.done() and now < probe["deadline"]:
                return
            self._exit_mode_query = None
            try:
                if not future.done() or future.exception() is not None:
                    raise ValueError("GET_STATUS timed out or failed")
                response = future.result()
                if response is None or response.msg.status != 0:
                    raise ValueError("GET_STATUS rejected")
                mode = json.loads(response.msg.body)["current_mode"]
                if type(mode) is not int:
                    raise ValueError("current_mode is not an integer")
                self._exit_mode_sample = (mode, probe["sent_ns"])
                if chord:
                    self._service_chord_shutdown(mode)
            except Exception as exc:
                self._exit_mode_sample = None
                if chord:
                    self.logger.warning("B+X shutdown pending: %s; current control retained", exc)
            finally:
                self._discard_mode_future({"probe": future}, "probe")
        if chord and (getattr(self, "_shutdown_complete", False) or (
                getattr(self, "_mode_request", None)
                and self._mode_request["target"] == _RobotModeInt.kWalking
                and self._mode_request["status"] == "pending")):
            return
        if now < getattr(self, "_exit_mode_next_query", 0.0):
            return
        if not self.rpc_service_client.service_is_ready():
            return
        self._exit_mode_next_query = now + 0.25
        try:
            request = RpcService.Request()
            request.msg = BoosterApiReqMsg()
            request.msg.api_id = _LOC_API_GET_STATUS
            request.msg.body = ""
            self._exit_mode_query = dict(
                future=self.rpc_service_client.call_async(request),
                sent_ns=time.monotonic_ns(), deadline=now + 1.0,
            )
        except Exception as exc:
            self._exit_mode_sample = None
            if chord:
                self.logger.warning("B+X GET_STATUS unavailable; shutdown pending: %s", exc)

    def _exit_safety_state(self) -> dict:
        sample = getattr(self, "_exit_mode_sample", None)
        fresh = bool(sample and 0 <= time.monotonic_ns() - sample[1] <= 2_000_000_000)
        return dict(robot_mode=sample[0] if fresh else None,
                    robot_mode_mono_ns=sample[1] if fresh else None,
                    shutdown_confirmed=bool(getattr(self, "_shutdown_complete", False)))

    def _confirm_shutdown(self) -> None:
        self._shutdown_complete = True
        diag = getattr(self, "_diag_ctrl", None)
        if diag is not None:
            diag.emit("shutdown confirmed", alerts=(),
                      extra={"state": {"state": self.control_state.value,
                                       **self._exit_safety_state()}}, force=True)
        self.exit_event.set()

    def _request_normal_shutdown(self) -> None:
        # May be called by the keyboard reader/signal handler. Main loop owns RPCs.
        if not getattr(self, "_shutdown_requested", False):
            self._shutdown_intent = dict(started_ns=time.monotonic_ns())
        self._shutdown_requested = True

    def _service_shutdown(self) -> None:
        if not getattr(self, "_shutdown_requested", False):
            return
        if getattr(self, "_shutdown_complete", False):
            return
        if getattr(self, "_shutdown_source", None) == "chord":
            if not self._shutdown_intent.get("refusal_reported"):
                self.logger.info("B+X shutdown pending: waiting for SDK GET_STATUS; current control retained")
                self._shutdown_intent["refusal_reported"] = True
            return
        intent = self._shutdown_intent
        sample = getattr(self, "_exit_mode_sample", None)
        if (not sample or sample[1] < intent["started_ns"]
                or self._exit_safety_state()["robot_mode"] not in {0, 1, 2}):
            # Custom (3), stale and unknown are never permission to kill or switch.
            if not intent.get("refusal_reported"):
                self.logger.warning("Shutdown refused/pending: fresh actual mode 0/1/2 required; no automatic mode change")
                intent["refusal_reported"] = True
            return
        # A submitted mode CHANGE can still execute remotely after cancellation.
        changing = getattr(self, "_mode_request", None)
        if changing and changing["status"] == "pending" and changing["sent"]:
            return
        if sample[0] in {0, 1}:
            self._shutdown_zero_status = "not_required"
            self.logger.info("Normal shutdown: fresh actual mode %s; no CHANGE/MOVE", sample[0])
            self._confirm_shutdown()
            return
        status = getattr(self, "_shutdown_zero_status", None)
        if status in {"confirmed", "failed"}:
            return
        pending = getattr(self, "_velocity_rpc", None)
        now = time.monotonic()
        if status is None:
            # Drain an earlier MOVE before submitting this operation's own zero.
            # Neither its ACK nor local cancellation proves our zero was applied.
            if pending:
                if not pending["future"].done() and now < pending["deadline"]:
                    return
                self._discard_mode_future({"old_move": pending["future"]}, "old_move")
                self._velocity_rpc = None
            self._call_booster_rpc(
                _LOC_API_MOVE, {"vx": 0.0, "vy": 0.0, "vyaw": 0.0},
                timeout_sec=float(self.remoteControlService.pico_walking_config.rpc_timeout_s),
            )
            if getattr(self, "_velocity_rpc", None):
                self._shutdown_zero_status = "pending"
            return
        future = pending["future"]
        if not future.done() and now < pending["deadline"]:
            return
        self._shutdown_zero_status = "failed"
        try:
            if not future.done():
                raise TimeoutError("MOVE zero timed out; remote result unknown")
            if future.exception() is not None:
                raise future.exception()
            response = future.result()
            if response is None or response.msg.status != 0:
                raise ValueError("MOVE zero rejected or returned no response")
            self._shutdown_zero_status = "confirmed"
        except Exception as exc:
            self.logger.error("Normal shutdown pending: zero velocity NOT confirmed: %s", exc)
        finally:
            self._discard_mode_future({"zero_move": future}, "zero_move")
            self._velocity_rpc = None
        if self._shutdown_zero_status == "confirmed":
            self.logger.info("Normal shutdown: fresh Walking and zero velocity ACK confirmed")
            self._confirm_shutdown()

    # The outer lifecycle owns writer finalization after producer cleanup.
    def cleanup(self, *, force: bool = False) -> bool:
        from booster_deploy.telemetry.runtime import get_writer,close_process
        if getattr(self,'_cleanup_done',False): return True
        producers=[(name,getattr(self,name,None)) for name in (
            'inference_process','low_state_thread',
            'low_cmd_process','policy_output_thread')]
        error=None
        result=False
        try:
            result=self._cleanup_resources(force=force)
        except BaseException as exc:
            error=repr(exc)
            raise
        finally:
            if getattr(self,'_telemetry_cleanup_started',False):
                alive=[name for name,producer in producers if producer is not None and producer.is_alive()]
                reason=error or ('cleanup producers still alive: '+','.join(alive) if alive else getattr(self,'_telemetry_cleanup_error',None))
                writer=get_writer()
                if reason and hasattr(writer,'mark_incomplete'): writer.mark_incomplete(reason)
                status=close_process(permanent=True)
                self._cleanup_done=bool(result and not reason and status['state'] in ('closed','not_started') and not status.get('incomplete'))
        return self._cleanup_done if getattr(self,'_telemetry_cleanup_started',False) else result

    def _cleanup_resources(self, *, force: bool = False) -> bool:
        if self._cleanup_done:
            return True
        if not force and rclpy.ok():
            self._request_normal_shutdown()
            self._service_exit_mode_query()
            self._service_mode_request()
            self._service_shutdown()
            if not getattr(self, "_shutdown_complete", False):
                self.logger.warning("Cleanup pending: fresh actual mode 0/1/2 (Walking also needs zero ACK) required")
                return False
        if not getattr(self, "_shutdown_complete", False):
            self.logger.error("Forced/ROS cleanup: actual safe mode NOT confirmed; releasing local resources only")
        intent = getattr(self, "_mode_request", None)
        if intent:
            for key in ("query", "change"):
                self._discard_mode_future(intent, key)
            if intent["status"] == "pending":
                intent["status"] = "forced_unconfirmed"
        probe = getattr(self, "_exit_mode_query", None)
        if probe:
            self._discard_mode_future({"probe": probe["future"]}, "probe")
            self._exit_mode_query = None
        velocity = getattr(self, "_velocity_rpc", None)
        if velocity:
            self._discard_mode_future({"velocity": velocity["future"]}, "velocity")
            self._velocity_rpc = None
        self._telemetry_cleanup_started = True
        self.logger.info("Doing cleanup...")
        self.is_running = False
        self.exit_event.set()
        self.policy_action_updated.set()

        self._stop_inference_process()

        # Low-state callbacks read hand input and may use the retarget socket.
        # Stop them before closing those resources or shutting ROS down.
        if self.low_state_thread is not None and self.low_state_thread.is_alive():
            self.low_state_thread.join(timeout=2.0)

        try:
            self.remoteControlService.close()
        except Exception as e:
            self._telemetry_cleanup_error=repr(e)
            self.logger.error(f"Error closing remote control: {e}")

        try:
            if self.hand_retarget_subscriber is not None:
                self.hand_retarget_subscriber.close()
        except Exception as e:
            self._telemetry_cleanup_error=repr(e)
            self.logger.error(f"Error closing hand retarget subscriber: {e}")

        if self.low_cmd_process is not None and self.low_cmd_process.is_alive():
            self.logger.info("Waiting for low cmd publisher process...")
            self.low_cmd_process.join(timeout=2.0)
            if self.low_cmd_process.is_alive():
                self.logger.warning(
                    "Low cmd publisher process did not stop, terminating...")
                self.low_cmd_process.terminate()
                self.low_cmd_process.join(timeout=1.0)

        try:
            thread = self.policy_output_thread
            if thread is not None and thread.is_alive():
                thread.join(timeout=2.0)

        except Exception as e:
            self._telemetry_cleanup_error=repr(e)
            self.logger.error(f"Error waiting for low state thread: {e}")

        if rclpy.ok():
            rclpy.shutdown()

        self.logger.info("Local cleanup complete (not a robot mode acknowledgement)")

        diag = getattr(self, "_diag_ctrl", None)
        if diag is not None:
            diag.close()

        for name, metric in self.metrics.items():
            stats = metric.compute()
            print(
                f"METRICS {name}: count={stats['count']}, "
                f"freq={stats['freq_hz']:.3f}Hz, "
                f"mean_period={stats['mean_period_s']}, "
                f"min={stats['min_period_s']}, max={stats['max_period_s']}"
            )
        return True

    def _request_stack_shutdown(self, reason: str) -> None:
        """Spawn the ordered shutdown script: diagnostics → stop panes → kill session.

        Runs detached (own session) so killing the tmux session cannot take the
        caller with it, and so the script survives our own Ctrl-C.
        """
        import subprocess
        from pathlib import Path

        if reason == "chord" and not getattr(self, "_shutdown_complete", False):
            if getattr(self, "_shutdown_source", None) == "chord":
                return
            self._shutdown_source = "chord"
            self._reference_exit = None
            self._request_normal_shutdown()
            pending=getattr(self,'_tracking_transition',None) or {}
            if pending.get('cycle'):
                self._diag_ctrl.event('offline_prepare_cancelled',dict(cycle=pending['cycle'],reason='shutdown_chord'))
                from booster_deploy.utils.replay_control import write_abort
                write_abort(os.getenv('BOOSTER_DIAG_DIR'))
            self._tracking_transition = None
            probe = getattr(self, "_exit_mode_query", None)
            if probe:
                self._discard_mode_future({"probe": probe["future"]}, "probe")
            self._exit_mode_query = None
            self._exit_mode_sample = None
            self._exit_mode_next_query = 0.0
            changing = getattr(self, "_mode_request", None)
            if changing:
                self._discard_mode_future(changing, "query")
                changing["next_query"] = 0.0
                if not changing["sent"]:
                    changing["status"] = "cancelled"
                if changing["sent"] and changing["target"] == _RobotModeInt.kCustom:
                    self.logger.warning("B+X: Custom already submitted; draining RPC before query. Local cancellation cannot recall remote CHANGE or guarantee vendor ordering")
            return
        session = (os.getenv("BOOSTER_TMUX_SESSION") or "").strip()
        if not session:
            # Fallback: the diag dir is logs/diagnostics/<session>, which is set for
            # every pane. Keeps the chord working on sessions started by older
            # launchers (or respawned without the exported env).
            diag = (os.getenv("BOOSTER_DIAG_DIR") or "").strip().rstrip("/")
            if diag:
                session = os.path.basename(diag)
        if not session:
            self.logger.warning("Stack shutdown ignored: BOOSTER_TMUX_SESSION is not set")
            if reason == "chord":
                self._request_normal_shutdown()
            return
        repo_root = Path(__file__).resolve().parents[2]
        script = repo_root / "scripts" / "teleop" / "teleop_shutdown.sh"
        if not script.is_file():
            self.logger.warning("Stack shutdown ignored: %s is missing", script)
            if reason == "chord":
                self._request_normal_shutdown()
            return
        try:
            subprocess.Popen(
                ["setsid", "bash", str(script), "--session", session, "--reason", reason],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd=str(repo_root),
                start_new_session=True,
            )
            self.logger.info(
                "Stack shutdown requested (%s): diagnostics → stop panes → kill session",
                reason,
            )
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("Stack shutdown could not start: %s", exc)
            if reason == "chord":
                self._request_normal_shutdown()

    def run(self):

        print("Initialization complete. Waiting in Walking mode.")
        print(self.remoteControlService.get_operation_hint())
        if self.remoteControlService.pico_walking_control_enabled:
            walk_cfg = self.remoteControlService.pico_walking_config
            self.logger.info(
                "PICO Walking velocity enabled: max=(%.2f, %.2f, %.2f), "
                "deadzone=%.1f%%, rate=%.1fHz, stale=%.2fs",
                walk_cfg.max_vx,
                walk_cfg.max_vy,
                walk_cfg.max_vyaw,
                100.0 * walk_cfg.axis_deadzone,
                walk_cfg.update_rate_hz,
                walk_cfg.stale_timeout_s,
            )
        self._set_output_mode("none")
        self._leave_to_walking()
        try:
            while self.is_running and not self.exit_event.is_set() and rclpy.ok():
                if self.remoteControlService.consume_shutdown_request():
                    # Discard a pending X-first reset before it can wait for an
                    # absent worker. Standalone X retains its existing behavior.
                    self.remoteControlService.consume_retarget_reset_request()
                    self._request_stack_shutdown("chord")
                if (self.remoteControlService.consume_retarget_reset_request()
                        and not getattr(self, "_shutdown_requested", False)):
                    if self.control_state == BoosterControlState.WALKING:
                        try:
                            self.remoteControlService.publish_retarget_reset()
                            self.logger.info(
                                "PICO X: retarget history reset from the next human frame"
                            )
                        except Exception as exc:
                            self.logger.warning(
                                "PICO retarget reset request could not be sent: %s",
                                exc,
                            )
                    else:
                        self.logger.info(
                            "Ignoring PICO retarget reset because state=%s",
                            self.control_state.value,
                        )
                event = self.remoteControlService.consume_control_event()
                if event is not None:
                    self._handle_control_event(event)
                self._service_exit_mode_query()
                self._service_mode_request()
                self._service_reference_exit()
                self._service_walking_transition()
                self._service_shutdown()
                self._service_tracking_transition()
                from booster_deploy.telemetry.events import Transitions
                if not hasattr(self,'_telemetry_transitions'): self._telemetry_transitions=Transitions('ctrl')
                self._telemetry_transitions.update(control_state=self.control_state.value)
                if self.exit_event.is_set():
                    continue
                if self.control_state in {
                    BoosterControlState.TRACKING,
                    BoosterControlState.TRACKING_PAUSED,
                }:
                    replay_finished = getattr(self, "offline_replay_finished", None)
                    if replay_finished is not None and replay_finished.is_set():
                        replay_finished.clear()
                        self.logger.info("Offline replay finished; starting smooth exit to Walking")
                        self._begin_reference_exit()
                    if (
                        self.inference_process is None
                        or not self.inference_process.is_alive()
                    ):
                        self.logger.error("Inference process died unexpectedly")
                        self._set_output_mode("hold")
                        self.control_state = BoosterControlState.FAULT
                        self._leave_to_walking()
                if self.control_state == BoosterControlState.WALKING:
                    self._tick_pico_walking_velocity()
                self._emit_ctrl_diag()
                self.exit_event.wait(timeout=0.05)
        except BaseException as exc:
            from booster_deploy.telemetry.runtime import get_writer
            writer=get_writer()
            if hasattr(writer,'mark_incomplete'): writer.mark_incomplete(repr(exc))
            raise
        finally:
            diag = getattr(self, "_diag_ctrl", None)
            if diag is not None:
                diag.emit(
                    "left tracking",
                    alerts=(),
                    extra={"state": {"exit": "left_tracking",
                                     "mode": self._output_mode,
                                     "state": self.control_state.value,
                                     **self._exit_safety_state()}},
                    force=True,
                )
                diag.close()
            if not getattr(self, "_shutdown_complete", False):
                self.logger.error("Control loop ended without confirmed shutdown; no safe hand-back claimed")
            # This is a forced/ROS/exception path; never wait indefinitely here.
            self.cleanup(force=True)

    def __enter__(self) -> BoosterRobotPortal:

        return self

    def __exit__(self, *args) -> None:

        self.cleanup(force=bool(args and args[0]) or self.exit_event.is_set() or not rclpy.ok())

    @staticmethod
    def inference_process_func(
        cfg: ControllerCfg,
        portal: BoosterRobotPortal,
    ) -> None:

        BoosterRobotController(cfg, portal).run()
        portal.logger.info("Inference process stopped.")


class BoosterRobotController(BaseController):
    def __init__(self, cfg: ControllerCfg, portal: BoosterRobotPortal) -> None:

        self.portal = portal
        super().__init__(cfg)
        try:
            self._last_pause_toggle_seq = int(
                portal.synced_command.read()[0]["pause_toggle"]
            )
        except Exception:
            self._last_pause_toggle_seq = 0

    def update_vel_command(self):

        cmd = self.portal.synced_command.read()[0]

        self.vel_command.lin_vel_x = cmd["vx"] * self.vel_command.vx_max
        self.vel_command.lin_vel_y = cmd["vy"] * self.vel_command.vy_max
        self.vel_command.ang_vel_yaw = cmd["vyaw"] * self.vel_command.vyaw_max

    def check_pause_toggle(self):

        cmd = self.portal.synced_command.read()[0]
        seq = int(cmd["pause_toggle"])
        token = bytes(cmd['reference_exit_token']).decode() if 'reference_exit_token' in cmd.dtype.names else ''
        session = bytes(cmd['reference_exit_session']).decode() if 'reference_exit_session' in cmd.dtype.names else ''
        policy_session = getattr(getattr(self.policy, '_diag', None), 'policy_session', '') or ''
        if token and session == policy_session:
            self._last_pause_toggle_seq = seq
            if token != getattr(self, '_reference_exit_seen', None):
                self._reference_exit_seen = token
                begin = getattr(self.policy, 'begin_reference_exit', None)
                if not callable(begin):
                    raise RuntimeError('Policy does not support reference exit; parent will fall back')
                begin(token)
            return
        if seq != self._last_pause_toggle_seq:
            self._last_pause_toggle_seq = seq
            toggle = getattr(self.policy, "toggle_reference_pause", None)
            if callable(toggle):
                paused = toggle()
                status = "paused" if paused else "resumed"
                print(f"[Booster Control] Reference {status} by pause key")
                # Freeze/resume the offline replay too. Without this the clip keeps
                # advancing while the policy holds a frozen window, so a pause either
                # jumps the reference ahead on resume or runs the clip out and the
                # publisher silently starts a new cycle (progress bar jumps to 0).
                self._write_replay_control(paused)
            else:
                print("[Booster RT Control] Warning: toggle_reference_pause method not available")

    def _probe_entry_jump(self, dof_targets) -> None:
        """Report how far the first policy targets sit from the measured joints.

        There is no simulator in the loop for real-robot task switches, so this is
        the cheap measurement: a policy that takes over without an entry ramp shows
        a large first-tick delta, a smooth handover shows a few degrees.
        """

        # This runs inside the *controller* (inference process), which has no
        # control_state -- reaching for it crashed the policy process on the first
        # tick (A looked dead). Instrumentation must never be able to do that, so
        # everything here is best-effort and state is read from the portal.
        state = getattr(getattr(self, "portal", None), "control_state", None)
        try:
            tracking = state == BoosterControlState.TRACKING
        except Exception:  # noqa: BLE001
            tracking = False
        if not tracking:
            self._entry_probe_ticks = 0
            self._entry_probe_max_deg = 0.0
            return
        ticks = getattr(self, "_entry_probe_ticks", 0)
        if ticks >= 10:
            return
        try:
            measured = self.robot.data.joint_pos
            delta = (dof_targets - measured).abs().max().item() * 180.0 / math.pi
        except Exception:  # noqa: BLE001
            return
        self._entry_probe_ticks = ticks + 1
        self._entry_probe_max_deg = max(getattr(self, "_entry_probe_max_deg", 0.0), float(delta))
        if ticks == 0:
            print(f"[entry] tick0 |target - measured| max = {delta:.2f} deg", flush=True)
        elif ticks == 9:
            print(f"[entry] first 10 ticks max = {self._entry_probe_max_deg:.2f} deg", flush=True)

    def _write_replay_control(self, paused: bool) -> None:
        """Publish pause state for the offline publisher (file, cross-process)."""

        from booster_deploy.utils.replay_control import write_state

        write_state(paused)

    def update_state(self, *, record: bool = True) -> None:

        state = self.portal.synced_state.read()[0]
        self.state_recv_id = int(state["state_recv_id"]) if "state_recv_id" in state.dtype.names else None
        self.state_recv_ns = int(state["state_recv_ns"]) if "state_recv_ns" in state.dtype.names else None
        size=int(state['low_state_size']) if 'low_state_size' in state.dtype.names else 0
        self._telemetry_low_state=bytes(state['low_state_payload'])[:size] if size else None

        self.robot.data.joint_pos = torch.from_numpy(
            state["joint_pos"]).to(dtype=torch.float32).to(
                self.robot.data.device)
        self.robot.data.joint_vel = torch.from_numpy(
            state["joint_vel"]).to(dtype=torch.float32).to(
                self.robot.data.device)
        self.robot.data.feedback_torque = torch.from_numpy(
            state["feedback_torque"]).to(dtype=torch.float32).to(
                self.robot.data.device)
        self.robot.data.root_pos_w = torch.from_numpy(
            state["root_pos_w"]).to(dtype=torch.float32).to(
                self.robot.data.device)
        rpy_t = torch.from_numpy(state["root_rpy_w"]).to(
            dtype=torch.float32).to(self.robot.data.device)
        self.robot.data.root_quat_w = lab_math.quat_from_euler_xyz(
            *rpy_t
        ).squeeze()
        self.robot.data.root_lin_vel_b = lab_math.quat_apply_inverse(
            self.robot.data.root_quat_w,
            torch.from_numpy(
                state["root_lin_vel_w"]).to(dtype=torch.float32).to(
                    self.robot.data.device)
        )
        self.robot.data.root_ang_vel_b = torch.from_numpy(
            state["root_ang_vel_b"]).to(dtype=torch.float32).to(
                self.robot.data.device)

    def ctrl_step(self, dof_targets: torch.Tensor) -> None:
        # SMPL may return a placeholder before the first human frame.  It must
        # not satisfy policy_ready and let Walking hand over to Custom.
        if getattr(self.policy, "policy_action_ready", True) is False:
            return
        if not bool(torch.isfinite(dof_targets).all()):
            return
        diag = getattr(self.policy, "_diag", None)
        self.portal._telemetry_policy_session = getattr(diag, "policy_session", "") or ""
        self.portal._telemetry_policy_tick = getattr(self.policy, "_ticks", getattr(self, "_step_count", 0))
        self.portal._telemetry_tick = getattr(self,'_telemetry_policy_tick','')
        self.portal._telemetry_state_recv_id = getattr(self, "state_recv_id", 0)
        self.portal._reference_exit_ack = (getattr(self.policy, '_exit_reference_token', '') or '') if getattr(self.policy, '_exit_reference_input_ready', False) else ''
        self.portal._reference_entry_ready = bool(getattr(self.policy, "entry_prepared", False))
        self.portal._reference_entry_ack = int(getattr(self.portal, "offline_entry_cycle", 0)) if getattr(self.policy, "entry_complete", False) else 0
        self.portal._write_policy_action(dof_targets)

    def stop(self):

        super().stop()
        self.portal.policy_stop_event.set()

    def run(self):
        from booster_deploy.telemetry.runtime import get_writer,close_process
        try:
            return self._run_policy()
        except BaseException as exc:
            writer=get_writer()
            if hasattr(writer,'mark_incomplete'): writer.mark_incomplete(repr(exc))
            raise
        finally:
            try:
                self.stop()
            except BaseException as exc:
                writer=get_writer()
                if hasattr(writer,'mark_incomplete'): writer.mark_incomplete(repr(exc))
                raise
            finally:
                close_process(permanent=True)

    def _run_policy(self):

        web_publisher = None
        web_endpoint = os.getenv("BOOSTER_WEB_VIS_ENDPOINT", "").strip()
        if web_endpoint:
            from booster_deploy.visualization.web.runtime import DirectWebSnapshotPublisher

            web_publisher = DirectWebSnapshotPublisher(web_endpoint)
        self.update_state(record=False)
        if self.vel_command is not None:
            self.update_vel_command()
        self.start()

        policy_deadline = (
            None
            if self.portal.timer.use_sim_time
            else PeriodicDeadline(float(self.cfg.policy_dt))
        )
        next_inference_time = self.portal.timer.get_time()
        try:
            while self.is_running and not self.portal.policy_stop_event.is_set():
                if policy_deadline is None:
                    if self.portal.timer.get_time() < next_inference_time:
                        time.sleep(0.0002)
                        continue
                    next_inference_time += self.cfg.policy_dt
                else:
                    policy_deadline.wait()
                    if self.portal.policy_stop_event.is_set():
                        break

                self.update_state()
                if web_publisher is not None:
                    root_pos = self.robot.data.root_pos_w.detach().cpu().numpy().copy()
                    # Booster low state has attitude but no world translation.  Keep
                    # the measured pose standing at a stable display anchor.
                    if (float(np.linalg.norm(root_pos)) < 1e-6
                            or not (0.3 < float(root_pos[2]) < 2.0)):
                        # Either all-zero (usual) or an implausible odometry height:
                        # anchor it so the echo never renders underground/airborne.
                        root_pos[2] = 1.0
                    joint_pos = self.robot.data.joint_pos.detach().cpu().numpy()
                    # No permutation here: the SDK/CONTROL joint order is exactly the
                    # MJCF qpos order (tests/test_t2_contract asserts it), while
                    # real2sim_joint_indexes maps into the *policy* order — feeding
                    # that into qpos scrambles the limbs.
                    sim_joint_pos = joint_pos
                    qpos = np.concatenate((
                        root_pos,
                        self.robot.data.root_quat_w.detach().cpu().numpy(),
                        sim_joint_pos,
                    ))
                    if np.isfinite(qpos).all():
                        web_publisher.publish(qpos)
                if self.vel_command is not None:
                    self.update_vel_command()

                self.check_pause_toggle()
                self.portal.metrics["policy_step"].mark()
                dof_targets = self.policy_step()
                try:  # instrumentation must never be able to kill the policy loop
                    self._probe_entry_jump(dof_targets)
                except Exception:  # noqa: BLE001
                    pass
                # Offline replay: the clip ran out, ask the control process to
                # leave Tracking. The policy only reports this when
                # BOOSTER_OFFLINE_REPLAY is set, so live teleop never fires it.
                reference_finished = getattr(self.policy, "reference_finished", None)
                playback_allowed = not getattr(self.portal, 'offline_entry_cycle', 0) or self.portal.offline_playback_active.is_set()
                if playback_allowed and callable(reference_finished) and reference_finished():
                    replay_finished = getattr(self.portal, "offline_replay_finished", None)
                    if replay_finished is not None:
                        replay_finished.set()
                self.ctrl_step(dof_targets)
                if policy_deadline is not None:
                    policy_deadline.complete()
        finally:
            if web_publisher is not None:
                web_publisher.close()
