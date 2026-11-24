
from enum import Enum
from typing import Any, Optional, Sequence
import math
import os
import evdev
import threading
from dataclasses import dataclass
import time
import termios
import tty
import select
import atexit
import sys

from booster_deploy.retargeting.rt_config import RealtimeRetargetZmqConfig
from booster_deploy.retargeting.rt_zmq import PicoControlSubscriber, RetargetResetPublisher
from booster_deploy.types.pico_types import PicoButtonBits, PicoControlFrameNs


@dataclass
class JoystickConfig:

    max_vx: float = 1.
    max_vy: float = 1.
    max_vyaw: float = 1.
    control_threshold: float = 0.1








    rl_gait_button: evdev.ecodes = evdev.ecodes.BTN_A
    x_axis: evdev.ecodes = evdev.ecodes.ABS_Y
    y_axis: evdev.ecodes = evdev.ecodes.ABS_X
    yaw_axis: evdev.ecodes = evdev.ecodes.ABS_RX



@dataclass
class PicoRemoteControlConfig:

    host: str = "127.0.0.1"
    port: int = 5555
    topic: str = "pico_rt.control"
    recv_timeout_ms: int = 50
    stale_timeout_s: float = 0.5
    button_debounce_s: float = 0.25
    trigger_deadzone: float = 0.02
    retarget_reset_host: str = "127.0.0.1"
    retarget_reset_port: int = 5559
    retarget_reset_topic: str = "retarget_rt.reset"


@dataclass
class PicoWalkingVelocityConfig:

    axis_deadzone: float = 0.05
    stale_timeout_s: float = 0.25
    max_vx: float = 0.5
    max_vy: float = 0.2
    max_vyaw: float = 0.3
    filter_time_constant_s: float = 0.12
    max_linear_accel: float = 0.8
    max_linear_decel: float = 1.2
    max_yaw_accel: float = 1.2
    max_yaw_decel: float = 1.8
    update_rate_hz: float = 20.0
    rpc_timeout_s: float = 0.1
    stop_epsilon: float = 1e-3

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.axis_deadzone) < 1.0:
            raise ValueError("PICO walking axis_deadzone must be in [0, 1)")
        if (
            not math.isfinite(float(self.stale_timeout_s))
            or float(self.stale_timeout_s) < 0.0
        ):
            raise ValueError(
                "PICO walking stale_timeout_s must be non-negative"
            )
        for name in ("max_vx", "max_vy", "max_vyaw"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"PICO walking {name} must be positive and finite"
                )
        for name in (
            "filter_time_constant_s",
            "max_linear_accel",
            "max_linear_decel",
            "max_yaw_accel",
            "max_yaw_decel",
            "update_rate_hz",
            "rpc_timeout_s",
            "stop_epsilon",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"PICO walking {name} must be positive and finite"
                )


class PicoWalkingVelocityFilter:

    _AXIS_COUNT = 3

    def __init__(self, config: PicoWalkingVelocityConfig) -> None:
        self.config = config
        self._value = (0.0, 0.0, 0.0)
        self._last_update_s: float | None = None

    @property
    def value(self) -> tuple[float, float, float]:
        return self._value

    def reset(self) -> None:
        self._value = (0.0, 0.0, 0.0)
        self._last_update_s = None

    def step(
        self,
        target: Sequence[float],
        *,
        now_s: float,
    ) -> tuple[float, float, float]:

        if len(target) != self._AXIS_COUNT:
            raise ValueError(
                f"Expected {self._AXIS_COUNT} velocity axes, "
                f"got {len(target)}"
            )
        now = float(now_s)
        if not math.isfinite(now):
            raise ValueError(
                "PICO walking filter requires a finite monotonic timestamp"
            )

        nominal_dt = 1.0 / float(self.config.update_rate_hz)
        if self._last_update_s is None:
            dt = nominal_dt
        else:
            dt = max(0.0, now - self._last_update_s)

            dt = min(dt, 2.0 * nominal_dt)
        self._last_update_s = now

        limits = (
            float(self.config.max_vx),
            float(self.config.max_vy),
            float(self.config.max_vyaw),
        )
        target_values = tuple(
            _finite_clamp(value, -limit, limit)
            for value, limit in zip(target, limits)
        )
        current = self._value
        if dt <= 0.0:
            return current

        alpha = 1.0 - math.exp(-dt / float(self.config.filter_time_constant_s))
        damped_target = tuple(
            current_i + alpha * (target_i - current_i)
            for current_i, target_i in zip(current, target_values)
        )
        accel_limits = (
            float(self.config.max_linear_accel),
            float(self.config.max_linear_accel),
            float(self.config.max_yaw_accel),
        )
        decel_limits = (
            float(self.config.max_linear_decel),
            float(self.config.max_linear_decel),
            float(self.config.max_yaw_decel),
        )
        next_value = []
        for current_i, desired_i, target_i, accel_i, decel_i in zip(
            current,
            damped_target,
            target_values,
            accel_limits,
            decel_limits,
        ):
            delta = desired_i - current_i
            if abs(delta) <= float(self.config.stop_epsilon):
                result_i = desired_i
            else:
                accelerating = current_i == 0.0 or (
                    current_i * desired_i > 0.0
                    and abs(desired_i) > abs(current_i)
                )
                rate = accel_i if accelerating else decel_i
                max_delta = rate * dt
                result_i = current_i + max(
                    -max_delta,
                    min(max_delta, delta),
                )
            if (
                abs(result_i) <= float(self.config.stop_epsilon)
                and abs(target_i) <= float(self.config.stop_epsilon)
            ):
                result_i = 0.0
            next_value.append(result_i)

        result = tuple(next_value)
        self._value = result
        return result


@dataclass
class RobotJoystickConfig:

    topic: str = "/remote_controller_state"
    stale_timeout_s: float = 0.5
    button_debounce_s: float = 0.25


class ControlEventType(str, Enum):

    ENTER_TRACKING = "enter_tracking"
    ENTER_PVT = "enter_pvt"
    RETURN_TO_WALK = "return_to_walk"
    TOGGLE_TRACKING_PAUSE = "toggle_tracking_pause"


_ROBOT_JOYSTICK_ACTIONS = {
    "a": "rl",
    "b": "exit_rl",
    "y": "pause",
    "x": "reset_retarget",
}


def _finite_clamp(value: float, lower: float, upper: float) -> float:

    value = float(value)
    if not math.isfinite(value):
        return 0.0
    return max(float(lower), min(float(upper), value))


def _rescaled_axis_deadzone(value: float, deadzone: float) -> float:

    value = _finite_clamp(value, -1.0, 1.0)
    magnitude = abs(value)
    if magnitude <= float(deadzone):
        return 0.0
    scaled = (magnitude - float(deadzone)) / (1.0 - float(deadzone))
    return math.copysign(min(scaled, 1.0), value)


def _rescaled_radial_deadzone(
    x: float,
    y: float,
    deadzone: float,
) -> tuple[float, float]:

    x = _finite_clamp(x, -1.0, 1.0)
    y = _finite_clamp(y, -1.0, 1.0)
    magnitude = math.hypot(x, y)
    if magnitude <= float(deadzone):
        return 0.0, 0.0
    clamped_magnitude = min(magnitude, 1.0)
    scaled_magnitude = (
        clamped_magnitude - float(deadzone)
    ) / (1.0 - float(deadzone))
    direction_scale = scaled_magnitude / magnitude
    return x * direction_scale, y * direction_scale


def map_pico_walking_axes(
    left_axis: Sequence[float],
    right_axis: Sequence[float],
    config: PicoWalkingVelocityConfig,
) -> tuple[float, float, float]:

    if len(left_axis) < 2 or len(right_axis) < 1:
        return 0.0, 0.0, 0.0
    raw_axes = (left_axis[0], left_axis[1], right_axis[0])
    if not all(math.isfinite(float(value)) for value in raw_axes):
        return 0.0, 0.0, 0.0
    left_x, left_y = _rescaled_radial_deadzone(
        left_axis[0], left_axis[1], float(config.axis_deadzone)
    )
    right_x = _rescaled_axis_deadzone(
        right_axis[0], float(config.axis_deadzone)
    )
    return (
        left_y * float(config.max_vx),
        -left_x * float(config.max_vy),
        -right_x * float(config.max_vyaw),
    )


class RemoteControlService:

    def __init__(self, config: Optional[JoystickConfig] = None):
        self.config = config or JoystickConfig()
        self.pico_config = self._load_pico_config()
        self.pico_walking_config = self._load_pico_walking_config()
        self.robot_joystick_config = self._load_robot_joystick_config()
        self._lock = threading.Lock()
        self._running = True
        self._backend = "uninitialized"
        self.joystick = None
        self.joystick_runner = None
        self.keyboard_runner = None
        self.pico_runner = None
        self._pico_subscriber: PicoControlSubscriber | None = None
        self._retarget_reset_publisher: RetargetResetPublisher | None = None
        self._stdin_tty = False
        self._old_termios = None
        self.shutdown_callback = None

        self.vx = 0.0
        self.vy = 0.0
        self.vyaw = 0.0
        self.keyboard_start_custom_mode = False
        self.keyboard_start_rl_gait = False
        self._exit_rl_gait_requested = False
        self.keyboard_pause_toggle = False
        self.left_trigger = 0.0
        self.right_trigger = 0.0
        self._pico_last_buttons = PicoButtonBits.NONE
        self._pico_last_local_recv_s = 0.0
        self._pico_latest_seq: int | None = None
        self._pico_retarget_reset_requested = False
        # B+X held together for 1 s = "close the whole teleop stack" (walking only;
        # the controller checks the state before acting on the request).
        # Never combine buttons, release frames, or hold clocks across devices.
        self._shutdown_chords: dict[str, dict] = {}
        self.shutdown_requested = False
        self._pico_last_retarget_reset_s = -float("inf")
        self._pico_walking_target = (0.0, 0.0, 0.0)
        self._pico_last_event_time = {
            PicoButtonBits.X: -float("inf"),
            PicoButtonBits.A: -float("inf"),
            PicoButtonBits.Y: -float("inf"),
            PicoButtonBits.B: -float("inf"),
        }
        self._robot_joystick_last_recv_s = 0.0
        self._robot_joystick_vx = 0.0
        self._robot_joystick_vy = 0.0
        self._robot_joystick_vyaw = 0.0
        self._robot_joystick_left_trigger = 0.0
        self._robot_joystick_right_trigger = 0.0
        self._robot_joystick_button_latched = {
            name: False for name in _ROBOT_JOYSTICK_ACTIONS
        }
        self._mocap_recording = os.getenv("BOOSTER_MOCAP_RECORDING") == "1"
        self._robot_joystick_last_event_time = {
            name: -float("inf") for name in _ROBOT_JOYSTICK_ACTIONS
        }
        backend = os.getenv("BOOSTER_REMOTE_CONTROL_BACKEND", "auto").strip().lower()
        if backend == "pico":
            self._backend = "pico"
            self._init_pico_control()
            self._start_pico_control_thread()
        elif backend == "keyboard":
            self._backend = "keyboard"
            self._init_keyboard_control()
            self._start_keyboard_thread()
        elif backend == "joystick":
            self._backend = "joystick"
            self._init_joystick()
            self._start_joystick_thread()
        elif backend == "joystick-robot":
            self._backend = "joystick-robot"
            self._init_robot_joystick_control()
        elif backend == "auto":
            try:
                self._backend = "joystick"
                self._init_joystick()
                self._start_joystick_thread()
            except Exception as e:
                print(f"{e}, downgrade to keyboard control")
                self._backend = "keyboard"
                self._init_keyboard_control()
                self._start_keyboard_thread()
        else:
            raise ValueError(
                "Unsupported BOOSTER_REMOTE_CONTROL_BACKEND="
                f"{backend!r}; expected auto, keyboard, joystick, "
                "joystick-robot, or pico"
            )

        # Opt-in dashboard keyboard input shares the existing control flags,
        # without replacing joystick/PICO input or duplicating keyboard listeners.
        if self._backend != "keyboard" and os.getenv("BOOSTER_CONTROL_KEYBOARD_OVERLAY", "0") == "1":
            self._start_keyboard_thread()

    @staticmethod
    def _env_float(name: str, default: float) -> float:
        value = os.getenv(name)
        if value is None or value == "":
            return float(default)
        return float(value)

    @staticmethod
    def _env_int(name: str, default: int) -> int:
        value = os.getenv(name)
        if value is None or value == "":
            return int(default)
        return int(value)

    @staticmethod
    def _env_bool(name: str, default: bool = False) -> bool:
        value = os.getenv(name)
        if value is None or value == "":
            return bool(default)
        return value.strip().lower() in {"1", "true", "yes", "on"}

    def _load_pico_config(self) -> PicoRemoteControlConfig:
        return PicoRemoteControlConfig(
            host=os.getenv("BOOSTER_PICO_CONTROL_HOST", "127.0.0.1"),
            port=self._env_int("BOOSTER_PICO_CONTROL_PORT", 5555),
            topic=os.getenv("BOOSTER_PICO_CONTROL_TOPIC", "pico_rt.control"),
            recv_timeout_ms=self._env_int("BOOSTER_PICO_RECV_TIMEOUT_MS", 50),
            stale_timeout_s=self._env_float("BOOSTER_PICO_CONTROL_STALE_TIMEOUT", 0.5),
            button_debounce_s=self._env_float("BOOSTER_PICO_BUTTON_DEBOUNCE", 0.25),
            trigger_deadzone=self._env_float("BOOSTER_PICO_TRIGGER_DEADZONE", 0.02),
            retarget_reset_host=os.getenv(
                "BOOSTER_RT_RESET_HOST",
                os.getenv("BOOSTER_RT_REFERENCE_HOST", "127.0.0.1"),
            ),
            retarget_reset_port=self._env_int("BOOSTER_RT_RESET_PORT", 5559),
            retarget_reset_topic=os.getenv("BOOSTER_RT_RESET_TOPIC", "retarget_rt.reset"),
        )

    def _load_pico_walking_config(self) -> PicoWalkingVelocityConfig:
        return PicoWalkingVelocityConfig(
            axis_deadzone=self._env_float("BOOSTER_PICO_WALK_DEADZONE", 0.05),
            stale_timeout_s=self._env_float(
                "BOOSTER_PICO_WALK_STALE_TIMEOUT", 0.25
            ),
            max_vx=self._env_float("BOOSTER_PICO_WALK_MAX_VX", 0.5),
            max_vy=self._env_float("BOOSTER_PICO_WALK_MAX_VY", 0.2),
            max_vyaw=self._env_float("BOOSTER_PICO_WALK_MAX_VYAW", 0.3),
            filter_time_constant_s=self._env_float(
                "BOOSTER_PICO_WALK_FILTER_TAU", 0.12
            ),
            max_linear_accel=self._env_float(
                "BOOSTER_PICO_WALK_MAX_LINEAR_ACCEL", 0.8
            ),
            max_linear_decel=self._env_float(
                "BOOSTER_PICO_WALK_MAX_LINEAR_DECEL", 1.2
            ),
            max_yaw_accel=self._env_float(
                "BOOSTER_PICO_WALK_MAX_YAW_ACCEL", 1.2
            ),
            max_yaw_decel=self._env_float(
                "BOOSTER_PICO_WALK_MAX_YAW_DECEL", 1.8
            ),
            update_rate_hz=self._env_float("BOOSTER_PICO_WALK_RATE_HZ", 20.0),
            rpc_timeout_s=self._env_float(
                "BOOSTER_PICO_WALK_RPC_TIMEOUT", 0.1
            ),
            stop_epsilon=self._env_float("BOOSTER_PICO_WALK_STOP_EPS", 1e-3),
        )

    def _load_robot_joystick_config(self) -> RobotJoystickConfig:
        return RobotJoystickConfig(
            topic=os.getenv(
                "BOOSTER_ROBOT_JOYSTICK_TOPIC",
                "/remote_controller_state",
            ),
            stale_timeout_s=self._env_float(
                "BOOSTER_ROBOT_JOYSTICK_STALE_TIMEOUT",
                0.5,
            ),
            button_debounce_s=self._env_float(
                "BOOSTER_ROBOT_JOYSTICK_BUTTON_DEBOUNCE",
                0.25,
            ),
        )

    @property
    def robot_joystick_enabled(self) -> bool:



        return True

    @property
    def robot_joystick_topic(self) -> str:
        return self.robot_joystick_config.topic

    @property
    def pico_walking_control_enabled(self) -> bool:
        return self._backend == "pico"

    def get_operation_hint(self) -> str:

        if self._backend == "pico":
            return (
                "In Walking, the left stick controls x/y velocity and the "
                "right "
                "stick controls yaw; X resets retargeting in Walking, A enters tracking, Y pauses/"
                "resumes reference, and B returns to Walking."
            )
        if self._backend == "joystick-robot":
            return (
                "Robot joystick axes control velocity; A enters tracking, "
                "Y pauses/resumes reference, and B returns to Walking."
            )
        if self._backend == "joystick" and self.joystick is not None:
            return (
                "Joystick axes control velocity; A enters tracking, "
                "Y pauses/resumes reference, and B returns to Walking."
            )
        return (
            "Keyboard controls velocity; 'r' enters tracking, "
            "'y' pauses/resumes reference, and 'b' returns to Walking."
        )

    def get_custom_mode_operation_hint(self) -> str:

        if self._backend == "pico":
            return "Press A to enter tracking directly."
        if self._backend == "joystick-robot":
            return "Press robot joystick A to enter tracking directly."
        if self._backend == "joystick" and self.joystick is not None:
            return "Press A to enter tracking directly."
        return "Press keyboard 'r' to enter tracking directly."

    def get_rl_gait_operation_hint(self) -> str:

        if self._backend == "pico":
            return "Press PICO or robot joystick A to enter tracking."
        if self._backend == "joystick-robot":
            return "Press robot joystick A to enter tracking."
        if self._backend == "joystick" and self.joystick is not None:
            return "Press joystick or robot joystick A to enter tracking."
        return "Press keyboard 'r' or robot joystick A to enter tracking."

    def _init_keyboard_control(self):

        self.joystick = None
        self.joystick_runner = None
        self.keyboard_start_custom_mode = False
        self.keyboard_start_rl_gait = False
        self.keyboard_pause_toggle = False

    def _start_keyboard_thread(self):


        self.keyboard_runner = threading.Thread(target=self._keyboard_listener, daemon=True)

        try:
            if sys.stdin.isatty():
                self._stdin_tty = True
                self._old_termios = termios.tcgetattr(sys.stdin.fileno())
            else:
                self._stdin_tty = False
                self._old_termios = None
        except Exception:
            self._stdin_tty = False
            self._old_termios = None


        try:
            atexit.register(self.close)
        except Exception:
            pass

        self.keyboard_runner.start()

    def _init_pico_control(self) -> None:

        self.joystick = None
        self.joystick_runner = None
        cfg = RealtimeRetargetZmqConfig(
            input_host=self.pico_config.host,
            input_port=int(self.pico_config.port),
            input_control_topic=self.pico_config.topic,
        )
        self._pico_subscriber = PicoControlSubscriber(
            cfg,
            recv_timeout_ms=int(self.pico_config.recv_timeout_ms),
            recv_hwm=1,
        )
        self._retarget_reset_publisher = RetargetResetPublisher(
            self.pico_config.retarget_reset_host,
            self.pico_config.retarget_reset_port,
            self.pico_config.retarget_reset_topic,
        )

    def _init_robot_joystick_control(self) -> None:
        pass

    def handle_robot_joystick_state(
        self,
        state: Any,
        *,
        local_recv_s: float | None = None,
    ) -> None:

        recv_s = time.monotonic() if local_recv_s is None else float(local_recv_s)
        vx = self._normalize_robot_joystick_axis(
            -float(state.ly), self.config.max_vx
        )
        vy = self._normalize_robot_joystick_axis(
            -float(state.lx), self.config.max_vy
        )
        vyaw = self._normalize_robot_joystick_axis(
            -float(state.rx), self.config.max_vyaw
        )

        with self._lock:
            self._robot_joystick_vx = vx
            self._robot_joystick_vy = vy
            self._robot_joystick_vyaw = vyaw
            self._robot_joystick_left_trigger = 1.0 if bool(state.lt) else 0.0
            self._robot_joystick_right_trigger = 1.0 if bool(state.rt) else 0.0
            self._robot_joystick_last_recv_s = recv_s

            # B+X held together closes the whole stack, including in recording mode.
            self._track_shutdown_chord(bool(getattr(state, "b", False)),
                                       bool(getattr(state, "x", False)), recv_s,
                                       "joystick")

            recording_modifier = self._mocap_recording and bool(state.lt) and bool(state.rt)
            for button, action in _ROBOT_JOYSTICK_ACTIONS.items():
                pressed = bool(getattr(state, button))
                if button in {"b", "x"} and bool(state.b) and bool(state.x):
                    self._robot_joystick_button_latched[button] = True
                    continue  # Chord is not a standalone B (Walking) or X.
                if recording_modifier and button in {"x", "y"}:
                    self._robot_joystick_button_latched[button] = pressed
                    continue  # Do not reset or pause tracking for a recording chord.
                was_pressed = self._robot_joystick_button_latched[button]
                if pressed and not was_pressed:
                    self._robot_joystick_button_latched[button] = True
                    last_event_s = self._robot_joystick_last_event_time[button]
                    if (
                        recv_s - last_event_s
                        >= self.robot_joystick_config.button_debounce_s
                    ):
                        self._robot_joystick_last_event_time[button] = recv_s
                        self._mark_robot_joystick_action(action)
                elif not pressed:
                    self._robot_joystick_button_latched[button] = False

    def _normalize_robot_joystick_axis(self, value: float, limit: float) -> float:
        scaled = min(max(float(value), -1.0), 1.0) * float(limit)
        if abs(scaled) < self.config.control_threshold:
            return 0.0
        return scaled

    def _mark_robot_joystick_action(self, action: str) -> None:

        if action == "rl":
            self.keyboard_start_rl_gait = True
        elif action == "exit_rl":
            self._exit_rl_gait_requested = True
        elif action == "pause":
            self.keyboard_pause_toggle = True
        elif action == "reset_retarget":
            self._pico_retarget_reset_requested = True
        self._report_key_event(
            {"rl": "A", "exit_rl": "B", "pause": "Y", "reset_retarget": "X"}.get(action, action),
            action,
            "joystick",
        )

    def _start_pico_control_thread(self) -> None:

        self.pico_runner = threading.Thread(
            target=self._run_pico_control,
            daemon=True,
            name="pico_remote_control",
        )
        self.pico_runner.start()

    def _run_pico_control(self) -> None:

        while self._running:
            subscriber = self._pico_subscriber
            if subscriber is None:
                time.sleep(0.05)
                continue
            try:
                frame = subscriber.recv_latest(timeout_s=0.1)
            except TimeoutError:
                continue
            except Exception as e:
                if not self._running:
                    break
                print(f"Error in PICO control polling loop: {e}")
                time.sleep(0.05)
                continue
            self._handle_pico_control_frame(frame, local_recv_s=time.monotonic())

    def _handle_pico_control_frame(self, frame: PicoControlFrameNs, *, local_recv_s: float) -> None:

        buttons = frame.buttons
        left_trigger = self._normalize_trigger(frame.left_trigger)
        right_trigger = self._normalize_trigger(frame.right_trigger)
        walking_target = map_pico_walking_axes(
            frame.left_axis,
            frame.right_axis,
            self.pico_walking_config,
        )
        with self._lock:
            rising = PicoButtonBits(int(buttons) & ~int(self._pico_last_buttons))
            if rising & PicoButtonBits.X and not buttons & PicoButtonBits.B:
                self._mark_pico_button_event(
                    PicoButtonBits.X, local_recv_s, "reset_retarget"
                )
            if rising & PicoButtonBits.A:
                self._mark_pico_button_event(PicoButtonBits.A, local_recv_s, "rl")
            if rising & PicoButtonBits.B and not buttons & PicoButtonBits.X:
                self._mark_pico_button_event(PicoButtonBits.B, local_recv_s, "exit_rl")
            if rising & PicoButtonBits.Y:
                self._mark_pico_button_event(PicoButtonBits.Y, local_recv_s, "pause")
            self.left_trigger = left_trigger
            self.right_trigger = right_trigger
            self._pico_walking_target = walking_target
            self._track_bx_shutdown_hold(buttons, float(local_recv_s))
            self._pico_last_buttons = buttons
            self._pico_last_local_recv_s = float(local_recv_s)
            self._pico_latest_seq = int(frame.seq)

    def _mark_pico_button_event(self, button: PicoButtonBits, now_s: float, action: str) -> None:

        if action == "reset_retarget":
            last_s = self._pico_last_retarget_reset_s
        else:
            last_s = self._pico_last_event_time.get(button, 0.0)
        if now_s - last_s < self.pico_config.button_debounce_s:
            return
        if action == "reset_retarget":
            self._pico_last_retarget_reset_s = now_s
        else:
            self._pico_last_event_time[button] = now_s
        if action == "rl":
            self.keyboard_start_rl_gait = True
        elif action == "exit_rl":
            self._exit_rl_gait_requested = True
        elif action == "pause":
            self.keyboard_pause_toggle = True
        elif action == "reset_retarget":
            self._pico_retarget_reset_requested = True
        self._report_key_event(button, action, "pico")

    def _report_key_event(self, button: "PicoButtonBits | str", action: str, source: str = "deploy") -> None:
        """Echo an accepted button press to the operator.

        Two consumers: the terminal pane (one readable line) and
        ``logs/diagnostics/<session>/keys.jsonl``, which the dashboard tails for its
        "Key feedback" panel.  Best effort only — a logging failure must never affect
        robot control.
        """

        names = {
            PicoButtonBits.X: "X",
            PicoButtonBits.Y: "Y",
            PicoButtonBits.A: "A",
            PicoButtonBits.B: "B",
        }
        labels = {
            "rl": "Enter teleoperation",
            "pause": "Pause/resume",
            "exit_rl": "Return to Walking",
            "reset_retarget": "Reset retargeting",
            "shutdown_request": "Shut down the entire teleoperation stack",
            "shutdown_hold_started": "B+X held together; hold for 1 second to shut down",
            "shutdown_hold_short": "B+X released before 1 second",
            "shutdown_hold_gap": "B+X input interrupted; restarting the timer",
            "shutdown_refused": "Shutdown refused (actual mode is not 0/1/2 or is unknown)",
        }
        name = button if isinstance(button, str) else names.get(button, "?")
        label = labels.get(action, action)
        try:
            print(f"[key] {name} → {label}", flush=True)
        except Exception:  # noqa: BLE001
            pass
        from booster_deploy.telemetry.keys import key
        key(name,phase='received',origin=source,action=action,label=label)

    def _track_bx_shutdown_hold(self, buttons: int, now_s: float) -> None:
        """B+X held together for 1 s requests a full teleop shutdown.

        Only *requests*: the controller checks fresh actual mode 0/1/2
        before allowing the stack to close (a hard kill while tracking would leave the robot
        without a controller).
        """

        self._track_shutdown_chord(
            bool(buttons & int(PicoButtonBits.B)),
            bool(buttons & int(PicoButtonBits.X)),
            now_s,
        )

    def _track_shutdown_chord(self, b_down: bool, x_down: bool, now_s: float,
                              source: str = "pico") -> None:
        """Backend-agnostic B+X chord (PICO buttons or robot controller buttons)."""

        states = getattr(self, '_shutdown_chords', None)
        if states is None:
            states = self._shutdown_chords = {}
        state = states.setdefault(source, dict(start=None, last=None, fired=False))
        both = bool(b_down) and bool(x_down)
        config = getattr(self, 'robot_joystick_config' if source == 'joystick' else 'pico_config', None)
        fresh_s = float(getattr(config, 'stale_timeout_s', 1.5))  # legacy callers without backend config; live backends use their explicit freshness
        if state['last'] is not None and (now_s < state['last'] or (source != 'joystick' and now_s-state['last'] > fresh_s)):
            if state['start'] is not None and not state['fired']:
                self._report_key_event('B+X', 'shutdown_hold_gap', source)
            state.update(start=None, fired=False)
        state['last'] = now_s
        if not both:
            if state['start'] is not None and not state['fired']:
                self._report_key_event('B+X', 'shutdown_hold_short', source)
            state.update(start=None, fired=False)
            return
        if state['start'] is None:
            state['start'] = now_s
            self._report_key_event('B+X', 'shutdown_hold_started', source)
            return
        self._advance_shutdown_chord(state, now_s, source)

    def _advance_shutdown_chord(self, state: dict, now_s: float, source: str) -> None:
        if state['start'] is None:
            return
        if now_s < state['start']:
            state.update(start=None, last=None, fired=False)
            return
        hold_s = max(1.0, self._env_float("BOOSTER_PICO_SHUTDOWN_HOLD_S", 1.0))
        if state['fired'] or now_s - state['start'] < hold_s:
            return
        state['fired'] = True
        self.shutdown_requested = True
        try:
            print(f"[key] B+X held for {hold_s:.1f}s → requesting teleoperation shutdown", flush=True)
        except Exception:  # noqa: BLE001
            pass
        self._report_key_event("B+X", "shutdown_request", source)

    def consume_shutdown_request(self) -> bool:
        """True once per B+X long-press (edge-triggered)."""

        with self._lock:
            # Native SDL input reports state changes, not a hold heartbeat.
            state = getattr(self, '_shutdown_chords', {}).get('joystick')
            if state is not None:
                self._advance_shutdown_chord(state, time.monotonic(), 'joystick')
            requested = bool(self.shutdown_requested)
            self.shutdown_requested = False
            return requested

    def _normalize_trigger(self, value: float) -> float:

        normalized = min(max(float(value), 0.0), 1.0)
        if normalized < self.pico_config.trigger_deadzone:
            return 0.0
        return normalized

    def _keyboard_listener(self):


        fd = None
        try:
            if not getattr(self, "_stdin_tty", False):

                return
            fd = sys.stdin.fileno()
            tty.setcbreak(fd)
            while self._running:

                rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
                if rlist:
                    ch = sys.stdin.read(1)
                    if ch == "\x03":
                        callback = self.shutdown_callback
                        if callback is not None:
                            callback()
                        continue
                    if ch == " ":
                        key = "space"
                    else:
                        key = ch
                    try:
                        self._handle_keyboard_press(key)
                    except Exception:

                        pass
        finally:

            try:
                if fd is not None and getattr(self, "_old_termios", None) is not None:
                    termios.tcsetattr(fd, termios.TCSADRAIN, self._old_termios)
            except Exception:
                pass

    def _handle_keyboard_press(self, key):

        with self._lock:
            if key == "r":
                self.keyboard_start_rl_gait = True
                self._report_key_event("A", "rl", "keyboard")
            if key == "b":
                self._exit_rl_gait_requested = True
                self._report_key_event("B", "exit_rl", "keyboard")
            if key in {"y", "space"}:
                self.keyboard_pause_toggle = True
                self._report_key_event("Y", "pause", "keyboard")
            if key == "w":
                old_x = self.vx
                self.vx += 0.1
                self.vx = min(self.vx, self.config.max_vx)
                print(f"VX: {old_x:.1f} => {self.vx:.1f}")
            if key == "s":
                old_x = self.vx
                self.vx -= 0.1
                self.vx = max(self.vx, -self.config.max_vx)
                print(f"VX: {old_x:.1f} => {self.vx:.1f}")
            if key == "a":
                old_y = self.vy
                self.vy += 0.1
                self.vy = min(self.vy, self.config.max_vy)
                print(f"VY: {old_y:.1f} => {self.vy:.1f}")
            if key == "d":
                old_y = self.vy
                self.vy -= 0.1
                self.vy = max(self.vy, -self.config.max_vy)
                print(f"VY: {old_y:.1f} => {self.vy:.1f}")
            if key == "q":
                old_yaw = self.vyaw
                self.vyaw += 0.1
                self.vyaw = min(self.vyaw, self.config.max_vyaw)
                print(f"VYaw: {old_yaw:.1f} => {self.vyaw:.1f}")
            if key == "e":
                old_yaw = self.vyaw
                self.vyaw -= 0.1
                self.vyaw = max(self.vyaw, -self.config.max_vyaw)
                print(f"VYaw: {old_yaw:.1f} => {self.vyaw:.1f}")
            if key == "space":
                self.vx = 0
                self.vy = 0
                self.vyaw = 0
                print("FULL STOP")

    def _init_joystick(self) -> None:
        try:
            devices = [evdev.InputDevice(path) for path in evdev.list_devices()]
            joystick = None

            for device in devices:
                caps = device.capabilities()




                if evdev.ecodes.EV_ABS in caps and evdev.ecodes.EV_KEY in caps:
                    abs_info = caps.get(evdev.ecodes.EV_ABS, [])

                    axes = [code for (code, info) in abs_info]
                    if all(code in axes for code in [self.config.x_axis, self.config.y_axis, self.config.yaw_axis]):
                        absinfo = {}
                        for code, info in abs_info:
                            absinfo[code] = info
                        self.axis_ranges = {
                            self.config.x_axis: absinfo[self.config.x_axis],
                            self.config.y_axis: absinfo[self.config.y_axis],
                            self.config.yaw_axis: absinfo[self.config.yaw_axis],
                        }
                        print(f"Found suitable joystick: {device.name}")
                        joystick = device
                        break

            if not joystick:
                raise RuntimeError("No suitable joystick found")

            self.joystick = joystick
            print(f"Selected joystick: {joystick.name}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize joystick: {e}")

    def _start_joystick_thread(self):
        self.joystick_runner = threading.Thread(target=self._run_joystick)
        self.joystick_runner.daemon = True
        self.joystick_runner.start()

    def start_custom_mode(self) -> bool:
        with self._lock:
            requested = self.keyboard_start_custom_mode
            self.keyboard_start_custom_mode = False
        return bool(requested)

    def start_rl_gait(self) -> bool:
        joystick_requested = (
            self._backend == "joystick"
            and self.joystick is not None
            and self.joystick.active_keys() == [self.config.rl_gait_button]
        )
        with self._lock:
            requested = self.keyboard_start_rl_gait
            self.keyboard_start_rl_gait = False
        return bool(joystick_requested or requested)

    def exit_rl_gait(self) -> bool:

        with self._lock:
            requested = self._exit_rl_gait_requested
            self._exit_rl_gait_requested = False
        return requested

    def consume_control_event(self) -> ControlEventType | None:

        if self.exit_rl_gait():
            return ControlEventType.RETURN_TO_WALK
        if self.start_custom_mode():
            return ControlEventType.ENTER_PVT
        if self.check_pause_toggle():
            return ControlEventType.TOGGLE_TRACKING_PAUSE
        if self.start_rl_gait():
            return ControlEventType.ENTER_TRACKING
        return None

    def consume_retarget_reset_request(self) -> bool:

        with self._lock:
            requested = self._pico_retarget_reset_requested
            self._pico_retarget_reset_requested = False
        return requested

    def publish_retarget_reset(self) -> None:

        publisher = self._retarget_reset_publisher
        if publisher is None:
            raise RuntimeError("PICO retarget reset publisher is not initialized")
        # REQ/REP: a False here means the worker never acked, so "sent" would be a
        # lie.  Raising lets the caller log the truth instead of a success line.
        cached = getattr(self, "_reset_port_ok", None)
        tried: set[int] = set()
        order = [cached] if cached else []
        order += [int(self.pico_config.retarget_reset_port), 5559, 15559]
        first = True
        for port in order:
            if port is None or port in tried:
                continue
            tried.add(int(port))
            try:
                if first:
                    target = publisher
                    first = False
                else:
                    target = RetargetResetPublisher(
                        self.pico_config.retarget_reset_host,
                        int(port),
                        self.pico_config.retarget_reset_topic,
                    )
                ok = target.publish_from_current()
            except Exception:  # noqa: BLE001 - a dead port must not mask the next one
                ok = False
            if ok:
                self._reset_port_ok = int(port)
                print(f"[X] retarget reset acked on port {port}", flush=True)
                return
        raise RuntimeError("retarget reset was not acknowledged by the worker")

    def clear_exit_rl_gait(self) -> None:

        with self._lock:
            self._exit_rl_gait_requested = False

    def check_pause_toggle(self) -> bool:
        with self._lock:
            if self.keyboard_pause_toggle:
                self.keyboard_pause_toggle = False
                return True
        return False

    def _run_joystick(self):
        while self._running:
            try:

                event = self.joystick.read_one()
                if event:
                    if event.type == evdev.ecodes.EV_ABS:
                        self._handle_axis(event.code, event.value)
                else:
                    time.sleep(0.01)
            except Exception as e:
                if not self._running:
                    break
                print(f"Error in joystick polling loop: {e}")
                time.sleep(0.05)

    def _handle_axis(self, code: int, value: int):

        try:
            with self._lock:
                if code == self.config.x_axis:
                    self.vx = self._scale(value, self.config.max_vx, self.config.control_threshold, code)

                elif code == self.config.y_axis:
                    self.vy = self._scale(value, self.config.max_vy, self.config.control_threshold, code)

                elif code == self.config.yaw_axis:
                    self.vyaw = self._scale(value, self.config.max_vyaw, self.config.control_threshold, code)

        except Exception:
            raise

    def _scale(self, value: float, max: float, threshold: float, axis_code: int) -> float:
        absinfo = self.axis_ranges[axis_code]
        min_in = absinfo.min
        max_in = absinfo.max

        mapped_value = ((value - min_in) / (max_in - min_in) * 2 - 1) * max


        if abs(mapped_value) < threshold:
            return 0.0
        return -mapped_value

    def get_vx_cmd(self) -> float:
        with self._lock:
            if self._robot_joystick_axes_active_locked(time.monotonic()):
                return self._robot_joystick_vx
            return self.vx

    def get_vy_cmd(self) -> float:
        with self._lock:
            if self._robot_joystick_axes_active_locked(time.monotonic()):
                return self._robot_joystick_vy
            return self.vy

    def get_vyaw_cmd(self) -> float:
        with self._lock:
            if self._robot_joystick_axes_active_locked(time.monotonic()):
                return self._robot_joystick_vyaw
            return self.vyaw

    def get_left_trigger_cmd(self) -> float:

        with self._lock:
            if self._backend != "pico" and self._robot_joystick_fresh_locked(time.monotonic()):
                return self._robot_joystick_left_trigger
            return self.left_trigger

    def get_right_trigger_cmd(self) -> float:

        with self._lock:
            if self._backend != "pico" and self._robot_joystick_fresh_locked(time.monotonic()):
                return self._robot_joystick_right_trigger
            return self.right_trigger

    def _robot_joystick_axes_active_locked(self, now_s: float) -> bool:
        if not self._robot_joystick_fresh_locked(now_s):
            return False
        return any(
            value != 0.0
            for value in (
                self._robot_joystick_vx,
                self._robot_joystick_vy,
                self._robot_joystick_vyaw,
            )
        )

    def _robot_joystick_fresh_locked(self, now_s: float) -> bool:
        return (
            self._robot_joystick_last_recv_s > 0.0
            and now_s - self._robot_joystick_last_recv_s
            <= self.robot_joystick_config.stale_timeout_s
        )

    def robot_joystick_control_fresh(self) -> bool:

        with self._lock:
            return self._robot_joystick_fresh_locked(time.monotonic())

    def robot_joystick_axes_active(self) -> bool:

        with self._lock:
            return self._robot_joystick_axes_active_locked(time.monotonic())

    def pico_control_fresh(self) -> bool:

        if self._backend != "pico":
            return False
        with self._lock:
            last_recv_s = self._pico_last_local_recv_s
        return last_recv_s > 0.0 and time.monotonic() - last_recv_s <= self.pico_config.stale_timeout_s

    def get_pico_walking_velocity_target(
        self,
        *,
        now_s: float | None = None,
    ) -> tuple[float, float, float]:

        if not self.pico_walking_control_enabled:
            return 0.0, 0.0, 0.0
        now = time.monotonic() if now_s is None else float(now_s)
        with self._lock:
            last_recv_s = self._pico_last_local_recv_s
            target = self._pico_walking_target
        age_s = now - last_recv_s
        if last_recv_s <= 0.0 or age_s < 0.0 or age_s > float(
            self.pico_walking_config.stale_timeout_s
        ):
            return 0.0, 0.0, 0.0
        return target

    def close(self):
        self._running = False
        with self._lock:
            self._shutdown_chords = {}
            self.shutdown_requested = False

        try:
            if getattr(self, "_stdin_tty", False) and getattr(self, "_old_termios", None) is not None:
                termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._old_termios)
        except Exception:
            pass
        if hasattr(self, "_pico_subscriber") and self._pico_subscriber is not None:
            try:
                self._pico_subscriber.close()
            except Exception as e:
                print(f"Error closing PICO control subscriber: {e}")
            self._pico_subscriber = None
        if self._retarget_reset_publisher is not None:
            try:
                self._retarget_reset_publisher.close()
            except Exception as e:
                print(f"Error closing retarget reset publisher: {e}")
            self._retarget_reset_publisher = None
        if hasattr(self, "joystick") and getattr(self, "joystick") is not None:
            try:
                self.joystick.close()
            except Exception as e:
                print(f"Error closing joystick: {e}")
        if hasattr(self, "pico_runner") and getattr(self, "pico_runner") is not None:
            try:
                self.pico_runner.join(timeout=1.0)
                if self.pico_runner.is_alive():
                    print("PICO control thread didn't exit within the time limit")
            except Exception as e:
                print(f"Error waiting for PICO control thread to end: {e}")
        if hasattr(self, "joystick_runner") and getattr(self, "joystick_runner") is not None:
            try:
                self.joystick_runner.join(timeout=1.0)
                if self.joystick_runner.is_alive():
                    print("Joystick thread didn't exit within the time limit")
            except Exception as e:
                print(f"Error waiting for joystick thread to end: {e}")
        if hasattr(self, "keyboard_runner") and getattr(self, "keyboard_runner") is not None:
            try:
                self.keyboard_runner.join(timeout=1.0)
                if self.keyboard_runner.is_alive():
                    print("Keyboard thread didn't exit within the time limit")
            except Exception as e:
                print(f"Error waiting for keyboard thread to end: {e}")

    def __enter__(self):

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):

        self.close()
