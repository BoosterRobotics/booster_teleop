#!/usr/bin/env python3
"""Control Revo2 hands directly from Pico triggers.

Onboard usage (Pico connects to the board's LAN IP):
    source scripts/env_set.sh onboard pico
    python scripts/diagnostics/pico_hand_control.py

No body tracking, retarget worker, deployment policy, or robot mode service is
used. Only the selected hand position-controller topics receive ROS commands.
Ctrl+C, SIGTERM, and stale Pico input stop publishing without a final hand move.
When enabled, the script also subscribes to hand JointState feedback and uses
the same position-history stall recovery as the full controller.
BOOSTER_HAND_TURBO_* configures recovery (position values are radians).
Use --disable-stall-recovery to publish the original Pico targets directly.
Missing/stale feedback bypasses recovery, and its status is printed.
Use --hands left or --hands right to control just one side. Use --dry-run to
print the selected Revo2 mappings offline without ROS or the Pico SDK.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import logging
import math
import os
from pathlib import Path
import signal
import sys
import threading
import time
from typing import Any, Sequence


# Same ROS/SDK conversion as hand_retargeting/real_hand_mapping.py.
MAX_RAD = 1.7453292519943296
SDK_SCALE = 1000.0 / MAX_RAD
OPEN_POSITION = tuple(value / SDK_SCALE for value in (0, 750, 0, 0, 0, 0))
CLOSED_POSITION = tuple(value / SDK_SCALE for value in (850, 750, 850, 1000, 1000, 1000))
DEFAULT_TOPICS = {
    "left": "/hand_left/left_revo2_hand_position_controller/commands",
    "right": "/hand_right/right_revo2_hand_position_controller/commands",
}


@dataclass(frozen=True)
class HandConfig:
    side: str
    topic: str
    feedback_topic: str
    joint_names: tuple[str, ...]
    opened: tuple[float, ...]
    closed: tuple[float, ...]


@dataclass(frozen=True)
class ControllerSample:
    received_s: float
    triggers: dict[str, float]


def _positions_from_env(name: str, default: Sequence[float]) -> tuple[float, ...]:
    raw = os.getenv(name, "").strip()
    values = tuple(float(value) for value in raw.split(",")) if raw else tuple(default)
    if len(values) != 6 or any(not math.isfinite(v) or not 0 <= v <= MAX_RAD for v in values):
        raise ValueError(f"{name} must contain six finite radians in [0, {MAX_RAD}]")
    return values


def _joint_names_from_env(name: str, side: str) -> tuple[str, ...]:
    default = tuple(
        f"{side}_{suffix}"
        for suffix in (
            "thumb_proximal_joint",
            "thumb_metacarpal_joint",
            "index_proximal_joint",
            "middle_proximal_joint",
            "ring_proximal_joint",
            "pinky_proximal_joint",
        )
    )
    raw = os.getenv(name, "").strip()
    values = tuple(value.strip() for value in raw.split(",")) if raw else default
    if len(values) != 6 or any(not value for value in values) or len(set(values)) != 6:
        raise ValueError(f"{name} must contain six unique joint names")
    return values


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float_alias(name: str, legacy_name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        raw = os.getenv(legacy_name)
    return float(raw) if raw is not None and raw.strip() else float(default)


def build_hand_configs(args: argparse.Namespace) -> tuple[HandConfig, ...]:
    sides = ("left", "right") if args.hands == "both" else (args.hands,)
    common_open = _positions_from_env("BOOSTER_HAND_OPEN_POSITION", OPEN_POSITION)
    common_closed = _positions_from_env("BOOSTER_HAND_CLOSED_POSITION", CLOSED_POSITION)
    configs = []
    for side in sides:
        prefix = f"BOOSTER_{side.upper()}_HAND"
        topic = getattr(args, f"{side}_topic") or os.getenv(f"{prefix}_TOPIC") or DEFAULT_TOPICS[side]
        if not topic.startswith("/"):
            raise ValueError(f"{side} hand topic must be an absolute ROS topic")
        feedback_topic = os.getenv(f"{prefix}_FEEDBACK_TOPIC") or f"/hand_{side}/joint_states"
        if not feedback_topic.startswith("/"):
            raise ValueError(f"{side} hand feedback topic must be an absolute ROS topic")
        configs.append(HandConfig(
            side=side,
            topic=topic,
            feedback_topic=feedback_topic,
            joint_names=_joint_names_from_env(f"{prefix}_JOINT_NAMES", side),
            opened=_positions_from_env(f"{prefix}_OPEN_POSITION", common_open),
            closed=_positions_from_env(f"{prefix}_CLOSED_POSITION", common_closed),
        ))
    if len({config.topic for config in configs}) != len(configs):
        raise ValueError("Left and right hand command topics must differ")
    return tuple(configs)


def trigger_positions(trigger: float, config: HandConfig, deadzone: float) -> list[float]:
    if not math.isfinite(trigger):
        raise ValueError("Pico trigger must be finite")
    ratio = min(max(trigger, 0.0), 1.0)
    if ratio < deadzone:
        ratio = 0.0
    return [opened + ratio * (closed - opened)
            for opened, closed in zip(config.opened, config.closed)]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hands", choices=("both", "left", "right"), default="both")
    parser.add_argument("--left-topic", help="Override the left hand command topic.")
    parser.add_argument("--right-topic", help="Override the right hand command topic.")
    parser.add_argument("--input-mode", choices=("auto", "callback", "polling"), default="auto",
                        help="auto prefers SDK callbacks, otherwise polls controller APIs only.")
    parser.add_argument("--publish-hz", type=float,
                        default=os.getenv("BOOSTER_HAND_PUBLISH_RATE_HZ", "50"))
    parser.add_argument("--stale-timeout", type=float,
                        default=os.getenv("BOOSTER_PICO_CONTROL_STALE_TIMEOUT", "0.5"),
                        help="Stop publishing after this many seconds without a new Pico timestamp.")
    parser.add_argument("--trigger-deadzone", type=float,
                        default=os.getenv("BOOSTER_PICO_TRIGGER_DEADZONE", "0.01"))
    parser.add_argument("--print-interval", type=float, default=1.0, help="Status interval in seconds.")
    parser.add_argument("--duration", type=float, default=0.0, help="Run for N seconds; 0 runs until Ctrl+C.")
    parser.add_argument("--qos-reliability", choices=("reliable", "best_effort"), default="reliable")
    parser.add_argument("--disable-stall-recovery", action="store_true",
                        help="Do not subscribe to hand feedback or apply stall backoff.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print mapping for trigger 0, 0.5, 1; import no ROS or native SDK.")
    args = parser.parse_args(argv)
    for name in ("publish_hz", "stale_timeout", "print_interval"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if not math.isfinite(args.duration) or args.duration < 0:
        parser.error("--duration must be finite and non-negative")
    if not math.isfinite(args.trigger_deadzone) or not 0 <= args.trigger_deadzone < 1:
        parser.error("--trigger-deadzone must be in [0, 1)")
    return args


def build_stall_guard(configs: Sequence[HandConfig]) -> HandStallGuard:
    # Allow running the standalone file without installing the project package.
    repo_root = str(Path(__file__).resolve().parents[2])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from booster_deploy.hand_retargeting.stall_recovery import (
        HandStallRecovery,
        HandStallRecoveryConfig,
    )

    def setting(suffix: str, default: float) -> float:
        return _env_float_alias(f"BOOSTER_HAND_TURBO_{suffix}",
                                f"BOOSTER_HAND_SOFT_TURBO_{suffix}", default)

    window_frames = setting("WINDOW_FRAMES", 5.0)
    if not math.isfinite(window_frames) or not window_frames.is_integer():
        raise ValueError("BOOSTER_HAND_TURBO_WINDOW_FRAMES must be an integer")
    recovery_config = HandStallRecoveryConfig(
        feedback_stale_s=setting("FEEDBACK_STALE_S", 0.15),
        window_frames=int(window_frames),
        stationary_tolerance=setting("STATIONARY_TOLERANCE", 1.0 / SDK_SCALE),
        stationary_duration_s=setting("STATIONARY_DURATION_S", 0.2),
        close_error_min=setting("CLOSE_ERROR_MIN", 10.0 / SDK_SCALE),
        backoff_units=setting("BACKOFF_UNITS", 1.0 / SDK_SCALE),
        retry_period_s=setting("RETRY_PERIOD_S", 0.5),
        command_min=0.0,
        command_max=MAX_RAD,
    )
    by_side = {config.side: config for config in configs}
    logger = logging.getLogger("pico_hand_control")
    recovery = HandStallRecovery(
        recovery_config,
        joint_names={side: config.joint_names for side, config in by_side.items()},
        open_positions={side: config.opened for side, config in by_side.items()},
        closed_positions={side: config.closed for side, config in by_side.items()},
        logger=logger,
    )
    return HandStallGuard(recovery, [config.side for config in configs], logger)


class HandStallGuard:
    """Connect JointState callbacks to recovery without using robot state."""

    def __init__(self, recovery: Any, sides: Sequence[str], logger: logging.Logger) -> None:
        self.recovery = recovery
        self.config = recovery.config
        self.sides = tuple(sides)
        self.logger = logger
        self.received_s: dict[str, float] = {}

    def on_feedback(self, side: str, msg: Any) -> None:
        now = time.monotonic()
        try:
            self.recovery.update_feedback(side, msg.name, msg.position, now_s=now)
        except (AttributeError, TypeError, ValueError, OverflowError) as exc:
            self.logger.warning("Ignoring invalid %s hand feedback: %s", side, exc)
            self.recovery.reset(side, clear_feedback=True)
            self.received_s.pop(side, None)
        else:
            self.received_s[side] = now

    def apply(self, side: str, positions: Sequence[float], *, now_s: float) -> list[float]:
        return self.recovery.apply(side, positions, now_s=now_s).tolist()

    def reset(self, side: str | None = None) -> None:
        self.recovery.reset(side)

    def feedback_status(self, now_s: float) -> str:
        states = []
        for side in self.sides:
            received = self.received_s.get(side)
            state = "waiting" if received is None else (
                "fresh" if now_s - received <= self.config.feedback_stale_s else "stale")
            states.append(f"{side}={state}")
        return "stall feedback: " + ", ".join(states)


class PicoControllerInput:
    """Read only controller fields; source timestamps distinguish cached SDK data."""

    def __init__(self, sdk: Any, sides: Sequence[str], mode: str) -> None:
        self.sdk = sdk
        self.sides = tuple(sides)
        callback_available = all(callable(getattr(sdk, name, None)) for name in
                                 ("register_frame_callback", "clear_frame_callback"))
        self.mode = ("callback" if callback_available else "polling") if mode == "auto" else mode
        required = ["init"]
        if self.mode == "callback":
            required += ["register_frame_callback", "clear_frame_callback"]
        else:
            required += ["get_time_stamp_ns"] + [f"get_{side}_trigger" for side in self.sides]
        missing = [name for name in required if not callable(getattr(sdk, name, None))]
        if missing:
            raise RuntimeError(f"Pico SDK is missing required APIs: {', '.join(missing)}")
        self._lock = threading.Lock()
        self._sample: ControllerSample | None = None
        self._timestamp: int | None = None
        self._accepting = False
        self.initialized = False

    def start(self) -> None:
        self.sdk.init()
        self.initialized = True
        self._accepting = True
        if self.mode == "callback":
            self.sdk.register_frame_callback(self._on_frame)

    def _accept(self, timestamp: int, triggers: dict[str, float]) -> None:
        if timestamp <= 0 or any(not math.isfinite(value) for value in triggers.values()):
            return
        with self._lock:
            if self._accepting and timestamp != self._timestamp:
                self._timestamp = timestamp
                self._sample = ControllerSample(time.monotonic(), triggers)

    def _on_frame(self, snapshot: Any) -> None:
        try:
            # Use the controller/headset timestamp, never the body timestamp.
            timestamp = int(snapshot["timestamp_ns"])
            triggers = {side: float(snapshot["controllers"][side]["trigger"]) for side in self.sides}
        except (KeyError, TypeError, ValueError, OverflowError):
            return
        self._accept(timestamp, triggers)

    def latest(self) -> ControllerSample | None:
        if self.mode == "polling":
            timestamp = int(self.sdk.get_time_stamp_ns())
            # Do not treat repeatedly polled cached values as fresh input.
            if timestamp > 0 and timestamp != self._timestamp:
                triggers = {side: float(getattr(self.sdk, f"get_{side}_trigger")()) for side in self.sides}
                self._accept(timestamp, triggers)
        with self._lock:
            return self._sample

    def stop(self) -> None:
        with self._lock:
            self._accepting = False
        if self.initialized and self.mode == "callback":
            self.sdk.clear_frame_callback()
        # The bundled SDK's close() holds the Python GIL during PXREADeinit,
        # while its native callback thread can be waiting for the same GIL.
        # The standalone CLI releases the native client at process exit instead.


def run_controller(args: argparse.Namespace, configs: Sequence[HandConfig],
                   source: PicoControllerInput, ros: Any, node: Any,
                   publishers: dict[str, Any], message_type: Any,
                   stop: threading.Event, recovery: Any = None) -> None:
    period = 1.0 / args.publish_hz
    started = time.monotonic()
    next_tick = started
    next_print = started
    previous_status = ""
    while ros.ok() and not stop.is_set():
        now = time.monotonic()
        if args.duration and now - started >= args.duration:
            break
        ros.spin_once(node, timeout_sec=0.0)
        sample = source.latest()
        now = time.monotonic()
        status = "waiting for Pico controllers"
        if sample is not None:
            status = "Pico input stale; publishing stopped"
            if now - sample.received_s <= args.stale_timeout:
                status = "active"
        missing = [config.topic for config in configs
                   if publishers[config.side].get_subscription_count() == 0]
        active = status == "active"
        if active and missing:
            status += " | waiting for ROS hand subscribers: " + ", ".join(missing)
        positions: dict[str, list[float]] = {}
        if active:
            for config in configs:
                if config.topic in missing:
                    if recovery is not None:
                        recovery.reset(config.side)
                    continue
                positions[config.side] = trigger_positions(
                    sample.triggers[config.side], config, args.trigger_deadzone)
                if recovery is not None:
                    positions[config.side] = recovery.apply(
                        config.side, positions[config.side], now_s=now)
            for config in configs:
                if config.side not in positions:
                    continue
                message = message_type()
                message.data = positions[config.side]
                publishers[config.side].publish(message)
        elif recovery is not None:
            # Pico disconnect/subscriber loss must discard pending backoff/retry.
            recovery.reset()
        if recovery is not None:
            status += " | " + recovery.feedback_status(now)
        if status != previous_status or now >= next_print:
            text = status
            if positions:
                text += " | " + " | ".join(
                    f"{side} trigger={sample.triggers[side]:.3f} position="
                    f"[{', '.join(f'{value:.4f}' for value in values)}]"
                    for side, values in positions.items())
            print(text, flush=True)
            previous_status = status
            next_print = now + args.print_interval
        next_tick = max(next_tick + period, time.monotonic())
        stop.wait(max(0.0, next_tick - time.monotonic()))


def main() -> int:
    args = parse_args()
    try:
        configs = build_hand_configs(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    for config in configs:
        print(f"{config.side}: {config.topic}\n"
              f"  open(rad)={list(config.opened)}\n"
              f"  closed(rad)={list(config.closed)}", flush=True)
    if args.dry_run:
        for trigger in (0.0, 0.5, 1.0):
            for config in configs:
                positions = trigger_positions(trigger, config, args.trigger_deadzone)
                print(f"{config.side} trigger={trigger:g} rad={positions} "
                      f"SDK-equivalent={[round(v * SDK_SCALE, 6) for v in positions]}", flush=True)
        return 0

    try:
        import rclpy
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
        from rclpy.signals import SignalHandlerOptions
        from std_msgs.msg import Float64MultiArray
        from sensor_msgs.msg import JointState
        # Match onboard runtime import order before loading the native XR library.
        import torch  # noqa: F401
        import xrobotoolkit_sdk as sdk
    except ImportError as exc:
        raise SystemExit("Missing onboard dependencies. Run: source scripts/env_set.sh onboard pico\n"
                         f"Import failed: {exc}") from exc

    stop = threading.Event()
    previous_signals = {sig: signal.signal(sig, lambda *_: stop.set())
                        for sig in (signal.SIGINT, signal.SIGTERM)}
    node = None
    source = None
    ros_initialized = False
    result = 0
    try:
        rclpy.init(args=[], signal_handler_options=SignalHandlerOptions.NO)
        ros_initialized = True
        node = rclpy.create_node("pico_hand_control")
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST, depth=1,
            reliability=(ReliabilityPolicy.RELIABLE if args.qos_reliability == "reliable"
                         else ReliabilityPolicy.BEST_EFFORT),
            durability=DurabilityPolicy.VOLATILE,
        )
        publishers = {config.side: node.create_publisher(Float64MultiArray, config.topic, qos)
                      for config in configs}
        recovery = None
        if not args.disable_stall_recovery and _env_bool(
                "BOOSTER_HAND_TURBO_ENABLED", _env_bool("BOOSTER_HAND_SOFT_TURBO", True)):
            logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
            recovery = build_stall_guard(configs)
            feedback_qos = QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=max(recovery.config.window_frames, 5),
                reliability=ReliabilityPolicy.BEST_EFFORT,
                durability=DurabilityPolicy.VOLATILE,
            )
            for config in configs:
                node.create_subscription(
                    JointState, config.feedback_topic,
                    lambda msg, side=config.side: recovery.on_feedback(side, msg),
                    feedback_qos,
                )
                print(f"{config.side} stall feedback: {config.feedback_topic} "
                      f"joints={list(config.joint_names)}", flush=True)
            print("Hand stall recovery enabled: one publish cycle of backoff, then retry. "
                  "Missing/stale feedback uses the original Pico target.", flush=True)
        else:
            print("Hand stall recovery disabled.", flush=True)
        source = PicoControllerInput(sdk, [config.side for config in configs], args.input_mode)
        source.start()
        print(f"Pico hand control started: SDK={source.mode}, rate={args.publish_hz:g}Hz. "
              "Connect Pico to this board; press triggers to move hands. Ctrl+C stops publishing.",
              flush=True)
        run_controller(args, configs, source, rclpy, node, publishers, Float64MultiArray, stop,
                       recovery)
    except Exception as exc:
        print(f"Pico hand control failed: {exc}", file=sys.stderr, flush=True)
        result = 1
    finally:
        try:
            if source is not None:
                source.stop()
        finally:
            if node is not None:
                node.destroy_node()
            if ros_initialized:
                rclpy.shutdown()
            for sig, handler in previous_signals.items():
                signal.signal(sig, handler)
        print("Hand publishing stopped; no final motion command sent.", flush=True)
    # Bypass native callback teardown after ROS cleanup (see PicoControllerInput.stop).
    if source is not None and source.initialized:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(result)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
