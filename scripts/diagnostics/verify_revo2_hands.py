#!/usr/bin/env python3
"""Verify BrainCo Revo2 hands through ROS 2 commands."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import time
from typing import Any, Sequence


LEFT_POSITION_TOPIC = "/hand_left/left_revo2_hand_position_controller/commands"
RIGHT_POSITION_TOPIC = "/hand_right/right_revo2_hand_position_controller/commands"
LEFT_JOINT_STATE_TOPIC = "/hand_left/joint_commands"
RIGHT_JOINT_STATE_TOPIC = "/hand_right/joint_commands"
LEFT_POSITION_HAND_NAME = "left"
RIGHT_POSITION_HAND_NAME = "right"
LEFT_JOINT_STATE_HAND_NAME = "hand_left"
RIGHT_JOINT_STATE_HAND_NAME = "hand_right"
PROFILE_TITLE = "BrainCo Revo2 hand"
POSITION_PROFILE_UNIT = "radians, 0 open and [0.8, 0.1, 1.4, 1.4, 1.4, 1.4] closed"
JOINT_STATE_PROFILE_UNIT = "raw SDK units, 0 open and 1000 closed"
COMMAND_INTERFACE_DEFAULT = "position_controller"
POSITION_JOINT_SUFFIXES = (
    "thumb_proximal_joint",
    "thumb_metacarpal_joint",
    "index_proximal_joint",
    "middle_proximal_joint",
    "ring_proximal_joint",
    "pinky_proximal_joint",
)
JOINT_STATE_SUFFIXES = (
    "thumb_joint",
    "thumb_aux_joint",
    "index_joint",
    "middle_joint",
    "ring_joint",
    "pinky_joint",
)
OPEN_POSITIONS = (0.0,) * len(POSITION_JOINT_SUFFIXES)
POSITION_CLOSE_POSITIONS = (0.8, 0.1, 1.4, 1.4, 1.4, 1.4)
JOINT_STATE_CLOSE_POSITIONS = (1000.0,) * len(JOINT_STATE_SUFFIXES)
PUBLISH_NAMES_DEFAULT = True


@dataclass(frozen=True, slots=True)
class FingerGroup:
    label: str
    indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class PlannedCommand:
    side: str
    label: str
    topic: str
    names: tuple[str, ...]
    indices: tuple[int, ...]


GROUPS = (
    FingerGroup("thumb", (0, 1)),
    FingerGroup("index", (2,)),
    FingerGroup("middle", (3,)),
    FingerGroup("ring", (4,)),
    FingerGroup("pinky", (5,)),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"Close and reopen {PROFILE_TITLE} digits one at a time."
    )
    parser.add_argument(
        "--command-interface",
        choices=("position_controller", "joint_state"),
        default=COMMAND_INTERFACE_DEFAULT,
        help=(
            "Command interface to use. 'position_controller' publishes "
            "std_msgs/msg/Float64MultiArray to ros2_control; 'joint_state' "
            "publishes sensor_msgs/msg/JointState to the legacy bridge."
        ),
    )
    parser.add_argument("--left-topic", default=None, help="Override the left command topic.")
    parser.add_argument("--right-topic", default=None, help="Override the right command topic.")
    parser.add_argument("--left-hand-name", default=None, help="Override the left joint-name prefix.")
    parser.add_argument("--right-hand-name", default=None, help="Override the right joint-name prefix.")
    parser.add_argument(
        "--hands",
        default="both",
        choices=("both", "left", "right"),
        help="Which side to test.",
    )
    parser.add_argument(
        "--close-position",
        type=float,
        default=None,
        help="Override every close target in the selected command-interface units.",
    )
    parser.add_argument(
        "--open-position",
        type=float,
        default=None,
        help="Override every open target in the selected command-interface units.",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=None,
        help="Optional JointState velocity value for every commanded joint.",
    )
    parser.add_argument(
        "--effort",
        type=float,
        default=None,
        help="Optional JointState effort value for every commanded joint.",
    )
    parser.add_argument(
        "--hold-seconds",
        type=float,
        default=1.0,
        help="Time to hold after each close and open command.",
    )
    parser.add_argument(
        "--publish-count",
        type=int,
        default=3,
        help="Number of repeated publishes for every command.",
    )
    parser.add_argument(
        "--publish-interval",
        type=float,
        default=0.05,
        help="Seconds between repeated publishes.",
    )
    parser.add_argument(
        "--discovery-timeout",
        type=float,
        default=5.0,
        help="Seconds to wait for subscribers on selected topics.",
    )
    parser.add_argument(
        "--qos-reliability",
        choices=("reliable", "best_effort"),
        default="best_effort",
        help="Publisher reliability QoS.",
    )
    parser.add_argument(
        "--publish-names",
        action=argparse.BooleanOptionalAction,
        default=PUBLISH_NAMES_DEFAULT,
        help="Publish JointState.name.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the interactive real-robot safety confirmation.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the sequence without importing ROS 2 or publishing.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only create publishers and wait for subscribers; publish no motion commands.",
    )
    parser.add_argument(
        "--verbose-positions",
        action="store_true",
        help="Also print the full 6-joint close/open arrays for each command.",
    )
    args = parser.parse_args()
    _validate_args(args, parser)
    return args


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.publish_count < 1:
        parser.error("--publish-count must be at least 1")
    for field in (
        "close_position",
        "open_position",
        "speed",
        "effort",
        "hold_seconds",
        "publish_interval",
        "discovery_timeout",
    ):
        value = getattr(args, field)
        if value is None:
            continue
        if not math.isfinite(float(value)):
            parser.error(f"--{field.replace('_', '-')} must be finite")
        if field in {"hold_seconds", "publish_interval", "discovery_timeout"} and value < 0.0:
            parser.error(f"--{field.replace('_', '-')} must be non-negative")
    if args.speed is not None and args.speed < 0.0:
        parser.error("--speed must be non-negative")
    if args.command_interface == "position_controller":
        for field in ("close_position", "open_position"):
            value = getattr(args, field)
            if value is not None and not 0.0 <= value <= 1.6:
                parser.error(f"--{field.replace('_', '-')} must be in the range 0..1.6 radians")
    for field in ("left_topic", "right_topic"):
        topic = getattr(args, field)
        if topic is not None and not str(topic).startswith("/"):
            parser.error(f"--{field.replace('_', '-')} must be an absolute ROS topic name")


def _selected_sides(args: argparse.Namespace) -> tuple[str, ...]:
    if args.hands == "left":
        return ("left",)
    if args.hands == "right":
        return ("right",)
    return ("left", "right")


def _hand_name(args: argparse.Namespace, side: str) -> str:
    hand_name = args.left_hand_name if side == "left" else args.right_hand_name
    if hand_name is not None:
        return hand_name
    if args.command_interface == "position_controller":
        return LEFT_POSITION_HAND_NAME if side == "left" else RIGHT_POSITION_HAND_NAME
    return LEFT_JOINT_STATE_HAND_NAME if side == "left" else RIGHT_JOINT_STATE_HAND_NAME


def _topic(args: argparse.Namespace, side: str) -> str:
    topic = args.left_topic if side == "left" else args.right_topic
    if topic is not None:
        return topic
    if args.command_interface == "position_controller":
        return LEFT_POSITION_TOPIC if side == "left" else RIGHT_POSITION_TOPIC
    return LEFT_JOINT_STATE_TOPIC if side == "left" else RIGHT_JOINT_STATE_TOPIC


def _joint_suffixes(args: argparse.Namespace) -> tuple[str, ...]:
    if args.command_interface == "position_controller":
        return POSITION_JOINT_SUFFIXES
    return JOINT_STATE_SUFFIXES


def _positions(base: Sequence[float], override: float | None) -> tuple[float, ...]:
    if override is None:
        return tuple(base)
    return tuple(float(override) for _ in base)


def _plan(args: argparse.Namespace) -> tuple[PlannedCommand, ...]:
    commands = []
    for side in _selected_sides(args):
        names = tuple(f"{_hand_name(args, side)}_{suffix}" for suffix in _joint_suffixes(args))
        for group in GROUPS:
            commands.append(
                PlannedCommand(
                    side=side,
                    label=group.label,
                    topic=_topic(args, side),
                    names=names,
                    indices=group.indices,
                )
            )
    return tuple(commands)


def _base_positions(args: argparse.Namespace, *, closed: bool) -> tuple[float, ...]:
    close_positions = (
        POSITION_CLOSE_POSITIONS
        if args.command_interface == "position_controller"
        else JOINT_STATE_CLOSE_POSITIONS
    )
    return _positions(
        close_positions if closed else OPEN_POSITIONS,
        args.close_position if closed else args.open_position,
    )


def _target_positions(
    args: argparse.Namespace,
    command: PlannedCommand,
    *,
    closed: bool,
) -> tuple[float, ...]:
    positions = list(_base_positions(args, closed=False))
    if closed:
        close_positions = _base_positions(args, closed=True)
        for index in command.indices:
            positions[index] = close_positions[index]
    return tuple(positions)


def _format_position_values(values: Sequence[float], indices: Sequence[int]) -> str:
    selected = [values[index] for index in indices]
    return "[" + ", ".join(f"{value:.6g}" for value in selected) + "]"


def _format_all_position_values(values: Sequence[float]) -> str:
    return "[" + ", ".join(f"{value:.6g}" for value in values) + "]"


def _print_sequence(args: argparse.Namespace, commands: Sequence[PlannedCommand]) -> None:
    total = len(commands)
    unit = (
        POSITION_PROFILE_UNIT
        if args.command_interface == "position_controller"
        else JOINT_STATE_PROFILE_UNIT
    )
    print(f"Profile: {PROFILE_TITLE} ({unit})", flush=True)
    if args.command_interface == "position_controller":
        print("Command interface: std_msgs/msg/Float64MultiArray (ros2_control position)", flush=True)
    else:
        print("Command interface: sensor_msgs/msg/JointState (legacy joint_commands)", flush=True)
        print(f"JointState.name publishing: {'on' if args.publish_names else 'off'}", flush=True)
    open_positions = _base_positions(args, closed=False)
    close_positions = _base_positions(args, closed=True)
    for index, command in enumerate(commands, start=1):
        command_close_positions = _target_positions(args, command, closed=True)
        command_open_positions = _target_positions(args, command, closed=False)
        joint_names = ", ".join(command.names[joint_i] for joint_i in command.indices)
        print(
            f"[{index:02d}/{total:02d}] {command.side:>5} {command.label:<7} "
            f"topic={command.topic} joints=[{joint_names}] "
            f"close={_format_position_values(close_positions, command.indices)} "
            f"-> open={_format_position_values(open_positions, command.indices)}",
            flush=True,
        )
        if args.verbose_positions:
            print(
                f"           publish_close={_format_all_position_values(command_close_positions)}",
                flush=True,
            )
            print(
                f"           publish_open ={_format_all_position_values(command_open_positions)}",
                flush=True,
            )


def _confirm_real_robot(sides: Sequence[str]) -> None:
    side_text = ", ".join(sides)
    print(f"WARNING: This script will move real {PROFILE_TITLE} hardware on: {side_text}.")
    print("Remove hands, tools, cables, and other objects from the finger workspace.")
    answer = input("Type YES to start the verification test: ").strip()
    if answer != "YES":
        raise SystemExit("Cancelled; no ROS commands were published.")


def _make_command_message(
    message_type: Any,
    node: Any,
    command: PlannedCommand,
    args: argparse.Namespace,
    *,
    closed: bool,
) -> Any:
    message = message_type()
    if args.command_interface == "position_controller":
        message.data = list(_target_positions(args, command, closed=closed))
        return message

    message.header.stamp = node.get_clock().now().to_msg()
    if args.publish_names:
        message.name = list(command.names)
    message.position = list(_target_positions(args, command, closed=closed))
    if args.speed is not None:
        message.velocity = [float(args.speed)] * len(message.position)
    if args.effort is not None:
        message.effort = [float(args.effort)] * len(message.position)
    return message


def _publish_repeated(
    publisher: Any,
    message: Any,
    *,
    publish_count: int,
    publish_interval: float,
) -> None:
    for publish_index in range(publish_count):
        publisher.publish(message)
        if publish_index + 1 < publish_count and publish_interval > 0.0:
            time.sleep(publish_interval)


def _wait_for_subscribers(
    node: Any,
    publishers: dict[str, Any],
    topics: dict[str, str],
    *,
    timeout: float,
    rclpy_module: Any,
) -> None:
    deadline = time.monotonic() + timeout
    while True:
        missing = [
            side
            for side, publisher in publishers.items()
            if publisher.get_subscription_count() < 1
        ]
        if not missing:
            counts = ", ".join(
                f"{side}={publishers[side].get_subscription_count()}" for side in publishers
            )
            print(f"ROS subscribers discovered: {counts}", flush=True)
            return
        if time.monotonic() >= deadline:
            missing_topics = ", ".join(topics[side] for side in missing)
            raise RuntimeError(
                f"No subscriber discovered within {timeout:g}s for: {missing_topics}. "
                "Check the hand driver or bridge and `ros2 topic info -v` before retrying."
            )
        rclpy_module.spin_once(node, timeout_sec=0.05)


def _publish_all_open(
    message_type: Any,
    node: Any,
    publishers: dict[str, Any],
    args: argparse.Namespace,
    commands: Sequence[PlannedCommand],
) -> None:
    seen: set[str] = set()
    for command in commands:
        if command.side in seen:
            continue
        seen.add(command.side)
        message = _make_command_message(message_type, node, command, args, closed=False)
        _publish_repeated(
            publishers[command.side],
            message,
            publish_count=args.publish_count,
            publish_interval=args.publish_interval,
        )


def main() -> int:
    args = parse_args()
    commands = _plan(args)
    _print_sequence(args, commands)

    if args.dry_run:
        print("Dry run complete; no ROS commands were published.", flush=True)
        return 0
    if not args.yes:
        _confirm_real_robot(_selected_sides(args))

    try:
        import rclpy
        from rclpy.qos import (
            DurabilityPolicy,
            HistoryPolicy,
            QoSProfile,
            ReliabilityPolicy,
        )
        if args.command_interface == "position_controller":
            from std_msgs.msg import Float64MultiArray
            message_type = Float64MultiArray
        else:
            from sensor_msgs.msg import JointState
            message_type = JointState
    except ImportError as exc:
        raise SystemExit(
            "ROS 2 Python interfaces are unavailable. Source the Booster ROS 2 environment first."
        ) from exc

    rclpy.init()
    node = rclpy.create_node("verify_revo2_hands")
    reliability = (
        ReliabilityPolicy.BEST_EFFORT
        if args.qos_reliability == "best_effort"
        else ReliabilityPolicy.RELIABLE
    )
    qos = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=reliability,
        durability=DurabilityPolicy.VOLATILE,
    )
    sides = _selected_sides(args)
    topics = {side: _topic(args, side) for side in sides}
    publishers = {
        side: node.create_publisher(message_type, topic, qos)
        for side, topic in topics.items()
    }
    completed = False
    published_any = False

    try:
        _wait_for_subscribers(
            node,
            publishers,
            topics,
            timeout=args.discovery_timeout,
            rclpy_module=rclpy,
        )
        if args.check_only:
            print("Subscriber check complete; no motion commands were published.", flush=True)
            return 0
        total = len(commands)
        for index, command in enumerate(commands, start=1):
            publisher = publishers[command.side]
            print(f"[{index:02d}/{total:02d}] CLOSE {command.side} {command.label}", flush=True)
            published_any = True
            _publish_repeated(
                publisher,
                _make_command_message(message_type, node, command, args, closed=True),
                publish_count=args.publish_count,
                publish_interval=args.publish_interval,
            )
            time.sleep(args.hold_seconds)

            print(f"[{index:02d}/{total:02d}] OPEN  {command.side} {command.label}", flush=True)
            _publish_repeated(
                publisher,
                _make_command_message(message_type, node, command, args, closed=False),
                publish_count=args.publish_count,
                publish_interval=args.publish_interval,
            )
            time.sleep(args.hold_seconds)
        completed = True
    except KeyboardInterrupt:
        print("Interrupted by user.", flush=True)
        return 130
    finally:
        if published_any:
            try:
                print("Sending final all-open command to selected hands...", flush=True)
                _publish_all_open(message_type, node, publishers, args, commands)
                time.sleep(max(args.publish_interval, 0.1))
            except Exception as exc:
                print(f"WARNING: final all-open command failed: {exc}", flush=True)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    if completed:
        print(f"{PROFILE_TITLE} close/open verification sequence completed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
