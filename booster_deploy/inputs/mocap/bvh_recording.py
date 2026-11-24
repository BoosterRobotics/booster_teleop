"""
Record PN-Link mocap RT raw local joint data as a BVH file.

The output stores the SDK BVH-space local offsets and local quaternions directly:
- local_pos is written in PN-Link centimeters.
- local_rot is interpreted as a wxyz quaternion and emitted as ZYX Euler channels.
- hand joints are part of the recorded hierarchy.
"""

from __future__ import annotations

import argparse
import os
import select
import sys
import termios
import time
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path
import tty
from typing import Any

import numpy as np

from .adapter import MOCAP_HIERARCHY
from .capture import MocapRtCapture, MocapRtCaptureConfig
from booster_deploy.types.mocap_types import MocapJoints
from booster_deploy.utils.project_path import resolve_project_path


_SYNTHETIC_BVH_JOINTS = frozenset(("Head1", "LeftTiptoe", "RightTiptoe"))
_DEFAULT_HIERARCHY: "OrderedDict[str, str]" = OrderedDict(
    (joint, parent)
    for joint, parent in MOCAP_HIERARCHY.items()
    if joint not in _SYNTHETIC_BVH_JOINTS
)
_DEFAULT_END_OFFSETS: dict[str, list[float]] = {
    "RightFoot": [0.0, -10.0, 15.12],
    "LeftFoot": [0.0, -10.0, 15.12],
    "Head": [0.0, 19.12, 0.0],
    "RightHandThumb3": [-2.0, 0.0, 0.0],
    "RightHandIndex3": [-2.0, 0.0, 0.0],
    "RightHandMiddle3": [-2.0, 0.0, 0.0],
    "RightHandRing3": [-2.0, 0.0, 0.0],
    "RightHandPinky3": [-2.0, 0.0, 0.0],
    "LeftHandThumb3": [2.0, 0.0, 0.0],
    "LeftHandIndex3": [2.0, 0.0, 0.0],
    "LeftHandMiddle3": [2.0, 0.0, 0.0],
    "LeftHandRing3": [2.0, 0.0, 0.0],
    "LeftHandPinky3": [2.0, 0.0, 0.0],
}


def _float_values(value: Any, width: int, default: list[float]) -> list[float]:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size < width or not np.all(np.isfinite(arr[:width])):
        return list(default)
    return [float(item) for item in arr[:width]]


def _quat_to_euler_zyx(w: float, x: float, y: float, z: float) -> tuple[float, float, float]:
    """Convert a wxyz quaternion to BVH Zrotation/Yrotation/Xrotation degrees."""

    quat = np.asarray([w, x, y, z], dtype=np.float64)
    norm = float(np.linalg.norm(quat))
    if norm <= 1e-12:
        w, x, y, z = 1.0, 0.0, 0.0, 0.0
    else:
        w, x, y, z = (quat / norm).tolist()

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return float(np.degrees(yaw)), float(np.degrees(pitch)), float(np.degrees(roll))


class BvhRecorder:
    """Accumulate PN-Link raw mocap joints and write a BVH file."""

    def __init__(
        self,
        hierarchy: Mapping[str, str] | None = None,
        *,
        target_fps: float = 50.0,
    ) -> None:
        if target_fps <= 0.0:
            raise ValueError(f"target_fps must be > 0, got {target_fps}")

        self._hierarchy = OrderedDict((str(joint), str(parent)) for joint, parent in (hierarchy or _DEFAULT_HIERARCHY).items())
        self._target_fps = float(target_fps)
        self._joint_order: list[str] = []
        self._children: dict[str, list[str]] = {}
        for joint, parent in self._hierarchy.items():
            self._children.setdefault(parent, [])
            self._children.setdefault(joint, [])
            self._children[parent].append(joint)

        roots = [joint for joint, parent in self._hierarchy.items() if parent == "world"]
        if len(roots) != 1:
            raise ValueError(f"Expected exactly one BVH root, got {roots}")
        self._root_name = roots[0]
        self._dfs_order(self._root_name)

        self._frames: list[dict[str, dict[str, list[float]]]] = []
        self._offsets: dict[str, list[float]] | None = None

    @property
    def frame_count(self) -> int:
        return len(self._frames)

    def add_frame(self, joints_data: Mapping[str, Mapping[str, Any]], timestamp_s: float | None = None) -> None:
        """Record one BVH sample. The timestamp is accepted for call-site symmetry and not written."""

        del timestamp_s
        if self._offsets is None:
            self._offsets = {
                joint: _float_values(
                    joints_data.get(joint, {}).get("local_pos", [0.0, 0.0, 0.0]),
                    3,
                    [0.0, 0.0, 0.0],
                )
                for joint in self._joint_order
            }

        frame: dict[str, dict[str, list[float]]] = {}
        for joint in self._joint_order:
            joint_data = joints_data.get(joint)
            if joint_data is None:
                continue
            frame[joint] = {
                "local_pos": _float_values(joint_data.get("local_pos", [0.0, 0.0, 0.0]), 3, [0.0, 0.0, 0.0]),
                "local_rot": _float_values(joint_data.get("local_rot", [1.0, 0.0, 0.0, 0.0]), 4, [1.0, 0.0, 0.0, 0.0]),
            }
        self._frames.append(frame)

    def save(self, output_path: str | Path) -> Path | None:
        """Write accumulated data as a standard BVH file."""

        if not self._frames or self._offsets is None:
            print("[BvhRecorder] No frames to save", flush=True)
            return None

        path = resolve_project_path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        lines: list[str] = ["HIERARCHY"]
        self._write_joint(lines, self._root_name, 0, is_root=True)
        lines.append("MOTION")
        lines.append(f"Frames: {len(self._frames)}")
        lines.append(f"Frame Time: {1.0 / self._target_fps:.6f}")

        for frame_data in self._frames:
            values: list[float] = []
            for joint in self._joint_order:
                joint_data = frame_data.get(joint, {})
                rot = joint_data.get("local_rot", [1.0, 0.0, 0.0, 0.0])
                zr, yr, xr = _quat_to_euler_zyx(rot[0], rot[1], rot[2], rot[3])
                if joint == self._root_name:
                    pos = joint_data.get("local_pos", [0.0, 0.0, 0.0])
                    values.extend([pos[0], pos[1], pos[2], zr, yr, xr])
                else:
                    values.extend([zr, yr, xr])
            lines.append(" ".join(f"{value:.6f}" for value in values))

        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"[BvhRecorder] Saved {len(self._frames)} frames to {path}", flush=True)
        return path

    def _dfs_order(self, joint: str) -> None:
        self._joint_order.append(joint)
        for child in self._children.get(joint, []):
            self._dfs_order(child)

    def _write_joint(self, lines: list[str], joint: str, depth: int, *, is_root: bool = False) -> None:
        if self._offsets is None:
            raise RuntimeError("Cannot write BVH hierarchy before the first frame sets offsets")

        indent = "    " * depth
        child_indent = "    " * (depth + 1)
        offset = self._offsets.get(joint, [0.0, 0.0, 0.0])

        lines.append(f"ROOT {joint}" if is_root else f"{indent}JOINT {joint}")
        lines.append(f"{indent}{{")
        lines.append(f"{child_indent}OFFSET {offset[0]:.6f} {offset[1]:.6f} {offset[2]:.6f}")
        if is_root:
            lines.append(f"{child_indent}CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation")
        else:
            lines.append(f"{child_indent}CHANNELS 3 Zrotation Yrotation Xrotation")

        children = self._children.get(joint, [])
        for child in children:
            self._write_joint(lines, child, depth + 1)

        if not children:
            end_offset = _DEFAULT_END_OFFSETS.get(joint, [0.0, 0.0, 0.0])
            lines.append(f"{child_indent}End Site")
            lines.append(f"{child_indent}{{")
            lines.append(f"{child_indent}    OFFSET {end_offset[0]:.6f} {end_offset[1]:.6f} {end_offset[2]:.6f}")
            lines.append(f"{child_indent}}}")

        lines.append(f"{indent}}}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record PN-Link raw local joints to BVH. Press terminal A to start; B saves and exits (no Enter needed)."
    )
    parser.add_argument("--output", required=True, help="Output BVH path.")
    parser.add_argument("--frames", type=int, default=0, help="Number of BVH samples to write. Use <=0 for unlimited.")
    parser.add_argument("--duration", type=float, default=0.0, help="Recording duration after A is pressed, in seconds. Use <=0 for unlimited.")
    parser.add_argument("--target-fps", type=float, default=50.0, help="BVH sampling FPS and Frame Time metadata.")
    parser.add_argument("--timeout", type=float, default=30.0, help="Initial raw body receive timeout in seconds.")
    parser.add_argument("--print-interval", type=int, default=100, help="Print status every N BVH samples.")
    parser.add_argument("--local-ip", default="10.42.0.1", help="Local NIC IP used by PN-Link UDP receive.")
    parser.add_argument("--local-port", type=int, default=8002, help="Local UDP receive port.")
    parser.add_argument("--server-ip", default="10.42.0.202", help="PN-Link server IP.")
    parser.add_argument("--server-port", type=int, default=8080, help="PN-Link server port.")
    parser.add_argument("--poll-hz", type=float, default=500.0, help="Internal SDK polling frequency in Hz.")
    return parser.parse_args(argv)


def _duration_remaining_s(start_time_s: float, duration_s: float) -> float | None:
    if duration_s <= 0.0:
        return None
    return max(float(duration_s) - (time.monotonic() - start_time_s), 0.0)


def _latest_raw_joints(body_frame) -> MocapJoints:
    return body_frame.raw_joints


class TerminalKeyReader:
    """Read single keys without requiring Enter and restore the terminal on exit."""

    def __init__(self) -> None:
        self._fd: int | None = None
        self._settings = None

    def __enter__(self) -> "TerminalKeyReader":
        if not sys.stdin.isatty():
            raise RuntimeError("BVH recording requires an interactive terminal for A/B controls")
        self._fd = sys.stdin.fileno()
        self._settings = termios.tcgetattr(self._fd)
        try:
            tty.setcbreak(self._fd)
        except BaseException:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._settings)
            raise
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self._fd is not None and self._settings is not None:
            try:
                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._settings)
            except (OSError, termios.error):
                pass
            self._fd = None

    def read_key(self, timeout_s: float) -> str | None:
        if self._fd is None:
            raise RuntimeError("TerminalKeyReader must be entered before reading keys")
        ready, _, _ = select.select([self._fd], [], [], max(float(timeout_s), 0.0))
        if not ready:
            return None
        key = os.read(self._fd, 1)
        if not key:
            raise EOFError("Terminal input closed during BVH recording")
        return key.decode(errors="ignore").lower()


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.target_fps <= 0.0:
        raise ValueError("--target-fps must be > 0")

    output_path = resolve_project_path(args.output)
    recorder = BvhRecorder(target_fps=float(args.target_fps))
    capture: MocapRtCapture | None = None

    sample_period_s = 1.0 / float(args.target_fps)
    recording_start_s: float | None = None
    next_sample_s: float | None = None
    latest_joints: MocapJoints | None = None
    last_seq: int | None = None
    interrupted = False

    print(
        f"Recording PN-Link raw BVH output={output_path} target_fps={args.target_fps:.3f} "
        f"local={args.local_ip}:{args.local_port} server={args.server_ip}:{args.server_port}",
        flush=True,
    )
    try:
        with TerminalKeyReader() as keyboard:
            capture = MocapRtCapture(
                MocapRtCaptureConfig(
                    local_ip=args.local_ip,
                    local_port=args.local_port,
                    server_ip=args.server_ip,
                    server_port=args.server_port,
                    poll_hz=args.poll_hz,
                    timeout_s=float(args.timeout),
                    max_cached_frames=1,
                )
            )
            capture.start()
            print("Press A to start recording; press B to save and exit.", flush=True)
            next_body_timeout = time.monotonic() + max(float(args.timeout), 0.001)
            while True:
                if args.frames > 0 and recorder.frame_count >= args.frames:
                    print(f"Reached frame limit ({args.frames}); stopping recording.", flush=True)
                    break
                now_s = time.monotonic()
                wait_s = 0.05
                if recording_start_s is not None and next_sample_s is not None:
                    wait_s = min(wait_s, max(next_sample_s - now_s, 0.0))
                if recording_start_s is not None:
                    remaining = _duration_remaining_s(recording_start_s, float(args.duration))
                    if remaining is not None:
                        wait_s = min(wait_s, remaining)
                key = keyboard.read_key(wait_s)

                if recording_start_s is None:
                    if key == "a":
                        recording_start_s = time.monotonic()
                        next_sample_s = recording_start_s
                        last_seq = capture.body_seq
                        next_body_timeout = recording_start_s + max(float(args.timeout), 0.001)
                        print(f"A pressed; recording started at {args.target_fps:.1f} Hz.", flush=True)
                    continue

                if key == "b":
                    print("B pressed; saving and exiting.", flush=True)
                    break

                remaining_duration = _duration_remaining_s(recording_start_s, float(args.duration))
                if remaining_duration is not None and remaining_duration <= 0.0:
                    break

                body = capture.latest_body()
                if body is not None and int(body.seq) != last_seq:
                    last_seq = int(body.seq)
                    if int(body.body_recv_ns) >= int(recording_start_s * 1.0e9):
                        latest_joints = _latest_raw_joints(body)
                        if recorder.frame_count == 0:
                            next_sample_s = time.monotonic()
                        next_body_timeout = time.monotonic() + max(float(args.timeout), 0.001)
                elif latest_joints is None and time.monotonic() >= next_body_timeout:
                    print(
                        f"Waiting for PN-Link raw body data | body_seq={capture.body_seq} "
                        f"callback_count={capture.callback_count}",
                        flush=True,
                    )
                    next_body_timeout = time.monotonic() + max(float(args.timeout), 0.001)

                now_s = time.monotonic()
                if latest_joints is None:
                    next_sample_s = now_s + min(sample_period_s, 0.05)
                elif next_sample_s is not None and now_s >= next_sample_s:
                    recorder.add_frame(latest_joints, now_s)
                    if args.print_interval > 0 and recorder.frame_count % int(args.print_interval) == 0:
                        elapsed_s = max(now_s - recording_start_s, 1e-9)
                        print(
                            f"recorded={recorder.frame_count} fps={recorder.frame_count / elapsed_s:.2f} "
                            f"latest_seq={last_seq}",
                            flush=True,
                        )

                    next_sample_s += sample_period_s
                    if next_sample_s < now_s - sample_period_s:
                        next_sample_s = now_s
    except KeyboardInterrupt:
        interrupted = True
        print("\nInterrupted; saving partial BVH recording.", flush=True)
    finally:
        try:
            if recorder.frame_count > 0:
                recorder.save(output_path)
            else:
                print("No frames recorded; output file was not written.", flush=True)
        finally:
            if capture is not None:
                capture.close(close_sdk=not interrupted)
