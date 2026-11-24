"""
Publish recorded PN-Link mocap BVH data as realtime mocap_rt ZMQ input.

Input BVH files are expected to come from scripts/bvh_recorder.py. The script
reconstructs PN-Link raw local joints from BVH local offsets/quaternions, runs the
same mocap_rt adapter used by the live capture path, and publishes
MocapRetargetInputFrameNs payloads on the same topic/port defaults as
scripts/mocap_rt_zmq_publisher.py.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import math
import os
import re
import time

import numpy as np

from .adapter import (
    MOCAP_HIERARCHY,
    PNLinkMocapAdapter,
    add_synthetic_tip_joints,
)
from .codecs import encode_mocap_retarget_input
from booster_deploy.inputs.control.start_gate import RobotButtonMonitor
from booster_deploy.retargeting.gmr.utils.lafan_vendor.extract import read_bvh
from booster_deploy.types.mocap_types import MocapJoints, MocapRetargetInputFrameNs
from booster_deploy.utils.project_path import resolve_project_path


_SYNTHETIC_BVH_JOINTS = frozenset(("Head1", "LeftTiptoe", "RightTiptoe"))
_REQUIRED_RECORDED_BVH_JOINTS = tuple(
    joint for joint in MOCAP_HIERARCHY if joint not in _SYNTHETIC_BVH_JOINTS
)


@dataclass(frozen=True, slots=True)
class RecordedMocapBvh:
    path: Path
    frame_time_s: float
    bone_names: tuple[str, ...]
    local_pos_cm: np.ndarray
    local_quat_wxyz: np.ndarray

    @property
    def frame_count(self) -> int:
        return int(self.local_pos_cm.shape[0])

    @property
    def fps(self) -> float:
        return 1.0 / self.frame_time_s


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Publish scripts/bvh_recorder.py BVH files as mocap_rt.retarget_input ZMQ frames."
    )
    parser.add_argument("bvh", help="Input BVH path recorded by scripts/bvh_recorder.py.")
    parser.add_argument("--bind", default="0.0.0.0", help="ZMQ bind address, matching mocap_rt_zmq_publisher.py.")
    parser.add_argument("--port", type=int, default=5557, help="ZMQ PUB port, matching mocap RT default.")
    parser.add_argument(
        "--retarget-topic",
        default="mocap_rt.retarget_input",
        help="Mocap RT retarget-input topic.",
    )
    parser.add_argument("--send-hwm", type=int, default=1, help="ZMQ send high-water mark.")
    parser.add_argument("--human-format", default="bvh", help="Published MocapRetargetInputFrameNs.human_format.")
    parser.add_argument("--fps", type=float, default=0.0, help="Override publish FPS. Use <=0 to use BVH Frame Time.")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier applied to selected FPS.")
    parser.add_argument("--start-frame", type=int, default=0, help="First BVH frame index to publish.")
    parser.add_argument("--end-frame", type=int, default=None, help="Exclusive BVH frame index to stop at.")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Maximum frames to publish. Use <=0 for unlimited or until clip end with --no-loop.",
    )
    parser.add_argument(
        "--loop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Loop the selected frame range to emulate a continuous mocap source.",
    )
    parser.add_argument(
        "--start-delay",
        type=float,
        default=0.0,
        help="Seconds to wait after binding before the first publish, useful for SUB connection warmup.",
    )
    parser.add_argument(
        "--wait-for-start-button",
        action="store_true",
        help=(
            "Replay one cycle per robot controller A press on /remote_controller_state; "
            "B aborts the cycle in progress. Implies --no-loop."
        ),
    )
    parser.add_argument("--print-interval", type=int, default=300, help="Print status every N published frames.")
    return parser.parse_args(argv)


def _parse_bvh_frame_time(path: Path) -> float:
    pattern = re.compile(r"^\s*Frame\s+Time:\s*([0-9.+\-eE]+)\s*$")
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            match = pattern.match(line)
            if match:
                frame_time_s = float(match.group(1))
                if frame_time_s <= 0.0:
                    raise ValueError(f"{path} has non-positive Frame Time: {frame_time_s}")
                return frame_time_s
    raise ValueError(f"{path} missing BVH 'Frame Time:' line")


def _normalize_quat_array(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64).copy()
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    ok = norm > 1e-12
    quat = np.divide(quat, norm, out=np.zeros_like(quat), where=ok)
    quat[..., 0] = np.where(ok[..., 0], quat[..., 0], 1.0)
    return quat


def _validate_recorded_bvh_bones(path: Path, bone_names: tuple[str, ...]) -> None:
    present = set(bone_names)
    missing = [joint for joint in _REQUIRED_RECORDED_BVH_JOINTS if joint not in present]
    if missing:
        preview = ", ".join(missing[:12])
        suffix = "" if len(missing) <= 12 else f", ... ({len(missing)} total)"
        raise ValueError(
            f"{path} is missing required recorded mocap BVH joints: {preview}{suffix}. "
            "Use BVH files produced by scripts/bvh_recorder.py."
        )


def load_recorded_mocap_bvh(path: str | Path) -> RecordedMocapBvh:
    bvh_path = resolve_project_path(path)
    if not bvh_path.exists():
        raise FileNotFoundError(f"BVH file not found: {bvh_path}")

    frame_time_s = _parse_bvh_frame_time(bvh_path)
    data = read_bvh(str(bvh_path))
    bone_names = tuple(str(name) for name in data.bones)
    _validate_recorded_bvh_bones(bvh_path, bone_names)

    local_pos_cm = np.asarray(data.pos, dtype=np.float64)
    local_quat_wxyz = _normalize_quat_array(np.asarray(data.quats, dtype=np.float64))
    if local_pos_cm.ndim != 3 or local_pos_cm.shape[-1] != 3:
        raise ValueError(f"{bvh_path} local positions must have shape [T, N, 3], got {local_pos_cm.shape}")
    if local_quat_wxyz.ndim != 3 or local_quat_wxyz.shape[-1] != 4:
        raise ValueError(f"{bvh_path} local quaternions must have shape [T, N, 4], got {local_quat_wxyz.shape}")
    if local_pos_cm.shape[:2] != local_quat_wxyz.shape[:2]:
        raise ValueError(
            f"{bvh_path} local position/quaternion shapes differ: "
            f"{local_pos_cm.shape} vs {local_quat_wxyz.shape}"
        )
    if local_pos_cm.shape[1] != len(bone_names):
        raise ValueError(f"{bvh_path} has {len(bone_names)} bones but local_pos width {local_pos_cm.shape[1]}")
    if local_pos_cm.shape[0] <= 0:
        raise ValueError(f"{bvh_path} contains no motion frames")

    return RecordedMocapBvh(
        path=bvh_path,
        frame_time_s=frame_time_s,
        bone_names=bone_names,
        local_pos_cm=local_pos_cm,
        local_quat_wxyz=local_quat_wxyz,
    )


def raw_joints_at(clip: RecordedMocapBvh, frame_idx: int) -> MocapJoints:
    joints: MocapJoints = {}
    for bone_idx, bone_name in enumerate(clip.bone_names):
        joints[bone_name] = {
            "local_pos": clip.local_pos_cm[frame_idx, bone_idx].tolist(),
            "local_rot": clip.local_quat_wxyz[frame_idx, bone_idx].tolist(),
        }
    return add_synthetic_tip_joints(joints)


class OfflineMocapZmqPublisher:
    """ZMQ PUB wrapper matching MocapRtZmqPublisher's wire protocol."""

    def __init__(self, *, bind: str, port: int, retarget_topic: str, send_hwm: int = 1) -> None:
        import zmq

        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.PUB)
        self._sock.setsockopt(zmq.SNDHWM, max(int(send_hwm), 1))
        self._sock.bind(f"tcp://{bind}:{int(port)}")
        self._topic = retarget_topic.encode("utf-8")
        import uuid
        from booster_deploy.utils.diag_report import reporter
        self._stream_id = uuid.uuid4().hex
        self._source_epoch = 0
        self._last_seq = None
        self._diag = reporter('offline_mocap_publisher')
        from booster_deploy.telemetry.publish import Source
        self._source = Source('offline_mocap')
        self._closed = False

    def publish(self, frame: MocapRetargetInputFrameNs) -> None:
        from dataclasses import replace
        if self._last_seq is not None and frame.body_seq < self._last_seq:
            self._source_epoch += 1
        self._last_seq = frame.body_seq
        # This class originates offline events. Decoding legacy network input,
        # in contrast, must never invent identity or a device timestamp.
        loop=frame.loop if frame.loop is not None else self._source_epoch
        identity = frame.source_ids[0] if frame.source_ids else self._source.next(native_seq=frame.body_seq,loop=loop,native_index=frame.native_index)
        frame = replace(frame, source_ids=(identity,), loop=loop,stream_id=frame.stream_id or self._stream_id,
                        source_epoch=self._source_epoch if frame.source_epoch is None else frame.source_epoch,
                        clock_domain=frame.clock_domain or self._diag.clock_domain)
        self._sock.send_multipart([self._topic, encode_mocap_retarget_input(frame)])
        if identity != getattr(self, '_last_published_identity', None):
            self._source.published(identity,channel='retarget_input',human_frame=frame.human_frame,
                                   body_recv_ns=frame.body_recv_ns,source_ns=frame.source_ns,offline_sample_ns=frame.offline_sample_ns)
            self._last_published_identity = identity
        return frame

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._sock.close(linger=0)
        self._ctx.term()
        self._diag.close()


def make_retarget_input_frame(
    clip: RecordedMocapBvh,
    frame_idx: int,
    *,
    body_seq: int,
    human_format: str,
    adapter: PNLinkMocapAdapter,
) -> MocapRetargetInputFrameNs:
    body_recv_ns = time.monotonic_ns()
    raw_joints = raw_joints_at(clip, frame_idx)
    human_frame = adapter.joints_to_human_frame(raw_joints)
    convert_done_ns = time.monotonic_ns()
    return MocapRetargetInputFrameNs(
        body_seq=int(body_seq),native_index=int(frame_idx),
        body_recv_ns=int(body_recv_ns),
        convert_done_ns=int(convert_done_ns),
        human_format=str(human_format),
        human_frame=human_frame,
    )


def _selected_frame_range(frame_count: int, *, start_frame: int, end_frame: int | None) -> tuple[int, int]:
    start = max(int(start_frame), 0)
    end = frame_count if end_frame is None else min(int(end_frame), frame_count)
    if not 0 <= start < end <= frame_count:
        raise ValueError(f"invalid frame range start={start}, end={end}, frame_count={frame_count}")
    return start, end


def _publish_fps(clip: RecordedMocapBvh, *, fps_override: float, speed: float) -> float:
    if not math.isfinite(speed) or speed <= 0.0:
        raise ValueError(f"--speed must be finite and > 0, got {speed}")
    if not math.isfinite(fps_override):
        raise ValueError(f"--fps must be finite, got {fps_override}")
    base_fps = clip.fps if fps_override <= 0.0 else float(fps_override)
    if base_fps <= 0.0:
        raise ValueError(f"publish FPS must be > 0, got {base_fps}")
    return base_fps * float(speed)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    clip = load_recorded_mocap_bvh(args.bvh)
    start, end = _selected_frame_range(
        clip.frame_count,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
    )
    publish_fps = _publish_fps(clip, fps_override=float(args.fps), speed=float(args.speed))
    publish_dt = 1.0 / publish_fps
    start_delay = float(args.start_delay)
    if not math.isfinite(start_delay):
        raise ValueError(f"--start-delay must be finite, got {start_delay}")
    adapter = PNLinkMocapAdapter()
    publisher = OfflineMocapZmqPublisher(
        bind=args.bind,
        port=int(args.port),
        retarget_topic=args.retarget_topic,
        send_hwm=int(args.send_hwm),
    )

    print(
        f"Publishing offline mocap BVH {clip.path} frames=[{start},{end}) "
        f"recorded_fps={clip.fps:.3f} publish_fps={publish_fps:.3f} speed={float(args.speed):.3f} "
        f"output=tcp://{args.bind}:{int(args.port)}/{args.retarget_topic}",
        flush=True,
    )

    if args.wait_for_start_button and args.loop:
        raise ValueError("--wait-for-start-button replays one cycle per A press; pass --no-loop")
    if args.wait_for_start_button:
        from booster_deploy.utils.replay_control import write_input
        if not write_input(os.getenv('BOOSTER_DIAG_DIR'),kind='mocap',path=str(clip.path),start_frame=start,
                end_frame=end,fps=float(args.fps),speed=float(args.speed),human_format=str(args.human_format)):
            raise RuntimeError('offline preparation selection could not be published')
    monitor = None
    if args.wait_for_start_button:
        monitor = RobotButtonMonitor()
        monitor.start()

    from booster_deploy.utils.diag_report import reporter
    from booster_deploy.utils.replay_control import control_path, read_start_ns, read_state, read_abort, write_state, replay_restart_boundary
    diag = reporter('offline')
    diag.note(recording_path=str(clip.path))
    run_dir = os.getenv('BOOSTER_DIAG_DIR')
    replay_control_path = control_path()
    seen_start_ns = replay_restart_boundary(run_dir)
    cycle = 0
    sent = 0
    frame_idx = start
    aborted = False
    if monitor is None:
        from booster_deploy.telemetry.run_config import lock_offline_playback_for_autostart
        if not lock_offline_playback_for_autostart(os.getenv("BOOSTER_TELEMETRY_DIR") or run_dir or ""):
            raise RuntimeError("offline playback lock could not be committed before automatic replay")
    try:
        while True:
            cycle_token = 0
            preparation = None
            if monitor is not None:
                print("[replay] idle; press robot controller A to replay one cycle", flush=True)
                started = False
                while not started:
                    diag.emit('idle · Waiting for A to start', alerts=[], extra={'state': dict(
                        cycle=cycle, frame_index=0, frames_left=end-start, publish_total=end-start, stopped=aborted,
                        waiting_for_start=True, paused=False)})
                    token = read_start_ns(run_dir)
                    from booster_deploy.utils.replay_control import read_prepare_ns,read_signal,PreparationCycle,play_authorized
                    prepare = read_prepare_ns(run_dir)
                    if (prepare > seen_start_ns and not read_abort(run_dir, prepare)
                            and read_signal(run_dir, 'replay.ready') == prepare):
                        seen_start_ns = cycle_token = prepare
                        preparation = PreparationCycle(run_dir,prepare,monitor.is_stopped)
                        started = True
                        break
                    if prepare and not play_authorized(run_dir,token):
                        token = 0
                    if token and token > seen_start_ns:
                        from booster_deploy.telemetry.run_config import commit_offline_playback_start
                        if not commit_offline_playback_start(os.getenv("BOOSTER_TELEMETRY_DIR") or run_dir or ""):
                            print("[replay] start rejected: no matching telemetry playback reservation", flush=True)
                            continue
                        seen_start_ns = cycle_token = token
                        started = True
                        break
                    started = monitor.wait_for_start(timeout_s=0.05) and not prepare and not os.getenv("BOOSTER_OFFLINE_REPLAY")
                    if monitor.is_stopped():
                        break
                if not started:
                    break
                from booster_deploy.telemetry.run_config import commit_offline_playback_start
                if preparation is None and not commit_offline_playback_start(os.getenv("BOOSTER_TELEMETRY_DIR") or run_dir or ""):
                    print("[replay] start rejected: no matching telemetry playback reservation", flush=True)
                    continue
                print("[replay] A pressed; replaying one cycle", flush=True)
            cycle += 1
            write_state(False)
            publisher._source_epoch += 1
            frame_idx = start
            next_time_s = time.perf_counter() + max(start_delay, 0.0)
            sample_origin_ns = time.monotonic_ns()
            aborted = False
            while True:
                token = read_start_ns(run_dir)
                if token > seen_start_ns:
                    seen_start_ns = cycle_token = token
                if read_abort(run_dir, cycle_token) or (monitor is not None and monitor.abort_requested()):
                    aborted = True
                    print("[replay] B pressed; cycle aborted", flush=True)
                    break
                if read_state(replay_control_path):
                    next_time_s = time.perf_counter()
                    diag.emit('paused · Replay is frozen (press Y to resume)', alerts=[], extra={'state': dict(
                        cycle=cycle, frame_index=frame_idx-start, frames_left=end-frame_idx,
                        publish_total=end-start, waiting_for_start=False, paused=True, stopped=False)})
                    time.sleep(0.02)
                    continue
                if preparation is not None and frame_idx > start:
                    try:
                        next_time_s += preparation.before_next()
                    except (InterruptedError,TimeoutError) as exc:
                        from booster_deploy.utils.replay_control import write_abort
                        write_abort(run_dir,token=preparation.cycle)
                        diag.event('offline_prepare_cancelled',dict(cycle=preparation.cycle,reason=str(exc)))
                        aborted=True
                        break
                now_s = time.perf_counter()
                if now_s < next_time_s:
                    time.sleep(min(next_time_s - now_s, 0.001))
                    continue
                next_time_s += publish_dt

                from dataclasses import replace
                frame = make_retarget_input_frame(clip,frame_idx,body_seq=sent+1,
                    human_format=str(args.human_format),adapter=adapter)
                if preparation:
                    frame = replace(frame,offline_sample_ns=sample_origin_ns+int((frame_idx-start)*publish_dt*1e9),offline_cycle=preparation.cycle)
                frame = publisher.publish(frame)
                if preparation is not None and frame_idx == start:
                    try:
                        next_time_s += preparation.first(lambda:publisher.publish(frame))
                    except (InterruptedError,TimeoutError) as exc:
                        from booster_deploy.utils.replay_control import write_abort
                        write_abort(run_dir,token=preparation.cycle)
                        diag.event('offline_prepare_cancelled',dict(cycle=preparation.cycle,reason=str(exc)))
                        aborted=True
                        break
                sent += 1
                diag.mark('sent')
                diag.emit('playing', alerts=[], extra={'state': dict(
                    cycle=cycle, frame_index=frame_idx-start+1, frames_left=end-frame_idx-1,
                    publish_total=end-start, waiting_for_start=False, paused=False, stopped=False)})

                if args.print_interval > 0 and sent % int(args.print_interval) == 0:
                    print(f"OFFLINE_MOCAP_ZMQ sent={sent} frame_idx={frame_idx}", flush=True)
                if args.max_frames > 0 and sent >= int(args.max_frames):
                    aborted = True
                    break

                frame_idx += 1
                if frame_idx >= end:
                    if not args.loop:
                        break
                    frame_idx = start
                    publisher._source_epoch += 1
                if next_time_s < time.perf_counter() - publish_dt:
                    next_time_s = time.perf_counter()
            if monitor is None:
                break
            diag.emit('idle · Waiting for A to start', alerts=[], force=True, extra={'state': dict(
                cycle=cycle, frame_index=0, frames_left=end-start, publish_total=end-start, stopped=aborted,
                waiting_for_start=True, paused=False)})
            if not aborted:
                print("[replay] cycle finished; waiting for the next A press", flush=True)
    except KeyboardInterrupt:
        print("Stopping offline mocap publisher...", flush=True)
    finally:
        if monitor is not None:
            monitor.stop()
        publisher.close()
        diag.close()
