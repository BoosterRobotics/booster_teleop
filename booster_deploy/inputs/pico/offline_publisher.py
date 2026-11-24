"""Publish recorded raw Pico body poses as realtime ``pico_rt`` input.

Input NPZ files are expected to come from
``scripts/diagnostics/record_pico_raw.py`` or ``record_pico_stream.py``. Each
recorded SDK pose is converted with the same coordinate adapter used by live
Pico capture, then encoded and published with the same topic and payload as
``scripts/teleop/pico_rt_zmq_publisher.py``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
import time
import uuid

import numpy as np

from .codecs import encode_pico_retarget_input
from .coordinate import BODY_JOINT_NAMES
from .offline_recording import PicoRawRecording, load_pico_raw_recording
from .raw_stream import PICO_RAW_TOPIC, encode_pico_raw
from booster_deploy.inputs.control.start_gate import RobotButtonMonitor
from booster_deploy.types.pico_types import PicoRetargetInputFrameNs
from booster_deploy.utils.diag_report import below_floor, format_gap, reporter


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Publish raw Pico NPZ recordings as pico_rt.retarget_input ZMQ frames."
    )
    parser.add_argument("recording", type=Path, help="Raw Pico NPZ recording.")
    parser.add_argument("--bind", default="0.0.0.0", help="ZMQ bind address, matching pico_rt_zmq_publisher.py.")
    parser.add_argument("--port", type=int, default=5555, help="ZMQ PUB port, matching the Pico RT default.")
    parser.add_argument(
        "--retarget-topic",
        default="pico_rt.retarget_input",
        help="Pico RT retarget-input topic.",
    )
    parser.add_argument("--send-hwm", type=int, default=1, help="ZMQ send high-water mark.")
    parser.add_argument(
        "--raw-bind",
        default=None,
        help="Also publish the pre-retarget PICO body stream on this endpoint (e.g. tcp://0.0.0.0:5581).",
    )
    parser.add_argument("--human-format", default="xrobot", help="Published PicoRetargetInputFrameNs.human_format.")
    parser.add_argument(
        "--fps",
        type=float,
        default=50.0,
        help="Resample the recorded timeline to this publish rate. Use <=0 to publish every source frame at its recorded time.",
    )
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier applied to recorded time.")
    parser.add_argument("--start-frame", type=int, default=0, help="First NPZ frame index to publish.")
    parser.add_argument("--end-frame", type=int, default=None, help="Exclusive NPZ frame index to stop at.")
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
        help="Loop the selected frame range to emulate a continuous Pico source.",
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


class OfflinePicoZmqPublisher:
    """ZMQ PUB wrapper matching ``PicoRtZmqPublisher``'s body protocol.

    ``raw_bind`` additionally publishes the pre-retarget body stream
    (``pico_rt.raw``) for diagnostics. The publisher can feed the same
    retargeting worker on a real robot or in simulation.
    """

    def __init__(self, *, bind: str, port: int, retarget_topic: str, send_hwm: int = 1,
                 raw_bind: str | None = None) -> None:
        import zmq

        self._zmq = zmq
        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.PUB)
        self._sock.setsockopt(zmq.SNDHWM, max(int(send_hwm), 1))
        self._sock.bind(f"tcp://{bind}:{int(port)}")
        self._topic = retarget_topic.encode("utf-8")
        self._raw_sock = None
        self._raw_stream_id = uuid.uuid4().hex
        self._diag = reporter("offline_publisher")
        from booster_deploy.telemetry.publish import Source
        self._source = Source("offline_pico")
        self._last_identity = None
        self._raw_seq = 0
        self._raw_dropped = 0
        self._raw_metadata_ns = 0
        if raw_bind:
            self._raw_sock = self._ctx.socket(zmq.XPUB)
            self._raw_sock.setsockopt(zmq.SNDHWM, 2048)
            self._raw_sock.setsockopt(zmq.XPUB_NODROP, 1)
            self._raw_sock.setsockopt(zmq.LINGER, 0)
            self._raw_sock.bind(raw_bind)
        self._closed = False

    @property
    def raw_enabled(self) -> bool:
        return self._raw_sock is not None

    def _send_raw(self, kind: str, **values: object) -> None:
        if self._raw_sock is None:
            return
        self._raw_seq += 1
        payload = encode_pico_raw({
            "kind": kind, "stream_id": self._raw_stream_id, "packet_seq": self._raw_seq,
            "source_drops": self._raw_dropped, "publish_ns": time.monotonic_ns(),
            "clock_domain": self._diag.clock_domain, **values,
        })
        try:
            self._raw_sock.send_multipart([PICO_RAW_TOPIC, payload], flags=self._zmq.NOBLOCK)
            if kind == 'body' and self._last_identity != getattr(self, '_last_raw_published_identity', None):
                self._source.published(self._last_identity, channel='raw_body', **values)
                self._last_raw_published_identity = self._last_identity
        except self._zmq.Again:
            self._raw_dropped += 1

    def publish(self, frame: PicoRetargetInputFrameNs) -> None:
        from dataclasses import replace
        identity = frame.source_ids[0] if frame.source_ids else self._source.next(native_seq=frame.body_seq, loop=frame.loop, native_index=frame.native_index)
        self._last_identity = identity
        frame = replace(frame, stream_id=self._source.stream, source_ids=(identity,), clock_domain=self._diag.clock_domain)
        self._sock.send_multipart([self._topic, encode_pico_retarget_input(frame)])
        if identity != getattr(self, '_last_published_identity', None):
            self._source.published(identity, channel='retarget_input', human_frame=frame.human_frame,
                                   body_recv_ns=frame.body_recv_ns, convert_done_ns=frame.convert_done_ns,offline_sample_ns=frame.offline_sample_ns)
            self._last_published_identity = identity
        return frame

    def publish_raw_metadata(self, recording: PicoRawRecording, *, human_format: str,
                             fps: float, speed: float) -> None:
        if self._raw_sock is None:
            return
        self._raw_metadata_ns = time.monotonic_ns()
        self._send_raw(
            "metadata",
            joint_names=list(BODY_JOINT_NAMES),
            pose_columns=["x", "y", "z", "qx", "qy", "qz", "qw"],
            position_unit="m",
            coordinates="XRoboToolkit SDK (before INPUT_TO_DEPLOY transform)",
            human_format=str(human_format),
            source="offline_replay",
            recording_path=str(recording.path),
            recording_sha256=_file_sha256(recording.path),
            playback_fps=float(fps),
            playback_speed=float(speed),
        )

    def publish_raw_body(self, *, seq: int, poses: np.ndarray, source_ns: int | None, offline_sample_ns=None, offline_cycle=0) -> None:
        if self._raw_sock is None:
            return
        self._send_raw(
            "body",
            seq=int(seq),
            source_ids=(self._last_identity,),
            body_recv_ns=time.monotonic_ns(),
            source_ns=None if source_ns is None else int(source_ns),
            offline_sample_ns=offline_sample_ns,offline_cycle=offline_cycle,
            poses=np.asarray(poses, dtype=float).tolist(),
        )

    def raw_metadata_due(self, interval_s: float = 0.5) -> bool:
        return time.monotonic_ns() - self._raw_metadata_ns >= int(interval_s * 1e9)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._sock.close(linger=0)
        if self._raw_sock is not None:
            self._raw_sock.close(linger=0)
        self._ctx.term()
        self._diag.close()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class PlaybackTimeline:
    frame_indices: np.ndarray
    offsets_s: np.ndarray
    source_duration_s: float
    cycle_period_s: float


def make_retarget_input_frame(
    recording: PicoRawRecording,
    frame_idx: int,
    *,
    body_seq: int,
    human_format: str,
    native_index: int | None = None, loop: int | None = None,
) -> PicoRetargetInputFrameNs:
    body_recv_ns = time.monotonic_ns()
    human_frame = recording.frame_at(frame_idx)
    convert_done_ns = time.monotonic_ns()
    return PicoRetargetInputFrameNs(
        body_seq=int(body_seq),
        native_index=frame_idx if native_index is None else native_index, loop=loop,
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


def _build_playback_timeline(
    recording: PicoRawRecording,
    *,
    start: int,
    end: int,
    target_fps: float,
    speed: float,
) -> PlaybackTimeline:
    speed = float(speed)
    target_fps = float(target_fps)
    if not math.isfinite(speed) or speed <= 0.0:
        raise ValueError(f"--speed must be finite and > 0, got {speed}")
    if not math.isfinite(target_fps):
        raise ValueError(f"--fps must be finite, got {target_fps}")

    source_offsets_s = np.asarray(recording.time_s[start:end], dtype=np.float64)
    source_offsets_s = source_offsets_s - source_offsets_s[0]
    source_duration_s = float(source_offsets_s[-1])
    if target_fps > 0.0 and source_duration_s > 0.0:
        sample_count = int(math.floor(source_duration_s * target_fps)) + 1
        sample_offsets_s = np.arange(sample_count, dtype=np.float64) / target_fps
        if source_duration_s - float(sample_offsets_s[-1]) > 1.0e-9:
            sample_offsets_s = np.append(sample_offsets_s, source_duration_s)
        relative_indices = np.searchsorted(source_offsets_s, sample_offsets_s, side="right") - 1
        relative_indices = np.clip(relative_indices, 0, len(source_offsets_s) - 1)
        tail_s = 1.0 / target_fps
    else:
        sample_offsets_s = source_offsets_s
        relative_indices = np.arange(len(source_offsets_s), dtype=np.int64)
        if len(source_offsets_s) > 1 and source_duration_s > 0.0:
            tail_s = source_duration_s / (len(source_offsets_s) - 1)
        else:
            tail_s = 1.0 / recording.fps

    playback_offsets_s = sample_offsets_s / speed
    return PlaybackTimeline(
        frame_indices=np.asarray(relative_indices + start, dtype=np.int64),
        offsets_s=np.asarray(playback_offsets_s, dtype=np.float64),
        source_duration_s=source_duration_s,
        cycle_period_s=float(playback_offsets_s[-1] + tail_s / speed),
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    recording = load_pico_raw_recording(args.recording)
    frame_count = int(recording.poses.shape[0])
    start, end = _selected_frame_range(
        frame_count,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
    )
    timeline = _build_playback_timeline(
        recording,
        start=start,
        end=end,
        target_fps=float(args.fps),
        speed=float(args.speed),
    )
    start_delay = float(args.start_delay)
    if not math.isfinite(start_delay):
        raise ValueError(f"--start-delay must be finite, got {start_delay}")
    publisher = OfflinePicoZmqPublisher(
        bind=args.bind,
        port=int(args.port),
        retarget_topic=args.retarget_topic,
        send_hwm=int(args.send_hwm),
        raw_bind=args.raw_bind,
    )

    print(
        f"Publishing offline Pico raw recording {recording.path} frames=[{start},{end}) "
        f"source_frames={end - start} publish_frames={len(timeline.frame_indices)} "
        f"source_duration={timeline.source_duration_s:.3f}s playback_duration="
        f"{timeline.source_duration_s / float(args.speed):.3f}s timing={recording.time_source} "
        f"target_fps={float(args.fps):.3f} speed={float(args.speed):.3f} "
        f"output=tcp://{args.bind}:{int(args.port)}/{args.retarget_topic}"
        + (f" raw={args.raw_bind}/{PICO_RAW_TOPIC.decode()}" if publisher.raw_enabled else ""),
        flush=True,
    )

    if args.wait_for_start_button and args.loop:
        raise ValueError("--wait-for-start-button replays one cycle per A press; pass --no-loop")
    if args.wait_for_start_button:
        from booster_deploy.utils.replay_control import write_input
        if not write_input(os.getenv('BOOSTER_DIAG_DIR'), kind='pico', path=str(recording.path),
                start_frame=start, end_frame=end, fps=float(args.fps), speed=float(args.speed),
                human_format=str(args.human_format)):
            raise RuntimeError('could not publish offline preparation selection')
    monitor = None
    if args.wait_for_start_button:
        monitor = RobotButtonMonitor()
        monitor.start()

    diag = reporter("offline")
    diag.note(recording_path=str(recording.path))
    cycle = 0
    sent = 0
    publish_total = int(len(timeline.frame_indices))
    # Set by the deploy (Y pause) so the replay itself freezes, not just the policy.
    from booster_deploy.utils.replay_control import control_path, read_start_ns, read_state, read_abort, replay_restart_boundary

    replay_control_path = control_path()
    seen_start_ns = replay_restart_boundary(os.getenv('BOOSTER_DIAG_DIR'))
    timeline_idx = 0
    aborted = False
    warned_failed = False
    if monitor is None:
        from booster_deploy.telemetry.run_config import lock_offline_playback_for_autostart
        if not lock_offline_playback_for_autostart(os.getenv("BOOSTER_TELEMETRY_DIR") or os.getenv("BOOSTER_DIAG_DIR", "")):
            raise RuntimeError("offline playback lock could not be committed before automatic replay")
    try:
        while True:
            cycle_token = 0
            preparation = None
            if monitor is not None:
                print("[replay] idle; press robot controller A to replay one cycle", flush=True)
                # Tell the dashboard right away, then keep saying it: without this
                # the progress bar kept the finished cycle's last value until the
                # next A press (diag only emitted inside the publish loop).
                publish_total = int(len(timeline.frame_indices))
                started = False
                while not started:
                    # The deploy drops replay.start when it accepts A. That token is
                    # the primary trigger: this pane's own ROS subscriber can fail
                    # silently (rclpy missing) and then the replay waited forever
                    # while the deploy sat in arming_tracking -- robot never moved.
                    token = read_start_ns(os.getenv("BOOSTER_DIAG_DIR"))
                    from booster_deploy.utils.replay_control import read_prepare_ns, play_authorized
                    prepare = read_prepare_ns(os.getenv("BOOSTER_DIAG_DIR"))
                    if prepare and not play_authorized(os.getenv("BOOSTER_DIAG_DIR"), token):
                        token = 0
                    from booster_deploy.utils.replay_control import read_signal, PreparationCycle
                    if (prepare > seen_start_ns and not read_abort(os.getenv('BOOSTER_DIAG_DIR'), prepare)
                            and read_signal(os.getenv('BOOSTER_DIAG_DIR'), 'replay.ready') == prepare):
                        seen_start_ns = prepare
                        cycle_token = prepare
                        preparation = PreparationCycle(os.getenv('BOOSTER_DIAG_DIR'), prepare, monitor.is_stopped)
                        started = True
                        break
                    if token and token > seen_start_ns:
                        from booster_deploy.telemetry.run_config import commit_offline_playback_start
                        if not commit_offline_playback_start(os.getenv("BOOSTER_TELEMETRY_DIR") or os.getenv("BOOSTER_DIAG_DIR", "")):
                            print("[replay] start rejected: no matching telemetry playback reservation", flush=True)
                            continue
                        seen_start_ns = cycle_token = token
                        started = True
                        print("[replay] start token from deploy; replaying one cycle", flush=True)
                        break
                    button_start = monitor.wait_for_start(timeout_s=0.05)
                    started = button_start and not prepare and not os.getenv("BOOSTER_OFFLINE_REPLAY")
                    if started or monitor.is_stopped():
                        break
                    if monitor.failed():
                        if not warned_failed:
                            warned_failed = True
                            print(
                                f"[replay] button monitor failed ({monitor.error}); "
                                "waiting for the deploy start token instead",
                                flush=True,
                            )
                        continue
                    diag.emit(
                        "idle · Waiting for A to start",
                        alerts=[],
                        extra={"state": {
                            "cycle": cycle,
                            "frame_index": 0,
                            "frames_left": publish_total,
                            "stopped": aborted,
                            "publish_total": publish_total,
                            "waiting_for_start": True,
                            "paused": False,
                        }},
                    )
                if not started:
                    break
                from booster_deploy.telemetry.run_config import commit_offline_playback_start
                if preparation is None and not commit_offline_playback_start(os.getenv("BOOSTER_TELEMETRY_DIR") or os.getenv("BOOSTER_DIAG_DIR", "")):
                    print("[replay] start rejected: no matching telemetry playback reservation", flush=True)
                    continue
                print("[replay] cycle started; replaying", flush=True)
            cycle += 1
            cycle_start_s = time.perf_counter() + max(start_delay, 0.0)
            sample_origin_ns = time.monotonic_ns()
            timeline_idx = 0
            aborted = False
            next_pause_check_s = 0.0
            paused = False
            # A new cycle always starts running: a leftover "paused" from the previous
            # run froze the replay right after A, so the deploy waited 30 s for a
            # reference window and the whole state machine looked unresponsive.
            if replay_control_path is not None and read_state(replay_control_path):
                from booster_deploy.utils.replay_control import write_state

                write_state(False)
                print("[replay] cleared stale pause at cycle start", flush=True)
            while True:
                # Accepted B is common to keyboard/web/native/PICO and wins over Y.
                token = read_start_ns(os.getenv("BOOSTER_DIAG_DIR"))
                if token > seen_start_ns:
                    seen_start_ns = cycle_token = token
                if read_abort(os.getenv("BOOSTER_DIAG_DIR"), cycle_token) or (monitor is not None and monitor.abort_requested()):
                    aborted = True
                    print("[replay] accepted B; cycle aborted", flush=True)
                    break
                # Y pause freezes the replay itself, not just the policy.
                now_check = time.perf_counter()
                if now_check >= next_pause_check_s:
                    next_pause_check_s = now_check + 0.1
                    paused = read_state(replay_control_path)
                if paused:
                    # Re-anchor the schedule so resuming does not burst-drain the
                    # frames that "should" have gone out while paused.
                    cycle_start_s = max(
                        cycle_start_s,
                        time.perf_counter() - float(timeline.offsets_s[timeline_idx]),
                    )
                    if diag.due():
                        diag.emit(
                            "paused · Replay is frozen (press Y to resume)",
                            alerts=[],
                            extra={"state": {
                                "cycle": cycle,
                                "frame_index": timeline_idx,
                                "frames_left": int(len(timeline.frame_indices) - timeline_idx),
                                "publish_total": publish_total,
                                "paused": True,
                                "waiting_for_start": False,
                                "stopped": False,
                            }},
                        )
                    time.sleep(0.02)
                    continue

                if timeline_idx >= len(timeline.frame_indices):
                    if not args.loop:
                        break
                    timeline_idx = 0
                    cycle += 1
                    cycle_start_s += timeline.cycle_period_s

                if preparation is not None and timeline_idx:
                    try:
                        cycle_start_s += preparation.before_next()
                    except (InterruptedError,TimeoutError) as exc:
                        from booster_deploy.utils.replay_control import write_abort
                        write_abort(os.getenv('BOOSTER_DIAG_DIR'),token=preparation.cycle)
                        diag.event('offline_prepare_cancelled',dict(cycle=preparation.cycle,reason=str(exc)))
                        aborted = True
                        break
                next_time_s = cycle_start_s + float(timeline.offsets_s[timeline_idx])
                now_s = time.perf_counter()
                if now_s < next_time_s:
                    time.sleep(min(next_time_s - now_s, 0.001))
                    continue

                frame_idx = int(timeline.frame_indices[timeline_idx])

                from dataclasses import replace
                frame = make_retarget_input_frame(recording, frame_idx, body_seq=sent+1,
                    loop=cycle, human_format=str(args.human_format))
                if preparation:
                    frame = replace(frame, offline_sample_ns=sample_origin_ns+int(float(timeline.offsets_s[timeline_idx])*1e9),offline_cycle=preparation.cycle)
                frame = publisher.publish(frame)
                if publisher.raw_enabled:
                    if publisher.raw_metadata_due():
                        publisher.publish_raw_metadata(
                            recording,
                            human_format=str(args.human_format),
                            fps=float(args.fps),
                            speed=float(args.speed),
                        )
                    publisher.publish_raw_body(
                        seq=sent + 1,
                        poses=recording.poses[frame_idx],
                        source_ns=int(recording.time_s[frame_idx] * 1e9),
                        offline_sample_ns=frame.offline_sample_ns,offline_cycle=frame.offline_cycle,
                    )
                if preparation is not None and timeline_idx == 0:
                    def resend_first():
                        publisher.publish(frame)
                        if publisher.raw_enabled:
                            publisher.publish_raw_body(seq=sent+1,poses=recording.poses[frame_idx],
                                source_ns=int(recording.time_s[frame_idx]*1e9),offline_sample_ns=frame.offline_sample_ns,offline_cycle=frame.offline_cycle)
                    try:
                        cycle_start_s += preparation.first(resend_first)
                    except (InterruptedError,TimeoutError) as exc:
                        from booster_deploy.utils.replay_control import write_abort
                        write_abort(os.getenv('BOOSTER_DIAG_DIR'),token=preparation.cycle)
                        diag.event('offline_prepare_cancelled',dict(cycle=preparation.cycle,reason=str(exc)))
                        aborted = True
                        break
                sent += 1

                # Schedule slip: how late this frame actually went out.
                diag.mark("sent")
                diag.sample("slip", max((time.perf_counter() - next_time_s) * 1e3, 0.0))
                if diag.due() and (preparation is None or preparation.playing()):
                    now_ns = time.monotonic_ns()
                    window = diag.interval("sent")
                    hz = window.hz(now_ns)
                    gap = window.gap_stats(now_ns)
                    slip = diag.latency("slip", now_ns)
                    alerts: list[str] = []
                    if below_floor(hz):
                        alerts.append(f"pub {hz:.1f} Hz < 50")
                    if slip.p95 > 10.0:
                        alerts.append(f"slip_p95 {slip.p95:.1f} ms")
                    diag.emit(
                        f"pub {hz:5.1f} Hz   gap {format_gap(gap)} ms",
                        alerts=alerts,
                        extra={"state": {"cycle": cycle, "frame_index": timeline_idx + 1,
                                         "frames_left": int(len(timeline.frame_indices) - timeline_idx - 1),
                                         "publish_total": int(len(timeline.frame_indices)),
                                         "waiting_for_start": False, "paused": False, "stopped": False,
                                         "slip_p95_ms": round(slip.p95, 2)}},
                        now_ns=now_ns,
                    )

                if args.print_interval > 0 and sent % int(args.print_interval) == 0:
                    print(f"OFFLINE_PICO_ZMQ sent={sent} frame_idx={frame_idx}", flush=True)
                if args.max_frames > 0 and sent >= int(args.max_frames):
                    aborted = True
                    break
                timeline_idx += 1
            if monitor is None:
                break
            diag.emit('idle · Waiting for A to start', alerts=[], force=True, extra={'state': dict(
                cycle=cycle, frame_index=0, frames_left=publish_total, publish_total=publish_total, stopped=aborted,
                waiting_for_start=True, paused=False)})
            if not aborted:
                print("[replay] cycle finished; waiting for the next A press", flush=True)
    except KeyboardInterrupt:
        print("Stopping offline Pico publisher...", flush=True)
    finally:
        diag.close()
        if monitor is not None:
            monitor.stop()
        publisher.close()
