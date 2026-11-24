"""Record Pico/XRoboToolkit body poses before HumanFrame conversion.

The existing ``pico_rt_zmq_publisher.py`` publishes a converted HumanFrame.
This tool records the SDK body-pose array itself so that GMR and future UMR
adapters can be replayed from exactly the same source capture.

Example::

    python scripts/diagnostics/record_pico_raw.py \
        --output logs/pico_raw/pico_raw_01.npz

Hold both triggers and press X on the controllers to start; hold both triggers
and press Y to save and exit. The trigger modifier keeps the chord away from the
bare A/B buttons the robot state machine listens to.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np

from .capture import PicoRtCapture, PicoRtCaptureConfig
from .coordinate import BODY_JOINT_NAMES
from .recording import RecordingChordDetector


FORMAT_VERSION = 2
RECORD_HZ = 50.0
RAW_POSE_COLUMNS = ("px", "py", "pz", "qx", "qy", "qz", "qw")
START_CHORD_HINT = "Hold both triggers and press X to start recording; hold both and press Y to save and exit."


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="Output compressed NPZ path.")
    parser.add_argument("--frames", type=int, default=0, help="Optional frame limit; 0 means stop only with Pico B or Ctrl-C.")
    parser.add_argument("--timeout", type=float, default=2.0, help="Seconds to wait for each new body frame.")
    parser.add_argument("--input-mode", choices=("auto", "callback", "polling"), default="auto")
    parser.add_argument("--poll-hz", type=float, default=50.0, help="Polling rate when callback is unavailable.")
    parser.add_argument("--print-interval", type=int, default=50, help="Print every N recorded frames; <=0 disables.")
    return parser.parse_args(argv)


def advance_sample_deadline(frame_ns: int, next_sample_ns: int, sample_period_ns: int) -> tuple[bool, int]:
    if int(frame_ns) < int(next_sample_ns):
        return False, int(next_sample_ns)
    elapsed_periods = (int(frame_ns) - int(next_sample_ns)) // int(sample_period_ns)
    return True, int(next_sample_ns) + (elapsed_periods + 1) * int(sample_period_ns)


def _wait_for_start_chord(capture: PicoRtCapture, timeout_s: float, detector: RecordingChordDetector):
    last_seq: int | None = None
    while True:
        try:
            control = capture.wait_next_pico_controller(last_seq, max(float(timeout_s), 0.001))
        except TimeoutError as exc:
            print(f"{exc}; {START_CHORD_HINT}", flush=True)
            continue
        last_seq = int(control.seq)
        if "c" in detector.update(control):
            return control, last_seq


def _estimate_fps(timestamps_ns: np.ndarray, fallback_hz: float) -> float:
    if timestamps_ns.size < 2:
        return float(fallback_hz)
    duration_ns = int(timestamps_ns[-1]) - int(timestamps_ns[0])
    if duration_ns <= 0:
        return float(fallback_hz)
    return float((timestamps_ns.size - 1) * 1.0e9 / duration_ns)


def _recording_time_ns(source_ns: list[int], body_recv_ns: list[int]) -> tuple[np.ndarray, str]:
    source = np.asarray(source_ns, dtype=np.int64)
    if source.size > 0 and np.all(source > 0) and np.all(np.diff(source) > 0):
        return source, "source_ns"
    return np.asarray(body_recv_ns, dtype=np.int64), "body_recv_ns"


def save_raw_recording(
    output: Path,
    *,
    poses: list[np.ndarray],
    body_seq: list[int],
    source_ns: list[int],
    body_recv_ns: list[int],
    fallback_hz: float,
) -> None:
    if not poses:
        raise ValueError("No Pico body frames were recorded")
    pose_array = np.asarray(poses, dtype=np.float64)
    if pose_array.ndim != 3 or pose_array.shape[1] != len(BODY_JOINT_NAMES) or pose_array.shape[2] < 7:
        raise ValueError(
            "raw poses must have shape [frames, 24, columns>=7], "
            f"got {pose_array.shape}"
        )
    if not (len(body_seq) == len(source_ns) == len(body_recv_ns) == pose_array.shape[0]):
        raise ValueError("metadata arrays and poses have different frame counts")

    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    column_names = list(RAW_POSE_COLUMNS)
    if pose_array.shape[2] > len(column_names):
        column_names.extend(f"column_{idx:02d}" for idx in range(len(column_names), pose_array.shape[2]))
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp.npz")
    time_ns, fps_time_source = _recording_time_ns(source_ns, body_recv_ns)
    np.savez_compressed(
        temporary,
        format_version=np.asarray(FORMAT_VERSION, dtype=np.int32),
        input_source=np.asarray("pico_sdk_raw"),
        human_format=np.asarray("xrobot"),
        fps=np.asarray([_estimate_fps(time_ns, fallback_hz)], dtype=np.float64),
        fps_time_source=np.asarray(fps_time_source),
        sampling_hz=np.asarray([fallback_hz], dtype=np.float64),
        raw_quaternion_convention=np.asarray("xyzw"),
        raw_pose_columns=np.asarray(column_names, dtype=str),
        joint_names=np.asarray(BODY_JOINT_NAMES, dtype=str),
        poses=pose_array,
        body_seq=np.asarray(body_seq, dtype=np.int64),
        source_ns=np.asarray(source_ns, dtype=np.int64),
        body_recv_ns=np.asarray(body_recv_ns, dtype=np.int64),
    )
    os.replace(temporary, output)


def record_raw_capture(
    capture: PicoRtCapture,
    output: Path,
    *,
    frame_limit: int = 0,
    timeout_s: float = 2.0,
    print_interval: int = 50,
) -> int:
    """Record an already started capture, save on stop, and return the frame count.

    The caller owns the capture lifecycle. Only the CLI ``main`` terminates the
    process to work around the native SDK callback teardown issue.
    """

    if frame_limit < 0:
        raise ValueError("frame_limit must be >= 0")
    poses: list[np.ndarray] = []
    body_seq: list[int] = []
    source_ns: list[int] = []
    body_recv_ns: list[int] = []
    last_seq: int | None = None
    last_control_seq: int | None = None
    chord = RecordingChordDetector()
    print(
        f"Pico raw recorder ready | mode={capture.input_mode} output={output.expanduser().resolve()}",
        flush=True,
    )
    try:
        print(START_CHORD_HINT, flush=True)
        start_control, last_control_seq = _wait_for_start_chord(capture, timeout_s, chord)
        recording_start_ns = int(start_control.recv_ns)
        sample_period_ns = int(round(1.0e9 / RECORD_HZ))
        next_sample_ns: int | None = None
        sample_clock_source: str | None = None
        last_sample_clock_ns: int | None = None
        started = time.monotonic()
        next_body_timeout = started + max(float(timeout_s), 0.001)
        print(f"Pico triggers + X pressed; recording started at {RECORD_HZ:.1f} Hz.", flush=True)

        while True:
            if frame_limit > 0 and len(poses) >= frame_limit:
                print(f"Reached frame limit ({frame_limit}); stopping recording.", flush=True)
                break

            control = capture.latest_pico_controller()
            if control is not None and int(control.seq) != last_control_seq:
                last_control_seq = int(control.seq)
                if "x" in chord.update(control):
                    print("Chord (both triggers + Y) pressed; saving and exiting.", flush=True)
                    break

            raw = capture.latest_body()
            if raw is None or int(raw.seq) == last_seq:
                now = time.monotonic()
                if now >= next_body_timeout:
                    print(f"No new Pico raw body frame received within {timeout_s:.3f}s", flush=True)
                    next_body_timeout = now + max(float(timeout_s), 0.001)
                time.sleep(0.001)
                continue
            last_seq = int(raw.seq)
            if int(raw.body_recv_ns) < recording_start_ns:
                continue
            if raw.source_ns is not None and int(raw.source_ns) > 0:
                sample_clock_ns = int(raw.source_ns)
                current_clock_source = "source_ns"
            else:
                sample_clock_ns = int(raw.body_recv_ns)
                current_clock_source = "body_recv_ns"
            if (
                next_sample_ns is None
                or sample_clock_source != current_clock_source
                or (last_sample_clock_ns is not None and sample_clock_ns <= last_sample_clock_ns)
            ):
                next_sample_ns = sample_clock_ns
                sample_clock_source = current_clock_source
            should_record, next_sample_ns = advance_sample_deadline(sample_clock_ns, next_sample_ns, sample_period_ns)
            last_sample_clock_ns = sample_clock_ns
            if not should_record:
                continue

            pose = np.asarray(raw.poses, dtype=np.float64).copy()
            if poses and pose.shape != poses[0].shape:
                raise ValueError(f"Pico raw pose shape changed from {poses[0].shape} to {pose.shape}")
            poses.append(pose)
            body_seq.append(int(raw.seq))
            source_ns.append(-1 if raw.source_ns is None else int(raw.source_ns))
            body_recv_ns.append(int(raw.body_recv_ns))
            next_body_timeout = time.monotonic() + max(float(timeout_s), 0.001)
            if print_interval > 0 and len(poses) % print_interval == 0:
                elapsed = max(time.monotonic() - started, 1.0e-9)
                print(f"frames={len(poses)} rate={len(poses) / elapsed:.1f} seq={last_seq}", flush=True)
    except KeyboardInterrupt:
        print("\nStopping raw Pico recording...", flush=True)
    finally:
        # Save before touching the native SDK so B, frame-limit, and Ctrl-C
        # stops leave a usable artifact even if SDK teardown blocks.
        if poses:
            save_raw_recording(
                output,
                poses=poses,
                body_seq=body_seq,
                source_ns=source_ns,
                body_recv_ns=body_recv_ns,
                fallback_hz=RECORD_HZ,
            )
            print(f"Saved {len(poses)} raw frames to {output.expanduser().resolve()}", flush=True)
        else:
            print("No raw Pico frames were captured; no NPZ was written.", flush=True)

    return len(poses)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.frames < 0:
        raise ValueError("--frames must be >= 0")

    capture = PicoRtCapture(
        PicoRtCaptureConfig(
            input_mode=str(args.input_mode),
            poll_hz=float(args.poll_hz),
            max_cached_frames=4,
        )
    )
    exit_code = 1
    try:
        capture.start()
        count = record_raw_capture(
            capture,
            args.output,
            frame_limit=args.frames,
            timeout_s=args.timeout,
            print_interval=args.print_interval,
        )
        exit_code = 0 if count else 2
    except KeyboardInterrupt:
        exit_code = 130
    except Exception:
        traceback.print_exc()

    # XRoboToolkit's callback teardown can block, and returning normally while
    # its native callback thread is still alive can crash Python during GIL
    # finalization. Saving has finished (or reported an error); terminate the
    # process without interpreter teardown so the shell gets control back.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
