#!/usr/bin/env python3
"""Normalise an NPZ capture into the canonical ``pico_raw`` recording format.

The offline retarget pipeline (``load_pico_raw_recording``) accepts exactly
what ``save_raw_recording`` writes:

    format_version, input_source, human_format, fps, fps_time_source,
    sampling_hz, raw_quaternion_convention, raw_pose_columns, joint_names,
    poses[frames,24,7], body_seq[frames], source_ns[frames], body_recv_ns[frames]

SDK captures written by ``record_raw_capture`` carry four extra ``capture_*``
keys; nothing else may differ (24 joints in SMPL order, xyzw quaternions,
monotonic time). This tool rewrites any such file through the canonical writer
so the offline publisher and raw visualizer use the same schema.

    python scripts/diagnostics/convert_to_pico_raw.py IN.npz OUT.npz
    python scripts/diagnostics/convert_to_pico_raw.py IN.npz OUT.npz --fallback-hz 50
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from booster_deploy.inputs.pico.raw_recording import (
    BODY_JOINT_NAMES,
    RAW_POSE_COLUMNS,
    save_raw_recording,
)


def _scalar(value) -> object:
    array = np.asarray(value)
    return array.item() if array.shape == () or array.size == 1 else array


def convert(source: Path, target: Path, fallback_hz: float) -> dict[str, object]:
    with np.load(source, allow_pickle=True) as data:
        for key in ("poses", "body_seq", "source_ns", "body_recv_ns"):
            if key not in data:
                raise SystemExit(f"{source}: missing required array {key!r}")
        poses = np.asarray(data["poses"], dtype=np.float64)
        body_seq = np.asarray(data["body_seq"], dtype=np.int64)
        source_ns = np.asarray(data["source_ns"], dtype=np.int64)
        body_recv_ns = np.asarray(data["body_recv_ns"], dtype=np.int64)
        joint_names = [str(name) for name in data["joint_names"]] if "joint_names" in data else list(BODY_JOINT_NAMES)
        columns = [str(name) for name in data["raw_pose_columns"]] if "raw_pose_columns" in data else list(RAW_POSE_COLUMNS)
        quat = str(_scalar(data["raw_quaternion_convention"])) if "raw_quaternion_convention" in data else "xyzw"
        declared_hz = float(np.asarray(data["sampling_hz"]).reshape(-1)[0]) if "sampling_hz" in data else fallback_hz
        extras = sorted(set(data.files) - {
            "format_version", "input_source", "human_format", "fps", "fps_time_source",
            "sampling_hz", "raw_quaternion_convention", "raw_pose_columns", "joint_names",
            "poses", "body_seq", "source_ns", "body_recv_ns",
        })

    # --- contract checks: refuse to write a file the loader would reject later ---
    if poses.ndim != 3 or poses.shape[1] != len(BODY_JOINT_NAMES) or poses.shape[2] < 7:
        raise SystemExit(f"poses must be [frames,{len(BODY_JOINT_NAMES)},>=7], got {poses.shape}")
    if not (len(body_seq) == len(source_ns) == len(body_recv_ns) == poses.shape[0]):
        raise SystemExit("poses / body_seq / source_ns / body_recv_ns length mismatch")
    if joint_names != list(BODY_JOINT_NAMES):
        raise SystemExit(
            "joint order does not match the pipeline (expected SMPL 24 order):\n"
            f"  got     : {joint_names}\n  expected: {list(BODY_JOINT_NAMES)}"
        )
    if quat != "xyzw":
        raise SystemExit(f"quaternion convention must be xyzw, got {quat!r}")
    if not np.all(np.diff(source_ns) > 0):
        raise SystemExit("source_ns must be strictly increasing")
    if not np.isfinite(poses).all():
        raise SystemExit("poses contain NaN/Inf")

    save_raw_recording(
        target,
        poses=[frame for frame in poses],
        body_seq=[int(seq) for seq in body_seq],
        source_ns=[int(ns) for ns in source_ns],
        body_recv_ns=[int(ns) for ns in body_recv_ns],
        fallback_hz=declared_hz or fallback_hz,
    )

    span_s = (int(source_ns[-1]) - int(source_ns[0])) / 1e9
    return {
        "frames": int(poses.shape[0]),
        "span_s": round(span_s, 3),
        "effective_fps": round((len(source_ns) - 1) / span_s, 3) if span_s > 0 else 0.0,
        "declared_hz": declared_hz,
        "dropped_extra_keys": extras,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument("--fallback-hz", type=float, default=50.0,
                        help="sampling grid used when the source declares none")
    args = parser.parse_args()

    info = convert(args.source.expanduser().resolve(), args.target.expanduser().resolve(),
                   args.fallback_hz)
    print(f"[convert] {args.source} -> {args.target}")
    print(f"[convert] frames={info['frames']} span={info['span_s']}s "
          f"eff_fps={info['effective_fps']} declared_hz={info['declared_hz']}")
    if info["dropped_extra_keys"]:
        print(f"[convert] dropped extra keys: {info['dropped_extra_keys']}")

    # Round-trip through the pipeline loader: the write is only useful if it loads.
    from booster_deploy.inputs.pico.offline_recording import load_pico_raw_recording

    recording = load_pico_raw_recording(args.target.expanduser().resolve())
    print(f"[convert] loader OK: poses={recording.poses.shape} fps={recording.fps:.3f} "
          f"duration={recording.time_s[-1] - recording.time_s[0]:.3f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
