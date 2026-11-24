"""Per-robot consistency report: real low-state feedback compared with one shared MuJoCo replay."""
from __future__ import annotations

import argparse
import base64
import csv
import json
import re
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor
from html import escape
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from booster_deploy.telemetry import reader  # noqa: E402
from booster_deploy.telemetry.trajectory import REPORT_STEM, committed_playbacks, is_unlimited, playback_at  # noqa: E402

DEFAULT_MJCF = REPO_ROOT / "booster_deploy" / "assets" / "t2" / "t2_31dof_fc.xml"
LAG_SEARCH_S = 3.0
COMPLETE_COVERAGE = 0.95
TIME_BINS = 240
COMMON_JOINT_MIN_DEG = 2.0    # a joint counts as "everyone is bad here" above this row median
OUTLIER_DEV_DEG = 1.5         # a robot counts as "outlier on this joint" above this deviation
ROBOT_BIAS_DEG = 0.3          # a robot counts as globally biased above this mean deviation
MINIMUM_PAIRS = 100


def joint_names_from_mjcf(path: Path) -> list[str]:
    text = path.read_text(errors="ignore")
    names = []
    for match in re.finditer(r"<joint\b([^>]*)>", text):
        attrs = match.group(1)
        name = re.search(r'name="([^"]+)"', attrs)
        kind = re.search(r'type="([^"]+)"', attrs)
        if not name or (kind and kind.group(1) in ("fixed", "free")):
            continue
        names.append(name.group(1))
    return names


def group_of(name: str) -> str:
    if "head" in name:
        return "head"
    if name.startswith("left_") or "_left" in name:
        side = "left"
    elif name.startswith("right_") or "_right" in name:
        side = "right"
    else:
        side = ""
    if any(k in name for k in ("shoulder", "elbow", "wrist")):
        return f"{side}_arm"
    if "waist" in name:
        return "waist"
    if any(k in name for k in ("hip", "knee", "ankle")):
        return f"{side}_leg"
    return "other"


def wrap(angle: np.ndarray) -> np.ndarray:
    """Wrap to (-pi, pi]; only used for base roll/pitch, joint angles are limit-bounded and never need it."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def lag_scan(real: np.ndarray, sim: np.ndarray, max_lag: int) -> tuple[int, float]:
    """Return (k, r) with real[t] ~ sim[t-k] and r the Pearson correlation of the overlapping window.

    The correlation is normalised per lag on the overlap: a raw dot product would reward the
    longest overlap (k=0) and systematically underestimate the real delay.
    """
    n = len(real)
    finite = np.isfinite(real) & np.isfinite(sim)
    if finite.mean() < 0.5:
        return 0, float("nan")
    a = np.where(finite, real, np.nanmean(real[finite]))
    b = np.where(finite, sim, np.nanmean(sim[finite]))
    best_r, best_lag = -np.inf, 0
    for k in range(-max_lag, max_lag + 1):
        x, y = (a[k:], b[: n - k]) if k >= 0 else (a[: n + k], b[-k:])
        if len(x) < 0.5 * n:          # require at least half the window to overlap
            continue
        x = x - x.mean()
        y = y - y.mean()
        denom = np.linalg.norm(x) * np.linalg.norm(y)
        if denom <= 0:
            continue
        r = float(x @ y) / denom
        if r > best_r:
            best_r, best_lag = r, k
    return best_lag, best_r


def shift(values: np.ndarray, k: int) -> np.ndarray:
    out = np.full_like(values, np.nan)
    if k > 0:
        out[k:] = values[: len(values) - k]
    elif k < 0:
        out[:k] = values[-k:]
    else:
        out[:] = values
    return out


def quat_to_rpy(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    return np.column_stack([
        np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)),
        np.arcsin(np.clip(2 * (w * y - z * x), -1, 1)),
        np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)),
    ])


def read_events(run: Path) -> dict:
    events, keys = {}, {}
    for path in sorted(run.glob("events.*.msgpack")):
        for rec in reader.iter_file(str(path)):
            name = rec["data"].get("event")
            if name in ("policy_session_start", "policy_session_end", "return_to_walk_requested", "tracking_requested"):
                events.setdefault(name, int(rec["t_mono_ns"]))
    for path in sorted(run.glob("keys.*.msgpack")):
        for rec in reader.iter_file(str(path)):
            if rec["data"].get("phase") != "received":
                continue
            key = rec["data"].get("key")
            if key in ("A", "B"):
                keys.setdefault(key, int(rec["t_mono_ns"]))
    return {"events": events, "keys": keys}


def read_state(run: Path, lo: int, hi: int) -> dict:
    """Read the dominant state writer only: several processes log state and the 50 Hz policy mirror would skew rates."""
    buckets: dict[str, dict] = {}
    for path in sorted(run.glob("state.*.msgpack")):
        for rec in reader.iter_file(str(path)):
            t = int(rec["t_mono_ns"])
            if not (lo <= t <= hi):
                continue
            data = rec["data"]
            q = data.get("q")
            if not isinstance(q, np.ndarray) or q.size != 31:
                continue
            token = str(rec.get("writer_token") or "?")
            bucket = buckets.setdefault(token, {"t": [], "q": [], "dq": [], "rpy": [], "bad_rpy": 0})
            bucket["t"].append(t)
            bucket["q"].append(q.astype(np.float64))
            dq = data.get("dq")
            bucket["dq"].append(dq.astype(np.float64) if getattr(dq, "size", 0) == 31 else np.full(31, np.nan))
            rpy = data.get("root_rpy_w")
            if getattr(rpy, "size", 0) == 3:
                bucket["rpy"].append(rpy.reshape(-1).astype(np.float64))
            else:
                bucket["bad_rpy"] += 1
                bucket["rpy"].append(np.full(3, np.nan))
    if not buckets:
        return {"t": np.zeros(0, dtype=np.int64), "q": np.zeros((0, 31)), "dq": np.zeros((0, 31)),
                "rpy": np.zeros((0, 3)), "records": 0, "bad_root_rpy": 0, "writers": {}}
    token = max(buckets, key=lambda k: len(buckets[k]["t"]))
    bucket = buckets[token]
    t = np.asarray(bucket["t"], dtype=np.int64)
    order = np.argsort(t, kind="stable")
    t = t[order]
    keep = np.concatenate([[True], np.diff(t) > 0]) if len(t) > 1 else np.ones(len(t), dtype=bool)
    return {
        "t": t[keep],
        "q": np.vstack(bucket["q"])[order][keep],
        "dq": np.vstack(bucket["dq"])[order][keep],
        "rpy": np.vstack(bucket["rpy"])[order][keep],
        "records": int(keep.sum()),
        "bad_root_rpy": bucket["bad_rpy"],
        "writers": {k: len(v["t"]) for k, v in buckets.items()},
    }


def health_of(t_ns: np.ndarray) -> dict:
    if len(t_ns) < 10:
        return {}
    dt_ms = np.diff(t_ns) / 1e6
    span = (t_ns[-1] - t_ns[0]) / 1e9
    gaps = dt_ms > 10.0
    return {
        "samples": int(len(t_ns)),
        "span_s": float(span),
        "effective_hz": float(len(t_ns) / span) if span > 0 else float("nan"),
        "dt_p50_ms": float(np.median(dt_ms)),
        "dt_p95_ms": float(np.percentile(dt_ms, 95)),
        "dt_p99_ms": float(np.percentile(dt_ms, 99)),
        "dt_max_ms": float(dt_ms.max()),
        "gaps_gt_10ms": int(gaps.sum()),
        "gap_time_ms": float(dt_ms[gaps].sum()),
    }


def read_publish(run: Path) -> tuple[np.ndarray, np.ndarray]:
    """(t_ns, native_index) of the recorded offline-trajectory frames, in order."""
    rows = []
    for path in sorted(run.glob("publish.*.msgpack")):
        for rec in reader.iter_file(str(path)):
            data = rec["data"]
            index = data.get("native_index")
            if index is not None:
                rows.append((int(rec["t_mono_ns"]), int(index)))
    rows.sort()
    if not rows:
        raise RuntimeError(f"{run.name}: no publish.native_index records")
    return np.array([r[0] for r in rows], dtype=np.int64), np.array([r[1] for r in rows], dtype=np.int64)


def robot_of(run: Path) -> str:
    """Trailing run-name token, e.g. ..._T2-132-3 -> T2-132-3."""
    return run.name.rsplit("_", 1)[-1]


def _source_key(value):
    """Normalise a lineage source id ([namespace, index]) into a hashable key."""
    if isinstance(value, (list, tuple)) and len(value) == 2 and not isinstance(value[0], (list, tuple, dict)):
        try:
            return (str(value[0]), int(value[1]))
        except (TypeError, ValueError):
            return None
    return None


def _collect_source_keys(node, found):
    """Collect every lineage source id reachable from a record (dicts, lists, nested origins)."""
    if isinstance(node, dict):
        for field in ("source_ids", "source_id"):
            if field not in node:
                continue
            value = node.get(field)
            items = (value if isinstance(value, (list, tuple)) and value and isinstance(value[0], (list, tuple))
                     else (value,))
            for item in items:
                key = _source_key(item)
                if key:
                    found.add(key)
        for value in node.values():
            _collect_source_keys(value, found)
    elif isinstance(node, (list, tuple)):
        key = _source_key(node)
        if key:
            found.add(key)
            return
        for item in node:
            _collect_source_keys(item, found)


def _window_frames(window):
    """Per-frame lineage entries of an encoder window: retarget list form or SMPL dict form."""
    if isinstance(window, dict):
        identity = window.get("identity")
        positions = identity.get("positions") if isinstance(identity, dict) else None
        return list(positions or []), identity
    return list(window or []), None


def _transition_reference(node):
    if isinstance(node, dict):
        return node.get('kind') in {'blend', 'activation_hold'} or any(
            _transition_reference(value) for value in node.values())
    if isinstance(node, (list, tuple)):
        return any(_transition_reference(value) for value in node)
    return False


def resolve_consumed_index(window, native_of):
    """Map an encoder window to (loop, native clip index) via recorded lineage source ids.

    ``sample_idx`` is the 50 Hz resampler output counter and never a native frame number;
    only the publish table's ``source_id -> (loop, native_index)`` mapping is authoritative.
    """
    positions, _ = _window_frames(window)
    if positions and _transition_reference(positions[0]):
        return None
    found = set()
    if positions:
        _collect_source_keys(positions[0], found)
    hits = [native_of[key] for key in found if key in native_of]
    if not hits or len({item[0] for item in hits}) != 1:
        return None
    loop = hits[0][0]
    return loop, min(index for _, index in hits)


def read_executions(run: Path, playbacks=None) -> list[dict]:
    """Map observed consumer frames to clip indices, isolated by playback cycle when recorded."""
    starts = [play.start_ns for play in playbacks] if playbacks is not None else []
    by_cycle = {play.cycle: play for play in playbacks} if playbacks is not None else {}
    seen_sources = set()
    pub = []
    native_of = {}
    for path in sorted(run.glob("publish.*.msgpack")):
        for rec in reader.iter_file(str(path)):
            data = rec["data"]
            if data.get("native_seq") is None or data.get("native_index") is None:
                continue
            stamp = int(rec["t_mono_ns"])
            key = _source_key(data.get("source_id"))
            if playbacks is None:
                loop = int(data.get("loop") or 0)
            else:
                play = playback_at(playbacks, starts, stamp)
                if play is None or data.get("offline_cycle") not in (None, 0, play.cycle):
                    continue
                loop = play.cycle
                if key and key in seen_sources:
                    continue  # PICO publishes raw and converted channels with the same source ID.
                if key:
                    seen_sources.add(key)
            index = int(data["native_index"])
            pub.append((stamp, int(data["native_seq"]), index, loop))
            if key:
                native_of[key] = (loop, index)
    if not pub and playbacks is None:
        return []
    pub.sort()
    ticks = []
    pub_by_loop = {}
    for row in pub:
        pub_by_loop.setdefault(row[3], []).append(row)
    consumed_by_loop = {}
    excluded_by_loop = {}
    for path in sorted(run.glob("encoder.*.msgpack")):
        for rec in reader.iter_file(str(path)):
            data = rec["data"]
            if data.get("phase") != "entry":
                continue
            positions, _ = _window_frames(data.get("window"))
            stamp = int(rec["t_mono_ns"])
            play = playback_at(playbacks, starts, stamp) if playbacks is not None else None
            if playbacks is not None and play is None:
                continue
            keys = set()
            _collect_source_keys(positions, keys)
            for key in keys:
                hit = native_of.get(key)
                if hit and (play is None or hit[0] == play.cycle):
                    consumed_by_loop.setdefault(hit[0], set()).add(hit[1])
            if positions and _transition_reference(positions[0]):
                first_keys = set()
                _collect_source_keys(positions[0], first_keys)
                loops = {native_of[key][0] for key in first_keys if key in native_of}
                if len(loops) == 1 and (play is None or loops == {play.cycle}):
                    loop = next(iter(loops))
                    excluded_by_loop[loop] = excluded_by_loop.get(loop, 0) + 1
            resolved = resolve_consumed_index(data.get("window"), native_of)
            if resolved is not None and (play is None or resolved[0] == play.cycle):
                ticks.append((stamp, resolved[0], resolved[1]))
    ticks.sort()
    out = []
    for loop in (by_cycle if playbacks is not None else sorted(pub_by_loop)):
        rows = pub_by_loop.get(loop, [])
        play = by_cycle.get(loop)
        first = {}
        for t, tick_loop, clip in ticks:
            if tick_loop == loop:
                first.setdefault(int(clip), int(t))
        clips = np.array(sorted(first), dtype=np.int64) if first else np.array([], dtype=np.int64)
        clip_t = {}
        for t, _, clip, _ in rows:
            clip_t.setdefault(int(clip), int(t))
        out.append({"loop": loop, "first": first, "clips": clips, "pub_clip_t": clip_t,
                    "consumed_clips": sorted(consumed_by_loop.get(loop, set())),
                    "excluded_transition_ticks": excluded_by_loop.get(loop, 0),
                    "t0": play.start_ns if play else rows[0][0],
                    "t1": (play.end_ns if play else rows[-1][0]), "published": len(rows)})
        if play is not None:
            out[-1].update(trajectory_index=play.trajectory_index, trajectory_path=play.trajectory_path,
                           playback_cycle=play.cycle)
    return out


def segment_label(run: Path, nth: int, robot_id: str | None = None) -> str:
    """Build per-execution robot IDs; continue a numeric suffix when one is present."""
    if robot_id == "unknown" or (robot_id and re.fullmatch(r"\d+", str(robot_id))):
        base = f"T2-{robot_id}"
    else:
        base = robot_of(run)
    match = re.match(r"^(T2-(?:\d+|unknown))(?:-(\d+))?$", base)
    if not match:
        return f"{base}-{nth + 1}"
    stem, index = match.group(1), match.group(2)
    if index is None:
        return f"{stem}-{nth + 1}"
    return f"{stem}-{int(index) + nth}"


def select_complete_executions(executions: list[dict], reference_indices) -> tuple[list[dict], list[dict]]:
    """Require observed encoder-window lineage for at least 95% of reference frames."""
    reference = {int(index) for index in reference_indices if int(index) >= 0}
    denominator = max(len(reference), 1)
    complete, skipped = [], []
    for execution in executions:
        item = dict(execution)
        clips = {int(index) for index in execution.get("clips", [])}
        item["tick_coverage"] = len(reference & clips) / denominator
        consumed = {int(index) for index in execution.get("consumed_clips", clips)}
        item["coverage"] = len(reference & consumed) / denominator
        (complete if clips and item["coverage"] >= COMPLETE_COVERAGE else skipped).append(item)
    return complete, skipped


def execution_reference_indices(run: Path, fallback, playback=None) -> np.ndarray:
    """Use the recorded input selection, not a truncated simulated clip, for completeness."""
    if playback is not None:
        return _recorded_reference_indices(run, playback.trajectory_path, playback.input_selection or {})
    config_path = run / "execution_config.json"
    if not config_path.is_file():
        return np.asarray(fallback, dtype=np.int64)
    config = json.loads(config_path.read_text())
    if not config.get("offline"):
        return np.asarray(fallback, dtype=np.int64)
    trajectory = config.get("trajectory") or {}
    declared = trajectory.get("path")
    selection_path = run / "replay.input.json"
    selection = json.loads(selection_path.read_text()) if selection_path.is_file() else {}
    return _recorded_reference_indices(run, declared, selection)


def _recorded_reference_indices(run, declared, selection):
    if declared and selection.get("path") and selection["path"] != declared:
        raise ValueError(f"{run.name}: recorded input selection does not match execution trajectory")
    start = selection.get("start_frame", 0)
    end = selection.get("end_frame")
    if end is None:
        from booster_deploy.telemetry.simulation import _recorded_path
        if not declared:
            raise ValueError(f"{run.name}: offline trajectory extent is missing; refusing truncated-reference coverage")
        source = Path(_recorded_path(str(declared), run))
        if not source.is_file():
            raise FileNotFoundError(f"{run.name}: trajectory extent unavailable: {declared}")
        if source.suffix.lower() == ".bvh":
            with source.open(encoding="utf-8") as handle:
                for line in handle:
                    match = re.fullmatch(r"\s*Frames:\s*(\d+)\s*", line, re.IGNORECASE)
                    if match:
                        end = int(match[1])
                        break
        else:
            with np.load(source, allow_pickle=False) as archive:
                if "poses" not in archive.files:
                    raise ValueError(f"{run.name}: trajectory has no poses; record an explicit input frame range")
                end = len(archive["poses"])
    if type(start) is not int or type(end) is not int or not 0 <= start < end:
        raise ValueError(f"{run.name}: invalid recorded trajectory frame range: {start}..{end}")
    return np.arange(start, end, dtype=np.int64)


def sim_frame_indices(sim_run: Path, sim_npz: Path) -> tuple[np.ndarray, np.ndarray]:
    """Map every simulation frame to the offline-trajectory index it consumed, plus its reference timestamp."""
    manifest = json.loads((sim_run / "manifest.json").read_text())
    origin = int(manifest["origin_mono_ns"])
    sim = np.load(sim_npz)
    seconds = sim["seconds"].astype(np.float64)
    t_ns = origin + (seconds * 1e9).astype(np.int64)
    if "native_index" in sim.files:
        indices = sim["native_index"].astype(np.int64)
    else:
        # Legacy npz: no per-frame lineage, so fall back to nearest publish time and say so.
        pub_t, pub_i = read_publish(sim_run)
        right = np.searchsorted(pub_t, t_ns).clip(0, len(pub_i) - 1)
        left = (right - 1).clip(0, len(pub_i) - 1)
        closest = np.where(np.abs(pub_t[left] - t_ns) <= np.abs(pub_t[right] - t_ns), left, right)
        indices = pub_i[closest]
        print(f"warning: {sim_npz} has no per-frame native_index (legacy schema); "
              "pairing falls back to nearest publish time", file=sys.stderr)
    if len(indices) > 1 and not np.all(np.diff(indices) == 1):
        gaps = int((np.diff(indices) != 1).sum())
        print(f"warning: simulation frames are not index-contiguous ({gaps} breaks); pairing still uses per-frame indices",
              file=sys.stderr)
    return indices, t_ns


def sim_segments(indices: np.ndarray) -> list[tuple[int, int]]:
    """Split simulation frames into replays: native_index restarts at each replay."""
    if not len(indices):
        return []
    starts = [0]
    for position in range(1, len(indices)):
        if int(indices[position]) < int(indices[position - 1]):
            starts.append(position)
    return [(start, starts[i + 1] if i + 1 < len(starts) else len(indices))
            for i, start in enumerate(starts)]


def select_simulation_frames(indices, stamps, execution=None, segment_ids=None):
    """Select one simulated replay and its first observation of each native frame."""
    indices, stamps = np.asarray(indices), np.asarray(stamps)
    if len(indices) != len(stamps):
        raise ValueError("simulation timestamps/indices length mismatch")
    if segment_ids is None:
        groups = [np.arange(start, end) for start, end in sim_segments(indices)]
    else:
        segment_ids = np.asarray(segment_ids)
        if len(segment_ids) != len(indices):
            raise ValueError("simulation segment/index length mismatch")
        groups = [np.flatnonzero(segment_ids == key) for key in dict.fromkeys(segment_ids.tolist())]
    candidates = []
    for group in groups:
        group = group[indices[group] >= 0]
        if execution is not None:
            group = group[(stamps[group] >= execution["t0"]) & (stamps[group] <= execution["t1"])]
        if len(group):
            candidates.append(group)
    if not candidates:
        raise ValueError("no simulated replay overlaps the requested execution")
    selected = max(candidates, key=lambda group: len(set(indices[group].tolist())))
    first = {}
    for frame in selected:
        first.setdefault(int(indices[frame]), int(frame))
    return np.asarray(list(first.values()), dtype=np.int64)


def analyse_run(job: dict) -> dict:
    run = Path(job["run"])
    sim = np.load(job["sim_npz"])
    sim_qpos = sim["qpos"].astype(np.float64)
    sim_q = sim_qpos[:, 7:]
    sim_root = sim_qpos[:, :7]
    sim_indices = np.asarray(job["sim_indices"], dtype=np.int64)

    first_tick = job["execution"]["first"]
    if not first_tick:
        raise RuntimeError(f"{run.name}: execution {job['execution']['loop']} has no paired ticks")

    # frame alignment: simulation frame i <-> the tick that consumed reference index i
    reference_frames = job.get("sim_frames", range(len(sim_indices)))
    pairs = [(k, int(sim_indices[k])) for k in reference_frames if int(sim_indices[k]) in first_tick]
    if len(pairs) < MINIMUM_PAIRS:
        raise RuntimeError(f"{run.name}: only {len(pairs)} frames share a reference index with the robot")
    frame_no = np.array([p[0] for p in pairs])
    pair_index = np.array([p[1] for p in pairs], dtype=np.int64)
    t_tick = np.array([first_tick[i] for i in pair_index], dtype=np.int64)
    tick_t = t_tick  # response-lag search shifts inside this execution only

    lo, hi = int(t_tick.min()) - 100_000_000, int(t_tick.max()) + 100_000_000
    if "playback_cycle" in job["execution"]:
        lo = max(lo, job["execution"]["t0"])
        boundary = job["execution"]["t1"]
        if boundary is not None:
            hi = min(hi, boundary - 1)
    state = read_state(run, lo, hi)
    if len(state["t"]) < 100:
        raise RuntimeError(f"{run.name}: no state samples in the execution window")

    def state_at(times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pos = np.searchsorted(state["t"], times).clip(0, len(state["t"]) - 1)
        return state["q"][pos], state["rpy"][pos]

    q_sim = sim_q[frame_no]
    root_sim = sim_root[frame_no]
    q_real, rpy_real = state_at(t_tick)

    # zero-shift (frame) alignment is the primary comparison
    err = q_real - q_sim
    rmse = np.degrees(np.sqrt((err ** 2).mean(0)))
    p95 = np.degrees(np.percentile(np.abs(err), 95, axis=0))
    maxabs = np.degrees(np.abs(err).max(0))
    bias = np.degrees(err.mean(0))

    # reference offset: publish of clip frame i -> tick that consumed i (contains the 10-frame look-ahead by design).
    # Only frames published inside this execution count: a multi-run log restarts native_index per replay.
    pub_map = job["execution"]["pub_clip_t"]
    offsets = np.array([t_tick[m] - pub_map[i] for m, i in enumerate(pair_index) if i in pub_map], dtype=np.int64)
    ref_offset_ms = float(np.median(offsets) / 1e6) if len(offsets) else float("nan")

    # residual response delay: best uniform tick shift after frame alignment
    def tick_positions(shift: int) -> np.ndarray:
        pos = np.searchsorted(tick_t, t_tick).clip(0, len(tick_t) - 1) + shift
        return tick_t[pos.clip(0, len(tick_t) - 1)]

    best_shift, best_rmse = 0, float(np.median(rmse))
    for shift in range(-10, 11):
        qr, _ = state_at(tick_positions(shift))
        value = float(np.median(np.degrees(np.sqrt(((qr - q_sim) ** 2).mean(0)))))
        if value < best_rmse:
            best_shift, best_rmse = shift, value
    per_joint_shift = np.zeros(31, dtype=int)
    for j in range(31):
        best_j, best_v = 0, float("inf")
        for shift in range(-10, 11):
            qr, _ = state_at(tick_positions(shift))
            v = float(np.sqrt(((qr[:, j] - q_sim[:, j]) ** 2).mean()))
            if v < best_v:
                best_j, best_v = shift, v
        per_joint_shift[j] = best_j
    tick_ms = float(np.median(np.diff(tick_t)) / 1e6) if len(tick_t) > 1 else 20.0

    frame_time = (sim["seconds"][frame_no].astype(np.float64) - sim["seconds"][frame_no[0]])
    settled = frame_time >= 2.0
    settled_rmse = (np.degrees(np.sqrt((err[settled] ** 2).mean(0))) if settled.sum() > 20
                    else np.full(31, np.nan))

    if "playback_cycle" in job["execution"]:
        dq_sim = np.gradient(q_sim, frame_time, axis=0)
    else:
        dq_sim = np.gradient(sim_q, float(np.median(np.diff(sim["seconds"]))), axis=0)[frame_no]
    moving = np.abs(dq_sim) > 0.5
    static_rmse = np.full(31, np.nan)
    moving_rmse = np.full(31, np.nan)
    for j in range(31):
        if (~moving[:, j]).sum() > 20:
            static_rmse[j] = np.degrees(np.sqrt((err[~moving[:, j], j] ** 2).mean()))
        if moving[:, j].sum() > 20:
            moving_rmse[j] = np.degrees(np.sqrt((err[moving[:, j], j] ** 2).mean()))

    rpy_sim = quat_to_rpy(root_sim[:, 3:7])
    both = ~np.isnan(rpy_sim).any(1) & ~np.isnan(rpy_real).any(1)
    root = {"roll_rms_deg": None, "pitch_rms_deg": None, "yaw_rms_deg": None,
            "yaw_offset_deg": None, "valid_samples": int(both.sum()), "valid_ratio": float(both.mean())}
    if both.sum() > 20:
        droll = np.degrees(wrap(rpy_real[both, 0] - rpy_sim[both, 0]))
        dpitch = np.degrees(wrap(rpy_real[both, 1] - rpy_sim[both, 1]))
        dyaw = np.unwrap(rpy_real[both, 2] - rpy_sim[both, 2])
        offset = float(np.mean(dyaw))
        dyaw = dyaw - offset
        root.update({"roll_rms_deg": float(np.sqrt((droll ** 2).mean())),
                     "pitch_rms_deg": float(np.sqrt((dpitch ** 2).mean())),
                     "yaw_rms_deg": float(np.sqrt((dyaw ** 2).mean())),
                     "yaw_offset_deg": float(np.degrees(offset))})

    bin_edges = np.linspace(0, len(err), TIME_BINS + 1).astype(int)
    heat = np.full((31, TIME_BINS), np.nan)
    for b in range(TIME_BINS):
        a, z = bin_edges[b], max(bin_edges[b] + 1, bin_edges[b + 1])
        if a < len(err):
            heat[:, b] = np.degrees(np.abs(err[a:z]).mean(0))

    health = health_of(state["t"])
    health.update({
        "records": state["records"],
        "bad_root_rpy": state["bad_root_rpy"],
        "root_rpy_valid_ratio": float(1 - state["bad_root_rpy"] / max(1, state["records"])),
        "replay_window_s": float((t_tick[-1] - t_tick[0]) / 1e9),
        "state_writers": state.get("writers") or {},
        "nominal_hz": 1000.0 if health.get("dt_p50_ms", 1.0) < 1.5 else 500.0,
    })
    health["rate_ratio"] = health.get("effective_hz", float("nan")) / health["nominal_hz"]

    return {
        "run": run.name,
        "robot": job["label"],
        "sn": job.get("sn"),
        "execution": job["execution"]["loop"],
        **{key: job["execution"][key] for key in ("playback_cycle", "trajectory_index", "trajectory_path")
           if key in job["execution"]},
        "framewise": {
            "rmse_deg": rmse, "p95_deg": p95, "maxabs_deg": maxabs, "bias_deg": bias,
            "resp_lag_ms": per_joint_shift * tick_ms,
            "static_rmse_deg": static_rmse, "moving_rmse_deg": moving_rmse,
            "settled_rmse_deg": settled_rmse,
        },
        "summary": {
            "ref_offset_ms": ref_offset_ms,
            "resp_lag_ms": float(best_shift * tick_ms),
            "tick_ms": tick_ms,
            "rmse_median_deg": float(np.median(rmse)),
            "rmse_bestshift_median_deg": best_rmse,
            "rmse_worst_joint": int(np.argmax(rmse)),
            "rmse_worst_deg": float(rmse.max()),
            "p95_median_deg": float(np.median(p95)),
            "joint_l2_median_rad": float(np.median(np.linalg.norm(err, axis=1))),
            "rmse_settled_median_deg": float(np.median(settled_rmse)),
            "frames": int(len(err)),
            "settled_frames": int(settled.sum()),
            "index_coverage": float(len(err) / max(1, len(reference_frames))),
            "window_lineage_coverage": job["execution"].get("coverage"),
            "window_start_coverage": job["execution"].get("tick_coverage"),
            "simulation_reference_frames": len(reference_frames),
            "excluded_transition_ticks": job["execution"].get("excluded_transition_ticks", 0),
            "pairing_exclusion_policy": "exclude_blend_activation_hold",
        },
        "root": root,
        "health": health,
        "heatmap": heat,
    }


def collect_runs(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.is_dir() and (p / "manifest.json").exists())


def fmt(value, digits=2):
    if value is None:
        return "-"
    if isinstance(value, float):
        return "-" if value != value else f"{value:.{digits}f}"
    if isinstance(value, (np.floating,)):
        value = float(value)
        return "-" if value != value else f"{value:.{digits}f}"
    return str(value)


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


LEGEND_ENTRIES = [('参考帧偏移 ms', '真机把第 i 帧参考喂给策略的时刻 − 该帧的 publish 时刻。含策略固定的 10 帧前瞻（10×20 ms = 200 ms）与约 20 ms 链路开销，属设计值，只用于检查链路是否正常，不参与机器人优劣排序'), ('响应延迟 ms', '帧对齐之后，再把真机整体错开 k 拍（±10 帧内搜索）使误差最小的 k。表示「同一帧参考进来后，关节需要多久追上仿真的响应」，是跨机器人可比的量'), ('RMSE 中位(全段)°', '先算 31 个关节各自的 RMSE，再取这 31 个值的中位数；用整个比较窗口。单位：度'), ('RMSE 中位(稳态)°', '同上，但先去掉对齐后前 2 s（两侧从不同初始姿态收敛的瞬态）再算。单位：度'), ('RMSE最大的关节', '31 个关节里 RMSE 最大的那一根（给出关节名），用于定位最差的单关节'), ('该关节RMSE°', '「RMSE最大的关节」那一根关节的 RMSE 数值，不是瞬时最大误差。单位：度'), ('P95中位°', '逐关节取 |误差| 的 95 分位（该关节 95% 的时刻误差都比它小），再对 31 个关节取中位数。单位：度'), ('左腿° / 右腿° / 腰° / 左臂° / 右臂° / 头°', '该组内各关节 RMSE 的中位数（同一条 RMSE 口径，只是在组内聚合）。单位：度'), ('基座 roll° / pitch° / yaw°', '骨盆（root 自由体）的整机姿态 RMS 误差，来自真机 IMU 与仿真基座四元数对比，不是单个关节的量；yaw 已 unwrap 并去掉常量偏置。单位：度'), ('名义Hz', '固件标称反馈频率：dt 中位≈1 ms → 1000 Hz 固件，≈2 ms → 500 Hz 固件'), ('有效 Hz', '比较窗口内的样本数 ÷ 窗口时长，即实际交付率对应的频率'), ('有效/名义%', '实际交付率，1 kHz 档当前 76–86%，500 Hz 档 80–93%；比绝对 Hz 更能反映链路质量'), ('缺口次数', '相邻样本间隔 dt > 10 ms 的段数。注意阈值含义随固件不同：500 Hz 固件正常 2 ms，10 ms ≈ 连丢 4 帧；1 kHz 正常 1 ms，10 ms ≈ 连丢 9 帧'), ('缺口 ms', '上述所有缺口的时间之和，反映这段时间里反馈总共「停」了多久')]

LEGEND_HTML = """<dt>参考帧偏移 ms</dt><dd>真机把第 i 帧参考喂给策略的时刻 − 该帧的 publish 时刻。含策略固定的 10 帧前瞻（10×20 ms = 200 ms）与约 20 ms 链路开销，属设计值，只用于检查链路是否正常，不参与机器人优劣排序</dd><dt>响应延迟 ms</dt><dd>帧对齐之后，再把真机整体错开 k 拍（±10 帧内搜索）使误差最小的 k。表示「同一帧参考进来后，关节需要多久追上仿真的响应」，是跨机器人可比的量</dd><dt>RMSE 中位(全段)°</dt><dd>先算 31 个关节各自的 RMSE，再取这 31 个值的中位数；用整个比较窗口。单位：度</dd><dt>RMSE 中位(稳态)°</dt><dd>同上，但先去掉对齐后前 2 s（两侧从不同初始姿态收敛的瞬态）再算。单位：度</dd><dt>RMSE最大的关节</dt><dd>31 个关节里 RMSE 最大的那一根（给出关节名），用于定位最差的单关节</dd><dt>该关节RMSE°</dt><dd>「RMSE最大的关节」那一根关节的 RMSE 数值，不是瞬时最大误差。单位：度</dd><dt>P95中位°</dt><dd>逐关节取 |误差| 的 95 分位（该关节 95% 的时刻误差都比它小），再对 31 个关节取中位数。单位：度</dd><dt>左腿° / 右腿° / 腰° / 左臂° / 右臂° / 头°</dt><dd>该组内各关节 RMSE 的中位数（同一条 RMSE 口径，只是在组内聚合）。单位：度</dd><dt>基座 roll° / pitch° / yaw°</dt><dd>骨盆（root 自由体）的整机姿态 RMS 误差，来自真机 IMU 与仿真基座四元数对比，不是单个关节的量；yaw 已 unwrap 并去掉常量偏置。单位：度</dd><dt>名义Hz</dt><dd>固件标称反馈频率：dt 中位≈1 ms → 1000 Hz 固件，≈2 ms → 500 Hz 固件</dd><dt>有效 Hz</dt><dd>比较窗口内的样本数 ÷ 窗口时长，即实际交付率对应的频率</dd><dt>有效/名义%</dt><dd>实际交付率，1 kHz 档当前 76–86%，500 Hz 档 80–93%；比绝对 Hz 更能反映链路质量</dd><dt>缺口次数</dt><dd>相邻样本间隔 dt &gt; 10 ms 的段数。注意阈值含义随固件不同：500 Hz 固件正常 2 ms，10 ms ≈ 连丢 4 帧；1 kHz 正常 1 ms，10 ms ≈ 连丢 9 帧</dd><dt>缺口 ms</dt><dd>上述所有缺口的时间之和，反映这段时间里反馈总共「停」了多久</dd>"""

CSS = """
:root { color-scheme: dark; }
body { margin:0; background:#14161a; color:#e7e9ee;
       font-family: -apple-system, "Segoe UI", "Noto Sans CJK SC", "Microsoft YaHei", sans-serif; }
.wrap { max-width:1500px; margin:0 auto; padding:28px 20px 60px; }
h1 { font-size:22px; margin:0 0 4px; }
h2 { font-size:16px; margin:30px 0 10px; color:#9fd8d4; font-weight:600; }
.sub { color:#8b93a7; font-size:13px; margin-bottom:18px; }
.cards { display:flex; gap:12px; flex-wrap:wrap; margin:14px 0 6px; }
.card { background:#1b1f27; border:1px solid #2a303c; border-radius:10px; padding:10px 16px; min-width:170px; }
.card .k { color:#8b93a7; font-size:12px; }
.card .v { font-size:21px; font-weight:600; margin-top:2px; }
.card .v small { font-size:12px; color:#8b93a7; font-weight:400; }
.scroll { overflow-x:auto; border:1px solid #2a303c; border-radius:10px; }
.scroll.narrow { display:inline-block; max-width:100%; }
table { border-collapse:collapse; width:100%; font-size:13px; font-variant-numeric:tabular-nums; }
th, td { padding:5px 10px; border-bottom:1px solid #232935; text-align:right; white-space:nowrap; }
th { background:#1b1f27; color:#9fb0c7; font-weight:600; position:sticky; top:0; }
td:first-child, th:first-child { text-align:left; position:sticky; left:0; background:#1b1f27; z-index:1; }
tr:hover td:first-child { background:#222833; }
.scroll.narrow td:first-child, .scroll.narrow th:first-child { position:static; background:transparent; }
tr:hover td { background:#1b2029; }
tr.grp td { background:#191d25; }
.bad { color:#ff8080; }
.warn { color:#ffc46b; }
.good { color:#6ee7a8; }
.fig { margin:14px 0 26px; }
.fig img { max-width:100%; border:1px solid #2a303c; border-radius:10px; display:block; }
.fig .cap { color:#8b93a7; font-size:12px; margin-top:6px; }
.note { color:#8b93a7; font-size:13px; line-height:1.7; }
.legend { display:grid; grid-template-columns:230px 1fr; gap:5px 16px; font-size:13px; line-height:1.5;
          background:#1b1f27; border:1px solid #2a303c; border-radius:10px; padding:14px 18px; }
.legend dt { color:#9fb0c7; font-weight:600; }
.legend dd { margin:0; color:#c6ccd8; }
nav.toc { background:#1b1f27; border:1px solid #2a303c; border-radius:10px; padding:10px 16px; margin:6px 0 24px;
          font-size:13px; color:#c6ccd8; }
nav.toc a { color:#9fd8d4; text-decoration:none; margin-right:14px; white-space:nowrap; }
nav.toc a:hover { text-decoration:underline; }
section { border-top:1px solid #2a303c; padding-top:26px; margin-top:26px; }
section:first-of-type { border-top:none; margin-top:0; padding-top:0; }
code { background:#1b1f27; padding:1px 5px; border-radius:4px; font-size:12px; }
"""


def _page(title: str, subtitle: str, body: str) -> str:
    return ("<!doctype html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\">"
            f"<title>{title}</title><style>{CSS}</style></head><body><div class=\"wrap\">"
            f"<h1>{title}</h1><div class=\"sub\">{subtitle}</div>{body}</div></body></html>")


def _table(headers: list[str], rows: list[list], row_class: list[str] | None = None, narrow: bool = False) -> str:
    out = [f"<div class=\"scroll{' narrow' if narrow else ''}\"><table><thead><tr>"]
    out += [f"<th>{h}</th>" for h in headers]
    out.append("</tr></thead><tbody>")
    for i, row in enumerate(rows):
        cls = f' class="{row_class[i]}"' if row_class and row_class[i] else ""
        out.append(f"<tr{cls}>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>")
    out.append("</tbody></table></div>")
    return "".join(out)


def _img(path: str | Path, caption: str) -> str:
    path = Path(path)
    if not path.exists():
        return ""
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return (f'<div class="fig"><img src="data:image/png;base64,{data}" alt="{path.name}">'
            f'<div class="cap">{caption}</div></div>')


def _txt_legend(entries: list[tuple[str, str]]) -> str:
    width = max(_wid(k) for k, _ in entries) + 2
    out = []
    for key, value in entries:
        out.append(f"  {_pad(key, width)}  {value}")
    return "\n".join(out)


def _wid(text: str) -> int:
    return sum(2 if ord(ch) > 0x2E80 else 1 for ch in str(text))


def _pad(text: str, width: int, right: bool = False) -> str:
    gap = max(0, width - _wid(text))
    return (" " * gap + str(text)) if right else (str(text) + " " * gap)


def _txt_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [max([_wid(headers[i])] + [_wid(r[i]) for r in rows]) for i in range(len(headers))]
    def line(cells):
        return "  ".join(_pad(c, widths[i], right=i > 0) for i, c in enumerate(cells))
    out = [line(headers), "  ".join("-" * w for w in widths)]
    out += [line(r) for r in rows]
    return "\n".join(out)


def run_report_body(result: dict, names: list[str], figures: list[str] | None = None):
    """HTML body for one robot; shared by the per-run page and the publishable single file."""
    fw, summary, health, root = result["framewise"], result["summary"], result["health"], result["root"]
    cards = [
        ("RMSE 中位(全段)", f"{fmt(summary['rmse_median_deg'])}<small> °</small>"),
        ("参考帧偏移", f"{fmt(summary['ref_offset_ms'], 0)}<small> ms</small>"),
        ("响应延迟", f"{fmt(summary['resp_lag_ms'], 0)}<small> ms</small>"),
        ("稳态 RMSE 中位", f"{fmt(summary['rmse_settled_median_deg'])}<small> °</small>"),
        ("RMSE 最大的关节", f"j{summary['rmse_worst_joint']} {names[summary['rmse_worst_joint']].replace('_joint', '')}"
                            f"<small> {fmt(summary['rmse_worst_deg'])}°</small>"),
        ("反馈率", f"{fmt(health.get('effective_hz'), 0)}<small> Hz</small>"),
        ("root 姿态", f"{fmt(root['roll_rms_deg'])}/{fmt(root['pitch_rms_deg'])}/{fmt(root['yaw_rms_deg'])}<small> °</small>"),
    ]
    body = ['<div class="cards">']
    body += [f'<div class="card"><div class="k">{k}</div><div class="v">{v}</div></div>' for k, v in cards]
    body.append("</div>")

    body.append("<h2>反馈链路</h2>")
    body.append(_table(["指标", "值"], [
        ["名义 / 有效 Hz", f"{fmt(health.get('nominal_hz'), 0)} / {fmt(health.get('effective_hz'), 0)}"],
        ["有效 / 名义", f"{fmt(health.get('rate_ratio') * 100, 1)} %"],
        ["样本数 / 窗口", f"{health.get('samples')} / {fmt(health.get('span_s'))} s"],
        ["dt p50 / p95 / p99 / max", f"{fmt(health.get('dt_p50_ms'))} / {fmt(health.get('dt_p95_ms'))} / {fmt(health.get('dt_p99_ms'))} / {fmt(health.get('dt_max_ms'))} ms"],
        ["&gt;10ms 缺口", f"{health.get('gaps_gt_10ms')} 次，共 {fmt(health.get('gap_time_ms'), 0)} ms"],
        ["窗口来源覆盖率 / observed source coverage", f"{fmt((summary.get('window_lineage_coverage') or 0) * 100, 1)} %"],
        ["非过渡帧匹配率 / non-transition pairing", f"{fmt(summary.get('index_coverage') * 100, 1)} %"],
        ["合成参考排除拍数 / excluded transition ticks", str(summary.get('excluded_transition_ticks', 0))],
        ["零位移 / 最佳位移 RMSE", f"{fmt(summary['rmse_median_deg'])} / {fmt(summary['rmse_bestshift_median_deg'])} °"],
        ["root_rpy 有效率", f"{fmt(root['valid_ratio'] * 100, 1)} %"],
        ["全段 / 稳态帧数", f"{summary['frames']} / {summary['settled_frames']}"],
    ]))

    body.append("<h2>逐关节</h2>")
    headers = ["#", "关节", "组", "RMSE全段°", "RMSE稳态°", "P95°", "max°", "bias°",
               "响应ms", "静止段°", "运动段°"]
    rows, txt_rows = [], []
    for j, name in enumerate(names):
        rows.append([j, name, group_of(name), fmt(fw["rmse_deg"][j]), fmt(fw["settled_rmse_deg"][j]),
                     fmt(fw["p95_deg"][j]), fmt(fw["maxabs_deg"][j]), fmt(fw["bias_deg"][j]),
                     fmt(fw["resp_lag_ms"][j], 0),
                     fmt(fw["static_rmse_deg"][j]), fmt(fw["moving_rmse_deg"][j])])
        txt_rows.append([str(c) for c in rows[-1]])
    body.append(_table(headers, rows))

    if figures:
        body.append("<h2>图</h2>")
        body.append(_img(figures[0], "逐关节 RMSE / bias / P95（红色为 RMSE 超过 31 关节中位 1.5 倍的关节）"))
        if len(figures) > 1:
            body.append(_img(figures[1], "关节 × 时间的平均绝对误差热图（颜色按 p99 截断）"))

    body.append("<h2>口径</h2><div class=\"note\">"
                "真机取 state 流（low_state 反馈，只取窗口内样本数最多的那个 writer），仿真取 derived_simulation.npz；"
                "对齐用离线轨迹帧号（publish.native_index ↔ encoder.window[0]），幅值指标在帧对齐（零位移）下计算。<br>"
                "参考帧偏移含策略固定的 10 帧前瞻（200 ms），属设计而非缺陷；响应延迟是帧对齐后残余的最优 tick 位移。<br>"
                "稳态 = 去掉前 2 s（两侧从不同初始姿态收敛的瞬态）。<br>"
                "root yaw 已 unwrap 并去除常量偏置，roll/pitch 按 ±180° 折返对齐。真机无 root 世界位置，不做绝对位移误差。"
                "</div>")
    return "".join(body), cards, headers, txt_rows


def write_run_report(result: dict, names: list[str], out_dir: Path, figures: list[str] | None = None) -> None:
    run_dir = out_dir / result["robot"]
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.json").write_text(json.dumps(jsonable(result), indent=2, ensure_ascii=False))
    body, cards, headers, txt_rows = run_report_body(result, names, figures)
    where = f"运行：{escape(result['run'])}"
    if "playback_cycle" in result:
        where += f" · 轨迹 {result['trajectory_index']} · 回放 {result['playback_cycle']}"
    elif int(result.get("execution") or 1) != 1:
        where += f" · 该日志第 {result['execution']} 次回放"
    html = _page(f"机器人一致性报告 {result['robot']}", where, body)
    (run_dir / "report.html").write_text(html)

    health, summary, root = result["health"], result["summary"], result["root"]
    where = f"运行：{result['run']}"
    if "playback_cycle" in result:
        where += f" · 轨迹 {result['trajectory_index']} · 回放 {result['playback_cycle']}"
    elif int(result.get("execution") or 1) != 1:
        where += f" · 该日志第 {result['execution']} 次回放"
    txt = [f"机器人一致性报告 {result['robot']}", where, "", "概要"]
    key_width = max(len(k) + 2 for k, _ in cards)
    txt += [f"  {_pad(k, key_width)}  {v.replace('<small>', '').replace('</small>', '')}" for k, v in cards]
    txt += ["", "反馈链路"]
    txt += [f"  名义/有效 Hz      {fmt(health.get('nominal_hz'), 0)} / {fmt(health.get('effective_hz'), 0)}  ({fmt(health.get('rate_ratio') * 100, 1)} %)",
            f"  dt p50/p95/p99    {fmt(health.get('dt_p50_ms'))} / {fmt(health.get('dt_p95_ms'))} / {fmt(health.get('dt_p99_ms'))} ms",
            f"  >10ms 缺口        {health.get('gaps_gt_10ms')} 次 / {fmt(health.get('gap_time_ms'), 0)} ms",
            f"  参考帧匹配率      {fmt(summary.get('index_coverage') * 100, 1)} %",
            f"  零位移/最佳位移   {fmt(summary['rmse_median_deg'])} / {fmt(summary['rmse_bestshift_median_deg'])} deg",
            f"  root_rpy 有效率   {fmt(root['valid_ratio'] * 100, 1)} %",
            "", "逐关节"]
    txt.append(_txt_table(headers, txt_rows))
    txt += ["", "口径：幅值指标均在帧对齐（零位移）下计算；稳态去掉前 2s；真机无 root 位置。"]
    (run_dir / "report.txt").write_text("\n".join(txt))

    for stale in ("report.zh-CN.md", "report.en.md"):
        old = run_dir / stale
        if old.exists():
            old.unlink()


def diagnose(results: list[dict], names: list[str]) -> dict:
    """Turn the 31 x N matrix into two plain-language findings: shared weak joints and single-robot outliers."""
    mat = np.vstack([res["framewise"]["rmse_deg"] for res in results]).T      # joints x robots
    med = np.nanmedian(mat, axis=1)
    dev = mat - med[:, None]
    robots = [res["robot"] for res in results]

    order = np.argsort(-med)
    common = [int(j) for j in order if med[j] >= COMMON_JOINT_MIN_DEG][:5] or [int(j) for j in order[:3]]
    outliers = []
    for i, robot in enumerate(robots):
        j = int(np.argmax(dev[:, i]))
        if dev[j, i] >= OUTLIER_DEV_DEG:
            outliers.append({"robot": robot, "joint": j, "name": names[j], "rmse": float(mat[j, i]),
                             "dev": float(dev[j, i]), "median": float(med[j])})
    outliers.sort(key=lambda row: -row["dev"])
    overall = sorted(({"robot": robot, "mean_dev": float(dev[:, i].mean())} for i, robot in enumerate(robots)),
                     key=lambda row: -row["mean_dev"])
    return {
        "common": [{"joint": int(j), "name": names[j], "median": float(med[j]),
                    "lo": float(np.nanmin(mat[j])), "hi": float(np.nanmax(mat[j]))} for j in common],
        "outliers": outliers[:6],
        "overall": [row for row in overall if abs(row["mean_dev"]) >= ROBOT_BIAS_DEG],
        "thresholds": {"common_deg": COMMON_JOINT_MIN_DEG, "outlier_deg": OUTLIER_DEV_DEG, "robot_bias_deg": ROBOT_BIAS_DEG},
    }


def aggregate_robot_rmse(results: list[dict]):
    """Average per-joint RMSE across executions sharing a recorded robot serial number."""
    grouped = {}
    for res in results:
        sn = str(res.get("sn") or "").strip()
        if not sn or sn.lower() == "unknown" or not sn.isdigit():
            continue
        grouped.setdefault(sn, []).append(np.asarray(res["framewise"]["rmse_deg"], dtype=float))
    if not any(len(values) > 1 for values in grouped.values()):
        return None
    labels, matrices, counts = [], [], []
    for sn, values in grouped.items():
        matrices.append(np.nanmean(np.stack(values), axis=0))
        counts.append(len(values))
        labels.append(f"T2-{sn} ({len(values)}次)")
    return labels, np.column_stack(matrices), counts


def _diag_txt(findings: dict | None) -> list[str]:
    if not findings:
        return []
    lines = ["自动诊断 — 共性偏高（行中位 >= %.1f deg）" % findings["thresholds"]["common_deg"],
             _txt_table(["#", "关节", "组", "行中位", "9台范围"],
                        [[f"j{c['joint']}", c["name"], group_of(c["name"]), fmt(c["median"]),
                          f"{fmt(c['lo'])}-{fmt(c['hi'])}"] for c in findings["common"]]), "",
             "自动诊断 — 单台离群（偏离行中位 >= %.1f deg）" % findings["thresholds"]["outlier_deg"],
             _txt_table(["机器人", "关节", "RMSE", "偏离", "行中位"],
                        [[o["robot"], f"j{o['joint']} {o['name']}", fmt(o["rmse"]), f"+{fmt(o['dev'])}", fmt(o["median"])]
                         for o in findings["outliers"]]) if findings["outliers"] else "  无", ""]
    return lines


def _append_diagnosis(body: list[str], findings: dict) -> None:
    """Auto findings (common-high joints / single-robot outliers); opt-in via --auto-diag."""
    cards2 = [("最共性关节", f"{findings['common'][0]['name'].replace('_joint', '')}<small> {fmt(findings['common'][0]['median'])}°</small>")
              if findings["common"] else ("最共性关节", "-"),
              ("最大单台离群", f"{findings['outliers'][0]['robot']} j{findings['outliers'][0]['joint']}<small> +{fmt(findings['outliers'][0]['dev'])}°</small>")
              if findings["outliers"] else ("最大单台离群", "无"),
              ("整体最高", f"{findings['overall'][0]['robot']}<small> {findings['overall'][0]['mean_dev']:+.2f}°</small>")
              if findings["overall"] else ("整体最高", "各台接近")]
    body.append("<h2>自动诊断</h2><div class=\"cards\">")
    body += [f'<div class="card"><div class="k">{k}</div><div class="v">{v}</div></div>' for k, v in cards2]
    body.append("</div>")

    rows_common = [[f"j{c['joint']}", c["name"], group_of(c["name"]), fmt(c["median"]),
                    f"{fmt(c['lo'])}–{fmt(c['hi'])}"] for c in findings["common"]]
    body.append("<p class=\"note\">共性偏高：这些关节在所有机器人的行中位都超过 "
                f"{fmt(findings['thresholds']['common_deg'], 1)}°，更像参考轨迹/模型/负载问题，不是单台硬件。</p>")
    body.append(_table(["关节号", "关节", "组", "行中位°", "9 台范围°"], rows_common, narrow=True))

    if findings["outliers"]:
        rows_out = [[o["robot"], f"j{o['joint']} {o['name']}", fmt(o["rmse"]), f"+{fmt(o['dev'])}", fmt(o["median"])]
                    for o in findings["outliers"]]
        body.append(f"<p class=\"note\">单台离群：该台在此关节比 9 台中位高 {fmt(findings['thresholds']['outlier_deg'], 1)}° 以上。</p>")
        body.append(_table(["机器人", "关节", "该台 RMSE°", "偏离中位°", "行中位°"], rows_out, narrow=True))
    else:
        body.append("<p class=\"note\">单台离群：没有机器人超过阈值。</p>")

    if findings["overall"]:
        body.append("<p class=\"note\">整体偏差（该台 31 关节偏离中位的均值）："
                    + "，".join(f"{row['robot']} {row['mean_dev']:+.2f}°" for row in findings["overall"])
                    + f"（阈值 ±{fmt(findings['thresholds']['robot_bias_deg'], 1)}°）。</p>")


def _heatmap_sections(figures: list[str] | None) -> list[str]:
    if not figures or len(figures) < 3:
        return []
    sections = ["<h2>关节 × 执行 RMSE 热图</h2>",
                _img(figures[2], "31 关节 × 每次执行：左=逐关节 RMSE，"
                             "中=跨执行每关节中位（共性），右=偏离中位（执行间离群）")]
    if len(figures) >= 4:
        sections.extend(["<h2>关节 × 机器人平均 RMSE 热图</h2>",
                         _img(figures[3], "每个 SN 一列；同 SN 多次测试的逐关节 RMSE 算术平均。"
                                      "颜色按每行归一，格内数字为平均 RMSE（度）。")])
    return sections


def write_comparison(results: list[dict], names: list[str], out_dir: Path, figures: list[str] | None = None,
                     auto_diag: bool = False, skipped: list | None = None) -> None:
    groups = ["left_leg", "right_leg", "waist", "left_arm", "right_arm", "head"]
    group_idx = {g: [j for j, n in enumerate(names) if group_of(n) == g] for g in groups}

    rows = []
    for res in results:
        fw, summary = res["framewise"], res["summary"]
        row = {
            "robot": res["robot"],
            "ref_offset_ms": round(summary["ref_offset_ms"], 1),
            "resp_lag_ms": round(summary["resp_lag_ms"], 1),
            "rmse_median_deg": round(summary["rmse_median_deg"], 2),
            "rmse_settled_median_deg": round(summary["rmse_settled_median_deg"], 2),
            "rmse_worst_deg": round(summary["rmse_worst_deg"], 2),
            "worst_joint": names[summary["rmse_worst_joint"]],
            "p95_median_deg": round(summary["p95_median_deg"], 2),
            "eff_hz": round(res["health"].get("effective_hz", float("nan")), 0),
            "nominal_hz": res["health"].get("nominal_hz"),
            "rate_ratio": round(res["health"].get("rate_ratio", float("nan")) * 100, 1),
            "gaps_gt_10ms": res["health"].get("gaps_gt_10ms"),
            "gap_time_ms": round(res["health"].get("gap_time_ms", float("nan")), 0),
            "roll_rms_deg": round(res["root"]["roll_rms_deg"], 2) if res["root"]["roll_rms_deg"] is not None else None,
            "pitch_rms_deg": round(res["root"]["pitch_rms_deg"], 2) if res["root"]["pitch_rms_deg"] is not None else None,
            "yaw_rms_deg": round(res["root"]["yaw_rms_deg"], 2) if res["root"]["yaw_rms_deg"] is not None else None,
            "window_lineage_coverage_pct": round((summary.get('window_lineage_coverage') or 0) * 100, 2),
            "nontransition_pair_coverage_pct": round(summary['index_coverage'] * 100, 2),
            "excluded_transition_ticks": summary.get('excluded_transition_ticks', 0),
        }
        for g, idxs in group_idx.items():
            row[f"{g}_rmse"] = round(float(np.median(fw["rmse_deg"][idxs])), 2) if idxs else None
        rows.append(row)

    with (out_dir / "comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    headers = ["机器人", "参考帧偏移 ms", "响应延迟 ms", "RMSE中位(全段)°", "RMSE中位(稳态)°",
               "RMSE最大的关节", "该关节RMSE°",
               "P95中位°",
               "名义Hz", "有效 Hz", "有效/名义%", "缺口次数", "缺口 ms", "基座 roll°", "基座 pitch°", "基座 yaw°",
               "左腿°", "右腿°", "腰°", "左臂°", "右臂°", "头°"]
    headers += ["窗口来源覆盖率%", "非过渡帧匹配率%", "合成参考排除拍数"]
    keys = ["robot", "ref_offset_ms", "resp_lag_ms", "rmse_median_deg", "rmse_settled_median_deg", "worst_joint", "rmse_worst_deg",
            "p95_median_deg", "nominal_hz", "eff_hz", "rate_ratio", "gaps_gt_10ms", "gap_time_ms",
            "roll_rms_deg", "pitch_rms_deg", "yaw_rms_deg", "left_leg_rmse", "right_leg_rmse", "waist_rmse",
            "left_arm_rmse", "right_arm_rmse", "head_rmse"]
    keys += ['window_lineage_coverage_pct', 'nontransition_pair_coverage_pct', 'excluded_transition_ticks']
    int_like = {"ref_offset_ms", "resp_lag_ms", "eff_hz", "nominal_hz", "gaps_gt_10ms", "gap_time_ms"}
    int_like.add('excluded_transition_ticks')

    resp_vals = [r["resp_lag_ms"] for r in rows]
    hz_vals = [r["eff_hz"] for r in rows]
    rmse_vals = [r["rmse_settled_median_deg"] for r in rows]

    # transposed layout: metrics as rows, robots as columns (reads far better than 21 columns)
    robots = [r["robot"] for r in rows]
    transposed_headers = ["指标"] + robots
    transposed, txt_rows = [], []
    for k, label in zip(keys[1:], headers[1:]):
        values = [r.get(k) for r in rows]
        cells, tcell = [f"<td>{label}</td>"], [label]
        for v, row in zip(values, rows):
            text = fmt(v, 0 if k in int_like else 2)
            cls = ""
            if k == "resp_lag_ms" and v >= max(resp_vals):
                cls = "bad"
            elif k == "eff_hz" and v <= min(hz_vals):
                cls = "warn"
            elif k == "rmse_settled_median_deg" and v >= max(rmse_vals):
                cls = "bad"
            cells.append(f'<td class="{cls}">{text}</td>' if cls else f"<td>{text}</td>")
            tcell.append(text)
        transposed.append("".join(cells))
        txt_rows.append(tcell)

    table = ("<div class=\"scroll\"><table><thead><tr>" + "".join(f"<th>{h}</th>" for h in transposed_headers)
             + "</tr></thead><tbody>" + "".join(f"<tr>{r}</tr>" for r in transposed) + "</tbody></table></div>")

    ref_vals = [r["ref_offset_ms"] for r in rows]
    serials = {str(res.get("sn")) for res in results
               if res.get("sn") and str(res.get("sn")).isdigit()}
    robot_count = len(serials)
    cards = [("可识别 SN / 有效执行", f"{robot_count} / {len(results)}"),
             ("参考帧偏移范围", f"{fmt(min(ref_vals), 0)}–{fmt(max(ref_vals), 0)}<small> ms</small>"),
             ("稳态 RMSE 范围", f"{fmt(min(rmse_vals))}–{fmt(max(rmse_vals))}<small> °</small>"),
             ("反馈率范围", f"{fmt(min(hz_vals), 0)}–{fmt(max(hz_vals), 0)}<small> Hz</small>")]
    body = ['<div class="cards">']
    body += [f'<div class="card"><div class="k">{k}</div><div class="v">{v}</div></div>' for k, v in cards]
    body.append("</div>")
    if skipped:
        detail = "；".join(f"{name.rsplit('_', 1)[-1]} 第 {loop} 段 {cov * 100:.0f}%" for name, loop, cov in skipped)
        body.append(f'<p class="note">只统计完整跑完参考轨迹的执行段（覆盖率 ≥ {COMPLETE_COVERAGE * 100:.0f}%）；'
                    f'已跳过 {len(skipped)} 段：{detail}。</p>')
    findings = diagnose(results, names) if auto_diag else None
    if auto_diag:
        _append_diagnosis(body, findings)
    body.extend(_heatmap_sections(figures))

    body.append("<h2>逐机器人</h2>")
    body.append(table)

    body.append("<h2 style=\"font-size:13px;color:#8b93a7;font-weight:600;margin:16px 0 8px\">指标说明（表中每一行的含义与单位）</h2>")
    body.append("<dl class=\"legend\">" + LEGEND_HTML + "</dl>")


    # per-joint matrices: every joint of the 31, one column per robot
    def joint_matrix(title: str, getter, worst: str = "max", digits: int = 2):
        head = ["#", "关节", "组"] + robots
        rows_html, txt = [], []
        for j, name in enumerate(names):
            values = [getter(res, j) for res in results]
            cells = [f"<td>{j}</td>", f"<td>{name}</td>", f"<td>{group_of(name)}</td>"]
            trow = [str(j), name, group_of(name)]
            finite = [v for v in values if v == v]
            target = (max(finite) if worst == "max" else min(finite)) if finite else None
            for v in values:
                text = fmt(v, digits)
                cls = ""
                if target is not None and v == target:
                    cls = "bad" if worst == "max" else "good"
                cells.append(f'<td class="{cls}">{text}</td>' if cls else f"<td>{text}</td>")
                trow.append(text)
            rows_html.append("".join(cells))
            txt.append(trow)
        html = ("<div class=\"scroll\"><table><thead><tr>" + "".join(f"<th>{h}</th>" for h in head)
                + "</tr></thead><tbody>" + "".join(f"<tr>{r}</tr>" for r in rows_html) + "</tbody></table></div>")
        return f"<h2>{title}</h2>" + html, txt

    rmse_html, rmse_txt = joint_matrix("逐关节 RMSE（°，每行 RMSE 最大的机器人标红）", lambda res, j: res["framewise"]["rmse_deg"][j])
    lag_html, lag_txt = joint_matrix("逐关节响应延迟（ms，每行延迟最大的机器人标红）",
                                     lambda res, j: res["framewise"]["resp_lag_ms"][j], digits=0)
    bias_html, bias_txt = joint_matrix("逐关节 |bias|（°，每行 |bias| 最大的机器人标红）", lambda res, j: abs(res["framewise"]["bias_deg"][j]))
    body.append(rmse_html)
    body.append(lag_html)
    body.append(bias_html)
    if figures:
        body.append("<h2>图</h2>")
        captions = ("跨机器人四个关键指标", "按身体分组的 RMSE 中位", "RMSE 热图（关节 × 执行段）",
                    "按 SN 聚合的逐关节 RMSE 算术平均热图（度）")
        for path, cap in zip(figures, captions):
            if path == figures[2] or (len(figures) >= 4 and path == figures[3]):
                continue
            body.append(_img(path, cap))
    multi = "playback_cycle" in results[0]
    reference_note = ("只比较同一轨迹选择下的执行；每次执行与同一回放周期的仿真帧配对。" if multi else
                      "全部机器人对比同一份仿真参考（derived_simulation.npz）——同一离线轨迹只重跑一次仿真，因此跨机器人差异即硬件差异。")
    body.append(f"<h2>口径</h2><div class=\"note\">{reference_note}<br>"
                "lag 正=真机落后于仿真；幅值类指标均在 lag 对齐后计算；稳态去掉前 2 s。<br>"
                "<b>基座 roll/pitch/yaw</b> 指骨盆（root/free joint）的整机姿态误差：真机取 IMU 的 root_rpy_w，"
                "仿真取 free joint 四元数换算成 rpy；它<b>不是</b>某个关节的量。T2 的 31 个关节都是单自由度铰链，"
                "每个关节只有一个角度，因此没有各自的 roll/pitch/yaw，只有下面三张逐关节矩阵里的角度误差、lag 和 bias。<br>"
                "红色 = 该项最差，橙色 = 反馈率最低，绿色 = 该项最好。<br><br>"
                "<b>怎么读 31×9 的 RMSE 热图</b>：左图颜色越亮 = 误差越大，某一列整体偏亮说明那台机器整体偏差；"
                "中图每行取 9 台的中位数，条越长说明<b>所有机器在这根关节上都偏高</b>（共性 —— 往往是参考轨迹、模型参数或该关节负载，不是某一台的硬件）；"
                "右图是“该台误差 − 同关节 9 台中位”，红 = 比同伴差、蓝 = 比同伴好，某一列出现成片的红就是<b>单台离群</b>。"
                "</div>")
    html = _page("机器人一致性对比", "同轨迹回放对比" if multi else "同一仿真参考下的硬件差异排序", "".join(body))
    (out_dir / "comparison.html").write_text(html)

    reference_label = "按回放周期配对仿真参考" if multi else "共用同一份仿真参考"
    txt = ["机器人一致性对比", f"可识别 SN：{robot_count}；有效执行：{len(results)}；{reference_label}", ""]
    if skipped:
        txt.append("只统计完整跑完参考轨迹的执行段（覆盖率 >= %.0f%%）；已跳过：%s" % (
            COMPLETE_COVERAGE * 100,
            "；".join(f"{n.rsplit('_', 1)[-1]} 第 {l} 段 {c * 100:.0f}%" for n, l, c in skipped)))
        txt.append("")
    txt += [
           _txt_table(transposed_headers, txt_rows), "",
           "指标说明",
           _txt_legend(LEGEND_ENTRIES), "",
           *_diag_txt(findings),
           "逐关节 RMSE (deg)", _txt_table(["#", "关节", "组"] + robots, rmse_txt), "",
           "逐关节响应延迟 (ms)", _txt_table(["#", "关节", "组"] + robots, lag_txt), "",
           "逐关节 |bias| (deg)", _txt_table(["#", "关节", "组"] + robots, bias_txt), "",
           "说明：lag 正=真机落后；RMSE 稳态去掉前 2s；基座 roll/pitch/yaw 是骨盆整机姿态，不是关节量。"]
    (out_dir / "comparison.txt").write_text("\n".join(txt))

    for stale in ("comparison.zh-CN.md", "comparison.en.md"):
        old = out_dir / stale
        if old.exists():
            old.unlink()


def _doc_body(html: str) -> str:
    """Take the inner content of a generated page so pages can be concatenated into one document."""
    start = html.index('<div class="wrap">') + len('<div class="wrap">')
    end = html.rindex('</div></body>')
    return html[start:end]


def write_publishable(out_dir: Path, results: list[dict], *, full: bool) -> Path:
    """One self-contained HTML (images inlined, no scripts, no external files); full adds every robot."""
    comparison = _doc_body((out_dir / "comparison.html").read_text())
    reference = "按回放周期配对仿真参考" if "playback_cycle" in results[0] else "共用同一份仿真参考"
    subtitle_tail = f"{reference} · 单文件，无外部依赖"
    robot_count = len({str(res["sn"]) for res in results
                       if res.get("sn") and str(res.get("sn")).isdigit()})
    cohort = f"{robot_count} 个可识别 SN，{len(results)} 次有效执行"
    if not full:
        html = _page("机器人一致性总览", f"{cohort} · {subtitle_tail}", comparison)
        path = out_dir / "report-summary.html"
        path.write_text(html)
        return path

    toc = ['<a href="#cmp">总览对比</a>']
    parts = ['<section id="cmp">' + comparison + "</section>"]
    for res in results:
        robot = res["robot"]
        toc.append(f'<a href="#r-{robot}">{robot}</a>')
        parts.append(f'<section id="r-{robot}">' + _doc_body((out_dir / robot / "report.html").read_text()) + "</section>")
    body = '<nav class="toc">' + "".join(toc) + "</nav>" + "".join(parts)
    html = _page("机器人一致性完整版", f"{cohort} · 总览 + 逐次执行明细 · {subtitle_tail}", body)
    path = out_dir / "report-full.html"
    path.write_text(html)
    return path


def plot_robot(result: dict, names: list[str], out_dir: Path) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patheffects as pe
    except Exception:
        return []
    out = out_dir / result["robot"]
    out.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.font_manager as fm
        if any("CJK" in f.name for f in fm.fontManager.ttflist):
            matplotlib.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
    except Exception:
        pass
    matplotlib.rcParams["axes.unicode_minus"] = False
    plt.style.use("dark_background")
    written = []

    fw = result["framewise"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 9), sharey=True)
    for ax, key, title in zip(axes, ("rmse_deg", "bias_deg", "p95_deg"),
                              ("RMSE (deg)", "bias (deg)", "P95 (deg)")):
        values = fw[key]
        colors = ["#ff6b6b" if v > np.nanmedian(values) * 1.5 else "#4ecdc4" for v in values]
        ax.barh(range(len(values)), values, color=colors)
        ax.set_title(title)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.25)
    axes[0].set_yticks(range(len(names)))
    axes[0].set_yticklabels([f"{j} {n}" for j, n in enumerate(names)], fontsize=7)
    fig.suptitle(f"{result['robot']} per-joint consistency")
    fig.tight_layout()
    path = out / "per_joint.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    written.append(str(path))

    heat = np.asarray(result["heatmap"], dtype=float)
    span = float(result["health"].get("replay_window_s", 58.2))
    vmax = float(np.nanpercentile(heat, 99)) if np.isfinite(heat).any() else 1.0
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(heat, aspect="auto", cmap="inferno", origin="upper",
                   vmin=0.0, vmax=max(vmax, 1e-6))
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([f"{j} {n}" for j, n in enumerate(names)], fontsize=7)
    ax.set_xticks(np.linspace(0, heat.shape[1] - 1, 7))
    ax.set_xticklabels([f"{span * v / (heat.shape[1] - 1):.0f}" for v in np.linspace(0, heat.shape[1] - 1, 7)])
    ax.set_xlabel("replay time (s)")
    for j, name in enumerate(names):
        if j and group_of(name) != group_of(names[j - 1]):
            ax.axhline(j - 0.5, color="#888888", lw=0.8)
    ax.set_title(f"{result['robot']} |joint error| over time (deg, colour clipped at p99)")
    fig.colorbar(im, ax=ax, label="deg", extend="max")
    fig.tight_layout()
    path = out / "error_heatmap.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    written.append(str(path))
    return written


def plot_comparison(results: list[dict], names: list[str], out_dir: Path) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patheffects as pe
    except Exception:
        return []
    try:
        import matplotlib.font_manager as fm
        if any("CJK" in f.name for f in fm.fontManager.ttflist):
            matplotlib.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
    except Exception:
        pass
    matplotlib.rcParams["axes.unicode_minus"] = False
    plt.style.use("dark_background")
    robots = [r["robot"] for r in results]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    series = [
        ("reference offset (ms)", lambda r: r["summary"]["ref_offset_ms"]),
        ("response delay (ms)", lambda r: r["summary"]["resp_lag_ms"]),
        ("median RMSE (deg)", lambda r: r["summary"]["rmse_median_deg"]),
        ("effective feedback Hz", lambda r: r["health"].get("effective_hz", float("nan"))),
    ]
    for ax, (title, getter) in zip(axes.ravel(), series):
        values = [getter(r) for r in results]
        ax.bar(robots, values, color="#4ecdc4")
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Cross-robot consistency (shared replay reference)")
    fig.tight_layout()
    path = out_dir / "comparison_summary.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    width = 0.2
    groups = ["left_leg", "right_leg", "waist", "left_arm", "right_arm", "head"]
    for i, res in enumerate(results):
        values = []
        for g in groups:
            idxs = [j for j, n in enumerate(names) if group_of(n) == g]
            values.append(float(np.nanmedian(res["framewise"]["rmse_deg"][idxs])) if idxs else float("nan"))
        ax.bar(np.arange(len(groups)) + (i - len(results) / 2) * width / 2, values, width / 2, label=res["robot"])
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups)
    ax.set_ylabel("median RMSE (deg)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=7, ncol=3)
    ax.set_title("Per-group median RMSE by robot")
    fig.tight_layout()
    path2 = out_dir / "comparison_groups.png"
    fig.savefig(path2, dpi=130)
    plt.close(fig)

    # 31 joints x robots: raw RMSE, per-joint median (common bias), deviation (single-robot outlier)
    mat = np.vstack([res["framewise"]["rmse_deg"] for res in results]).T
    row_med = np.nanmedian(mat, axis=1)
    dev = mat - row_med[:, None]
    lim = float(np.nanpercentile(np.abs(dev), 90)) or 1.0
    fig = plt.figure(figsize=(17, 10))
    grid = fig.add_gridspec(1, 3, width_ratios=[3.4, 1.0, 3.4], wspace=0.06)
    ax0 = fig.add_subplot(grid[0, 0])
    ax1 = fig.add_subplot(grid[0, 1], sharey=ax0)
    ax2 = fig.add_subplot(grid[0, 2], sharey=ax0)
    # left panel uses the same per-joint comparison logic as the right one: colour by rank inside the row,
    # blue = smallest RMSE among the robots for that joint, red = largest; the cell text keeps the real value.
    row_min = np.nanmin(mat, axis=1)
    row_max = np.nanmax(mat, axis=1)
    span = np.where((row_max - row_min) > 1e-6, row_max - row_min, 1.0)
    norm = np.clip((mat - row_min[:, None]) / span[:, None], 0.0, 1.0)
    # same palette as the right panel; label colour is chosen per cell so it stays readable on light mid-tones
    im0 = ax0.imshow(norm, aspect="auto", cmap="RdBu_r", origin="upper", vmin=0, vmax=1)
    im2 = ax2.imshow(dev, aspect="auto", cmap="RdBu_r", origin="upper", vmin=-lim, vmax=lim)
    short = [f"{j} {n.replace('_joint', '')}" for j, n in enumerate(names)]
    ax0.set_yticks(range(len(names)))
    ax0.set_yticklabels(short, fontsize=7)
    ax0.set_xticks(range(len(results)))
    ax0.set_xticklabels(robots, rotation=45, ha="right", fontsize=8)
    ax2.set_xticks(range(len(results)))
    ax2.set_xticklabels(robots, rotation=45, ha="right", fontsize=8)
    ax0.set_title("RMSE per joint x robot (deg)\ncolour normalised per row: blue = smallest, red = largest", fontsize=9)
    ax2.set_title("deviation from that joint's median (deg)\nred = worse than the other robots", fontsize=9)
    ax1.barh(range(len(names)), row_med, color="#4ecdc4")
    ax1.set_title("per-joint median (deg)")
    ax1.grid(axis="x", alpha=0.25)
    ax1.tick_params(labelleft=False)
    for ax in (ax0, ax2):
        for j, name in enumerate(names):
            if j and group_of(name) != group_of(names[j - 1]):
                ax.axhline(j - 0.5, color="#ff9f43", lw=1.1, alpha=0.85)
                ax1.axhline(j - 0.5, color="#ff9f43", lw=1.1, alpha=0.85)
    for j in range(mat.shape[0]):
        for i in range(mat.shape[1]):
            v = mat[j, i]
            if not np.isfinite(v):
                continue
            t = float(norm[j, i])
            r, g, b, _ = im0.cmap(t)
            light_cell = (0.2126 * r + 0.7152 * g + 0.0722 * b) > 0.55
            ax0.text(i, j, f"{v:.1f}", ha="center", va="center", fontsize=6.2,
                     color="#10161f" if light_cell else "#f7fbff",
                     path_effects=[pe.withStroke(linewidth=0.8,
                                                 foreground="#f7fbff" if light_cell else "#10161f")])
    fig.colorbar(im2, ax=[ax2], fraction=0.025, pad=0.01, label="deg")
    path = out_dir / "comparison_joint_heatmap.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    written = [str(p) for p in (out_dir / "comparison_summary.png", out_dir / "comparison_groups.png", path)]

    aggregate = aggregate_robot_rmse(results)
    if aggregate:
        labels, mean_mat, counts = aggregate
        fig, ax = plt.subplots(figsize=(max(10, 1.15 * len(labels)), 10))
        row_min = np.nanmin(mean_mat, axis=1)
        row_max = np.nanmax(mean_mat, axis=1)
        span = np.where((row_max - row_min) > 1e-6, row_max - row_min, 1.0)
        norm = np.clip((mean_mat - row_min[:, None]) / span[:, None], 0.0, 1.0)
        image = ax.imshow(norm, aspect="auto", cmap="RdBu_r", origin="upper", vmin=0, vmax=1)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(short, fontsize=8)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
        ax.set_title("Mean per-joint RMSE across tests for the same robot (deg)\n"
                     "colour normalised per joint: blue = smallest, red = largest")
        for j, name in enumerate(names):
            if j and group_of(name) != group_of(names[j - 1]):
                ax.axhline(j - 0.5, color="#ff9f43", lw=1.1, alpha=0.85)
        for j in range(mean_mat.shape[0]):
            for i in range(mean_mat.shape[1]):
                value = mean_mat[j, i]
                if not np.isfinite(value):
                    continue
                red, green, blue, _ = image.cmap(float(norm[j, i]))
                light = 0.2126 * red + 0.7152 * green + 0.0722 * blue > 0.55
                ax.text(i, j, f"{value:.1f}", ha="center", va="center", fontsize=7,
                        color="#10161f" if light else "#f7fbff",
                        path_effects=[pe.withStroke(linewidth=0.8,
                                                    foreground="#f7fbff" if light else "#10161f")])
        fig.colorbar(image, ax=ax, fraction=0.025, pad=0.01, label="row-normalised RMSE")
        fig.tight_layout()
        aggregate_path = out_dir / "comparison_joint_heatmap_by_robot.png"
        fig.savefig(aggregate_path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        written.append(str(aggregate_path))
    return written


def load_run_config(run):
    path = run / "execution_config.json"
    return json.loads(path.read_text()) if path.is_file() else {}


def write_multi_trajectory_report(run, sim_npz, sim_indices, names, out_dir, config, args):
    events = [row for path in sorted(run.glob("events.*.msgpack")) for row in reader.iter_file(path)]
    playbacks = committed_playbacks(config, events)
    with np.load(sim_npz, allow_pickle=False) as archive:
        if not {"playback_cycle", "trajectory_index", "native_index"} <= set(archive.files):
            raise ValueError("regenerate derived_simulation.npz: unlimited playback provenance is missing")
        sim_cycles, sim_trajectories = archive["playback_cycle"], archive["trajectory_index"]
    if sim_cycles.shape != sim_indices.shape or sim_trajectories.shape != sim_indices.shape:
        raise ValueError("simulation playback provenance length mismatch")
    playback_by_cycle = {play.cycle: play for play in playbacks}
    for cycle in np.unique(sim_cycles):
        play = playback_by_cycle.get(cycle)
        if play is None or np.any(sim_trajectories[sim_cycles == cycle] != play.trajectory_index):
            raise ValueError(f"simulation playback provenance disagrees with recorded cycle {cycle}")

    robot_id = str(args.sn or config.get("sn") or "unknown")
    if robot_id != "unknown" and not re.fullmatch(r"[0-9]+", robot_id):
        raise ValueError("robot id must contain digits only or be unknown")
    executions = read_executions(run, playbacks)
    index = [dict(trajectory_index=n, **{k: item[k] for k in ("path", "name", "stem")}, playbacks=[])
             for n, item in enumerate(config["trajectory_history"])]
    jobs_by_index, skipped_by_index = {}, {}
    for execution in executions:
        cycle = execution["playback_cycle"]
        play = playback_by_cycle[cycle]
        n = play.trajectory_index
        record = {"cycle": cycle}
        index[n]["playbacks"].append(record)
        # No lineage means incomplete regardless of the declared frame extent.
        if not execution["consumed_clips"]:
            record.update(coverage=0.0, status="incomplete")
        else:
            reference = execution_reference_indices(run, sim_indices, play)
            complete, rejected = select_complete_executions([execution], reference)
            record["coverage"] = (complete or rejected)[0]["coverage"]
            if not complete:
                record["status"] = "incomplete"
            else:
                first = {}
                allowed = set(reference.tolist())
                for frame in np.flatnonzero(sim_cycles == cycle):
                    clip = int(sim_indices[frame])
                    if clip in allowed:
                        first.setdefault(clip, int(frame))
                if sum(clip in execution["first"] for clip in first) < MINIMUM_PAIRS:
                    record["status"] = "insufficient_simulation_pairs"
                else:
                    group = jobs_by_index.setdefault(n, [])
                    label = segment_label(run, len(group), robot_id)
                    record.update(status="reported", report=f"trajectory-{n:03d}/{label}/report.html")
                    group.append({"run": str(run), "label": label, "sn": robot_id, "execution": complete[0],
                                  "sim_npz": str(sim_npz), "sim_indices": sim_indices,
                                  "sim_frames": list(first.values())})
        if record["status"] != "reported":
            skipped_by_index.setdefault(n, []).append((run.name, cycle, record["coverage"]))
            print(f"skip playback cycle {cycle} (trajectory {n}): {record['status']}", file=sys.stderr)

    for n, jobs in jobs_by_index.items():
        directory = out_dir / f"trajectory-{n:03d}"
        render_reports(jobs, skipped_by_index.get(n, []), names, directory, args)
        index[n]["report"] = f"trajectory-{n:03d}/comparison.html"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "trajectory_index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n")
    statuses = {"incomplete": "覆盖率不足", "insufficient_simulation_pairs": "仿真配对帧不足", "reported": "已生成报告"}
    rows = []
    for item in index:
        trajectory = escape(item["name"])
        if item.get("report"):
            trajectory = f'<a href="{item["report"]}">{trajectory}</a>'
        details = []
        for p in item["playbacks"]:
            label = f'{p["cycle"]}：{statuses[p["status"]]}（{p["coverage"]:.0%}）'
            details.append(f'<a href="{p["report"]}">{label}</a>' if p.get("report") else label)
        rows.append([item["trajectory_index"], trajectory, "<br>".join(details) or "未开始回放"])
    (out_dir / "index.html").write_text(_page("多轨迹一致性报告", escape(run.name),
        _table(["选择序号（从 0 开始）", "轨迹", "回放及窗口来源覆盖率"], rows)))
    print(f"trajectory report index -> {out_dir / 'index.html'}")
    return 0 if jobs_by_index else 4


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Compare real teleop feedback with a shared MuJoCo replay reference.")
    parser.add_argument("--run", action="append", default=[], help="single run directory (repeatable)")
    parser.add_argument("--all", type=Path, help="folder holding several run directories")
    parser.add_argument("--sim-run", type=Path, help="run directory that owns derived_simulation.npz (default: auto)")
    parser.add_argument("--out", type=Path, help="report output directory (default: per-run *_Report under the run)")
    parser.add_argument("--robot-id", dest="sn", type=str, help="numeric robot serial number for report label")
    parser.add_argument("--trajectory-name", help="offline trajectory stem for the report label")
    parser.add_argument("--report-timestamp", help="run start timestamp for the report label")
    parser.add_argument("--mjcf", type=Path, default=DEFAULT_MJCF, help="MJCF used for joint names")
    parser.add_argument("--lag-max-s", type=float, default=LAG_SEARCH_S)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--auto-diag", action="store_true",
                        help="also render the automatic findings section (common-high joints / single-robot outliers)")
    parser.add_argument("--no-report", action="store_true",
                        help="skip the self-contained HTML reports (report-summary.html / report-full.html); written by default")
    args = parser.parse_args(argv)

    runs = [Path(p).expanduser().resolve() for p in args.run]
    folder = None
    if args.all:
        folder = args.all.expanduser().resolve()
        runs.extend(collect_runs(folder))
    if not runs:
        parser.error("give --run DIR and/or --all FOLDER")
    configs = {run: load_run_config(run) for run in runs}
    multi = any(is_unlimited(config) for config in configs.values())
    if multi and (len(runs) != 1 or folder is not None):
        parser.error("unlimited trajectory runs require individual --run reports")

    if args.sim_run:
        sim_run = args.sim_run.expanduser().resolve()
    else:
        with_sim = [r for r in runs if (r / "derived_simulation.npz").exists()]
        if not with_sim:
            parser.error("no run has derived_simulation.npz; pass --sim-run pointing at the replayed reference")
        sim_run = with_sim[0]
    if multi and sim_run != runs[0]:
        parser.error("an unlimited trajectory run needs its own derived simulation")
    sim_npz = sim_run / "derived_simulation.npz"
    if not sim_npz.exists():
        parser.error(f"{sim_npz} missing")
    sim_indices, sim_stamps = sim_frame_indices(sim_run, sim_npz)
    with np.load(sim_npz, allow_pickle=False) as archive:
        sim_segment_ids = archive["segment_id"] if "segment_id" in archive.files else None
    if not len(sim_indices):
        parser.error(f"{sim_npz} has no simulated frames")
    print(f"sim reference frames: {len(sim_indices)}  index {sim_indices.min()}..{sim_indices.max()}")

    out_dir = args.out.expanduser().resolve() if args.out else None
    if out_dir is None and len(runs) == 1 and folder is None:
        run = runs[0]
        execution_config = configs[run]
        robot_id = args.sn or execution_config.get("sn") or "unknown"
        if robot_id != "unknown" and not str(robot_id).isdigit():
            parser.error("single-run robot ID must contain digits only")
        trajectory = REPORT_STEM if multi else (args.trajectory_name or (execution_config.get("trajectory") or {}).get("stem"))
        started_at = args.report_timestamp or execution_config.get("started_at")
        if not trajectory or not started_at:
            parser.error("single-run report requires trajectory name and start timestamp in execution_config.json")
        report_label = f"T2-{robot_id}-{trajectory}-{started_at}_Report"
        out_dir = run / report_label
    if out_dir is None:
        base = folder if folder else runs[0].parent
        out_dir = base.parent / f"{base.name}_report"
    names = joint_names_from_mjcf(args.mjcf)
    if len(names) != 31:
        print(f"warning: {args.mjcf} yields {len(names)} actuated joints, expected 31", file=sys.stderr)
    names = (names + [f"joint_{i}" for i in range(len(names), 31)])[:31]
    if multi:
        return write_multi_trajectory_report(runs[0], sim_npz, sim_indices, names, out_dir, configs[runs[0]], args)

    jobs, skipped = [], []
    for run in runs:
        reference_indices = execution_reference_indices(run, sim_indices)
        complete, rejected = select_complete_executions(read_executions(run), reference_indices)
        skipped.extend((run.name, item["loop"], item["coverage"]) for item in rejected)
        if not complete:
            print(f"warning: {run.name}: no execution covers enough of the reference "
                  f"(best {max([item['coverage'] for item in rejected] or [0]) * 100:.0f}%), skipped",
                  file=sys.stderr)
            continue
        run_config = configs[run]
        base_robot_id = args.sn or run_config.get("sn")
        if base_robot_id and str(base_robot_id) != "unknown" and not str(base_robot_id).isdigit():
            parser.error("robot id must contain digits only or be unknown")
        labels = [segment_label(run, nth, str(base_robot_id) if base_robot_id else "unknown")
                  for nth in range(len(complete))]
        for execution, label in zip(complete, labels):
            frames = select_simulation_frames(
                sim_indices, sim_stamps, execution if run == sim_run else None, sim_segment_ids)
            jobs.append({"run": str(run), "label": label, "execution": execution,
                         "sn": str(args.sn or run_config.get("sn")) if (args.sn or run_config.get("sn")) else None,
                         "sim_npz": str(sim_npz), "sim_indices": sim_indices.tolist(),
                         "sim_frames": frames.tolist()})
    for name, loop, coverage in skipped:
        print(f"skip incomplete execution: {name} loop {loop} (coverage {coverage * 100:.0f}%)", file=sys.stderr)

    if not jobs:
        print(f"没有满足完整性条件的执行 / No complete executions: "
              f"required observed window lineage coverage >= {COMPLETE_COVERAGE:.0%}. "
              "No RMSE report generated; analysis and plotting skipped.", file=sys.stderr)
        return 4
    render_reports(jobs, skipped, names, out_dir, args)
    return 0


def render_reports(jobs, skipped, names, out_dir, args):
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    if args.jobs > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            for res in pool.map(analyse_run, jobs):
                results.append(res)
    else:
        for job in jobs:
            results.append(analyse_run(job))

    for res in results:
        figures = plot_robot(res, names, out_dir) if not args.no_plots else []
        write_run_report(res, names, out_dir, figures)
        summary = res["summary"]
        print(f"{res['robot']:>6s}  refOff {summary['ref_offset_ms']:4.0f} ms  respLag {summary['resp_lag_ms']:4.0f} ms  "
              f"RMSE {summary['rmse_median_deg']:5.2f} deg  "
              f"worst j{summary['rmse_worst_joint']:<2d} {summary['rmse_worst_deg']:5.2f} deg  "
              f"hz {res['health'].get('effective_hz', 0):4.0f}")
    figures = plot_comparison(results, names, out_dir) if not args.no_plots else []
    write_comparison(results, names, out_dir, figures, auto_diag=args.auto_diag, skipped=skipped)
    if not args.no_report:
        print(f"publishable report -> {write_publishable(out_dir, results, full=False)}")
        print(f"publishable report -> {write_publishable(out_dir, results, full=True)}")
        stale = out_dir / "report.html"
        if stale.exists():
            stale.unlink()
    print(f"report -> {out_dir}")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
