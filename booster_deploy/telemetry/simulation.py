"""Regenerate real-run policy execution in MuJoCo from recorded human inputs."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

from booster_deploy.utils.project_path import resolve_project_path

from .reader import iter_file
from .run_config import model_sha256, provenance_path, read_execution_config
from .trajectory import committed_playbacks, is_unlimited, playback_at


ACTIVE_STATES = frozenset(("arming_tracking", "tracking", "tracking_paused", "exiting_to_walk"))
CTRL_EVIDENCE_SOURCE = "ctrl_state_transitions"
FALLBACK_SOURCE = "whole_clip_fallback"


def _records(directory: Path, stages: set[str]):
    rows = []
    errors = []
    for path in sorted(directory.glob("*.msgpack")):
        for row in iter_file(path, errors):
            if row.get("stage") in stages and row.get("t_mono_ns") is not None:
                rows.append(row)
    if errors:
        raise ValueError("cannot generate simulation from damaged telemetry: " + "; ".join(errors[:3]))
    rows.sort(key=lambda row: int(row["t_mono_ns"]))
    return rows


def _intervals(rows, origin: int, input_times: list[int]):
    """Return ``(intervals, provenance)`` for A->Walking simulation windows.

    ``provenance`` says whether the windows are backed by recorded ``ctrl``
    state transitions or were guessed from the whole human-input span, so a
    derived artifact can be labelled instead of passing an inferred window off
    as a real teleoperation interval.
    """
    intervals = []
    start = None
    transitions = 0
    active_seen = False
    degenerate = 0
    for row in rows:
        if row.get("stage") != "events":
            continue
        data = row.get("data", {})
        if data.get("event") != "state_transition" or data.get("component") != "ctrl":
            continue
        state = data.get("changes", {}).get("control_state")
        stamp = int(row["t_mono_ns"])
        transitions += 1
        if state in ACTIVE_STATES:
            if start is None:
                active_seen = True
                start = stamp
        elif state == "walking" and start is not None:
            if stamp > start:
                intervals.append((start, stamp))
            else:
                degenerate += 1
            start = None
    if start is not None and input_times:
        intervals.append((start, max(input_times[-1], start)))
    observed = [(max(a, origin), max(b, a)) for a, b in intervals]
    if observed:
        return observed, {"inferred": False, "source": CTRL_EVIDENCE_SOURCE, "reason": None}
    if not input_times:
        return [], {"inferred": False, "source": CTRL_EVIDENCE_SOURCE, "reason": None}
    if transitions == 0:
        reason = "recording has no ctrl state_transition events"
    elif not active_seen:
        reason = "no arming_tracking/tracking ctrl transition"
    elif degenerate:
        reason = "ctrl reported arming and walking on the same timestamp (zero-length active window)"
    else:
        reason = "no complete A->Walking ctrl transition pair"
    return [(input_times[0], input_times[-1])], {"inferred": True, "source": FALLBACK_SOURCE, "reason": reason}


def _task_config():
    from tasks.gmc import T2GmcRtControllerCfg

    cfg = T2GmcRtControllerCfg()
    cfg.mujoco.log_states = None
    return cfg


def _run_entry(controller):
    for tick in range(int(45. / controller.cfg.policy_dt)):
        controller.update_state()
        targets = controller.policy_step()
        prepared=getattr(controller.policy,'entry_prepared',False)
        if prepared:
            controller.ctrl_step(targets)
        if getattr(controller.policy, 'entry_complete', False):
            controller.portal.offline_playback_active.set()
            return tick+1
    raise RuntimeError('simulation entry transition timed out')


def _prepare_controller(controller, startup):
    import threading
    from types import SimpleNamespace
    controller.portal=SimpleNamespace(offline_entry_cycle=int(startup['cycle']),
        reference_tracking_active=threading.Event(),offline_playback_active=threading.Event())
    controller.portal.reference_tracking_active.set()
    state=startup.get('state',{})
    if state.get('joint_pos') is not None:
        controller.mj_data.qpos[controller._body_joint_qposadr]=state['joint_pos']
    if state.get('quat') is not None:
        controller.mj_data.qpos[3:7]=state['quat']
    import mujoco
    mujoco.mj_forward(controller.mj_model,controller.mj_data)


def _native_lookup(rows):
    """publish source_id -> native clip index: the only authoritative frame identity in a log."""
    table = {}
    for row in rows:
        data = row.get("data", {})
        index = data.get("native_index")
        key = data.get("source_id")
        if index is None or not isinstance(key, (list, tuple)) or len(key) != 2:
            continue
        try:
            table[(str(key[0]), int(key[1]))] = int(index)
        except (TypeError, ValueError):
            continue
    return table


def _frame_native(frame, native_of):
    """Oldest native clip index behind one reference frame's lineage (None when unrecorded)."""
    found = []
    for value in getattr(frame, "source_ids", ()) or ():
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            continue
        try:
            key = (str(value[0]), int(value[1]))
        except (TypeError, ValueError):
            continue
        if key in native_of:
            found.append(native_of[key])
    return min(found) if found else None


class _OfflineReferenceReceiver:
    """The production consumer sees the same growing 50 Hz window without ZMQ time."""
    def __init__(self, frames, target_lag):
        self.frames = {int(frame.sample_idx): frame for frame in frames}
        self.indices = sorted(self.frames)
        if not self.indices:
            raise ValueError("retargeting produced no 50 Hz references")
        self.position = min(int(target_lag), len(self.indices) - 1)
        self.first_sample_idx = self.indices[0]

    def start(self):
        return None

    def close(self):
        return None

    def latest_ready_idx(self):
        return self.indices[self.position]

    def get_window(self, indices):
        values = [self.frames.get(int(index)) for index in indices]
        return None if any(value is None for value in values) else values

    def wait_until_ready(self, min_samples, timeout_s, *, is_ready=None):
        if len(self.frames) < int(min_samples) or (is_ready is not None and not is_ready()):
            raise ValueError(f"retarget segment has {len(self.frames)} references; needs {min_samples}")

    def advance(self):
        if self.position + 1 < len(self.indices):
            self.position += 1
            return True
        return False


def _retarget_segment(cfg, segment, origin: int, native_of=None, *, startup=None):
    from booster_deploy.retargeting.rt_config import default_realtime_config_from_policy
    from booster_deploy.retargeting.rt_worker import RealtimeRetargetWorker
    from booster_deploy.types.pico_types import PicoRetargetInputFrameNs
    from booster_deploy.controllers.mujoco_controller import MujocoController

    worker = RealtimeRetargetWorker(default_realtime_config_from_policy(cfg.policy, robot_name="t2"))
    references = []
    try:
        for index, item in enumerate(segment):
            stamp, data = item[0], item[1]
            source_ids = tuple(tuple(value) for value in data.get("source_ids") or ())
            frame = PicoRetargetInputFrameNs(
                body_seq=int(data.get("native_seq", index)), body_recv_ns=int(stamp),
                offline_sample_ns=data.get("offline_sample_ns"), offline_cycle=int(data.get("offline_cycle") or 0),
                convert_done_ns=int(stamp), human_format=str(data.get("human_format") or "xrobot"),
                human_frame=data["human_frame"], source_ids=source_ids,loop=data.get("loop"),stream_id=data.get("stream_id"),
                clock_domain=data.get("clock_domain"),
            )
            references.extend(worker.process_one(frame))
    finally:
        worker.close()
    target_lag = int(cfg.policy.rt_reference_consumer.target_lag_frames)
    source = _OfflineReferenceReceiver(references, target_lag)
    controller = MujocoController(cfg)
    controller.policy._reference_receiver = source
    qpos = []
    seconds = []
    natives = []
    try:
        if startup:
            _prepare_controller(controller,startup)
        controller.update_state()
        controller.start()
        if startup:
            _run_entry(controller)
        while True:
            controller.update_state()
            targets = controller.policy_step()
            controller.ctrl_step(targets)
            base = controller.policy._last_consumed_base_idx
            if base is not None and int(base) in source.frames:
                frame = source.frames[int(base)]
                seconds.append((int(frame.sample_ns) - origin) / 1e9)
                qpos.append(np.asarray(controller.mj_data.qpos, dtype=np.float32).copy())
                native = _frame_native(frame, native_of or {})
                natives.append(int(native) if native is not None else -1)
            if not source.advance():
                break
    finally:
        controller.stop()
    if not qpos:
        raise RuntimeError("retarget policy simulation produced no frames")
    return np.asarray(seconds, dtype=np.float64), np.stack(qpos), np.asarray(natives, dtype=np.int64)


def generate(directory, *, output=None, model=None, force=False):
    directory = Path(directory).expanduser().resolve()
    manifest = json.loads((directory / "manifest.json").read_text())
    metadata = manifest.get("metadata", {})
    exec_config = read_execution_config(directory)
    execution = exec_config.get("execution") or metadata.get("execution")
    if execution == "sim":
        raise ValueError("simulation logs already contain closed-loop MuJoCo state")
    if execution not in (None, "real"):
        raise ValueError("simulation generation requires a real-run log")
    topology = exec_config.get("topology") or manifest.get("topology") or metadata.get("topology")
    if topology not in (None, "unknown", "retarget"):
        raise ValueError(f"unsupported telemetry topology {topology!r}; only retarget runs can be simulated")
    task = exec_config.get("task") or metadata.get("task") or "t2_gmc_rt"
    if task != "t2_gmc_rt":
        raise ValueError(f"unsupported recorded task {task!r}; expected t2_gmc_rt")
    model_config = exec_config.get("model") or {}
    if model_config.get("action_offset_source") not in (None, "command"):
        raise ValueError("recorded policy does not use the M0 command-offset action protocol")
    recorded_model = provenance_path(directory, model_config.get("resolved_path") or metadata.get("onnx_model"),
                                     model_config.get("repo_relative_path"))
    resolved_model = Path(model).expanduser().resolve() if model is not None else recorded_model
    if resolved_model is None:
        raise ValueError("log has no resolved ONNX provenance; pass --model for a copy of the bundled M0 model")
    if not resolved_model.is_file():
        raise FileNotFoundError(f"M0 model is missing: {resolved_model}")
    bundled = resolve_project_path(_task_config().policy.onnx_model_path)
    digest = model_sha256(bundled)
    if model_config.get("sha256") not in (None, digest):
        raise ValueError("recorded ONNX digest does not match the bundled T2 M0-1010 policy")
    for path in {resolved_model, recorded_model} - {None, bundled}:
        if path.is_file() and model_sha256(path) != digest:
            raise ValueError(f"ONNX does not match the bundled T2 M0-1010 policy: {path}")
    model_source = "offline_override" if model is not None else (
        model_config.get("source") or metadata.get("model_source") or "legacy")
    output = Path(output or directory / "derived_simulation.npz").expanduser().resolve()
    origin = int(manifest["origin_mono_ns"])
    unlimited = is_unlimited(exec_config)
    if output.exists() and not force and model is None:
        with np.load(output, allow_pickle=False) as archive:
            cached = json.loads(archive['metadata'].item()) if 'metadata' in archive.files else {}
            if (cached.get('origin_mono_ns') == origin and cached.get('clock_domain') == manifest.get('clock_domain')
                    and cached.get('model_sha256') == digest and cached.get('resolved_model') == str(resolved_model)
                    and cached.get('model_source') == model_source
                    and (not unlimited or {'trajectory_index', 'playback_cycle'} <= set(archive.files))):
                return output

    rows = _records(directory, {"publish", "retarget", "events"})
    events = [row for row in rows if row["stage"] == "events"]
    playbacks = committed_playbacks(exec_config, events)
    if playbacks == []:
        raise ValueError(f"{directory}: no committed offline playback events")
    if playbacks is not None:
        starts = [play.start_ns for play in playbacks]
        samples_by_cycle = {play.cycle: [] for play in playbacks}
        publish_cycles = {}
        for row in rows:
            if row["stage"] != "publish":
                continue
            data = row["data"]
            key = data.get("source_id")
            play = playback_at(playbacks, starts, int(row["t_mono_ns"]))
            if key and play is not None and data.get("native_index") is not None:
                publish_cycles[tuple(key)] = play.cycle
        for row in rows:
            data = row["data"]
            stamp = int(row["t_mono_ns"])
            if row["stage"] != "retarget" or data.get("human_frame") is None:
                continue
            play = playback_at(playbacks, starts, stamp)
            if play is None or data.get("offline_cycle") not in (None, 0, play.cycle):
                continue
            sources = data.get("source_ids") or ()
            if not sources or {publish_cycles.get(tuple(key)) for key in sources} != {play.cycle}:
                continue
            samples_by_cycle[play.cycle].append((stamp, data))
        segments = []
        for play in playbacks:
            segment = samples_by_cycle[play.cycle]
            if len(segment) < 2:
                print(f"warning: playback cycle {play.cycle} has fewer than two input samples; skipped", file=sys.stderr)
                continue
            segments.append((None, segment, play))
        interval_provenance = {"inferred": False, "source": "offline_playback_started"}
    else:
        segments, interval_provenance = _legacy_segments(directory, rows, origin)
    if not segments:
        raise ValueError("no playback interval contains enough human input samples")

    return _simulate_segments(directory, output, exec_config, manifest, metadata, resolved_model,
                              model_source, digest, origin, rows, segments, interval_provenance)


def _legacy_segments(directory, rows, origin):
    samples = [(int(row["t_mono_ns"]), row["data"]) for row in rows
               if row.get("stage") == "retarget" and row.get("data", {}).get("human_frame") is not None]
    if not samples:
        raise ValueError("log does not contain retarget.human_frame; enable that telemetry field before recording")
    intervals, interval_provenance = _intervals(rows, origin, [sample[0] for sample in samples])
    if interval_provenance["inferred"]:
        print(
            f"warning: {directory}: simulation intervals are INFERRED, not observed -- "
            f"interval_source={interval_provenance['source']} (intervals_inferred=true); "
            f"reason: {interval_provenance['reason']}. The derived npz falls back to the whole "
            "human-input clip, so its A->Walking window is NOT ctrl-confirmed teleoperation.",
            file=sys.stderr, flush=True,
        )
    segments = []
    for bounds in intervals:
        segment = [sample for sample in samples if bounds[0] <= sample[0] <= bounds[1]]
        if len(segment) >= 2:
            segments.append((bounds, segment, None))
    return segments, interval_provenance


def _simulate_segments(directory, output, exec_config, manifest, metadata, resolved_model,
                       model_source, digest, origin, rows, segments, interval_provenance):

    all_seconds = []
    all_qpos = []
    all_natives = []
    all_segments = []
    all_playback_cycles = []
    all_trajectory_indices = []
    segment_playbacks = []
    native_of = _native_lookup(rows)
    startup_events=[row['data'] for row in rows if row.get('stage')=='events' and row.get('data',{}).get('event')=='offline_startup_snapshot']
    playback_events=[row for row in rows if row.get('stage')=='events' and row.get('data',{}).get('event')=='offline_playback_started']
    startup_status=[]
    old_env = {key: os.environ.get(key) for key in ("BOOSTER_TELEMETRY_DIR", "BOOSTER_TELEMETRY_CONFIG", "BOOSTER_RUN_ID", "BOOSTER_DIAG_DIR")}
    with tempfile.TemporaryDirectory(prefix="booster-telemetry-sim-") as temp:
        work = Path(temp)
        config = work / "telemetry-off.yaml"
        config.write_text("enabled: false\n")
        os.environ.update(BOOSTER_TELEMETRY_DIR=str(work / "internal"), BOOSTER_DIAG_DIR=str(work / "internal"),
                          BOOSTER_TELEMETRY_CONFIG=str(config), BOOSTER_RUN_ID="offline-simulation")
        try:
            for index, (bounds, segment, committed) in enumerate(segments):
                cfg = _task_config()
                cfg.policy.onnx_model_path = str(resolved_model)
                policy_config = exec_config.get('policy_config') or {}
                if 'rt_reference_transition_s' in policy_config:
                    cfg.policy.rt_reference_transition_s = float(policy_config['rt_reference_transition_s'])
                robot_config = exec_config.get("robot_config") or {}
                for key in ("joint_names", "default_joint_pos", "joint_stiffness", "joint_damping",
                            "effort_limit", "action_scale_factor"):
                    if key in robot_config:
                        setattr(cfg.robot, key, robot_config[key])
                mjcf = provenance_path(directory, robot_config.get("mjcf") or exec_config.get("mjcf")
                                       or metadata.get("mjcf"), robot_config.get("mjcf_repo_relative"))
                if mjcf is not None:
                    cfg.robot.mjcf_path = str(mjcf)
                model_config = exec_config.get("model") or {}
                if "action_scale_factor" not in robot_config and model_config.get("action_scale_factor") is not None:
                    cfg.robot.action_scale_factor = float(model_config["action_scale_factor"])
                if committed is None:
                    play = next((row for row in playback_events if bounds[0] <= int(row['t_mono_ns']) <= bounds[1]), None)
                    cycle = play['data'].get('cycle') if play else None
                    played_ns = int(play['t_mono_ns']) if play else None
                else:
                    cycle, played_ns = committed.cycle, committed.played_ns
                startup = next((event for event in startup_events if cycle is not None and event.get('cycle') == cycle), None)
                if startup:
                    if startup.get('policy', {}).get('action_offset_source') not in (None, 'command'):
                        raise ValueError('recorded startup does not use the M0 command-offset action protocol')
                    transition_s = startup.get('policy', {}).get('rt_reference_transition_s')
                    if transition_s is not None:
                        cfg.policy.rt_reference_transition_s = float(transition_s)
                startup_status.append('recorded_entry_rebuilt' if startup else 'legacy_approximate_no_startup_snapshot')
                options={'startup':startup} if startup else {}
                seconds, qpos, natives = _retarget_segment(cfg, segment, origin, native_of, **options)
                if startup and len(seconds):
                    seconds = seconds - seconds[0] + (played_ns - origin) / 1e9
                count = min(len(seconds), len(qpos), len(natives))
                all_seconds.append(np.asarray(seconds[:count], dtype=np.float64))
                all_qpos.append(np.asarray(qpos[:count], dtype=np.float32))
                all_natives.append(np.asarray(natives[:count], dtype=np.int64))
                all_segments.append(np.full(count, index, dtype=np.int64))
                if committed is not None:
                    all_playback_cycles.append(np.full(count, committed.cycle, dtype=np.int64))
                    all_trajectory_indices.append(np.full(count, committed.trajectory_index, dtype=np.int64))
                    segment_playbacks.append({"segment_id": index, "playback_cycle": committed.cycle,
                                              "trajectory_index": committed.trajectory_index,
                                              "trajectory_path": committed.trajectory_path})
        finally:
            from .runtime import close_process
            close_process()
            for key, value in old_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
    seconds = np.concatenate(all_seconds)
    qpos = np.concatenate(all_qpos)
    natives = np.concatenate(all_natives)
    segment_ids = np.concatenate(all_segments)
    order = np.argsort(seconds, kind="stable")
    seconds, qpos, natives = seconds[order], qpos[order], natives[order]
    segment_ids = segment_ids[order]
    provenance = {
        "schema": "booster_telemetry_derived_simulation_v2", "run_id": manifest.get("run_id"),
        "origin_mono_ns": origin, "clock_domain": manifest.get("clock_domain"),
        "topology": "retarget", "task": "t2_gmc_rt", "source": "recorded retarget input -> M0 policy -> MujocoController",
        "resolved_model": str(resolved_model), "model_source": model_source,
        "model_sha256": digest,
        "action_offset_source": cfg.policy.action_offset_source,
        "action_scale_factor": float(cfg.robot.action_scale_factor),
        "segments": len(segments), "frames": len(seconds),
        "startup": startup_status,
        "semantics": "same-input MuJoCo closed-loop; not exact command replay or physical equivalence",
        "native_index_source": "recorded publish lineage (-1 if unavailable)",
        "unknown_native_frames": int((natives < 0).sum()),
        "intervals_inferred": bool(interval_provenance["inferred"]),
        "interval_source": interval_provenance["source"],
    }
    if interval_provenance["inferred"]:
        provenance["interval_fallback_reason"] = interval_provenance["reason"]
    playback_columns = {}
    if segment_playbacks:
        provenance["segment_playbacks"] = segment_playbacks
        playback_columns = {"playback_cycle": np.concatenate(all_playback_cycles)[order],
                            "trajectory_index": np.concatenate(all_trajectory_indices)[order]}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
        np.savez_compressed(handle, seconds=seconds, qpos=qpos, native_index=natives, segment_id=segment_ids,
                            metadata=np.array(json.dumps(provenance, sort_keys=True)), **playback_columns)
    os.replace(temporary, output)
    return output
