"""Aggregate existing teleoperation streams into render-facing latest frames."""
from __future__ import annotations

import os
import time
import threading
from pathlib import Path
from typing import Callable

import mujoco

from booster_deploy.inputs.mocap.adapter import skeleton_connections as mocap_connections
from booster_deploy.inputs.pico.coordinate import pico_skeleton_connections
from booster_deploy.retargeting.rt_config import RealtimeRetargetZmqConfig
from booster_deploy.retargeting.rt_zmq import (
    MocapRetargetInputSubscriber,
    PicoRetargetInputSubscriber,
    Reference50HzSubscriber,
)
from booster_deploy.utils.mjcf_path import t2_mjcf_path
from booster_deploy.visualization.revo2_combined_qpos import Revo2CombinedQposMapper
from .frames import human_frame_payload, qpos_frame_payload, reference_frame_payload, t2_body_qpos
from .model import build_model_manifest, compiled_mesh_bytes, geometry_bundle
from .protocol import TOPIC, decode_sim_snapshot
from .store import LatestFrameStore


class WebVisualizationHub:
    """Read-only subscriber hub. Every worker owns its socket and MuJoCo data."""

    def __init__(
        self,
        repo_root: Path,
        *,
        source_kind: str = "pico",
        execution: str | None = None,
        sim_endpoint: str | None = None,
    ) -> None:
        self.repo_root = Path(repo_root)
        self.source_kind = str(source_kind)
        self.execution = execution
        self.model_path = t2_mjcf_path(self.repo_root)
        body_model_path = self.model_path.with_name("t2_31dof_fc.xml")
        self.hand_mapper = (
            Revo2CombinedQposMapper(body_xml_path=body_model_path, combined_xml_path=self.model_path)
            if self.model_path != body_model_path else None
        )
        self.model = (self.hand_mapper.combined_model if self.hand_mapper is not None
                      else mujoco.MjModel.from_xml_path(str(self.model_path)))
        self.manifest = build_model_manifest(self.model)
        self.manifest["geometry"] = self._geometry_url()
        self.store = LatestFrameStore(stale_after_s=0.5)
        # Display-only FK, never a controller target or policy/inference instance.
        from booster_deploy.robots.t2_contract import CONTROL_JOINT_NAMES, DEFAULT_JOINT_POS_CONTROL
        data = mujoco.MjData(self.model)
        qpos = data.qpos.copy()
        qpos[:7] = [0, 0, 1.0, 1, 0, 0, 0]
        for name, value in zip(CONTROL_JOINT_NAMES, DEFAULT_JOINT_POS_CONTROL):
            qpos[self.model.jnt_qposadr[self.model.joint(name).id]] = value
        self.default_frame = qpos_frame_payload(self.model, data, qpos, seq=-1, sample_ns=0)
        self.manifest["default_frame"] = self.default_frame
        self.state_getter = None
        self._idle_at = time.monotonic()
        self._display_key = None
        self._display_generation = 0
        self._render_generation = 0
        self._replay_waiting = False
        self._preview_frame = None
        self._preview_pending = None
        self._preview_worker = None
        self._preview_lock = threading.Lock()
        self.sim_endpoint = sim_endpoint or os.getenv(
            "BOOSTER_WEB_VIS_ENDPOINT", "tcp://127.0.0.1:5566"
        )
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self.errors: dict[str, str] = {}
        self.human_dropped = 0
        self._mesh_cache: dict[int, bytes] = {}
        self._geometry: bytes | None = None
        self._geometry_lock = threading.Lock()

    def _geometry_url(self) -> str:
        """Versioned URL so a browser cache never outlives the model it belongs to."""
        import hashlib

        from .model import GEOMETRY_VERSION

        try:
            stat = self.model_path.stat()
            stamp = (stat.st_size, int(stat.st_mtime))
        except OSError:
            stamp = (0, 0)
        digest = hashlib.sha1(
            f"{self.model_path.name}:{stamp}:{int(self.model.nmesh)}:{int(self.model.ngeom)}:"
            f"v{GEOMETRY_VERSION}".encode()
        ).hexdigest()[:12]

        return f"/api/visualization/geometry?v={digest}"

    def start(self) -> None:
        if self._threads:
            return
        targets = (
            ("human", self._human_loop),
            ("retarget", self._reference_loop),
            ("sim", self._sim_loop),
        )
        for name, target in targets:
            thread = threading.Thread(
                target=self._guarded(name, target), name=f"webviz-{name}", daemon=True
            )
            thread.start()
            self._threads.append(thread)

    def _guarded(self, name: str, target: Callable[[], None]):
        def run() -> None:
            try:
                target()
            except Exception as exc:  # keep dashboard alive; surface the exact error
                if not self._stop.is_set():
                    self.errors[name] = f"{type(exc).__name__}: {exc}"
        return run

    def _human_loop(self) -> None:
        if self.source_kind == "mocap":
            cfg = RealtimeRetargetZmqConfig(
                input_port=int(os.getenv("BOOSTER_MOCAP_INPUT_PORT", "5557")),
                input_retarget_topic="mocap_rt.retarget_input",
            )
            subscriber = MocapRetargetInputSubscriber(cfg, recv_timeout_ms=250)
            connections = tuple(mocap_connections())
        else:
            cfg = RealtimeRetargetZmqConfig(
                input_port=int(os.getenv(
                    "BOOSTER_PICO_INPUT_PORT",
                    os.getenv("BOOSTER_PICO_CONTROL_PORT", "5555"),
                )),
                input_retarget_topic="pico_rt.retarget_input",
            )
            subscriber = PicoRetargetInputSubscriber(cfg, recv_timeout_ms=250)
            connections = tuple(pico_skeleton_connections())
        try:
            while not self._stop.is_set():
                try:
                    frame = subscriber.recv_latest(timeout_s=0.25)
                except TimeoutError:
                    continue
                if not frame.human_frame:
                    self.human_dropped += 1
                    continue
                try:
                    payload = human_frame_payload(
                        frame.human_frame,
                        seq=int(frame.seq),
                        sample_ns=int(frame.recv_ns),
                        connections=connections,
                    )
                except (ValueError, TypeError):
                    # Keep the last good skeleton on screen instead of blanking the
                    # viewport when PICO reports NaN joints for a few frames.
                    self.human_dropped += 1
                    continue
                self.store.publish("human", payload)
        finally:
            subscriber.close()

    def _reference_loop(self) -> None:
        subscriber = Reference50HzSubscriber(
            host=os.getenv("BOOSTER_RT_REFERENCE_HOST", "127.0.0.1"),
            port=int(os.getenv("BOOSTER_RT_REFERENCE_PORT", "5556")),
            recv_timeout_ms=250,
            recv_hwm=1,
        )
        data = mujoco.MjData(self.model)
        last_idx: int | None = None
        try:
            while not self._stop.is_set():
                try:
                    frames = subscriber.recv_newer_than(last_idx, timeout_s=0.25)
                except TimeoutError:
                    continue
                frame = frames[-1]
                last_idx = int(frame.sample_idx)
                self.store.publish("retarget", reference_frame_payload(
                    self.model, data, frame, hand_mapper=self.hand_mapper,
                ))
        finally:
            subscriber.close()

    def _sim_loop(self) -> None:
        import zmq

        ctx = zmq.Context()
        sock = ctx.socket(zmq.SUB)
        sock.setsockopt(zmq.RCVHWM, 1)
        sock.setsockopt(zmq.RCVTIMEO, 250)
        sock.setsockopt(zmq.LINGER, 0)
        sock.connect(self.sim_endpoint)
        sock.subscribe(TOPIC)
        data = mujoco.MjData(self.model)
        try:
            while not self._stop.is_set():
                try:
                    parts = sock.recv_multipart()
                except zmq.Again:
                    continue
                raw = decode_sim_snapshot(parts[-1])
                qpos = raw["qpos"]
                if self.execution == "real" and self.hand_mapper is not None:
                    qpos = t2_body_qpos(self.model, qpos)
                self.store.publish(
                    "sim",
                    qpos_frame_payload(
                        self.model, data, qpos, seq=raw["seq"], sample_ns=raw["sample_ns"]
                    ),
                )
                if raw["reference_qpos"] is not None:
                    self.store.publish(
                        "sim_reference",
                        qpos_frame_payload(
                            self.model, data, raw["reference_qpos"],
                            seq=raw["seq"], sample_ns=raw["sample_ns"],
                        ),
                    )
        finally:
            sock.close(linger=0)
            ctx.term()

    def _recording_path(self, value) -> str | None:
        if not value:
            return None
        path = Path(value).expanduser()
        return str((path if path.is_absolute() else self.repo_root / path).resolve())

    def _prepare_display(self, state: dict) -> None:
        session = state.get('session') or {}
        path = self._recording_path(session.get('recording'))
        key = (session.get('name'), session.get('version'), session.get('task'), path, session.get('replay'))
        with self._preview_lock:
            if key == self._display_key:
                return
            if self._display_key is not None:
                self.store.discard('human', 'retarget')
            self._display_key = key
            self._display_generation += 1
            self._render_generation += 1
            self._replay_waiting = False
            self._preview_frame = None
            self.errors.pop('preview', None)
            self._preview_pending = ((self._display_generation, path)
                if session.get('replay') == 'offline' and path else None)
            if self._preview_pending and self._preview_worker is None:
                self._preview_worker = threading.Thread(target=self._load_display,
                    daemon=True, name='webviz-preview')
                self._preview_worker.start()

    def _load_display(self) -> None:
        # At most one decode in flight plus one latest pending generation.
        while not self._stop.is_set():
            with self._preview_lock:
                pending = self._preview_pending
                self._preview_pending = None
                if pending is None:
                    self._preview_worker = None
                    return
            generation, path = pending
            try:
                from booster_deploy.dashboard.preview import load_preview
                clip = load_preview(Path(path), max_frames=1)
                frame = dict(kind='human', names=clip['names'], edges=clip['edges'], positions=clip['positions'][0])
                with self._preview_lock:
                    if generation == self._display_generation and not self._stop.is_set():
                        self._preview_frame = frame
                        self.errors.pop('preview', None)
            except Exception as exc:
                with self._preview_lock:
                    if generation == self._display_generation:
                        self.errors['preview'] = str(exc)

    def snapshot(self) -> dict:
        state = (self.state_getter() if self.state_getter else {}) or {}
        self._prepare_display(state)
        replay = state.get('replay') or {}
        session = state.get('session') or {}
        same_clip = (session.get('replay') == 'offline' and session.get('recording')
            and self._recording_path(replay.get('recording')) == self._recording_path(session.get('recording')))
        if same_clip and replay.get('age_s') is not None and 0 <= replay['age_s'] <= 3.0:
            waiting = bool(replay.get('waiting')) and not replay.get('paused')
            if waiting != self._replay_waiting:
                self.store.discard('human', 'retarget')
                self._render_generation += 1
                self._replay_waiting = waiting
        if self._replay_waiting:
            self.store.discard('human', 'retarget')
        result = self.store.snapshot()
        with self._preview_lock:
            if 'human' not in result['streams'] and self._preview_frame is not None:
                result['streams']['human'] = self._preview_frame
                result['status']['human'] = dict(display='preview', stale=False,
                    generation=self._render_generation,
                    recording=(state.get('session') or {}).get('recording'))
        # A paused control state or held input alone is not evidence of a
        # paused producer. Require its fresh diagnostic and this effective clip.
        if (session.get('replay') == 'offline' and session.get('recording')
                and replay.get('paused')
                and replay.get('age_s') is not None and 0 <= replay['age_s'] <= 3.0
                and self._recording_path(replay.get('recording')) == self._recording_path(session.get('recording'))):
            for name in ('human', 'retarget'):
                if name in result['streams'] and result['status'].get(name, {}).get('display') != 'preview':
                    result['status'][name]['display'] = 'frozen'
        control = state.get('control_state') or {}
        active = control.get('state') in {'tracking', 'tracking_paused'}
        # Sim has no Booster low-state/control reporter: its existing deploy
        # phase is inference/paused. Never use robot_mode or low-state freshness
        # as a rendering prerequisite, and never treat a standby frame as active.
        if (state.get('session') or {}).get('mode') == 'sim':
            active = active or control.get('state') in {'inference', 'paused'}
        now = time.monotonic()
        if not active:
            self._idle_at = now
        sim_status = result['status'].get('sim') or {}
        fresh_active = (active and not sim_status.get('stale', True)
                        and sim_status.get('received_mono_s', 0) > self._idle_at)
        if not fresh_active:
            result['streams']['sim'] = self.default_frame
            result['status']['sim'] = {'display': 'default', 'stale': False, 'age_s': 0}
            result['streams'].pop('sim_reference', None)
        else:
            result['status']['sim']['display'] = 'echo'
        if 'retarget' not in result['streams']:
            result['streams']['retarget'] = self.default_frame
            result['status']['retarget'] = {'display': 'default', 'stale': False, 'age_s': 0}
        for status in result['status'].values():
            status['generation'] = self._render_generation
        result["errors"] = dict(self.errors)
        return result

    def mesh(self, mesh_id: int) -> bytes:
        mesh_id = int(mesh_id)
        if mesh_id not in self._mesh_cache:
            self._mesh_cache[mesh_id] = compiled_mesh_bytes(self.model, mesh_id)
        return self._mesh_cache[mesh_id]

    def geometry(self) -> bytes:
        """Whole-robot geometry in one atomic payload (see model.geometry_bundle)."""
        with self._geometry_lock:
            if self._geometry is None:
                self._geometry = geometry_bundle(self.model)
            return self._geometry

    def stop(self) -> None:
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=0.6)
        self._threads.clear()
