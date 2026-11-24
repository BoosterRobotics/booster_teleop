"""Assemble the dashboard snapshot from read-only sources.

The snapshot has a deliberately small *stable subset* (see the design notes):

    session / panes / pipeline / control_state / replay / alerts / viz / ports / ts

Everything else in the diagnostics records is passed through untouched under
``panes[].state`` so a later schema change cannot break the page: missing keys
render as an em dash instead of raising.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .sources import DEFAULT_PORTS, DiagTail, PaneInfo, PortProbe, TmuxProbe, role_for

def consumer_health(records, now_ns):
    candidates = [r for k, r in records.items() if k.split('.')[0] == 'deploy'
                  and isinstance((r.get('state') or {}).get('input_progress'), dict)]
    if not candidates:
        return dict(known=False, effective_hz=None, stall_ms=None, state='unknown')
    record = max(candidates, key=lambda r: r.get('health_mono_ns', 0))
    facts = record['state']['input_progress']
    known = bool(facts.get('known'))
    last = facts.get('last_advance_ns')
    health = dict(known=known, state=facts.get('state', 'unknown'),
                  effective_hz=float(sum(now_ns-1_000_000_000 < stamp <= now_ns
                                         for stamp in facts.get('events', ()))) if known else None,
                  stall_ms=None if last is None else max(0., (now_ns-last)/1e6),
                  last_advance_ns=last, clock=facts.get('clock'))
    observed = record.get('health_mono_ns')
    if observed is None or not 0 <= now_ns-observed <= STREAM_STALE_S*1e9:
        health.update(known=False, effective_hz=None, state='unknown', diagnostics_stale=True)
    return health


RATE_FLOOR_HZ = 49.5          # mirrors booster_deploy.utils.diag_report.RATE_FLOOR_HZ
WARMUP_S = 2.0                # same warm-up window as the diagnostics
STREAM_STALE_S = 3.0          # stream panes emit every second; silence = broken
ALERT_DEDUPE_S = 60.0
HISTORY_MAX = 400

#: Panes that emit every second while merely running. State panes (ctrl/deploy)
#: stay silent when idle by design, so silence there is not an alert.
CONTINUOUS_PANES = frozenset({"publisher", "offline", "worker"})

#: Stages allowed to report 0 Hz while idle without turning the row red.
IDLE_CAPABLE_STAGES = frozenset({"ctrl", "policy"})

CONTROL_LABELS: dict[str, str] = {
    "walking": "Walking",
    "arming_tracking": "Preparing to enter teleoperation",
    "tracking": "Live teleoperation",
    "tracking_paused": "HOLD (paused)",
    "exiting_to_walk": "Exiting",
    "arming_pvt": "Establishing PVT",
    "pvt_hold": "PVT hold",
    "fault": "FAULT",
    # RealControlState (secondary state machine)
    "wait_start": "Waiting to start",
    "prepare": "Preparing",
    "wait_activate": "Waiting for activation",
    "policy_standing": "Fixed-pose standing",
    "policy_tracking": "Policy tracking",
    "stopping": "Stopping",
}

#: (stage key, label, pane role, primary hz metric, secondary hz metrics)
#: Pane roles come from the tmux title, diagnostic files come from the pane name
#: inside each record (``logs/diagnostics/<session>/<name>.jsonl``).  The publisher
#: pane is titled "publisher (offline pico)" but writes ``offline.jsonl``, so the
#: role alone used to hide its rates from the dashboard.
ROLE_RECORDS: dict[str, tuple[str, ...]] = {
    "publisher": ("publisher", "offline", "pico", "mocap"),
    "worker": ("worker",),
    "deploy": ("deploy",),
    "ctrl": ("ctrl",),
}

STAGES: tuple[tuple[str, str, str, tuple[str, ...], tuple[str, ...]], ...] = (
    ("input", "Input device", "publisher", ("body", "sent"), ("ctl",)),
    ("retarget", "Retargeting", "worker", ("solve", "out"), ("in",)),
    ("policy", "Policy closed loop", "deploy", ("infer",), ("tick",)),
    ("ctrl", "Hardware control", "ctrl", ("tick",), ()),
    ("record", "Data recording", "record", ("clip",), ()),
)

VIZ_ROUTES: tuple[tuple[str, str, int], ...] = (
    ("Human skeleton", "0", 8080),
    ("Retargeted robot", "1", 8081),
    ("Simulation / hardware feedback", "2", 8082),
)


def _f(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _lat_p95(record: dict[str, Any], key: str) -> float | None:
    entry = (record.get("lat_ms") or {}).get(key)
    if isinstance(entry, dict):
        return _f(entry.get("p95"))
    return None


def _age(record, now_mono, *, rates=False):
    stamp = _f(record.get("health_mono_ns" if rates else "t_mono_ns"))
    if stamp is None and rates:
        stamp = _f(record.get("t_mono_ns"))
    return None if stamp is None or stamp > now_mono * 1e9 else max(0., now_mono - stamp / 1e9)


def _idle(state):
    return bool(state.get('held') or state.get('paused') or state.get('waiting_for_start')
                or (state.get('input_progress') or {}).get('state') in ('paused', 'held', 'warmup', 'walking')
                or state.get('phase') in ('paused', 'walking', 'idle', 'held')
                or state.get('state') in ('walking', 'tracking_paused', 'pvt_hold'))


@dataclass
class Dashboard:
    """Holds the probes and produces snapshots.  No writes, no threads."""

    repo_root: Path
    session: str | None = None
    diag_dir: Path | None = None
    allow_control: bool = True
    tmux: TmuxProbe = field(default=None, init=False)  # type: ignore[assignment]
    diag: DiagTail = field(default=None, init=False)  # type: ignore[assignment]
    ports: PortProbe = field(default=None, init=False)  # type: ignore[assignment]
    _seen_alerts: dict[tuple[str, str], float] = field(default_factory=dict, init=False)
    _history: list[dict[str, Any]] = field(default_factory=list, init=False)
    _history_loaded: bool = field(default=False, init=False)
    history_error: str | None = field(default=None, init=False)
    _last_snapshot: dict[str, Any] | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.tmux = TmuxProbe(self.session)
        self.ports = PortProbe()
        self.diag = DiagTail(Path(self.diag_dir) if self.diag_dir else None)

    # ---------------------------------------------------------------- lifecycle
    @property
    def session_name(self) -> str | None:
        if self.session:
            return self.session
        name = self.tmux.session()
        self.session = name
        return name

    def ensure_diag_dir(self) -> Path | None:
        if self.diag.diag_dir and self.diag.diag_dir.is_dir():
            return self.diag.diag_dir
        name = self.session_name
        if not name:
            return None
        configured = self.tmux.option(name, "@booster_diag_dir")
        candidate = Path(configured) if configured else Path(self.repo_root) / "logs" / name
        if candidate != self.diag.diag_dir:
            self.diag.diag_dir = candidate
        return candidate

    # ---------------------------------------------------------------- snapshot
    def snapshot(self, now: float | None = None, tail_lines: int = 24) -> dict[str, Any]:
        now = time.time() if now is None else now
        now_mono = time.monotonic()
        session = self.session_name
        alive = bool(session) and self.tmux.alive(session)
        self.ensure_diag_dir()
        records = self.diag.poll()
        reset_port = self.tmux.option(session, "@booster_rt_reset_port") if alive else None
        if reset_port and str(reset_port).strip().isdigit():
            self.ports.override("retarget_reset", int(str(reset_port).strip()))
        ports = self.ports.snapshot(now_mono)

        panes = self._panes(session, records, now_mono, tail_lines) if alive else []
        record_option = self.tmux.option(session, "@booster_live_record") if alive else None
        pipeline = self._pipeline(records, now_mono, live_record=record_option)
        control = self._control(records)
        alerts = self._alerts(panes, pipeline, ports, alive, now)
        replay = self._replay(records, now_mono)

        created = self.tmux.created_at(session) if alive else 0.0
        meta = self._session_meta(panes, session, created, now, alive)
        snapshot = {
            "ts": now,
            "session": meta,
            "control_state": control,
            "pipeline": pipeline,
            "replay": replay,
            "panes": panes,
            "alerts": alerts,
            "ports": ports,
            "viz": [
                {"name": label, "route": route, "port": port, "up": bool(ports.get(f"viz_{key}"))}
                for (label, route, port), key in zip(VIZ_ROUTES, ("human", "robot", "sim"))
            ],
            # Discrete operator events (PICO A/Y/B/X) echoed by the control process.
            "keys": list(reversed(self.diag.events))[:12],
            "input_health": consumer_health(records, time.monotonic_ns()),
            "diag": {
                "dir": str(self.diag.diag_dir) if self.diag.diag_dir else None,
                "files": len(self.diag.files()),
                "read_errors": self.diag.read_errors,
                "history_error": self.history_error,
            },
            "allow_control": bool(self.allow_control),
            "dashboard": {
                # Whether the launcher let this session accept the web keystrokes.
                # Read from tmux so the page can warn before the operator clicks.
                "overlay": (
                    str(self.tmux.option(session, "@booster_dashboard_overlay") or "1").strip() == "1"
                    if (alive and session)
                    else False
                ),
                "control": bool(self.allow_control),
            },
            "floor_hz": RATE_FLOOR_HZ,
        }
        self._last_snapshot = snapshot
        return snapshot

    # ---------------------------------------------------------------- internals
    def _panes(
        self,
        session: str | None,
        records: dict[str, dict[str, Any]],
        now: float,
        tail_lines: int,
    ) -> list[dict[str, Any]]:
        if not session:
            return []
        rows: list[dict[str, Any]] = []
        for pane in self.tmux.panes(session):
            if pane.role == "dashboard":
                continue
            record = self._record_for_role(records, pane.role)
            age = _age(record, now)
            rate_age = _age(record, now, rates=True)
            rows.append(
                {
                    "id": pane.pane_id,
                    "title": pane.title,
                    "role": pane.role,
                    "alive": not pane.dead,
                    "dead": pane.dead,
                    # A wrapper shell can still own a live service child. Prefer
                    # role-specific process evidence. Unreadable trees remain
                    # unknown, never infer death merely from a shell wrapper.
                    "service_down": (not pane.dead)
                    and pane.service_alive is False,
                    "pid": pane.pid,
                    "service_alive": pane.service_alive,
                    "service_pid": pane.service_pid,
                    "command": pane.command,
                    "hz": (record.get("hz") or {}) if rate_age is not None and rate_age <= STREAM_STALE_S else {},
                    "total": record.get("total") or {},
                    "gap_ms": record.get("gap_ms") or {},
                    "lat_ms": record.get("lat_ms") or {},
                    "drop": record.get("drop") or {},
                    "state": record.get("state") or {},
                    "screen": record.get("screen") or "",
                    "age_s": age,
                    "idle": _idle(record.get("state") or {}),
                    "tail": self.tmux.capture(pane.pane_id, tail_lines) if not pane.dead else "",
                }
            )
        return rows

    def _replay(self, records: dict[str, dict[str, Any]], now: float) -> dict[str, Any] | None:
        """Clip progress for offline replays (publisher pane owns the timeline).

        The publisher reports ``frame_index``/``frames_left``/``publish_total`` in
        its diag ``state`` once a second; that is enough for a simple progress bar
        without touching the publisher's data path.
        """
        record = self._record_for_role(records, "publisher")
        state = record.get("state") or {}
        index = _f(state.get("frame_index"))
        left = _f(state.get("frames_left"))
        total = _f(state.get("publish_total"))
        if index is None and left is None and total is None:
            return None
        done = None if (total is None or left is None) else max(0.0, total - left)
        return {
            "frame_index": None if index is None else int(index),
            "remaining": None if left is None else int(left),
            "total": None if total is None else int(total),
            "done": None if done is None else int(done),
            "cycle": state.get("cycle"),
            "sent": (record.get("total") or {}).get("sent"),
            "recording": state.get("recording_path"),
            "fraction": None if not total or done is None else round(done / total, 4),
            "age_s": _age(record, now),
            "waiting": bool(state.get("waiting_for_start")),
            "paused": bool(state.get("paused")),
            "stopped": bool(state.get("stopped")),
        }

    @staticmethod
    def _record_for_role(records: dict[str, dict[str, Any]], role: str) -> dict[str, Any]:
        names = ROLE_RECORDS.get(role, (role,))
        candidates = [r for k, r in records.items() if k in names or k.split(".")[0] in names]
        if candidates:
            return max(candidates, key=lambda r: r.get("t_mono_ns", 0))
        return {}

    def _pipeline(self, records: dict[str, dict[str, Any]], now_mono=None,
                  *, live_record: str | None = None) -> list[dict[str, Any]]:
        now_mono = time.monotonic() if now_mono is None else now_mono
        rows: list[dict[str, Any]] = []
        for key, label, role, primary, secondary in STAGES:
            record = self._record_for_role(records, role)
            is_record = key == "record"
            if is_record and (live_record == "off" or (live_record != "on" and not record)):
                continue
            hz_map = record.get("hz") or {}
            state = record.get("state") or {}
            value: float | None = None
            metric = ""
            for name in primary:
                candidate = _f(hz_map.get(name))
                if candidate is not None:
                    value, metric = candidate, name
                    break
            if value is None:
                for name in secondary:
                    candidate = _f(hz_map.get(name))
                    if candidate is not None:
                        value, metric = candidate, name
                        break
            floor = RATE_FLOOR_HZ
            # The gap window is not always named after the rate metric: the worker
            # measures gaps on "in" while its rate of interest is "solve"/"out".
            gap_p95 = gap_stall = None
            for candidate in (metric, "in", "tick", "body", "solve", "out"):
                entry = (record.get("gap_ms") or {}).get(candidate) if candidate else None
                if isinstance(entry, dict):
                    gap_p95 = _f(entry.get("p95"))
                    gap_stall = _f(entry.get("stall_ms"))
                    break
            idle = _idle(state) or bool(
                value == 0
                and key in IDLE_CAPABLE_STAGES
                and ((record.get('state') or {}).get('state')!='tracking'
                     or (record.get('state') or {}).get('held'))
            )
            if is_record:
                idle = state.get("clip_frames") is None
            ok = None if idle or value is None else value >= floor
            age = _age(record, now_mono, rates=True)
            stale = age is None or age > STREAM_STALE_S or (is_record and "clip_frames" not in state)
            if is_record:
                # Recording is optional, not part of the robot control rate contract.
                ok = None
                if value is not None and value < 0:  # Counter reset after process restart.
                    value = None
            if stale or idle:
                value, ok = None, None
            recording = {}
            if is_record:
                saved = None if stale else state.get("saved", 0)
                if stale:
                    status = "unavailable"
                elif not idle:
                    status = "recording"
                else:
                    status = "saved" if saved else "ready"
                recording = {"record_status": status, "record_count": saved}
            rows.append(
                {
                    "key": key,
                    "label": label,
                    "pane": role,
                    "hz": value,
                    "metric": metric,
                    "floor": floor,
                    "ok": ok,
                    "gap_p95": gap_p95,
                    "stall_ms": gap_stall,
                    "idle": idle,
                    "stale": stale,
                    "age_s": age,
                    "lat_p95": _lat_p95(record, "e2e") if key == "policy" else None,
                    "present": bool(record),
                    **recording,
                }
            )
        return rows

    def _control(self, records: dict[str, dict[str, Any]]) -> dict[str, Any]:
        ctrl_state = (records.get("ctrl") or {}).get("state") or {}
        policy_state = (records.get("deploy") or {}).get("state") or {}
        raw = ctrl_state.get("state") or policy_state.get("phase")
        held = policy_state.get("held")
        if held and raw == "tracking":
            raw = "tracking_paused"
        label = CONTROL_LABELS.get(str(raw), str(raw) if raw else "Unknown")
        mode, stamp = ctrl_state.get("robot_mode"), ctrl_state.get("robot_mode_mono_ns")
        shutdown_allowed = (type(mode) is int and mode in {0, 1, 2}
                            and type(stamp) is int
                            and 0 <= time.monotonic_ns() - stamp <= 2_000_000_000)
        return {
            "observed_mono_ns": (records.get("ctrl") or {}).get("t_mono_ns"),
            "robot_mode": mode,
            "robot_mode_mono_ns": stamp,
            "shutdown_allowed": shutdown_allowed,
            "state": raw,
            "label": label,
            "mode": ctrl_state.get("mode") or policy_state.get("mode"),
            "held": bool(held),
            "phase": policy_state.get("phase"),
            "late_ticks": ctrl_state.get("late_ticks"),
            "gap_p95": ctrl_state.get("gap_p95_ms"),
            "holds_total": policy_state.get("holds_total"),
            "known": bool(raw),
        }

    def _session_meta(
        self,
        panes: list[dict[str, Any]],
        session: str | None,
        created: float,
        now: float,
        alive: bool,
    ) -> dict[str, Any]:
        option = (lambda name: self.tmux.option(session, name)) if (alive and session) else (lambda name: None)
        meta_opt = option("@booster_session_meta") or ""
        bits = meta_opt.split("/") if meta_opt else []
        source = bits[1] if len(bits) > 1 else None
        replay = bits[3] if len(bits) > 3 else None
        if not bits:
            # Fallback for sessions started before these options existed.
            first_title = next((p["title"] for p in panes if p["role"] == "publisher"), "")
            raw = first_title.replace("publisher (", "").replace(")", "").split()
            replay = raw[0] if raw else None
            source = raw[1] if len(raw) > 1 else None
        mode = option("@booster_session_mode")
        if not mode:
            # `ctrl` is a diagnostics pane name, not a tmux pane: never infer from it.
            titles = " ".join(p["title"].lower() for p in panes)
            mode = "real" if "real robot" in titles else ("sim" if "mujoco" in titles else None)
        return {
            "name": session,
            "alive": alive,
            "uptime_s": max(0.0, now - created) if created else None,
            "roles": sorted({p["role"] for p in panes}),
            "replay": replay,
            "source": source,
            "mode": mode,
            "recording": option("@booster_recording"),
            "version": str(created),
        }

    def _alerts(
        self,
        panes: list[dict[str, Any]],
        pipeline: list[dict[str, Any]],
        ports: dict[str, bool],
        alive: bool,
        now: float,
    ) -> list[dict[str, Any]]:
        self._load_history()
        fresh: list[dict[str, Any]] = []

        def push(level: str, pane: str, text: str, at: float | None = None, detail: str | None = None,
                 code: str | None = None, params: dict | None = None, detail_code: str | None = None,
                 origin: str | None = None) -> None:
            stamp = now if at is None else at
            # Key on the *text only*: keep dynamic numbers in `detail`, otherwise
            # every poll invents a "new" alert and the history fills with noise.
            key = (pane, text)
            last = self._seen_alerts.get(key)
            if last is not None and now - last < ALERT_DEDUPE_S:
                return
            self._seen_alerts[key] = now
            item = {"level": level, "pane": pane, "text": text, "at": stamp}
            if code:
                item.update(code=code, params=params or {})
            if detail_code:
                item['detail_code'] = detail_code
            if origin:
                item['origin'] = origin
            if detail:
                item["detail"] = detail
            fresh.append(item)

        # 0) A service pane that fell back to a shell prompt has died. This is the
        #    loud version of "robot does not move": no diagnostics, no error, just a
        #    bash prompt where the deploy should be.
        for pane in panes:
            if pane.get("service_down"):
                push(
                    "error",
                    str(pane.get("role") or "pane"),
                    f"{pane.get('title') or pane.get('role')} process has exited (pane returned to {pane.get('command')})",
                    detail="The robot will receive no commands; click this pane to inspect recent logs",
                    code="alert_process_exit", params={'p0':pane.get('title') or pane.get('role'), 'p1':pane.get('command')},
                    detail_code="alert_no_commands",
                )

        # 0b) Heartbeat: a pane whose process lives but stopped writing diagnostics is
        #     wedged (deadlock, blocked wait). Without this the operator only sees a
        #     frozen robot and a "0 Hz" row.
        for pane in panes:
            if pane.get("service_down") or pane.get("dead") or pane.get("idle"):
                continue
            age = pane.get("age_s")
            if age is None or float(age) < 2.0:
                continue
            role = str(pane.get("role") or "")
            if role not in {"deploy", "worker", "publisher", "ctrl"}:
                continue
            push(
                "warn",
                role,
                f"{pane.get('title') or role} heartbeat interrupted (last diagnostic was {float(age):.1f}s ago)",
                detail="The process may be stuck in a synchronous wait; press B to check whether it still responds",
                code="alert_heartbeat", params={'p0':pane.get('title') or role, 'p1':f'{float(age):.1f}'},
                detail_code="alert_waiting",
            )

        # 1) diagnostics alerts, verbatim (they are already health-only).  These are
        # surfaced even when the tmux session is gone: a crashed session is exactly
        # when the last few health lines matter most.
        for item in self.diag.recent_alerts[-40:]:
            if now - float(item["at"]) > 300:
                continue
            text = str(item["text"])
            level = "error" if "fault" in text.lower() else "warn"
            push(level, str(item["pane"]), text, float(item["at"]), origin="diagnostic")

        for stage in pipeline:
            if stage.get('present') and stage.get('stale') and not stage.get('idle'):
                push('warn', stage['pane'], f"{stage['label']} facts are stale or unknown", code='alert_stream_stale')
            if stage['ok'] is False and not stage.get('idle'):
                push('warn',stage['pane'],f"{stage['label']} rate is below the required threshold",
                     detail=f"{stage['hz']:.1f} Hz < {stage['floor']:.0f} Hz",code='alert_rate',params={'p0':stage['label']})

        if alive:
            # 2) pane death
            for pane in panes:
                if pane["dead"]:
                    push("error", pane["role"], f"{pane['title']} has exited", code="alert_pane_exit", params={'p0':pane['title']})

            # 3) stream pane went silent (state panes are allowed to be quiet)
            for pane in panes:
                role = pane["role"]
                age = pane["age_s"]
                if role in CONTINUOUS_PANES and not pane.get("idle") and age is not None and age > STREAM_STALE_S:
                    push("warn", role, "Fact heartbeat stopped updating (this stage may have lost its stream)", detail=f"{age:.1f}s", code="alert_stream_stale")


            # Legacy terminal mode still uses independent viser panes.  Dashboard
            # mode renders from ZMQ snapshots and intentionally has no :8080-8082.
            if not any(p["role"] == "dashboard" for p in panes):
                for name, route, port in VIZ_ROUTES:
                    key = {"0": "viz_human", "1": "viz_robot", "2": "viz_sim"}[route]
                    if not ports.get(key) and any(p["role"] == "viewer_" + ("human" if route == "0" else "robot") for p in panes):
                        push("info", f"viz{route}", f"{name} :{port} is not listening (it may have been closed)", code="alert_viewer", params={'p0':name, 'p1':port})

        self._append_history(fresh)
        view = list(reversed(self._history[-HISTORY_MAX:]))
        # Mark the entries added this round so the page can highlight them.
        fresh_keys = {(item["pane"], item["text"], item["at"]) for item in fresh}
        for item in view:
            item["fresh"] = (item.get("pane"), item.get("text"), item.get("at")) in fresh_keys
        return view[:60]

    # ------------------------------------------------------------------ history
    def _load_history(self) -> None:
        self._history_loaded=True

    def _append_history(self,items: list[dict[str, Any]]) -> None:
        self._history.extend(items)
        del self._history[:-HISTORY_MAX]
