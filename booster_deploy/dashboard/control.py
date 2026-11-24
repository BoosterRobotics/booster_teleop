"""Whitelisted control actions for the dashboard.

Design rules agreed with the operator:

* Only actions that already exist as a physical/keyboard/zmq path are exposed.
  Nothing here invents a new way to move the robot.
* Editing joint targets or bypassing the A/B/Y state machine is never exposed.
* Explicit replay changes may restart deploy only after fresh Walking;
  no automatic activation follows a restart.
* Every failure is reported to the caller.  A swallowed keystroke would make an
  operator believe the robot received a command it never got.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
import threading
import shlex
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

RESET_TOPIC = b"retarget_rt.reset"
RESET_PAYLOAD = b"reset_from_current"
#: The launcher binds the worker's reset socket on 5559 for offline replays and
#: on 15559 for onboard sessions.  Never hard-code one of them.
RESET_PORT_CANDIDATES = (15559, 5559)

#: dashboard action -> (pane role, key, human label)
KEY_ACTIONS: dict[str, tuple[str, str, str]] = {
    "enter_tracking": ("deploy", "r", "Enter teleoperation (equivalent to pressing A)"),
    "pause_toggle": ("deploy", "y", "Pause / resume (equivalent to pressing Y)"),
    "return_to_walk": ("deploy", "b", "Return to Walking (equivalent to pressing B)"),
}

ALLOWED = (
    set(KEY_ACTIONS)
    | {"reset_retarget", "export_summary", "switch_trajectory", "shutdown_stack"}
)


def _failure(action: str, error: str, **extra: Any) -> dict[str, Any]:
    return {"ok": False, "action": action, "error": error, **extra}


@dataclass
class Controller:
    repo_root: Path
    session_getter: Callable[[], str | None]
    pane_lookup: Callable[[str], str | None]   # role -> pane_id
    enabled: bool = True
    option_getter: Callable[[str], str | None] | None = None
    snapshot_getter: Callable[[], dict[str, Any] | None] | None = None
    _switch_lock: Any = field(default_factory=threading.Lock, init=False, repr=False)
    _restart_ns: int = field(default=0, init=False)

    # ------------------------------------------------------------------ entry
    def _run_directory(self, session=None):
        session=session or self.session_getter()
        if not session or Path(session).name!=session or session in ('.','..'): return None
        configured=self._option('@booster_diag_dir')
        return Path(configured) if configured else Path(self.repo_root)/'logs'/session

    def _fact(self,directory,stage,**facts):
        if directory is None: return False
        from booster_deploy.telemetry.runtime import record_for
        return record_for(directory,stage,**facts)

    def dispatch(self, action: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        directory=self._run_directory()
        request_id=f'{os.getpid()}:{time.monotonic_ns()}'
        self._fact(directory,'dashboard',phase='request',action=action,payload=payload or {},request_id=request_id)
        if not self._switch_lock.acquire(blocking=False):
            result=_failure(action, "Another control/switch transaction is in progress")
            self._fact(directory,'dashboard',phase='result',request_id=request_id,result=result)
            return result
        self._event_directory=directory
        try:
            result=self._dispatch(action,payload)
            self._fact(directory,'dashboard',phase='result',action=action,request_id=request_id,result=result)
            return result
        except Exception as exc:
            self._fact(directory,'dashboard',phase='result',action=action,request_id=request_id,error=str(exc))
            raise
        finally:
            self._event_directory=None
            self._switch_lock.release()

    def _dispatch(self, action: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        if not self.enabled:
            return _failure(action, "Control is disabled by --no-control")
        if action not in ALLOWED:
            return _failure(action, f"Unsupported action: {action}")
        if action in KEY_ACTIONS:
            return self._key(action)
        if action == "reset_retarget":
            return self._reset()
        if action == "switch_trajectory":
            try:
                return self.switch_trajectory(str((payload or {}).get("path") or "").strip())
            except Exception as exc:
                return _failure(action, str(exc))
        if action == "shutdown_stack":
            return self.shutdown_stack()
        return self._summary()

    # ------------------------------------------------------------------ shutdown
    def shutdown_stack(self) -> dict[str, Any]:
        """Shut down the entire teleoperation stack (fresh measured robot mode 0/1/2).

        The shutdown logic is in scripts/teleop/teleop_shutdown.sh: it stops services and waits for producer flush/drain,
        then analyzes and saves pane logs before destroying the tmux session. This method only checks the gate and triggers shutdown,
        and returns immediately: the page will soon disconnect from this process.
        """

        session = self.session_getter() if self.session_getter else None
        if not session:
            return _failure("shutdown_stack", "No tmux session is available")
        session_mode=self._option('@booster_session_mode')
        if session_mode not in ('sim','real'):
            return _failure('shutdown_stack','Session lacks explicit sim/real metadata; refusing to infer the safety gate')
        snapshot = self.snapshot_getter() if self.snapshot_getter else None
        state = (snapshot or {}).get("control_state") or {}
        mode, stamp = state.get("robot_mode"), state.get("robot_mode_mono_ns")
        if session_mode=='real' and not (type(mode) is int and mode in {0, 1, 2} and type(stamp) is int
                and 0 <= time.monotonic_ns() - stamp <= 2_000_000_000):
            return _failure(
                "shutdown_stack",
                "Shutdown denied: actual robot mode is 3, unknown, or stale",
                hint="Requires a fresh measured mode 0(Damping)/1(PVT)/2(Walking); local PVT state does not imply actual mode 1",
            )
        script = Path(self.repo_root) / "scripts" / "teleop" / "teleop_shutdown.sh"
        if not script.is_file():
            return _failure("shutdown_stack", f"Missing {script}")
        try:
            subprocess.Popen(
                ["setsid", "bash", str(script), "--session", str(session), "--reason", "web"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd=str(self.repo_root),
                start_new_session=True,
            )
        except Exception as exc:  # noqa: BLE001
            return _failure("shutdown_stack", f"Failed to start the shutdown script: {exc}")
        return {
            "ok": True,
            "action": "shutdown_stack",
            "detail": "Shutdown started: stop services → drain → analyze → close session (this page will disconnect)",
        }

    # ------------------------------------------------------------------ replay
    def _option(self, name: str) -> str | None:
        if self.option_getter is None:
            return None
        try:
            return self.option_getter(name)
        except Exception:  # noqa: BLE001
            return None

    def _require_walking(self) -> None:
        if not self.session_getter():
            raise ValueError("No tmux session is available")
        snapshot = self.snapshot_getter() if self.snapshot_getter else {}
        state = (snapshot or {}).get("control_state") or {}
        if state.get("state") != "walking":
            raise ValueError("Switching is allowed only in Walking; wait for confirmation")
        session_mode = self._option("@booster_session_mode")
        if session_mode not in {'sim', 'real'}:
            raise ValueError('Session lacks an explicit sim/real mode; refusing to infer the safety gate')
        real = session_mode == "real"
        stamp = state.get("robot_mode_mono_ns") if real else state.get("observed_mono_ns")
        if (type(stamp) is not int or not 0 <= time.monotonic_ns() - stamp <= 2_000_000_000
                or stamp <= self._restart_ns or (real and state.get("robot_mode") != 2)):
            raise ValueError("Requires fresh measured Walking (actual mode=2 on hardware); wait for state from the new process after restarting")

    @staticmethod
    def _replace_value(command: str, flag: str, old: str, new: str) -> str:
        # Preserve the launcher's shell setup verbatim. Do not shlex.join a shell
        # program: that would quote &&/exports and silently break the second launch.
        token = r'''(?:[^\s"'\\]|\\.|"(?:\\.|[^"])*"|'[^']*')+'''
        matches = list(re.finditer(token, command))
        values = [shlex.split(m.group())[0] for m in matches]
        indices = [i + 1 for i, v in enumerate(values[:-1]) if v == flag and values[i + 1] == old]
        if len(indices) != 1:
            raise ValueError(f"Saved command has a non-unique {flag} argument or a metadata mismatch; refusing to rewrite")
        m = matches[indices[0]]
        return command[:m.start()] + shlex.quote(new) + command[m.end():]

    @staticmethod
    def _tmux(*args: str) -> str:
        result = subprocess.run(["tmux", *args], capture_output=True, text=True, timeout=10, check=False)
        if result.returncode:
            raise RuntimeError((result.stderr or result.stdout).strip() or "tmux failed")
        return result.stdout.strip()

    _stop_timeout_s = 8.0

    @staticmethod
    def _descendants(tree, root):
        pending, found = [root], set()
        while pending:
            pid = pending.pop()
            if pid in found:
                continue
            found.add(pid)
            pending.extend(tree.children.get(pid, ()))
        return {pid for pid in found if tree.nodes.get(pid) not in (None, 'Z', 'X')}

    def _stop_pane(self, pane, role):
        """SIGINT only the actual pane tree, then prove exit and writer drain.

        No kill escalation, invented terminals, marker removal, or unbounded waits.
        Unknown process inventory or historical writer drain fails closed.
        """
        import signal
        from .sources import ProcSnapshot, SHELL_COMMANDS
        from booster_deploy.telemetry.reader import iter_file
        directory = self._run_directory()
        if directory is None:
            raise RuntimeError('Run directory is missing; writer drain cannot be verified')
        started = time.monotonic_ns()
        deadline = time.monotonic() + self._stop_timeout_s
        self._tmux('set-option', '-p', '-t', pane, 'remain-on-exit', 'on')
        inventory = self._tmux('list-panes', '-s', '-t', f'={self.session_getter()}', '-F', '#{pane_id}|#{pane_pid}|#{pane_dead}')
        entries = [line.split('|') for line in inventory.splitlines()]
        own = next((entry for entry in entries if entry[0] == pane), None)
        if own is None:
            raise RuntimeError('The pane to stop has disappeared; refusing to infer the process')
        root = int(own[1])
        tracked, signalled = set(), set()
        while True:
            tree = ProcSnapshot()
            if not tree.complete:
                raise RuntimeError('The full process tree cannot be observed; preserving the pane')
            current = self._descendants(tree, root) if own[2] != '1' else set()
            tracked.update(current)
            live = {pid for pid in tracked if tree.nodes.get(pid) not in (None, 'Z', 'X')}
            active = set(live)
            if root in active:
                try:
                    executable = Path(Path(f'/proc/{root}/cmdline').read_bytes().split(b'\0')[0].decode()).name
                except FileNotFoundError:
                    executable = ''
                if executable in SHELL_COMMANDS:
                    active.discard(root)  # An idle wrapper is safe only with no live descendants.
            for pid in active - signalled:
                try:
                    os.kill(pid, signal.SIGINT)
                except ProcessLookupError:
                    pass
                signalled.add(pid)
            other_pids = {os.getpid()}
            for entry in entries:
                if entry[0] != pane and entry[2] != '1':
                    other_pids.update(self._descendants(tree, int(entry[1])))
            registered = {}
            for marker in directory.glob('writer.*.json'):
                data = json.loads(marker.read_text())
                if data['pid'] not in other_pids or data['pid'] in tracked:
                    registered[data['writer_token']] = data
            terminal = {r['writer_token']: r['data'] for path in directory.glob('terminal.*.msgpack') for r in iter_file(path)}
            bad = [token for token in registered if token in terminal and
                   (terminal[token].get('state') != 'closed' or terminal[token].get('incomplete'))]
            if bad:
                raise RuntimeError(f'{role} writer terminal state indicates failure: {bad}')
            pending = set(registered) - terminal.keys()
            if not active and not pending:
                break
            if time.monotonic() >= deadline:
                raise RuntimeError(f'{role} stop/drain timed out; preserving the safe state active={sorted(active)} writers={sorted(pending)}')
            time.sleep(.05)
        if role == 'deploy' and signalled and self._option('@booster_session_mode') == 'real':
            from booster_deploy.telemetry.dashboard import control_state
            state = control_state(directory)
            mode, stamp = state.get('robot_mode'), state.get('robot_mode_mono_ns')
            if not (state.get('shutdown_confirmed') is True and type(mode) is int and mode in (0, 1, 2)
                    and type(stamp) is int and started <= stamp <= time.monotonic_ns()
                    and time.monotonic_ns() - stamp <= 30_000_000_000):
                raise RuntimeError('deploy has not confirmed a safe exit for this request; preserving the pane and not restarting the source')

    def _restart(self, action: str, commands: dict[str, str], updates: dict[str, str]) -> dict[str, Any]:
        session = self.session_getter()
        panes = {role: self.pane_lookup(role) for role in commands}
        if not all(panes.values()):
            raise ValueError("The pane to restart is missing; services will not be created automatically")
        previous = {role: self._option(f"@booster_{role}_cmd") for role in commands}
        if not all(previous.values()):
            raise ValueError("The existing session has no saved command; reopen the session with the original CLI")
        saved = {key: self._option(key) for key in updates}
        # Stop deploy first; never leave it consuming a half-switched source.
        order = ['deploy'] + [role for role in commands if role != 'deploy']
        try:
            self._require_walking()
            self._restart_ns = time.monotonic_ns()
            for role in order:
                self._stop_pane(panes[role], role)
            if 'publisher' in commands and self._option('@booster_session_mode') == 'real':
                from booster_deploy.utils.replay_control import fence_replay_restart
                directory = self._run_directory()
                if directory is None or not fence_replay_restart(directory):
                    raise RuntimeError('Cannot fence previous replay tokens; refusing to restart the publisher')
        except Exception as exc:
            # A timeout/unknown drain must not trigger a second stop or respawn.
            return _failure(action, str(exc), ready=False, rollback_errors=[],
                            hint='Stop/finalization is unconfirmed; preserving the current safe state; manual inspection required')
        try:
            for role in reversed(order):
                if role != 'deploy':
                    self._tmux('respawn-pane', '-k', '-t', panes[role], commands[role])
            for key, value in updates.items():
                self._tmux('set-option', '-t', f'={session}:', key, value)
            self._tmux('respawn-pane', '-k', '-t', panes['deploy'], commands['deploy'])
        except Exception as exc:
            errors = []
            # Rollback uses exactly the same stop/drain contract. No rollback
            # command can kill a replacement producer that has already started.
            for role in order:
                try:
                    self._stop_pane(panes[role], role)
                except Exception as rollback:
                    errors.append(str(rollback))
                    break
            if not errors:
                try:
                    for key, value in saved.items():
                        if value is None:
                            self._tmux('set-option', '-u', '-t', f'={session}:', key)
                        else:
                            self._tmux('set-option', '-t', f'={session}:', key, value)
                    for role in reversed(order):
                        self._tmux('respawn-pane', '-k', '-t', panes[role], previous[role])
                except Exception as rollback:
                    errors.append(str(rollback))
            return _failure(action, str(exc), rollback_errors=errors, ready=False,
                            hint='Rollback to the previous command ' + ('failed; preserving the safe state; manual inspection required' if errors else 'requested; this does not confirm model readiness'))
        return {"ok": True, "action": action, "ready": False,
                "detail": "Previous producers have stopped and drained; restart requested, but model readiness is not yet confirmed",
                "hint": "Wait for Walking/model readiness from the new process before pressing A; automatic entry into teleoperation is forbidden"}

    def _recording_info(self, path: str) -> dict[str, Any]:
        import zipfile
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            candidate = Path(self.repo_root) / candidate
        candidate = candidate.resolve(strict=True)
        if not candidate.is_file():
            raise ValueError("Recording path is not a file")
        if zipfile.is_zipfile(candidate):
            from booster_deploy.inputs.pico.offline_recording import load_pico_raw_recording
            clip = load_pico_raw_recording(candidate)
            return {'path': str(candidate), 'source': 'pico', 'format': 'raw PICO [T,24,7] XYZW', 'frames': len(clip.poses), 'input_file': str(candidate)}
        from booster_deploy.inputs.mocap.offline_publisher import load_recorded_mocap_bvh
        clip = load_recorded_mocap_bvh(candidate)
        return {'path': str(candidate), 'source': 'mocap', 'format': 'PN-Link recorded BVH', 'frames': clip.frame_count, 'input_file': str(candidate)}

    def trajectories(self) -> dict[str, Any]:
        from booster_deploy.telemetry.run_config import read_execution_config, trajectory_switch_block_reason
        bits = (self._option('@booster_session_meta') or '').split('/')
        offline = len(bits) == 4 and bits[3] == 'offline'
        source = bits[1] if len(bits) == 4 else None
        current = self._option('@booster_recording')
        directory = self._run_directory()
        execution_config = read_execution_config(directory) if directory else {}
        telemetry_lock = bool(execution_config.get('telemetry_enabled') and offline)
        switch_count = int(execution_config.get('trajectory_switch_count', 0))
        playback_started = bool(execution_config.get('first_offline_playback_started'))
        playback_pending = bool(execution_config.get('playback_start_pending'))
        switch = execution_config.get('trajectory_switch', 'once')
        blocked_reason = trajectory_switch_block_reason(execution_config)
        rows = []
        if offline:
            paths = {Path(current)} if current else set()
            for relative in ('logs/pico_raw', 'logs/mocap_raw', 'data/pico_raw', 'data/mocap', 'booster_deploy/assets/trajs'):
                folder = Path(self.repo_root) / relative
                if folder.is_dir():
                    paths.update(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in {'.npz', '.bvh'})
            for path in sorted(paths):
                try:
                    row = self._recording_info(str(path))
                    row['selectable'] = row['source'] == source
                    if row['selectable'] and row['path'] != current and telemetry_lock:
                        if blocked_reason:
                            row['selectable'] = False
                            row['reason'] = blocked_reason
                        elif switch == 'unlimited':
                            row['reason'] = 'Return to confirmed Walking before switching clips'
                        else:
                            row['reason'] = 'One clip change is available before the first playback'
                    else:
                        row['reason'] = None if row['selectable'] else 'Incompatible with the current input source; sources will not be switched automatically'
                except Exception as exc:
                    row = {'path': str(path), 'selectable': False, 'format': 'unsupported',
                           'reason': f'Incompatible content schema: {exc}; robot FK/retarget NPZ is not raw PICO or PN-Link BVH'}
                row.setdefault('name', Path(row['path']).name)
                row['current'] = row['path'] == current
                if row['current'] and execution_config.get('trajectory'):
                    row['name'] = execution_config['trajectory'].get('name') or row['name']
                rows.append(row)
        return {'offline': offline, 'source': source, 'current': current, 'trajectories': rows,
                'telemetry_enabled': telemetry_lock,
                'trajectory_switch': switch,
                'trajectory_switch_count': switch_count,
                'first_offline_playback_started': playback_started,
                'playback_start_pending': playback_pending,
                'hint': 'PICO: raw PICO NPZ; mocap: PN-Link BVH. Robot joint/FK NPZ files are not supported by this replay entry point.'
                        if offline else 'Online input does not select offline trajectories'}

    def switch_trajectory(self, path: str) -> dict[str, Any]:
        self._require_walking()
        bits = (self._option('@booster_session_meta') or '').split('/')
        if len(bits) != 4 or bits[3] != 'offline':
            raise ValueError('Only offline sessions may switch recordings; hardware sources will not be started automatically')
        if bits[2] != self._option('@booster_session_mode'):
            raise ValueError('Session mode and replay metadata disagree; refusing to restart')
        clip = self._recording_info(path)
        source = bits[1]
        if clip['source'] != source:
            raise ValueError('Recording format is incompatible with this session input source')
        from booster_deploy.telemetry.run_config import trajectory_switch
        directory = self._run_directory()
        with trajectory_switch(directory, clip['path']) as trajectory_change:
            old = self._option('@booster_recording')
            deploy = self._option('@booster_deploy_cmd')
            publisher = self._option('@booster_publisher_cmd')
            if not old or not deploy or not publisher:
                raise ValueError('The existing session has no saved recording, deploy command, or publisher command; reopen it with the original CLI')
            script = str(Path(self.repo_root) / f'scripts/diagnostics/offline_{source}_publisher.py')
            if bits[2] == 'real' and not {'--no-loop', '--wait-for-start-button'}.issubset(shlex.split(publisher)):
                raise ValueError('The hardware offline publisher lacks the press-A single-replay safety gate; refusing to restart')
            new_publisher = self._replace_value(publisher, script, old, clip['path'])
            worker = self._option('@booster_worker_cmd')
            if not worker:
                raise ValueError('The existing worker command is missing; no additional service will be started')
            commands = {'publisher': new_publisher, 'worker': worker, 'deploy': deploy}
            result = self._restart('switch_trajectory', commands,
                                   {'@booster_recording': clip['path'], '@booster_publisher_cmd': new_publisher})
            if result.get('ok'):
                trajectory_change.commit()
            return result

    # ------------------------------------------------------------------ keys
    def _key(self, action: str) -> dict[str, Any]:
        role, key, label = KEY_ACTIONS[action]
        pane = self.pane_lookup(role)
        if not pane:
            return _failure(action, f"No {role} pane found (session is not running or this stage is disabled)", label=label)
        try:
            result = subprocess.run(
                ["tmux", "send-keys", "-t", pane, key],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except Exception as exc:  # noqa: BLE001
            return _failure(action, f"tmux send-keys failed: {exc}", label=label)
        if result.returncode != 0:
            return _failure(action, (result.stderr or result.stdout).strip() or "tmux failed", label=label)
        self._log_web_key(key, label, action)
        return {
            "ok": True,
            "action": action,
            "label": label,
            "detail": f"Sent to {pane}: '{key}'",
        }

    def _log_web_key(self, key: str, label: str, action: str) -> None:
        directory=getattr(self,'_event_directory',None) or self._run_directory()
        self._fact(directory,'keys',key=key,label=label,action=action,origin='dashboard',phase='sent_to_tmux',execution_ack=False)

    # ------------------------------------------------------------------ reset
    def _reset(self) -> dict[str, Any]:
        try:
            import zmq
        except Exception as exc:  # noqa: BLE001
            return _failure("reset_retarget", f"pyzmq is unavailable: {exc}")
        # A PUB with no peer silently drops the message.  Check the worker's
        # subscriber socket first so the operator is told the truth, and use the
        # port this session's worker actually bound.
        import socket as _socket

        port = self._live_reset_port(_socket)
        if port is None:
            probed = "/".join(str(p) for p in self._reset_port_candidates())
            detail = getattr(self, "_reset_port_error", None)
            return _failure(
                "reset_retarget",
                detail or f"{probed} has no subscribers (is the retarget worker running?); reset was not sent",
                hint=f"Probed {probed}",
            )
        from booster_deploy.retargeting.rt_zmq import RetargetResetPublisher

        # Use the same publisher the PICO X button uses: a single PUB send right
        # after connect() is dropped by ZMQ's slow-joiner rule, and that is exactly
        # what made "Reset retargeting" report success while the worker never reset.
        publisher = None
        try:
            publisher = RetargetResetPublisher("127.0.0.1", port, RESET_TOPIC.decode())
            acknowledged = publisher.publish_from_current()
        except Exception as exc:  # noqa: BLE001
            return _failure("reset_retarget", f"Failed to publish reset: {exc}")
        finally:
            if publisher is not None:
                publisher.close()
        if not acknowledged:
            return _failure(
                "reset_retarget",
                f"Worker did not acknowledge reset (no reply from 127.0.0.1:{port}; it may still be starting or incompatible)",
            )
        return {
            "ok": True,
            "action": "reset_retarget",
            "label": "Reset retargeting (equivalent to pressing X)",
            "detail": f"Published retarget_rt.reset → 127.0.0.1:{port}",
        }

    def _reset_port_candidates(self) -> list[int]:
        ordered: list[int] = []
        recorded = self._option("@booster_rt_reset_port")
        for value in (recorded, os.getenv("BOOSTER_RT_RESET_PORT"), *map(str, RESET_PORT_CANDIDATES)):
            text = (value or "").strip()
            if text.isdigit() and int(text) not in ordered:
                ordered.append(int(text))
        return ordered

    def _live_reset_port(self, module) -> int | None:
        """First candidate port whose listener is *this session's* retarget worker.

        A plain "something is listening" check is not enough: a stale worker from
        an earlier session can hold 5559/15559 and would happily swallow the reset
        while the dashboard reports success.
        """
        self._reset_port_error = None
        for port in self._reset_port_candidates():
            pid = self._listener_pid(port)
            if pid is None:
                continue
            if self._listener_is_our_worker(pid):
                return port
            self._reset_port_error = f"The process listening on port {port} (pid {pid}) is not this session's retarget worker"
        return None

    @staticmethod
    def _listener_pid(port: int) -> int | None:
        try:
            out = subprocess.run(
                ["ss", "-tlnp", f"sport = :{int(port)}"],
                capture_output=True, text=True, timeout=3, check=False,
            ).stdout
        except Exception:  # noqa: BLE001
            return None
        for line in out.splitlines()[1:]:
            if "pid=" not in line:
                continue
            text = line.split("pid=", 1)[1].split(",", 1)[0]
            if text.isdigit():
                return int(text)
        return None

    def _listener_is_our_worker(self, pid: int) -> bool:
        pane = self.pane_lookup("worker")
        if not pane:
            # Legacy session without the worker pane record: the best available
            # evidence is that the listener really is a retarget worker.
            return self._process_cmdline_has(pid, "retarget_rt_zmq_worker")
        pane_pid = self._pane_pid(pane)
        if pane_pid is None:
            return self._process_cmdline_has(pid, "retarget_rt_zmq_worker")
        return self._is_descendant(pid, pane_pid)

    @staticmethod
    def _process_cmdline_has(pid: int, needle: str) -> bool:
        try:
            raw = Path(f"/proc/{int(pid)}/cmdline").read_bytes()
        except OSError:
            return False
        return needle.encode() in raw

    @staticmethod
    def _pane_pid(pane_id: str) -> int | None:
        try:
            out = subprocess.run(
                ["tmux", "display-message", "-p", "-t", pane_id, "#{pane_pid}"],
                capture_output=True, text=True, timeout=5, check=False,
            ).stdout.strip()
        except Exception:  # noqa: BLE001
            return None
        return int(out) if out.isdigit() else None

    @staticmethod
    def _is_descendant(pid: int, ancestor: int, depth: int = 12) -> bool:
        current = int(pid)
        for _ in range(depth):
            if current == ancestor:
                return True
            if current <= 1:
                return False
            try:
                stat = Path(f"/proc/{current}/stat").read_text()
                current = int(stat.rsplit(")", 1)[1].split()[1])
            except (OSError, IndexError, ValueError):
                return False
        return False

    # ------------------------------------------------------------------ summary
    def _summary(self) -> dict[str, Any]:
        session = self.session_getter()
        if not session:
            return _failure("export_summary", "No tmux session is available")
        directory=self._run_directory(session)
        if not directory or not directory.is_dir():
            return _failure('export_summary',f'Run directory does not exist: {directory}')
        try:
            from booster_deploy.telemetry.runtime import close_process
            from booster_deploy.telemetry.analysis import save
            close_process()
            save(directory)
        except Exception as exc:
            return _failure('export_summary',str(exc))
        return dict(ok=True,action='export_summary',detail=str(directory/'analysis.en.md'))
