"""Read-only telemetry sources for the teleoperation dashboard.

Three sources, all side-effect free:

* :class:`TmuxProbe`  -- session/pane inventory, optional service flags, pane tails.
* :class:`DiagTail`   -- incremental reader of ``$BOOSTER_DIAG_DIR/<pane>.jsonl``.
* :class:`PortProbe`  -- TCP reachability of the well-known teleop ports.

The dashboard must never become a second control plane: nothing in this module
writes to tmux, to the robot, or to the diagnostics files.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------- roles

# Pane titles are set by scripts/teleop_tmux.sh via `tmux select-pane -T`.
_TITLE_ROLES: tuple[tuple[str, str], ...] = (
    ("publisher", "publisher"),
    ("offline", "offline"),
    ("hand retarget worker", "hand_worker"),
    ("retarget worker", "worker"),
    ("robot viewer", "viewer_robot"),
    ("human viewer", "viewer_human"),
    ("mujoco simulation", "deploy"),
    ("real robot deploy", "deploy"),
    ("raw pico recorder", "record"),
    ("dashboard", "dashboard"),
)


def role_for(title: str, command: str = "") -> str:
    """Best-effort role for a pane, from its title (then its command)."""
    text = f"{title} {command}".lower()
    for needle, role in _TITLE_ROLES:
        if needle in text:
            return role
    return "other"


# --------------------------------------------------------------------------- tmux


def _tmux(*args: str, timeout: float = 5.0) -> str:
    result = subprocess.run(
        ["tmux", *args], capture_output=True, text=True, timeout=timeout, check=False
    )
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout).strip() or "tmux failed")
    return result.stdout


# Only deployed entry points count as service evidence, never arbitrary children.
_ROLE_ENTRIES = {
    "publisher": {"pico_rt_zmq_publisher.py", "mocap_rt_zmq_publisher.py",
                  "offline_pico_publisher.py", "offline_mocap_publisher.py"},
    "offline": {"offline_pico_publisher.py", "offline_mocap_publisher.py"},
    "deploy": {"deploy.py"},
    "worker": {"retarget_rt_zmq_worker.py"},
    "hand_worker": {"hand_retarget_rt_zmq_worker.py"},
    "viewer_robot": {"retarget_rt_zmq_robot_viewer.py"},
    "viewer_human": {"pico_rt_zmq_human_viewer.py", "mocap_rt_zmq_human_viewer.py"},
    "record": {"record_pico_stream.py"},
    "dashboard": {"dashboard.py"},
}
SHELL_COMMANDS = {"bash", "zsh", "sh", "fish", "dash"}


class ProcSnapshot:
    """One bounded PID/PPid inventory shared by all panes in an observation.

    Does not require CONFIG_PROC_CHILDREN. Kernel stat PPid is the parent's
    thread-group ID, including children created by non-main threads. Unknown
    visibility/bounds cannot prove absence; positive matching evidence can.
    """
    def __init__(self, root=None, max_nodes=8192, budget_s=.15):
        self.root = root if root is not None else Path('/proc')
        self.nodes = {}
        self.children = {}
        self.complete = True
        begin = time.monotonic()
        try:
            count = 0
            for entry in self.root.iterdir():
                if not entry.name.isdigit():
                    continue
                count += 1
                if count > max_nodes or time.monotonic()-begin > budget_s:
                    self.complete = False
                    break
                try:
                    fields = (entry/'stat').read_text().rsplit(')',1)[1].split()
                    pid, parent = int(entry.name), int(fields[1])
                    self.nodes[pid] = fields[0]
                    self.children.setdefault(parent, []).append(pid)
                except FileNotFoundError:
                    continue  # raced with exit, not an unreadable live node
                except (OSError, ValueError, IndexError):
                    self.complete = False
        except OSError:
            self.complete = False


def _service_process(pid: int, role: str, tree=None) -> tuple[bool | None, int | None]:
    """Match live role entries in a shared PPid inventory; no ps or SDK calls."""
    entries = _ROLE_ENTRIES.get(role)
    if not entries or not pid:
        return None, None
    tree = tree if tree is not None else ProcSnapshot()
    pending, seen = [(pid, 0)], set()
    complete = tree.complete
    while pending:
        if len(seen) >= 128:
            complete = False
            break
        current, depth = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        state = tree.nodes.get(current)
        if state is None or state in {'Z','X'}:
            continue
        try:
            argv = (tree.root/str(current)/'cmdline').read_bytes().decode(errors='replace').rstrip('\0').split('\0')
            executable = Path(argv[0]).name
            script = executable
            if executable.startswith('python'):
                args = iter(argv[1:])
                for arg in args:
                    if arg in {'-c','-m'}:
                        break
                    if arg in {'-W','-X'}:
                        next(args, None)
                    elif not arg.startswith('-'):
                        script = Path(arg).name
                        break
            if script in entries:
                return True, current
        except FileNotFoundError:
            continue
        except (OSError, ValueError, IndexError):
            complete = False
        children = tree.children.get(current, [])
        if depth >= 8 and children:
            complete = False
        else:
            pending.extend((child, depth+1) for child in children)
    return (False if complete else None), None


@dataclass
class PaneInfo:
    pane_id: str
    title: str
    role: str
    dead: bool
    command: str
    width: int = 0
    height: int = 0
    pid: int = 0
    service_alive: bool | None = None
    service_pid: int | None = None


class TmuxProbe:
    """Query the tmux server.  Read-only: list/show-options/capture-pane."""

    PREFIXES = ("booster-teleop-", "dash-")

    def __init__(self, session: str | None = None, allow_any: bool = False) -> None:
        self.requested_session = session
        self.allow_any = allow_any

    # -- discovery ---------------------------------------------------------
    def session(self) -> str | None:
        if self.requested_session:
            return self.requested_session
        try:
            out = _tmux("list-sessions", "-F", "#{session_name}")
        except Exception:
            return None
        candidates = [name for name in out.splitlines() if name.strip()]
        owned = [n for n in candidates if n.startswith(self.PREFIXES)]
        if owned:
            return owned[-1]
        # Never hijack an unrelated tmux session (an operator's shell, other work):
        # only an explicit --session or --allow-any-session may point elsewhere.
        return candidates[-1] if (self.allow_any and candidates) else None

    def alive(self, session: str | None = None) -> bool:
        name = session or self.session()
        if not name:
            return False
        try:
            _tmux("has-session", "-t", f"={name}")
            return True
        except Exception:
            return False

    def created_at(self, session: str | None = None) -> float:
        name = session or self.session()
        if not name:
            return 0.0
        try:
            out = _tmux("display-message", "-p", "-t", f"={name}:", "#{session_created}")
            return float(out.strip() or 0)
        except Exception:
            return 0.0

    # -- panes -------------------------------------------------------------
    def panes(self, session: str | None = None) -> list[PaneInfo]:
        name = session or self.session()
        if not name:
            return []
        fmt = "|".join(
            [
                "#{pane_id}",
                "#{pane_title}",
                "#{pane_dead}",
                "#{pane_current_command}",
                "#{pane_width}",
                "#{pane_height}",
                "#{pane_pid}",
            ]
        )
        try:
            out = _tmux("list-panes", "-s", "-t", f"={name}", "-F", fmt)
        except Exception:
            return []
        panes: list[PaneInfo] = []
        process_tree = ProcSnapshot() if out.strip() else None
        for line in out.splitlines():
            parts = line.split("|")
            if len(parts) < 6:
                continue
            pane_id, title, dead, command, width, height = parts[:6]
            pid = int(parts[6]) if len(parts) > 6 and parts[6].isdigit() else 0
            role = role_for(title, command)
            service_alive, service_pid = _service_process(pid, role, process_tree) if dead.strip() != "1" else (False, None)
            panes.append(
                PaneInfo(
                    pane_id=pane_id,
                    title=title.strip(),
                    role=role,
                    pid=pid,
                    service_alive=service_alive,
                    service_pid=service_pid,
                    dead=dead.strip() == "1",
                    command=command.strip(),
                    width=int(width or 0),
                    height=int(height or 0),
                )
            )
        return panes

    def capture(self, pane_id: str, lines: int = 40) -> str:
        """Tail of one pane (last ``lines`` rows), or '' when unavailable."""
        try:
            return _tmux("capture-pane", "-p", "-t", pane_id, "-S", f"-{max(1, lines)}")
        except Exception:
            return ""

    def option(self, session: str, name: str) -> str | None:
        """Read one tmux session option (``@booster_*``); None when unset."""
        try:
            return _tmux("show-options", "-v", "-t", f"={session}:", name).strip()
        except Exception:
            return None


# --------------------------------------------------------------------------- diag


from booster_deploy.telemetry.dashboard import FactTail as DiagTail


# --------------------------------------------------------------------------- ports

DEFAULT_PORTS: tuple[tuple[str, int], ...] = (
    ("pico_control", 15555),
    ("reference", 15556),
    ("hand_reference", 15558),
    ("retarget_reset", 15559),
    ("viz_human", 8080),
    ("viz_robot", 8081),
    ("viz_sim", 8082),
)


class PortProbe:
    """Cached TCP connect probe (0.15 s timeout, 1 s cache)."""

    #: ZMQ sockets never answer an HTTP request, so an HTTP probe reports them
    #: permanently down (and costs a timeout every poll).  Only the viser/browser
    #: ports get the HTTP/1.1 handshake that keeps their logs clean.
    HTTP_PORTS = frozenset({"viz_human", "viz_robot", "viz_sim"})

    def __init__(self, ports: tuple[tuple[str, int], ...] = DEFAULT_PORTS) -> None:
        self.ports = ports
        self._cache: dict[str, tuple[float, bool]] = {}

    def override(self, name: str, port: int) -> None:
        """Replace one probed port after the launcher recorded the real value."""
        port = int(port)
        self.ports = tuple((n, port if n == name else p) for n, p in self.ports)
        self._cache.pop(name, None)

    def _check(self, port: int, timeout: float = 0.35, http: bool = True) -> bool:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout) as sock:
                if not http:
                    return True
                sock.settimeout(timeout)
                # A bare connect+close (and HEAD) makes viser's websockets server log
                # "did not receive a valid HTTP request" into the viewer pane, burying
                # real errors.  viser only answers a full HTTP/1.1 request, so send one
                # and accept the first bytes of the response.
                sock.sendall(b"GET / HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n")
                return sock.recv(16).startswith(b"HTTP/")
        except OSError:
            return False

    def snapshot(self, now: float | None = None, ttl: float = 1.0) -> dict[str, bool]:
        now = time.monotonic() if now is None else now
        result: dict[str, bool] = {}
        for name, port in self.ports:
            cached = self._cache.get(name)
            if cached and now - cached[0] < ttl:
                result[name] = cached[1]
                continue
            up = self._check(port, http=name in self.HTTP_PORTS)
            self._cache[name] = (now, up)
            result[name] = up
        return result


# --------------------------------------------------------------------------- helpers


def repo_root_from(module_file: str | os.PathLike[str]) -> Path:
    """Repository root for ``booster_deploy/dashboard/<this file>``."""
    return Path(module_file).resolve().parents[2]


def default_diag_dir(repo_root: Path, session: str | None) -> Path | None:
    if not session:
        return None
    return Path(repo_root) / "logs" / session
