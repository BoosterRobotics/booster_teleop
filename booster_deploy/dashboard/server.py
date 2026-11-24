"""Single-port dashboard server (stdlib only).

Routes
------
``GET  /``                embedded dark page (static/page.html)
``GET  /api/status``      telemetry snapshot (polled at 1 Hz by the page)
``POST /api/control``     whitelisted action, see :mod:`booster_deploy.dashboard.control`
``GET  /healthz``         liveness
``GET  /viz/<n>/...``     reverse proxy to the local viser viewers (HTTP + WebSocket)

Why a raw socket server instead of ``BaseHTTPRequestHandler``: the viser viewers
need their WebSocket upgrade relayed byte-for-byte after a prefix rewrite, and a
handler that has already consumed the request head cannot hand the socket over
cleanly.  Everything else is plain HTTP/1.1 with ``Connection: close``.
"""

from __future__ import annotations

import errno
import json
import os
import socket
import threading
import time
import traceback
from dataclasses import dataclass, field
from mimetypes import guess_type
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

from .control import Controller
from .proxy import read_head, relay, rewrite
from .snapshot import VIZ_ROUTES, Dashboard
from .system_health import SystemHealth
from booster_deploy.utils.mjcf_path import t2_mjcf_path
from booster_deploy.visualization.web.websocket import accept_value, encode_text_frame

MAX_BODY = 4096
STATUS_CACHE_S = 0.4
PAGE_CACHE_S = 30.0


def _port_holder(port: int) -> str:
    """Best-effort 'who owns this TCP port' string for a bind failure."""
    import subprocess

    try:
        out = subprocess.run(
            ["ss", "-tlnp", f"sport = :{port}"],
            capture_output=True, text=True, timeout=3, check=False,
        ).stdout
    except Exception:  # noqa: BLE001
        return "Unknown process"
    for line in out.splitlines()[1:]:
        if "users:" not in line:
            continue
        users = line.split("users:", 1)[1]
        entry = users.split("((", 1)[-1]
        name = entry.split('"')[1] if '"' in entry else entry
        pid = ""
        if "pid=" in entry:
            pid = entry.split("pid=", 1)[1].split(",", 1)[0]
        return f"{name} (pid {pid})" if pid else name
    return "Unknown process"


def _response(status: int, content_type: str, body: bytes, *,
              extra: tuple[tuple[str, str], ...] = (),
              cache_control: str = "no-store") -> bytes:
    """Single-shot HTTP/1.1 response.

    ``cache_control`` replaces the default instead of appending a second header:
    two Cache-Control values are merged by the client and ``no-store`` wins, which
    silently defeats caching of the (large, immutable) geometry payload.
    """
    reason = {200: "OK", 400: "Bad Request", 404: "Not Found", 405: "Method Not Allowed",
              413: "Payload Too Large", 500: "Internal Server Error"}.get(status, "OK")
    head = [
        f"HTTP/1.1 {status} {reason}",
        f"Content-Type: {content_type}",
        f"Content-Length: {len(body)}",
        f"Cache-Control: {cache_control}",
        "Connection: close",
    ]
    head.extend(f"{name}: {value}" for name, value in extra)
    return ("\r\n".join(head) + "\r\n\r\n").encode("latin-1") + body


@dataclass
class DashboardServer:
    repo_root: Path
    bind: str = "0.0.0.0"
    port: int = 8095
    session: str | None = None
    diag_dir: Path | None = None
    allow_control: bool = True
    language: str | None = None
    device_host: str | None = None
    interval: float = 1.0
    page_path: Path | None = None
    viz_ports: tuple[int, int, int] | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _preview_lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _status_cache: tuple[float, dict[str, Any]] | None = field(default=None, init=False)
    _page_cache: tuple[float, bytes] | None = field(default=None, init=False)
    _panes_cache: tuple[float, dict[str, str]] = field(default=(0.0, {}), init=False)
    _server: socket.socket | None = field(default=None, init=False)
    _stop: threading.Event = field(default_factory=threading.Event, init=False)
    webviz: Any = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.system_health = SystemHealth(self.device_host)
        if self.language not in (None, "zh", "en"):
            raise ValueError("language must be zh or en")
        self.dashboard = Dashboard(
            repo_root=Path(self.repo_root),
            session=self.session,
            diag_dir=self.diag_dir,
            allow_control=self.allow_control,
        )
        self.controller = Controller(
            repo_root=Path(self.repo_root),
            session_getter=lambda: self.dashboard.session_name,
            pane_lookup=self.pane_lookup,
            enabled=self.allow_control,
            option_getter=self.option,
            snapshot_getter=self.status,
        )
        if self.page_path is None:
            self.page_path = Path(__file__).resolve().parent / "static" / "page.html"
        if self.viz_ports is None:
            raw = os.getenv("BOOSTER_DASHBOARD_VIZ_PORTS", "")
            parts = [int(p) for p in raw.replace(",", " ").split() if p.strip().isdigit()]
            self.viz_ports = tuple(parts) if len(parts) == 3 else (8080, 8081, 8082)  # type: ignore[assignment]
        self.routes: dict[str, int] = {
            route: port for (_, route, _), port in zip(VIZ_ROUTES, self.viz_ports)
        }

    # ------------------------------------------------------------------ lookup
    def pane_lookup(self, role: str) -> str | None:
        reserved = self.option(f"@booster_{role}_pane")
        if reserved:
            panes = self.dashboard.tmux.panes(self.dashboard.session_name)
            return next((pane.pane_id for pane in panes if pane.pane_id == reserved), None)
        now = time.monotonic()
        stamp, mapping = self._panes_cache
        if now - stamp > 1.0:
            mapping = {}
            for pane in self.dashboard.tmux.panes(self.dashboard.session_name):
                if pane.role not in mapping and not pane.dead:
                    mapping[pane.role] = pane.pane_id
            self._panes_cache = (now, mapping)
        return mapping.get(role)

    # ------------------------------------------------------------------ payloads
    def option(self, name: str) -> str | None:
        session = self.dashboard.session_name
        if not session:
            return None
        return self.dashboard.tmux.option(session, name)

    def _shutdown_phase(self) -> dict:
        from .shutdown_state import read
        now = time.monotonic()
        cached = getattr(self, '_shutdown_cache', None)
        if cached and now-cached[0]<0.1:
            return cached[1]
        session = self.dashboard.session_name
        directory = getattr(self, '_shutdown_directory', None) or self.dashboard.ensure_diag_dir() or self.diag_dir or os.getenv('BOOSTER_TELEMETRY_DIR') or os.getenv('BOOSTER_DIAG_DIR') or Path(self.repo_root)/'logs'/str(session)
        self._shutdown_directory = directory
        phase = read(directory, session)
        self._shutdown_cache = (now, phase)
        if phase.get('phase') in {'draining', 'analyzing', 'closed'} and not getattr(self, '_writer_sealed', False):
            from booster_deploy.telemetry.runtime import close_process
            close_process(permanent=True)
            self._writer_sealed = True
        return phase

    def status(self) -> dict[str, Any]:
        shutdown = self._shutdown_phase()
        now = time.monotonic()
        with self._lock:
            if self._status_cache and now - self._status_cache[0] < STATUS_CACHE_S:
                return {**self._status_cache[1], "system_health": self.system_health.snapshot(), "shutdown": shutdown}
            snapshot = self.dashboard.snapshot()
            self.system_health.update(snapshot)
            self._status_cache = (now, snapshot)
            return {**snapshot, "system_health": self.system_health.snapshot(), "shutdown": shutdown}

    def trajectory_preview(self) -> dict[str, Any]:
        if not self._preview_lock.acquire(blocking=False):
            raise ValueError('Preview reader is busy; please try again later')
        try:
            return self._load_trajectory_preview()
        finally:
            self._preview_lock.release()

    def _load_trajectory_preview(self) -> dict[str, Any]:
        from .preview import load_preview

        bits = (self.option('@booster_session_meta') or '').split('/')
        if len(bits) != 4 or bits[3] != 'offline':
            raise ValueError('Recordings can be previewed only in offline sessions')
        current = self.option('@booster_recording')
        if not current:
            raise ValueError('No recording is currently active in this session')
        # No caller-supplied path: only the launcher's effective recording option.
        path = Path(current).expanduser()
        if not path.is_absolute():
            path = Path(self.repo_root) / path
        return load_preview(path.resolve(strict=True))

    def page(self) -> bytes:
        now = time.monotonic()
        with self._lock:
            if self._page_cache and now - self._page_cache[0] < PAGE_CACHE_S:
                return self._page_cache[1]
            body = Path(self.page_path).read_bytes() if self.page_path else b"dashboard page missing"
            if self.language is not None:
                body = body.replace(b'<script src="/webviz/i18n.js"></script>',
                                    ("<script>window.DASHBOARD_LANGUAGE=" + json.dumps(self.language) + ";</script>"
                                     + '<script src="/webviz/i18n.js"></script>').encode("utf-8"), 1)
            self._page_cache = (now, body)
            return body

    def _send_file(self, conn: socket.socket, root: Path, relative: str) -> bool:
        """Serve one file below root; traversal and directories are rejected."""
        root = root.resolve()
        candidate = (root / unquote(relative)).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            return False
        if not candidate.is_file():
            return False
        content_type = guess_type(candidate.name)[0] or "application/octet-stream"
        conn.sendall(_response(200, content_type, candidate.read_bytes()))
        return True

    # ------------------------------------------------------------------ routing
    def handle(self, conn: socket.socket) -> None:
        try:
            request = read_head(conn)
            if request is None:
                return
            path = request.target.split("?", 1)[0]
            if path == "/":
                conn.sendall(_response(200, "text/html; charset=utf-8", self.page()))
                return
            if path == "/healthz":
                conn.sendall(_response(200, "text/plain; charset=utf-8", b"ok"))
                return
            if path == '/api/shutdown-ack':
                from .shutdown_state import acknowledge
                phase = self._shutdown_phase()
                query = parse_qs(urlsplit(request.target).query)
                if phase.get('phase') not in {'closed', 'failed'} or query.get('id') != [phase.get('id')]:
                    conn.sendall(_response(400, 'application/json', b'{"ok":false}'))
                    return
                directory = self.dashboard.ensure_diag_dir() or self.diag_dir or os.getenv('BOOSTER_TELEMETRY_DIR') or os.getenv('BOOSTER_DIAG_DIR') or Path(self.repo_root)/'logs'/str(self.dashboard.session_name)
                acknowledge(directory, phase)
                conn.sendall(_response(200, 'application/json', b'{"ok":true}'))
                return
            if path == "/api/status":
                body = json.dumps(self.status(), separators=(",", ":")).encode("utf-8")
                conn.sendall(_response(200, "application/json; charset=utf-8", body))
                return
            if path == "/api/trajectory-preview":
                if request.method != 'GET' or urlsplit(request.target).query:
                    conn.sendall(_response(400, 'application/json', b'{"error":"use GET without parameters"}'))
                    return
                try:
                    payload = self.trajectory_preview()
                    status = 200
                except Exception as exc:
                    payload = {'error': f'unsupported / preview failed: {exc}'}
                    status = 400
                conn.sendall(_response(status, 'application/json; charset=utf-8',
                                       json.dumps(payload, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode('utf-8')))
                return
            if path == "/api/trajectories":
                body = json.dumps(self.controller.trajectories(), ensure_ascii=False).encode("utf-8")
                conn.sendall(_response(200, "application/json; charset=utf-8", body))
                return
            if path == "/api/visualization/model" and self.webviz is not None:
                body = json.dumps(self.webviz.manifest, separators=(",", ":")).encode("utf-8")
                conn.sendall(_response(200, "application/json; charset=utf-8", body))
                return
            if path == "/api/visualization/geometry" and self.webviz is not None:
                # One atomic payload: the robot must never arrive half-loaded.
                conn.sendall(_response(
                    200, "application/octet-stream", self.webviz.geometry(),
                    cache_control="public, max-age=86400, immutable",
                ))
                return
            if path == "/api/visualization/frame" and self.webviz is not None:
                body = json.dumps(self.webviz.snapshot(), separators=(",", ":")).encode("utf-8")
                conn.sendall(_response(200, "application/json; charset=utf-8", body))
                return
            if path == "/ws/visualization" and self.webviz is not None:
                key = request.headers.get("sec-websocket-key", "")
                if request.headers.get("upgrade", "").lower() != "websocket" or not key:
                    conn.sendall(_response(400, "text/plain", b"websocket upgrade required"))
                    return
                conn.sendall((
                    "HTTP/1.1 101 Switching Protocols\r\n"
                    "Upgrade: websocket\r\nConnection: Upgrade\r\n"
                    f"Sec-WebSocket-Accept: {accept_value(key)}\r\n\r\n"
                ).encode("ascii"))
                while not self._stop.is_set():
                    body = json.dumps(
                        self.webviz.snapshot(), separators=(",", ":")
                    ).encode("utf-8")
                    try:
                        conn.sendall(encode_text_frame(body))
                    except OSError:
                        return
                    self._stop.wait(0.04)
                return
            if path.startswith("/api/visualization/mesh/") and self.webviz is not None:
                try:
                    mesh_id = int(path.rsplit("/", 1)[-1])
                    mesh = self.webviz.mesh(mesh_id)
                except (ValueError, IndexError, AttributeError):
                    conn.sendall(_response(404, "application/json", b'{"error":"mesh not found"}'))
                    return
                conn.sendall(_response(200, "application/octet-stream", mesh))
                return
            if path == "/api/control":
                self._control(conn, request)
                return
            if path.startswith("/webviz/"):
                static_root = Path(__file__).resolve().parents[1] / "visualization/web/static"
                if self._send_file(conn, static_root, path[len("/webviz/"):]):
                    return
            mapped = rewrite(request.target, "/viz/", self.routes)
            if mapped is not None:
                from .proxy import rewrite_request

                port, new_target = mapped
                relay(conn, port, rewrite_request(request, new_target))
                return
            conn.sendall(_response(404, "application/json", b'{"error":"not found"}'))
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            try:
                conn.sendall(_response(500, "text/plain", b"internal error"))
            except OSError:
                pass
        finally:
            try:
                conn.close()
            except OSError:
                pass

    def _control(self, conn: socket.socket, request: Any) -> None:
        if request.method != "POST":
            conn.sendall(_response(405, "application/json", b'{"error":"use POST"}'))
            return
        try:
            length = int(request.headers.get("content-length", "0"))
        except ValueError:
            length = 0
        if not 0 < length <= MAX_BODY:
            conn.sendall(_response(400, "application/json", b'{"error":"bad length"}'))
            return
        body = request.rest
        while len(body) < length:
            chunk = conn.recv(length - len(body))
            if not chunk:
                break
            body += chunk
        try:
            payload = json.loads(body[:length].decode("utf-8"))
            action = str(payload["action"])
        except Exception:  # noqa: BLE001
            conn.sendall(_response(400, "application/json", b'{"error":"bad payload"}'))
            return
        if self._shutdown_phase().get('phase') in {'exiting', 'draining', 'analyzing', 'closed'}:
            conn.sendall(_response(409, 'application/json', b'{"ok":false,"error":"Shutdown already in progress"}'))
            return
        result = self.controller.dispatch(action, payload)
        status = 200 if result.get("ok") else 400
        conn.sendall(
            _response(status, "application/json; charset=utf-8",
                      json.dumps(result, ensure_ascii=False).encode("utf-8"))
        )

    # ------------------------------------------------------------------ serving
    def start(self) -> None:
        model_path = t2_mjcf_path(Path(self.repo_root))
        if self.webviz is None and model_path.is_file():
            try:
                from booster_deploy.visualization.web.hub import WebVisualizationHub

                session = self.status().get("session") or {}
                self.webviz = WebVisualizationHub(
                    Path(self.repo_root), source_kind=str(session.get("source") or "pico"),
                    execution=os.getenv("BOOSTER_TELEMETRY_EXECUTION") or session.get("mode"),
                )
                self.webviz.state_getter = self.status
                self.webviz.start()
            except Exception:  # visualization failure must not take down control/status
                traceback.print_exc()
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            server.bind((self.bind, self.port))
        except OSError as exc:
            server.close()
            if exc.errno != errno.EADDRINUSE:
                raise
            holder = _port_holder(self.port)
            raise SystemExit(
                f"dashboard cannot listen on {self.bind}:{self.port} — the port is already in use by {holder}.\n"
                f"  Common causes: a dashboard from the previous session is still running, or an SSH port forward was left behind"
                f" (ssh -L {self.port}:...).\n"
                f"  Resolution: kill <pid>, or restart with --port {self.port + 1}."
            ) from None
        server.listen(64)
        server.settimeout(0.5)
        self._server = server
        self.system_health.start()
        thread = threading.Thread(target=self._accept_loop, name="dashboard-accept", daemon=True)
        thread.start()
        # Prime caches so the first page load is not slower than the rest.
        try:
            self.status()
            self.page()
        except Exception:  # noqa: BLE001
            traceback.print_exc()

    def _accept_loop(self) -> None:
        assert self._server is not None
        while not self._stop.is_set():
            self._shutdown_phase()
            try:
                conn, _ = self._server.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            threading.Thread(target=self.handle, args=(conn,), name="dashboard-conn", daemon=True).start()

    def serve_forever(self) -> None:
        self.start()
        try:
            while not self._stop.is_set():
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def stop(self) -> None:
        self._stop.set()
        self.system_health.stop()
        if self.webviz is not None and hasattr(self.webviz, "stop"):
            self.webviz.stop()
        if self._server is not None:
            try:
                self._server.close()
            except OSError:
                pass


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Booster teleop dashboard (single port, read-mostly).")
    parser.add_argument("--bind", default=os.getenv("BOOSTER_DASHBOARD_BIND", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("BOOSTER_DASHBOARD_PORT", "8095")))
    parser.add_argument("--session", default=os.getenv("BOOSTER_DASHBOARD_SESSION") or None)
    parser.add_argument("--device-host", default=None,
                        help="Motion device IP; overrides verified PICO peer discovery; never browser/SSH/gateway")
    parser.add_argument("--diag-dir", default=None)
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--no-control", action="store_true", help="read-only: reject every control action")
    parser.add_argument(
        "--viz-ports", default=os.getenv("BOOSTER_DASHBOARD_VIZ_PORTS", ""),
        help="three ports for /viz/0|1|2 (human,robot,sim); default 8080,8081,8082",
    )
    parser.add_argument("--lang", choices=("zh", "en"), default=None,
                        help="UI language: zh or en (default: browser preference, then zh)")
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parents[2]
    server = DashboardServer(
        repo_root=repo_root,
        bind=args.bind,
        port=args.port,
        session=args.session,
        diag_dir=Path(args.diag_dir) if args.diag_dir else None,
        allow_control=not args.no_control,
        interval=args.interval,
        language=args.lang,
        device_host=args.device_host,
        viz_ports=tuple(int(p) for p in args.viz_ports.replace(",", " ").split())  # type: ignore[arg-type]
        if args.viz_ports.strip() else None,
    )
    server.start()
    session = server.dashboard.session_name or "(no session found)"
    ports = server.routes
    # 0.0.0.0 is a bind address, not something you can open in a browser.
    shown = "127.0.0.1" if args.bind in ("0.0.0.0", "::") else args.bind
    print(f"dashboard: http://{shown}:{args.port}/   session={session}  (listening on {args.bind})", flush=True)
    print(
        f"  viz: /viz/0 human :{ports['0']}   /viz/1 robot :{ports['1']}   /viz/2 sim :{ports['2']}",
        flush=True,
    )
    print(f"  control: {'enabled' if server.allow_control else 'DISABLED'}", flush=True)
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
