"""Read-only HTTP service for synchronized telemetry replay."""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
from urllib.parse import parse_qs, unquote, urlsplit

from .replay import ReplayRun


REPO_ROOT = Path(__file__).resolve().parents[2]
STATIC_ROOT = Path(__file__).with_name("static")
WEBVIZ_ROOT = REPO_ROOT / "booster_deploy/visualization/web/static"
DEFAULT_MJCF = REPO_ROOT / "booster_deploy/assets/t2/t2_31dof_fc.xml"


def _json(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


class ReplayService:
    def __init__(self, directory, *, execution=None, mjcf=None, generated=None):
        self.run = ReplayRun(
            directory, execution=execution, mjcf=mjcf, generated=generated)
        if self.run.mjcf is None:
            self.run.mjcf = DEFAULT_MJCF
        self._geometry = None
        # MuJoCo forward kinematics shares one mjData; concurrent HTTP threads segfault
        # without this lock, so every model-touching call is serialised.
        self._lock = threading.Lock()

    def frame(self, seconds):
        with self._lock:
            return self.run.snapshot(seconds)

    def manifest(self):
        value = self.run.describe()
        value["model_available"] = bool(self.run.mjcf)
        labels = {
            "human": ("原始人体", "Raw human"),
            "retarget": ("重定向", "Retargeted"),
            "sim": ("仿真执行", "Simulation"),
            "real": ("真机反馈", "Robot feedback"),
        }
        value["tracks"] = [
            {
                "id": stream,
                "kind": "human" if stream == "human" else "robot",
                "label_zh": labels[stream][0],
                "label_en": labels[stream][1],
            }
            for stream in self.run.streams
        ]
        value["camera"] = {
            "mode": "robot_local_front",
            "forward_axis": "+X",
            "shared_controls": True,
        }
        return value

    def geometry(self):
        if self._geometry is None:
            with self._lock:
                if self._geometry is None:
                    self._geometry = self.run.geometry()
        return self._geometry


class ReplayHandler(BaseHTTPRequestHandler):
    server_version = "BoosterTelemetryReplay/1"

    def log_message(self, fmt, *args):
        return

    @property
    def service(self) -> ReplayService:
        return self.server.service

    def do_HEAD(self):
        self._handle(head=True)

    def do_GET(self):
        self._handle(head=False)

    def do_POST(self):
        self._send(405, b'{"error":"read-only service"}', "application/json", head=False)

    def _send(self, status, body, content_type, *, head):
        try:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            if not head:
                self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            # Seeking cancels stale frame requests; a disconnected client is not a replay failure.
            return

    def _file(self, root: Path, relative: str, *, head: bool):
        root = root.resolve()
        candidate = (root / unquote(relative)).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            self._send(404, b"not found", "text/plain", head=head)
            return
        if not candidate.is_file():
            self._send(404, b"not found", "text/plain", head=head)
            return
        types = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
                 ".css": "text/css; charset=utf-8", ".json": "application/json"}
        self._send(200, candidate.read_bytes(), types.get(candidate.suffix, "application/octet-stream"), head=head)

    def _handle(self, *, head: bool):
        target = urlsplit(self.path)
        path = target.path
        try:
            if path == "/":
                self._file(STATIC_ROOT, "replay.html", head=head)
            elif path in ("/replay.js", "/replay.css"):
                self._file(STATIC_ROOT, path[1:], head=head)
            elif path.startswith("/webviz/"):
                self._file(WEBVIZ_ROOT, path[len("/webviz/"):], head=head)
            elif path == "/api/replay/manifest":
                self._send(200, _json(self.service.manifest()), "application/json; charset=utf-8", head=head)
            elif path == "/api/replay/frame":
                query = parse_qs(target.query)
                seconds = float(query.get("time_s", query.get("t", ["0"]))[0])
                self._send(200, _json(self.service.frame(seconds)), "application/json; charset=utf-8", head=head)
            elif path == "/api/visualization/model":
                self._send(200, _json(self.service.run.model_manifest()), "application/json; charset=utf-8", head=head)
            elif path == "/api/visualization/geometry":
                self._send(200, self.service.geometry(), "application/octet-stream", head=head)
            elif path == "/healthz":
                self._send(200, b"ok", "text/plain", head=head)
            else:
                self._send(404, b'{"error":"not found"}', "application/json", head=head)
        except (ValueError, KeyError, TypeError) as exc:
            self._send(400, _json({"error": str(exc)}), "application/json; charset=utf-8", head=head)
        except Exception as exc:  # keep a replay failure visible without exposing a traceback
            self._send(500, _json({"error": f"{type(exc).__name__}: {exc}"}),
                       "application/json; charset=utf-8", head=head)


class ReplayHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, service):
        self.service = service
        super().__init__(address, ReplayHandler)


def serve(directory, *, bind="127.0.0.1", port=8096, execution=None, mjcf=None, generated=None):
    service = ReplayService(directory, execution=execution, mjcf=mjcf, generated=generated)
    server = ReplayHTTPServer((bind, int(port)), service)
    return server
