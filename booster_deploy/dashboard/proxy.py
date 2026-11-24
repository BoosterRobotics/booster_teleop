"""Path-prefix reverse proxy used to fold every viser viewer onto one port.

viser 1.0.30 serves its HTML **and** its WebSocket on the same port at ``/``
(verified: ``GET /`` -> 200, ``GET /`` + ``Upgrade`` -> 101).  Therefore a single
listener can host several viewers only if the prefix is stripped before the
bytes reach the upstream socket -- which is exactly what this module does: it
parses nothing beyond the request head, then relays bytes in both directions.

Because the relay is protocol-agnostic, viser upgrades will not break it.
"""

from __future__ import annotations

import select
import socket
from dataclasses import dataclass

MAX_HEAD_BYTES = 64 * 1024


@dataclass
class Request:
    method: str
    target: str
    version: str
    headers: dict[str, str]
    head_lines: list[bytes]
    rest: bytes


def read_head(conn: socket.socket, max_bytes: int = MAX_HEAD_BYTES) -> Request | None:
    """Read one HTTP request head.  Returns None on EOF/oversize."""
    buf = b""
    while b"\r\n\r\n" not in buf:
        try:
            chunk = conn.recv(16384)
        except OSError:
            return None
        if not chunk:
            return None
        buf += chunk
        if len(buf) > max_bytes:
            return None
    head, rest = buf.split(b"\r\n\r\n", 1)
    lines = head.split(b"\r\n")
    try:
        method, target, version = lines[0].split(b" ")
    except ValueError:
        return None
    headers: dict[str, str] = {}
    for line in lines[1:]:
        if b":" not in line:
            continue
        name, _, value = line.partition(b":")
        headers[name.decode("latin-1").strip().lower()] = value.decode("latin-1").strip()
    return Request(
        method=method.decode("latin-1"),
        target=target.decode("latin-1"),
        version=version.decode("latin-1"),
        headers=headers,
        head_lines=lines,
        rest=rest,
    )


def rewrite(target: str, prefix: str, routes: dict[str, int]) -> tuple[int, str] | None:
    """Map ``<prefix><name>/...`` onto ``(upstream_port, "/...")``."""
    if not target.startswith(prefix):
        return None
    remainder = target[len(prefix) :]
    name, _, tail = remainder.partition("/")
    port = routes.get(name)
    if port is None:
        return None
    return port, "/" + tail if tail else "/"


def relay(conn: socket.socket, port: int, request: Request, *, timeout: float = 120.0) -> None:
    """Forward the (rewritten) request to 127.0.0.1:port and pump bytes both ways."""
    try:
        upstream = socket.create_connection(("127.0.0.1", port), 5)
    except OSError:
        # The viewer is usually still importing mujoco/viser for a few seconds while
        # the dashboard is already up.  Return a self-refreshing page instead of a
        # dead-end 502, otherwise the embedded iframe stays broken until a reload.
        body = (
            "<!doctype html><meta charset=utf-8>"
            "<meta http-equiv=refresh content=3>"
            "<style>html,body{height:100%;margin:0;background:#05070a;color:#8b9bb0;"
            "font:13px ui-sans-serif,system-ui;display:grid;place-items:center;text-align:center}"
            "</style>"
            f"<div>View :{port} is still starting… (retrying automatically every 3 seconds)<br>"
            "<span style='font-size:11px;opacity:.7'>viz upstream unavailable</span></div>"
        ).encode("utf-8")
        conn.sendall(
            b"HTTP/1.1 502 Bad Gateway\r\nContent-Type: text/html; charset=utf-8\r\n"
            + f"Content-Length: {len(body)}\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n".encode()
            + body
        )
        return
    try:
        upstream.sendall(_head_bytes(request))
        if request.rest:
            upstream.sendall(request.rest)
        sockets = [conn, upstream]
        while True:
            readable, _, errored = select.select(sockets, [], sockets, timeout)
            if not readable and not errored:
                return
            for current in readable:
                try:
                    data = current.recv(262144)
                except OSError:
                    return
                if not data:
                    return
                other = upstream if current is conn else conn
                try:
                    other.sendall(data)
                except OSError:
                    return
    finally:
        try:
            upstream.close()
        except OSError:
            pass


def _head_bytes(request: Request) -> bytes:
    """Rebuild the head with only the request line rewritten."""
    first = request.head_lines[0]
    method = request.method.encode("latin-1")
    # keep the query string; the upstream viewer expects it (e.g. ?websocket=…)
    path = request.target.encode("latin-1")
    return b" ".join([method, path, request.version.encode("latin-1")]) + b"\r\n" + b"\r\n".join(
        request.head_lines[1:]
    ) + b"\r\n\r\n"


def rewrite_request(request: Request, new_target: str) -> Request:
    lines = list(request.head_lines)
    lines[0] = b" ".join(
        [request.method.encode("latin-1"), new_target.encode("latin-1"), request.version.encode("latin-1")]
    )
    return Request(
        method=request.method,
        target=new_target,
        version=request.version,
        headers=request.headers,
        head_lines=lines,
        rest=request.rest,
    )
