from __future__ import annotations

import hmac
import json
import logging
import queue
import threading
import time
from collections import deque
from collections.abc import Callable
from typing import Any, ClassVar

import zmq

logger = logging.getLogger(__name__)


class T2WebControlServer:
    ACTIONS: ClassVar[frozenset[str]] = frozenset(
        {"walking", "start_vla", "pause", "resume", "shutdown"}
    )

    def __init__(
        self,
        *,
        bind: str,
        port: int,
        token: str,
        state_provider: Callable[[], str],
        recording_provider: Callable[[], dict[str, Any]] | None = None,
    ) -> None:
        if not token:
            raise ValueError("T2 web control requires a non-empty authentication token")
        self.endpoint = f"tcp://{bind}:{int(port)}"
        self._token = token.encode("utf-8")
        self._state_provider = state_provider
        self._recording_provider = recording_provider
        self._queue: queue.Queue[dict[str, str]] = queue.Queue(maxsize=32)
        self._lock = threading.Lock()
        self._commands: dict[str, dict[str, Any]] = {}
        self._command_order: deque[str] = deque()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._serve, name="t2-web-control-server", daemon=True
        )
        self._thread.start()
        logger.info("T2 web state-machine control listening on %s", self.endpoint)

    def take_pending(self) -> dict[str, str] | None:
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None

    def update_command(
        self, command_id: str, *, status: str, message: str, state: str
    ) -> None:
        with self._lock:
            command = self._commands.get(command_id)
            if command is None:
                return
            command.update(
                status=status,
                message=message,
                state=state,
                updated_wall_ns=time.time_ns(),
            )

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        with self._lock:
            for command in self._commands.values():
                if command["status"] in {"queued", "running"}:
                    command.update(
                        status="failed",
                        message="robot deployment is shutting down",
                        updated_wall_ns=time.time_ns(),
                    )

    def _snapshot(self) -> dict[str, Any]:
        with self._lock:
            commands = [dict(self._commands[key]) for key in self._command_order]
        snapshot = {"state": self._state_provider(), "commands": commands}
        if self._recording_provider is not None:
            snapshot["recording"] = self._recording_provider()
        return snapshot

    def _remember(self, command: dict[str, Any]) -> None:
        command_id = command["command_id"]
        self._commands[command_id] = command
        self._command_order.append(command_id)
        while len(self._command_order) > 256:
            expired = self._command_order.popleft()
            self._commands.pop(expired, None)

    def _handle_request(self, request: Any) -> dict[str, Any]:
        if not isinstance(request, dict):
            return {"ok": False, "error": "request must be a JSON object"}
        supplied_token = request.get("token")
        if not isinstance(supplied_token, str) or not hmac.compare_digest(
            supplied_token.encode("utf-8"), self._token
        ):
            return {"ok": False, "error": "unauthorized"}
        operation = request.get("op")
        if operation == "status":
            return {"ok": True, **self._snapshot()}
        if operation == "command_status":
            command_id = request.get("command_id")
            with self._lock:
                command = self._commands.get(command_id) if isinstance(command_id, str) else None
                result = None if command is None else dict(command)
            return {"ok": True, "command": result, "state": self._state_provider()}
        if operation != "command":
            return {"ok": False, "error": "unsupported operation"}

        command_id = request.get("command_id")
        action = request.get("action")
        if not isinstance(command_id, str) or not 1 <= len(command_id) <= 64:
            return {"ok": False, "error": "command_id must contain 1-64 characters"}
        if action not in self.ACTIONS:
            return {"ok": False, "error": "unsupported state-machine action"}
        with self._lock:
            existing = self._commands.get(command_id)
            if existing is not None:
                if existing["action"] != action:
                    return {"ok": False, "error": "command_id was already used for another action"}
                return {"ok": True, "command": dict(existing)}
            command = {
                "command_id": command_id,
                "action": action,
                "status": "queued",
                "message": "command queued for the robot control loop",
                "state": self._state_provider(),
                "created_wall_ns": time.time_ns(),
                "updated_wall_ns": time.time_ns(),
            }
            try:
                self._queue.put_nowait({"command_id": command_id, "action": action})
            except queue.Full:
                return {"ok": False, "error": "robot command queue is full"}
            self._remember(command)
        return {"ok": True, "command": dict(command)}

    def _serve(self) -> None:
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.RCVTIMEO, 200)
        socket.setsockopt(zmq.SNDTIMEO, 1000)
        try:
            socket.bind(self.endpoint)
            while not self._stop.is_set():
                try:
                    request = socket.recv_json()
                except zmq.Again:
                    continue
                except (ValueError, json.JSONDecodeError):
                    response = {"ok": False, "error": "invalid JSON request"}
                else:
                    try:
                        response = self._handle_request(request)
                    except Exception:
                        logger.exception("T2 web control request failed")
                        response = {"ok": False, "error": "internal control service error"}
                try:
                    socket.send_json(response)
                except zmq.ZMQError:
                    if not self._stop.is_set():
                        logger.exception("Failed to send T2 web control response")
        except zmq.ZMQError:
            if not self._stop.is_set():
                logger.exception("T2 web control service stopped unexpectedly")
        finally:
            socket.close(linger=0)
            context.term()
