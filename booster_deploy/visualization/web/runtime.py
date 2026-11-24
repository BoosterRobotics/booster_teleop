"""Latest-only simulation snapshot publisher compatible with MujocoController."""
from __future__ import annotations

import queue
import threading
import time
from typing import Callable

import numpy as np

from booster_deploy.utils.async_viewer import ViewerSnapshot
from .protocol import TOPIC, encode_sim_snapshot


class WebSnapshotRuntime:
    """ViewerRuntime-shaped PUB adapter; never waits for a browser."""

    def __init__(self, *, model, endpoint: str) -> None:
        self._model = model
        self._endpoint = str(endpoint)
        self._queue: queue.Queue[ViewerSnapshot] = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._running = threading.Event()
        self._seq = 0
        self._zero_forces = np.zeros((int(model.nbody), 6), dtype=np.float64)

    def run(self, *, on_started: Callable[[], None] | None = None) -> None:
        import zmq

        ctx = zmq.Context()
        sock = ctx.socket(zmq.PUB)
        sock.setsockopt(zmq.SNDHWM, 1)
        sock.setsockopt(zmq.LINGER, 0)
        sock.bind(self._endpoint)
        self._running.set()
        if on_started is not None:
            on_started()
        try:
            while not self._stop.is_set():
                try:
                    snapshot = self._queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                while True:
                    try:
                        snapshot = self._queue.get_nowait()
                    except queue.Empty:
                        break
                self._seq += 1
                payload = encode_sim_snapshot(
                    seq=self._seq,
                    sample_ns=time.monotonic_ns(),
                    qpos=snapshot.qpos,
                    reference_qpos=snapshot.reference_qpos,
                )
                sock.send_multipart([TOPIC, payload])
        finally:
            self._running.clear()
            sock.close(linger=0)
            ctx.term()

    def publish(self, snapshot: ViewerSnapshot) -> None:
        try:
            self._queue.put_nowait(snapshot)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(snapshot)
            except queue.Full:
                pass

    def is_running(self) -> bool:
        return self._running.is_set() and not self._stop.is_set()

    def drain_key_events(self, _handler) -> None:
        return

    def latest_xfrc_applied(self) -> np.ndarray:
        return self._zero_forces.copy()

    def close(self) -> None:
        self._stop.set()


class DirectWebSnapshotPublisher:
    """Non-blocking latest-value publisher for the real-robot control loop."""

    def __init__(self, endpoint: str) -> None:
        import zmq

        self._zmq = zmq
        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.PUB)
        self._sock.setsockopt(zmq.SNDHWM, 1)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.bind(str(endpoint))
        self._seq = 0

    def publish(self, qpos: np.ndarray) -> None:
        self._seq += 1
        payload = encode_sim_snapshot(
            seq=self._seq, sample_ns=time.monotonic_ns(), qpos=qpos
        )
        try:
            self._sock.send_multipart([TOPIC, payload], flags=self._zmq.NOBLOCK)
        except self._zmq.Again:
            pass

    def close(self) -> None:
        self._sock.close(linger=0)
        self._ctx.term()
