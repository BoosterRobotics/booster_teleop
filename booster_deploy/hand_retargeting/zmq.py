
from __future__ import annotations

import time

from booster_deploy.hand_retargeting.codecs import decode_hand_retarget_frame, encode_hand_retarget_frame
from booster_deploy.types.hand_retarget_types import HandRetargetFrameNs


class HandRetargetFramePublisher:

    def __init__(self, host: str, port: int, topic: str, *, send_hwm: int = 8) -> None:
        import zmq

        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.PUB)
        self._sock.setsockopt(zmq.SNDHWM, max(int(send_hwm), 1))
        self._sock.bind(f"tcp://{host}:{int(port)}")
        self._topic = topic.encode("utf-8")

    def publish(self, frame: HandRetargetFrameNs) -> None:
        self._sock.send_multipart([self._topic, encode_hand_retarget_frame(frame)])

    def close(self) -> None:
        self._sock.close(linger=0)
        self._ctx.term()


class HandRetargetFrameSubscriber:

    def __init__(self, host: str, port: int, topic: str, *, recv_timeout_ms: int = 0) -> None:
        import zmq

        self._zmq = zmq
        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.SUB)
        self._sock.setsockopt(zmq.RCVTIMEO, int(recv_timeout_ms))
        self._sock.setsockopt(zmq.RCVHWM, 1)
        self._sock.connect(f"tcp://{host}:{int(port)}")
        self._sock.subscribe(topic.encode("utf-8"))

    def _recv_payload(self) -> bytes:
        parts = self._sock.recv_multipart()
        while True:
            try:
                parts = self._sock.recv_multipart(flags=self._zmq.NOBLOCK)
            except self._zmq.Again:
                break
        if len(parts) == 1:
            return parts[0].split(maxsplit=1)[-1]
        return parts[-1]

    def try_recv_latest(self) -> HandRetargetFrameNs | None:
        try:
            return decode_hand_retarget_frame(self._recv_payload())
        except self._zmq.Again:
            return None

    def recv_latest(self, timeout_s: float | None = None) -> HandRetargetFrameNs:
        if timeout_s is not None and timeout_s <= 0.0:
            frame = self.try_recv_latest()
            if frame is None:
                raise TimeoutError("No hand retarget input available")
            return frame
        deadline = None if timeout_s is None else time.monotonic() + max(float(timeout_s), 0.0)
        while True:
            frame = self.try_recv_latest()
            if frame is not None:
                return frame
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(f"No hand retarget input received within {timeout_s:.1f}s")
            time.sleep(0.001)

    def close(self) -> None:
        self._sock.close(linger=0)
        self._ctx.term()


__all__ = [
    "HandRetargetFramePublisher",
    "HandRetargetFrameSubscriber",
]
