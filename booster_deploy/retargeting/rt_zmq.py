
from __future__ import annotations

import time

from booster_deploy.inputs.mocap.codecs import decode_mocap_retarget_input
from booster_deploy.inputs.pico.codecs import decode_pico_control, decode_pico_retarget_input
from booster_deploy.types.mocap_types import MocapRetargetInputFrameNs
from booster_deploy.types.pico_types import PicoControlFrameNs, PicoRetargetInputFrameNs

from .rt_codecs import decode_reference_50hz, encode_reference_50hz
from .rt_config import RealtimeRetargetZmqConfig
from booster_deploy.types.retarget_types import ReferenceFrame50HzNs


class PicoRetargetInputSubscriber:

    def __init__(self, config: RealtimeRetargetZmqConfig, *, recv_timeout_ms: int = 2000) -> None:

        import zmq

        self._zmq = zmq
        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.SUB)
        self._sock.setsockopt(zmq.RCVTIMEO, int(recv_timeout_ms))
        self._sock.setsockopt(zmq.RCVHWM, 1)
        self._sock.connect(f"tcp://{config.input_host}:{int(config.input_port)}")
        self._sock.subscribe(config.input_retarget_topic.encode("utf-8"))

    def recv_latest(self, timeout_s: float | None = None) -> PicoRetargetInputFrameNs:

        deadline = None if timeout_s is None else time.monotonic() + max(float(timeout_s), 0.0)
        while True:
            try:
                parts = self._sock.recv_multipart()
                while True:
                    try:
                        parts = self._sock.recv_multipart(flags=self._zmq.NOBLOCK)
                    except self._zmq.Again:
                        break
                if len(parts) == 1:
                    payload = parts[0].split(maxsplit=1)[-1]
                else:
                    payload = parts[-1]
                return decode_pico_retarget_input(payload)
            except self._zmq.Again:
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError(f"No Pico retarget input received within {timeout_s:.1f}s")

    def close(self) -> None:

        self._sock.close(linger=0)
        self._ctx.term()


class PicoControlSubscriber:

    def __init__(
        self,
        config: RealtimeRetargetZmqConfig,
        *,
        recv_timeout_ms: int = 50,
        recv_hwm: int = 1,
    ) -> None:

        import zmq

        self._zmq = zmq
        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.SUB)
        self._sock.setsockopt(zmq.RCVTIMEO, int(recv_timeout_ms))
        self._sock.setsockopt(zmq.RCVHWM, max(int(recv_hwm), 1))
        self._sock.connect(f"tcp://{config.input_host}:{int(config.input_port)}")
        self._sock.subscribe(config.input_control_topic.encode("utf-8"))

    def recv_latest(self, timeout_s: float | None = None) -> PicoControlFrameNs:

        deadline = None if timeout_s is None else time.monotonic() + max(float(timeout_s), 0.0)
        while True:
            try:
                parts = self._sock.recv_multipart()
                while True:
                    try:
                        parts = self._sock.recv_multipart(flags=self._zmq.NOBLOCK)
                    except self._zmq.Again:
                        break
                if len(parts) == 1:
                    payload = parts[0].split(maxsplit=1)[-1]
                else:
                    payload = parts[-1]
                return decode_pico_control(payload)
            except self._zmq.Again:
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError(f"No Pico control input received within {timeout_s:.1f}s")

    def close(self) -> None:

        self._sock.close(linger=0)
        self._ctx.term()


class RetargetResetPublisher:

    """Confirmed reset request (REQ/REP).

    This used to be a fire-and-forget PUB whose peer is a bound SUB.  On that
    topology ZMQ's slow-joiner rule drops the request: measured with these very
    classes, a single send right after ``connect()`` was lost 100% of the time and
    even 30 sends in a tight loop could miss, while the caller still reported
    success.  That is why the X / Reset retargeting button sometimes did nothing.

    REQ/REP queues the request until the worker is ready and returns an ack, so
    "sent" now means "the worker received it".
    """

    def __init__(self, host: str, port: int, topic: str) -> None:

        import zmq

        self._ctx = zmq.Context()
        self._host = host
        self._port = int(port)
        self._sock = self._new_socket()
        self._topic = topic.encode("utf-8")
        self._closed = False

    def _new_socket(self):
        sock = self._ctx.socket(self._zmq_module().REQ)
        sock.setsockopt(self._zmq_module().LINGER, 0)
        sock.setsockopt(self._zmq_module().REQ_RELAXED, 1)
        sock.setsockopt(self._zmq_module().RCVTIMEO, 400)
        sock.connect(f"tcp://{self._host}:{self._port}")
        return sock

    @staticmethod
    def _zmq_module():
        import zmq

        return zmq

    def publish_from_current(self, *, attempts: int = 3, timeout_s: float = 0.4) -> bool:
        """Ask the worker to re-seed its history; True when the worker acked."""
        if self._closed:
            raise RuntimeError("RetargetResetPublisher is closed")
        zmq = self._zmq_module()
        for _ in range(max(1, int(attempts))):
            try:
                self._sock.send_multipart([self._topic, b"reset_from_current"])
                if self._sock.poll(timeout=int(max(0.05, timeout_s) * 1000), flags=zmq.POLLIN):
                    self._sock.recv()
                    return True
            except Exception:  # noqa: BLE001 - retry with a fresh REQ socket
                pass
            # A timed-out REQ socket cannot send again; rebuild it for the retry.
            try:
                self._sock.close(linger=0)
            except Exception:  # noqa: BLE001
                pass
            self._sock = self._new_socket()
            time.sleep(0.05)
        return False

    def close(self) -> None:

        if self._closed:
            return
        self._closed = True
        self._sock.close(linger=0)
        self._ctx.term()


class RetargetResetSubscriber:

    """Worker side of the confirmed reset channel (REP, binds the port)."""

    def __init__(self, config: RealtimeRetargetZmqConfig, *, recv_timeout_ms: int = 100) -> None:

        import zmq

        self._zmq = zmq
        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.REP)
        self._sock.setsockopt(zmq.RCVTIMEO, int(recv_timeout_ms))
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.bind(f"tcp://{config.reset_bind}:{int(config.reset_port)}")

    def recv_latest(self, timeout_s: float | None = None) -> bytes:

        deadline = None if timeout_s is None else time.monotonic() + max(float(timeout_s), 0.0)
        while True:
            try:
                parts = self._sock.recv_multipart()
                payload = parts[-1] if len(parts) > 1 else parts[0].split(maxsplit=1)[-1]
                # Ack so the publisher knows the request was consumed, not queued.
                try:
                    self._sock.send(b"ack")
                except Exception:  # noqa: BLE001
                    pass
                return payload
            except self._zmq.Again:
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError("No retarget reset request received")

    def close(self) -> None:

        self._sock.close(linger=0)
        self._ctx.term()


class MocapRetargetInputSubscriber:

    def __init__(self, config: RealtimeRetargetZmqConfig, *, recv_timeout_ms: int = 2000) -> None:

        import zmq

        self._zmq = zmq
        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.SUB)
        self._sock.setsockopt(zmq.RCVTIMEO, int(recv_timeout_ms))
        self._sock.setsockopt(zmq.RCVHWM, 1)
        self._sock.connect(f"tcp://{config.input_host}:{int(config.input_port)}")
        self._sock.subscribe(config.input_retarget_topic.encode("utf-8"))

    def recv_latest(self, timeout_s: float | None = None) -> MocapRetargetInputFrameNs:

        deadline = None if timeout_s is None else time.monotonic() + max(float(timeout_s), 0.0)
        while True:
            try:
                parts = self._sock.recv_multipart()
                while True:
                    try:
                        parts = self._sock.recv_multipart(flags=self._zmq.NOBLOCK)
                    except self._zmq.Again:
                        break
                if len(parts) == 1:
                    payload = parts[0].split(maxsplit=1)[-1]
                else:
                    payload = parts[-1]
                return decode_mocap_retarget_input(payload)
            except self._zmq.Again:
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError(f"No Mocap retarget input received within {timeout_s:.1f}s")

    def close(self) -> None:

        self._sock.close(linger=0)
        self._ctx.term()


class Reference50HzPublisher:

    def __init__(self, config: RealtimeRetargetZmqConfig, *, send_hwm: int = 64) -> None:

        import zmq

        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.PUB)
        self._sock.setsockopt(zmq.SNDHWM, max(int(send_hwm), 1))
        self._sock.bind(f"tcp://{config.output_bind}:{int(config.output_port)}")
        self._topic = config.output_reference_topic.encode("utf-8")
        from booster_deploy.utils.diag_report import reporter
        self._diag = reporter('reference_publisher')

    def publish(self, frame: ReferenceFrame50HzNs) -> None:

        self._sock.send_multipart([self._topic, encode_reference_50hz(frame)])


    def close(self) -> None:

        self._sock.close(linger=0)
        self._ctx.term()
        self._diag.close()


class Reference50HzSubscriber:

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 5556,
        topic: str = "retarget_rt.reference_50hz",
        *,
        recv_hwm: int = 128,
        recv_timeout_ms: int = 2000,
    ) -> None:

        import zmq

        self._zmq = zmq
        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.SUB)
        self._sock.setsockopt(zmq.RCVTIMEO, int(recv_timeout_ms))
        self._sock.setsockopt(zmq.RCVHWM, max(int(recv_hwm), 1))
        self._sock.connect(f"tcp://{host}:{int(port)}")
        self._sock.subscribe(topic.encode("utf-8"))

    def recv_newer_than(
        self,
        last_sample_idx: int | None,
        timeout_s: float | None = None,
    ) -> list[ReferenceFrame50HzNs]:

        deadline = None if timeout_s is None else time.monotonic() + max(float(timeout_s), 0.0)
        while True:
            frames = self.drain_available()
            if last_sample_idx is None:
                if frames:
                    return frames
            else:
                newer = [frame for frame in frames if frame.sample_idx > last_sample_idx]
                if newer:
                    return newer
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(f"No reference frame received within {timeout_s:.1f}s")
            time.sleep(0.001)

    def drain_available(self) -> list[ReferenceFrame50HzNs]:

        frames: list[ReferenceFrame50HzNs] = []
        while True:
            try:
                parts = self._sock.recv_multipart(flags=self._zmq.NOBLOCK)
            except self._zmq.Again:
                break
            payload = parts[-1] if len(parts) > 1 else parts[0].split(maxsplit=1)[-1]
            frames.append(decode_reference_50hz(payload))
        if not frames:
            try:
                parts = self._sock.recv_multipart()
            except self._zmq.Again:
                return frames
            payload = parts[-1] if len(parts) > 1 else parts[0].split(maxsplit=1)[-1]
            frames.append(decode_reference_50hz(payload))
        return frames

    def close(self) -> None:

        self._sock.close(linger=0)
        self._ctx.term()
