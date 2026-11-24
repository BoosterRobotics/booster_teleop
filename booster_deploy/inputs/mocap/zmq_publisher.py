
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

from .capture import MocapRtCapture, MocapRtCaptureConfig
from .codecs import encode_mocap_retarget_input


@dataclass(frozen=True)
class MocapRtZmqPublisherConfig:

    bind: str = "0.0.0.0"
    port: int = 5557
    retarget_topic: str = "mocap_rt.retarget_input"
    provider_timeout_s: float = 30.0
    send_hwm: int = 1
    print_interval: int = 300
    capture: MocapRtCaptureConfig | None = None
    record_dir: Path | None = None


class MocapRtZmqPublisher:

    def __init__(self, config: MocapRtZmqPublisherConfig | None = None) -> None:

        import zmq

        self.config = config or MocapRtZmqPublisherConfig()
        self._zmq = zmq
        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.PUB)
        self._sock.setsockopt(zmq.SNDHWM, max(int(self.config.send_hwm), 1))
        self._sock.bind(f"tcp://{self.config.bind}:{int(self.config.port)}")
        self._retarget_topic = self.config.retarget_topic.encode("utf-8")

        capture_cfg = self.config.capture or MocapRtCaptureConfig()
        if capture_cfg.timeout_s != float(self.config.provider_timeout_s):
            capture_cfg = MocapRtCaptureConfig(
                human_format=capture_cfg.human_format,
                local_ip=capture_cfg.local_ip,
                local_port=capture_cfg.local_port,
                server_ip=capture_cfg.server_ip,
                server_port=capture_cfg.server_port,
                poll_hz=capture_cfg.poll_hz,
                timeout_s=float(self.config.provider_timeout_s),
                dedupe_source_ns=capture_cfg.dedupe_source_ns,
                max_cached_frames=capture_cfg.max_cached_frames,
                synthetic_tip_quat_mode=capture_cfg.synthetic_tip_quat_mode,
            )
        self._capture = MocapRtCapture(capture_cfg)
        self._closed = False
        self._recorder = None
        if self.config.record_dir is not None:
            from .live_recording import MocapLiveRecorder

            self._recorder = MocapLiveRecorder(self.config.record_dir, self._capture.latest_body)
        from booster_deploy.utils.diag_report import reporter
        self._diag = reporter('mocap_publisher')
        from booster_deploy.telemetry.publish import Source
        self._source = Source('online_mocap')

    def publish(self, body):
        from dataclasses import replace
        identity=self._source.next(native_seq=body.body_seq)
        body=replace(body,source_ids=(identity,))
        self._sock.send_multipart([self._retarget_topic, encode_mocap_retarget_input(body)])
        self._source.published(identity,channel='retarget_input',human_frame=body.human_frame,
            body_recv_ns=body.body_recv_ns,source_ns=body.source_ns,body_unique_known=body.body_unique_known)
        return body

    def run_forever(self) -> None:

        last_body_seq: int | None = None
        published_body = 0
        start_ns = time.monotonic_ns()
        last_wait_print_ns = 0
        interrupted = False
        try:
            if self._recorder is not None:
                self._recorder.start()
            self._capture.start()
            while not self._closed:
                try:
                    body = self._capture.wait_next_retarget_input(
                        last_body_seq,
                        timeout_s=0.02,
                    )
                    self.publish(body)
                    last_body_seq = body.body_seq
                    published_body += 1
                except TimeoutError:
                    now_ns = time.monotonic_ns()
                    if published_body <= 0 and now_ns - last_wait_print_ns >= 1_000_000_000:
                        last_wait_print_ns = now_ns
                        print(
                            "Waiting for Mocap RT body data | "
                            f"body_seq={self._capture.body_seq} "
                            f"callback_count={self._capture.callback_count}",
                            flush=True,
                        )

                if (
                    published_body > 0
                    and self.config.print_interval > 0
                    and published_body % int(self.config.print_interval) == 0
                ):
                    elapsed_s = max((time.monotonic_ns() - start_ns) / 1e9, 1e-9)
                    print(
                        f"MOCAP_RT published body={published_body} "
                        f"({published_body / elapsed_s:.1f} fps) "
                        f"last_body_seq={last_body_seq}",
                        flush=True,
                    )
        except KeyboardInterrupt:
            interrupted = True
            print("Stopping Mocap RT publisher...", flush=True)
        finally:
            self.close(fast=interrupted)

    def close(self, *, fast: bool = False) -> None:

        if self._closed:
            return
        self._closed = True
        try:
            if self._recorder is not None:
                self._recorder.close()
        finally:
            try:
                self._capture.close(close_sdk=not fast)
            finally:
                self._sock.close(linger=0)
                self._ctx.term()
                self._diag.close()


__all__ = [
    "MocapRtZmqPublisher",
    "MocapRtZmqPublisherConfig",
]
