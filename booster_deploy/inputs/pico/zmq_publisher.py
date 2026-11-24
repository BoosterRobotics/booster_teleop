
from __future__ import annotations

from dataclasses import asdict, dataclass
import time
import queue
import logging
import threading
import uuid

from .capture import PicoRtCapture, PicoRtCaptureConfig
from .codecs import encode_pico_control, encode_pico_retarget_input
from booster_deploy.types.pico_types import PicoControlFrameNs
from .recording import PicoRecordingControlRouter
from .coordinate import BODY_JOINT_NAMES, INPUT_TO_DEPLOY_MATRIX, INPUT_TO_DEPLOY_QUAT
from .raw_stream import PICO_RAW_TOPIC, encode_pico_raw
from booster_deploy.utils.diag_report import below_floor, format_gap, reporter

logger = logging.getLogger(__name__)

_RAW_QUEUE_FRAMES = 32
_RAW_MAX_AGE_NS = 500_000_000
_RAW_SEND_HWM = 1024


@dataclass(frozen=True)
class PicoRtZmqPublisherConfig:

    bind: str = "0.0.0.0"
    port: int = 5555
    retarget_topic: str = "pico_rt.retarget_input"
    control_topic: str = "pico_rt.control"
    provider_timeout_s: float = 30.0
    send_hwm: int = 1
    control_send_hwm: int = 16
    print_interval: int = 300
    input_mode: str = "auto"
    poll_hz: float = 50.0
    # Mask trigger+X/Y on the robot control topic; the recorder uses raw controls.
    control_chord_mask: bool = False
    raw_bind: str | None = None


class PicoControlZmqPublisher:

    def __init__(
        self,
        *,
        bind: str = "0.0.0.0",
        port: int = 5555,
        topic: str = "pico_rt.control",
        send_hwm: int = 16,
    ) -> None:
        import zmq

        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.PUB)
        self._sock.setsockopt(zmq.SNDHWM, max(int(send_hwm), 1))
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.bind(f"tcp://{bind}:{int(port)}")
        self._topic = topic.encode("utf-8")
        self._closed = False
        self.send_again = 0

    def publish(self, frame: PicoControlFrameNs) -> bool:

        if self._closed:
            raise RuntimeError("PicoControlZmqPublisher is closed")
        import zmq
        try:
            self._sock.send_multipart([self._topic, encode_pico_control(frame)], flags=zmq.NOBLOCK)
            return True
        except zmq.Again:
            self.send_again += 1
            return False

    def close(self) -> None:

        if self._closed:
            return
        self._closed = True
        self._sock.close(linger=0)
        self._ctx.term()


class PicoRtZmqPublisher:

    def __init__(self, config: PicoRtZmqPublisherConfig | None = None) -> None:

        import zmq

        self.config = config or PicoRtZmqPublisherConfig()
        self._zmq = zmq
        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.PUB)
        self._sock.setsockopt(zmq.SNDHWM, max(int(self.config.send_hwm), int(self.config.control_send_hwm), 1))
        self._sock.bind(f"tcp://{self.config.bind}:{int(self.config.port)}")
        self._retarget_topic = self.config.retarget_topic.encode("utf-8")
        self._control_topic = self.config.control_topic.encode("utf-8")
        self._recording_router = PicoRecordingControlRouter() if self.config.control_chord_mask else None
        self._diag = reporter("publisher")
        self._seq_gap_events = 0
        self._raw_socket = None
        self._raw_stream_id = uuid.uuid4().hex
        from booster_deploy.telemetry.publish import Source
        self._source = Source("online_pico")
        self._raw_seq = 0
        self._raw_send_drops = 0
        self._raw_metadata_ns = 0
        self._raw_controls = queue.Queue(maxsize=_RAW_QUEUE_FRAMES)
        self._raw_queue_drops = 0
        self._raw_stale_drops = 0
        self._raw_shutdown_drops = 0
        self._raw_last_metadata: dict[str, object] | None = None
        self._raw_again = 0
        self._raw_encode_errors = 0
        self._raw_error: str | None = None
        self._online_send_again = 0
        self._raw_stop = threading.Event()
        self._raw_ready = threading.Event()
        self._raw_thread: threading.Thread | None = None
        self._capture = PicoRtCapture(
            PicoRtCaptureConfig(
                timeout_s=float(self.config.provider_timeout_s),
                input_mode=str(self.config.input_mode),
                poll_hz=float(self.config.poll_hz),
                raw_recording=bool(self.config.raw_bind),
            )
        )
        if self.config.raw_bind:
            self._capture.raw_frames = queue.Queue(maxsize=_RAW_QUEUE_FRAMES)
            self._capture.raw_snapshots = queue.Queue(maxsize=_RAW_QUEUE_FRAMES)
        self._closed = False

    def _send_online(self, topic: bytes, payload: bytes) -> bool:
        try:
            self._sock.send_multipart([topic, payload], flags=self._zmq.NOBLOCK)
            return True
        except self._zmq.Again:
            self._online_send_again += 1
            return False

    def _enqueue_raw_control(self, raw: PicoControlFrameNs, routed: PicoControlFrameNs) -> None:
        if not self.config.raw_bind:
            return
        try:
            self._raw_controls.put_nowait((raw, routed))
            self._capture.raw_activity_event.set()
        except queue.Full:
            self._raw_queue_drops += 1

    def _start_raw_worker(self) -> None:
        if not self.config.raw_bind or self._raw_thread is not None:
            return
        self._raw_thread = threading.Thread(target=self._raw_loop, name="pico-raw-publisher", daemon=True)
        self._raw_thread.start()
        if not self._raw_ready.wait(2.0):
            raise RuntimeError("PICO raw publisher startup timed out")
        if self._raw_error is not None:
            raise RuntimeError(self._raw_error)

    def _raw_loop(self) -> None:
        ctx = self._zmq.Context()
        try:
            self._raw_socket = ctx.socket(self._zmq.XPUB)
            self._raw_socket.setsockopt(self._zmq.SNDHWM, _RAW_SEND_HWM)
            self._raw_socket.setsockopt(self._zmq.XPUB_NODROP, 1)
            self._raw_socket.setsockopt(self._zmq.LINGER, 0)
            self._raw_socket.bind(self.config.raw_bind)
            self._raw_ready.set()
            while not self._raw_stop.is_set():
                self._capture.raw_activity_event.clear()
                self._drain_raw()
                try:
                    raw, routed = self._raw_controls.get_nowait()
                except queue.Empty:
                    pass
                else:
                    self._publish_raw("control", raw_control=_control_values(raw), routed_control=_control_values(routed))
                if all(q.empty() for q in (
                    self._capture.raw_frames, self._capture.raw_snapshots, self._raw_controls,
                )):
                    self._capture.raw_activity_event.wait(0.02)
        except Exception as exc:
            self._raw_error = str(exc)
            logger.exception("PICO raw publisher failed")
        finally:
            self._raw_ready.set()
            try:
                self._raw_shutdown_drops = sum(q.qsize() for q in (
                    self._capture.raw_frames, self._capture.raw_snapshots, self._raw_controls,
                ))
                if self._raw_error is None and self._raw_last_metadata is not None:
                    self._publish_raw("metadata", **{
                        **self._raw_last_metadata,
                        "publisher_wall_ns": time.time_ns(),
                        "publisher_monotonic_ns": time.monotonic_ns(),
                    })
            except Exception as exc:
                self._raw_error = str(exc)
                logger.exception("PICO raw publisher final audit failed")
            finally:
                if self._raw_socket is not None:
                    self._raw_socket.close(linger=0)
                ctx.term()

    def _publish_raw(self, kind: str, **values: object) -> None:
        if self._raw_socket is None:
            return
        self._raw_seq += 1
        if kind == "body":
            values['source_ids']=(self._source.for_sample(values['seq'],values['body_recv_ns']),)
        if kind == "metadata":
            self._raw_last_metadata = values
        recv_ns = values.get("body_recv_ns", values.get("recv_ns"))
        if kind == "control":
            recv_ns = values["raw_control"]["recv_ns"]
        if recv_ns is not None and time.monotonic_ns() - int(recv_ns) > _RAW_MAX_AGE_NS:
            self._raw_stale_drops += 1
            return
        try:
            payload = encode_pico_raw({
                "kind": kind, "stream_id": getattr(self._capture, "stream_id", self._raw_stream_id), "packet_seq": self._raw_seq,
                "source_drops": (self._capture.raw_dropped_frames + self._raw_send_drops
                                 + getattr(self, "_raw_queue_drops", 0) + getattr(self, "_raw_stale_drops", 0)
                                 + getattr(self, "_raw_shutdown_drops", 0)),
                "publish_ns": time.monotonic_ns(),
                "clock_domain": self._diag.clock_domain,
                **values,
            })
            if recv_ns is not None and time.monotonic_ns() - int(recv_ns) > _RAW_MAX_AGE_NS:
                self._raw_stale_drops += 1
                return
            self._raw_socket.send_multipart([PICO_RAW_TOPIC, payload], flags=self._zmq.NOBLOCK)
            if kind == "body":
                self._source.published(values['source_ids'][0],channel='raw_body',**values)
        except self._zmq.Again:
            self._raw_send_drops += 1
            self._raw_again = getattr(self, "_raw_again", 0) + 1
        except (TypeError, ValueError):
            self._raw_send_drops += 1
            self._raw_encode_errors = getattr(self, "_raw_encode_errors", 0) + 1

    def _drain_raw(self) -> None:
        if self._raw_socket is None:
            return
        for _ in range(16):
            if not self._raw_socket.poll(0):
                break
            self._raw_socket.recv(flags=self._zmq.NOBLOCK)
        now_ns = time.monotonic_ns()
        if now_ns - self._raw_metadata_ns >= 500_000_000:
            self._raw_metadata_ns = now_ns
            self._publish_raw(
                "metadata", joint_names=BODY_JOINT_NAMES,
                pose_columns=["x", "y", "z", "qx", "qy", "qz", "qw"],
                position_unit="m", coordinates="XRoboToolkit SDK (before INPUT_TO_DEPLOY transform)",
                input_to_deploy_matrix=INPUT_TO_DEPLOY_MATRIX.tolist(),
                input_to_deploy_quat_wxyz=INPUT_TO_DEPLOY_QUAT.tolist(),
                human_format=self._capture.config.human_format,
                input_mode=self._capture._input_mode, poll_hz=self.config.poll_hz,
                recording_chords=bool(self._recording_router),
                capture_config=asdict(self._capture.config),
                publisher_config=asdict(self.config),
                raw_queue_frames=_RAW_QUEUE_FRAMES, raw_max_age_ns=_RAW_MAX_AGE_NS,
                raw_send_hwm=_RAW_SEND_HWM,
                publisher_wall_ns=time.time_ns(), publisher_monotonic_ns=now_ns,
            )
        try:
            snapshot = self._capture.raw_snapshots.get_nowait()
        except queue.Empty:
            pass
        else:
            self._publish_raw("sdk", **_json_values(snapshot))
        try:
            body = self._capture.raw_frames.get_nowait()
        except queue.Empty:
            pass
        else:
            self._publish_raw(
                "body", seq=body.seq, body_recv_ns=body.body_recv_ns,
                source_ns=body.source_ns, poses=body.poses.tolist(),
            )

    def run_forever(self) -> None:

        last_body_seq: int | None = None
        last_pico_controller_seq: int | None = None
        published_body = 0
        published_pico_controller = 0
        start_ns = time.monotonic_ns()
        last_wait_print_ns = 0
        interrupted = False
        try:
            self._capture.start()
            self._start_raw_worker()
            while not self._closed:
                pico_controller = self._capture.latest_pico_controller()
                if pico_controller is not None and pico_controller.seq != last_pico_controller_seq:
                    control = self._recording_router.route(pico_controller) if self._recording_router else pico_controller
                    self._send_online(self._control_topic, encode_pico_control(control))
                    self._enqueue_raw_control(pico_controller, control)
                    last_pico_controller_seq = pico_controller.seq
                    published_pico_controller += 1
                    self._diag.count("ctl")
                try:
                    body = self._capture.wait_next_retarget_input(
                        last_body_seq,
                        timeout_s=0.0,
                    )
                    from dataclasses import replace
                    identity=self._source.for_sample(body.body_seq,body.body_recv_ns)
                    body=replace(body,source_ids=(identity,))
                    sent=self._send_online(self._retarget_topic, encode_pico_retarget_input(body))
                    if sent:
                        self._source.published(identity,channel='retarget_input',human_frame=body.human_frame,
                            body_recv_ns=body.body_recv_ns,convert_done_ns=body.convert_done_ns)
                    else:
                        from booster_deploy.telemetry.events import event
                        event('publish_send_refused',source_id=identity,channel='retarget_input')
                    # PICO's body_seq is not a per-delivery counter (it jumps
                    # by 2 within one 22 ms frame), so it cannot measure loss.
                    if last_body_seq is not None and int(body.body_seq) > last_body_seq + 1:
                        self._seq_gap_events += int(body.body_seq) - last_body_seq - 1
                    self._diag.mark("body")
                    self._diag.sample(
                        "sdk_to_pub",
                        max((time.monotonic_ns() - int(body.body_recv_ns)) / 1e6, 0.0),
                    )
                    last_body_seq = body.body_seq
                    published_body += 1
                except TimeoutError:
                    now_ns = time.monotonic_ns()
                    if published_body <= 0 and now_ns - last_wait_print_ns >= 1_000_000_000:
                        last_wait_print_ns = now_ns
                        print(
                            "Waiting for Pico RT body data | "
                            f"sdk_body_available={self._capture.sdk_body_available} "
                            f"body_seq={self._capture.body_seq} callback_count={self._capture.callback_count}",
                            flush=True,
                        )

                self._emit_diag()

                if (
                    published_body > 0
                    and self.config.print_interval > 0
                    and published_body % int(self.config.print_interval) == 0
                ):
                    elapsed_s = max((time.monotonic_ns() - start_ns) / 1e9, 1e-9)
                    print(
                        f"PICO_RT published body={published_body} ({published_body / elapsed_s:.1f} fps) "
                        f"pico_controller={published_pico_controller} ({published_pico_controller / elapsed_s:.1f} fps) "
                        f"last_body_seq={last_body_seq}",
                        flush=True,
                    )
                self._capture.wait_input_activity(last_body_seq, last_pico_controller_seq, 0.02)
        except KeyboardInterrupt:
            interrupted = True
            print("Stopping Pico RT publisher...", flush=True)
        finally:
            self._diag.close()
            self.close(fast=interrupted)

    def _emit_diag(self) -> None:
        """Human line: publish rate + gap tail; drops live in the jsonl/alert."""

        diag = self._diag
        if not diag.due():
            return
        now_ns = time.monotonic_ns()
        window = diag.interval("body")
        pub_hz = window.hz(now_ns)
        gap = window.gap_stats(now_ns)
        stall = window.stall_ms(now_ns)
        alerts: list[str] = []
        if below_floor(pub_hz):
            alerts.append(f"source {pub_hz:.1f} Hz < 50")
        if self._raw_send_drops:
            alerts.append(f"raw_drop {self._raw_send_drops}")
        if stall is not None and stall > 200.0:
            alerts.append(f"stall {stall:.0f} ms")
        diag.emit(
            f"pub {pub_hz:5.1f} Hz   gap {format_gap(gap)} ms",
            alerts=alerts,
            extra={"state": {"ctl_hz": round(diag.rate("ctl").hz(now_ns), 2),
                             "raw_drop": self._raw_send_drops,
                             "raw_queue_drop": self._raw_queue_drops,
                             "raw_stale_drop": self._raw_stale_drops,
                             "raw_shutdown_drop": self._raw_shutdown_drops,
                             "raw_capture_drop": self._capture.raw_dropped_frames,
                             "raw_again": self._raw_again,
                             "raw_encode_error": self._raw_encode_errors,
                             "raw_error": self._raw_error,
                             "online_send_again": self._online_send_again,
                             "seq_skips": self._seq_gap_events,
                             "sdk_to_pub_p95_ms": round(diag.latency("sdk_to_pub", now_ns).p95, 2)}},
            now_ns=now_ns,
        )

    def close(self, *, fast: bool = False) -> None:

        if self._closed:
            return
        self._closed = True
        self._raw_stop.set()
        self._capture.raw_activity_event.set()
        try:
            self._capture.close(close_sdk=not fast, clear_callback=not fast)
        finally:
            if self._raw_thread is not None:
                self._raw_thread.join(timeout=2.0)
                if self._raw_thread.is_alive():
                    self._raw_error = "raw publisher did not stop within 2s"
                    logger.error(self._raw_error)
            self._sock.close(linger=0)
            self._ctx.term()


__all__ = [
    "PicoControlZmqPublisher",
    "PicoRtZmqPublisher",
    "PicoRtZmqPublisherConfig",
]


def _control_values(frame: PicoControlFrameNs) -> dict[str, object]:
    return {
        "seq": frame.seq, "recv_ns": frame.recv_ns, "source_ns": frame.source_ns,
        "buttons": int(frame.buttons), "left_axis": frame.left_axis,
        "right_axis": frame.right_axis, "left_trigger": frame.left_trigger,
        "right_trigger": frame.right_trigger, "left_grip": frame.left_grip,
        "right_grip": frame.right_grip,
    }


def _json_values(value):
    if isinstance(value, dict):
        return {str(key): _json_values(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_values(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return value
