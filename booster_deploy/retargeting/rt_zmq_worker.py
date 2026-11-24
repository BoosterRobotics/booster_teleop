
from __future__ import annotations

import time
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Protocol

import numpy as np

from booster_deploy.hand_retargeting.rt_worker import HandRetargetWorker, build_hand_retarget_worker_config, parse_hands
from booster_deploy.types.retarget_types import ReferenceFrame50HzNs, RetargetInputFrameProtocol, RetargetQposFrameNs
from booster_deploy.utils.diag_report import below_floor, format_gap, format_lat, reporter

from .rt_config import RealtimeRetargetConfig, RealtimeRetargetZmqConfig
from .rt_resampler import Strict50HzRetargetPostprocessor
from .rt_worker import RealtimeRetargetWorker
from .rt_zmq import (
    MocapRetargetInputSubscriber,
    PicoRetargetInputSubscriber,
    Reference50HzPublisher,
    RetargetResetSubscriber,
)


class RetargetInputSubscriber(Protocol):

    def recv_latest(self, timeout_s: float | None = None) -> RetargetInputFrameProtocol:

        ...

    def close(self) -> None:

        ...


SubscriberFactory = Callable[[RealtimeRetargetZmqConfig], RetargetInputSubscriber]

@dataclass(frozen=True)
class RealtimeRetargetInputSourceSpec:

    port: int
    topic: str
    subscriber_factory: SubscriberFactory
    label: str


@dataclass
class RealtimeRetargetZmqStats:

    processed: int = 0
    published: int = 0
    start_ns: int = 0

    def elapsed_s(self) -> float:

        if self.start_ns <= 0:
            return 0.0
        return max((time.monotonic_ns() - self.start_ns) / 1e9, 1e-9)


INPUT_SOURCE_DEFAULTS: dict[str, RealtimeRetargetInputSourceSpec] = {
    "pico_rt": RealtimeRetargetInputSourceSpec(
        port=5555,
        topic="pico_rt.retarget_input",
        subscriber_factory=PicoRetargetInputSubscriber,
        label="Pico RT",
    ),
    "mocap_rt": RealtimeRetargetInputSourceSpec(
        port=5557,
        topic="mocap_rt.retarget_input",
        subscriber_factory=MocapRetargetInputSubscriber,
        label="Mocap RT",
    ),
}

class RealtimeQposZmqRunner:

    def __init__(
        self,
        config: RealtimeRetargetConfig,
        *,
        publisher: Reference50HzPublisher | None = None,
        postprocessor: Strict50HzRetargetPostprocessor | None = None,
    ) -> None:
        self.config = config
        self.postprocessor = postprocessor or Strict50HzRetargetPostprocessor(config)
        self.publisher = publisher or Reference50HzPublisher(config.zmq)
        self.stats = RealtimeRetargetZmqStats(start_ns=time.monotonic_ns())

    def push_qpos_and_publish(
        self,
        robot_qpos,
        *,
        source_seq: int,
        source_sample_ns: int,
        robot_name: str | None = None,
        retarget_recv_ns: int | None = None,
        retarget_done_ns: int | None = None,
    ) -> list[ReferenceFrame50HzNs]:

        resolved_robot_name = str(robot_name or self.config.robot_name)
        if resolved_robot_name != self.config.robot_name:
            raise ValueError(
                f"qpos robot_name={resolved_robot_name!r} does not match pipeline "
                f"robot_name={self.config.robot_name!r}"
            )
        qpos = np.asarray(robot_qpos, dtype=np.float64)
        if qpos.ndim != 1 or qpos.shape[0] < 8:
            raise ValueError(f"robot_qpos must be a complete one-dimensional free-root qpos, got {qpos.shape}")
        if not np.isfinite(qpos).all():
            raise ValueError("robot_qpos contains non-finite values")
        done_ns = int(retarget_done_ns if retarget_done_ns is not None else time.monotonic_ns())
        frame = RetargetQposFrameNs(
            body_seq=int(source_seq),
            body_recv_ns=int(source_sample_ns),
            retarget_recv_ns=int(retarget_recv_ns if retarget_recv_ns is not None else done_ns),
            retarget_done_ns=done_ns,
            robot_name=resolved_robot_name,
            robot_qpos=qpos.copy(),
        )
        return self.postprocess_qpos_and_publish(frame)

    def postprocess_qpos_and_publish(self, frame: RetargetQposFrameNs) -> list[ReferenceFrame50HzNs]:

        refs = self.postprocessor.push(frame)
        self.stats.processed += 1
        self.publish_references(refs)
        return refs

    def publish_references(self, refs: Sequence[ReferenceFrame50HzNs]) -> int:

        for ref in refs:
            self.publisher.publish(ref)
        sent = len(refs)
        self.stats.published += sent
        return sent

    def close(self) -> None:
        self.publisher.close()


class RealtimeRetargetZmqRunner(RealtimeQposZmqRunner):

    def __init__(
        self,
        config: RealtimeRetargetConfig,
        *,
        subscriber: RetargetInputSubscriber | None = None,
        publisher: Reference50HzPublisher | None = None,
        reset_subscriber: RetargetResetSubscriber | None = None,
    ) -> None:

        self.config = config
        self.worker = RealtimeRetargetWorker(config)
        super().__init__(config, publisher=publisher, postprocessor=self.worker.postprocessor)
        self.hand_worker = (
            HandRetargetWorker(
                build_hand_retarget_worker_config(
                    hands=config.hand_sides,
                    left_config_path=config.hand_left_config_path,
                    right_config_path=config.hand_right_config_path,
                    somehand_repo=config.hand_somehand_repo,
                    human_format=config.human_format,
                    tip_mode=config.hand_tip_mode,
                    skip_missing_hands=config.hand_skip_missing_hands,
                )
            )
            if config.include_hand_frame
            else None
        )
        self._diag = reporter("worker")
        self._diag.note(robot=config.robot_name, input_source=config.source)
        self.subscriber = subscriber
        self.reset_subscriber = reset_subscriber or RetargetResetSubscriber(config.zmq)
        self._reset_stop_event = threading.Event()
        self._reset_pending = threading.Event()
        self._reset_thread = threading.Thread(
            target=self._run_reset_listener,
            name="retarget_reset_listener",
            daemon=True,
        )
        self._reset_thread.start()

    @classmethod
    def from_config(
        cls,
        config: RealtimeRetargetConfig,
        *,
        input_source: str = "pico_rt",
        subscriber: RetargetInputSubscriber | None = None,
        publisher: Reference50HzPublisher | None = None,
        reset_subscriber: RetargetResetSubscriber | None = None,
    ) -> "RealtimeRetargetZmqRunner":

        if subscriber is None:
            spec = input_source_spec(input_source)
            subscriber = spec.subscriber_factory(config.zmq)
        return cls(
            config,
            subscriber=subscriber,
            publisher=publisher,
            reset_subscriber=reset_subscriber,
        )

    def recv_process_publish_once(self, *, timeout_s: float | None = None) -> list[ReferenceFrame50HzNs]:

        if self.subscriber is None:
            raise RuntimeError("recv_process_publish_once requires a subscriber")
        frame = self.subscriber.recv_latest(timeout_s=timeout_s)
        return self.process_and_publish(frame)

    def process_and_publish(self, frame: RetargetInputFrameProtocol) -> list[ReferenceFrame50HzNs]:

        self._apply_pending_reset()
        if getattr(frame, 'offline_sample_ns', None) is not None:
            key = (getattr(frame, 'stream_id', None), int(frame.seq), int(frame.offline_sample_ns))
            if key == getattr(self, '_offline_last_key', None):
                self.publish_references(self._offline_last_refs)
                return self._offline_last_refs
        else:
            key = None
        started = time.perf_counter()
        refs = self.worker.process_one(frame)
        if self.config.include_human_frame or self.hand_worker is not None:
            hand_frame = self.hand_worker.retarget_one(frame) if self.hand_worker is not None else None
            refs = [
                replace(ref,
                    human_format=frame.human_format if self.config.include_human_frame else ref.human_format,
                    human_frame=frame.human_frame if self.config.include_human_frame else ref.human_frame,
                    hand_frame=hand_frame if hand_frame is not None and hand_frame.commands else ref.hand_frame,
                )
                for ref in refs
            ]
        self._offline_last_key, self._offline_last_refs = key, refs
        self.stats.processed += 1
        self.publish_references(refs)
        now_ns = time.monotonic_ns()
        self._diag.mark("in", now_ns)
        self._diag.count("solve", now_ns)
        for _ in refs:
            self._diag.count("out", now_ns)
        self._diag.sample("ik", (time.perf_counter() - started) * 1e3)
        if frame.clock_domain == self._diag.clock_domain:
            self._diag.sample("in_age", max((now_ns - int(frame.body_recv_ns)) / 1e6, 0.0))
        if refs and frame.clock_domain == self._diag.clock_domain:
            self._diag.sample(
                "in_to_out",
                max((int(refs[-1].retarget_done_ns) - int(frame.body_recv_ns)) / 1e6, 0.0),
            )
        return refs

    def _run_reset_listener(self) -> None:

        while not self._reset_stop_event.is_set():
            try:
                payload = self.reset_subscriber.recv_latest(timeout_s=0.1)
            except TimeoutError:
                continue
            except Exception as exc:
                if self._reset_stop_event.is_set():
                    return
                print(f"Error in retarget reset listener: {exc}", flush=True)
                time.sleep(0.05)
                continue
            if payload == b"reset_from_current":
                self._reset_pending.set()

    def _apply_pending_reset(self) -> None:

        if not self._reset_pending.is_set():
            return
        self._reset_pending.clear()
        self.worker.reset()
        print("RETARGET_RT reset history from current frame", flush=True)

    def run_forever(
        self,
        *,
        input_timeout_s: float,
        print_interval: int = 300,
        source_label: str = "retarget input",
    ) -> None:

        try:
            while True:
                try:
                    refs = self.recv_process_publish_once(timeout_s=input_timeout_s)
                except TimeoutError:
                    print(f"Waiting for {source_label} retarget input...", flush=True)
                    self._emit_diag()
                    continue
                if print_interval > 0 and self.stats.processed % int(print_interval) == 0:
                    print(self.status_line(refs), flush=True)
                self._emit_diag()
        finally:
            self._diag.close()

    def _emit_diag(self) -> None:
        """Human line: retarget solve rate, input rate, 50 Hz contract output."""

        diag = self._diag
        if not diag.due():
            return
        now_ns = time.monotonic_ns()
        window = diag.interval("in")
        in_hz = window.hz(now_ns)
        solve_hz = diag.rate("solve").hz(now_ns)
        out_hz = diag.rate("out").hz(now_ns)
        gap = window.gap_stats(now_ns)
        stall = window.stall_ms(now_ns)
        ik = diag.latency("ik", now_ns)
        alerts: list[str] = []
        if below_floor(solve_hz):
            alerts.append(f"solve {solve_hz:.1f} Hz < 50")
        if below_floor(out_hz):
            alerts.append(f"out {out_hz:.1f} Hz < 50")
        if ik.p95 > 12.0:
            alerts.append(f"ik_p95 {ik.p95:.1f} ms")
        if stall is not None and stall > 200.0:
            alerts.append(f"stall {stall:.0f} ms")
        diag.emit(
            f"solve {solve_hz:5.1f} Hz   in {in_hz:5.1f}→{out_hz:5.1f} Hz",
            alerts=alerts,
            extra={"state": {
                "ik_p50_ms": round(ik.p50, 2),
                "ik_p95_ms": round(ik.p95, 2),
                "in_age_p95_ms": round(diag.latency("in_age", now_ns).p95, 1),
                "in_to_out_p95_ms": round(diag.latency("in_to_out", now_ns).p95, 1),
                "gap_p95_ms": round(gap.p95, 1),
            }},
            now_ns=now_ns,
        )

    def status_line(self, latest_refs: Sequence[ReferenceFrame50HzNs] | None = None) -> str:

        elapsed_s = max(self.stats.elapsed_s(), 1e-9)
        latest_sample = latest_refs[-1].sample_idx if latest_refs else "none"
        return (
            f"RETARGET_RT processed={self.stats.processed} ({self.stats.processed / elapsed_s:.1f} fps) "
            f"published_50hz={self.stats.published} latest_sample={latest_sample}"
        )

    def close(self) -> None:

        self._reset_stop_event.set()
        if self._reset_thread.is_alive():
            self._reset_thread.join(timeout=1.0)
        self.reset_subscriber.close()
        self.worker.close()
        if self.hand_worker is not None:
            self.hand_worker.close()
        if self.subscriber is not None:
            self.subscriber.close()
        self._diag.close()
        super().close()


def input_source_spec(input_source: str) -> RealtimeRetargetInputSourceSpec:

    try:
        return INPUT_SOURCE_DEFAULTS[input_source]
    except KeyError as exc:
        choices = ", ".join(sorted(INPUT_SOURCE_DEFAULTS))
        raise KeyError(f"Unknown input_source={input_source!r}; choices: {choices}") from exc


def build_realtime_retarget_zmq_config(
    *,
    input_source: str = "pico_rt",
    input_host: str = "127.0.0.1",
    input_port: int | None = None,
    input_retarget_topic: str | None = None,
    output_bind: str = "0.0.0.0",
    output_port: int = 5556,
    output_reference_topic: str = "retarget_rt.reference_50hz",
    reset_bind: str = "0.0.0.0",
    reset_port: int = 5559,
    reset_topic: str = "retarget_rt.reset",
) -> tuple[RealtimeRetargetZmqConfig, RealtimeRetargetInputSourceSpec]:

    spec = input_source_spec(input_source)
    return (
        RealtimeRetargetZmqConfig(
            input_host=input_host,
            input_port=int(input_port if input_port is not None else spec.port),
            input_retarget_topic=str(input_retarget_topic or spec.topic),
            output_bind=output_bind,
            output_port=int(output_port),
            output_reference_topic=output_reference_topic,
            reset_bind=reset_bind,
            reset_port=int(reset_port),
            reset_topic=reset_topic,
        ),
        spec,
    )


def build_realtime_retarget_worker_config(
    *,
    target_fps: float = 50.0,
    robot_name: str = "g1",
    actual_human_height: float | None = None,
    solver: str = "daqp",
    ground_detection_mode: str = "mesh",
    qpos_filter_alpha: float = 0.85,
    joint_vel_filter_alpha: float = 0.75,
    max_joint_speed: float | None = 20.0,
    input_timeout_s: float = 30.0,
    include_human_frame: bool = False,
    include_hand_frame: bool = False,
    hand_sides: str | tuple[str, ...] = ("left", "right"),
    hand_left_config_path: str | None = None,
    hand_right_config_path: str | None = None,
    hand_somehand_repo: str | None = None,
    hand_tip_mode: str = "extrapolate",
    hand_skip_missing_hands: bool = True,
    zmq: RealtimeRetargetZmqConfig | None = None,
) -> RealtimeRetargetConfig:

    return RealtimeRetargetConfig(
        target_fps=float(target_fps),
        robot_name=robot_name,
        actual_human_height=actual_human_height,
        solver=solver,
        ground_detection_mode=ground_detection_mode,
        qpos_filter_alpha=float(qpos_filter_alpha),
        joint_vel_filter_alpha=float(joint_vel_filter_alpha),
        max_joint_speed=max_joint_speed,
        input_timeout_s=float(input_timeout_s),
        include_human_frame=bool(include_human_frame),
        include_hand_frame=bool(include_hand_frame),
        hand_sides=parse_hands(",".join(hand_sides) if isinstance(hand_sides, tuple) else str(hand_sides)),
        hand_left_config_path=hand_left_config_path,
        hand_right_config_path=hand_right_config_path,
        hand_somehand_repo=hand_somehand_repo,
        hand_tip_mode=hand_tip_mode,
        hand_skip_missing_hands=bool(hand_skip_missing_hands),
        zmq=zmq or RealtimeRetargetZmqConfig(),
    )


__all__ = [
    "INPUT_SOURCE_DEFAULTS",
    "RealtimeRetargetInputSourceSpec",
    "RealtimeQposZmqRunner",
    "RealtimeRetargetZmqRunner",
    "RealtimeRetargetZmqStats",
    "RetargetInputSubscriber",
    "build_realtime_retarget_worker_config",
    "build_realtime_retarget_zmq_config",
    "input_source_spec",
]
