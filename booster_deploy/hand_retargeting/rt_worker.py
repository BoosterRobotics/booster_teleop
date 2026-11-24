
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from booster_deploy.retargeting.rt_config import RealtimeRetargetZmqConfig
from booster_deploy.retargeting.rt_zmq import MocapRetargetInputSubscriber
from booster_deploy.types.hand_retarget_types import HandRetargetCommandNs, HandRetargetFrameNs
from booster_deploy.types.mocap_types import MocapRetargetInputFrameNs

from .somehand_adapter import Revo2SomehandRetargeter
from .zmq import HandRetargetFramePublisher


@dataclass(frozen=True)
class HandRetargetWorkerConfig:

    hands: tuple[str, ...] = ("right",)
    left_config_path: str | None = None
    right_config_path: str | None = None
    somehand_repo: str | None = None
    human_format: str | None = "bvh"
    tip_mode: str = "extrapolate"
    skip_missing_hands: bool = True
    zmq: RealtimeRetargetZmqConfig = RealtimeRetargetZmqConfig(
        input_host="127.0.0.1",
        input_port=5557,
        input_retarget_topic="mocap_rt.retarget_input",
        output_bind="0.0.0.0",
        output_port=5558,
        output_reference_topic="hand_retarget.revo2_commands",
    )


@dataclass
class HandRetargetZmqStats:
    processed: int = 0
    published: int = 0
    skipped: int = 0
    start_ns: int = 0

    def elapsed_s(self) -> float:
        if self.start_ns <= 0:
            return 0.0
        return max((time.monotonic_ns() - self.start_ns) / 1e9, 1e-9)


class HandRetargetWorker:

    def __init__(self, config: HandRetargetWorkerConfig) -> None:
        self.config = config
        self._retargeters: dict[str, Revo2SomehandRetargeter] = {}
        for side in config.hands:
            config_path = config.left_config_path if side == "left" else config.right_config_path
            self._retargeters[side] = Revo2SomehandRetargeter(
                side,
                config_path=config_path,
                somehand_repo=config.somehand_repo,
                tip_mode=config.tip_mode,
            )

    def retarget_one(self, frame: MocapRetargetInputFrameNs) -> HandRetargetFrameNs:

        input_format = str(frame.human_format)
        if self.config.human_format is not None and input_format != self.config.human_format:
            raise ValueError(
                f"Input human_format={input_format!r} does not match config.human_format={self.config.human_format!r}"
            )
        retarget_recv_ns = time.monotonic_ns()
        commands: list[HandRetargetCommandNs] = []
        for retargeter in self._retargeters.values():
            try:
                command, _ = retargeter.retarget(frame.human_frame)
            except KeyError:
                if not self.config.skip_missing_hands:
                    raise
                continue
            commands.append(command)
        retarget_done_ns = time.monotonic_ns()
        return HandRetargetFrameNs(
            body_seq=int(frame.body_seq),
            body_recv_ns=int(frame.body_recv_ns),
            retarget_recv_ns=retarget_recv_ns,
            retarget_done_ns=retarget_done_ns,
            human_format=input_format,
            commands=tuple(commands),
        )

    def close(self) -> None:
        self._retargeters.clear()


class HandRetargetZmqRunner:

    def __init__(
        self,
        config: HandRetargetWorkerConfig,
        *,
        subscriber: MocapRetargetInputSubscriber | None = None,
        publisher: HandRetargetFramePublisher | None = None,
    ) -> None:
        self.config = config
        self.worker = HandRetargetWorker(config)
        self.subscriber = subscriber or MocapRetargetInputSubscriber(config.zmq)
        self.publisher = publisher or HandRetargetFramePublisher(
            config.zmq.output_bind,
            config.zmq.output_port,
            config.zmq.output_reference_topic,
        )
        self.stats = HandRetargetZmqStats(start_ns=time.monotonic_ns())

    def recv_process_publish_once(self, *, timeout_s: float | None = None) -> HandRetargetFrameNs:
        frame = self.subscriber.recv_latest(timeout_s=timeout_s)
        return self.process_and_publish(frame)

    def process_and_publish(self, frame: MocapRetargetInputFrameNs) -> HandRetargetFrameNs:
        output = self.worker.retarget_one(frame)
        self.stats.processed += 1
        if output.commands:
            self.publisher.publish(output)
            self.stats.published += 1
        else:
            self.stats.skipped += 1
        return output

    def run_forever(self, *, input_timeout_s: float, print_interval: int = 300) -> None:
        while True:
            try:
                frame = self.recv_process_publish_once(timeout_s=input_timeout_s)
            except TimeoutError:
                print("Waiting for mocap_rt hand input...", flush=True)
                continue
            if print_interval > 0 and self.stats.processed % int(print_interval) == 0:
                print(self.status_line(frame), flush=True)

    def status_line(self, latest_frame: HandRetargetFrameNs | None = None) -> str:
        elapsed_s = self.stats.elapsed_s()
        hands = ",".join(command.hand_side for command in latest_frame.commands) if latest_frame else "none"
        return (
            f"HAND_RETARGET_RT processed={self.stats.processed} ({self.stats.processed / elapsed_s:.1f} fps) "
            f"published={self.stats.published} skipped={self.stats.skipped} latest_hands={hands}"
        )

    def close(self) -> None:
        self.worker.close()
        self.subscriber.close()
        self.publisher.close()



def parse_hands(value: str) -> tuple[str, ...]:

    normalized: list[str] = []
    for item in str(value).split(","):
        side = item.strip().lower()
        if not side:
            continue
        if side in ("both", "bihand"):
            normalized.extend(["left", "right"])
        elif side in ("left", "right"):
            normalized.append(side)
        else:
            raise ValueError(f"Unsupported hand side: {item!r}")
    if not normalized:
        raise ValueError("At least one hand side is required")
    return tuple(dict.fromkeys(normalized))



def build_hand_retarget_zmq_config(
    *,
    input_host: str = "127.0.0.1",
    input_port: int = 5557,
    input_retarget_topic: str = "mocap_rt.retarget_input",
    output_bind: str = "0.0.0.0",
    output_port: int = 5558,
    output_reference_topic: str = "hand_retarget.revo2_commands",
) -> RealtimeRetargetZmqConfig:
    return RealtimeRetargetZmqConfig(
        input_host=input_host,
        input_port=int(input_port),
        input_retarget_topic=input_retarget_topic,
        output_bind=output_bind,
        output_port=int(output_port),
        output_reference_topic=output_reference_topic,
    )



def build_hand_retarget_worker_config(
    *,
    hands: str | tuple[str, ...] = "right",
    left_config_path: str | Path | None = None,
    right_config_path: str | Path | None = None,
    somehand_repo: str | Path | None = None,
    human_format: str | None = "bvh",
    tip_mode: str = "extrapolate",
    skip_missing_hands: bool = True,
    zmq: RealtimeRetargetZmqConfig | None = None,
) -> HandRetargetWorkerConfig:
    parsed_hands = parse_hands(",".join(hands) if isinstance(hands, tuple) else str(hands))
    return HandRetargetWorkerConfig(
        hands=parsed_hands,
        left_config_path=None if left_config_path is None else str(left_config_path),
        right_config_path=None if right_config_path is None else str(right_config_path),
        somehand_repo=None if somehand_repo is None else str(somehand_repo),
        human_format=human_format,
        tip_mode=tip_mode,
        skip_missing_hands=skip_missing_hands,
        zmq=zmq or build_hand_retarget_zmq_config(),
    )


__all__ = [
    "HandRetargetWorker",
    "HandRetargetWorkerConfig",
    "HandRetargetZmqRunner",
    "HandRetargetZmqStats",
    "build_hand_retarget_worker_config",
    "build_hand_retarget_zmq_config",
    "parse_hands",
]
