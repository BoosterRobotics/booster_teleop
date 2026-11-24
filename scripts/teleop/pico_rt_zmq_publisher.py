
from __future__ import annotations

import argparse
import sys

sys.path.append(".")

from booster_deploy.inputs.pico import PicoRtZmqPublisher, PicoRtZmqPublisherConfig


def main() -> None:

    parser = argparse.ArgumentParser(description="Publish Pico RT body/control packets over ZMQ.")
    parser.add_argument("--bind", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--retarget-topic", default="pico_rt.retarget_input")
    parser.add_argument("--control-topic", default="pico_rt.control")
    parser.add_argument("--provider-timeout", type=float, default=30.0)
    parser.add_argument("--send-hwm", type=int, default=1)
    parser.add_argument("--control-send-hwm", type=int, default=16)
    parser.add_argument("--print-interval", type=int, default=300)
    parser.add_argument(
        "--input-mode",
        choices=("auto", "callback", "polling"),
        default="auto",
        help="XRoboToolkit SDK input mode; auto prefers callback and falls back to polling.",
    )
    parser.add_argument("--xr-poll-hz", type=float, default=50.0, help="SDK polling rate in polling mode.")
    parser.add_argument(
        "--control-chord-mask",
        action="store_true",
        help=(
            "Mask trigger+X/Y recording chords out of pico_rt.control when recording from "
            "pico_rt.raw. A/B are never masked."
        ),
    )
    parser.add_argument("--raw-bind", default=None, help="Optional dedicated pre-retarget PICO stream endpoint.")
    args = parser.parse_args()

    publisher = PicoRtZmqPublisher(
        PicoRtZmqPublisherConfig(
            bind=args.bind,
            port=args.port,
            retarget_topic=args.retarget_topic,
            control_topic=args.control_topic,
            provider_timeout_s=args.provider_timeout,
            send_hwm=args.send_hwm,
            control_send_hwm=args.control_send_hwm,
            print_interval=args.print_interval,
            input_mode=args.input_mode,
            poll_hz=args.xr_poll_hz,
            control_chord_mask=args.control_chord_mask,
            raw_bind=args.raw_bind,
        )
    )
    print(
        f"Publishing Pico RT data on tcp://{args.bind}:{args.port} "
        f"retarget_topic={args.retarget_topic} control_topic={args.control_topic}",
        flush=True,
    )
    try:
        publisher.run_forever()
    except KeyboardInterrupt:
        print("Stopping Pico RT publisher...", flush=True)
        publisher.close(fast=True)


if __name__ == "__main__":
    main()
