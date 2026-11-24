#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys

sys.path.append(".")

from booster_deploy.hand_retargeting.rt_worker import (
    HandRetargetZmqRunner,
    build_hand_retarget_worker_config,
    build_hand_retarget_zmq_config,
    parse_hands,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Realtime mocap_rt -> somehand/Revo2 hand retarget ZMQ worker.")
    parser.add_argument("--input-host", default="127.0.0.1")
    parser.add_argument("--input-port", type=int, default=5557)
    parser.add_argument("--input-retarget-topic", default="mocap_rt.retarget_input")
    parser.add_argument("--output-bind", default="0.0.0.0")
    parser.add_argument("--output-port", type=int, default=5558)
    parser.add_argument("--output-topic", default="hand_retarget.revo2_commands")
    parser.add_argument("--hands", default="both", help="Hand side: right, left, both, or comma-separated sides.")
    parser.add_argument("--left-config", default=None, help="Override somehand left Revo2 retarget YAML.")
    parser.add_argument("--right-config", default=None, help="Override somehand right Revo2 retarget YAML.")
    parser.add_argument("--human-format", default="bvh", help="Expected HumanFrame format; use empty string to accept any.")
    parser.add_argument("--tip-mode", choices=("extrapolate", "distal"), default="extrapolate")
    require_hands = parser.add_mutually_exclusive_group()
    require_hands.add_argument(
        "--require-hands",
        dest="require_hands",
        action="store_true",
        help="Raise when a selected hand is absent instead of skipping it (default).",
    )
    require_hands.add_argument(
        "--no-require-hands",
        dest="require_hands",
        action="store_false",
        help="Allow selected hands to be absent and skip them.",
    )
    parser.set_defaults(require_hands=True)
    parser.add_argument("--input-timeout", type=float, default=30.0)
    parser.add_argument("--print-interval", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hands = parse_hands(args.hands)
    zmq_cfg = build_hand_retarget_zmq_config(
        input_host=args.input_host,
        input_port=args.input_port,
        input_retarget_topic=args.input_retarget_topic,
        output_bind=args.output_bind,
        output_port=args.output_port,
        output_reference_topic=args.output_topic,
    )
    human_format = None if args.human_format == "" else args.human_format
    cfg = build_hand_retarget_worker_config(
        hands=hands,
        left_config_path=args.left_config,
        right_config_path=args.right_config,
        human_format=human_format,
        tip_mode=args.tip_mode,
        skip_missing_hands=not args.require_hands,
        zmq=zmq_cfg,
    )
    runner = HandRetargetZmqRunner(cfg)
    print(
        "Hand RT worker "
        f"hands={','.join(cfg.hands)} "
        f"input=tcp://{zmq_cfg.input_host}:{zmq_cfg.input_port}/{zmq_cfg.input_retarget_topic} "
        f"output=tcp://{zmq_cfg.output_bind}:{zmq_cfg.output_port}/{zmq_cfg.output_reference_topic} "
        f"tip_mode={cfg.tip_mode}",
        flush=True,
    )
    try:
        runner.run_forever(input_timeout_s=args.input_timeout, print_interval=args.print_interval)
    finally:
        runner.close()


if __name__ == "__main__":
    main()
