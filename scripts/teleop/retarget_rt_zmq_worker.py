
from __future__ import annotations

import argparse
import os
import sys

sys.path.append(".")

from booster_deploy.retargeting.rt_zmq_worker import (
    INPUT_SOURCE_DEFAULTS,
    RealtimeRetargetZmqRunner,
    build_realtime_retarget_worker_config,
    build_realtime_retarget_zmq_config,
)


def main() -> None:

    parser = argparse.ArgumentParser(description="Realtime RT input -> GMR -> 50Hz reference ZMQ worker.")
    parser.add_argument(
        "--input-source",
        choices=sorted(INPUT_SOURCE_DEFAULTS.keys()),
        default="mocap_rt",
        help="Select the realtime human input source, which determines the default subscriber port/topic and type.",
    )
    parser.add_argument("--input-host", default="127.0.0.1")
    parser.add_argument(
        "--input-port",
        type=int,
        default=None,
        help="Override the input source default port; use INPUT_SOURCE_DEFAULTS when None.",
    )
    parser.add_argument(
        "--input-retarget-topic",
        default=None,
        help="Override the input source default topic; use INPUT_SOURCE_DEFAULTS when None.",
    )
    parser.add_argument("--output-bind", default="0.0.0.0")
    parser.add_argument("--output-port", type=int, default=5556)
    parser.add_argument("--output-reference-topic", default="retarget_rt.reference_50hz")
    parser.add_argument("--reset-bind", default=os.getenv("BOOSTER_RT_RESET_BIND", "0.0.0.0"))
    parser.add_argument("--reset-port", type=int, default=int(os.getenv("BOOSTER_RT_RESET_PORT", "5559")))
    parser.add_argument("--reset-topic", default=os.getenv("BOOSTER_RT_RESET_TOPIC", "retarget_rt.reset"))
    parser.add_argument("--robot-name", default="t2")
    parser.add_argument("--target-fps", type=float, default=50.0)
    parser.add_argument("--solver", default="daqp")
    parser.add_argument("--ground-detection-mode", choices=["mesh", "link", "off"], default="mesh")
    parser.add_argument(
        "--actual-human-height", type=float, default=None,
        help="Override human height in meters; PICO otherwise estimates the configured skeleton path on the first frame.",
    )
    parser.add_argument("--qpos-filter-alpha", type=float, default=0.99)
    parser.add_argument("--joint-vel-filter-alpha", type=float, default=0.99)
    parser.add_argument("--max-joint-speed", type=float, default=20.0)
    parser.add_argument("--input-timeout", type=float, default=30.0)
    parser.add_argument("--print-interval", type=int, default=300)
    parser.add_argument("--include-human-frame", action="store_true", help="Embed the source HumanFrame in each 50Hz reference packet.")
    parser.add_argument("--include-hand-frame", action="store_true", help="Retarget Revo2 hands in the same worker and embed hand commands in each 50Hz reference packet. Required for t2_gmc_rt MuJoCo hand control.")
    parser.add_argument("--hands", default="both", help="Hand side for --include-hand-frame: left, right, both, or comma-separated sides.")
    parser.add_argument("--left-hand-config", default=None, help="Override left Revo2 retarget YAML for embedded hand retarget.")
    parser.add_argument("--right-hand-config", default=None, help="Override right Revo2 retarget YAML for embedded hand retarget.")
    parser.add_argument("--hand-tip-mode", choices=("extrapolate", "distal"), default="extrapolate")
    parser.add_argument("--require-hands", action="store_true", help="Raise when selected hand landmarks are absent.")
    args = parser.parse_args()

    zmq_cfg, source_spec = build_realtime_retarget_zmq_config(
        input_source=args.input_source,
        input_host=args.input_host,
        input_port=args.input_port,
        input_retarget_topic=args.input_retarget_topic,
        output_bind=args.output_bind,
        output_port=args.output_port,
        output_reference_topic=args.output_reference_topic,
        reset_bind=args.reset_bind,
        reset_port=args.reset_port,
        reset_topic=args.reset_topic,
    )
    cfg = build_realtime_retarget_worker_config(
        target_fps=args.target_fps,
        robot_name=args.robot_name,
        actual_human_height=args.actual_human_height,
        solver=args.solver,
        ground_detection_mode=args.ground_detection_mode,
        qpos_filter_alpha=args.qpos_filter_alpha,
        joint_vel_filter_alpha=args.joint_vel_filter_alpha,
        max_joint_speed=args.max_joint_speed,
        input_timeout_s=args.input_timeout,
        include_human_frame=args.include_human_frame,
        include_hand_frame=args.include_hand_frame,
        hand_sides=args.hands,
        hand_left_config_path=args.left_hand_config,
        hand_right_config_path=args.right_hand_config,
        hand_tip_mode=args.hand_tip_mode,
        hand_skip_missing_hands=not args.require_hands,
        zmq=zmq_cfg,
    )
    runner = RealtimeRetargetZmqRunner.from_config(cfg, input_source=args.input_source)
    source_label = source_spec.label

    extra = []
    if args.include_human_frame:
        extra.append("human_frame")
    if args.include_hand_frame:
        extra.append(f"hand_frame:{','.join(cfg.hand_sides)}")
    extra_text = " embeds=" + ",".join(extra) if extra else ""
    print(
        f"Retarget RT worker source={source_label} "
        f"input=tcp://{args.input_host}:{zmq_cfg.input_port}/{zmq_cfg.input_retarget_topic} "
        f"output=tcp://{args.output_bind}:{args.output_port}/{args.output_reference_topic} "
        f"reset=tcp://{args.reset_bind}:{args.reset_port}/{args.reset_topic} "
        f"robot={args.robot_name} target_fps={args.target_fps}{extra_text}",
        flush=True,
    )
    try:
        runner.run_forever(
            input_timeout_s=args.input_timeout,
            print_interval=args.print_interval,
            source_label=source_label,
        )
    finally:
        runner.close()


if __name__ == "__main__":
    main()
