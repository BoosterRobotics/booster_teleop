
from __future__ import annotations

import argparse
from pathlib import Path
import os
import sys

sys.path.append(".")

from booster_deploy.inputs.mocap import (
    MocapRtCaptureConfig,
    MocapRtZmqPublisher,
    MocapRtZmqPublisherConfig,
)


def main() -> None:

    parser = argparse.ArgumentParser(description="Publish PN-Link mocap RT body packets over ZMQ.")
    parser.add_argument("--bind", default=os.environ.get("BOOSTER_MOCAP_RT_BIND", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=os.environ.get("BOOSTER_MOCAP_RT_PORT", "5557"))
    parser.add_argument(
        "--retarget-topic",
        default=os.environ.get("BOOSTER_MOCAP_RT_TOPIC", "mocap_rt.retarget_input"),
    )
    parser.add_argument(
        "--provider-timeout",
        type=float,
        default=os.environ.get("BOOSTER_MOCAP_PROVIDER_TIMEOUT_S", "30.0"),
    )
    parser.add_argument(
        "--send-hwm",
        type=int,
        default=os.environ.get("BOOSTER_MOCAP_RT_SEND_HWM", "1"),
    )
    parser.add_argument(
        "--print-interval",
        type=int,
        default=os.environ.get("BOOSTER_MOCAP_RT_PRINT_INTERVAL", "300"),
    )
    parser.add_argument(
        "--local-ip",
        default=os.environ.get("BOOSTER_MOCAP_LOCAL_IP", "10.42.0.101"),
        help="Local NIC IP used by PN-Link UDP receive.",
    )
    parser.add_argument(
        "--local-port",
        type=int,
        default=os.environ.get("BOOSTER_MOCAP_LOCAL_PORT", "8002"),
        help="Local UDP receive port.",
    )
    parser.add_argument(
        "--server-ip",
        default=os.environ.get("BOOSTER_MOCAP_SERVER_IP", "10.42.0.202"),
        help="PN-Link server IP.",
    )
    parser.add_argument(
        "--server-port",
        type=int,
        default=os.environ.get("BOOSTER_MOCAP_SERVER_PORT", "8080"),
        help="PN-Link server port.",
    )
    parser.add_argument(
        "--poll-hz",
        type=float,
        default=os.environ.get("BOOSTER_MOCAP_POLL_HZ", "500.0"),
        help="Internal SDK polling frequency in Hz.",
    )
    parser.add_argument(
        "--human-format",
        default=os.environ.get("BOOSTER_MOCAP_HUMAN_FORMAT", "bvh"),
        help="Human format tag attached to MocapRetargetInputFrameNs.human_format.",
    )
    parser.add_argument(
        "--max-cached-frames",
        type=int,
        default=os.environ.get("BOOSTER_MOCAP_RT_MAX_CACHED_FRAMES", "4"),
        help="Ring buffer size for raw mocap frames; 0 disables history caching.",
    )
    parser.add_argument("--record-dir", type=Path, help="Enable robot remote-controlled raw BVH recording here.")
    args = parser.parse_args()

    capture_cfg = MocapRtCaptureConfig(
        human_format=args.human_format,
        local_ip=args.local_ip,
        local_port=args.local_port,
        server_ip=args.server_ip,
        server_port=args.server_port,
        poll_hz=args.poll_hz,
        timeout_s=float(args.provider_timeout),
        max_cached_frames=args.max_cached_frames,
    )
    publisher = MocapRtZmqPublisher(
        MocapRtZmqPublisherConfig(
            bind=args.bind,
            port=args.port,
            retarget_topic=args.retarget_topic,
            provider_timeout_s=args.provider_timeout,
            send_hwm=args.send_hwm,
            print_interval=args.print_interval,
            capture=capture_cfg,
            record_dir=args.record_dir,
        )
    )
    print(
        f"Publishing Mocap RT data on tcp://{args.bind}:{args.port} "
        f"retarget_topic={args.retarget_topic} "
        f"local={args.local_ip}:{args.local_port} server={args.server_ip}:{args.server_port}",
        flush=True,
    )
    try:
        publisher.run_forever()
    except KeyboardInterrupt:
        print("Stopping Mocap RT publisher...", flush=True)
        publisher.close(fast=True)


if __name__ == "__main__":
    main()
