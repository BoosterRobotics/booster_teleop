
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.append(".")

from booster_deploy.inputs.mocap.codecs import decode_mocap_retarget_input
from booster_deploy.inputs.mocap.adapter import skeleton_connections
from booster_deploy.inputs.mocap.ground_alignment import MocapGroundAligner
from booster_deploy.types.mocap_types import MocapRetargetInputFrameNs
from booster_deploy.visualization import add_viewer_backend_args, create_human_frame_viewer


class MocapRtRetargetInputSubscriber:

    def __init__(
        self,
        host: str,
        port: int,
        topic: str,
        *,
        recv_timeout_ms: int,
        recv_hwm: int = 1,
    ) -> None:

        import zmq

        self._zmq = zmq
        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.SUB)
        self._sock.setsockopt(zmq.RCVTIMEO, int(recv_timeout_ms))
        self._sock.setsockopt(zmq.RCVHWM, max(int(recv_hwm), 1))
        self._sock.connect(f"tcp://{host}:{int(port)}")
        self._sock.subscribe(topic.encode("utf-8"))

    def recv_latest(self) -> tuple[MocapRetargetInputFrameNs, int]:

        parts = self._sock.recv_multipart()
        drained = 0
        while True:
            try:
                parts = self._sock.recv_multipart(flags=self._zmq.NOBLOCK)
                drained += 1
            except self._zmq.Again:
                break
        payload = parts[-1] if len(parts) > 1 else parts[0].split(maxsplit=1)[-1]
        return decode_mocap_retarget_input(payload), drained

    def close(self) -> None:

        self._sock.close(linger=0)
        self._ctx.term()


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description="Visualize live PN-Link mocap RT/ZMQ HumanFrame data.")
    parser.add_argument(
        "--host",
        default=os.environ.get("BOOSTER_MOCAP_RT_HOST", "127.0.0.1"),
        help="Mocap RT ZMQ publisher host.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=os.environ.get("BOOSTER_MOCAP_RT_PORT", "5557"),
        help="Mocap RT ZMQ publisher port.",
    )
    parser.add_argument(
        "--topic",
        default=os.environ.get("BOOSTER_MOCAP_RT_TOPIC", "mocap_rt.retarget_input"),
        help="Mocap RT retarget-input topic.",
    )
    parser.add_argument("--frames", type=int, default=0, help="Frames to visualize. Use <=0 to run until interrupted.")
    parser.add_argument("--timeout", type=float, default=2.0, help="Receive timeout in seconds.")
    parser.add_argument(
        "--align-ground",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Align inferred PN-Link foot ground to MuJoCo world z=0.",
    )
    parser.add_argument("--ground-z", type=float, default=None, help="Manual mocap ground z override.")
    parser.add_argument("--ground-target-z", type=float, default=0.0, help="Target world ground z.")
    parser.add_argument("--ground-calibration-frames", type=int, default=30, help="Frames used for median ground calibration.")
    parser.add_argument(
        "--safe-visualizer-exit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use os._exit after finite runs to avoid native MuJoCo/GLFW shutdown issues.",
    )
    parser.add_argument("--viewer-period", type=float, default=0.0, help="Optional viewer step sleep in seconds.")
    parser.add_argument(
        "--print-interval",
        type=int,
        default=0,
        help="Print status every N frames; <=0 disables periodic status (default).",
    )
    parser.add_argument(
        "--wait-status-interval",
        type=float,
        default=2.0,
        help="Print waiting status every N seconds before first frame.",
    )
    add_viewer_backend_args(parser)
    parser.add_argument(
        "--viser",
        action="store_true",
        help="Shortcut for --viewer-backend viser.",
    )
    args = parser.parse_args()
    if args.viser:
        args.viewer_backend = "viser"
    return args


def wait_for_next_packet(
    subscriber: MocapRtRetargetInputSubscriber,
    *,
    last_seq: int | None,
    wait_status_interval_s: float,
) -> MocapRetargetInputFrameNs:

    import zmq

    last_status_time = time.monotonic()
    while True:
        try:
            frame, drained = subscriber.recv_latest()
        except zmq.Again:
            now = time.monotonic()
            if now - last_status_time >= wait_status_interval_s:
                last_status_time = now
                print(f"Waiting for Mocap RT/ZMQ frame | last_seq={last_seq}")
            continue
        if last_seq is None or frame.body_seq > last_seq:
            if drained > 0:
                print(f"Drained {drained} queued Mocap RT frames before seq={frame.body_seq}")
            return frame


def main() -> None:

    args = parse_args()
    print(f"Connecting to Mocap RT/ZMQ tcp://{args.host}:{args.port} topic={args.topic}")
    subscriber = MocapRtRetargetInputSubscriber(
        host=args.host,
        port=args.port,
        topic=args.topic,
        recv_timeout_ms=int(max(args.timeout, 0.001) * 1000),
    )

    ground_aligner = None
    if args.align_ground:
        ground_aligner = MocapGroundAligner(
            calibration_frames=args.ground_calibration_frames,
            target_ground_z=args.ground_target_z,
            ground_z=args.ground_z,
        )
        print(
            "Mocap ground alignment enabled | target_z=%.6f%s"
            % (
                ground_aligner.target_ground_z,
                "" if ground_aligner.ground_z is None else " ground_z=%.6f" % ground_aligner.ground_z,
            )
        )

    frame_count = 0
    last_seq: int | None = None
    reported_ground_alignment = bool(ground_aligner is not None and ground_aligner.is_calibrated)
    start_time = time.monotonic()
    print(f"Waiting for first Mocap RT/ZMQ HumanFrame before opening {args.viewer_backend} viewer...")
    first_packet = wait_for_next_packet(
        subscriber,
        last_seq=None,
        wait_status_interval_s=args.wait_status_interval,
    )
    print(
        f"Received first Mocap RT/ZMQ frame | seq={first_packet.body_seq} joints={len(first_packet.human_frame)}"
    )

    viewer = create_human_frame_viewer(
        args.viewer_backend,
        title="Mocap RT HumanFrame",
        skeleton_connections=skeleton_connections(),
        release_camera_after_first_frame=True,
        camera_azimuth=args.camera_azimuth,
        camera_elevation=args.camera_elevation,
        host=args.viser_host,
        port=args.viser_port,
    )
    try:
        while (args.frames <= 0 or frame_count < args.frames) and viewer.is_running():
            packet = first_packet if frame_count == 0 else wait_for_next_packet(
                subscriber,
                last_seq=last_seq,
                wait_status_interval_s=args.wait_status_interval,
            )
            frame = packet.human_frame
            if ground_aligner is not None:
                frame = ground_aligner.align(frame)
                if ground_aligner.is_calibrated and not reported_ground_alignment:
                    print(
                        "Mocap ground alignment calibrated | samples=%d ground_z=%.6f target_z=%.6f z_offset=%.6f"
                        % (
                            ground_aligner.sample_count,
                            ground_aligner.ground_z,
                            ground_aligner.target_ground_z,
                            ground_aligner.z_offset,
                        )
                    )
                    reported_ground_alignment = True

            viewer.step(frame, rate_limit_s=args.viewer_period)
            if args.print_interval > 0 and frame_count % args.print_interval == 0:
                elapsed_s = max(time.monotonic() - start_time, 1e-9)
                print(
                    "seq=%s source=body_recv_ns:%s viewer_fps=%.2f joints=%d"
                    % (packet.body_seq, packet.body_recv_ns, (frame_count + 1) / elapsed_s, len(frame))
                )
            last_seq = packet.body_seq
            frame_count += 1
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        viewer.close()
        subscriber.close()

    print(f"Visualized {frame_count} live Mocap RT/ZMQ frames.")
    if args.frames > 0 and args.safe_visualizer_exit and args.viewer_backend == "mujoco":
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
