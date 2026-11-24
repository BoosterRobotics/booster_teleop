
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

from booster_deploy.retargeting.gmr.params import ROBOT_XML_DICT
from booster_deploy.types.retarget_types import ReferenceFrame50HzNs
from booster_deploy.retargeting.rt_zmq import Reference50HzSubscriber
from booster_deploy.hand_retargeting.zmq import HandRetargetFrameSubscriber
from booster_deploy.types.hand_retarget_types import HandRetargetFrameNs
from booster_deploy.types import RetargetFrame
from booster_deploy.utils.project_path import resolve_project_path
from booster_deploy.visualization import add_viewer_backend_args, create_retargeting_viewer
from booster_deploy.visualization.revo2_combined_qpos import (
    DEFAULT_T2_REVO2_XML_PATH,
    LatestHandFrameSelector,
    Revo2CombinedQposMapper,
)


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description="Visualize realtime retarget RT/ZMQ robot reference frames.")
    parser.add_argument(
        "--host",
        default=os.environ.get("BOOSTER_RT_REFERENCE_HOST", "127.0.0.1"),
        help="Retarget RT ZMQ publisher host.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=os.environ.get("BOOSTER_RT_REFERENCE_PORT", "5556"),
        help="Retarget RT ZMQ publisher port.",
    )
    parser.add_argument(
        "--topic",
        default=os.environ.get("BOOSTER_RT_REFERENCE_TOPIC", "retarget_rt.reference_50hz"),
        help="ReferenceFrame50HzNs topic.",
    )
    parser.add_argument("--robot-name", default="t2", help="Robot name used to resolve default GMR XML.")
    parser.add_argument("--robot-xml-path", default=None, help="Override MuJoCo XML path.")
    parser.add_argument(
        "--with-hands",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the T2+Revo2 XML and embedded/independent hand commands; missing hands use a closed pose.",
    )
    parser.add_argument(
        "--hand-host",
        default=os.environ.get("BOOSTER_HAND_RETARGET_HOST"),
        help="Independent hand-command host; defaults to --host.",
    )
    parser.add_argument(
        "--hand-port",
        type=int,
        default=os.environ.get("BOOSTER_HAND_RETARGET_PORT", "5558"),
        help="Independent hand-command ZMQ port.",
    )
    parser.add_argument(
        "--hand-topic",
        default=os.environ.get(
            "BOOSTER_HAND_RETARGET_TOPIC",
            "hand_retarget.revo2_commands",
        ),
        help="Independent HandRetargetFrameNs topic.",
    )
    parser.add_argument(
        "--hand-stale-s",
        type=float,
        default=float(os.environ.get("BOOSTER_HAND_RETARGET_STALE_S", "0.25")),
        help="Use independent hand data only this long after local receipt.",
    )
    parser.add_argument("--root-body", default="trunk", help="Root body name for camera tracking.")
    parser.add_argument(
        "--follow-robot",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep the viewer camera following the robot root body.",
    )
    parser.add_argument("--camera-distance", type=float, default=1.8, help="Initial camera distance.")
    parser.add_argument("--camera-azimuth", type=float, default=150.0, help="Initial camera azimuth in degrees.")
    parser.add_argument("--camera-elevation", type=float, default=-12.0, help="Initial camera elevation in degrees.")
    parser.add_argument(
        "--show-support-base",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Draw robot CoM ground projection and actual ground-contact support base.",
    )
    parser.add_argument("--frames", type=int, default=0, help="Frames to visualize. Use <=0 to run until interrupted.")
    parser.add_argument("--timeout", type=float, default=2.0, help="Receive timeout in seconds.")
    parser.add_argument("--recv-hwm", type=int, default=128, help="ZMQ receive high-water mark.")
    parser.add_argument("--viewer-period", type=float, default=0.0, help="Optional viewer step sleep in seconds.")
    parser.add_argument("--print-interval", type=int, default=0, help="Print status every N rendered frames; <=0 disables periodic status (default).")
    parser.add_argument("--wait-status-interval", type=float, default=2.0, help="Print waiting status every N seconds.")
    parser.add_argument(
        "--safe-visualizer-exit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use os._exit after finite runs to avoid native MuJoCo/GLFW shutdown issues.",
    )
    parser.add_argument(
        "--vscode-browser",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Deprecated: print VS Code browser instructions without invoking the code CLI."
        ),
    )
    add_viewer_backend_args(parser)



    parser.set_defaults(viewer_backend="viser", viser_host="0.0.0.0", viser_port=8081)
    return parser.parse_args()


def open_vscode_browser(url: str, *, enabled: bool) -> bool:

    if not enabled:
        return False
    print(
        "--vscode-browser no longer opens or reuses a VS Code window. "
        f"Forward the Viser port, then open Simple Browser: Show manually: {url}"
    )
    return False


def reference_to_retarget_frame(
    frame: ReferenceFrame50HzNs,
    *,
    qpos_mapper: Revo2CombinedQposMapper | None = None,
    hand_frame: HandRetargetFrameNs | None = None,
) -> RetargetFrame:

    qpos = np.concatenate(
        [
            np.asarray(frame.root_pos, dtype=np.float64),
            np.asarray(frame.root_quat, dtype=np.float64),
            np.asarray(frame.joint_pos, dtype=np.float64),
        ],
        axis=0,
    )
    if qpos_mapper is not None:
        qpos = qpos_mapper.compose(
            qpos,
            frame.hand_frame if hand_frame is None else hand_frame,
        )
    return RetargetFrame.from_qpos(
        qpos,
        robot_name=("t2_revo2" if qpos_mapper is not None else frame.robot_name),
        source="retarget_rt.reference_50hz",
        timestamp_s=frame.sample_ns / 1e9,
        seq=frame.sample_idx,
        joint_vel=(
            None
            if qpos_mapper is not None
            else np.asarray(frame.joint_vel, dtype=np.float64)
        ),
    )


def wait_for_next_reference(
    subscriber: Reference50HzSubscriber,
    *,
    last_sample_idx: int | None,
    timeout_s: float,
    wait_status_interval_s: float,
) -> tuple[ReferenceFrame50HzNs, int]:

    last_status_time = time.monotonic()
    while True:
        try:
            frames = subscriber.recv_newer_than(last_sample_idx, timeout_s=timeout_s)
        except TimeoutError:
            now = time.monotonic()
            if now - last_status_time >= wait_status_interval_s:
                last_status_time = now
                print(f"Waiting for retarget RT reference frame | last_sample_idx={last_sample_idx}")
            continue
        if not frames:
            continue
        return frames[-1], max(len(frames) - 1, 0)


def main() -> None:

    args = parse_args()
    try:
        body_xml_path = str(resolve_project_path(ROBOT_XML_DICT[args.robot_name]))
    except KeyError as exc:
        raise KeyError(
            f"No default robot XML found for robot_name={args.robot_name!r}"
        ) from exc
    xml_path = args.robot_xml_path
    if xml_path is None:
        if args.with_hands:
            if args.robot_name != "t2":
                raise ValueError(
                    "--with-hands currently has a default combined XML only for "
                    f"robot_name='t2', got {args.robot_name!r}; pass --no-with-hands"
                )
            xml_path = str(DEFAULT_T2_REVO2_XML_PATH)
        else:
            xml_path = body_xml_path
    xml_path = str(resolve_project_path(xml_path))
    qpos_mapper = (
        Revo2CombinedQposMapper(
            body_xml_path=body_xml_path,
            combined_xml_path=xml_path,
        )
        if args.with_hands
        else None
    )
    hand_selector = LatestHandFrameSelector(stale_s=args.hand_stale_s)
    hand_host = args.host if args.hand_host in (None, "") else args.hand_host

    print(f"Connecting to retarget RT/ZMQ tcp://{args.host}:{args.port} topic={args.topic}")
    if args.with_hands:
        print(
            "Connecting to hand RT/ZMQ "
            f"tcp://{hand_host}:{args.hand_port} topic={args.hand_topic} "
            f"stale_s={args.hand_stale_s:g}"
        )
    print(f"Opening robot XML with {args.viewer_backend}: {xml_path}")
    subscriber = None
    hand_subscriber = None
    viewer = None
    frame_count = 0
    last_sample_idx: int | None = None
    start_time = time.monotonic()
    try:
        subscriber = Reference50HzSubscriber(
            host=args.host,
            port=args.port,
            topic=args.topic,
            recv_hwm=args.recv_hwm,
            recv_timeout_ms=int(max(args.timeout, 0.001) * 1000),
        )
        if args.with_hands:
            hand_subscriber = HandRetargetFrameSubscriber(
                hand_host,
                args.hand_port,
                args.hand_topic,
                recv_timeout_ms=0,
            )
        viewer = create_retargeting_viewer(
            args.viewer_backend,
            xml_path,
            robot_root_body=args.root_body,
            show_human=False,
            show_scaled_human=False,
            follow_robot=bool(args.follow_robot),
            camera_distance=args.camera_distance,
            camera_azimuth=args.camera_azimuth,
            camera_elevation=args.camera_elevation,
            show_robot_support_base=bool(args.show_support_base),
            host=args.viser_host,
            port=args.viser_port,
            show_joint_axes=args.viser_show_joint_axes,
        )
        if args.viewer_backend == "viser":
            vscode_browser_enabled = bool(args.vscode_browser)
            viser_url = f"http://127.0.0.1:{args.viser_port}"
            print(
                f"Viser URL: {viser_url} (on the viewer host). "
                "For SSH, forward this port and open it in your local browser."
            )
            open_vscode_browser(viser_url, enabled=vscode_browser_enabled)
        while (args.frames <= 0 or frame_count < args.frames) and viewer.is_running():
            ref, skipped = wait_for_next_reference(
                subscriber,
                last_sample_idx=last_sample_idx,
                timeout_s=args.timeout,
                wait_status_interval_s=args.wait_status_interval,
            )
            selected_hand_frame = None
            hand_source = "disabled"
            if qpos_mapper is not None:
                now_s = time.monotonic()
                if hand_subscriber is not None:
                    independent_hand_frame = hand_subscriber.try_recv_latest()
                    if independent_hand_frame is not None:
                        hand_selector.update_independent(
                            independent_hand_frame,
                            now_s=now_s,
                        )
                selected_hand_frame, hand_source = hand_selector.select(
                    ref.hand_frame,
                    now_s=now_s,
                )
            retarget_frame = reference_to_retarget_frame(
                ref,
                qpos_mapper=qpos_mapper,
                hand_frame=selected_hand_frame,
            )
            viewer.step(retarget_frame, rate_limit_s=args.viewer_period)
            last_sample_idx = ref.sample_idx
            if args.print_interval > 0 and frame_count % args.print_interval == 0:
                elapsed_s = max(time.monotonic() - start_time, 1e-9)
                hand_sides = (
                    set(qpos_mapper.available_hand_sides(selected_hand_frame))
                    if qpos_mapper is not None
                    else set()
                )
                hand_status = (
                    ",".join(
                        f"{side}={'retarget' if side in hand_sides else 'fist'}"
                        for side in ("left", "right")
                    )
                    if qpos_mapper is not None
                    else "disabled"
                )
                print(
                    "sample_idx=%d body_seq=%d render_fps=%.2f root_z=%.6f "
                    "skipped=%d hands=%s source=%s"
                    % (
                        ref.sample_idx,
                        ref.body_seq,
                        (frame_count + 1) / elapsed_s,
                        retarget_frame.root_pos[2],
                        skipped,
                        hand_status,
                        hand_source,
                    )
                )
            frame_count += 1
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        if viewer is not None:
            viewer.close()
        if subscriber is not None:
            subscriber.close()
        if hand_subscriber is not None:
            hand_subscriber.close()

    print(f"Visualized {frame_count} retarget RT robot frames.")
    if args.frames > 0 and args.safe_visualizer_exit and args.viewer_backend == "mujoco":
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
