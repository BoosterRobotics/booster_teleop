
from __future__ import annotations

import argparse
import os
import select
import sys
import termios
import time
import tty

sys.path.append(".")

from booster_deploy.inputs.mocap.calibration_provider import (
    CalibrationProgress,
    CommandResultDetail,
    PNLinkMocapProvider,
)


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description="Interactive PN-Link mocap calibration flow followed by a local HumanFrame viewer.",
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
        "--timeout",
        type=float,
        default=os.environ.get("BOOSTER_MOCAP_CALIBRATION_TIMEOUT_S", "240.0"),
        help="Calibration timeout in seconds.",
    )
    parser.add_argument(
        "--poll-hz",
        type=float,
        default=os.environ.get("BOOSTER_MOCAP_CALIBRATION_POLL_HZ", "100.0"),
        help="Provider polling frequency.",
    )
    parser.add_argument("--auto", action="store_true", help="Automatically run StartCapture -> Calibrate.")
    parser.add_argument(
        "--skip-calibration",
        action="store_true",
        help="Skip the calibration state machine and jump straight to the local viewer.",
    )

    parser.add_argument("--viewer-period", type=float, default=0.0, help="Optional viewer step sleep in seconds.")
    parser.add_argument("--print-interval", type=int, default=60, help="Print viewer status every N frames.")
    parser.add_argument(
        "--align-ground",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Align inferred PN-Link foot ground to MuJoCo world z=0 inside the viewer.",
    )
    parser.add_argument("--ground-z", type=float, default=None, help="Manual mocap ground z override.")
    parser.add_argument("--ground-target-z", type=float, default=0.0, help="Target world ground z.")
    parser.add_argument(
        "--ground-calibration-frames", type=int, default=30, help="Frames used for median ground calibration."
    )
    parser.add_argument(
        "--no-viewer",
        action="store_true",
        help="Skip the local viewer after calibration; useful for headless boxes.",
    )
    parser.add_argument(
        "--viser",
        action="store_true",
        help="Use the browser-based Viser viewer instead of the native MuJoCo/GLFW viewer.",
    )
    parser.add_argument(
        "--viser-host",
        default=os.environ.get("BOOSTER_MOCAP_VISER_HOST", "0.0.0.0"),
        help="Viser bind host used with --viser.",
    )
    parser.add_argument(
        "--viser-port",
        type=int,
        default=os.environ.get("BOOSTER_MOCAP_VISER_PORT", "8080"),
        help="Viser HTTP/WebSocket port used with --viser.",
    )
    parser.add_argument(
        "--safe-visualizer-exit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use os._exit on viewer close to avoid native MuJoCo/GLFW shutdown issues.",
    )
    return parser.parse_args()


class CalibrationStateMachine:

    def __init__(self, provider: PNLinkMocapProvider, *, timeout_s: float) -> None:

        self.provider = provider
        self.timeout_s = float(timeout_s)
        self._last_progress_seq = -1
        self._last_result_detail: CommandResultDetail | None = None
        self._avatar_reported = False
        self._phase2_reported = False
        self._start_time = time.monotonic()


    def run_interactive(self) -> bool:

        self._print_banner()
        print(
            "[Stage 1] Press N to start capture. After motion data is received, press C to calibrate.",
            flush=True,
        )
        while True:
            if self._timed_out():
                raise TimeoutError(f"PN-Link calibration did not complete within {self.timeout_s:.1f}s")
            self._tick_status()
            if self.provider.calibration_done:
                self._tick_status()
                print(
                    f"PN-Link calibration complete. last_command_result={self.provider.last_command_result}",
                    flush=True,
                )
                return True

            key = _read_key()
            if key is not None:
                key_lower = key.lower()
                if key_lower == "n":
                    self._send_command("Start Capture", self.provider.start_capture)
                elif key_lower == "c":
                    self._send_command("Calibrate Motion", self.provider.calibrate)
                elif key_lower in {"f", "s"}:
                    self._send_command("Stop Capture", self.provider.stop_capture)
                elif key_lower == "q":
                    print("Stopped.", flush=True)
                    return False
            time.sleep(0.02)

    def run_auto(self) -> bool:

        print("Auto mode: sending CommandStartCapture...", flush=True)
        self.provider.start_capture()
        _wait_for(
            lambda: self.provider.capture_started,
            timeout_s=min(self.timeout_s, 30.0),
            label="Start Capture",
        )
        print("Start Capture done. Waiting for first AvatarUpdated event...", flush=True)
        _wait_for(
            lambda: self.provider.avatar_seen,
            timeout_s=min(self.timeout_s, 30.0),
            label="AvatarUpdated",
        )
        print("AvatarUpdated received. Sending CommandCalibrateMotion...", flush=True)
        self.provider.calibrate()

        while not self.provider.calibration_done:
            if self._timed_out():
                raise TimeoutError(f"PN-Link calibration did not complete within {self.timeout_s:.1f}s")
            self._tick_status()
            time.sleep(0.02)
        self._tick_status()
        print(
            f"PN-Link calibration complete. last_command_result={self.provider.last_command_result}",
            flush=True,
        )
        return True


    def _print_banner(self) -> None:

        print("=" * 50)
        print(" PN-Link MoCap Calibration")
        print("=" * 50)
        print(" N - Start Capture")
        print(" C - Calibrate Motion")
        print(" F/S - Stop Capture")
        print(" Q - Exit")
        print("=" * 50)

    def _timed_out(self) -> bool:

        return time.monotonic() - self._start_time > self.timeout_s

    def _tick_status(self) -> None:

        if self.provider.avatar_seen and not self._avatar_reported:
            print(
                "[Stage 2] Motion data received from device; capture is already running. Press C to calibrate.",
                flush=True,
            )
            self._avatar_reported = True

        progress_events = self.provider.drain_calibration_progress(self._last_progress_seq)
        for progress in progress_events:
            print(progress.text, flush=True)
            if progress.progress == 100 and not self._phase2_reported:
                print(
                    "Waiting for the completion of the calibration of human posture command execution",
                    flush=True,
                )
                self._phase2_reported = True
            self._last_progress_seq = progress.seq

        result = self.provider.last_command_result_detail
        if result is not None and result != self._last_result_detail:
            title = self.provider.command_title(result.command)
            if result.result_code == 0:
                print(f"{title} done.", flush=True)
                if title == "Start Capture":
                    print("[Stage 2] Capture started. Waiting for motion data...", flush=True)
            elif title == "Start Capture" and result.result_message and "running" in result.result_message.lower():
                print("Start Capture already running. Press C to calibrate.", flush=True)
            else:
                print(
                    f"{title} failed: code={result.result_code} message={result.result_message}",
                    flush=True,
                )
            self._last_result_detail = result

    def _send_command(self, title: str, send) -> None:

        try:
            print(f"\n> Key: {title}", flush=True)
            send()
            print(f"Pending command {self.provider.command_title()} is running.", flush=True)
        except Exception as exc:
            print(f"{title} failed: {exc}", flush=True)


def run_local_human_viewer(
    provider: PNLinkMocapProvider,
    *,
    viewer_period: float,
    print_interval: int,
    align_ground: bool,
    ground_z: float | None,
    ground_target_z: float,
    ground_calibration_frames: int,
    use_viser: bool = False,
    viser_host: str = "0.0.0.0",
    viser_port: int = 8080,
) -> int:

    from booster_deploy.inputs.mocap.adapter import skeleton_connections
    from booster_deploy.inputs.mocap.ground_alignment import MocapGroundAligner
    from booster_deploy.visualization import create_human_frame_viewer

    ground_aligner: MocapGroundAligner | None = None
    if align_ground:
        ground_aligner = MocapGroundAligner(
            calibration_frames=ground_calibration_frames,
            target_ground_z=ground_target_z,
            ground_z=ground_z,
        )
        print(
            "Mocap ground alignment enabled | target_z=%.6f%s"
            % (
                ground_aligner.target_ground_z,
                "" if ground_aligner.ground_z is None else " ground_z=%.6f" % ground_aligner.ground_z,
            ),
            flush=True,
        )

    viewer_backend = "viser" if use_viser else "mujoco"
    print(
        f"Waiting for first PN-Link HumanFrame before opening {viewer_backend} viewer...",
        flush=True,
    )
    first_frame, first_ts, first_seq = provider.get_frame_packet()
    print(
        f"Received first frame | seq={first_seq} ts={first_ts:.6f} joints={len(first_frame)}",
        flush=True,
    )

    viewer = create_human_frame_viewer(
        viewer_backend,
        title="PN-Link MoCap HumanFrame",
        skeleton_connections=skeleton_connections(),
        release_camera_after_first_frame=True,
        host=viser_host,
        port=viser_port,
    )

    frame_count = 0
    last_seq: int | None = None
    last_timestamp: float | None = None
    reported_ground_alignment = bool(ground_aligner is not None and ground_aligner.is_calibrated)
    start_time = time.monotonic()
    try:
        while viewer.is_running():
            frame, timestamp, seq = _wait_next_frame(provider, last_seq=last_seq, last_timestamp=last_timestamp)
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
                        ),
                        flush=True,
                    )
                    reported_ground_alignment = True
            viewer.step(frame, rate_limit_s=viewer_period)
            frame_count += 1
            last_seq = seq
            last_timestamp = timestamp
            if print_interval > 0 and frame_count % max(int(print_interval), 1) == 0:
                elapsed_s = max(time.monotonic() - start_time, 1e-9)
                print(
                    "seq=%s provider_fps=%.2f viewer_fps=%.2f joints=%d"
                    % (seq, provider.fps, frame_count / elapsed_s, len(frame)),
                    flush=True,
                )
    except KeyboardInterrupt:
        print("\nStopped.", flush=True)
    finally:
        viewer.close()
    return frame_count


def _wait_next_frame(
    provider: PNLinkMocapProvider,
    *,
    last_seq: int | None,
    last_timestamp: float | None,
) -> tuple[dict, float, int]:

    while True:
        frame, timestamp, seq = provider.get_frame_packet()
        if last_seq is None and last_timestamp is None:
            return frame, timestamp, seq
        if seq is not None and last_seq is not None and seq > last_seq:
            return frame, timestamp, seq
        if seq is None and last_timestamp is not None and timestamp > last_timestamp + 1e-9:
            return frame, timestamp, seq
        time.sleep(0.001)


def _wait_for(predicate, *, timeout_s: float, label: str) -> None:

    deadline = time.monotonic() + float(timeout_s)
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.05)
    raise TimeoutError(f"Timed out waiting for {label}")


def _enter_cbreak():

    if not sys.stdin.isatty():
        return None
    old_settings = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())
    new_settings = termios.tcgetattr(sys.stdin)
    new_settings[3] = new_settings[3] & ~termios.ECHO
    termios.tcsetattr(sys.stdin, termios.TCSANOW, new_settings)
    return old_settings


def _read_key() -> str | None:

    if not select.select([sys.stdin], [], [], 0)[0]:
        return None
    key = sys.stdin.read(1)
    if key in ("\n", "\r"):
        return None
    return key


def main() -> None:

    args = parse_args()
    provider = PNLinkMocapProvider(
        local_ip=args.local_ip,
        local_port=args.local_port,
        server_ip=args.server_ip,
        server_port=args.server_port,
        skip_calibration=args.skip_calibration,
        auto_start_capture=False,
        timeout=1.0,
        poll_hz=args.poll_hz,
    )
    print(
        "PN-Link opened | local=%s:%d server=%s:%d"
        % (args.local_ip, args.local_port, args.server_ip, args.server_port),
        flush=True,
    )

    old_settings = None
    calibrated = bool(args.skip_calibration)
    try:
        if not args.skip_calibration:
            if args.auto:
                state_machine = CalibrationStateMachine(provider, timeout_s=float(args.timeout))
                calibrated = state_machine.run_auto()
            else:
                old_settings = _enter_cbreak()
                state_machine = CalibrationStateMachine(provider, timeout_s=float(args.timeout))
                calibrated = state_machine.run_interactive()
                if old_settings is not None:
                    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
                    old_settings = None

        if not calibrated:
            return
        if args.no_viewer:
            return

        rendered = run_local_human_viewer(
            provider,
            viewer_period=args.viewer_period,
            print_interval=args.print_interval,
            align_ground=args.align_ground,
            ground_z=args.ground_z,
            ground_target_z=args.ground_target_z,
            ground_calibration_frames=args.ground_calibration_frames,
            use_viser=args.viser,
            viser_host=args.viser_host,
            viser_port=args.viser_port,
        )
        print(f"Visualized {rendered} live PN-Link mocap frames.", flush=True)
    except KeyboardInterrupt:
        print("\nStopped.", flush=True)
    finally:
        if old_settings is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        provider.close()

    if args.safe_visualizer_exit and not args.no_viewer and not args.viser:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
