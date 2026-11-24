"""Run offline input, live retargeting, and the paired Viser viewers."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_SPECS = {
    "pico": ("pico_rt", 5555, "pico_rt_zmq_human_viewer.py", "offline_pico_publisher.py"),
    "mocap": ("mocap_rt", 5557, "mocap_rt_zmq_human_viewer.py", "offline_mocap_publisher.py"),
}


def _port_available(port: int, host: str = "127.0.0.1") -> bool:
    """Return whether a local TCP port can be bound for the child pipeline."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((host, int(port)))
        except OSError:
            return False
    return True


def _select_local_port(preferred: int, *, label: str, reserved: set[int] | None = None) -> int:
    """Use the requested port, or the next available private port."""
    if not 1 <= int(preferred) <= 65535:
        raise ValueError(f"invalid {label} port: {preferred}")
    reserved = reserved or set()
    if int(preferred) not in reserved and _port_available(preferred):
        return int(preferred)
    for candidate in range(int(preferred) + 1, min(int(preferred) + 101, 65536)):
        if candidate not in reserved and _port_available(candidate):
            print(
                f"Warning: {label} port {preferred} is busy; using {candidate}",
                file=sys.stderr,
                flush=True,
            )
            return candidate
    raise RuntimeError(
        f"{label} port {preferred} is busy and no free port was found in "
        f"{preferred + 1}-{preferred + 100}"
    )


def build_offline_visualization_commands(
    source: str,
    recording: str | Path,
    *,
    python: str = sys.executable,
    viser_host: str = "0.0.0.0",
    human_port: int = 8080,
    robot_port: int = 8081,
    input_port: int | None = None,
    reference_port: int = 5556,
    speed: float = 1.0,
    fps: float | None = None,
    loop: bool = True,
) -> dict[str, list[str]]:
    try:
        input_source, default_input_port, human_viewer_name, publisher_name = INPUT_SPECS[source]
    except KeyError as exc:
        raise ValueError(f"unsupported offline source {source!r}; expected pico or mocap") from exc

    selected_input_port = default_input_port if input_port is None else int(input_port)
    recording_path = str(Path(recording).expanduser().resolve())
    human_viewer = REPO_ROOT / "scripts" / "teleop" / human_viewer_name
    publisher = REPO_ROOT / "scripts" / "diagnostics" / publisher_name
    robot_viewer = REPO_ROOT / "scripts" / "teleop" / "retarget_rt_zmq_robot_viewer.py"
    worker = REPO_ROOT / "scripts" / "teleop" / "retarget_rt_zmq_worker.py"

    publisher_args = [str(publisher), recording_path, "--bind", "127.0.0.1", "--port", str(selected_input_port), "--start-delay", "1.5", "--speed", str(speed)]
    if fps is not None:
        publisher_args.extend(("--fps", str(fps)))
    if not loop:
        publisher_args.append("--no-loop")

    return {
        "human viewer": [
            python,
            str(human_viewer),
            "--host",
            "127.0.0.1",
            "--port",
            str(selected_input_port),
            "--viser",
            "--viser-host",
            viser_host,
            "--viser-port",
            str(human_port),
        ],
        "robot viewer": [
            python,
            str(robot_viewer),
            "--host",
            "127.0.0.1",
            "--port",
            str(reference_port),
            "--viser-host",
            viser_host,
            "--viser-port",
            str(robot_port),
        ],
        "retarget worker": [
            python,
            str(worker),
            "--input-source",
            input_source,
            "--input-host",
            "127.0.0.1",
            "--input-port",
            str(selected_input_port),
            "--output-bind",
            "127.0.0.1",
            "--output-port",
            str(reference_port),
            "--robot-name",
            "t2",
        ],
        "offline publisher": [python, *publisher_args],
    }


def _parse_args(source: str, argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"Show offline {source} input and the T2 robot after the realtime GMR retarget pipeline."
    )
    parser.add_argument("recording", type=Path, help="Raw PICO NPZ or PN-Link BVH recording.")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier.")
    parser.add_argument("--fps", type=float, default=None, help="Override source playback FPS.")
    parser.add_argument(
        "--loop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Loop the recording until the viewer process is stopped.",
    )
    parser.add_argument("--viser-host", default="0.0.0.0", help="Viser bind host for both viewers.")
    parser.add_argument("--human-port", type=int, default=8080)
    parser.add_argument("--robot-port", type=int, default=8081)
    parser.add_argument("--input-port", type=int, default=None, help="Override the input PUB port.")
    parser.add_argument("--reference-port", type=int, default=5556)
    return parser.parse_args(argv)


def _stop_processes(processes: list[tuple[str, subprocess.Popen]]) -> None:
    for _name, process in processes:
        if process.poll() is None:
            process.terminate()
    deadline = time.monotonic() + 3.0
    for _name, process in processes:
        if process.poll() is None:
            try:
                process.wait(timeout=max(deadline - time.monotonic(), 0.0))
            except subprocess.TimeoutExpired:
                process.kill()
    for _name, process in processes:
        try:
            process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def run_offline_visualization(source: str, args: argparse.Namespace) -> None:
    recording = args.recording.expanduser().resolve()
    if not recording.is_file():
        raise FileNotFoundError(f"recording not found: {recording}")
    if not math.isfinite(args.speed) or args.speed <= 0.0:
        raise ValueError("--speed must be finite and positive")
    if args.fps is not None and not math.isfinite(args.fps):
        raise ValueError("--fps override must be finite")
    # Resolve ports before starting any child. The old visualizer could start
    # three children and only then discover that the publisher cannot bind 5555.
    selected_input_port = _select_local_port(
        args.input_port if args.input_port is not None else INPUT_SPECS[source][1],
        label="input",
    )
    selected_reference_port = _select_local_port(args.reference_port, label="reference", reserved={selected_input_port})
    selected_human_port = _select_local_port(args.human_port, label="human Viser", reserved={selected_input_port, selected_reference_port})
    selected_robot_port = _select_local_port(args.robot_port, label="robot Viser", reserved={selected_input_port, selected_reference_port, selected_human_port})
    ports = {
        "input": selected_input_port,
        "reference": selected_reference_port,
        "human Viser": selected_human_port,
        "robot Viser": selected_robot_port,
    }
    for name, port in ports.items():
        if not 1 <= int(port) <= 65535:
            raise ValueError(f"invalid {name} port: {port}")
    if len(set(map(int, ports.values()))) != len(ports):
        raise ValueError("input, reference, human Viser, and robot Viser ports must be distinct")

    commands = build_offline_visualization_commands(
        source,
        recording,
        viser_host=args.viser_host,
        human_port=selected_human_port,
        robot_port=selected_robot_port,
        input_port=selected_input_port,
        reference_port=selected_reference_port,
        speed=args.speed,
        fps=args.fps,
        loop=args.loop,
    )
    child_env = os.environ.copy()
    existing_pythonpath = child_env.get("PYTHONPATH", "")
    child_env["PYTHONPATH"] = str(REPO_ROOT) + (os.pathsep + existing_pythonpath if existing_pythonpath else "")

    print(f"Offline {source} recording: {recording}", flush=True)
    print(
        "Viser viewers: "
        f"human http://localhost:{selected_human_port}/ | robot http://localhost:{selected_robot_port}/",
        flush=True,
    )
    print(
        f"-L {selected_human_port}:127.0.0.1:{selected_human_port} "
        f"-L {selected_robot_port}:127.0.0.1:{selected_robot_port}",
        flush=True,
    )

    processes: list[tuple[str, subprocess.Popen]] = []
    command_order = ("human viewer", "robot viewer", "retarget worker", "offline publisher")
    try:
        for name in command_order:
            process = subprocess.Popen(
                commands[name],
                cwd=REPO_ROOT,
                env=child_env,
            )
            processes.append((name, process))
            print(f"Started {name} pid={process.pid}", flush=True)

        publisher_finished_at: float | None = None
        while True:
            for name, process in processes:
                exit_code = process.poll()
                if exit_code is None:
                    continue
                if name == "offline publisher" and exit_code == 0 and not args.loop:
                    publisher_finished_at = publisher_finished_at or time.monotonic()
                    if time.monotonic() - publisher_finished_at >= 1.0:
                        return
                    continue
                raise RuntimeError(f"{name} exited with status {exit_code}")
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Stopping offline viewers and retarget pipeline...", flush=True)
    finally:
        _stop_processes(processes)


def main_pico(argv: list[str] | None = None) -> None:
    run_offline_visualization("pico", _parse_args("pico", argv))


def main_mocap(argv: list[str] | None = None) -> None:
    run_offline_visualization("mocap", _parse_args("mocap", argv))


__all__ = ["build_offline_visualization_commands", "main_mocap", "main_pico", "run_offline_visualization"]
