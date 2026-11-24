"""Record clips from the live publisher's existing PN-Link capture."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue
from threading import Thread
from time import monotonic_ns

from booster_deploy.inputs.control.start_gate import ButtonEdgeDetector, RobotButtonMonitor, RobotButtonMonitorCfg
from booster_deploy.types.mocap_types import MocapBodyRawFrameNs
from booster_deploy.utils.diag_report import reporter
from booster_deploy.utils.project_path import resolve_project_path

from .bvh_recording import BvhRecorder


class MocapLiveRecorder:
    """Receive robot button edges and sample raw body frames off the publisher thread."""

    FPS = 50
    PERIOD_NS = 1_000_000_000 // FPS

    def __init__(self, output_dir: Path, latest_body: Callable[[], MocapBodyRawFrameNs | None]) -> None:
        self._output_dir = resolve_project_path(output_dir)
        self._latest_body = latest_body
        self._edges = ButtonEdgeDetector(("x", "y"))
        self._commands: Queue[tuple[str, int]] = Queue()
        self._diag = reporter("record")
        self._saved = 0
        self._monitor = RobotButtonMonitor(
            RobotButtonMonitorCfg(node_name="mocap_recording_buttons"), on_state=self._on_state,
        )
        self._thread = Thread(target=self._run, name="mocap-bvh-recorder", daemon=True)

    def start(self) -> None:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._monitor.start()
        self._monitor.wait_until_ready()
        self._thread.start()

    def close(self) -> None:
        self._monitor.stop()
        if self._thread.ident is not None:
            self._commands.put(("close", monotonic_ns()))
            self._thread.join()

    def _on_state(self, pressed: dict[str, bool]) -> None:
        rising = self._edges.update(pressed)
        if not (pressed["lt"] and pressed["rt"]) or (pressed["b"] and pressed["x"]):
            return  # Preserve B+X shutdown; it must not start a clip.
        if "y" in rising:
            self._commands.put(("y", monotonic_ns()))
        elif "x" in rising:
            self._commands.put(("x", monotonic_ns()))

    def _run(self) -> None:
        diag = self._diag
        diag.rate("clip")
        clip: BvhRecorder | None = None
        started_ns = next_ns = 0
        try:
            while True:
                if diag.due():
                    frames = None if clip is None else clip.frame_count
                    diag.emit("idle" if frames is None else f"REC {frames} frames",
                              extra={"state": {"clip_frames": frames, "saved": self._saved}})
                timeout = 1.0 if clip is None else max(0, next_ns - monotonic_ns()) / 1e9
                try:
                    command, stamp_ns = self._commands.get(timeout=timeout)
                except Empty:
                    if clip is None:
                        continue
                    now_ns = monotonic_ns()
                    next_ns = max(next_ns + self.PERIOD_NS, now_ns + self.PERIOD_NS // 2)
                    body = self._latest_body()
                    if body is not None and body.body_recv_ns >= started_ns:
                        clip.add_frame(body.raw_joints)
                        diag.count("clip", now_ns)
                    continue
                if command == "close":
                    return
                if command == "x" and clip is None:
                    clip = BvhRecorder(target_fps=self.FPS)
                    started_ns, next_ns = stamp_ns, monotonic_ns()
                    print("[mocap-record] recording BVH", flush=True)
                elif command == "y" and clip is not None:
                    self._save(clip)
                    clip = None
        finally:
            if clip is not None:
                self._save(clip)
            diag.close()

    def _save(self, clip: BvhRecorder) -> None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = f"mocap_raw_{stamp}"
        path = self._output_dir / f"{prefix}.bvh"
        suffix = 1
        while path.exists():
            suffix += 1
            path = self._output_dir / f"{prefix}_{suffix:02d}.bvh"
        pending = path.with_suffix(".part")
        try:
            if clip.save(pending) is not None:
                pending.replace(path)
                self._saved += 1
                print(f"[mocap-record] saved {path}", flush=True)
        except OSError as exc:
            print(f"[mocap-record] could not save {path}: {exc}", flush=True)
