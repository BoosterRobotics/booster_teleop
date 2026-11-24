"""Record the onboard raw PICO stream into the same NPZ format as ``record_pico_raw.py``.

The live publisher already broadcasts every SDK body frame plus the controller
state on ``pico_rt.raw`` (see ``booster_deploy.inputs.pico.raw_stream``). This
recorder subscribes to that stream instead of opening a second
``xrobotoolkit_sdk`` client, so it can run in a teleoperation pane next to the
publisher without touching the SDK.

Recording is driven by the shared chord detector
(:class:`booster_deploy.inputs.pico.recording.RecordingChordDetector`): hold both
triggers and press X to start, hold both and press Y to stop. Every stop writes
one NPZ (``<prefix>_<timestamp>.npz``) and the recorder keeps listening, so a
session can capture several clips.
"""

from __future__ import annotations

import time
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from booster_deploy.types.pico_types import PicoButtonBits, PicoControlFrameNs

from .raw_recording import (
    RECORD_HZ,
    advance_sample_deadline,
    save_raw_recording,
)
from .raw_stream import PICO_RAW_TOPIC, decode_pico_raw
from .recording import DEFAULT_TRIGGER_THRESHOLD, RecordingChordDetector
from booster_deploy.utils.diag_report import below_floor, format_gap, reporter

#: A clip shorter than this is a phantom stop (vendor button glitch), not an edit.
MIN_CLIP_S = 0.5


@dataclass
class RecordedClip:
    poses: list[np.ndarray] = field(default_factory=list)
    body_seq: list[int] = field(default_factory=list)
    source_ns: list[int] = field(default_factory=list)
    body_recv_ns: list[int] = field(default_factory=list)


class PicoRawStreamRecorder:
    """Drain ``pico_rt.raw`` packets and save NPZ clips on the trigger chords."""

    def __init__(
        self,
        *,
        output_dir: Path,
        prefix: str = "pico_raw",
        endpoint: str = "tcp://127.0.0.1:5581",
        topic: bytes | str = PICO_RAW_TOPIC,
        trigger_threshold: float = DEFAULT_TRIGGER_THRESHOLD,
        sample_hz: float = RECORD_HZ,
        print_interval: int = 50,
        keyboard_control: bool = False,
    ) -> None:
        self.output_dir = Path(output_dir).expanduser()
        self.prefix = str(prefix)
        self.endpoint = str(endpoint)
        self.topic = topic.encode() if isinstance(topic, str) else topic
        self.sample_period_ns = int(round(1.0e9 / float(sample_hz)))
        self.print_interval = int(print_interval)
        self.chord = RecordingChordDetector(trigger_threshold)
        self.clip: RecordedClip | None = None
        self.saved: list[Path] = []
        self._clip_started_s = 0.0
        self.frames_seen = 0
        self.rejected = 0
        self._next_sample_ns: int | None = None
        self._last_sample_clock_ns: int | None = None
        self._sample_clock_source: str | None = None
        self._last_body_seq: int | None = None
        self._diag = reporter("record")
        self._ctx = None
        self._sock = None
        # Optional web-dashboard keystrokes ('x' start / 'y' stop).  Disabled unless
        # asked for and stdin is a TTY, so a background recorder never consumes input.
        self._stdin = sys.stdin if (keyboard_control and sys.stdin.isatty()) else None
        # Single-key control needs cbreak: a line-buffered TTY would hold 'x' until
        # Enter, so `tmux send-keys x` from the dashboard never reached the loop.
        self._tty_settings = None
        if self._stdin is not None:
            import termios
            import tty

            try:
                fd = self._stdin.fileno()
                self._tty_settings = termios.tcgetattr(fd)
                tty.setcbreak(fd)
            except (OSError, ValueError, termios.error):
                self._tty_settings = None

    def _restore_tty(self) -> None:
        if self._tty_settings is None or self._stdin is None:
            return
        import termios

        try:
            termios.tcsetattr(self._stdin.fileno(), termios.TCSADRAIN, self._tty_settings)
        except (OSError, ValueError, termios.error):
            pass
        self._tty_settings = None

    # ------------------------------------------------------------------ transport
    def start(self) -> None:
        if self._sock is not None:
            return
        import zmq

        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.SUB)
        self._sock.setsockopt(zmq.RCVHWM, 256)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.subscribe(self.topic)
        self._sock.connect(self.endpoint)

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close(linger=0)
            self._ctx.term()
            self._sock = self._ctx = None
        self._diag.close()

    def _emit_diag(self) -> None:
        """Human line: clip write rate (target 50 Hz) + gap tail; frames on disk."""

        diag = self._diag
        if not diag.due():
            return
        now_ns = time.monotonic_ns()
        window = diag.interval("clip")
        window_hz = window.hz(now_ns)
        gap = window.gap_stats(now_ns)
        # Grid compliance: frames written per clip-second, immune to the catch-up
        # bursts that make a 1 s window read 44 Hz then 57 Hz.
        clip_hz = window_hz
        if self.clip is not None and self._clip_started_s:
            age_s = max(time.monotonic() - self._clip_started_s, 1e-3)
            clip_hz = len(self.clip.poses) / age_s
        stall = window.stall_ms(now_ns)
        skip_hz = diag.rate("skip").hz(now_ns)
        alerts: list[str] = []
        if self.clip is not None:
            # Judge grid compliance in a band: bursts are legitimate catch-up.
            if clip_hz and not 45.0 <= clip_hz <= 55.0:
                alerts.append(f"clip {clip_hz:.1f} Hz not ~50")
            if stall is not None and stall > 200.0:
                alerts.append(f"stall {stall:.0f} ms")
        if self.rejected:
            alerts.append(f"rejected {self.rejected}")
        if self.clip is None:
            screen = "idle"
            state = {"saved": len(self.saved), "rejected": self.rejected, "clip_frames": None}
        else:
            screen = f"REC {len(self.clip.poses):5d} frames   {clip_hz:5.1f} Hz   gap {format_gap(gap)} ms"
            state = {"saved": len(self.saved), "rejected": self.rejected,
                     "clip_frames": len(self.clip.poses),
                     "clip_hz": round(clip_hz, 2),
                     "skip_hz": round(skip_hz, 1)}
        diag.emit(screen, alerts=alerts, extra={"state": state}, now_ns=now_ns)

    # ------------------------------------------------------------------ packet handling
    def handle_packet(self, packet: dict) -> Path | None:
        """Apply one decoded ``pico_rt.raw`` packet; returns a path when a clip was saved."""

        kind = packet.get("kind")
        if kind == "control":
            control = self._control_frame(packet)
            if control is None:
                return None
            events = self.chord.update(control)
            if events:
                # Keep the raw inputs in the log: a phantom chord is a vendor
                # mapping problem and can only be diagnosed from these numbers.
                print(
                    f"[record] chord event={events} buttons={int(control.buttons)}"
                    f"(X={int(bool(control.buttons & PicoButtonBits.X))}"
                    f" Y={int(bool(control.buttons & PicoButtonBits.Y))}"
                    f" A={int(bool(control.buttons & PicoButtonBits.A))}"
                    f" B={int(bool(control.buttons & PicoButtonBits.B))})"
                    f" triggers={control.left_trigger:.2f}/{control.right_trigger:.2f}",
                    flush=True,
                )
            if "c" in events and self.clip is None:
                self.start_clip(reason="chord")
            elif "x" in events and self.clip is not None:
                saved = self.stop_clip(reason="chord")
                if saved is None:
                    self.chord.reset()
                return saved
            return None
        if kind != "body" or self.clip is None:
            return None
        return self._record_body(packet)

    # ------------------------------------------------------------------ clip control
    def start_clip(self, *, reason: str = "chord") -> None:
        """Begin a clip.  ``reason`` is only for the operator log."""
        if self.clip is not None:
            return
        self.clip = RecordedClip()
        self._clip_started_s = time.monotonic()
        # Clip boundaries are not stutter: start a fresh gap window.
        self._diag.interval("clip").reset()
        self._next_sample_ns = None
        self._last_sample_clock_ns = None
        self._sample_clock_source = None
        label = "chord started" if reason == "chord" else f"started ({reason})"
        print(f"[record] {label} | output_dir={self.output_dir}", flush=True)

    def stop_clip(self, *, reason: str = "chord") -> Path | None:
        """Finish the in-flight clip; returns the saved path, or None if rejected."""
        if self.clip is None:
            return None
        age_s = time.monotonic() - getattr(self, "_clip_started_s", 0.0)
        if age_s < MIN_CLIP_S:
            print(
                f"[record] stop ignored ({reason}): clip only {age_s * 1e3:.0f} ms old"
                f" (min {MIN_CLIP_S * 1e3:.0f} ms)",
                flush=True,
            )
            return None
        return self._finish_clip()

    def _poll_keyboard(self) -> None:
        """Dashboard keystrokes: 'x' start, 'y' stop -- identical to the chords."""
        if self._stdin is None:
            return
        import select as _select

        try:
            while True:
                ready, _, _ = _select.select([self._stdin], [], [], 0)
                if not ready:
                    return
                char = self._stdin.read(1)
                if not char:
                    self._stdin = None
                    return
                if char == "x":
                    self.start_clip(reason="dashboard")
                elif char == "y":
                    saved = self.stop_clip(reason="dashboard")
                    if saved is not None:
                        print(f"[record] saved {saved}", flush=True)
        except (OSError, ValueError):
            self._stdin = None

    def _control_frame(self, packet: dict) -> PicoControlFrameNs | None:
        values = packet.get("raw_control") or packet.get("routed_control")
        if not isinstance(values, dict):
            return None
        try:
            return PicoControlFrameNs(
                seq=int(values.get("seq", 0)),
                recv_ns=int(values.get("recv_ns", time.monotonic_ns())),
                source_ns=values.get("source_ns"),
                buttons=PicoButtonBits(int(values.get("buttons", 0))),
                left_axis=tuple(values.get("left_axis") or (0.0, 0.0)),
                right_axis=tuple(values.get("right_axis") or (0.0, 0.0)),
                left_trigger=float(values.get("left_trigger", 0.0)),
                right_trigger=float(values.get("right_trigger", 0.0)),
                left_grip=float(values.get("left_grip", 0.0)),
                right_grip=float(values.get("right_grip", 0.0)),
            )
        except (TypeError, ValueError):
            self.rejected += 1
            return None

    def _record_body(self, packet: dict) -> None:
        try:
            poses = np.asarray(packet["poses"], dtype=np.float64)
        except (KeyError, TypeError, ValueError):
            self.rejected += 1
            return
        if poses.ndim != 2 or poses.shape[0] != 24 or poses.shape[1] < 7:
            self.rejected += 1
            return
        seq = int(packet.get("seq", -1))
        if seq < 0 or (self._last_body_seq is not None and seq <= self._last_body_seq):
            self.rejected += 1
            return
        self._last_body_seq = seq
        source_ns = packet.get("source_ns")
        if source_ns is not None and int(source_ns) > 0:
            clock_ns, clock_source = int(source_ns), "source_ns"
        else:
            clock_ns = int(packet.get("body_recv_ns", time.monotonic_ns()))
            clock_source = "body_recv_ns"
        if (
            self._next_sample_ns is None
            or self._sample_clock_source != clock_source
            or (self._last_sample_clock_ns is not None and clock_ns <= self._last_sample_clock_ns)
        ):
            self._next_sample_ns = clock_ns
            self._sample_clock_source = clock_source
        should_record, self._next_sample_ns = advance_sample_deadline(
            clock_ns, self._next_sample_ns, self.sample_period_ns
        )
        self._last_sample_clock_ns = clock_ns
        if not should_record:
            self._diag.count("skip")
            return

        clip = self.clip
        assert clip is not None
        if clip.poses and poses[:, :7].shape != clip.poses[0].shape:
            raise ValueError(
                f"raw pose shape changed from {clip.poses[0].shape} to {poses[:, :7].shape}"
            )
        clip.poses.append(poses[:, :7].copy())
        clip.body_seq.append(seq)
        clip.source_ns.append(-1 if source_ns is None else int(source_ns))
        clip.body_recv_ns.append(int(packet.get("body_recv_ns", time.monotonic_ns())))
        self.frames_seen += 1
        self._diag.mark("clip")
        if self.print_interval > 0 and len(clip.poses) % self.print_interval == 0:
            print(f"[record] frames={len(clip.poses)} seq={seq}", flush=True)

    # ------------------------------------------------------------------ output
    def _finish_clip(self) -> Path | None:
        clip = self.clip
        self.clip = None
        if clip is None or not clip.poses:
            print("[record] clip stopped without body frames; nothing saved", flush=True)
            return None
        self.output_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        output = self.output_dir / f"{self.prefix}_{stamp}.npz"
        suffix = 1
        while output.exists():
            suffix += 1
            output = self.output_dir / f"{self.prefix}_{stamp}_{suffix:02d}.npz"
        save_raw_recording(
            output,
            poses=clip.poses,
            body_seq=clip.body_seq,
            source_ns=clip.source_ns,
            body_recv_ns=clip.body_recv_ns,
            fallback_hz=float(RECORD_HZ),
        )
        self.saved.append(output)
        print(f"[record] saved {len(clip.poses)} frames -> {output}", flush=True)
        return output

    def save_pending(self) -> Path | None:
        """Flush an in-flight clip (Ctrl-C while recording)."""

        if self.clip is None:
            return None
        return self._finish_clip()


    # ------------------------------------------------------------------ loop
    def run_forever(self, *, idle_print_interval_s: float = 5.0) -> None:
        """Blocking receive loop; Ctrl-C flushes an in-flight clip."""

        import zmq

        self.start()
        last_idle_print = 0.0
        packets = 0
        try:
            while True:
                self._poll_keyboard()
                try:
                    parts = self._sock.recv_multipart(flags=zmq.NOBLOCK)
                except zmq.Again:
                    now = time.monotonic()
                    if now - last_idle_print >= idle_print_interval_s:
                        last_idle_print = now
                        state = "recording" if self.clip is not None else "idle"
                        print(
                            f"[record] state={state} packets={packets} rejected={self.rejected} "
                            f"saved={len(self.saved)} | source={self.endpoint}/{self.topic.decode()}",
                            flush=True,
                        )
                    self._emit_diag()
                    time.sleep(0.005)
                    continue
                if len(parts) == 2 and parts[0] != self.topic:
                    continue
                try:
                    packet = decode_pico_raw(parts[-1])
                except ValueError:
                    self.rejected += 1
                    continue
                packets += 1
                self._diag.count("pkt")
                self.handle_packet(packet)
                self._emit_diag()
        except KeyboardInterrupt:
            print("\n[record] stopping on Ctrl-C", flush=True)
        finally:
            self._restore_tty()
            saved = self.save_pending()
            self.close()
            if saved is None and not self.saved:
                print("[record] no clip was saved", flush=True)


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Record pico_rt.raw into record_pico_raw-compatible NPZ clips."
    )
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="Directory for the clip NPZ files.")
    parser.add_argument("--prefix", default="pico_raw", help="Clip filename prefix.")
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:5581",
                        help="pico_rt.raw endpoint of the running teleoperation stack.")
    parser.add_argument("--topic", default=PICO_RAW_TOPIC.decode(), help="Raw stream topic.")
    parser.add_argument("--trigger-threshold", type=float, default=DEFAULT_TRIGGER_THRESHOLD,
                        help="Both triggers must exceed this value to arm the chord.")
    parser.add_argument("--sample-hz", type=float, default=RECORD_HZ,
                        help="Resample the incoming stream onto this grid.")
    parser.add_argument("--print-interval", type=int, default=50,
                        help="Print every N recorded frames; <=0 disables.")
    parser.add_argument("--keyboard-control", action="store_true",
                        help="Accept 'x' (start) / 'y' (stop) on stdin, for the web dashboard.")
    args = parser.parse_args(argv)

    recorder = PicoRawStreamRecorder(
        output_dir=args.output_dir,
        prefix=args.prefix,
        endpoint=args.endpoint,
        topic=args.topic,
        trigger_threshold=float(args.trigger_threshold),
        sample_hz=float(args.sample_hz),
        print_interval=int(args.print_interval),
        keyboard_control=bool(args.keyboard_control),
    )
    print(
        f"[record] ready | source={args.endpoint}/{args.topic} output_dir={recorder.output_dir} "
        "| hold both triggers + X to start, hold both + Y to stop",
        flush=True,
    )
    recorder.run_forever()


if __name__ == "__main__":
    main()
