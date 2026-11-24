"""Detect raw clip recording chords and keep them out of robot controls."""

from __future__ import annotations

from dataclasses import replace

from booster_deploy.types.pico_types import PicoButtonBits, PicoControlFrameNs

# Recording chords: the trigger modifier plus a non-safety face button. A/B are
# deliberately excluded -- they drive Tracking entry and the return to Walking.
EVENTS = (
    (PicoButtonBits.X, "c"),
    (PicoButtonBits.Y, "x"),
)
CHORD_BUTTONS = PicoButtonBits.X | PicoButtonBits.Y
DEFAULT_TRIGGER_THRESHOLD = 0.7


def modifier_held(control: PicoControlFrameNs,
                  threshold: float = DEFAULT_TRIGGER_THRESHOLD) -> bool:
    """True when both triggers are held past ``threshold`` (the chord modifier)."""

    return float(control.left_trigger) >= threshold and float(control.right_trigger) >= threshold


class RecordingChordDetector:
    """Rising-edge chord buttons gated by the both-triggers modifier."""

    def __init__(
        self,
        trigger_threshold: float = DEFAULT_TRIGGER_THRESHOLD,
        stable_frames: int = 2,
        release_threshold: float | None = None,
    ) -> None:
        if not 0 < trigger_threshold <= 1:
            raise ValueError("Trigger threshold must be in (0, 1]")
        self.trigger_threshold = trigger_threshold
        # Hysteresis: arm at ``trigger_threshold``, disarm only below
        # ``release_threshold``.  On real hardware a held trigger dips to ~0.65,
        # and a hard threshold would drop the modifier mid-chord (and stop
        # masking X/Y from the robot).
        self.release_threshold = (
            max(trigger_threshold - 0.2, 0.0) if release_threshold is None
            else float(release_threshold)
        )
        # A chord must survive ``stable_frames`` consecutive control frames:
        # single-frame vendor glitches must not start or stop a clip.
        self.stable_frames = max(1, int(stable_frames))
        self._last_face_buttons = PicoButtonBits.NONE
        self._pending: dict[str, int] = {}
        self._armed = False

    @property
    def last_face_buttons(self) -> PicoButtonBits:
        return self._last_face_buttons

    @property
    def armed(self) -> bool:
        """True while the both-trigger modifier is held (with hysteresis)."""

        return self._armed

    def _update_armed(self, control: PicoControlFrameNs) -> bool:
        left = float(control.left_trigger)
        right = float(control.right_trigger)
        if left >= self.trigger_threshold and right >= self.trigger_threshold:
            self._armed = True
        elif left < self.release_threshold or right < self.release_threshold:
            self._armed = False
        return self._armed

    def reset(self) -> None:
        self._last_face_buttons = PicoButtonBits.NONE
        self._pending.clear()
        self._armed = False

    def update(self, control: PicoControlFrameNs) -> tuple[str, ...]:
        """Return the recording events produced by this control frame."""

        face = PicoButtonBits(int(control.buttons) & int(CHORD_BUTTONS))
        rising = PicoButtonBits(int(face) & ~int(self._last_face_buttons))
        self._last_face_buttons = face
        if not self._update_armed(control):
            self._pending.clear()
            return ()
        fired: list[str] = []
        for button, event in EVENTS:
            if not face & button:
                self._pending.pop(event, None)
                continue
            if rising & button:
                self._pending[event] = 1
            elif event in self._pending:
                self._pending[event] += 1
            if self._pending.get(event, 0) >= self.stable_frames:
                fired.append(event)
                self._pending.pop(event, None)
        return tuple(fired)


class PicoRecordingControlRouter:
    """Mask recording chords out of the control frame forwarded to the robot."""

    def __init__(self, trigger_threshold: float = DEFAULT_TRIGGER_THRESHOLD) -> None:
        self._detector = RecordingChordDetector(trigger_threshold)
        self._mask_until_release = False

    def route(self, frame: PicoControlFrameNs) -> PicoControlFrameNs:
        """Mask trigger+X/Y until the face buttons are released."""
        buttons = frame.buttons
        chord_face = PicoButtonBits(int(buttons) & int(CHORD_BUTTONS))
        if self._detector._update_armed(frame):
            self._mask_until_release |= chord_face != PicoButtonBits.NONE
            return replace(
                frame, buttons=PicoButtonBits(int(buttons) & ~int(CHORD_BUTTONS))
            )
        if self._mask_until_release:
            if chord_face != PicoButtonBits.NONE:
                return replace(
                    frame, buttons=PicoButtonBits(int(buttons) & ~int(CHORD_BUTTONS))
                )
            self._mask_until_release = False
        return frame


__all__ = [
    "CHORD_BUTTONS",
    "DEFAULT_TRIGGER_THRESHOLD",
    "EVENTS",
    "PicoRecordingControlRouter",
    "RecordingChordDetector",
    "modifier_held",
]
