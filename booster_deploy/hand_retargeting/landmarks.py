
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from booster_deploy.types import HumanFrame

RIGHT_HAND_LANDMARK_NAMES: tuple[str, ...] = (
    "RightHand",
    "RightHandThumb1",
    "RightHandThumb2",
    "RightHandThumb3",
    "RightHandThumb3_EndSite",
    "RightHandIndex1",
    "RightHandIndex2",
    "RightHandIndex3",
    "RightHandIndex3_EndSite",
    "RightHandMiddle1",
    "RightHandMiddle2",
    "RightHandMiddle3",
    "RightHandMiddle3_EndSite",
    "RightHandRing1",
    "RightHandRing2",
    "RightHandRing3",
    "RightHandRing3_EndSite",
    "RightHandPinky1",
    "RightHandPinky2",
    "RightHandPinky3",
    "RightHandPinky3_EndSite",
)
LEFT_HAND_LANDMARK_NAMES: tuple[str, ...] = (
    "LeftHand",
    "LeftHandThumb1",
    "LeftHandThumb2",
    "LeftHandThumb3",
    "LeftHandThumb3_EndSite",
    "LeftHandIndex1",
    "LeftHandIndex2",
    "LeftHandIndex3",
    "LeftHandIndex3_EndSite",
    "LeftHandMiddle1",
    "LeftHandMiddle2",
    "LeftHandMiddle3",
    "LeftHandMiddle3_EndSite",
    "LeftHandRing1",
    "LeftHandRing2",
    "LeftHandRing3",
    "LeftHandRing3_EndSite",
    "LeftHandPinky1",
    "LeftHandPinky2",
    "LeftHandPinky3",
    "LeftHandPinky3_EndSite",
)

HAND_LANDMARK_NAMES: dict[str, tuple[str, ...]] = {
    "right": RIGHT_HAND_LANDMARK_NAMES,
    "left": LEFT_HAND_LANDMARK_NAMES,
}

_FINGER_TIP_FALLBACKS: dict[int, tuple[str, str]] = {
    4: ("Thumb3", "Thumb2"),
    8: ("Index3", "Index2"),
    12: ("Middle3", "Middle2"),
    16: ("Ring3", "Ring2"),
    20: ("Pinky3", "Pinky2"),
}


def normalize_hand_side(hand_side: str) -> str:
    value = str(hand_side).strip().lower()
    if value in ("left", "l"):
        return "left"
    if value in ("right", "r"):
        return "right"
    raise ValueError(f"Unsupported hand side: {hand_side!r}")


def _joint_position(frame: Mapping[str, tuple[np.ndarray, np.ndarray]], name: str) -> np.ndarray:
    try:
        pos, _ = frame[name]
    except KeyError as exc:
        raise KeyError(name) from exc
    return np.asarray(pos, dtype=np.float64)


def _tip_position(
    frame: Mapping[str, tuple[np.ndarray, np.ndarray]],
    *,
    index: int,
    names: tuple[str, ...],
    tip_mode: str,
) -> np.ndarray:
    tip_name = names[index]
    if tip_name in frame:
        return _joint_position(frame, tip_name)

    distal_name = names[index - 1]
    middle_name = names[index - 2]
    distal = _joint_position(frame, distal_name)
    if tip_mode == "distal":
        return distal.copy()

    middle = None
    for candidate in (middle_name,):
        if candidate in frame:
            middle = _joint_position(frame, candidate)
            break
    if middle is None:
        return distal.copy()

    direction = distal - middle
    norm = float(np.linalg.norm(direction))
    if norm < 1e-12:
        return distal.copy()
    return distal + direction


def _frame_to_landmarks(
    frame: Mapping[str, tuple[np.ndarray, np.ndarray]],
    *,
    hand_side: str,
    tip_mode: str,
) -> np.ndarray:
    side = normalize_hand_side(hand_side)
    names = HAND_LANDMARK_NAMES[side]
    landmarks = np.empty((21, 3), dtype=np.float64)
    missing: list[str] = []
    for index, name in enumerate(names):
        if index in _FINGER_TIP_FALLBACKS:
            landmarks[index] = _tip_position(
                frame,
                index=index,
                names=names,
                tip_mode=tip_mode,
            )
            continue
        if name not in frame:
            missing.append(name)
            continue
        landmarks[index] = _joint_position(frame, name)
    if missing:
        sample = ", ".join(sorted(frame)[:12])
        raise KeyError(
            f"hand frame is missing required joints: {missing}. Available joint sample: {sample}"
        )
    if not np.all(np.isfinite(landmarks)):
        raise ValueError(f"hand landmarks contain non-finite values for hand_side={side!r}")
    return landmarks


def human_frame_to_landmarks(
    frame: HumanFrame,
    hand_side: str,
    *,
    tip_mode: str = "extrapolate",
) -> np.ndarray:

    return _frame_to_landmarks(frame, hand_side=hand_side, tip_mode=tip_mode)


def bvh_hand_frame_to_landmarks(
    frame: HumanFrame,
    hand_side: str,
    *,
    tip_mode: str = "extrapolate",
) -> np.ndarray:

    return _frame_to_landmarks(frame, hand_side=hand_side, tip_mode=tip_mode)


__all__ = [
    "HAND_LANDMARK_NAMES",
    "LEFT_HAND_LANDMARK_NAMES",
    "RIGHT_HAND_LANDMARK_NAMES",
    "bvh_hand_frame_to_landmarks",
    "human_frame_to_landmarks",
    "normalize_hand_side",
]
