"""Offline-replay session flags shared by the launchers and the policies.

The launcher sets ``BOOSTER_OFFLINE_REPLAY=1`` only for a replayed clip, so a
live teleoperation run can never end its own session. The policy uses the same
flag to decide whether a stalled input means "the clip finished".
"""

from __future__ import annotations

import os

OFFLINE_REPLAY_ENV = "BOOSTER_OFFLINE_REPLAY"
OFFLINE_END_TIMEOUT_ENV = "BOOSTER_OFFLINE_END_TIMEOUT_S"
DEFAULT_OFFLINE_END_TIMEOUT_S = 1.0


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None or value == "":
        return bool(default)
    return value.strip().lower() in {"1", "true", "yes", "on"}


def offline_replay_enabled() -> bool:
    return _env_bool(OFFLINE_REPLAY_ENV)


def offline_replay_end_timeout_s() -> float:
    value = os.getenv(OFFLINE_END_TIMEOUT_ENV)
    if value is None or value.strip() == "":
        return DEFAULT_OFFLINE_END_TIMEOUT_S
    try:
        timeout = float(value)
    except ValueError:
        return DEFAULT_OFFLINE_END_TIMEOUT_S
    return timeout if timeout > 0.0 else DEFAULT_OFFLINE_END_TIMEOUT_S


__all__ = [
    "DEFAULT_OFFLINE_END_TIMEOUT_S",
    "OFFLINE_END_TIMEOUT_ENV",
    "OFFLINE_REPLAY_ENV",
    "offline_replay_enabled",
    "offline_replay_end_timeout_s",
]
