
from __future__ import annotations

from enum import Enum


class RealControlState(str, Enum):

    WAIT_START = "wait_start"
    PREPARE = "prepare"
    WAIT_ACTIVATE = "wait_activate"
    POLICY_STANDING = "policy_standing"
    POLICY_TRACKING = "policy_tracking"
    STOPPING = "stopping"
