"""Browser visualization component.

This package owns render-facing model/frame contracts.  It intentionally does not
own teleoperation control state or the dashboard HTTP server.
"""

from .frames import human_frame_payload, qpos_frame_payload, reference_frame_payload
from .model import build_model_manifest
from .store import LatestFrameStore

__all__ = [
    "LatestFrameStore",
    "build_model_manifest",
    "human_frame_payload",
    "qpos_frame_payload",
    "reference_frame_payload",
]
