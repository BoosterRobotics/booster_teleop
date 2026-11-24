
from __future__ import annotations

import os
from pathlib import Path

from .project_path import PROJECT_ROOT, PathValue, resolve_project_path


DEFAULT_T2_MJCF_PATH = (
    PROJECT_ROOT / "booster_deploy" / "assets" / "t2" / "t2_31dof_fc.xml"
)
T2_REVO2_MJCF_PATH = DEFAULT_T2_MJCF_PATH.with_name("t2_31dof_revo2hand.xml")


def t2_mjcf_path(root: Path = PROJECT_ROOT) -> Path:
    """Select the display/simulation model for the launcher's hand switch."""
    path = T2_REVO2_MJCF_PATH if os.getenv("BOOSTER_HAND_CONTROL_ENABLED") == "1" else DEFAULT_T2_MJCF_PATH
    return root / path.relative_to(PROJECT_ROOT)


def resolve_mjcf_path(
    configured_path: PathValue,
) -> Path:
    candidate = resolve_project_path(configured_path)
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"MuJoCo XML not found: '{candidate}'")


__all__ = ["DEFAULT_T2_MJCF_PATH", "T2_REVO2_MJCF_PATH", "resolve_mjcf_path", "t2_mjcf_path"]
