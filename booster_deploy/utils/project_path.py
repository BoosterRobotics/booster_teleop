
from __future__ import annotations

import os
from pathlib import Path
from typing import TypeAlias


PathValue: TypeAlias = str | os.PathLike[str]


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_project_path(value: PathValue) -> Path:

    raw_value = os.fspath(value)
    if not raw_value.strip():
        raise ValueError("path must not be empty")
    expanded = os.path.expandvars(os.path.expanduser(raw_value))
    path = Path(expanded)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


__all__ = ["PROJECT_ROOT", "PathValue", "resolve_project_path"]
