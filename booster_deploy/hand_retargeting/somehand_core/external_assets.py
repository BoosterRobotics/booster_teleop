
from __future__ import annotations

from pathlib import Path
from typing import Iterable

PACKAGE_ROOT = Path(__file__).resolve().parent
RESOURCE_ROOT = PACKAGE_ROOT.parent / "resources"
SOURCE_ROOT = PACKAGE_ROOT.parents[2]
DATA_ROOT = RESOURCE_ROOT
PROJECT_ROOT = SOURCE_ROOT
DEFAULT_MODELSCOPE_REPO_ID = ""
DEFAULT_HUGGINGFACE_REPO_ID = ""


class AssetEntry:
    def __init__(self, remote_path: str, local_path: str, mode: str = "copy") -> None:
        self.remote_path = remote_path
        self.local_path = local_path
        self.mode = mode


ASSET_GROUPS: dict[str, list[AssetEntry]] = {}


def iter_asset_entries(groups: Iterable[str] | None = None) -> list[tuple[str, AssetEntry]]:
    return []


def resolve_asset_path(local_path: str | Path) -> Path:
    candidate = Path(local_path)
    if candidate.is_absolute():
        return candidate
    if candidate.parts and candidate.parts[0] == "assets":
        candidate = Path(*candidate.parts[1:])
    return RESOURCE_ROOT / candidate


def infer_asset_group(path: str | Path) -> str | None:
    return None


def build_download_command(*, group: str | None = None, source: str = "modelscope") -> str:
    return "vendored resources are already included"


def build_missing_asset_message(path: str | Path, *, label: str = "Asset") -> str:
    return f"{label} not found: {path}. Vendored Revo2 resources should be present under {RESOURCE_ROOT}."
