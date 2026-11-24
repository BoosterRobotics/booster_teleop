
from __future__ import annotations

from pathlib import Path

from booster_deploy.utils.project_path import resolve_project_path

PACKAGE_ROOT = Path(__file__).resolve().parent
RESOURCE_ROOT = PACKAGE_ROOT.parent / "resources"
PROJECT_ROOT = PACKAGE_ROOT.parents[2]
SOURCE_ROOT = PROJECT_ROOT
DATA_ROOT = RESOURCE_ROOT
CONFIG_ROOT = RESOURCE_ROOT
DEFAULT_CONFIG_PATH = CONFIG_ROOT / "revo2_right.yaml"
DEFAULT_BIHAND_CONFIG_PATH = CONFIG_ROOT / "revo2_bihand.yaml"
DEFAULT_HC_MOCAP_REFERENCE_BVH = "__builtin_hc_mocap__"
DEFAULT_HAND_LANDMARKER_MODEL = RESOURCE_ROOT / "models" / "hand_landmarker.task"
DEFAULT_LINKERHAND_SDK_PATH = RESOURCE_ROOT / "third_party" / "linkerhand-python-sdk"


def resolve_asset_path(local_path: str | Path) -> Path:
    candidate = Path(local_path)
    if candidate.is_absolute():
        return candidate
    if candidate.parts and candidate.parts[0] == "assets":
        candidate = Path(*candidate.parts[1:])
    return RESOURCE_ROOT / candidate


def resolve_config_path(config_path: str | Path) -> Path:
    candidate = Path(config_path).expanduser()
    if candidate.is_absolute():
        return candidate
    project_candidate = resolve_project_path(candidate)
    if project_candidate.exists():
        return project_candidate
    if candidate.parts[:2] == ("configs", "retargeting"):
        candidate = Path(*candidate.parts[2:])
    if len(candidate.parts) >= 2 and candidate.parts[-2] in {"left", "right", "base", "bihand"}:
        candidate = Path(candidate.parts[-1])
    bundled_candidate = CONFIG_ROOT / candidate
    return bundled_candidate if bundled_candidate.exists() else project_candidate
