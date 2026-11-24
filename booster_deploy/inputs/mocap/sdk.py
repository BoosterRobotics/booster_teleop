
from __future__ import annotations

import sys
from pathlib import Path


_SDK_PYTHON_DIR = Path(__file__).resolve().parents[3] / "third_party" / "pnlink_mocap" / "python"


def load_mocap_api():

    sdk_path = str(_SDK_PYTHON_DIR)
    if sdk_path not in sys.path:
        sys.path.insert(0, sdk_path)
    import mocap_api

    return mocap_api


__all__ = ["load_mocap_api"]
