#!/usr/bin/env python3
"""User entry point for booster_deploy.telemetry."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from booster_deploy.telemetry.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
