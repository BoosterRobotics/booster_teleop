#!/usr/bin/env python3
"""Visualize a raw Pico recording and its GMR-retargeted T2 motion."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from booster_deploy.visualization.offline_teleop import main_pico


if __name__ == "__main__":
    main_pico()
