#!/usr/bin/env python3
"""Publish a recorded PN-Link BVH on the live mocap RT topic."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from booster_deploy.inputs.mocap.offline_publisher import main


if __name__ == "__main__":
    main()
