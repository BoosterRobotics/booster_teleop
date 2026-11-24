#!/usr/bin/env python3
"""Record raw PN-Link local joints to BVH using terminal A/B controls."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from booster_deploy.inputs.mocap.bvh_recording import main


if __name__ == "__main__":
    main()
