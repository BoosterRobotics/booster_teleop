#!/usr/bin/env python3
"""Record raw Pico/XRoboToolkit body poses to an NPZ file."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from booster_deploy.inputs.pico.raw_recording import main


if __name__ == "__main__":
    main()
