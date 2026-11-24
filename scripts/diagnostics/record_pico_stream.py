#!/usr/bin/env python3
"""Record the onboard pico_rt.raw stream into record_pico_raw-compatible NPZ clips."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from booster_deploy.inputs.pico.stream_recording import main


if __name__ == "__main__":
    main()
