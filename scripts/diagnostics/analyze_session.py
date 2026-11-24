"""Analyze a current MessagePack deployment run."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from booster_deploy.telemetry.cli import main

if __name__=='__main__': main()
