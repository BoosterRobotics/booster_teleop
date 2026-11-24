"""A run shares the launcher's monotonic origin across processes."""
from pathlib import Path
import time

def domain():
    try: return Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    except OSError: return 'unknown'

def display(ns, origin):
    ms=max(0,(ns-origin)//1_000_000)
    return f'{ms//3600000:02d}:{ms//60000%60:02d}:{ms//1000%60:02d}:{ms%1000:03d}'
