"""Bounded latest-value business IPC remains available with logging disabled."""
import atexit
import json
import os
from pathlib import Path
import threading
from .runtime import initialize

_mailbox=None
_guard=threading.Lock()

class ControlMailbox:
    def __init__(self,directory):
        self.path=Path(directory)/'.control_state.json'
        self.pending=None
        self.lock=threading.Lock()
        self.wakeup=threading.Event()
        self.stopping=False
        self.thread=threading.Thread(target=self._run,daemon=True,name='control-state-ipc')
        self.thread.start()

    def submit(self,fields):
        owned={k:v for k,v in fields.items() if k in ('state','mode','robot_mode','robot_mode_mono_ns','shutdown_confirmed','output_mode') and isinstance(v,(str,bool,int,float,type(None)))}
        with self.lock: self.pending=owned
        self.wakeup.set()

    def _run(self):
        while True:
            self.wakeup.wait(.1); self.wakeup.clear()
            with self.lock: value=self.pending; self.pending=None
            if value is not None:
                try:
                    temp=self.path.with_suffix('.tmp')
                    temp.write_text(json.dumps(value)); os.replace(temp,self.path)
                except OSError: pass
            if self.stopping and self.pending is None: return

    def close(self):
        self.stopping=True; self.wakeup.set(); self.thread.join(2)


def submit(fields):
    global _mailbox
    if not os.getenv('BOOSTER_TELEMETRY_DIR'):
        return
    with _guard:
        if _mailbox is None:
            initialize()
            _mailbox=ControlMailbox(os.environ['BOOSTER_TELEMETRY_DIR'])
        _mailbox.submit(fields)


def read(directory):
    try: return json.loads((Path(directory)/'.control_state.json').read_text())
    except (OSError,ValueError): return {}


def close():
    global _mailbox
    if _mailbox is not None: _mailbox.close(); _mailbox=None


def _fork_child():
    global _mailbox,_guard
    _mailbox=None; _guard=threading.Lock()

os.register_at_fork(after_in_child=_fork_child)
atexit.register(close)
