"""UI-only rolling diagnostics; persistent facts use booster_deploy.telemetry."""
from __future__ import annotations
import json
import os
import queue
import threading
import sys
import time
import uuid
from pathlib import Path
from .diag_windows import GapStats, IntervalWindow, LatencyStats, RateWindow, SampleWindow, SeqTracker
from booster_deploy.telemetry.clock import domain, display
from booster_deploy.telemetry.events import Transitions, event

DIAG_ENV='BOOSTER_DIAG'
DIAG_DIR_ENV='BOOSTER_DIAG_DIR'
DIAG_HZ_ENV='BOOSTER_DIAG_HZ'
DEFAULT_INTERVAL_S=1.0
RATE_FLOOR_HZ=49.5
WARMUP_S=2.0
_STATE_NAMES=frozenset(('phase','mode','state','control_state','robot_mode','robot_mode_mono_ns','shutdown_confirmed',
    'actual_mode','actual_mode_id','actual_mode_name','actual_mode_recv_ns','actual_mode_fresh','sdk_confirmed',
    'output_mode','paused','stopped','held','exit','waiting_for_start','task','recording','recording_path'))
_PROGRESS_NAMES=frozenset(('cycle','frame_index','frames_left','publish_total','clip_frames','saved','rejected'))


def below_floor(hz):
    return bool(hz) and float(hz)<RATE_FLOOR_HZ


def format_gap(stats):
    return 'n/a' if stats.count==0 else f'{stats.p95:.1f}/{stats.maximum:.1f}'


def format_lat(stats):
    return 'n/a' if stats.count==0 else f'{stats.p50:.1f}/{stats.p95:.1f}/{stats.maximum:.1f}'


class DiagReporter:
    def __init__(self,pane,*,interval_s=None,jsonl_path=None,enabled=None,clock_ns=time.monotonic_ns,
                 wall_clock_ns=time.time_ns,stream=None):
        self.pane=pane
        self.enabled=(os.getenv(DIAG_ENV,'1').lower() not in ('0','off','false','no')) if enabled is None else enabled
        self.interval_ns=int((interval_s if interval_s is not None else 1/max(float(os.getenv(DIAG_HZ_ENV,'1')),0.01))*1e9)
        self._clock_ns=clock_ns
        self._stream=stream or sys.stdout
        self._intervals={}; self._rates={}; self._samples={}; self._seqs={}; self._state={}
        self._last_emit_ns=None
        self._transitions=Transitions(pane)
        self.policy_session=None
        self.clock_domain=domain()
        self.run_id=os.getenv('BOOSTER_RUN_ID')
        self._lines_written=0
        self._console=queue.Queue(maxsize=8)
        self._console_thread=None
        self._fact_state={}

    _clock_domain=staticmethod(domain)

    def interval(self,name,**kwargs):
        if name not in self._intervals: self._intervals[name]=IntervalWindow()
        return self._intervals[name]

    def rate(self,name):
        if name not in self._rates: self._rates[name]=RateWindow()
        return self._rates[name]

    def seq(self,name):
        if name not in self._seqs: self._seqs[name]=SeqTracker()
        return self._seqs[name]

    def mark(self,name,now_ns=None):
        self.interval(name).mark(self._clock_ns() if now_ns is None else now_ns)

    def count(self,name,now_ns=None):
        self.rate(name).mark(self._clock_ns() if now_ns is None else now_ns)

    def sample(self,name,value,now_ns=None,**kwargs):
        if name not in self._samples: self._samples[name]=SampleWindow()
        self._samples[name].add(value,self._clock_ns() if now_ns is None else now_ns)

    def latency(self,name,now_ns=None):
        return self._samples[name].stats(self._clock_ns() if now_ns is None else now_ns) if name in self._samples else LatencyStats(0,0.,0.,0.)

    def gap(self,name,now_ns=None):
        return self._intervals[name].gap_stats(self._clock_ns() if now_ns is None else now_ns) if name in self._intervals else GapStats(0,0.,0.,0)

    def stall_ms(self,name,now_ns=None):
        return self._intervals[name].stall_ms(self._clock_ns() if now_ns is None else now_ns) if name in self._intervals else None

    def note(self,**state):
        self._state.update(state)
        facts={k:v for k,v in state.items() if k in _STATE_NAMES and isinstance(v,(str,bool,int,float,type(None)))}
        changed={k:v for k,v in facts.items() if k not in self._fact_state or self._fact_state[k]!=v}
        self._fact_state.update(facts)
        if changed:
            event('note',component=self.pane,**changed)
            if self.pane=='ctrl':
                from booster_deploy.telemetry.control_ipc import submit
                submit(self._fact_state)

    def due(self,now_ns=None):
        now=self._clock_ns() if now_ns is None else now_ns
        return self._last_emit_ns is None or now-self._last_emit_ns>=self.interval_ns

    def emit(self,screen,*,alerts=(),extra=None,now_ns=None,force=False):
        now=self._clock_ns() if now_ns is None else now_ns
        state=dict(self._state,**((extra or {}).get('state',{})))
        self.note(**state)
        if not force and not self.due(now): return None
        self._last_emit_ns=now
        from booster_deploy.telemetry.runtime import record
        counts={k:v.total for k,v in self._intervals.items()}
        counts.update({k:v.total for k,v in self._rates.items()})
        input_progress=state.get('input_progress')
        raw_progress={k:input_progress[k] for k in ('events','last_advance_ns','known','state','clock') if k in input_progress} if isinstance(input_progress,dict) else None
        record('health',component=self.pane,observed_ns=now,counts=counts,
            **({'input_progress':raw_progress} if raw_progress is not None else {}),
            progress={k:v for k,v in state.items() if k in _PROGRESS_NAMES and isinstance(v,(int,float,bool,str,type(None)))})
        line=f'[diag] {self.pane}  {screen}'
        self._print(line)
        self._lines_written+=1
        return line

    def event(self,name,payload=None):
        return event(name,component=self.pane,policy_session=self.policy_session,**(payload or {}))

    def start_policy_session(self):
        self.policy_session=uuid.uuid4().hex
        self.event('policy_session_start')
        return self.policy_session

    def _console_run(self):
        while True:
            line=self._console.get()
            if line is None: return
            try: print(line,file=self._stream,flush=True)
            except (OSError,ValueError): pass

    def _print(self,line):
        if not self.enabled: return
        if self._console_thread is None:
            self._console_thread=threading.Thread(target=self._console_run,daemon=True,name='ui-console')
            self._console_thread.start()
        try: self._console.put_nowait(line[:8192])
        except queue.Full: pass

    def console_payload(self,prefix,payload):
        self._print(prefix+json.dumps(payload,ensure_ascii=False))

    def close(self):
        self.event('component_close')
        if self.pane=='ctrl':
            from booster_deploy.telemetry.control_ipc import close
            close()
        if self._console_thread is not None:
            try: self._console.put_nowait(None)
            except queue.Full: pass

    @property
    def lines_written(self):
        return self._lines_written


def reporter(pane,**kwargs):
    return DiagReporter(pane,**kwargs)
