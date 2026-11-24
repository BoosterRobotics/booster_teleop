"""One bounded asynchronous streaming writer per process."""
import json
import os
import queue
import sys
import threading
import time
import uuid
from pathlib import Path
import msgpack
from .config import Config,STAGES
from .codec import snapshot, estimate, encode
from .clock import domain, display
from .recording import RecordingGate

class Writer:
    def __init__(self, directory, *, config=None, origin_ns=None, run_id=None, sink=None):
        self.directory=Path(directory)
        self.directory.mkdir(parents=True,exist_ok=True)
        self.config=config or Config.load()
        manifest_path=self.directory/'manifest.json'
        manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        self.gate=RecordingGate(self.directory,self.config.recording_mode,manifest.get('metadata',{}).get('execution'))
        settings=self.config.values.get('writer',{})
        self.capacity=int(settings.get('capacity_bytes',16777216))
        self.max_record=int(settings.get('max_record_bytes',2097152))
        self.flush_s=float(settings.get('flush_s',.25))
        self.origin_ns=origin_ns if origin_ns is not None else time.monotonic_ns()
        self.pid=os.getpid()
        self.token=f'{self.pid}-{uuid.uuid4().hex[:12]}'
        self.run_id=run_id or self.directory.name
        self.clock_domain=domain()
        self._queue=queue.Queue()
        self._lock=threading.Lock()
        self._closing=threading.Event()
        self._done=threading.Event()
        self._bytes=0
        self._accepted=self._written=self._rejected=0
        self._stage_counts={}
        self._discarded_prepare=0
        self._error=None
        self._state='open'
        self._sink=sink
        self._handles={}
        marker=self.directory/f'writer.{self.token}.json'
        marker.write_text(json.dumps(dict(writer_token=self.token,pid=self.pid,started_mono_ns=time.monotonic_ns(),origin_mono_ns=self.origin_ns)))
        self._thread=threading.Thread(target=self._run,daemon=True,name='telemetry-writer')
        self._thread.start()
        self.record('events',{'event':'process_start','writer_token':self.token})

    def record(self,stage,payload,*,t_mono_ns=None):
        if stage not in STAGES:
            if os.getpid()==self.pid:
                with self._lock: self._rejected+=1
            return False
        if os.getpid()!=self.pid:
            return False
        now=time.monotonic_ns() if t_mono_ns is None else int(t_mono_ns)
        self.gate.observe(stage,payload)
        if self.gate.error:
            self.mark_incomplete(self.gate.error)
        prepared_cycle = self.gate.preparing_cycle if stage in ("publish","retarget","postprocess") else 0
        if not self.gate.accepts(stage): return False
        selected=self.config.select(stage,payload)
        if selected is None: return False
        size=estimate(selected)+2048
        reserve = min(self.capacity//4,262144) if stage not in ('events','keys','health','dashboard') else 0
        with self._lock:
            if self._state!='open' or size>self.max_record or self._bytes+size>self.capacity-reserve:
                self._rejected+=1
                return False
            self._bytes+=size
        try:
            owned=snapshot(selected)
        except Exception as exc:
            with self._lock:
                self._bytes-=size
                self._rejected+=1
                self._error='snapshot: '+repr(exc)
            return False
        record=dict(schema_version=4,stage=stage,run_id=self.run_id,pid=self.pid,writer_token=self.token,
                    clock_domain=self.clock_domain,origin_mono_ns=self.origin_ns,t_mono_ns=now,
                    t=display(now,self.origin_ns),data=owned)
        with self._lock:
            if self._state!='open':
                self._bytes-=size
                self._rejected+=1
                return False
            self._accepted+=1
            self._queue.put_nowait((stage,record,size,prepared_cycle))
        return True

    # Cheap pre-check so callers can skip building a payload at all.
    def accepts(self,stage):
        if stage not in STAGES or os.getpid()!=self.pid: return False
        with self._lock:
            if self._state!='open': return False
        return self.gate.accepts(stage) and self.config.enabled(stage)

    # Preserve lifecycle failures even when the byte queue drains successfully.
    def mark_incomplete(self,reason):
        with self._lock:
            self._error=self._error or str(reason)

    def status(self):
        if os.getpid()!=self.pid: return {'state':'inherited','incomplete':True}
        with self._lock:
            return dict(state=self._state,accepted=self._accepted,written=self._written,
                        rejected=self._rejected,discarded_prepare=self._discarded_prepare,queued_bytes=self._bytes,error=self._error,
                        incomplete=bool(self._error or self._rejected or self._state not in ('closed','open')))

    def _write(self,stage,record):
        blob=msgpack.packb(record,use_bin_type=True,default=encode)
        if self._sink is not None:
            self._sink(stage,blob)
        else:
            handle=self._handles.get(stage)
            if handle is None:
                handle=open(self.directory/f'{stage}.{self.token}.msgpack','ab',buffering=262144)
                self._handles[stage]=handle
            handle.write(blob)

    def _run(self):
        last=time.monotonic()
        pending={}
        def finish(stage, record, size, write=True):
            if write:
                self._write(stage,record)
            with self._lock:
                if write:
                    self._written+=1
                    self._stage_counts[stage]=self._stage_counts.get(stage,0)+1
                else:
                    self._discarded_prepare+=1
                self._bytes-=size
        try:
            while not self._closing.is_set() or not self._queue.empty() or pending:
                try: stage,record,size,cycle=self._queue.get(timeout=min(.05,self.flush_s))
                except queue.Empty:
                    stage=None
                if stage is not None:
                    if cycle:
                        pending.setdefault(cycle,[]).append((stage,record,size))
                    else:
                        finish(stage,record,size)
                for cycle in tuple(pending):
                    committed=self.gate.committed(cycle)
                    if committed or cycle != self.gate.preparing_cycle or self._closing.is_set():
                        if not committed:
                            committed=self.gate.committed(cycle)
                        for stage,record,size in pending.pop(cycle):
                            finish(stage,record,size,committed)
                if time.monotonic()-last>=self.flush_s:
                    for h in self._handles.values(): h.flush()
                    last=time.monotonic()
            for h in self._handles.values(): h.flush(); h.close()
            self._handles={}
            status=self.status()
            status.update(state='closed',incomplete=bool(self._error or self._rejected),stage_counts=self._stage_counts)
            terminal=dict(schema_version=4,stage='terminal',run_id=self.run_id,pid=self.pid,
                          writer_token=self.token,origin_mono_ns=self.origin_ns,clock_domain=self.clock_domain,
                          t_mono_ns=time.monotonic_ns(),data=status)
            self._write('terminal',terminal)
            for h in self._handles.values(): h.flush(); h.close()
            self._handles={}
            with self._lock: self._state='closed' 
        except Exception as exc:
            with self._lock:
                self._error=repr(exc)
                self._state='failed'
            print('[telemetry] writer failed: '+repr(exc),file=sys.stderr,flush=True)
            try:
                self._sink=None
                self._write('terminal',dict(stage='terminal',writer_token=self.token,pid=self.pid,
                    run_id=self.run_id,t_mono_ns=time.monotonic_ns(),data=self.status()))
            except Exception: pass
        finally:
            for h in self._handles.values():
                try: h.flush(); h.close()
                except OSError as exc:
                    self._error=repr(exc); self._state='failed'
            self._done.set()

    def close(self,timeout=5.):
        if os.getpid()!=self.pid: return {'state':'inherited','incomplete':True}
        with self._lock:
            if self._state=='open': self._state='draining'
            self._closing.set()
        self._done.wait(timeout)
        if not self._done.is_set():
            with self._lock: self._state='close_timeout'
        self.gate.close()
        return self.status()
