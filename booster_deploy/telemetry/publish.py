"""Publication IDs advance independently of logging and native device counters."""
from collections import OrderedDict
import os
import threading
import uuid
from .runtime import get_writer

class Source:
    def __init__(self,name,*,writer=None):
        self.name=name
        self.stream=f'{name}:{uuid.uuid4().hex}'
        self.index=0
        self.pid=os.getpid()
        self.writer=writer
        if writer is not None:
            self.directory=writer.directory; self.namespace='source:'+writer.run_id
        elif os.getenv('BOOSTER_TELEMETRY_DIR'):
            from .runtime import initialize
            from pathlib import Path
            manifest=initialize()
            self.directory=Path(os.environ['BOOSTER_TELEMETRY_DIR']); self.namespace='source:'+manifest['run_id']
        else:
            self.directory=None
            self.namespace='source:'+self.stream
        self._lock=threading.Lock()
        self._samples=OrderedDict()
        self._native=OrderedDict()

    def next(self,*,native_seq=None,loop=None,native_index=None):
        if os.getpid()!=self.pid:
            self.stream=f'{self.name}:{uuid.uuid4().hex}'
            self.index=0
            self.pid=os.getpid()
            if self.directory is None:
                self.namespace='source:'+self.stream
            self._lock=threading.Lock()
            self._samples=OrderedDict()
            self._native=OrderedDict()
        if self.directory is None:
            self.index+=1
        else:
            from .source_counter import next_id
            self.index=next_id(self.directory)
        identity=(self.namespace,self.index)
        self._native[identity]=dict(native_seq=native_seq,loop=loop,native_index=native_index,source_stream=self.stream)
        if len(self._native)>4096: self._native.popitem(last=False)
        return identity

    def for_sample(self,native_seq,recv_ns):
        with self._lock:
            key=(int(native_seq),int(recv_ns))
            if key not in self._samples:
                self._samples[key]=self.next(native_seq=int(native_seq))
                if len(self._samples)>4096: self._samples.popitem(last=False)
            return self._samples[key]

    def published(self,identity,*,channel,**fields):
        facts=dict(self._native.get(identity,{}),**fields)
        facts.update(source_id=identity,source_ids=[identity],kind='source',channel=channel)
        return (self.writer or get_writer()).record('publish',facts)
