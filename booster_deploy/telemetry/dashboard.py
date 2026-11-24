"""Incremental fact projection for the UI, without scanning tensor streams."""
from dataclasses import dataclass,field
from pathlib import Path
import time
import msgpack


@dataclass
class FactTail:
    diag_dir: Path | None
    offsets: dict = field(default_factory=dict)
    latest: dict = field(default_factory=dict)
    recent_alerts: list = field(default_factory=list)
    max_alerts: int = 200
    read_errors: int = 0
    events: list = field(default_factory=list)
    max_events: int = 40
    _unpackers: dict = field(default_factory=dict)
    _counts: dict = field(default_factory=dict)
    _directory: object = None

    def files(self):
        if self.diag_dir is None: return {}
        return {str(p):p for prefix in ('health','events','keys','dashboard') for p in sorted(Path(self.diag_dir).glob(prefix+'.*.msgpack'))}

    def poll(self,now_ns=None,max_bytes=1<<20):
        if self._directory!=self.diag_dir:
            self.offsets.clear(); self._unpackers.clear(); self.latest.clear(); self._counts.clear()
            self._directory=self.diag_dir
        for key,path in self.files().items():
            try:
                size=path.stat().st_size; offset=self.offsets.get(key,0)
                if size<offset: offset=0; self._unpackers.pop(key,None)
                with path.open('rb') as handle:
                    handle.seek(offset); chunk=handle.read(min(max_bytes,size-offset))
                self.offsets[key]=offset+len(chunk)
                unpacker=self._unpackers.setdefault(key,msgpack.Unpacker(raw=False,max_buffer_size=2<<20))
                unpacker.feed(chunk)
                for row in unpacker: self._apply(row)
            except (OSError,ValueError,msgpack.UnpackException): self.read_errors+=1
        if self.diag_dir:
            from .control_ipc import read
            ipc=read(self.diag_dir)
            if ipc: self.latest.setdefault('ctrl',dict(pane='ctrl',state={},hz={},total={},alerts=[]))['state'].update(ipc)
        return self.latest

    def _apply(self,row):
        data=row.get('data',{}); stage=row.get('stage'); component=data.get('component','keys' if stage=='keys' else 'dashboard')
        stamp=row.get('t_mono_ns',0)
        view=self.latest.setdefault(component,dict(pane=component,state={},hz={},total={},alerts=[]))
        view['t_mono_ns']=max(stamp,view.get('t_mono_ns',0))
        view['t_session']=row.get('t_session')
        if stage=='events':
            if data.get('event')=='state_transition': view['state'].update(data.get('changes',{}))
            if data.get('event')=='note': view['state'].update({k:v for k,v in data.items() if k not in ('event','component')})
            else: view['msg']=data.get('event','')
        if stage=='health':
            view['health_mono_ns']=data.get('observed_ns',stamp)
            view['state'].update(data.get('facts',{}))
            view['state'].update(data.get('progress',{}))
            # Replace, including removal by a field switch; never scan encoder tensors.
            view['state'].pop('input_progress',None)
            if isinstance(data.get('input_progress'),dict):
                view['state']['input_progress']=data['input_progress']
            for metric,value in data.get('counts',{}).items():
                previous=self._counts.get((component,metric))
                if previous and stamp>previous[0]: view['hz'][metric]=(value-previous[1])*1e9/(stamp-previous[0])
                self._counts[(component,metric)]=(stamp,value)
                view['total'][metric]=value
        if stage=='keys':
            self.events.append(dict(at=stamp/1e9,text='t:key '+str(data.get('key')),button=data.get('key'),label=data.get('label',data.get('phase')),source=data.get('source')))
            del self.events[:-self.max_events]


def control_state(directory):
    from .control_ipc import read
    ipc=read(directory)
    if ipc: return ipc
    tail=FactTail(Path(directory))
    while True:
        before=dict(tail.offsets); tail.poll()
        if tail.offsets==before: break
    return (tail.latest.get('ctrl') or {}).get('state',{})
