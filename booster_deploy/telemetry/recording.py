"""Shared mmap gate changes recording only, never business IPC or source identity."""
import fcntl
import mmap
import os
from pathlib import Path

ALWAYS=frozenset(('events','keys','health','dashboard'))
ACTIVE=frozenset(('arming_tracking','tracking','tracking_paused','exiting_to_walk'))
ARMED=(ord('1'),ord('2'))
GATE_NAME='.recording_gate'
OWNER_NAME='.recording_gate.owner'
_DEAD_STATES=('Z','X','x')


def _proc_stat_fields(pid):
    """Fields of /proc/<pid>/stat after the command name, or None when unavailable."""
    try: raw=Path(f'/proc/{int(pid)}/stat').read_text()
    except (OSError,TypeError,ValueError): return None
    try: return raw[raw.rindex(')')+2:].split()
    except ValueError: return None


def _proc_start_time(pid):
    """Boot-relative start ticks used to tell a real owner from a recycled pid."""
    fields=_proc_stat_fields(pid)
    try: return int(fields[19])
    except (TypeError,IndexError,ValueError): return None


def _owner_alive(pid,start):
    """os.kill(pid,0) plus identity checks: zombies and recycled pids are dead owners."""
    try: pid=int(pid)
    except (TypeError,ValueError): return False
    if pid<=0: return False
    try: os.kill(pid,0)
    except ProcessLookupError: return False
    except PermissionError: pass
    except OSError: return False
    fields=_proc_stat_fields(pid)
    if fields and fields[0] in _DEAD_STATES: return False
    if start is not None:
        current=_proc_start_time(pid)
        if current is not None and current!=start: return False
    return True


def _read_owner(path):
    try: parts=path.read_text().split()
    except OSError: return None
    if not parts: return None
    try: pid=int(parts[0])
    except ValueError: return None
    start=None
    if len(parts)>1 and parts[1]!='-':
        try: start=int(parts[1])
        except ValueError: start=None
    return pid,start


def _write_owner(path):
    start=_proc_start_time(os.getpid())
    body=f'{os.getpid()} {"-" if start is None else start}\n'
    temporary=path.with_name(f'{OWNER_NAME}.{os.getpid()}.tmp')
    try:
        temporary.write_text(body)
        os.replace(temporary,path)
    except OSError:
        try: temporary.unlink()
        except OSError: pass


class RecordingGate:
    def __init__(self,directory,mode,execution=None):
        self.mode=mode
        self.sim=execution=='sim'
        self.memory=None
        self.directory=Path(directory)
        self._committed=set()
        self.error=None
        self.cycle_memory=None
        self._owner=None
        self._claimed=False
        if mode=='tracking' and not self.sim:
            path=Path(directory)/GATE_NAME
            self._owner=path.with_name(OWNER_NAME)
            fd=os.open(path,os.O_CREAT|os.O_RDWR,0o600)
            try:
                fcntl.flock(fd,fcntl.LOCK_EX)
                created=os.fstat(fd).st_size==0
                if created:
                    os.write(fd,b'0')
                self.memory=mmap.mmap(fd,1)
                cycle_fd=os.open(path.with_name('.recording_cycle'),os.O_CREAT|os.O_RDWR,0o600)
                try:
                    if os.fstat(cycle_fd).st_size < 8:
                        os.ftruncate(cycle_fd,8)
                    self.cycle_memory=mmap.mmap(cycle_fd,8)
                finally:
                    os.close(cycle_fd)
                # A reused run directory must not inherit an armed flag whose owner is gone.
                # Joining a live owner is left untouched, so adds never disarm a running gate.
                live=self._owner_live()
                if not live:
                    if self.memory[0] in (*ARMED,ord('3')): self.memory[0]=ord('0')
                    self._claim()
            finally:
                fcntl.flock(fd,fcntl.LOCK_UN)
                os.close(fd)

    def _owner_live(self):
        owner=_read_owner(self._owner) if self._owner is not None else None
        if owner is None: return False
        return _owner_alive(*owner)

    # Ownership follows the process that last armed the gate, not the first one to open it.
    def _claim(self):
        if self._owner is None or self._claimed: return
        owner=_read_owner(self._owner)
        if owner is not None and owner[0]==os.getpid() and _owner_alive(*owner):
            self._claimed=True; return
        _write_owner(self._owner)
        self._claimed=True

    def observe(self,stage,payload):
        if self.memory is None or stage!='events': return
        if payload.get('event')=='execution_metadata' and payload.get('metadata',{}).get('execution')=='sim':
            self.memory[0]=ord('2')
            self._claim()
            return
        event = payload.get('event')
        if event == 'offline_prepare':
            cycle = int(payload.get('cycle') or 0)
            if cycle <= 0:
                return
            self.memory[0] = ord('3')
            self.cycle_memory[:] = cycle.to_bytes(8, 'little')
            self._claim()
            return
        if event == 'offline_prepare_cancelled':
            if int(payload.get('cycle') or 0) == self.cycle and self.memory[0] == ord('3'):
                self.memory[0] = ord('4')
            return
        if event == 'offline_playback_started':
            cycle = int(payload.get('cycle') or 0)
            if cycle > 0 and cycle == int.from_bytes(self.cycle_memory[:], 'little') and self.memory[0] == ord('3'):
                try:
                    (self.directory/f'.recording_committed.{cycle}').touch(exist_ok=True)
                except OSError as exc:
                    self.error=repr(exc)
                    return
                self._committed.add(cycle)
                self.memory[0] = ord('1')
                self._claim()
            return
        if event!='state_transition' or payload.get('component')!='ctrl': return
        state=payload.get('changes',{}).get('control_state')
        if state in ACTIVE and self.memory[0] not in (ord('3'),ord('4')):
            self.memory[0]=ord('1')
            self._claim()
        elif state=='walking': self.memory[0]=ord('0')

    @property
    def preparing_cycle(self):
        if self.memory is not None and self.memory[0] == ord('3'):
            return int.from_bytes(self.cycle_memory[:], 'little')
        return 0

    def committed(self,cycle):
        if cycle in self._committed:
            return True
        if (self.directory/f'.recording_committed.{int(cycle)}').is_file():
            self._committed.add(cycle)
            return True
        return False

    @property
    def cycle(self):
        return int.from_bytes(self.cycle_memory[:], 'little') if self.memory is not None else 0

    def accepts(self, stage):
        return self.allows(stage) or bool(self.preparing_cycle and stage in ('publish','retarget','postprocess'))

    def allows(self,stage):
        if self.mode=='full_run' or self.sim or stage in ALWAYS: return True
        memory=self.memory
        return memory is not None and memory[0] in (ord('1'),ord('2'))

    # Release the shared mapping; the flag file itself is a run artifact.
    def close(self):
        for name in ('memory','cycle_memory'):
            memory=getattr(self,name)
            setattr(self,name,None)
            if memory is not None:
                try: memory.close()
                except (BufferError,ValueError): pass
