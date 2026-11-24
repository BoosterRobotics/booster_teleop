"""Run context and the fork-safe process singleton writer."""
import atexit
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import uuid
from .clock import domain
from .config import Config
from .writer import Writer

_writer=None
_pid=None
_sealed_pid=None
_guard=threading.RLock()
_finalized=set()
_switch_guard=threading.Lock()
_active_directory=None
_switching=False
_routed={}
_route_guard=threading.Lock()
_routed_dirs=set()

# Keep finalized runs sealed while allowing an explicit new process run.
class _ClosedWriter:
    def record(self,*args,**kwargs): return False
    def status(self): return {'state':'not_started','incomplete':False}
    def close(self,timeout=5.): return self.status()

_closed_writer=_ClosedWriter()


def initialize(name=None,*,directory=None,config=None,topology=None,metadata=None):
    name=name or os.getenv('BOOSTER_RUN_ID') or ('standalone_'+time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6])
    directory=Path(directory or os.getenv('BOOSTER_TELEMETRY_DIR') or Path(__file__).resolve().parents[2]/'logs'/name).resolve()
    global _writer,_active_directory,_switching
    with _switch_guard:
        with _guard:
            old_directory=_active_directory or os.getenv('BOOSTER_TELEMETRY_DIR')
            changed=old_directory is not None and str(directory)!=str(old_directory)
            previous=_writer if _pid==os.getpid() else None
            if changed: _finalized.add((os.getpid(),str(old_directory)))
            _switching=True
        try:
            if changed and previous is not None:
                result=previous.close()
                if result['state'] not in ('closed','failed'):
                    raise RuntimeError('previous telemetry run did not close: '+repr(result))
            with _guard:
                manifest=_initialize_manifest(
                    name,directory=directory,config=config,topology=topology,metadata=metadata)
                if changed: _writer=None
                _active_directory=str(directory)
                return manifest
        finally:
            with _guard: _switching=False


def _initialize_manifest(name,*,directory,config,topology,metadata=None,activate=True):
    directory.mkdir(parents=True,exist_ok=True)
    manifest_path=directory/'manifest.json'
    execution=os.getenv('BOOSTER_TELEMETRY_EXECUTION')
    replay=os.getenv('BOOSTER_TELEMETRY_REPLAY')
    existing_manifest=manifest_path.exists()
    if existing_manifest:
        cfg=Config(json.loads(manifest_path.read_text())['config'])
    else:
        cfg=Config.load(config or os.getenv('BOOSTER_TELEMETRY_CONFIG'))
    trajectory_switch=cfg.trajectory_switch if (execution,replay)==('real','offline') else 'once'
    if not existing_manifest and trajectory_switch == 'unlimited':
        cfg=Config({**cfg.values, 'enabled': True})
    resolved_topology=topology or os.getenv('BOOSTER_TELEMETRY_TOPOLOGY','unknown')
    from .run_config import initialize_execution_config
    execution_config_path=directory/'execution_config.json'
    if not execution_config_path.exists():
        initialize_execution_config(directory, {
            'telemetry_enabled': cfg.recording_enabled,
            'recording_mode': cfg.recording_mode,
            'sn': os.getenv('BOOSTER_TELEMETRY_SN'),
            'started_at': os.getenv('BOOSTER_TELEMETRY_STARTED_AT'),
            'task': os.getenv('BOOSTER_TELEMETRY_TASK'),
            'execution': execution,
            'input_source': os.getenv('BOOSTER_TELEMETRY_INPUT_SOURCE'),
            'input_path': os.getenv('BOOSTER_TELEMETRY_INPUT_PATH'),
            'replay': replay,
            'topology': resolved_topology,
            'mjcf': os.getenv('BOOSTER_TELEMETRY_MJCF'),
            'trajectory': os.getenv('BOOSTER_TELEMETRY_TRAJECTORY_PATH'),
            'trajectory_switch': trajectory_switch,
        })
    run_metadata={
        key:value for key,value in {
            'execution':os.getenv('BOOSTER_TELEMETRY_EXECUTION'),
            'task':os.getenv('BOOSTER_TELEMETRY_TASK'),
            'input_source':os.getenv('BOOSTER_TELEMETRY_INPUT_SOURCE'),
            'mjcf':os.getenv('BOOSTER_TELEMETRY_MJCF'),
            'sn': os.getenv('BOOSTER_TELEMETRY_SN'),
            'trajectory_path': os.getenv('BOOSTER_TELEMETRY_TRAJECTORY_PATH'),
            'trajectory_stem': os.getenv('BOOSTER_TELEMETRY_TRAJECTORY_STEM'),
        }.items() if value
    }
    if metadata:
        run_metadata.update({str(key):value for key,value in metadata.items() if value is not None})
    manifest=dict(schema_version=4,run_id=name,origin_mono_ns=int(os.getenv('BOOSTER_RUN_ORIGIN_NS') or time.monotonic_ns()),
                  clock_domain=domain(),topology=resolved_topology,metadata=run_metadata,config=cfg.values)
    fd,temp=tempfile.mkstemp(dir=directory,prefix='.manifest-')
    try:
        with os.fdopen(fd,'w') as handle:
            json.dump(manifest,handle,ensure_ascii=False,indent=2)
            handle.flush()
        try: os.link(temp,manifest_path)
        except FileExistsError: pass
    finally:
        Path(temp).unlink()
    manifest=json.loads(manifest_path.read_text())
    if manifest['clock_domain']!=domain(): raise RuntimeError('run directory belongs to a different boot/host')
    if activate:
        os.environ.update(BOOSTER_RUN_ID=manifest['run_id'],BOOSTER_TELEMETRY_DIR=str(directory),
                          BOOSTER_RUN_ORIGIN_NS=str(manifest['origin_mono_ns']))
        if config: os.environ['BOOSTER_TELEMETRY_CONFIG']=str(Path(config).resolve())
    return manifest


def get_writer():
    global _writer,_pid
    directory=os.getenv('BOOSTER_TELEMETRY_DIR')
    # Only explicit runs may create persistent telemetry.
    if not directory:
        return _closed_writer
    with _guard:
        key=(os.getpid(),str(Path(directory).resolve()))
        if _sealed_pid==os.getpid() or key in _finalized or _switching: return _closed_writer
        needs_init=_active_directory!=str(Path(directory).resolve()) or not (Path(directory)/'manifest.json').exists()
    if needs_init: initialize(directory=directory)
    with _guard:
        pid=os.getpid()
        directory=os.getenv('BOOSTER_TELEMETRY_DIR')
        if not directory: return _closed_writer
        key=(pid,str(Path(directory).resolve()))
        if _switching: return _closed_writer
        if _sealed_pid==pid or key in _finalized:
            return _closed_writer
        if _writer is None or _pid!=pid:
            manifest=json.loads((Path(directory)/'manifest.json').read_text())
            _pid=pid
            _writer=Writer(directory,config=Config(manifest['config']),
                           origin_ns=manifest['origin_mono_ns'],run_id=manifest['run_id'])
        return _writer


def record_for(directory,stage,*,run_id=None,**fields):
    directory=Path(directory).resolve()
    stamp=fields.pop('_event_mono_ns',None)
    with _guard:
        if _sealed_pid==os.getpid() or (os.getpid(),str(directory)) in _finalized: return False
        if _writer is not None and _pid==os.getpid() and _writer.directory==directory:
            return _writer.record(stage,fields,t_mono_ns=stamp)
    with _route_guard:
        with _guard:
            if _sealed_pid==os.getpid() or (os.getpid(),str(directory)) in _finalized: return False
            writer=_routed.get(str(directory))
            previous=next(iter(_routed.values()),None)
        if writer is None:
            if previous is not None:
                result=previous.close()
                if result['state'] not in ('closed','failed'): return False
            with _guard:
                if _sealed_pid==os.getpid() or (os.getpid(),str(directory)) in _finalized: return False
                manifest=_initialize_manifest(run_id or directory.name,directory=directory,config=None,topology=None,activate=False)
                writer=Writer(directory,config=Config(manifest['config']),origin_ns=manifest['origin_mono_ns'],run_id=manifest['run_id'])
                _routed.clear()
                _routed[str(directory)]=writer
                _routed_dirs.add(str(directory))
        return writer.record(stage,fields,t_mono_ns=stamp)


def record(stage,**fields):
    stamp=fields.pop('_event_mono_ns',None)
    return get_writer().record(stage,fields,t_mono_ns=stamp)


def close_process(timeout=5.,*,permanent=False):
    global _sealed_pid
    with _guard:
        pid=os.getpid()
        if permanent: _sealed_pid=pid
        directory=os.getenv('BOOSTER_TELEMETRY_DIR')
        _finalized.add((pid,str(Path(directory).resolve()) if directory else None))
        for routed_directory in _routed_dirs: _finalized.add((pid,routed_directory))
        writers=list(_routed.values())
        if _writer is not None and _pid==pid: writers.append(_writer)
        for writer in writers: _finalized.add((pid,str(writer.directory)))
    results=[writer.close(timeout) for writer in writers]
    return next((result for result in results if result['state']!='closed' or result.get('incomplete')),results[-1] if results else _closed_writer.status())


def _fork_child():
    global _writer,_pid,_guard,_sealed_pid,_finalized,_switch_guard,_switching,_active_directory,_routed,_route_guard,_routed_dirs
    _route_guard=threading.Lock()
    _routed_dirs=set()
    _routed={}
    _switch_guard=threading.Lock()
    _switching=False
    _active_directory=None
    _sealed_pid=None
    _finalized=set()
    _writer=None
    _pid=None
    _guard=threading.RLock()

os.register_at_fork(after_in_child=_fork_child)
atexit.register(close_process)
