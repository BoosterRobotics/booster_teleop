"""Script-owned atomic business shutdown status, served by the final HTTP stage.

This file records phases only. It is not a robot-mode gate or a drain heuristic.
"""
import json,os,tempfile,time,uuid
from pathlib import Path


def _atomic(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(dir=path.parent,prefix='.shutdown-')
    try:
        with os.fdopen(fd,'w') as f:json.dump(value,f,ensure_ascii=False);f.flush()
        os.replace(tmp,path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def origin(directory):
    try:return json.loads((Path(directory)/'manifest.json').read_text()).get('origin_mono_ns')
    except (OSError,ValueError):return None


def wait_writers(directory,timeout=20):
    from booster_deploy.telemetry.reader import iter_file
    root=Path(directory);deadline=time.monotonic()+timeout
    while True:
        registered={p.stem.split('writer.',1)[1] for p in root.glob('writer.*.json')}
        terminals={r.get('writer_token'):r.get('data',{}) for p in root.glob('terminal.*.msgpack') for r in iter_file(p)}
        bad={k:v for k,v in terminals.items() if v.get('state')!='closed' or v.get('incomplete',True) or v.get('error') or v.get('rejected',0)}
        if bad:raise RuntimeError('Writer terminal incomplete/failed: '+repr(bad))
        if registered<=terminals.keys():return
        if time.monotonic()>=deadline:raise RuntimeError('Writer drain unknown: '+repr(sorted(registered-terminals.keys())))
        time.sleep(.1)


def write(directory,session,phase,detail=''):
    value=dict(session=session,origin_mono_ns=origin(directory),phase=phase,detail=detail,id=uuid.uuid4().hex,t_mono_ns=time.monotonic_ns())
    _atomic(Path(directory)/'shutdown-state.json',value)
    return value


def read(directory,session):
    try:
        value=json.loads((Path(directory)/'shutdown-state.json').read_text())
        return value if value.get('session')==session and value.get('origin_mono_ns')==origin(directory) else {}
    except (OSError,ValueError):return {}


def acknowledge(directory,value):
    _atomic(Path(directory)/'shutdown-delivered.json',dict(id=value['id']))


def delivered(directory,value):
    try:return json.loads((Path(directory)/'shutdown-delivered.json').read_text()).get('id')==value.get('id')
    except (OSError,ValueError):return False


def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('directory');parser.add_argument('session');parser.add_argument('phase');parser.add_argument('detail',nargs='?',default='')
    args=parser.parse_args()
    if args.phase=='start-run':
        # Invoked once by the locked launcher, only after old session vanished.
        # Keep all telemetry data/manifest/writer markers; clear business UI state.
        for name in ('shutdown-state.json','shutdown-delivered.json'):
            (Path(args.directory)/name).unlink(missing_ok=True)
    elif args.phase=='wait-delivery':
        value=read(args.directory,args.session);deadline=time.monotonic()+3.
        while time.monotonic()<deadline and not delivered(args.directory,value):time.sleep(.1)
    else:write(args.directory,args.session,args.phase,detail=args.detail)

if __name__=='__main__':main()
