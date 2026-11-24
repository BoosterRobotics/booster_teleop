"""Project-scoped, ordinary-safe shutdown before any new tmux/telemetry run.

Called once by teleop_tmux.sh, under its local launch flock. No name-prefix kills.
"""
from __future__ import annotations
import argparse,os,subprocess,time
from pathlib import Path


def tmux(args):
    return subprocess.run(['tmux',*args],capture_output=True,text=True,timeout=3)


def owned_sessions(root,run=tmux):
    root=Path(root).resolve()
    result=run(['list-sessions','-F','#{session_name}'])
    if result.returncode:
        if result.returncode==1 and any(s in result.stderr.lower() for s in ('no server running','no such file or directory')):return []
        raise RuntimeError('Cannot list tmux sessions: '+result.stderr.strip())
    owned=[]
    for name in result.stdout.splitlines():
        target='='+name
        explicit=run(['show-options','-v','-t',target+':','@booster_repo_root']).stdout.strip()
        if explicit:
            if Path(explicit).resolve()==root:owned.append(name)
            continue
        meta=run(['show-options','-v','-t',target+':','@booster_session_meta']).stdout.strip()
        panes=run(['list-panes','-s','-t',target+':','-F','#{pane_current_path}|#{pane_start_command}'])
        if panes.returncode:raise RuntimeError('Cannot inspect session '+name)
        for row in panes.stdout.splitlines():
            cwd,_,entry=row.partition('|')
            if not cwd or not Path(cwd).resolve().is_relative_to(root):continue
            # Legacy sessions: exact project cwd plus explicit teleop metadata or
            # an absolute project entry; unrelated shells in the cwd aren't ours.
            old_entry=any(str(root/p) in entry for p in ('scripts/deploy.py','scripts/teleop_tmux.sh','scripts/dashboard.py'))
            if (meta and len(meta.split('/'))==4) or old_entry:
                owned.append(name);break
    return sorted(owned)


def cleanup(names,run,shutdown,*,timeout_s=30,dry_run=False,current_session=None):
    for name in names:
        if dry_run:
            print('Would request ordinary safe shutdown and wait for exact session:',name,flush=True);continue
        if current_session==name:
            raise RuntimeError('Launcher is inside old project session '+name+'; run from an independent terminal')
        print('Requesting ordinary safe shutdown:',name,flush=True)
        if shutdown(name)!=0:raise RuntimeError('Old session shutdown refused/failed: '+name)
        deadline=time.monotonic()+timeout_s
        while True:
            result = run(['has-session','-t','='+name])
            if result.returncode:
                gone = result.returncode == 1 and any(s in result.stderr.lower() for s in
                    ("can't find session", 'no server running', 'no such file or directory'))
                if not gone:
                    raise RuntimeError('Cannot confirm old session disappearance: '+result.stderr.strip())
                break
            if time.monotonic()>=deadline:raise RuntimeError('Old session still exists after shutdown: '+name)
            time.sleep(.1)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root',type=Path,required=True)
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args();root=args.repo_root.resolve()
    current=None
    if os.getenv('TMUX_PANE'):
        result=tmux(['display-message','-p','-t',os.environ['TMUX_PANE'],'#{session_name}'])
        if result.returncode:raise RuntimeError('Cannot identify launcher tmux session')
        current=result.stdout.strip()
    cleanup(owned_sessions(root),tmux,lambda name:subprocess.run(['bash',str(root/'scripts/teleop/teleop_shutdown.sh'),'--session',name,'--reason','cli']).returncode,dry_run=args.dry_run,current_session=current)

if __name__=='__main__':
    try:main()
    except (RuntimeError,subprocess.TimeoutExpired) as exc:raise SystemExit(str(exc))
