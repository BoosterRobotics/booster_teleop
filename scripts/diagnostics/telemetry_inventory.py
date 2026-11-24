"""Inventory onboard telemetry runs: windows, sample counts, gaps, warnings."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from booster_deploy.telemetry.replay import ReplayRun


def describe(directory, execution=None):
    row = {'run': directory.name}
    try:
        run = ReplayRun(directory, execution=execution)
    except Exception as exc:
        row['error'] = f'{type(exc).__name__}: {exc}'
        return row
    counts = run.describe()['counts']
    row.update(topology=run.topology, execution=run.execution, duration=round(run.duration, 1),
               windows=len(run.streams), streams='/'.join(run.streams),
               human=counts.get('human', 0), real=counts.get('real', 0), sim=counts.get('sim', 0),
               retarget=counts.get('retarget', 0),
               derived='yes' if (directory / 'derived_simulation.npz').exists() else 'no',
               warnings=','.join(run.warnings) or '-')
    return row


def main(argv):
    root = Path(argv[1]).expanduser() if len(argv) > 1 else ROOT / 'logs'
    directories = sorted(p for p in root.iterdir() if (p / 'manifest.json').exists()) if root.is_dir() else []
    if not directories:
        print(f'no telemetry runs under {root}')
        return 1
    rows = [describe(d) for d in directories]
    prefix = 'booster-teleop-onboard-pico-'
    for row in rows:
        row['short'] = row['run'][len(prefix):] if row['run'].startswith(prefix) else row['run']
    header = ('run', 'exec', 'windows', 'human', 'real', 'sim', 'retarget', 'derived', 'seconds', 'warnings')
    print('{:<32} {:>4} {:>7} {:>7} {:>6} {:>5} {:>8} {:>7} {:>8}  {}'.format(*header))
    for row in rows:
        if 'error' in row:
            print(f'{row["short"]:<32} ERROR {row["error"]}')
            continue
        print('{:<32} {:>4} {:>7} {:>7} {:>6} {:>5} {:>8} {:>7} {:>8}  {}'.format(
            row['short'][:32], row['execution'], row['windows'], row['human'], row['real'],
            row['sim'], row['retarget'], row['derived'], row['duration'], row['warnings']))
    Path(root / 'inventory.json').write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    print(f'\nwritten: {root / "inventory.json"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))
