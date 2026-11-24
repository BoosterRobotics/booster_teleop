#!/usr/bin/env python3
"""List registered tasks with the input path their policy consumes.

The launcher uses this to build the task menu and to reject a task/input-path
mismatch. The input path is declared by the task's own ``policy.input_path``
(``"retarget"`` / ``"smpl"``), never inferred from the task name.

Output: ``<name>\t<input_path>\t<policy class>\t<PICO input mode>`` per line.
"""

from __future__ import annotations

import argparse
import importlib
import pkgutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", default="", help="Only list tasks starting with this prefix.")
    parser.add_argument("--input-path", default=None,
                        help="Only list tasks declaring this input path.")
    args = parser.parse_args(argv)

    import tasks as tasks_pkg

    for module_info in pkgutil.walk_packages(tasks_pkg.__path__, prefix="tasks."):
        try:
            importlib.import_module(module_info.name)
        except Exception as exc:  # pragma: no cover - reported on stderr only
            print(f"# skipped {module_info.name}: {exc.__class__.__name__}: {exc}", file=sys.stderr)

    from booster_deploy.utils.registry import list_tasks

    rows = []
    for name, cfg in sorted(list_tasks().items()):
        if args.prefix and not name.startswith(args.prefix):
            continue
        policy = cfg.policy
        default_path = str(getattr(policy, 'input_path', '') or 'retarget')
        select_input_path = getattr(cfg, 'select_input_path', None)
        paths = (default_path,) if select_input_path is None else (
            default_path, 'retarget' if default_path == 'smpl' else 'smpl')
        for input_path in paths:
            if args.input_path and input_path != args.input_path:
                continue
            route_policy = policy if input_path == default_path else select_input_path(input_path).policy
            constructor = getattr(route_policy, 'constructor', None)
            policy_class = getattr(constructor, '__name__', '') or type(route_policy).__name__
            input_mode = str(getattr(route_policy, 'pico_input_mode', '') or '')
            if input_path == 'retarget':
                input_mode = 'retarget_rt' if getattr(route_policy, 'rt_reference_topic', None) else 'recording'
            rows.append((name, input_path, policy_class, input_mode))

    for name, input_path, policy_class, input_mode in rows:
        print(f"{name}\t{input_path}\t{policy_class}\t{input_mode}")
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
