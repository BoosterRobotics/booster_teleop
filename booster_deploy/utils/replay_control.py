"""Cross-process pause/resume signals for offline replay (atomic file + TTL, not a safety stop).

Why: pressing ``Y`` previously froze only the policy while replay kept advancing, so a finished clip would "return to the trajectory
start" (actually a new cycle). Pausing now also freezes replay; deploy and publisher are **separate
processes**, so state is passed through ``replay.state`` in the session diag directory.

Two review findings (gpt-6-astra) were addressed with minimal changes:
* Writes must be atomic (tmp + ``os.replace``), or readers may see a partial file;
* State must identify its owner and freshness (pid / cycle / write time); stale state is treated as running,
    preventing leftover state from a previous cycle from freezing a new session.

This is **not** a safety-stop mechanism: hardware safety still relies on emergency stop/damping and the state machine.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

FILENAME = "replay.state"


def control_path(diag_dir: str | os.PathLike[str] | None = None) -> Path | None:
    base = Path(diag_dir) if diag_dir else Path(os.getenv("BOOSTER_DIAG_DIR", "").strip() or ".")
    if not str(base) or str(base) == ".":
        return None
    return base / FILENAME


def write_state(
    paused: bool,
    diag_dir: str | os.PathLike[str] | None = None,
    *,
    cycle: int | None = None,
) -> None:
    path = control_path(diag_dir)
    if path is None:
        return
    payload = {
        "paused": bool(paused),
        "pid": os.getpid(),
        "updated_ns": time.time_ns(),
        "cycle": cycle,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        os.replace(tmp, path)  # atomic: readers never see a half-written file
    except Exception:  # noqa: BLE001
        try:
            tmp.unlink()
        except Exception:  # noqa: BLE001
            pass


def _write_token(run_dir, name, token):
    if not run_dir:
        return False
    target = Path(run_dir) / name
    tmp = target.with_name(f'.{name}.{os.getpid()}.tmp')
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(dict(pid=os.getpid(), at_ns=int(token))))
        os.replace(tmp, target)
        return True
    except OSError:
        return False
    finally:
        try: tmp.unlink(missing_ok=True)
        except OSError: pass


def write_input(run_dir, **selection):
    if not run_dir:
        return False
    target = Path(run_dir) / 'replay.input.json'
    tmp = target.with_name(f'.replay.input.{os.getpid()}.tmp')
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(selection))
        os.replace(tmp, target)
        return True
    except OSError:
        return False
    finally:
        try: tmp.unlink(missing_ok=True)
        except OSError: pass


def read_input(run_dir):
    if not run_dir:
        return None
    try:
        data = json.loads((Path(run_dir) / 'replay.input.json').read_text())
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def read_signal(run_dir, name):
    try:
        return int(json.loads((Path(run_dir) / name).read_text())['at_ns']) if run_dir else 0
    except (OSError, ValueError, TypeError, KeyError):
        return 0


def fence_replay_restart(run_dir: Path) -> bool:
    """Keep the old markers, but stop a replacement publisher from consuming them."""
    boundary = max(read_signal(run_dir, 'replay.restart'), read_prepare_ns(run_dir), read_start_ns(run_dir))
    return _write_token(run_dir, 'replay.restart', boundary)


def replay_restart_boundary(run_dir: str | Path | None) -> int:
    return read_signal(run_dir, 'replay.restart')


class PreparationCycle:
    def __init__(self, run_dir, cycle, stopped=lambda: False):
        self.directory, self.cycle, self.stopped = run_dir, int(cycle), stopped
        self.deadline = time.monotonic() + 45.

    def check(self):
        if (self.stopped() or read_prepare_ns(self.directory) != self.cycle
                or read_abort(self.directory, self.cycle)):
            raise InterruptedError('offline preparation cancelled')
        if time.monotonic() > self.deadline and not self.playing():
            raise TimeoutError('offline preparation/play acknowledgement timed out')

    def playing(self):
        return play_authorized(self.directory, self.cycle)

    def first(self, resend):
        start = time.monotonic()
        while read_signal(self.directory, 'replay.first') != self.cycle:
            self.check()
            resend()
            time.sleep(.02)
        return time.monotonic() - start

    def before_next(self):
        start = time.monotonic()
        self.check()
        while read_signal(self.directory, 'replay.prepared') == self.cycle and not self.playing():
            self.check()
            time.sleep(.01)
        return time.monotonic() - start


def read_prepare_ns(run_dir):
    if not run_dir:
        return 0
    try:
        return int(json.loads((Path(run_dir) / 'replay.prepare').read_text())['at_ns'])
    except (OSError, ValueError, TypeError, KeyError):
        return 0


def write_prepare(run_dir):
    token = max(time.time_ns(), read_prepare_ns(run_dir) + 1, replay_restart_boundary(run_dir) + 1)
    return token if _write_token(run_dir, 'replay.prepare', token) else 0


def play_authorized(run_dir, token):
    return bool(token and read_prepare_ns(run_dir) == token
                and read_start_ns(run_dir) == token and not read_abort(run_dir, token))


def write_start(run_dir: Path | str | None, *, token: int | None = None) -> bool:
    """Ask the replay to start one cycle (the deploy writes this on A).

    Why a token instead of relying on a second ROS subscriber: the publisher used
    to wait for its own ``/remote_controller_state`` A edge. When that subscriber
    failed to start (rclpy missing in that pane), the publisher waited forever
    while the deploy happily accepted A -- robot never moved, no error anywhere.
    """

    if token is not None and (read_prepare_ns(run_dir) != token or read_abort(run_dir, token)):
        return False
    return _write_token(run_dir, 'replay.start', token if token is not None else time.time_ns())


def read_start_ns(run_dir: Path | str | None) -> int:
    """Timestamp of the last start request (0 when there is none)."""

    if run_dir is None:
        return 0
    try:
        data = json.loads((Path(run_dir) / "replay.start").read_text(encoding="utf-8"))
        return int(data.get("at_ns") or 0)
    except (OSError, ValueError, TypeError):
        return 0


def write_abort(run_dir: Path | str | None, *, token=None) -> None:
    """Accepted B aborts only the current A token, not the next replay cycle."""
    if run_dir is None:
        return
    token = max(read_start_ns(run_dir), read_prepare_ns(run_dir)) if token is None else int(token)
    if not token:
        return
    _write_token(run_dir,f'replay.abort.{token}',token)
    target = Path(run_dir) / 'replay.abort'
    tmp = target.with_suffix(f'.{os.getpid()}.tmp')
    try:
        tmp.write_text(json.dumps(dict(start_ns=token)), encoding='utf-8')
        os.replace(tmp, target)
    except OSError:
        return


def read_abort(run_dir: Path | str | None, start_ns: int) -> bool:
    if run_dir is None or not start_ns:
        return False
    if read_signal(run_dir,f'replay.abort.{int(start_ns)}') == start_ns:
        return True
    try:
        data = json.loads((Path(run_dir) / 'replay.abort').read_text(encoding='utf-8'))
        return data.get('start_ns') == start_ns
    except (OSError, ValueError, TypeError):
        return False


def read_state(path: Path | None) -> bool:
    """True only for a ``paused`` record written by a still-running process."""

    if path is None:
        return False
    try:
        raw = path.read_text(encoding="utf-8").strip()
    except Exception:  # noqa: BLE001
        return False
    if not raw:
        return False
    if raw.startswith("{"):
        try:
            data = json.loads(raw)
        except Exception:  # noqa: BLE001
            return False
        pid = int(data.get("pid") or 0)
        if pid and pid != os.getpid():
            # A leftover from a run whose writer is gone must not freeze the replay.
            # Long pauses by a live writer stay honoured (no time-based expiry).
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return False
            except PermissionError:
                pass
        return bool(data.get("paused"))
    # Legacy plain-text format written by earlier builds.
    return raw.lower() == "paused"
