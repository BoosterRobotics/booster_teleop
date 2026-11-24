"""Atomic run configuration and offline-trajectory lock shared across processes."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
from typing import Any, Iterator

from booster_deploy.utils.project_path import PROJECT_ROOT, resolve_project_path

CONFIG_NAME = "execution_config.json"
LOCK_NAME = ".execution_config.lock"


def _paths(directory: str | Path) -> tuple[Path, Path]:
    root = Path(directory).expanduser().resolve()
    return root / CONFIG_NAME, root / LOCK_NAME


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except FileNotFoundError:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"invalid execution config: {path}")
    return value


def _write(path: Path, value: dict[str, Any]) -> None:
    fd, temporary = tempfile.mkstemp(prefix=".execution-config-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _trajectory(value: str | Path | dict[str, Any] | None) -> dict[str, str] | None:
    if not value:
        return None
    if isinstance(value, dict):
        value = value.get("path")
    if not value:
        return None
    path = Path(value).expanduser().resolve()
    return {"path": str(path), "name": path.name, "stem": path.stem}


def trajectory_switch_block_reason(config: dict[str, Any]) -> str | None:
    if not (config.get("telemetry_enabled") and config.get("offline")):
        return None
    unlimited = config.get("trajectory_switch") == "unlimited"
    if config.get("playback_start_pending") or config.get("trajectory_playback_active"):
        return "Trajectory switching is locked during offline playback setup or playback; return to Walking first"
    if not unlimited:
        if config.get("first_offline_playback_started"):
            return "Telemetry logging locks the clip after the first offline playback starts"
        if int(config.get("trajectory_switch_count", 0)) >= 1:
            return "Telemetry logging allows only one clip change before the first playback"
    return None


def offline_playback_provenance(directory: str | Path) -> dict[str, Any]:
    config = read_execution_config(directory)
    history = config.get("trajectory_history") or []
    if not history:
        return {}
    return {"trajectory_index": len(history) - 1, "trajectory_path": history[-1]["path"]}


def initialize_execution_config(directory: str | Path, values: dict[str, Any]) -> dict[str, Any]:
    """Create the initial run snapshot. Existing snapshots are never overwritten."""
    path, lock_path = _paths(directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            return _read(path)
        switch = values.get("trajectory_switch") or "once"
        if switch not in {"once", "unlimited"}:
            raise ValueError("trajectory switch must be once or unlimited")
        trajectory = _trajectory(values.get("trajectory"))
        if values.get("telemetry_enabled", True) and values.get("replay") == "offline" and not trajectory:
            raise ValueError("offline telemetry requires a selected trajectory")
        config = {
            "schema_version": 1,
            "started_at": values.get("started_at") or time.strftime("%Y%m%d-%H%M%S"),
            "offline": values.get("replay") == "offline",
            "telemetry_enabled": bool(values.get("telemetry_enabled", True)),
            "sn": values.get("sn") or None,
            "task": values.get("task"),
            "execution": values.get("execution"),
            "input_source": values.get("input_source"),
            "input_path": values.get("input_path"),
            "topology": values.get("topology"),
            "mjcf": values.get("mjcf"),
            "trajectory": trajectory,
            "trajectory_switch": switch,
            "trajectory_history": ([dict(trajectory, selected_ns=time.time_ns())]
                                   if trajectory and values.get("replay") == "offline" else []),
            "trajectory_switch_count": 0,
            "first_offline_playback_started": False,
            "trajectory_playback_active": False,
            "playback_start_pending": False,
            "playback_start_pending_owner": None,
            "model": {
                "requested_path": None,
                "source": "task_default",
                "resolved_path": None,
                "action_offset_source": None,
                "action_scale_factor": None,
            },
        }
        _write(path, config)
        return config


def read_execution_config(directory: str | Path) -> dict[str, Any]:
    path, _ = _paths(directory)
    return _read(path)


def update_execution_config(directory: str | Path, **updates: Any) -> dict[str, Any]:
    path, lock_path = _paths(directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        config = _read(path)
        if not config:
            raise FileNotFoundError(f"execution config missing: {path}")
        if "trajectory" in updates:
            updates["trajectory"] = _trajectory(updates["trajectory"])
        config.update(updates)
        if config.get("telemetry_enabled") and config.get("offline") and not config.get("trajectory"):
            raise ValueError("offline telemetry requires a selected trajectory")
        _write(path, config)
        return config


def provenance_path(directory: Path, value: str | None, repo_relative: str | None = None) -> Path | None:
    """Resolve a recorded path, with a repository fallback for copied runs."""
    candidates = []
    if value:
        path = Path(value).expanduser()
        candidates.extend((path,) if path.is_absolute() else (directory / path, PROJECT_ROOT / path))
    if repo_relative:
        candidates.append(PROJECT_ROOT / repo_relative)
    if not candidates:
        return None
    return next((path.resolve() for path in candidates if path.is_file()), candidates[0].resolve())



def model_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def record_execution_policy(directory: str | Path, task: str, cfg: Any, *, execution: str) -> dict[str, Any]:
    """Record the actual deploy configuration, regardless of which pane initialized the run."""
    model = resolve_project_path(cfg.policy.onnx_model_path)
    robot = cfg.robot
    mjcf = resolve_project_path(robot.mjcf_path)

    def repo_relative(path: Path) -> str | None:
        return str(path.relative_to(PROJECT_ROOT)) if path.is_relative_to(PROJECT_ROOT) else None

    scale = float(robot.action_scale_factor)
    return update_execution_config(
        directory, task=task, execution=execution, topology="retarget", input_path="retarget", mjcf=str(mjcf),
        model={
            "source": "task_default",
            "requested_path": None,
            "resolved_path": str(model),
            "repo_relative_path": repo_relative(model),
            "sha256": model_sha256(model),
            "action_offset_source": cfg.policy.action_offset_source,
            "action_scale_factor": scale,
        },
        policy_config={
            "rt_reference_transition_s": float(cfg.policy.rt_reference_transition_s),
        },
        robot_config={
            "joint_names": list(robot.joint_names),
            "default_joint_pos": list(robot.default_joint_pos),
            "joint_stiffness": list(robot.joint_stiffness),
            "joint_damping": list(robot.joint_damping),
            "effort_limit": list(robot.effort_limit),
            "action_scale_factor": scale,
            "mjcf": str(mjcf),
            "mjcf_repo_relative": repo_relative(mjcf),
        },
    )


def reserve_offline_playback_start(directory: str | Path) -> bool:
    """Reserve A against a concurrent trajectory switch."""
    path, lock_path = _paths(directory)
    if not path.is_file():
        return True
    try:
        lock = lock_path.open("a+")
    except OSError:
        return False
    with lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            config = _read(path)
        except (BlockingIOError, OSError, ValueError, json.JSONDecodeError):
            return False
        if not (config.get("telemetry_enabled") and config.get("offline")):
            return True
        if config.get("first_offline_playback_started") and config.get("trajectory_switch") != "unlimited":
            return True
        if config.get("playback_start_pending"):
            return config.get("playback_start_pending_owner") == os.getpid()
        config["playback_start_pending"] = True
        config["playback_start_pending_owner"] = os.getpid()
        try:
            _write(path, config)
        except (OSError, TypeError, ValueError):
            return False
        return True


def commit_offline_playback_start(directory: str | Path, *, nonblocking: bool = False) -> bool:
    """Commit the A reservation before starting replay; publishers may confirm it again."""
    path, lock_path = _paths(directory)
    if not path.is_file():
        return True
    try:
        lock = lock_path.open("a+")
    except OSError:
        return False
    with lock:
        try:
            flags = fcntl.LOCK_EX | (fcntl.LOCK_NB if nonblocking else 0)
            fcntl.flock(lock, flags)
            config = _read(path)
        except (BlockingIOError, OSError, ValueError, json.JSONDecodeError):
            return False
        if not (config.get("telemetry_enabled") and config.get("offline")):
            return True
        unlimited = config.get("trajectory_switch") == "unlimited"
        if not config.get("playback_start_pending"):
            return bool(config.get("trajectory_playback_active") if unlimited
                        else config.get("first_offline_playback_started"))
        config["playback_start_pending"] = False
        config["playback_start_pending_owner"] = None
        config["first_offline_playback_started"] = True
        if unlimited:
            config["trajectory_playback_active"] = True
        try:
            _write(path, config)
        except (OSError, TypeError, ValueError):
            return False
        return True


def lock_offline_playback_for_autostart(directory: str | Path) -> bool:
    """Lock an offline clip before an ungated simulator publisher emits its first frame."""
    return reserve_offline_playback_start(directory) and commit_offline_playback_start(directory)


def clear_offline_playback_start_pending(directory: str | Path) -> None:
    path, lock_path = _paths(directory)
    if not path.is_file():
        return
    try:
        lock = lock_path.open("a+")
    except OSError:
        return
    with lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            config = _read(path)
        except (BlockingIOError, OSError, ValueError, json.JSONDecodeError):
            return
        if config.get("playback_start_pending"):
            config["playback_start_pending"] = False
            config["playback_start_pending_owner"] = None
            try:
                _write(path, config)
            except (OSError, TypeError, ValueError):
                return


def release_offline_playback(directory: str | Path) -> bool:
    """Release an unlimited replay only after the controller confirms Walking."""
    path, lock_path = _paths(directory)
    if not path.is_file():
        return True
    try:
        with lock_path.open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            config = _read(path)
            if config.get("trajectory_playback_active") or config.get("playback_start_pending"):
                config["trajectory_playback_active"] = False
                config["playback_start_pending"] = False
                config["playback_start_pending_owner"] = None
                _write(path, config)
    except (OSError, ValueError):
        return False
    return True


@contextmanager
def trajectory_switch(directory: str | Path | None, new_path: str | Path) -> Iterator[Any]:
    """Reserve a trajectory change until publisher/deploy restart succeeds."""
    if directory is None:
        yield SimpleNamespace(commit=lambda: None)
        return
    path, lock_path = _paths(directory)
    if not path.is_file():
        yield SimpleNamespace(commit=lambda: None)
        return
    new_trajectory = _trajectory(new_path)
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        config = _read(path)
        old_trajectory = config.get("trajectory") or {}
        old_path = str(Path(old_trajectory.get("path", "")).expanduser().resolve()) if old_trajectory.get("path") else ""
        if new_trajectory and new_trajectory["path"] == old_path:
            yield SimpleNamespace(commit=lambda: None)
            return
        reason = trajectory_switch_block_reason(config)
        if reason:
            raise ValueError(reason)
        def commit() -> None:
            config["trajectory"] = new_trajectory
            if config.get("telemetry_enabled") and config.get("offline"):
                config["trajectory_switch_count"] = int(config.get("trajectory_switch_count", 0)) + 1
                history = config.setdefault("trajectory_history", [])
                if not history and old_trajectory:
                    history.append(dict(old_trajectory, selected_ns=None))
                history.append(dict(new_trajectory, selected_ns=time.time_ns()))
            _write(path, config)

        yield SimpleNamespace(commit=commit)
