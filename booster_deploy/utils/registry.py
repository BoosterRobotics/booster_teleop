
from __future__ import annotations

from typing import Dict
from ..controllers.controller_cfg import ControllerCfg


_TASK_REGISTRY: Dict[str, ControllerCfg] = {}


def register_task(name: str, cfg: ControllerCfg) -> None:
    if name in _TASK_REGISTRY:
        raise KeyError(f"Task '{name}' already registered")
    _TASK_REGISTRY[name] = cfg


def get_task(name: str) -> ControllerCfg:

    return _TASK_REGISTRY[name]


def list_tasks() -> Dict[str, ControllerCfg]:

    return dict(_TASK_REGISTRY)


__all__ = [
    "register_task", "get_task", "list_tasks",
]
