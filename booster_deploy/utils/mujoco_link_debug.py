
from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

from .live_terminal_panel import LiveTerminalPanel, PanelSection


class MujocoLinkDebug:

    def __init__(self, model: mujoco.MjModel, cfg: Any) -> None:

        self._model = model
        self._print_viewer_force = bool(getattr(cfg, "print_viewer_applied_force", False))
        self._viewer_force_threshold = max(float(getattr(cfg, "viewer_force_print_threshold", 1e-4)), 0.0)
        self._viewer_force_include_torque = bool(getattr(cfg, "viewer_force_print_torque", False))

        self._force_body_name = getattr(cfg, "force_apply_link", None)
        self._configured_force = self._parse_configured_force(getattr(cfg, "apply_force", None))
        self._force_body_id = self._resolve_force_body_id()
        self._last_configured_qfrc = np.zeros(int(model.nv), dtype=np.float64)

        position_names = getattr(cfg, "position_print_link", None) or []
        self._position_body_ids = [
            self._resolve_body_id(name, "position_print_link") for name in position_names
        ]
        self._position_body_names = [self._body_name(body_id) for body_id in self._position_body_ids]

        self._panel = LiveTerminalPanel(
            refresh_hz=float(getattr(cfg, "mujoco_debug_panel_refresh_hz", 10.0)),
            enabled=bool(getattr(cfg, "mujoco_debug_panel_enabled", True)),
        )

    @property
    def enabled(self) -> bool:

        return (
            self._print_viewer_force
            or self._configured_force is not None
            or bool(self._position_body_ids)
        )

    def apply_configured_force(self, data: mujoco.MjData) -> None:

        if self._configured_force is None or self._force_body_id is None:
            return

        data.qfrc_applied[:] -= self._last_configured_qfrc
        self._last_configured_qfrc[:] = 0.0

        torque = np.zeros(3, dtype=np.float64)
        point = np.asarray(data.xpos[self._force_body_id], dtype=np.float64)
        mujoco.mj_applyFT(
            self._model,
            data,
            self._configured_force,
            torque,
            point,
            self._force_body_id,
            self._last_configured_qfrc,
        )
        data.qfrc_applied[:] += self._last_configured_qfrc

    def update_display(self, data: mujoco.MjData) -> None:

        if not self.enabled:
            return
        self._panel.render("MuJoCo Link Debug", self._make_sections(data))

    def close(self, data: mujoco.MjData | None = None) -> None:

        if data is not None and np.any(self._last_configured_qfrc):
            data.qfrc_applied[:] -= self._last_configured_qfrc
            self._last_configured_qfrc[:] = 0.0
        self._panel.close()

    def _make_sections(self, data: mujoco.MjData) -> list[PanelSection]:

        sections: list[PanelSection] = []

        if self._print_viewer_force:
            viewer_force_lines = self._viewer_force_lines(data)
            if viewer_force_lines:
                sections.append(PanelSection("viewer applied force", viewer_force_lines))

        if self._configured_force is not None and self._force_body_id is not None:
            force = self._configured_force
            sections.append(
                PanelSection(
                    "configured force",
                    [
                        f"{self._body_name(self._force_body_id):<28} "
                        f"F=[{force[0]: .3f}, {force[1]: .3f}, {force[2]: .3f}] N"
                    ],
                )
            )

        if self._position_body_ids:
            position_lines = []
            for body_name, body_id in zip(self._position_body_names, self._position_body_ids):
                pos = np.asarray(data.xpos[body_id], dtype=np.float64)
                position_lines.append(
                    f"{body_name:<28} p=[{pos[0]: .3f}, {pos[1]: .3f}, {pos[2]: .3f}] m"
                )
            sections.append(PanelSection("link position", position_lines))

        if not sections:
            sections.append(PanelSection("status", ["no active debug output"]))
        return sections

    def _viewer_force_lines(self, data: mujoco.MjData) -> list[str]:

        xfrc = np.asarray(data.xfrc_applied, dtype=np.float64)
        if xfrc.ndim != 2 or xfrc.shape[1] < 6:
            return []

        force_norms = np.linalg.norm(xfrc[:, :3], axis=1)
        active_body_ids = np.flatnonzero(force_norms > self._viewer_force_threshold)
        lines = []
        for body_id in active_body_ids:
            wrench = xfrc[int(body_id)]
            line = (
                f"{self._body_name(int(body_id)):<28} "
                f"F=[{wrench[0]: .3f}, {wrench[1]: .3f}, {wrench[2]: .3f}] N"
            )
            if self._viewer_force_include_torque:
                line += f"  T=[{wrench[3]: .3f}, {wrench[4]: .3f}, {wrench[5]: .3f}] N*m"
            lines.append(line)
        return lines

    def _parse_configured_force(self, force: Any) -> np.ndarray | None:

        has_link = self._force_body_name is not None and str(self._force_body_name) != ""
        has_force = force is not None
        if has_link != has_force:
            raise ValueError("force_apply_link and apply_force must be configured together")
        if not has_force:
            return None

        force_array = np.asarray(force, dtype=np.float64).reshape(-1)
        if force_array.shape != (3,):
            raise ValueError(f"apply_force must contain exactly 3 values, got shape {force_array.shape}")
        return force_array

    def _resolve_force_body_id(self) -> int | None:

        if self._configured_force is None:
            return None
        return self._resolve_body_id(str(self._force_body_name), "force_apply_link")

    def _resolve_body_id(self, body_name: str, field_name: str) -> int:

        body_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, str(body_name))
        if body_id < 0:
            raise ValueError(f"{field_name} references unknown MuJoCo body/link: {body_name}")
        return int(body_id)

    def _body_name(self, body_id: int) -> str:

        name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_BODY, int(body_id))
        return str(name) if name else f"body_{int(body_id)}"
