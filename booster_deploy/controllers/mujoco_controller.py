
from __future__ import annotations

import csv
import logging
import os
import sys
import threading
from pathlib import Path
from time import perf_counter, sleep
import select
import numpy as np
import torch
import mujoco
import mujoco.viewer
from .base_controller import BaseController, ControllerCfg, VelocityCommand
from ..hand_retargeting.mujoco_control import (
    HandTargetConditioner,
    apply_hand_mujoco_passive_tuning,
    validate_disjoint_actuator_groups,
)
from ..inputs.control import ControlInputManager, LocomotionMapper, LocomotionMapperConfig
from ..types.hand_retarget_types import HandRetargetFrameNs
from ..utils.async_viewer import (
    AsyncViewerRuntime,
    AsyncViserViewerRuntime,
    ViewerSnapshot,
)
from ..utils.periodic_deadline import PeriodicDeadline
from ..utils.progress import print_progress
from ..utils.mjcf_path import resolve_mjcf_path
from ..utils.project_path import resolve_project_path
from ..utils.mujoco_link_debug import MujocoLinkDebug
from ..visualization.viewer import create_viewer, is_viewer_backend_available
from ..visualization.drawers import MujocoSceneDrawer
from ..visualization.layers import CommandVelocityArrowLayer, HumanSkeletonLayer
from ..inputs.pico.coordinate import pico_skeleton_connections


logger = logging.getLogger("booster_deploy")


class MujocoController(BaseController):

    def __init__(self, cfg: ControllerCfg):

        super().__init__(cfg)

        mjcf_path = str(resolve_mjcf_path(self.robot.cfg.mjcf_path))
        self.mj_model = mujoco.MjModel.from_xml_path(mjcf_path)
        self.mj_model.opt.timestep = self.cfg.mujoco.physics_dt

        if getattr(self.cfg.mujoco, 'disable_self_collision', False):
            for i in range(self.mj_model.ngeom):
                geom_name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, i)

                if geom_name and ('floor' in geom_name.lower() or 'ground' in geom_name.lower()):
                    continue

                if self.mj_model.geom_contype[i] == 1:
                    self.mj_model.geom_conaffinity[i] = 0
        self.decimation = self.cfg.mujoco.decimation
        self._body_joint_qposadr, self._body_joint_dofadr = self._resolve_robot_joint_addresses()
        self._actuator_by_joint_name = self._build_actuator_joint_map()
        self._body_actuator_ids = self._resolve_body_actuator_ids()
        self._hand_actuator_by_joint_name = self._resolve_hand_actuator_map()
        self._hand_qpos_by_joint_name = self._resolve_hand_qpos_map()
        hand_actuator_ids = np.asarray(sorted(set(self._hand_actuator_by_joint_name.values())), dtype=np.int32)
        self._body_actuator_ids, self._hand_actuator_ids = validate_disjoint_actuator_groups(
            self._body_actuator_ids,
            hand_actuator_ids,
            actuator_count=int(self.mj_model.nu),
        )
        self._hand_target_conditioner: HandTargetConditioner | None = None
        if self._hand_actuator_ids.size:
            mimic_dof_ids, equality_ids = self._resolve_hand_mimic_constraints()
            model_family = "revo2" if "revo2" in Path(mjcf_path).stem.lower() else None
            tuning = apply_hand_mujoco_passive_tuning(
                self.mj_model,
                actuator_ids=self._hand_actuator_ids,
                mimic_dof_ids=mimic_dof_ids,
                equality_ids=equality_ids,
                mjcf_path=mjcf_path,
                model_family=model_family,
            )
            ctrl_ranges = np.asarray(self.mj_model.actuator_ctrlrange, dtype=np.float64).copy()
            for actuator_id in self._hand_actuator_ids:
                if not bool(self.mj_model.actuator_ctrllimited[int(actuator_id)]):
                    ctrl_ranges[int(actuator_id)] = (-np.inf, np.inf)
            self._hand_target_conditioner = HandTargetConditioner(
                self._hand_actuator_ids,
                ctrl_ranges,
                max_velocity=float(self.cfg.mujoco.hand_target_max_velocity),
                filter_alpha=float(self.cfg.mujoco.hand_target_filter_alpha),
            )
            logger.info(
                "Configured %d hand actuators separately from %d body actuators "
                "(max_velocity=%.3frad/s, filter_alpha=%.3f, kp_cap=%.3f)",
                self._hand_actuator_ids.size,
                self._body_actuator_ids.size,
                self._hand_target_conditioner.max_velocity,
                self._hand_target_conditioner.filter_alpha,
                float(tuning["actuator_kp_cap"]),
            )
        self._hand_retarget_frame: HandRetargetFrameNs | None = None
        self._warned_missing_hand_actuators: set[str] = set()
        self.mj_data = mujoco.MjData(self.mj_model)
        mujoco.mj_resetData(self.mj_model, self.mj_data)

        if self.cfg.mujoco.init_joint_pos is not None:
            init_joint_pos = np.array(self.cfg.mujoco.init_joint_pos, dtype=np.float64)
        else:
            init_joint_pos = self.robot.default_joint_pos.detach().cpu().numpy().astype(np.float64, copy=True)
        if init_joint_pos.shape[0] != self.robot.num_joints:
            raise ValueError(
                f"init_joint_pos must have {self.robot.num_joints} body joints, got {init_joint_pos.shape[0]}"
            )

        init_qpos = np.asarray(self.mj_model.qpos0, dtype=np.float64).copy()
        init_qpos[:3] = np.array(self.cfg.mujoco.init_pos, dtype=np.float64)
        init_qpos[3:7] = np.array(self.cfg.mujoco.init_quat, dtype=np.float64)
        init_qpos[self._body_joint_qposadr] = init_joint_pos
        self.mj_data.qpos[:] = init_qpos
        mujoco.mj_forward(self.mj_model, self.mj_data)
        self._ghost_mj_data = mujoco.MjData(self.mj_model)
        self._ghost_mj_data.qpos[:] = self.mj_data.qpos
        self._ghost_mj_data.qvel[:] = 0.0
        mujoco.mj_forward(self.mj_model, self._ghost_mj_data)
        self._ghost_rgba = np.array(
            self.cfg.mujoco.ghost_rgba, dtype=np.float32)
        self._ghost_scene_option = mujoco.MjvOption()
        self._reference_qpos: np.ndarray | None = None
        self._generalized_csv_files = {}
        self._generalized_csv_writers = {}
        self._generalized_csv_rows = 0
        self._generalized_csv_flush_every = max(int(getattr(self.cfg.mujoco, "log_csv_flush_every", 50)), 1)
        self._last_joint_qcmd: np.ndarray | None = None
        self._init_generalized_csv_loggers()
        self._overlay_drawer = None
        self._remote_control = None
        self._control_input = None
        self._mujoco_link_debug = MujocoLinkDebug(self.mj_model, self.cfg.mujoco)
        if not self._mujoco_link_debug.enabled:
            self._mujoco_link_debug = None


    def _joint_qpos_width(self, joint_id: int) -> int:
        start = int(self.mj_model.jnt_qposadr[joint_id])
        next_addrs = [int(addr) for addr in self.mj_model.jnt_qposadr if int(addr) > start]
        stop = min(next_addrs) if next_addrs else int(self.mj_model.nq)
        return stop - start

    def _resolve_robot_joint_addresses(self) -> tuple[np.ndarray, np.ndarray]:
        qposadr: list[int] = []
        dofadr: list[int] = []
        for joint_name in self.robot.cfg.joint_names:
            joint_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id < 0:
                raise ValueError(f"Robot joint {joint_name!r} not found in MuJoCo model")
            width = self._joint_qpos_width(joint_id)
            if width != 1:
                raise ValueError(f"Robot joint {joint_name!r} must be scalar, got qpos width {width}")
            qposadr.append(int(self.mj_model.jnt_qposadr[joint_id]))
            dofadr.append(int(self.mj_model.jnt_dofadr[joint_id]))
        return np.asarray(qposadr, dtype=np.int32), np.asarray(dofadr, dtype=np.int32)

    def _build_actuator_joint_map(self) -> dict[str, int]:
        result: dict[str, int] = {}
        for actuator_id in range(int(self.mj_model.nu)):
            joint_id = int(self.mj_model.actuator_trnid[actuator_id, 0])
            if joint_id < 0 or joint_id >= int(self.mj_model.njnt):
                continue
            joint_name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            if joint_name:
                result.setdefault(str(joint_name), actuator_id)
        return result

    def _resolve_body_actuator_ids(self) -> np.ndarray:
        actuator_ids: list[int] = []
        missing: list[str] = []
        for joint_name in self.robot.cfg.joint_names:
            actuator_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, joint_name)
            if actuator_id < 0:
                actuator_id = self._actuator_by_joint_name.get(joint_name, -1)
            if actuator_id < 0:
                missing.append(joint_name)
                continue
            actuator_ids.append(int(actuator_id))
        if missing:
            if int(self.mj_model.nu) >= self.robot.num_joints:
                logger.warning(
                    "MuJoCo body actuator names missing for %s; falling back to first %d actuators",
                    ", ".join(missing),
                    self.robot.num_joints,
                )
                return np.arange(self.robot.num_joints, dtype=np.int32)
            raise ValueError(f"Missing MuJoCo actuators for robot joints: {missing}")
        return np.asarray(actuator_ids, dtype=np.int32)

    def _resolve_hand_actuator_map(self) -> dict[str, int]:
        body_joints = set(self.robot.cfg.joint_names)
        result: dict[str, int] = {}
        for actuator_id in range(int(self.mj_model.nu)):
            joint_id = int(self.mj_model.actuator_trnid[actuator_id, 0])
            joint_name = None
            if 0 <= joint_id < int(self.mj_model.njnt):
                joint_name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            if joint_name and joint_name not in body_joints:
                result.setdefault(str(joint_name), actuator_id)
            actuator_name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
            if actuator_name and actuator_name.startswith("act_"):
                stripped = actuator_name[4:]
                if stripped not in body_joints:
                    result.setdefault(stripped, actuator_id)
        return result

    def _resolve_hand_qpos_map(self) -> dict[str, int]:
        body_joints = set(self.robot.cfg.joint_names)
        result: dict[str, int] = {}
        for joint_id in range(int(self.mj_model.njnt)):
            joint_name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            if not joint_name or joint_name in body_joints:
                continue
            if self._joint_qpos_width(joint_id) != 1:
                continue
            result[str(joint_name)] = int(self.mj_model.jnt_qposadr[joint_id])
        return result

    def _resolve_hand_mimic_constraints(self) -> tuple[np.ndarray, np.ndarray]:
        hand_joint_names = set(self._hand_actuator_by_joint_name)
        mimic_dof_ids: list[int] = []
        equality_ids: list[int] = []
        joint_eq_type = int(mujoco.mjtEq.mjEQ_JOINT)
        for equality_id in range(int(self.mj_model.neq)):
            if int(self.mj_model.eq_type[equality_id]) != joint_eq_type:
                continue
            mimic_joint_id = int(self.mj_model.eq_obj1id[equality_id])
            source_joint_id = int(self.mj_model.eq_obj2id[equality_id])
            if not (0 <= mimic_joint_id < int(self.mj_model.njnt)):
                continue
            if not (0 <= source_joint_id < int(self.mj_model.njnt)):
                continue
            mimic_name = mujoco.mj_id2name(
                self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, mimic_joint_id
            )
            source_name = mujoco.mj_id2name(
                self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, source_joint_id
            )
            if mimic_name not in hand_joint_names and source_name not in hand_joint_names:
                continue
            mimic_dof_ids.append(int(self.mj_model.jnt_dofadr[mimic_joint_id]))
            equality_ids.append(equality_id)
        return (
            np.asarray(mimic_dof_ids, dtype=np.int32),
            np.asarray(equality_ids, dtype=np.int32),
        )

    def _set_hand_joint_qpos(self, qpos: np.ndarray, joint_name: str, value: float) -> None:
        qpos_index = self._hand_qpos_by_joint_name.get(joint_name)
        if qpos_index is not None:
            qpos[qpos_index] = float(value)

    def _apply_revo2_coupled_distal_qpos(self, qpos: np.ndarray, joint_name: str, value: float) -> None:
        if not joint_name.endswith("_proximal_joint"):
            return
        distal_name = joint_name.replace("_proximal_joint", "_distal_joint")
        if "_thumb_" in joint_name:
            distal_value = float(value)
        elif any(f"_{finger}_" in joint_name for finger in ("index", "middle", "ring", "pinky")):
            distal_value = 1.155 * float(value)
        else:
            return
        self._set_hand_joint_qpos(qpos, distal_name, distal_value)

    def set_hand_retarget_frame(self, frame: HandRetargetFrameNs | None) -> None:

        self._hand_retarget_frame = frame
        conditioner = self._hand_target_conditioner
        if frame is None or conditioner is None:
            return
        for command in frame.commands:
            names = tuple(str(name) for name in command.joint_names)
            positions = np.asarray(command.joint_pos, dtype=np.float64).reshape(-1)
            if len(names) != positions.shape[0]:
                key = f"{command.hand_side}:shape"
                if key not in self._warned_missing_hand_actuators:
                    logger.warning(
                        "Skipping MuJoCo hand command with %d names and %d positions",
                        len(names),
                        positions.shape[0],
                    )
                    self._warned_missing_hand_actuators.add(key)
                continue
            for joint_name, position in zip(names, positions, strict=True):
                actuator_id = self._hand_actuator_by_joint_name.get(joint_name)
                if actuator_id is None:
                    if joint_name not in self._warned_missing_hand_actuators:
                        logger.warning("MuJoCo hand actuator not found for joint %s", joint_name)
                        self._warned_missing_hand_actuators.add(joint_name)
                    continue
                if not np.isfinite(position):
                    key = f"{joint_name}:nonfinite"
                    if key not in self._warned_missing_hand_actuators:
                        logger.warning("Ignoring non-finite MuJoCo hand target for joint %s", joint_name)
                        self._warned_missing_hand_actuators.add(key)
                    continue
                joint_id = int(self.mj_model.actuator_trnid[int(actuator_id), 0])
                qpos_id = int(self.mj_model.jnt_qposadr[joint_id])
                conditioner.update_target(
                    int(actuator_id),
                    float(position),
                    current_position=float(self.mj_data.qpos[qpos_id]),
                )

    def _apply_hand_retarget_frame_to_ctrl(self) -> None:
        conditioner = self._hand_target_conditioner
        if conditioner is None:
            return
        actuator_ids, values = conditioner.step(float(self.mj_model.opt.timestep))
        self.mj_data.ctrl[actuator_ids] = values

    def _apply_hand_retarget_frame_to_qpos(self, qpos: np.ndarray) -> None:
        frame = self._hand_retarget_frame
        if frame is None or not self._hand_qpos_by_joint_name:
            return
        for command in frame.commands:
            names = tuple(str(name) for name in command.joint_names)
            positions = np.asarray(command.joint_pos, dtype=np.float64).reshape(-1)
            if len(names) != positions.shape[0]:
                continue
            for joint_name, position in zip(names, positions, strict=True):
                value = float(position)
                self._set_hand_joint_qpos(qpos, joint_name, value)
                self._apply_revo2_coupled_distal_qpos(qpos, joint_name, value)

    def _coerce_model_qpos(self, qpos: np.ndarray) -> np.ndarray:
        qpos = np.asarray(qpos, dtype=np.float64).reshape(-1)
        if qpos.shape[0] == int(self.mj_model.nq):
            return qpos.copy()
        expected_body_qpos = 7 + self.robot.num_joints
        if qpos.shape[0] != expected_body_qpos:
            raise ValueError(
                f"qpos must have shape (nq,) or ({expected_body_qpos},), got {qpos.shape} (nq={int(self.mj_model.nq)})"
            )
        full_qpos = np.asarray(self.mj_model.qpos0, dtype=np.float64).copy()
        full_qpos[:7] = qpos[:7]
        full_qpos[self._body_joint_qposadr] = qpos[7:]
        self._apply_hand_retarget_frame_to_qpos(full_qpos)
        return full_qpos

    def start(self):

        self._reference_qpos = None
        self._hand_retarget_frame = None
        if self._hand_target_conditioner is not None:
            self._hand_target_conditioner.reset()
        return super().start()

    def render_reference_robot(
        self,
        viewer,
        *,
        rgba: np.ndarray | None = None,
    ) -> None:
        if rgba is None:
            rgba = self._ghost_rgba

        if self._reference_qpos is not None:
            self._ghost_mj_data.qpos[:] = self._reference_qpos
            self._ghost_mj_data.qvel[:] = 0.0
            mujoco.mj_forward(self.mj_model, self._ghost_mj_data)

        if hasattr(viewer, 'add_ghost_robot'):

            viewer.add_ghost_robot(
                ghost_id="reference",
                ghost_data=self._ghost_mj_data,
                rgba=rgba,
            )
        else:
            mujoco.mjv_updateScene(
                self.mj_model,
                self._ghost_mj_data,
                self._ghost_scene_option,
                None,
                viewer.cam,
                int(mujoco.mjtCatBit.mjCAT_DYNAMIC),
                viewer.user_scn,
            )
            for i in range(viewer.user_scn.ngeom):
                viewer.user_scn.geoms[i].rgba[:] = rgba

    def set_reference_qpos(
        self,
        qpos: np.ndarray | torch.Tensor | None,
    ) -> None:
        if qpos is None:
            self._reference_qpos = None
            return

        if isinstance(qpos, torch.Tensor):
            qpos_np = qpos.detach().cpu().numpy()
        else:
            qpos_np = np.asarray(qpos)

        qpos_np = np.asarray(qpos_np, dtype=np.float64).reshape(-1)
        qpos_np = self._coerce_model_qpos(qpos_np)
        self._reference_qpos = qpos_np.copy()

    def render_realtime_human(self, viewer) -> None:

        if not hasattr(viewer, "user_scn"):
            return
        motion = getattr(self.policy, "motion", None)
        latest_human_frame = getattr(motion, "latest_human_frame", None)
        if not callable(latest_human_frame):
            return
        frame = latest_human_frame()
        if frame is None:
            return
        if self._overlay_drawer is None or getattr(self._overlay_drawer, "_scene", None) is not viewer.user_scn:
            self._overlay_drawer = MujocoSceneDrawer(viewer.user_scn)
        skeleton_connections = None
        get_skeleton_connections = getattr(motion, "skeleton_connections", None)
        if callable(get_skeleton_connections):
            skeleton_connections = get_skeleton_connections()
        if skeleton_connections is None:
            skeleton_connections = pico_skeleton_connections()
        HumanSkeletonLayer(
            frame,
            connections=skeleton_connections,
            point_rgba=(0.0, 0.85, 1.0, 0.85),
            line_rgba=(0.0, 0.75, 1.0, 0.45),
            point_size=0.018,
            line_width=0.004,
            bone_style="line",
        ).draw(self._overlay_drawer)

    def render_command_velocity_arrow(self, viewer) -> None:

        if not hasattr(viewer, "user_scn") or self.vel_command is None:
            return
        if self._overlay_drawer is None or getattr(self._overlay_drawer, "_scene", None) is not viewer.user_scn:
            self._overlay_drawer = MujocoSceneDrawer(viewer.user_scn)
        CommandVelocityArrowLayer(
            root_pos=self.mj_data.qpos.astype(np.float32)[:3],
            root_quat=self.mj_data.qpos.astype(np.float32)[3:7],
            vx=float(self.vel_command.lin_vel_x),
            vy=float(self.vel_command.lin_vel_y),
            z_offset=float(getattr(self.cfg.mujoco, "command_velocity_arrow_z", 0.55)),
            length_scale=float(getattr(self.cfg.mujoco, "command_velocity_arrow_scale", 0.55)),
        ).draw(self._overlay_drawer)

    def _init_generalized_csv_loggers(self) -> None:

        log_paths = {
            "qpos": getattr(self.cfg.mujoco, "log_generalized_qpos_csv", None),
            "qvel": getattr(self.cfg.mujoco, "log_generalized_qvel_csv", None),
            "qtor": getattr(self.cfg.mujoco, "log_generalized_qtor_csv", None),
            "qcmd": getattr(self.cfg.mujoco, "log_generalized_qcmd_csv", None),
            "qref": getattr(self.cfg.mujoco, "log_generalized_qref_csv", None),
        }
        try:
            for name, log_path in log_paths.items():
                if not log_path:
                    continue
                path = resolve_project_path(str(log_path))
                path.parent.mkdir(parents=True, exist_ok=True)
                fp = path.open("w", encoding="utf-8", newline="")
                self._generalized_csv_files[name] = fp
                self._generalized_csv_writers[name] = csv.writer(fp)
        except Exception:
            self._close_generalized_csv_loggers()
            raise

    @staticmethod
    def _generalized_qpos_xyzw_row(qpos_wxyz: np.ndarray) -> np.ndarray:

        row = np.asarray(qpos_wxyz, dtype=np.float32).reshape(-1).copy()
        if row.shape[0] < 7:
            raise ValueError(f"MuJoCo qpos must contain at least 7 values, got {row.shape[0]}")
        quat_wxyz = row[3:7].copy()
        row[3:7] = quat_wxyz[[1, 2, 3, 0]]
        return row

    def _record_generalized_csv_logs(self) -> None:

        if not self._generalized_csv_writers:
            return
        rows = {
            "qpos": self._generalized_qpos_xyzw_row(self.mj_data.qpos),
            "qvel": np.asarray(self.mj_data.qvel, dtype=np.float32).reshape(-1),
            "qtor": np.asarray(self.mj_data.qfrc_actuator, dtype=np.float32).reshape(-1),
        }
        if "qcmd" in self._generalized_csv_writers:
            if self._last_joint_qcmd is None:
                raise RuntimeError("qcmd CSV logging requires a joint target for the current policy step")
            rows["qcmd"] = self._last_joint_qcmd
        if "qref" in self._generalized_csv_writers:
            if self._reference_qpos is None:
                raise RuntimeError("qref CSV logging requires a reference qpos for the current policy step")
            rows["qref"] = self._generalized_qpos_xyzw_row(self._reference_qpos)
        for name, writer in self._generalized_csv_writers.items():
            writer.writerow([f"{float(v):.10f}" for v in rows[name]])
        self._generalized_csv_rows += 1
        if self._generalized_csv_rows % self._generalized_csv_flush_every == 0:
            for fp in self._generalized_csv_files.values():
                fp.flush()

    def _close_generalized_csv_loggers(self) -> None:

        for fp in self._generalized_csv_files.values():
            fp.flush()
            fp.close()
        self._generalized_csv_files.clear()
        self._generalized_csv_writers.clear()

    def update_vel_command(self):

        cmd: VelocityCommand = self.vel_command
        if self._control_input is not None:
            command = self._control_input.get_command()
            cmd.lin_vel_x = float(command.vx)
            cmd.lin_vel_y = float(command.vy)
            cmd.ang_vel_yaw = float(command.wz)
            return
        if self._remote_control is not None:
            cmd.lin_vel_x = self._remote_control.get_vx_cmd()
            cmd.lin_vel_y = self._remote_control.get_vy_cmd()
            cmd.ang_vel_yaw = self._remote_control.get_vyaw_cmd()
            return
        if select.select([sys.stdin], [], [], 0)[0]:
            try:
                parts = sys.stdin.readline().strip().split()
                if len(parts) == 3:
                    (cmd.lin_vel_x, cmd.lin_vel_y, cmd.ang_vel_yaw) = map(float, parts)
                    print(
                        f"Updated command to: x={cmd.lin_vel_x},"
                        f"y={cmd.lin_vel_y}, yaw={cmd.ang_vel_yaw}\n"
                        "Set command (x, y, yaw): ",
                        end="",
                    )
                else:
                    raise ValueError
            except ValueError:
                print(
                    "Invalid input. Enter three numeric values. "
                    "Set command (x, y, yaw): ",
                    end="",
                )

    def _init_velocity_command_source(self) -> None:

        if self.vel_command is None:
            return
        source = str(getattr(self.cfg.mujoco, "velocity_command_source", "stdin"))
        if source == "stdin":
            print("\nSet command (x, y, yaw): ", end="")
            return
        mapper = LocomotionMapper(
            LocomotionMapperConfig(
                max_vx=float(self.vel_command.vx_max),
                max_vy=float(self.vel_command.vy_max),
                max_wz=float(self.vel_command.vyaw_max),
                deadzone=float(getattr(self.cfg.mujoco, "velocity_command_deadzone", 0.08)),
                invert_vx=bool(getattr(self.cfg.mujoco, "velocity_command_invert_vx", False)),
                invert_vy=bool(getattr(self.cfg.mujoco, "velocity_command_invert_vy", False)),
                invert_wz=bool(getattr(self.cfg.mujoco, "velocity_command_invert_wz", False)),
                left_y_mode=str(getattr(self.cfg.mujoco, "velocity_command_left_y_mode", "bipolar")),
            )
        )
        self._control_input = ControlInputManager(backend=source, mapper=mapper)
        print(f"\nUsing control input source: {self._control_input.source}")

    def update_state(self) -> None:

        dof_pos = self.mj_data.qpos[self._body_joint_qposadr].astype(np.float32)
        dof_vel = self.mj_data.qvel[self._body_joint_dofadr].astype(np.float32)
        dof_torque = self.mj_data.qfrc_actuator[self._body_joint_dofadr].astype(np.float32)

        base_pos_w = self.mj_data.qpos.astype(np.float32)[:3]
        base_quat = self.mj_data.qpos.astype(np.float32)[3:7]
        base_lin_vel_b = self.mj_data.qvel.astype(np.float32)[:3]
        base_ang_vel_b = self.mj_data.qvel.astype(np.float32)[3:6]

        self.robot.data.joint_pos = torch.from_numpy(
            dof_pos).to(self.robot.data.device)
        self.robot.data.joint_vel = torch.from_numpy(
            dof_vel).to(self.robot.data.device)
        self.robot.data.feedback_torque = torch.from_numpy(
            dof_torque).to(self.robot.data.device)
        self.robot.data.root_pos_w = torch.from_numpy(
            base_pos_w).to(self.robot.data.device)
        self.robot.data.root_quat_w = torch.from_numpy(
            base_quat).to(self.robot.data.device)
        self.robot.data.root_lin_vel_b = torch.from_numpy(
            base_lin_vel_b).to(self.robot.data.device)
        self.robot.data.root_ang_vel_b = torch.from_numpy(
            base_ang_vel_b).to(self.robot.data.device)
        import time
        self.state_recv_id = getattr(self, "state_recv_id", 0) + 1
        self.state_recv_ns = time.monotonic_ns()

    def log_states(self, dof_targets: np.ndarray) -> None:

        path = self.cfg.mujoco.log_states
        if path is None:
            return
        recorder = getattr(self, '_state_recorder', None)
        if recorder is None:
            from booster_deploy.utils.mujoco_state_log import MujocoStateRecorder

            joints = int(np.asarray(self._body_joint_qposadr).size)
            recorder = MujocoStateRecorder(
                path,
                {
                    'root_pos_w': 3, 'root_quat_w': 4,
                    'root_lin_vel_b': 3, 'root_ang_vel_b': 3,
                    'joint_pos': joints, 'joint_vel': joints,
                    'joint_torque': joints, 'dof_targets': joints,
                },
                flush_every=100,
            )
            self._state_recorder = recorder
        # One row copy per field: the npz serialisation happens on the recorder's
        # writer thread, so the 50 Hz control loop is no longer stalled every 100
        # steps by a full-history np.stack + np.savez.
        recorder.append({
            'root_pos_w': self.mj_data.qpos[:3],
            'root_quat_w': self.mj_data.qpos[3:7],
            'root_lin_vel_b': self.mj_data.qvel[:3],
            'root_ang_vel_b': self.mj_data.qvel[3:6],
            'joint_pos': self.mj_data.qpos[self._body_joint_qposadr],
            'joint_vel': self.mj_data.qvel[self._body_joint_dofadr],
            'joint_torque': self.mj_data.qfrc_actuator[self._body_joint_dofadr],
            'dof_targets': np.asarray(dof_targets).reshape(-1),
        })

    def stop(self) -> None:

        recorder = getattr(self, '_state_recorder', None)
        if recorder is not None:
            self._state_recorder = None
            recorder.close()
        super().stop()

    def _set_joint_qcmd(self, dof_targets: np.ndarray | torch.Tensor) -> np.ndarray:

        if isinstance(dof_targets, torch.Tensor):
            target = dof_targets.detach().cpu().numpy()
        else:
            target = np.asarray(dof_targets)
        target = np.asarray(target, dtype=np.float32).reshape(-1)
        if target.shape[0] != self.robot.num_joints:
            raise ValueError(
                f"dof_targets must have {self.robot.num_joints} body joints, got {target.shape[0]}"
            )
        target = np.clip(
            target,
            self.robot.joint_pos_limit_low.numpy(),
            self.robot.joint_pos_limit_high.numpy(),
        )
        self._last_joint_qcmd = target.copy()
        from booster_deploy.telemetry.encoder import target as record_target
        record_target(self,target,backend='mujoco',semantics='pre_physics_target_not_execution_ack')
        return target

    def ctrl_step(self, dof_targets: torch.Tensor):

        dof_targets = self._set_joint_qcmd(dof_targets)
        self.log_states(dof_targets)

        kp = self.robot.joint_stiffness.numpy()
        kd = self.robot.joint_damping.numpy()
        ctrl_limit = self.robot.effort_limit.numpy()
        for i in range(self.decimation):
            dof_pos = self.mj_data.qpos[self._body_joint_qposadr].astype(np.float32)
            dof_vel = self.mj_data.qvel[self._body_joint_dofadr].astype(np.float32)
            body_ctrl = np.clip(
                kp * (dof_targets - dof_pos) - kd * dof_vel,
                -ctrl_limit,
                ctrl_limit,
            )
            self.mj_data.ctrl[self._body_actuator_ids] = body_ctrl
            self._apply_hand_retarget_frame_to_ctrl()
            if self._mujoco_link_debug is not None:
                self._mujoco_link_debug.apply_configured_force(self.mj_data)
            mujoco.mj_step(self.mj_model, self.mj_data)

        self._record_generalized_csv_logs()



    def create_viewer(
        self,
        *,
        model: mujoco.MjModel | None = None,
        data: mujoco.MjData | None = None,
        key_callback=None,
    ):
        backend = self.cfg.mujoco.viewer_backend
        if not is_viewer_backend_available(backend):
            print(f"Warning: Viewer backend '{backend}' is not available.")
            print("Falling back to 'mujoco' backend.")
            backend = "mujoco"

        viewer_kwargs = {}
        if backend == "mujoco":
            viewer_kwargs = {
                "show_left_ui": False,
                "show_right_ui": False,
            }
        elif backend == "viser":
            viewer_kwargs = {
                "host": self.cfg.mujoco.viser_host,
                "port": self.cfg.mujoco.viser_port,
                "show_joint_axes": self.cfg.mujoco.viser_show_joint_axes,
                "camera_azimuth": self.cfg.mujoco.viser_camera_azimuth,
                "camera_elevation": self.cfg.mujoco.viser_camera_elevation,
                "camera_distance": self.cfg.mujoco.viser_camera_distance,
                "camera_follow": self.cfg.mujoco.viser_camera_follow,
            }

        return create_viewer(
            backend=backend,
            model=self.mj_model if model is None else model,
            data=self.mj_data if data is None else data,
            key_callback=self._handle_viewer_key if key_callback is None else key_callback,
            **viewer_kwargs,
        )

    def _viewer_snapshot(self) -> ViewerSnapshot:

        human_frame = None
        skeleton_connections = None
        if self.cfg.mujoco.visualize_realtime_human:
            motion = getattr(self.policy, "motion", None)
            latest_human_frame = getattr(motion, "latest_human_frame", None)
            if callable(latest_human_frame):
                human_frame = latest_human_frame()
            get_skeleton_connections = getattr(motion, "skeleton_connections", None)
            if callable(get_skeleton_connections):
                skeleton_connections = get_skeleton_connections()
            if skeleton_connections is None:
                skeleton_connections = pico_skeleton_connections()
            else:
                skeleton_connections = tuple(skeleton_connections)

        velocity_command_xy = None
        if (
            self.cfg.mujoco.visualize_command_velocity_arrow
            and self.vel_command is not None
        ):
            velocity_command_xy = (
                float(self.vel_command.lin_vel_x),
                float(self.vel_command.lin_vel_y),
            )

        return ViewerSnapshot.create(
            qpos=self.mj_data.qpos,
            reference_qpos=self._reference_qpos,
            human_frame=human_frame,
            skeleton_connections=skeleton_connections,
            velocity_command_xy=velocity_command_xy,
        )

    def _create_async_viewer(
        self,
    ) -> AsyncViewerRuntime | AsyncViserViewerRuntime:

        # Dashboard mode owns rendering in the operator's browser.  This adapter
        # preserves the existing latest-only ViewerSnapshot boundary while avoiding
        # a third Python renderer / WebGL server on the onboard computer.
        web_endpoint = os.getenv("BOOSTER_WEB_VIS_ENDPOINT", "").strip()
        if web_endpoint:
            from booster_deploy.visualization.web.runtime import WebSnapshotRuntime

            return WebSnapshotRuntime(model=self.mj_model, endpoint=web_endpoint)

        if self.cfg.mujoco.viewer_backend == "viser" and is_viewer_backend_available(
            "viser"
        ):
            return AsyncViserViewerRuntime(
                model=self.mj_model,
                render_hz=float(self.cfg.mujoco.viewer_render_hz),
                host=self.cfg.mujoco.viser_host,
                port=int(self.cfg.mujoco.viser_port),
                show_joint_axes=bool(self.cfg.mujoco.viser_show_joint_axes),
                visualize_reference_ghost=bool(
                    self.cfg.mujoco.visualize_reference_ghost
                ),
                visualize_realtime_human=bool(
                    self.cfg.mujoco.visualize_realtime_human
                ),
                visualize_command_velocity_arrow=bool(
                    self.cfg.mujoco.visualize_command_velocity_arrow
                ),
                ghost_rgba=self._ghost_rgba,
                command_velocity_arrow_z=float(
                    self.cfg.mujoco.command_velocity_arrow_z
                ),
                command_velocity_arrow_scale=float(
                    self.cfg.mujoco.command_velocity_arrow_scale
                ),
            )

        def viewer_factory(model, data, key_callback):
            return self.create_viewer(
                model=model,
                data=data,
                key_callback=key_callback,
            )

        return AsyncViewerRuntime(
            model=self.mj_model,
            viewer_factory=viewer_factory,
            render_hz=float(self.cfg.mujoco.viewer_render_hz),
            visualize_reference_ghost=bool(
                self.cfg.mujoco.visualize_reference_ghost
            ),
            visualize_realtime_human=bool(
                self.cfg.mujoco.visualize_realtime_human
            ),
            visualize_command_velocity_arrow=bool(
                self.cfg.mujoco.visualize_command_velocity_arrow
            ),
            ghost_rgba=self._ghost_rgba,
            command_velocity_arrow_z=float(
                self.cfg.mujoco.command_velocity_arrow_z
            ),
            command_velocity_arrow_scale=float(
                self.cfg.mujoco.command_velocity_arrow_scale
            ),
        )

    def _handle_viewer_key(self, keycode: int) -> None:

        motion = getattr(self.policy, "motion", None)
        handle_key = getattr(motion, "handle_key", None)
        if callable(handle_key) and handle_key(keycode):
            return
        key = chr(keycode).lower()
        motion_is_realtime = bool(getattr(motion, "is_realtime", False))
        if key != "y" and not (keycode == ord(" ") and not motion_is_realtime):
            return
        toggle = getattr(self.policy, "toggle_reference_pause", None)
        if callable(toggle):
            paused = toggle()
            # Sim sessions pause from the viewer keys, so mirror the state for the
            # offline publisher (otherwise the clip keeps running to its end).
            from booster_deploy.utils.replay_control import write_state

            write_state(bool(paused))

    def _run_control_loop(
        self,
        viewer_runtime: AsyncViewerRuntime | AsyncViserViewerRuntime,
    ) -> None:

        self._init_velocity_command_source()
        self.update_state()
        viewer_runtime.publish(self._viewer_snapshot())
        self.start()

        motion = getattr(self.policy, "motion", None)
        total = (
            0
            if bool(getattr(motion, "is_realtime", False))
            else getattr(motion, "time_step_total", 0)
        )
        try:
            total = int(total)
        except Exception:
            total = 0
        motion_path = getattr(getattr(self.policy, "cfg", None), "motion_path", "motion")
        label = os.path.splitext(os.path.basename(str(motion_path)))[0]
        t0 = perf_counter()
        last_progress_step: int | None = None
        if total > 0 and self.vel_command is None:
            initial_step = int(getattr(self.policy, "current_frame", 0))
            print_progress(label, initial_step, total, t0)
            last_progress_step = initial_step

        policy_deadline = PeriodicDeadline(float(self.cfg.policy_dt))
        while viewer_runtime.is_running() and self.is_running:
            policy_deadline.wait()
            viewer_runtime.drain_key_events(self._handle_viewer_key)
            if self.vel_command is not None:
                self.update_vel_command()
            self.update_state()
            dof_targets = self.policy_step()
            self.mj_data.xfrc_applied[:] = viewer_runtime.latest_xfrc_applied()
            self.ctrl_step(dof_targets)
            viewer_runtime.publish(self._viewer_snapshot())

            if total > 0 and self.vel_command is None:
                step = getattr(self.policy, "current_frame", self._step_count)
                try:
                    step = int(step)
                except Exception:
                    step = int(self._step_count)
                if step != last_progress_step and (step % 10 == 0 or step >= total):
                    print_progress(label, step, total, t0)
                    last_progress_step = step

            if self._mujoco_link_debug is not None:
                self._mujoco_link_debug.update_display(self.mj_data)
            policy_deadline.complete()

        if total > 0 and self.vel_command is None:
            print()

    def run(self):

        viewer_runtime = self._create_async_viewer()
        control_errors: list[BaseException] = []

        def control_target() -> None:
            try:
                self._run_control_loop(viewer_runtime)
            except BaseException as exc:
                control_errors.append(exc)
            finally:
                viewer_runtime.close()

        control_thread = threading.Thread(
            target=control_target,
            name="mujoco-control-loop",
            daemon=False,
        )
        try:
            viewer_runtime.run(on_started=control_thread.start)
        finally:
            viewer_runtime.close()
            if control_thread.ident is not None:
                control_thread.join()
            self.stop()
            if self._control_input is not None:
                self._control_input.close()
                self._control_input = None
            if self._remote_control is not None:
                self._remote_control.close()
                self._remote_control = None
            if self._mujoco_link_debug is not None:
                self._mujoco_link_debug.close(self.mj_data)
                self._mujoco_link_debug = None
            self._close_generalized_csv_loggers()

        if control_errors:
            raise control_errors[0]
