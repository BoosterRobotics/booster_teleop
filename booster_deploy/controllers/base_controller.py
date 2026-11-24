
from __future__ import annotations
from abc import abstractmethod
import inspect
import os
import torch

from .controller_cfg import (
    ControllerCfg, PolicyCfg, RobotCfg, VelocityCommandCfg
)


class RobotData:

    joint_pos: torch.Tensor
    joint_vel: torch.Tensor
    feedback_torque: torch.Tensor
    root_pos_w: torch.Tensor
    root_quat_w: torch.Tensor
    root_lin_vel_b: torch.Tensor
    root_ang_vel_b: torch.Tensor

    def __init__(self, cfg: RobotCfg) -> None:

        self.cfg = cfg
        num_joints = len(self.cfg.joint_names)
        self.real2sim_joint_indexes = [cfg.joint_names.index(name) for name in cfg.sim_joint_names]
        self.sim2real_joint_indexes = [cfg.sim_joint_names.index(name) for name in cfg.joint_names]
        self.device = "cpu"

        self.joint_pos: torch.Tensor = torch.zeros(num_joints, dtype=torch.float32)
        self.joint_vel: torch.Tensor = torch.zeros(num_joints, dtype=torch.float32)
        self.feedback_torque: torch.Tensor = torch.zeros(num_joints, dtype=torch.float32)
        self.root_lin_vel_b: torch.Tensor = torch.zeros(3, dtype=torch.float32)
        self.root_ang_vel_b: torch.Tensor = torch.zeros(3, dtype=torch.float32)
        self.root_pos_w: torch.Tensor = torch.zeros(3, dtype=torch.float32)
        self.root_quat_w: torch.Tensor = torch.zeros(4, dtype=torch.float32)

    def to(self, device: torch.device | str) -> None:

        self.device = device
        self.joint_pos = self.joint_pos.to(device)
        self.joint_vel = self.joint_vel.to(device)
        self.feedback_torque = self.feedback_torque.to(device)
        self.root_lin_vel_b = self.root_lin_vel_b.to(device)
        self.root_ang_vel_b = self.root_ang_vel_b.to(device)
        self.root_pos_w = self.root_pos_w.to(device)
        self.root_quat_w = self.root_quat_w.to(device)


class BoosterRobot:

    cfg: RobotCfg
    data: RobotData
    joint_stiffness: torch.Tensor
    joint_damping: torch.Tensor
    action_scale: torch.Tensor
    default_joint_pos: torch.Tensor
    joint_pos_limit_low: torch.Tensor
    joint_pos_limit_high: torch.Tensor

    def __init__(self, cfg: RobotCfg) -> None:

        self.cfg = cfg
        self.data = RobotData(cfg)

        self.joint_stiffness = torch.tensor(cfg.joint_stiffness, dtype=torch.float32)

        self.joint_damping = torch.tensor(cfg.joint_damping, dtype=torch.float32)

        self.default_joint_pos = torch.tensor(cfg.default_joint_pos, dtype=torch.float32)
        self.effort_limit = torch.tensor(cfg.effort_limit, dtype=torch.float32)
        self.action_scale = (
            float(cfg.action_scale_factor) * self.effort_limit / self.joint_stiffness
        )
        self.joint_pos_limit_low = torch.tensor(cfg.joint_pos_limit_low, dtype=torch.float32)
        self.joint_pos_limit_high = torch.tensor(cfg.joint_pos_limit_high, dtype=torch.float32)

    @property
    def num_joints(self) -> int:

        return len(self.cfg.joint_names)

    @property
    def num_bodies(self) -> int:

        return len(self.cfg.body_names)


class Commands:

    pass


class VelocityCommand(Commands):

    lin_vel_x: float
    lin_vel_y: float
    ang_vel_yaw: float

    def __init__(self, cfg: VelocityCommandCfg) -> None:

        self.vx_max = cfg.vx_max
        self.vy_max = cfg.vy_max
        self.vyaw_max = cfg.vyaw_max

        self.lin_vel_x: float = 0.0
        self.lin_vel_y: float = 0.0
        self.ang_vel_yaw: float = 0.0


class Policy:

    def __init__(self, cfg: PolicyCfg, controller: BaseController):

        self.cfg = cfg
        self.controller = controller

        class_module = inspect.getmodule(self.__class__)
        self.task_path = os.path.dirname(class_module.__file__)
        self._offline_reference_paused = False

    def reset_motion_playback(self) -> int:

        self._offline_reference_paused = False
        motion = getattr(self, "motion", None)
        if motion is None or bool(getattr(motion, "is_realtime", False)):
            self.current_frame = 0
            return self.current_frame

        total = int(getattr(motion, "time_step_total", 0))
        start_frame = int(getattr(self.cfg, "motion_start_frame", 0))
        if start_frame < 0:
            raise ValueError(f"motion_start_frame must be >= 0, got {start_frame}")
        if total <= 0:
            raise ValueError("offline motion must contain at least one frame")
        if start_frame >= total:
            raise ValueError(
                f"motion_start_frame {start_frame} is out of range for "
                f"{total} motion frames (valid: 0..{total - 1})"
            )
        self.current_frame = start_frame
        return self.current_frame

    def toggle_reference_pause(self) -> bool:

        motion = getattr(self, "motion", None)
        if motion is None or bool(getattr(motion, "is_realtime", False)):
            return False
        self._offline_reference_paused = not self._offline_reference_paused
        status = "paused" if self._offline_reference_paused else "resumed"
        print(f"Offline reference {status} at frame {int(self.current_frame)}", flush=True)
        return self._offline_reference_paused

    def inference_with_reference_control(self) -> torch.Tensor:

        held_frame = getattr(self, "current_frame", None)
        action = self.inference()
        if self._offline_reference_paused and held_frame is not None:
            self.current_frame = held_frame
        return action

    def reference_finished(self) -> bool:
        """True when a replayed reference clip has run out.

        Only offline replays report this (``BOOSTER_OFFLINE_REPLAY``), and the
        controller turns it into "leave Tracking and go back to Walking". Live
        teleoperation keeps its stale-input hold behaviour instead.
        """

        return False

    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def inference(self) -> torch.Tensor:
        pass


class BaseController:

    cfg: ControllerCfg
    robot: BoosterRobot
    vel_command: VelocityCommand
    policy: Policy

    def __init__(self, cfg: ControllerCfg) -> None:

        self.cfg = cfg
        self._step_count: int = 0
        self._elapsed_s: float = 0.0
        self.is_running: bool = False
        self.robot = BoosterRobot(cfg.robot)
        self.vel_command = None
        if self.cfg.vel_command is not None:
            self.vel_command = VelocityCommand(cfg.vel_command)
        self.policy = self.cfg.policy.constructor(self.cfg.policy, self)

    def start(self):
        self._step_count = 0
        self._elapsed_s = 0.0
        self.is_running = True
        self.policy.reset()

    def policy_step(self) -> torch.Tensor:
        if not self.is_running:
            raise RuntimeError("Environment.step() called before start().")

        self._step_count += 1
        self._elapsed_s = self._step_count * self.cfg.policy_dt

        return self.policy.inference_with_reference_control()

    def stop(self) -> None:
        self.is_running = False
        close = getattr(self.policy, "close", None)
        if callable(close):
            close()

    @abstractmethod
    def ctrl_step(self, dof_targets: torch.Tensor) -> None:
        pass

    @abstractmethod
    def update_state(self) -> None:
        pass

    @abstractmethod
    def run(self) -> None:
        pass
