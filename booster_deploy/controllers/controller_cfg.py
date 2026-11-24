
from typing import Callable, List, Optional
from dataclasses import MISSING
import torch

from ..utils.isaaclab.configclass import configclass


@configclass
class PrepareStateCfg:

    stiffness: List[float] = MISSING
    damping: List[float] = MISSING
    joint_pos: List[float] = MISSING


@configclass
class MujocoControllerCfg:

    init_pos: List[float] = [0.0, 0.0, 0.6]
    init_quat: List[float] = [1.0, 0.0, 0.0, 0.0]
    decimation: int = 10

    physics_dt: float = None
    log_states: Optional[str] = None
    visualize_reference_ghost: bool = False
    visualize_realtime_human: bool = False
    visualize_command_velocity_arrow: bool = False
    print_viewer_applied_force: bool = False
    viewer_force_print_threshold: float = 1e-4
    viewer_force_print_torque: bool = False
    force_apply_link: Optional[str] = None
    apply_force: Optional[List[float]] = None
    position_print_link: List[str] = []
    mujoco_debug_panel_refresh_hz: float = 10.0
    mujoco_debug_panel_enabled: bool = True
    command_velocity_arrow_z: float = 0.55
    command_velocity_arrow_scale: float = 0.55
    ghost_rgba: List[float] = [0.2, 0.8, 0.2, 0.25]

    disable_self_collision: bool = False

    hand_target_max_velocity: float = 5.0
    hand_target_filter_alpha: float = 1.0

    log_generalized_qpos_csv: Optional[str] = None
    log_generalized_qvel_csv: Optional[str] = None
    log_generalized_qtor_csv: Optional[str] = None
    log_generalized_qcmd_csv: Optional[str] = None
    log_generalized_qref_csv: Optional[str] = None
    log_csv_flush_every: int = 50

    init_joint_pos: Optional[List[float]] = None

    viewer_backend: str = "mujoco"
    viewer_render_hz: float = 50.0
    viser_host: str = "0.0.0.0"
    viser_port: int = 8080
    viser_show_joint_axes: bool = False
    # Operator view: 180° azimuth is the project's "robot front" convention, and
    # follow locks the camera to the root so a stray drag cannot lose the view.
    viser_camera_azimuth: Optional[float] = 180.0
    viser_camera_elevation: Optional[float] = -10.0
    viser_camera_distance: Optional[float] = 2.4
    viser_camera_follow: bool = True

    velocity_command_source: str = "stdin"
    velocity_command_deadzone: float = 0.08
    velocity_command_invert_vx: bool = False
    velocity_command_invert_vy: bool = False
    velocity_command_invert_wz: bool = False
    velocity_command_left_y_mode: str = "bipolar"


@configclass
class BoosterRobotControllerCfg:

    low_state_dt: float = 0.002
    metrics_max_events: int = 2000
    hand_control_mode: str = "closed"


@configclass
class UnitreeG1RealControllerCfg:

    network_interface: str = "eth0"
    publish_hz: int = 200
    enter_debug_mode: bool = True
    pvt_hold_leg_kp_scale: float = 5.0
    pvt_hold_upper_kp_scale: float = 3.0


@configclass
class RobotCfg:

    name: str = MISSING

    joint_names: list[str] = MISSING
    body_names: list[str] = MISSING

    sim_joint_names: list[str] = MISSING
    sim_body_names: list[str] = MISSING

    joint_stiffness: List[float] = MISSING
    joint_damping: List[float] = MISSING
    action_scale_factor: float = 0.25

    default_joint_pos: List[float] = MISSING
    effort_limit: List[float] = MISSING
    joint_pos_limit_low: List[float] = None
    joint_pos_limit_high: List[float] = None

    mjcf_path: str = MISSING

    prepare_state: PrepareStateCfg = MISSING

    def __post_init__(self):

        assert (
            len(self.joint_names)
            == len(self.joint_stiffness)
            == len(self.joint_damping)
            == len(self.default_joint_pos)
            == len(self.effort_limit)
        )
        if self.joint_pos_limit_low is None:
            self.joint_pos_limit_low = [-float("inf")] * len(self.joint_names)
        if self.joint_pos_limit_high is None:
            self.joint_pos_limit_high = [float("inf")] * len(self.joint_names)
        assert len(self.joint_pos_limit_low) == len(self.joint_names)
        assert len(self.joint_pos_limit_high) == len(self.joint_names)


@configclass
class VelocityCommandCfg:

    vx_max: float = 1.0
    vy_max: float = 1.0
    vyaw_max: float = 1.0


@configclass
class PolicyCfg:

    constructor: Callable = MISSING
    checkpoint_path: str = MISSING
    enable_safety_fallback: bool = False
    device: str | torch.device = "cpu"
    motion_start_frame: int = 0
    input_path: str = ""


@configclass
class EvaluatorCfg:

    constructor: Callable = MISSING

    render: bool = True

    def update(self, controller: 'ControllerCfg'):

        pass


@configclass
class ControllerCfg:

    policy_dt: float = 0.02
    robot: RobotCfg = MISSING
    vel_command: Optional[VelocityCommandCfg] = None
    policy: PolicyCfg = MISSING

    mujoco: MujocoControllerCfg = MujocoControllerCfg()
    booster: BoosterRobotControllerCfg = BoosterRobotControllerCfg()
    unitree_g1_real: UnitreeG1RealControllerCfg = UnitreeG1RealControllerCfg()
    evaluator: Optional[EvaluatorCfg] = None

    def __post_init__(self):

        self.mujoco.physics_dt = self.policy_dt / self.mujoco.decimation
