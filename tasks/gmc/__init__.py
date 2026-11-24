import os

from booster_deploy.utils.isaaclab.configclass import configclass
from booster_deploy.utils.actuator_profiles import apply_t2_actuator_profile
from booster_deploy.utils.registry import register_task

from booster_deploy.controllers.controller_cfg import (
    ControllerCfg,
    MujocoControllerCfg,
)
from booster_deploy.robots.booster import T2_32DOF_CFG

from .gmc_rt import GmcRtPolicy, GmcRtPolicyCfg


@configclass
class T2GmcRtControllerCfg(ControllerCfg):

    robot = T2_32DOF_CFG
    enable_velocity_commands = False
    mujoco = MujocoControllerCfg(
        init_pos=[0.0, 0.0, 1.0],
        visualize_reference_ghost=False,
        disable_self_collision=True,
        decimation=10,
    )
    policy: GmcRtPolicyCfg = GmcRtPolicyCfg(
        traj_anchor_name="trunk",
        obs_anchor_name="trunk",
        num_joints=31,
        num_actions=31,
    )

    actuator_profile: str = "joint_config_release_1010"

    def __post_init__(self):

        super().__post_init__()
        apply_t2_actuator_profile(self.robot, self.actuator_profile)
        self.policy.rt_reference_host = os.getenv("BOOSTER_RT_REFERENCE_HOST", "127.0.0.1")
        self.policy.rt_reference_port = int(os.getenv("BOOSTER_RT_REFERENCE_PORT", "5556"))
        self.policy.rt_reference_topic = os.getenv("BOOSTER_RT_REFERENCE_TOPIC", "retarget_rt.reference_50hz")
        self.policy.rt_reference_transition_s = float(os.getenv("BOOSTER_RT_REFERENCE_TRANSITION_S", "1.5"))
        self.policy.constructor = GmcRtPolicy
        self.policy.history_length = 10
        self.policy.command_history_length = 1
        self.policy.state_history_length = 10
        self.policy.command_num_future_frames = 10
        self.policy.command_frame_interval = 0
        self.policy.action_offset_source = "command"
        self.policy.onnx_model_path = "booster_deploy/assets/models/model_gmc_release_1010.onnx"


register_task("t2_gmc_rt", T2GmcRtControllerCfg())
