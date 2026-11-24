
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RealtimeRetargetZmqConfig:

    input_host: str = "127.0.0.1"
    input_port: int = 5555
    input_retarget_topic: str = "pico_rt.retarget_input"
    input_control_topic: str = "pico_rt.control"
    output_bind: str = "127.0.0.1"
    output_port: int = 5556
    output_reference_topic: str = "retarget_rt.reference_50hz"
    output_status_topic: str = "retarget_rt.status"
    reset_bind: str = "0.0.0.0"
    reset_port: int = 5559
    reset_topic: str = "retarget_rt.reset"


@dataclass(frozen=True)
class RealtimeRetargetConfig:

    source: str = "pico_rt"
    target_fps: float = 50.0
    robot_name: str = "g1"
    robot_xml_path: str | None = None
    human_format: str | None = None
    ik_config_path: str | None = None
    actual_human_height: float | None = None
    solver: str = "daqp"
    ground_detection_mode: str = "mesh"
    use_velocity_limit: bool = False
    qpos_filter: str = "ema"
    qpos_filter_alpha: float = 0.85
    joint_vel_filter: str = "ema"
    joint_vel_filter_alpha: float = 0.75
    max_joint_speed: float | None = 20.0
    delay_frames: int | None = None
    future_frames: int | None = None
    command_frame_interval: int | None = None
    history_slack_frames: int = 64
    input_timeout_s: float = 30.0
    include_human_frame: bool = False
    include_hand_frame: bool = False
    hand_sides: tuple[str, ...] = ("left", "right")
    hand_left_config_path: str | None = None
    hand_right_config_path: str | None = None
    hand_somehand_repo: str | None = None
    hand_tip_mode: str = "extrapolate"
    hand_skip_missing_hands: bool = True
    max_retarget_gap_ns: int = 500_000_000
    startup_min_samples: int = 20
    zmq: RealtimeRetargetZmqConfig = field(default_factory=RealtimeRetargetZmqConfig)

    @property
    def sample_dt_ns(self) -> int:

        return int(round(1_000_000_000 / self.target_fps))


def default_realtime_config_from_policy(policy_cfg, *, robot_name: str = "g1") -> RealtimeRetargetConfig:

    future_frames = int(getattr(policy_cfg, "command_num_future_frames", 10))
    command_frame_interval = int(getattr(policy_cfg, "command_frame_interval", 0))
    delay_frames = future_frames
    startup_min_samples = delay_frames + (future_frames - 1) * (command_frame_interval + 1) + 1
    return RealtimeRetargetConfig(
        robot_name=robot_name,
        target_fps=float(getattr(policy_cfg, "realtime_retarget_fps", 50.0)),
        human_format=None,
        actual_human_height=getattr(policy_cfg, "realtime_retarget_actual_human_height", None),
        solver=str(getattr(policy_cfg, "realtime_retarget_solver", "daqp")),
        ground_detection_mode=str(getattr(policy_cfg, "realtime_retarget_ground_detection_mode", "mesh")),
        delay_frames=delay_frames,
        future_frames=future_frames,
        command_frame_interval=command_frame_interval,
        startup_min_samples=startup_min_samples,
    )
