import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))


def main():
    parser = argparse.ArgumentParser(description="Deploy the T2 M0 retarget policy.")
    parser.add_argument("--telemetry-config", help="Telemetry YAML configuration")
    parser.add_argument("--net", default="127.0.0.1", help="Network interface for SDK communication.")
    backend = parser.add_mutually_exclusive_group()
    backend.add_argument("--mujoco", action="store_true", help="Deploy in MuJoCo simulation.")
    backend.add_argument("--real", action="store_true", help="Deploy on the real robot (default).")
    parser.add_argument("--device", default="cpu", help="Inference device (e.g., cpu or cuda).")
    parser.add_argument("--viser", action="store_true", help="Use the Viser web viewer for MuJoCo simulation.")
    parser.add_argument("--viser-port", type=int, default=8080, help="Viser web viewer port.")
    parser.add_argument("--viser-host", default="0.0.0.0", help="Viser web viewer bind address.")
    parser.add_argument("--visualize-reference-ghost", action=argparse.BooleanOptionalAction,
                        help="Show the retargeted reference robot ghost in the MuJoCo viewer.")
    parser.add_argument("--visualize-realtime-human", action=argparse.BooleanOptionalAction,
                        help="Show the raw realtime HumanFrame skeleton in the MuJoCo viewer.")
    args = parser.parse_args()

    from tasks.gmc import T2GmcRtControllerCfg
    from booster_deploy.telemetry.run_config import record_execution_policy
    from booster_deploy.telemetry.runtime import initialize
    from booster_deploy.utils.mjcf_path import t2_mjcf_path
    from booster_deploy.utils.project_path import resolve_project_path

    task_cfg = T2GmcRtControllerCfg()
    task_cfg.robot.mjcf_path = str(t2_mjcf_path())
    task_cfg.policy.device = args.device
    if args.visualize_reference_ghost is not None:
        task_cfg.mujoco.visualize_reference_ghost = args.visualize_reference_ghost
    if args.visualize_realtime_human is not None:
        task_cfg.mujoco.visualize_realtime_human = args.visualize_realtime_human
    if args.viser:
        task_cfg.mujoco.viewer_backend = "viser"
        task_cfg.mujoco.viser_port = args.viser_port
        task_cfg.mujoco.viser_host = args.viser_host

    os.environ.update(
        BOOSTER_TELEMETRY_TASK="t2_gmc_rt",
        BOOSTER_TELEMETRY_EXECUTION="sim" if args.mujoco else "real",
        BOOSTER_TELEMETRY_TOPOLOGY="retarget",
        BOOSTER_TELEMETRY_INPUT_PATH="retarget",
        BOOSTER_TELEMETRY_MJCF=str(resolve_project_path(task_cfg.robot.mjcf_path)),
    )
    initialize(config=args.telemetry_config, metadata={
        "onnx_model": str(resolve_project_path(task_cfg.policy.onnx_model_path)),
        "model_source": "task_default",
    })
    record_execution_policy(os.environ["BOOSTER_TELEMETRY_DIR"], "t2_gmc_rt", task_cfg,
                            execution="sim" if args.mujoco else "real")

    if args.mujoco:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode = "offline" if os.getenv("BOOSTER_TELEMETRY_REPLAY") == "offline" else "online"
        hand_suffix = "_hand" if task_cfg.robot.mjcf_path.endswith("_revo2hand.xml") else ""
        log_prefix = f"logs/mujoco_state/teleop_{timestamp}_{mode}_retarget_sim{hand_suffix}"
        PROJECT_DIR.joinpath("logs", "mujoco_state").mkdir(parents=True, exist_ok=True)
        task_cfg.mujoco.log_states = log_prefix
        print(f"[deploy] MuJoCo state log: {log_prefix}.npz", flush=True)

    try:
        if args.mujoco:
            print("[low_state] MuJoCo: no real /low_state recording (simulation logs remain separate).", flush=True)
            from booster_deploy.controllers.mujoco_controller import MujocoController
            MujocoController(task_cfg).run()
        else:
            try:
                from booster_robotics_sdk_python import ChannelFactory
                ChannelFactory.Instance().Init(0, args.net)
                print(f"[deploy] booster_robotics_sdk_python initialized (net={args.net}).")
            except ImportError:
                print(
                    "[deploy] booster_robotics_sdk_python not installed; "
                    "going through ROS2 (rclpy + booster_interface) only. "
                    "Make sure booster_ros2 is running and /low_state, "
                    "/joint_ctrl, booster_rpc_service are reachable.",
                    flush=True,
                )

            from booster_deploy.controllers.booster_robot_controller import BoosterRobotPortal
            with BoosterRobotPortal(task_cfg, use_sim_time=False) as portal:
                portal.run()
    except KeyboardInterrupt:
        print("\n Interrupted by user. Shutting down gracefully...")


if __name__ == "__main__":
    main()
