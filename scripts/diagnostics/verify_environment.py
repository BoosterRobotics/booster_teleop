import argparse
import importlib
from pathlib import Path
import platform
import sys

import numpy as np
import mujoco
import onnxruntime as ort
import torch
from qpsolvers import solve_qp

parser = argparse.ArgumentParser()
parser.add_argument("role", choices=("host", "onboard"))
args = parser.parse_args()
root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root))

for name in ("scipy", "evdev", "zmq", "msgpack", "mink", "qpsolvers", "daqp", "yaml", "rich", "viser", "trimesh"):
    importlib.import_module(name)
    print(f"PASS import {name}")

from tasks.gmc import T2GmcRtControllerCfg
from booster_deploy.utils.project_path import resolve_project_path
from booster_deploy.utils.registry import list_tasks
from booster_deploy.retargeting.gmr.params import ROBOT_XML_DICT
from booster_deploy.hand_retargeting.somehand_adapter import Revo2SomehandRetargeter
from booster_deploy.hand_retargeting.real_hand_mapping import load_real_hand_command_mapping

assert set(list_tasks()) == {"t2_gmc_rt"}
cfg = T2GmcRtControllerCfg()
assert cfg.policy.num_actions == cfg.policy.num_joints == 31
assert len(cfg.robot.joint_names) == len(cfg.robot.default_joint_pos) == 31
assert cfg.policy.action_offset_source == "command"
print("PASS fixed M0 retarget task registration and 31D robot config")
model = mujoco.MjModel.from_xml_path(str(ROBOT_XML_DICT["t2"]))
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
assert np.isfinite(data.qpos).all()
for side in ("left", "right"):
    Revo2SomehandRetargeter(side)
load_real_hand_command_mapping()
assert torch.equal(torch.tensor([1.0]) + 1, torch.tensor([2.0]))
np.testing.assert_allclose(solve_qp(np.eye(2), np.zeros(2), solver="daqp"), np.zeros(2))

from booster_deploy.retargeting.gmr.motion_retarget import GeneralMotionRetargeting
for source in ("bvh", "xrobot"):
    GeneralMotionRetargeting(source, "t2", actual_human_height=1.7, verbose=False, ground_detection_mode="off")

model_path = resolve_project_path(cfg.policy.onnx_model_path)
assert model_path.is_file(), f"Configured M0 ONNX model missing: {model_path}"
policy_session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
assert len(policy_session.get_inputs()) == 2, "M0 policy requires motion command and robot-state inputs"
inputs = {
    item.name: np.zeros([dim if isinstance(dim, int) else 1 for dim in item.shape], dtype=np.float32)
    for item in policy_session.get_inputs()
}
outputs = policy_session.run(None, inputs)
assert outputs and outputs[0].shape[-1] == cfg.policy.num_actions
assert all(np.isfinite(output).all() for output in outputs), model_path
print(f"PASS M0 ONNX CPU inference: {model_path.name}")

import xrobotoolkit_sdk as xrt
required_pico_api = (
    "init", "close", "register_frame_callback", "clear_frame_callback",
    "is_body_data_available", "get_body_joints_pose", "get_A_button", "get_B_button", "get_X_button", "get_Y_button",
)
missing_pico_api = [name for name in required_pico_api if not callable(getattr(xrt, name, None))]
assert not missing_pico_api, f"Pico SDK is missing required APIs: {missing_pico_api}"
print(f"PASS import xrobotoolkit_sdk and Pico callback/polling APIs: {xrt.__file__}")

if args.role == "onboard":
    for name in ("rclpy", "booster_interface.msg", "sensor_msgs.msg", "booster_robotics_sdk_python"):
        importlib.import_module(name)
        print(f"PASS import {name}")
    importlib.import_module("booster_deploy.controllers.booster_robot_controller")

from booster_deploy.inputs.mocap.sdk import load_mocap_api
load_mocap_api()

print(f"PASS {args.role}: {platform.machine()} Python {platform.python_version()} at {sys.executable}")
print("PASS tasks, T2 XML/FK, hand solver construction, Torch and ONNX CPU inference")
print("No robot controllers, SDK sessions, ROS nodes, publishers or services were started.")
