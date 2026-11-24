
import json
import copy
import os

import mink
import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation as R

from .params import ROBOT_XML_DICT, IK_CONFIG_DICT

ROBOT_VELOCITY_LIMITS = {
    "t2": {
        "left_shoulder_pitch_joint": 5,
        "left_shoulder_roll_joint": 5,
        "left_elbow_pitch_joint": 5,
        "left_elbow_yaw_joint": 5,
        "right_shoulder_pitch_joint": 5,
        "right_shoulder_roll_joint": 5,
        "right_elbow_pitch_joint": 5,
        "right_elbow_yaw_joint": 5,
    },
}

class GeneralMotionRetargeting:

    def __init__(
        self,
        src_human: str,
        tgt_robot: str,
        actual_human_height: float = None,
        solver: str = "daqp",
        damping: float = 1e-1,
        verbose: bool = False,
        ik_config=None,
        ground_detection_mode: str = "mesh",
        use_velocity_limit: bool = False,
        model: mj.MjModel | None = None,
    ) -> None:

        self.xml_file = str(ROBOT_XML_DICT[tgt_robot])
        if verbose:
            print("Use robot model: ", self.xml_file)
        self.model = model if model is not None else mj.MjModel.from_xml_path(self.xml_file)

        config_source = ik_config if ik_config is not None else IK_CONFIG_DICT[src_human][tgt_robot]
        if isinstance(config_source, dict):
            ik_config = copy.deepcopy(config_source)
        else:
            with open(config_source) as f:
                ik_config = json.load(f)
        if verbose:
            print("Use IK config: ", config_source)

        ratio = float(ik_config["robot_height"]) / float(actual_human_height)

        for key in ik_config["human_scale_table"].keys():
            ik_config["human_scale_table"][key] = (np.array(ik_config["human_scale_table"][key]) * ratio).tolist()

        self.ik_joint_mappings = ik_config["ik_joint_mappings"]
        self.human_root_name = ik_config["human_root_name"]
        self.robot_root_name = ik_config["robot_root_name"]
        self.use_ik_optimization = ik_config.get("use_ik_optimization", True)
        self.human_scale_table = ik_config["human_scale_table"]
        self.ground = ik_config["ground_height"] * np.array([0, 0, 1])
        self.collision_pairs = ik_config["collision_pairs"]

        self.use_collision_avoidance = ik_config.get("use_collision_avoidance", False)
        self.use_velocity_limit = use_velocity_limit
        self.tgt_robot = tgt_robot

        self.max_iter = 500
        self.solver = solver
        self.damping = damping

        self.human_body_to_task1 = {}
        self.human_body_to_task2 = {}
        self.pos_offsets1 = {}
        self.rot_offsets1 = {}
        self.pos_offsets2 = {}
        self.rot_offsets2 = {}

        self.limits = []

        joint_limit = mink.ConfigurationLimit(self.model)
        self.limits.append(joint_limit)

        if self.use_velocity_limit:
            if tgt_robot in ROBOT_VELOCITY_LIMITS:
                velocity_config = ROBOT_VELOCITY_LIMITS[tgt_robot]
                velocity_limit = mink.VelocityLimit(model=self.model, velocities=velocity_config)
                self.limits.append(velocity_limit)
                if verbose:
                    print(f"Using velocity limits for robot '{tgt_robot}' in retargeting.")
            else:
                if verbose:
                    print(f"[Warning] No velocity limit configuration found for robot '{tgt_robot}', using model defaults.")
                velocity_limit = mink.VelocityLimit(model=self.model, velocities={})
                self.limits.append(velocity_limit)

        collision_limit = mink.CollisionAvoidanceLimit(
            model=self.model,
            geom_pairs=self.collision_pairs,
            gain=0.01,
            minimum_distance_from_collisions=0.015,
            collision_detection_distance=0.15,
            bound_relaxation=0.0,
        )

        if self.use_collision_avoidance:
            self.limits.append(collision_limit)
            if verbose:
                print("Using collision avoidance in retargeting.")
        self.setup_retarget_configuration()
        self.com_task = None
        com_config = ik_config.get("com_bias")
        if com_config is not None:
            self.com_weight = float(os.environ.get("BOOSTER_RETARGET_COM_WEIGHT", com_config["weight"]))
            if not np.isfinite(self.com_weight) or self.com_weight < 0:
                raise ValueError("BOOSTER_RETARGET_COM_WEIGHT must be finite and non-negative")
            if self.com_weight > 0:
                self.com_foot_frames = tuple(com_config["foot_frames"])
                self.com_sole_center = np.asarray(com_config["sole_center_offset"], dtype=float)
                self.com_task = mink.ComTask(cost=[self.com_weight, self.com_weight, 0.0], lm_damping=1)
                self.tasks2.append(self.com_task)
        if ground_detection_mode not in ["mesh", "link", "off"]:
            raise ValueError(f"Invalid ground_detection_mode: {ground_detection_mode}. Must be 'mesh', 'link', or 'off'.")

        self.ground_detection_mode = ground_detection_mode
        self.auto_ground_detection = ground_detection_mode != "off"
        self.is_first_retarget = True
        self.detected_ground_offset = 0.0
        self.ground_qpos_z_offset = 0.0

        if verbose:
            mode_descriptions = {
                "mesh": "using mesh lowest point (most accurate)",
                "link": "using body link center + manual ground_height",
                "off": "disabled (using config ground_height only)",
            }
            print(f"Ground detection mode: {ground_detection_mode} ({mode_descriptions[ground_detection_mode]})")

    def setup_retarget_configuration(self):

        self.configuration = mink.Configuration(self.model)

        self.tasks1 = []
        self.tasks2 = []

        for frame_name, joint_config in self.ik_joint_mappings.items():
            body_name = joint_config["human_body"]
            pos_offset = joint_config["pos_offset"]
            rot_offset = joint_config["rot_offset"]
            frame_type = joint_config["frame_type"]

            stage1_weights = joint_config["stage1_weights"]
            pos_weight1 = stage1_weights["position"]
            rot_weight1 = stage1_weights["orientation"]

            if pos_weight1 != 0 or rot_weight1 != 0:
                task = mink.FrameTask(
                    frame_name=frame_name,
                    frame_type=frame_type,
                    position_cost=pos_weight1,
                    orientation_cost=rot_weight1,
                    lm_damping=1,
                )
                self.human_body_to_task1[body_name] = task
                self.pos_offsets1[body_name] = np.array(pos_offset) - self.ground
                self.rot_offsets1[body_name] = R.from_quat(rot_offset, scalar_first=True)
                self.tasks1.append(task)

            stage2_weights = joint_config["stage2_weights"]
            pos_weight2 = stage2_weights["position"]
            rot_weight2 = stage2_weights["orientation"]

            if pos_weight2 != 0 or rot_weight2 != 0:
                task = mink.FrameTask(
                    frame_name=frame_name,
                    frame_type=frame_type,
                    position_cost=pos_weight2,
                    orientation_cost=rot_weight2,
                    lm_damping=1,
                )
                self.human_body_to_task2[body_name] = task
                self.pos_offsets2[body_name] = np.array(pos_offset) - self.ground
                self.rot_offsets2[body_name] = R.from_quat(rot_offset, scalar_first=True)
                self.tasks2.append(task)

    def update_targets(self, human_data, offset_to_ground=False):

        human_data = self.to_numpy(human_data)
        human_data = self.scale_human_data(human_data, self.human_root_name, self.human_scale_table)
        human_data = self.offset_human_data(human_data, self.pos_offsets1, self.rot_offsets1)
        self.scaled_human_data = human_data

        if self.use_ik_optimization:
            for body_name in self.human_body_to_task1.keys():
                task = self.human_body_to_task1[body_name]
                pos, rot = human_data[body_name]
                task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3(rot), pos))

            for body_name in self.human_body_to_task2.keys():
                task = self.human_body_to_task2[body_name]
                pos, rot = human_data[body_name]
                task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3(rot), pos))

            self._update_com_target(human_data)

    def _update_com_target(self, human_data):
        if self.com_task is None:
            return
        centers = []
        for frame in self.com_foot_frames:
            body = self.ik_joint_mappings[frame]["human_body"]
            pos, quat = human_data[body]
            centers.append(pos + R.from_quat(quat, scalar_first=True).apply(self.com_sole_center))
        centers = np.asarray(centers)
        # Infer stance from desired sole heights, not measured contacts.
        blend = np.clip((np.ptp(centers[:, 2]) - 0.03) / 0.05, 0.0, 1.0)
        weight = self.com_weight * (1.0 - blend * blend * (3.0 - 2.0 * blend))
        self.com_task.set_cost([weight, weight, 0.0])
        self.com_task.set_target(centers.mean(axis=0))
   
    def _compute_tasks_error(self, tasks):

        errors = []
        for task in tasks:
            if task is self.com_task:
                if np.any(task.cost):
                    errors.append(task.compute_error(self.configuration)[:2])
            else:
                errors.append(task.compute_error(self.configuration))
        return np.linalg.norm(np.concatenate(errors)) if errors else 0.0

    def _solve_tasks_iteratively(self, tasks):
        if self.com_task is not None and not np.any(self.com_task.cost):
            tasks = [task for task in tasks if task is not self.com_task]
        if not (self.use_ik_optimization and len(tasks) > 0):
            return
        curr_error = self._compute_tasks_error(tasks)
        dt = self.configuration.model.opt.timestep
        vel = mink.solve_ik(self.configuration, tasks, dt, self.solver, self.damping, limits=self.limits)
        self.configuration.integrate_inplace(vel, dt)
        self._clamp_to_joint_limits()
        next_error = self._compute_tasks_error(tasks)
        num_iter = 0
        while curr_error - next_error > 0.001 and num_iter < self.max_iter:
            curr_error = next_error
            dt = self.configuration.model.opt.timestep
            vel = mink.solve_ik(self.configuration, tasks, dt, self.solver, self.damping, limits=self.limits)
            self.configuration.integrate_inplace(vel, dt)
            self._clamp_to_joint_limits()
            next_error = self._compute_tasks_error(tasks)
            num_iter += 1

    def retarget_sequence(self, human_sequence, offset_to_ground=False):
        if human_sequence is None:
            raise ValueError("human_sequence is None")
        if len(human_sequence) == 0:
            return np.zeros((0,))

        qpos_list = []
        for human_data in human_sequence:
            qpos_list.append(self.retarget(human_data, offset_to_ground))
        return np.stack(qpos_list, axis=0)

    def retarget(self, human_data, offset_to_ground=False):

        if self.is_first_retarget and self.auto_ground_detection:
            print("[Auto Ground Detection] Processing first frame without ground offset...")

            original_ground = self.ground.copy()
            self.ground = np.array([0.0, 0.0, 0.0])

            for body_name in self.pos_offsets1.keys():
                joint_config = None
                for frame_name, config in self.ik_joint_mappings.items():
                    if config["human_body"] == body_name:
                        joint_config = config
                        break
                if joint_config:
                    self.pos_offsets1[body_name] = np.array(joint_config["pos_offset"]) - self.ground

            for body_name in self.pos_offsets2.keys():
                joint_config = None
                for frame_name, config in self.ik_joint_mappings.items():
                    if config["human_body"] == body_name:
                        joint_config = config
                        break
                if joint_config:
                    self.pos_offsets2[body_name] = np.array(joint_config["pos_offset"]) - self.ground

            self._clamp_to_joint_limits()
            self.update_targets(human_data, offset_to_ground)

            self._solve_tasks_iteratively(self.tasks1)
            self._solve_tasks_iteratively(self.tasks2)

            self.detected_ground_offset = self._detect_ground_offset()
            self.ground = original_ground
            self.ground_qpos_z_offset = float(original_ground[2] - self.detected_ground_offset)
            print(f"  Detected ground offset: {self.detected_ground_offset:.4f}")
            print(f"  Config ground_height: {original_ground[2]:.4f}")
            print(f"  IK ground preserved: {self.ground}")
            print(f"  Root qpos z correction: {self.ground_qpos_z_offset:.4f}")

            for body_name in self.pos_offsets1.keys():
                joint_config = None
                for frame_name, config in self.ik_joint_mappings.items():
                    if config["human_body"] == body_name:
                        joint_config = config
                        break
                if joint_config:
                    self.pos_offsets1[body_name] = np.array(joint_config["pos_offset"]) - self.ground

            for body_name in self.pos_offsets2.keys():
                joint_config = None
                for frame_name, config in self.ik_joint_mappings.items():
                    if config["human_body"] == body_name:
                        joint_config = config
                        break
                if joint_config:
                    self.pos_offsets2[body_name] = np.array(joint_config["pos_offset"]) - self.ground

            self.is_first_retarget = False


        self._clamp_to_joint_limits()
        self.update_targets(human_data, offset_to_ground)

        self._solve_tasks_iteratively(self.tasks1)
        self._solve_tasks_iteratively(self.tasks2)

        qpos = self.configuration.data.qpos.copy()
        if self.auto_ground_detection:
            qpos[2] += self.ground_qpos_z_offset
        return qpos

    def error1(self):

        return np.linalg.norm(np.concatenate([task.compute_error(self.configuration) for task in self.tasks1]))

    def error2(self):

        return self._compute_tasks_error(self.tasks2)

    def to_numpy(self, human_data):

        for body_name in human_data.keys():
            human_data[body_name] = [np.asarray(human_data[body_name][0]), np.asarray(human_data[body_name][1])]
        return human_data

    def offset_human_data(self, human_data, pos_offsets, rot_offsets):
        offset_human_data = {}
        for body_name in human_data.keys():
            pos, quat = human_data[body_name]
            offset_human_data[body_name] = [pos, quat]

            rotation_offset = rot_offsets.get(body_name, R.from_quat([1, 0, 0, 0], scalar_first=True))
            updated_quat = (R.from_quat(quat, scalar_first=True) * rotation_offset).as_quat(scalar_first=True)
            offset_human_data[body_name][1] = updated_quat

            local_offset = pos_offsets[body_name]
            global_pos_offset = R.from_quat(updated_quat, scalar_first=True).apply(local_offset)
            offset_human_data[body_name][0] = pos + global_pos_offset

        return offset_human_data


    def scale_human_data(self, human_data, human_root_name, human_scale_table):

        human_data_local = {}
        root_pos, root_quat = human_data[human_root_name]

        root_scale = human_scale_table[human_root_name]
        if isinstance(root_scale, (int, float)):
            scaled_root_pos = root_scale * root_pos
        else:
            scaled_root_pos = np.array(root_scale) * root_pos

        for body_name in human_data.keys():
            if body_name not in human_scale_table:
                continue
            if body_name == human_root_name:
                continue
            else:
                local_pos = human_data[body_name][0] - root_pos
                body_scale = human_scale_table[body_name]
                if isinstance(body_scale, (int, float)):
                    scaled_local_pos = body_scale * local_pos
                else:
                    scaled_local_pos = np.array(body_scale) * local_pos
                human_data_local[body_name] = scaled_local_pos

        human_data_global = {human_root_name: (scaled_root_pos, root_quat)}
        for body_name in human_data_local.keys():
            human_data_global[body_name] = (human_data_local[body_name] + scaled_root_pos, human_data[body_name][1])

        return human_data_global

        
    def _detect_ground_offset(self):

        possible_left_names = ["Left_Foot_Mod", "LeftFootMod", "Left_Foot", "left_foot", "LeftFoot", "LeftAnkle", "Foot_L", "left_ankle_roll_link", "Left_Ankle"]
        possible_right_names = ["Right_Foot_Mod", "RightFootMod", "Right_Foot", "right_foot", "RightFoot", "RightAnkle", "Foot_R", "right_ankle_roll_link", "Right_Ankle"]
        
        left_foot_link = None
        for name in possible_left_names:
            left_foot_link = self._get_foot_link_name(name)
            if left_foot_link:
                break
        
        right_foot_link = None
        for name in possible_right_names:
            right_foot_link = self._get_foot_link_name(name)
            if right_foot_link:
                break

        if left_foot_link is None or right_foot_link is None:
            print("[Warning] Cannot find foot links for auto ground detection")
            print(f"  Searched for left foot: {possible_left_names}")
            print(f"  Searched for right foot: {possible_right_names}")
            return 0.0

        mj.mj_forward(self.model, self.configuration.data)

        if self.ground_detection_mode == "mesh":

            left_foot_lowest = self._get_mesh_lowest_point(left_foot_link)
            right_foot_lowest = self._get_mesh_lowest_point(right_foot_link)

            if left_foot_lowest is None:
                left_foot_lowest = self.configuration.data.body(left_foot_link).xpos[2]
                print(f"  [Fallback] Using left foot body center")

            if right_foot_lowest is None:
                right_foot_lowest = self.configuration.data.body(right_foot_link).xpos[2]
                print(f"  [Fallback] Using right foot body center")

            ground_offset = min(left_foot_lowest, right_foot_lowest)

            print(f"[Auto Ground Detection - Mesh Mode]")
            print(f"  Left foot lowest point: {left_foot_lowest:.4f}")
            print(f"  Right foot lowest point: {right_foot_lowest:.4f}")
            print(f"  Detected ground offset: {ground_offset:.4f}")

        elif self.ground_detection_mode == "link":

            left_foot_z = self.configuration.data.body(left_foot_link).xpos[2]
            right_foot_z = self.configuration.data.body(right_foot_link).xpos[2]
            ground_offset = min(left_foot_z, right_foot_z)

            print(f"[Auto Ground Detection - Link Mode]")
            print(f"  Left foot link center: {left_foot_z:.4f}")
            print(f"  Right foot link center: {right_foot_z:.4f}")
            print(f"  Detected ground offset: {ground_offset:.4f}")
            print(f"  Note: Add manual ground_height in config for foot thickness compensation")

        else:

            ground_offset = 0.0

        return ground_offset        

    def _get_foot_link_name(self, human_foot_name):
        if human_foot_name in self.ik_joint_mappings:
            return human_foot_name
        for robot_link, config in self.ik_joint_mappings.items():
            if config["human_body"] == human_foot_name:
                return robot_link
        return None

    def _get_mesh_lowest_point(self, body_name):
        mj.mj_forward(self.model, self.configuration.data)

        body_id = self.model.body(body_name).id
        lowest_z = float("inf")

        for geom_id in range(self.model.ngeom):
            if self.model.geom_bodyid[geom_id] == body_id:
                geom_type = self.model.geom_type[geom_id]

                if geom_type == mj.mjtGeom.mjGEOM_MESH:
                    mesh_id = self.model.geom_dataid[geom_id]
                    if mesh_id >= 0:

                        mesh_vertadr = self.model.mesh_vertadr[mesh_id]
                        mesh_vertnum = self.model.mesh_vertnum[mesh_id]
                        geom_pos = self.configuration.data.geom_xpos[geom_id]
                        geom_mat = self.configuration.data.geom_xmat[geom_id].reshape(3, 3)

                        for i in range(mesh_vertnum):
                            vert_idx = mesh_vertadr + i
                            vert_local = self.model.mesh_vert[vert_idx]
                            vert_world = geom_pos + geom_mat @ vert_local
                            lowest_z = min(lowest_z, vert_world[2])

                elif geom_type in [mj.mjtGeom.mjGEOM_BOX, mj.mjtGeom.mjGEOM_CYLINDER, mj.mjtGeom.mjGEOM_SPHERE, mj.mjtGeom.mjGEOM_CAPSULE]:
                    geom_pos = self.configuration.data.geom_xpos[geom_id]
                    geom_size = self.model.geom_size[geom_id]
                    if geom_type == mj.mjtGeom.mjGEOM_SPHERE:
                        lowest_z = min(lowest_z, geom_pos[2] - geom_size[0])
                    elif geom_type == mj.mjtGeom.mjGEOM_BOX:
                        lowest_z = min(lowest_z, geom_pos[2] - geom_size[2])
                    elif geom_type in [mj.mjtGeom.mjGEOM_CYLINDER, mj.mjtGeom.mjGEOM_CAPSULE]:

                        lowest_z = min(lowest_z, geom_pos[2] - geom_size[1])

        return lowest_z if lowest_z != float("inf") else None        


    def _clamp_to_joint_limits(self):
        model = self.configuration.model
        data = self.configuration.data

        for i in range(model.njnt):
            if model.jnt_limited[i]:
                joint_range = model.jnt_range[i]
                qpos_idx = model.jnt_qposadr[i]
                joint_name = model.joint(i).name
                if model.jnt_type[i] == 0:
                    current_val = data.qpos[qpos_idx]
                    tolerance = 1e-5
                    clamped_val = np.clip(current_val, joint_range[0], joint_range[1])
                    if abs(current_val - clamped_val) > 1e-8:
                        print(f"Clamping joint {joint_name}: {current_val:.10f} -> {clamped_val:.10f} (range: [{joint_range[0]:.3f}, {joint_range[1]:.3f}])")
                    data.qpos[qpos_idx] = clamped_val        
