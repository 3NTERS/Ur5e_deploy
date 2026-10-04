"""MuJoCo adapter for the 69-observation, 7-action Ur5eRobotiq tasks."""

from pathlib import Path

import mujoco
import numpy as np
import yaml


GRIPPER_MULTIPLIERS = np.asarray([1.0, -1.0, 1.0, 1.0, -1.0, 1.0])
ARM_ACTUATORS = ("shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3")
FINGERTIPS = ("robotiq_left_inner_finger_pad", "robotiq_right_inner_finger_pad")


def wxyz_to_xyzw(quaternion):
    return np.asarray(quaternion, dtype=np.float64)[[1, 2, 3, 0]]


def xyzw_to_wxyz(quaternion):
    return np.asarray(quaternion, dtype=np.float64)[[3, 0, 1, 2]]


class Ur5eThrowAdapter:
    observation_dim = 69
    action_dim = 7
    episode_length = 600
    policy_period = 0.01667

    def __init__(self, scene_path, metadata_path, seed=0, object_scale=(1.0, 1.0, 1.0)):
        self.scene_path = Path(scene_path)
        self.metadata = yaml.safe_load(Path(metadata_path).read_text(encoding="utf-8"))
        self.task = str(self.metadata.get("task", "Ur5eRobotiqThrow"))
        self.is_grasp = self.task == "Ur5eRobotiqGrasp"
        self.is_hover = self.task == "Ur5eRobotiqHoverGripper"
        contract = self.metadata.get("task_contract", {})
        self.episode_length = int(contract.get("episode_steps", self.episode_length))
        self.policy_period = float(contract.get("policy_period_s", self.policy_period))
        self.arm_action_scale = float(contract.get("arm_action_scale", 10.0))
        self.observation_joint_order = np.asarray(
            contract.get("observation_joint_order_indices", list(range(12))), dtype=np.int64
        )
        self.palm_quaternion_reference = np.asarray(
            contract.get("palm_quaternion_reference_xyzw", (-1.0, 0.0, 0.0, 0.0)),
            dtype=np.float64,
        )
        self.max_command_joint_velocity = float(
            contract.get("max_command_joint_velocity_rad_s", float("inf"))
        )
        self.max_command_joint_acceleration = float(
            contract.get("max_command_joint_acceleration_rad_s2", float("inf"))
        )
        self.model = mujoco.MjModel.from_xml_path(str(self.scene_path))
        if bool(contract.get("disable_robot_gravity", False)):
            robot_root = self._id(mujoco.mjtObj.mjOBJ_BODY, "base")
            for body_id in range(1, self.model.nbody):
                ancestor = body_id
                while ancestor != 0 and ancestor != robot_root:
                    ancestor = int(self.model.body_parentid[ancestor])
                if ancestor == robot_root:
                    self.model.body_gravcomp[body_id] = 1.0
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.RandomState(int(seed))
        io = self.metadata["io"]
        if int(io["observation_dim"]) != self.observation_dim or int(io["action_dim"]) != self.action_dim:
            raise ValueError("UR5e policy metadata has an incompatible IO contract")
        self.joint_names = tuple(io["joint_order"])
        self.lower = np.asarray(io["training_joint_lower"], dtype=np.float64)
        self.upper = np.asarray(io["training_joint_upper"], dtype=np.float64)
        self.qpos_addr = np.asarray([self.model.jnt_qposadr[self._joint_id(name)] for name in self.joint_names])
        self.dof_addr = np.asarray([self.model.jnt_dofadr[self._joint_id(name)] for name in self.joint_names])
        self.arm_actuator_ids = np.asarray([self._actuator_id(name) for name in ARM_ACTUATORS])
        if self.is_hover:
            position_kp = float(contract.get("position_kp", 80.0))
            position_kd = float(contract.get("position_kd", 15.0))
            self.model.actuator_gainprm[self.arm_actuator_ids, 0] = position_kp
            self.model.actuator_biasprm[self.arm_actuator_ids, 1] = -position_kp
            self.model.actuator_biasprm[self.arm_actuator_ids, 2] = -position_kd
        self.gripper_actuator_id = self._actuator_id("fingers_actuator")
        self.palm_site = self._site_id("robotiq_pinch")
        self.palm_body = self._body_id("robotiq_arg2f_base_link")
        self.fingertip_bodies = np.asarray([self._body_id(name) for name in FINGERTIPS])
        self.object_body = self._body_id("object")
        self.object_geom = self._geom_id("object_geom")
        self.object_joint = self._joint_id("object_freejoint")
        self.object_qpos_addr = int(self.model.jnt_qposadr[self.object_joint])
        self.object_dof_addr = int(self.model.jnt_dofadr[self.object_joint])
        bucket_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "bucket")
        self.bucket_body = None if bucket_id < 0 else bucket_id
        self.goal_site = self._site_id("goal")
        self.home = np.asarray(
            contract.get("initial_arm_rad",
                         (-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0)),
            dtype=np.float64,
        )
        self.object_scale = np.asarray(object_scale, dtype=np.float64)
        self._set_object_scale(self.object_scale)
        self.progress = 0
        self.successes = 0.0
        self.lifted = False
        self.closest_keypoint = np.inf
        self.closest_fingertips = np.full(2, np.inf)
        self.furthest_hand = -np.inf
        self.previous_target = np.zeros(12, dtype=np.float64)
        self.last_action = np.zeros(7, dtype=np.float64)
        self.command_velocity = np.zeros(6, dtype=np.float64)
        self.initial_object_z = 0.15
        self.last_reward = 0.0

    def _id(self, object_type, name):
        value = mujoco.mj_name2id(self.model, object_type, name)
        if value < 0:
            raise ValueError("Scene is missing {}".format(name))
        return value

    def _joint_id(self, name):
        return self._id(mujoco.mjtObj.mjOBJ_JOINT, name)

    def _actuator_id(self, name):
        return self._id(mujoco.mjtObj.mjOBJ_ACTUATOR, name)

    def _body_id(self, name):
        return self._id(mujoco.mjtObj.mjOBJ_BODY, name)

    def _site_id(self, name):
        return self._id(mujoco.mjtObj.mjOBJ_SITE, name)

    def _geom_id(self, name):
        return self._id(mujoco.mjtObj.mjOBJ_GEOM, name)

    def _set_object_scale(self, scale):
        half_size = 0.02 * np.asarray(scale, dtype=np.float64)
        mass = 567.0 * float(np.prod(2.0 * half_size))
        self.model.geom_size[self.object_geom] = half_size
        self.model.body_mass[self.object_body] = mass
        self.model.body_inertia[self.object_body] = mass / 3.0 * np.asarray(
            [half_size[1] ** 2 + half_size[2] ** 2,
             half_size[0] ** 2 + half_size[2] ** 2,
             half_size[0] ** 2 + half_size[1] ** 2]
        )

    def _velocity(self, object_type, object_id):
        velocity = np.empty(6, dtype=np.float64)
        mujoco.mj_objectVelocity(self.model, self.data, object_type, object_id, velocity, 0)
        return velocity[3:6].copy(), velocity[0:3].copy()

    def _default_initial_state(self):
        if self.is_grasp or self.is_hover:
            object_position = np.asarray((0.55, 0.0, 0.15))
            hover_height = float(self.metadata.get("task_contract", {}).get("hover_height_m", 0.12))
            return (
                np.concatenate((self.home, np.zeros(6))),
                np.zeros(12),
                object_position,
                np.asarray((0.0, 0.0, 0.0, 1.0)),
                object_position + np.asarray((0.0, 0.0, hover_height)),
            )
        default_qpos = np.concatenate((self.home, np.zeros(6)))
        delta = self.lower - default_qpos + (self.upper - self.lower) * self.rng.uniform(0.0, 1.0, 12)
        qpos = default_qpos + 0.05 * delta
        qvel = 0.2 * self.rng.uniform(-1.0, 1.0, 12)
        object_position = np.asarray(
            [0.55 + self.rng.uniform(-0.05, 0.05), self.rng.uniform(-0.05, 0.05),
             0.15 + self.rng.uniform(-0.01, 0.01)]
        )
        uvw = self.rng.uniform(0.0, 1.0, 3)
        object_quaternion = np.asarray(
            [np.sqrt(1.0 - uvw[0]) * np.cos(2.0 * np.pi * uvw[1]),
             np.sqrt(uvw[0]) * np.sin(2.0 * np.pi * uvw[2]),
             np.sqrt(uvw[0]) * np.cos(2.0 * np.pi * uvw[2]),
             np.sqrt(1.0 - uvw[0]) * np.sin(2.0 * np.pi * uvw[1])]
        )
        side = 1.0 if self.rng.uniform(-1.0, 1.0) > 0.0 else -1.0
        goal = np.asarray(
            [self.rng.uniform(0.65, 0.95), side * self.rng.uniform(0.25, 0.45), 0.05]
        )
        return qpos, qvel, object_position, object_quaternion, goal

    def reset(self, initial=None):
        mujoco.mj_resetData(self.model, self.data)
        if initial is None:
            qpos, qvel, object_position, object_quaternion, goal = self._default_initial_state()
            object_velocity = np.zeros(6)
            joint_target = qpos
        else:
            qpos = np.asarray(initial["joint_position"], dtype=np.float64)
            qvel = np.asarray(initial["joint_velocity"], dtype=np.float64)
            object_state = np.asarray(initial["object_state"], dtype=np.float64)
            object_position = object_state[:3]
            object_quaternion = object_state[3:7]
            object_velocity = object_state[7:13]
            goal = np.asarray(initial["goal_position"], dtype=np.float64)
            joint_target = np.asarray(initial["joint_target"], dtype=np.float64)
        self.data.qpos[self.qpos_addr] = qpos
        self.data.qvel[self.dof_addr] = qvel
        address = self.object_qpos_addr
        self.data.qpos[address:address + 3] = object_position
        self.data.qpos[address + 3:address + 7] = xyzw_to_wxyz(object_quaternion)
        self.data.qvel[self.object_dof_addr:self.object_dof_addr + 6] = object_velocity
        if self.bucket_body is not None:
            self.model.body_pos[self.bucket_body] = [goal[0], goal[1], goal[2] - 0.05]
        self.model.site_pos[self.goal_site] = goal
        self.previous_target = joint_target.copy()
        self.data.ctrl[self.arm_actuator_ids] = joint_target[:6]
        self.data.ctrl[self.gripper_actuator_id] = np.clip(joint_target[6] / 0.72 * 255.0, 0.0, 255.0)
        self.progress = 0
        self.successes = 0.0
        self.lifted = False
        self.closest_keypoint = np.inf
        self.closest_fingertips[:] = np.inf
        self.furthest_hand = -np.inf
        self.last_action[:] = 0.0
        self.command_velocity[:] = 0.0
        self.last_reward = 0.0
        self.initial_object_z = float(object_position[2])
        mujoco.mj_forward(self.model, self.data)
        return self.observe(update_reward=False)

    def _state_parts(self):
        qpos = self.data.qpos[self.qpos_addr].copy()
        qvel = self.data.qvel[self.dof_addr].copy()
        palm_position = self.data.site_xpos[self.palm_site].copy()
        palm_quaternion = wxyz_to_xyzw(self.data.xquat[self.palm_body])
        palm_linear, palm_angular = self._velocity(mujoco.mjtObj.mjOBJ_BODY, self.palm_body)
        fingertips = self.data.xpos[self.fingertip_bodies].copy()
        object_position = self.data.xpos[self.object_body].copy()
        object_quaternion = wxyz_to_xyzw(self.data.xquat[self.object_body])
        object_linear, object_angular = self._velocity(mujoco.mjtObj.mjOBJ_BODY, self.object_body)
        goal = self.data.site_xpos[self.goal_site].copy()
        return (
            qpos, qvel, palm_position, palm_quaternion, palm_linear, palm_angular,
            fingertips, object_position, object_quaternion, object_linear, object_angular, goal,
        )

    def observe(self, update_reward=True):
        parts = self._state_parts()
        (qpos, qvel, palm_position, palm_quaternion, palm_linear, palm_angular,
         fingertips, object_position, object_quaternion, object_linear, object_angular, goal) = parts
        fingertip_distances = np.linalg.norm(fingertips - object_position, axis=1)
        keypoint_distance = float(np.linalg.norm(object_position - goal))
        if not np.isfinite(self.closest_keypoint):
            self.closest_keypoint = keypoint_distance
            self.closest_fingertips = fingertip_distances.copy()
            self.furthest_hand = float(fingertip_distances[0])
        closest_keypoint = self.closest_keypoint
        closest_fingertips = self.closest_fingertips.copy()
        lifted = self.lifted
        successes = self.successes
        reward = self._reward(qvel, object_position, keypoint_distance, fingertip_distances) if update_reward else 0.0
        if self.is_hover and np.dot(palm_quaternion, self.palm_quaternion_reference) < 0.0:
            palm_quaternion = -palm_quaternion
        order = self.observation_joint_order if self.is_hover else np.arange(12)
        scaled_qpos = (
            2.0 * (qpos[order] - self.lower[order])
            / (self.upper[order] - self.lower[order]) - 1.0
        )
        observation = np.concatenate(
            (
                scaled_qpos, qvel[order], palm_position,
                palm_quaternion, palm_linear, palm_angular,
                object_quaternion, object_linear, object_angular,
                (fingertips - palm_position).reshape(-1),
                object_position - palm_position, object_position - goal,
                self.object_scale, [closest_keypoint], closest_fingertips,
                [float(lifted)], [np.log(self.progress / 10.0 + 1.0)],
                [np.log(successes + 1.0)], [0.01 * reward],
            )
        ).astype(np.float32)
        if self.is_hover:
            for start, end in self.metadata["task_contract"].get(
                "observation_zero_slices", ((56, 59), (62, 66), (67, 69))
            ):
                observation[int(start):int(end)] = 0.0
        observation = np.clip(observation, -10.0, 10.0)
        if observation.shape != (self.observation_dim,) or not np.isfinite(observation).all():
            raise RuntimeError("Invalid UR5e observation")
        self.last_reward = reward
        return observation

    def _reward(self, qvel, object_position, keypoint_distance, fingertip_distances):
        was_lifted = self.lifted
        z_lift = 0.05 + object_position[2] - self.initial_object_z
        self.lifted = self.lifted or z_lift > 0.15
        lifting = float(np.clip(z_lift, 0.0, 0.5)) * (not self.lifted) * 20.0
        lift_bonus = 300.0 if self.lifted and not was_lifted else 0.0
        fingertip_delta = np.clip(self.closest_fingertips - fingertip_distances, 0.0, 10.0)
        fingertip_reward = float(fingertip_delta.sum()) * (not self.lifted) * 50.0
        keypoint_delta = float(np.clip(self.closest_keypoint - keypoint_distance, 0.0, 100.0))
        keypoint_reward = 0.0 if self.is_grasp else keypoint_delta * self.lifted * 200.0
        self.closest_fingertips = np.minimum(self.closest_fingertips, fingertip_distances)
        self.furthest_hand = max(self.furthest_hand, float(fingertip_distances[0]))
        self.closest_keypoint = min(self.closest_keypoint, keypoint_distance)
        near_goal = (keypoint_distance <= 0.075 * 1.5) if not self.is_grasp else False
        if self.is_grasp and self.lifted and not was_lifted:
            self.successes += 1.0
        elif near_goal:
            self.successes += 1.0
        penalty = -0.003 * np.abs(qvel[:6]).sum() - 0.0003 * np.abs(qvel[6:]).sum()
        return fingertip_reward + lifting + lift_bonus + keypoint_reward + 1000.0 * near_goal + penalty

    def apply_action(self, action):
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        if action.shape != (7,) or not np.isfinite(action).all():
            raise ValueError("Expected a finite 7-dimensional action")
        action = np.clip(action, -1.0, 1.0)
        target = self.previous_target.copy()
        desired_velocity = np.clip(
            self.arm_action_scale * action[:6],
            -self.max_command_joint_velocity, self.max_command_joint_velocity,
        )
        max_velocity_delta = self.max_command_joint_acceleration * self.policy_period
        self.command_velocity += np.clip(
            desired_velocity - self.command_velocity,
            -max_velocity_delta, max_velocity_delta,
        )
        target[:6] = np.clip(
            target[:6] + self.command_velocity * self.policy_period,
            self.lower[:6], self.upper[:6],
        )
        master = 0.5 * (action[6] + 1.0) * 0.72
        target[6:] = np.clip(master * GRIPPER_MULTIPLIERS, self.lower[6:], self.upper[6:])
        self.previous_target = target
        self.last_action = action
        self.data.ctrl[self.arm_actuator_ids] = target[:6]
        self.data.ctrl[self.gripper_actuator_id] = master / 0.72 * 255.0
        return target.copy()

    def step_physics_to(self, target_time):
        while self.data.time + 1e-12 < target_time:
            mujoco.mj_step(self.model, self.data)
        self.progress += 1

    def snapshot(self):
        parts = self._state_parts()
        (qpos, qvel, palm_position, palm_quaternion, palm_linear, palm_angular,
         fingertips, object_position, object_quaternion, object_linear, object_angular, goal) = parts
        return {
            "joint_position": qpos,
            "joint_velocity": qvel,
            "joint_target": self.previous_target.copy(),
            "palm_position": palm_position,
            "palm_state": np.concatenate((palm_quaternion, palm_linear, palm_angular)),
            "fingertip_position": fingertips,
            "object_state": np.concatenate((object_position, object_quaternion, object_linear, object_angular)),
            "goal_position": goal,
        }

    @property
    def done(self):
        return self.progress >= self.episode_length or self.data.xpos[self.object_body, 2] < 0.0


Ur5eGraspAdapter = Ur5eThrowAdapter
