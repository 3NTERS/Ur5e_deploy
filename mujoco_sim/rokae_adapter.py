from pathlib import Path

import mujoco
import numpy as np
import yaml


FINGERTIPS = ("pf4_tip", "mf4_tip", "if4_tip", "th4_tip")


def wxyz_to_xyzw(quaternion):
    quaternion = np.asarray(quaternion, dtype=np.float64)
    return quaternion[[1, 2, 3, 0]]


def matrix_to_xyzw(matrix):
    quaternion = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quaternion, np.asarray(matrix, dtype=np.float64).reshape(9))
    return wxyz_to_xyzw(quaternion)


class RokaeAdapter:
    observation_dim = 99
    action_dim = 23
    episode_length = 600
    policy_period = 0.01667

    def __init__(self, scene_path, metadata_path, seed=0):
        self.scene_path = Path(scene_path)
        self.metadata = yaml.safe_load(Path(metadata_path).read_text(encoding="utf-8"))
        self.model = mujoco.MjModel.from_xml_path(str(self.scene_path))
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(seed)
        self.joint_names = list(self.metadata["io"]["joint_order"])
        limits = self.metadata["io"]["joint_limits"]
        self.lower = np.asarray([item["lower"] for item in limits], dtype=np.float64)
        self.upper = np.asarray([item["upper"] for item in limits], dtype=np.float64)
        self.qpos_addr = np.asarray([self.model.jnt_qposadr[self._joint_id(name)] for name in self.joint_names])
        self.dof_addr = np.asarray([self.model.jnt_dofadr[self._joint_id(name)] for name in self.joint_names])
        self.actuator_ids = np.asarray(
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{name}_position") for name in self.joint_names]
        )
        if np.any(self.actuator_ids < 0):
            raise ValueError("MJCF does not expose all 23 position actuators")
        self.palm_site = self._site_id("palm_center")
        self.tip_sites = np.asarray([self._site_id(name) for name in FINGERTIPS])
        self.object_body = self._body_id("object")
        self.bucket_body = self._body_id("bucket")
        self.goal_site = self._site_id("goal")
        self.object_joint = self._joint_id("object_freejoint")
        self.object_qpos_addr = self.model.jnt_qposadr[self.object_joint]
        self.default_qpos = np.asarray(self.metadata["control"]["default_joint_position"], dtype=np.float64)
        self.progress = 0
        self.successes = 0.0
        self.previous_reward = 0.0
        self.previous_target = self.default_qpos.copy()
        self.closest_keypoint = np.inf
        self.closest_fingertips = np.full(4, np.inf)
        self.furthest_hand = -np.inf
        self.lifted = False
        self.initial_object_z = 0.555

    def _joint_id(self, name):
        value = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if value < 0:
            raise ValueError(f"Missing joint {name}")
        return value

    def _site_id(self, name):
        value = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
        if value < 0:
            raise ValueError(f"Missing site {name}")
        return value

    def _body_id(self, name):
        value = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if value < 0:
            raise ValueError(f"Missing body {name}")
        return value

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.qpos_addr] = self.default_qpos
        self.data.qvel[self.dof_addr] = 0.0
        self.previous_target = self.default_qpos.copy()
        self.data.ctrl[self.actuator_ids] = self.previous_target
        side = -1.0 if self.rng.random() < 0.5 else 1.0
        self.model.body_pos[self.bucket_body] = [self.rng.uniform(0.5, 1.5), side * self.rng.uniform(0.5, 0.9), 0.45]
        address = self.object_qpos_addr
        self.data.qpos[address : address + 3] = [
            0.75 + self.rng.uniform(-0.05, 0.05),
            self.rng.uniform(-0.05, 0.05),
            self.initial_object_z + self.rng.uniform(-0.01, 0.01),
        ]
        quaternion = self.rng.normal(size=4)
        self.data.qpos[address + 3 : address + 7] = quaternion / np.linalg.norm(quaternion)
        self.progress = 0
        self.successes = 0.0
        self.previous_reward = 0.0
        self.closest_keypoint = np.inf
        self.closest_fingertips[:] = np.inf
        self.furthest_hand = -np.inf
        self.lifted = False
        mujoco.mj_forward(self.model, self.data)
        return self.observe()

    def _velocity(self, object_type, object_id):
        value = np.empty(6, dtype=np.float64)
        mujoco.mj_objectVelocity(self.model, self.data, object_type, object_id, value, 0)
        return value[3:6], value[0:3]

    def observe(self):
        qpos = self.data.qpos[self.qpos_addr].copy()
        qvel = self.data.qvel[self.dof_addr].copy()
        scaled_qpos = 2.0 * (qpos - self.lower) / (self.upper - self.lower) - 1.0
        palm_pos = self.data.site_xpos[self.palm_site].copy()
        palm_quat = matrix_to_xyzw(self.data.site_xmat[self.palm_site])
        palm_linvel, palm_angvel = self._velocity(mujoco.mjtObj.mjOBJ_SITE, self.palm_site)
        object_pos = self.data.xpos[self.object_body].copy()
        object_quat = wxyz_to_xyzw(self.data.xquat[self.object_body])
        object_linvel, object_angvel = self._velocity(mujoco.mjtObj.mjOBJ_BODY, self.object_body)
        fingertip_positions = self.data.site_xpos[self.tip_sites].copy()
        fingertips_rel_palm = (fingertip_positions - palm_pos).reshape(-1)
        fingertip_distances = np.linalg.norm(fingertip_positions - object_pos, axis=1)
        goal_pos = self.data.site_xpos[self.goal_site].copy()
        keypoint_rel_palm = object_pos - palm_pos
        keypoint_rel_goal = object_pos - goal_pos
        keypoint_distance = float(np.linalg.norm(keypoint_rel_goal))
        if not np.isfinite(self.closest_keypoint):
            self.closest_keypoint = keypoint_distance
        if not np.isfinite(self.closest_fingertips).all():
            self.closest_fingertips = fingertip_distances.copy()
        if not np.isfinite(self.furthest_hand):
            self.furthest_hand = float(fingertip_distances[0])

        closest_keypoint_before_reward = self.closest_keypoint
        closest_fingertips_before_reward = self.closest_fingertips.copy()
        lifted_before_reward = self.lifted
        successes_before_reward = self.successes
        reward = self._reward(
            qvel,
            object_pos,
            keypoint_distance,
            fingertip_distances,
        )
        observation = np.concatenate(
            (
                scaled_qpos,
                qvel,
                palm_pos,
                palm_quat,
                palm_linvel,
                palm_angvel,
                object_quat,
                object_linvel,
                object_angvel,
                fingertips_rel_palm,
                keypoint_rel_palm,
                keypoint_rel_goal,
                np.ones(3),
                [closest_keypoint_before_reward],
                closest_fingertips_before_reward,
                [float(lifted_before_reward)],
                [np.log(self.progress / 10.0 + 1.0)],
                [np.log(successes_before_reward + 1.0)],
                [0.01 * reward],
            )
        ).astype(np.float32)
        observation = np.clip(observation, -10.0, 10.0)
        if observation.shape != (self.observation_dim,):
            raise RuntimeError(f"Observation shape mismatch: {observation.shape}")
        if not np.isfinite(observation).all():
            raise RuntimeError("Observation contains NaN or Inf")
        self.previous_reward = reward
        return observation

    def _reward(self, qvel, object_pos, keypoint_distance, fingertip_distances):
        was_lifted = self.lifted
        z_lift = 0.05 + object_pos[2] - self.initial_object_z
        self.lifted = self.lifted or z_lift > 0.15
        lift_reward = np.clip(z_lift, 0.0, 0.5) * (not self.lifted) * 20.0
        lift_bonus = 300.0 if self.lifted and not was_lifted else 0.0
        fingertip_delta = np.clip(self.closest_fingertips - fingertip_distances, 0.0, 10.0)
        fingertip_reward = float(fingertip_delta.sum()) * (not self.lifted) * 50.0
        keypoint_delta = np.clip(self.closest_keypoint - keypoint_distance, 0.0, 100.0)
        keypoint_reward = float(keypoint_delta) * self.lifted * 200.0
        self.closest_fingertips = np.minimum(self.closest_fingertips, fingertip_distances)
        self.furthest_hand = max(self.furthest_hand, float(fingertip_distances[0]))
        self.closest_keypoint = min(self.closest_keypoint, keypoint_distance)
        near_goal = keypoint_distance <= 0.075 * 1.5
        if near_goal:
            self.successes += 1.0
        action_penalty = -0.003 * np.abs(qvel[:7]).sum() - 0.0003 * np.abs(qvel[7:]).sum()
        return fingertip_reward + lift_reward + lift_bonus + keypoint_reward + 1000.0 * near_goal + action_penalty

    def apply_action(self, action):
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        if action.shape != (self.action_dim,) or not np.isfinite(action).all():
            raise ValueError(f"Expected finite action shape ({self.action_dim},), got {action.shape}")
        action = np.clip(action, -1.0, 1.0)
        target = self.previous_target.copy()
        target[:7] = np.clip(
            target[:7] + 10.0 * self.policy_period * action[:7],
            self.lower[:7],
            self.upper[:7],
        )
        target[7:] = self.lower[7:] + 0.5 * (action[7:] + 1.0) * (self.upper[7:] - self.lower[7:])
        target = np.clip(target, self.lower, self.upper)
        self.previous_target = target
        self.data.ctrl[self.actuator_ids] = target
        return target.copy()

    def step_physics_to(self, target_time):
        while self.data.time + 1e-12 < target_time:
            mujoco.mj_step(self.model, self.data)
        self.progress += 1

    @property
    def done(self):
        return self.progress >= self.episode_length or self.data.xpos[self.object_body, 2] < 0.0
