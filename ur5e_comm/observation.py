from __future__ import annotations

from dataclasses import dataclass
from time import monotonic

import numpy as np

from .robot import RobotSnapshot


ARM_JOINTS = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)
GRIPPER_JOINTS = (
    "finger_joint",
    "left_inner_finger_joint",
    "left_inner_knuckle_joint",
    "right_outer_knuckle_joint",
    "right_inner_finger_joint",
    "right_inner_knuckle_joint",
)
JOINT_ORDER = ARM_JOINTS + GRIPPER_JOINTS
GRIPPER_MULTIPLIERS = np.array([1.0, -1.0, 1.0, 1.0, -1.0, 1.0], dtype=np.float64)

# Isaac Gym task multiplies every URDF limit by 0.9 before normalization.
TRAINING_LOWER = np.array(
    [
        -5.654866776,
        -5.654866776,
        -2.827433388,
        -5.654866776,
        -5.654866776,
        -5.654866776,
        0.0,
        -0.78813,
        0.0,
        0.0,
        -0.78813,
        0.0,
    ],
    dtype=np.float64,
)
TRAINING_UPPER = np.array(
    [
        5.654866776,
        5.654866776,
        2.827433388,
        5.654866776,
        5.654866776,
        5.654866776,
        0.72,
        0.0,
        0.78813,
        0.729,
        0.0,
        0.78813,
    ],
    dtype=np.float64,
)


def _xyzw_from_wxyz(value):
    value = np.asarray(value, dtype=np.float64)
    return value[[1, 2, 3, 0]]


def _rotate_xyzw(quaternion, vector):
    x, y, z, w = np.asarray(quaternion, dtype=np.float64)
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    return rotation.dot(np.asarray(vector, dtype=np.float64))


@dataclass(frozen=True)
class ProjectedRobotState:
    joint_position: np.ndarray
    joint_velocity: np.ndarray
    palm_center_position: np.ndarray
    palm_state: np.ndarray
    fingertip_position: np.ndarray


class MujocoStateProjector:
    """Reconstruct the Isaac Gym full robot state from RTDE + gripper data."""

    def __init__(self, model_path, palm_offset=(0.0, 0.0, 0.145), gripper_range=0.72):
        try:
            import mujoco
        except ImportError as error:
            raise RuntimeError("mujoco is required for robot-state projection") from error
        self.mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self.palm_offset = np.asarray(palm_offset, dtype=np.float64)
        self.gripper_range = float(gripper_range)
        if self.palm_offset.shape != (3,) or not np.isfinite(self.palm_offset).all():
            raise ValueError("palm_offset must contain three finite values")
        if not np.isfinite(self.gripper_range) or self.gripper_range <= 0.0:
            raise ValueError("gripper_range must be positive")
        self.qpos_address = np.asarray([self._qpos_address(name) for name in JOINT_ORDER])
        self.dof_address = np.asarray([self._dof_address(name) for name in JOINT_ORDER])
        self.palm_body = self._body("robotiq_arg2f_base_link")
        self.fingertip_bodies = np.asarray(
            [self._body("robotiq_left_inner_finger_pad"), self._body("robotiq_right_inner_finger_pad")]
        )
        self.previous_gripper = None
        self.previous_timestamp = None

    def _joint(self, name):
        index = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_JOINT, name)
        if index < 0:
            raise ValueError("MJCF is missing joint {}".format(name))
        return index

    def _qpos_address(self, name):
        return int(self.model.jnt_qposadr[self._joint(name)])

    def _dof_address(self, name):
        return int(self.model.jnt_dofadr[self._joint(name)])

    def _body(self, name):
        index = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_BODY, name)
        if index < 0:
            raise ValueError("MJCF is missing body {}".format(name))
        return index

    def reset(self):
        self.previous_gripper = None
        self.previous_timestamp = None

    def _joint_state(self, snapshot: RobotSnapshot):
        master = float(snapshot.gripper_position) / 255.0 * self.gripper_range
        if self.previous_gripper is None or snapshot.timestamp <= self.previous_timestamp:
            master_velocity = 0.0
        else:
            master_velocity = (master - self.previous_gripper) / (snapshot.timestamp - self.previous_timestamp)
        qpos = np.concatenate((snapshot.joint_position, master * GRIPPER_MULTIPLIERS))
        qvel = np.concatenate((snapshot.joint_velocity, master_velocity * GRIPPER_MULTIPLIERS))
        self.previous_gripper = master
        self.previous_timestamp = snapshot.timestamp
        return qpos, qvel

    def _set_state(self, qpos, qvel):
        self.data.qpos[self.qpos_address] = qpos
        self.data.qvel[self.dof_address] = qvel
        self.mujoco.mj_forward(self.model, self.data)

    def _body_velocity(self, body):
        value = np.empty(6, dtype=np.float64)
        self.mujoco.mj_objectVelocity(
            self.model,
            self.data,
            self.mujoco.mjtObj.mjOBJ_BODY,
            int(body),
            value,
            0,
        )
        return value[3:6], value[0:3]

    def project(self, snapshot: RobotSnapshot) -> ProjectedRobotState:
        qpos, qvel = self._joint_state(snapshot)
        self._set_state(qpos, qvel)
        palm_quaternion = _xyzw_from_wxyz(self.data.xquat[self.palm_body])
        palm_position = self.data.xpos[self.palm_body].copy()
        palm_center = palm_position + _rotate_xyzw(palm_quaternion, self.palm_offset)
        linear, angular = self._body_velocity(self.palm_body)
        palm_state = np.concatenate((palm_quaternion, linear, angular))
        fingertips = self.data.xpos[self.fingertip_bodies].copy()
        return ProjectedRobotState(qpos, qvel, palm_center, palm_state, fingertips)

    def palm_position_for(self, arm_position, gripper_position):
        master = float(gripper_position) / 255.0 * self.gripper_range
        qpos = np.concatenate((np.asarray(arm_position, dtype=np.float64), master * GRIPPER_MULTIPLIERS))
        self._set_state(qpos, np.zeros(12, dtype=np.float64))
        quaternion = _xyzw_from_wxyz(self.data.xquat[self.palm_body])
        return self.data.xpos[self.palm_body].copy() + _rotate_xyzw(quaternion, self.palm_offset)


class Ur5eObservationBuilder:
    observation_dim = 69

    def __init__(self, projector, goal_position, object_quaternion=(0.0, 0.0, 0.0, 1.0), object_scale=(1.0, 1.0, 1.0), clamp=10.0):
        self.projector = projector
        self.goal_position = np.asarray(goal_position, dtype=np.float64)
        self.object_quaternion = np.asarray(object_quaternion, dtype=np.float64)
        self.object_scale = np.asarray(object_scale, dtype=np.float64)
        self.clamp = float(clamp)
        if not np.isfinite(self.clamp) or self.clamp <= 0.0:
            raise ValueError("observation clamp must be positive")
        for value, shape, name in (
            (self.goal_position, (3,), "goal_position"),
            (self.object_quaternion, (4,), "object_quaternion"),
            (self.object_scale, (3,), "object_scale"),
        ):
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError("{} has invalid shape or values".format(name))
        norm = np.linalg.norm(self.object_quaternion)
        if norm <= 1e-8:
            raise ValueError("object quaternion cannot be zero")
        self.object_quaternion = self.object_quaternion / norm
        self.object_position = None
        self.object_linear_velocity = np.zeros(3, dtype=np.float64)
        self.initial_object_z = None
        self.progress = 0
        self.successes = 0.0
        self.lifted = False
        self.reward = 0.0
        self.closest_keypoint = np.inf
        self.closest_fingertips = np.full(2, np.inf, dtype=np.float64)
        self.furthest_hand = -np.inf

    def begin_episode(self, initial_object_position):
        position = np.asarray(initial_object_position, dtype=np.float64)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError("initial object position must contain three finite values")
        self.object_position = position.copy()
        self.object_linear_velocity[:] = 0.0
        self.initial_object_z = float(position[2])
        self.progress = 0
        self.successes = 0.0
        self.lifted = False
        self.reward = 0.0
        self.closest_keypoint = float(np.linalg.norm(position - self.goal_position))
        self.closest_fingertips[:] = np.inf
        self.furthest_hand = -np.inf
        self.projector.reset()

    def update_object(self, position, linear_velocity):
        position = np.asarray(position, dtype=np.float64)
        velocity = np.asarray(linear_velocity, dtype=np.float64)
        if position.shape != (3,) or velocity.shape != (3,):
            raise ValueError("object position and linear velocity must contain three values")
        if not np.isfinite(position).all() or not np.isfinite(velocity).all():
            raise ValueError("object position and linear velocity must be finite")
        self.object_position = position.copy()
        self.object_linear_velocity = velocity.copy()

    def build(self, snapshot: RobotSnapshot, object_state=None):
        if self.object_position is None:
            raise RuntimeError("begin_episode must be called before building observations")
        if object_state is not None:
            self.update_object(object_state.position_base, object_state.linear_velocity_base)
        state = self.projector.project(snapshot)
        scaled_qpos = 2.0 * (state.joint_position - TRAINING_LOWER) / (TRAINING_UPPER - TRAINING_LOWER) - 1.0
        fingertips_relative_palm = state.fingertip_position - state.palm_center_position
        fingertip_distances = np.linalg.norm(state.fingertip_position - self.object_position, axis=1)
        if not np.isfinite(self.closest_fingertips).all():
            self.closest_fingertips = fingertip_distances.copy()
            self.furthest_hand = float(fingertip_distances[0])
        closest_fingertips_before_update = self.closest_fingertips.copy()
        keypoint_relative_palm = self.object_position - state.palm_center_position
        keypoint_relative_goal = self.object_position - self.goal_position
        keypoint_distance = float(np.linalg.norm(keypoint_relative_goal))
        closest_keypoint_before_update = self.closest_keypoint
        lifted_before_update = self.lifted
        successes_before_update = self.successes

        z_lift = 0.05 + self.object_position[2] - self.initial_object_z
        lifted = bool(z_lift > 0.15 or self.lifted)
        lifting_reward = float(np.clip(z_lift, 0.0, 0.5)) * (not lifted) * 20.0
        lift_bonus = 300.0 if lifted and not self.lifted else 0.0
        fingertip_delta = np.clip(
            self.closest_fingertips - fingertip_distances, 0.0, 10.0
        )
        fingertip_reward = float(fingertip_delta.sum()) * (not lifted) * 50.0
        keypoint_delta = float(np.clip(self.closest_keypoint - keypoint_distance, 0.0, 100.0))
        keypoint_reward = keypoint_delta * lifted * 200.0
        near_goal = keypoint_distance <= 0.075 * 1.5
        arm_penalty = -0.003 * float(np.abs(state.joint_velocity[:6]).sum())
        gripper_penalty = -0.0003 * float(np.abs(state.joint_velocity[6:]).sum())
        reward = (
            fingertip_reward
            + lifting_reward
            + lift_bonus
            + keypoint_reward
            + 1000.0 * near_goal
            + arm_penalty
            + gripper_penalty
        )
        object_state_values = np.concatenate(
            (self.object_quaternion, self.object_linear_velocity, np.zeros(3, dtype=np.float64))
        )
        observation = np.concatenate(
            (
                scaled_qpos,
                state.joint_velocity,
                state.palm_center_position,
                state.palm_state,
                object_state_values,
                fingertips_relative_palm.reshape(-1),
                keypoint_relative_palm,
                keypoint_relative_goal,
                self.object_scale,
                [closest_keypoint_before_update],
                closest_fingertips_before_update,
                [float(lifted_before_update)],
                [np.log(self.progress / 10.0 + 1.0)],
                [np.log(successes_before_update + 1.0)],
                [0.01 * reward],
            )
        ).astype(np.float32)
        if observation.shape != (self.observation_dim,):
            raise RuntimeError("Observation shape mismatch: {}".format(observation.shape))
        observation = np.clip(observation, -self.clamp, self.clamp)
        if not np.isfinite(observation).all():
            raise RuntimeError("Observation contains NaN or Inf")
        self.closest_fingertips = np.minimum(self.closest_fingertips, fingertip_distances)
        self.furthest_hand = max(self.furthest_hand, float(fingertip_distances[0]))
        self.closest_keypoint = min(self.closest_keypoint, keypoint_distance)
        self.lifted = lifted
        if near_goal:
            self.successes += 1.0
        self.reward = reward
        return observation, state

    def advance(self):
        self.progress += 1


class Ur5eActionMapper:
    action_dim = 7

    def __init__(self, period=0.01667, speed_scale=10.0, max_arm_step=0.05, arm_lower=None, arm_upper=None):
        self.period = float(period)
        self.speed_scale = float(speed_scale)
        self.max_arm_step = float(max_arm_step)
        self.lower = TRAINING_LOWER[:6].copy() if arm_lower is None else np.asarray(arm_lower, dtype=np.float64)
        self.upper = TRAINING_UPPER[:6].copy() if arm_upper is None else np.asarray(arm_upper, dtype=np.float64)
        if self.lower.shape != (6,) or self.upper.shape != (6,) or np.any(self.lower >= self.upper):
            raise ValueError("arm limits must be six valid lower/upper pairs")
        if self.period <= 0.0 or self.speed_scale <= 0.0 or self.max_arm_step <= 0.0:
            raise ValueError("period, speed scale, and maximum arm step must be positive")
        self.previous_arm_target = None

    def reset(self, snapshot: RobotSnapshot):
        self.previous_arm_target = np.clip(snapshot.joint_position.copy(), self.lower, self.upper)

    def map(self, action):
        if self.previous_arm_target is None:
            raise RuntimeError("Action mapper must be reset from the measured robot state")
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        if action.shape != (self.action_dim,) or not np.isfinite(action).all():
            raise ValueError("Expected finite action shape (7,), got {}".format(action.shape))
        action = np.clip(action, -1.0, 1.0)
        delta = self.speed_scale * self.period * action[:6]
        delta = np.clip(delta, -self.max_arm_step, self.max_arm_step)
        arm_target = np.clip(self.previous_arm_target + delta, self.lower, self.upper)
        self.previous_arm_target = arm_target
        gripper_position = int(np.clip(round((action[6] + 1.0) * 127.5), 0, 255))
        return arm_target.copy(), gripper_position


class SafetyMonitor:
    def __init__(
        self,
        arm_lower,
        arm_upper,
        max_joint_velocity=2.0,
        max_target_error=0.2,
        max_state_age=0.25,
        workspace_min=(-1.0, -1.0, 0.0),
        workspace_max=(1.0, 1.0, 1.5),
        object_position_min=(-1.0, -1.0, 0.0),
        object_position_max=(1.0, 1.0, 1.5),
    ):
        self.lower = np.asarray(arm_lower, dtype=np.float64)
        self.upper = np.asarray(arm_upper, dtype=np.float64)
        self.max_joint_velocity = float(max_joint_velocity)
        self.max_target_error = float(max_target_error)
        self.max_state_age = float(max_state_age)
        self.workspace_min = np.asarray(workspace_min, dtype=np.float64)
        self.workspace_max = np.asarray(workspace_max, dtype=np.float64)
        self.object_position_min = np.asarray(object_position_min, dtype=np.float64)
        self.object_position_max = np.asarray(object_position_max, dtype=np.float64)
        for lower, upper, name in (
            (self.lower, self.upper, "arm"),
            (self.workspace_min, self.workspace_max, "workspace"),
            (self.object_position_min, self.object_position_max, "object position"),
        ):
            expected = (6,) if name == "arm" else (3,)
            if lower.shape != expected or upper.shape != expected or not np.isfinite(lower).all() or not np.isfinite(upper).all() or np.any(lower >= upper):
                raise ValueError("{} limits are invalid".format(name))
        if min(self.max_joint_velocity, self.max_target_error, self.max_state_age) <= 0.0:
            raise ValueError("safety velocity, target error, and state age must be positive")

    def check_initial_object(self, position):
        self.check_object_position(position)

    def check_object_position(self, position):
        position = np.asarray(position, dtype=np.float64)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise RuntimeError("Detected object position is invalid")
        if np.any(position < self.object_position_min) or np.any(position > self.object_position_max):
            raise RuntimeError("Detected object position {} is outside configured bounds".format(position.tolist()))

    def check_state(self, snapshot: RobotSnapshot, now=None):
        now = monotonic() if now is None else float(now)
        if snapshot.protective_stopped or snapshot.emergency_stopped:
            raise RuntimeError("UR safety/emergency stop is active")
        if now - snapshot.timestamp > self.max_state_age:
            raise RuntimeError("Robot state is stale by {:.3f}s".format(now - snapshot.timestamp))
        if np.any(snapshot.joint_position < self.lower) or np.any(snapshot.joint_position > self.upper):
            raise RuntimeError("Measured arm joint is outside configured hard limits")
        if np.max(np.abs(snapshot.joint_velocity)) > self.max_joint_velocity:
            raise RuntimeError("Measured arm velocity exceeds safety limit")

    def check_target(self, snapshot, arm_target, gripper_position, projector):
        arm_target = np.asarray(arm_target, dtype=np.float64)
        if arm_target.shape != (6,) or not np.isfinite(arm_target).all():
            raise RuntimeError("Commanded arm target is invalid")
        if not 0 <= int(gripper_position) <= 255:
            raise RuntimeError("Commanded gripper target is invalid")
        if np.max(np.abs(arm_target - snapshot.joint_position)) > self.max_target_error:
            raise RuntimeError("Command target is too far from measured joints")
        palm = projector.palm_position_for(arm_target, gripper_position)
        if np.any(palm < self.workspace_min) or np.any(palm > self.workspace_max):
            raise RuntimeError("Commanded palm position {} leaves safety workspace".format(palm.tolist()))
        return palm
