#!/usr/bin/env python3
"""Collect a versioned single-environment Isaac Gym trajectory for MuJoCo replay."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from time import sleep

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--isaac-root",
        default="/home/liang/Workspace/Isaacgym_cjlu/IsaacGymEnvs",
        help="directory containing the isaacgymenvs package and assets",
    )
    parser.add_argument("--model", default="resources/models/ur5e_robotiq/policy.onnx")
    parser.add_argument("--provider", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--task", default="Ur5eRobotiqGrasp")
    parser.add_argument("--sim-device", default="cuda:0")
    parser.add_argument("--rl-device", default="cuda:0")
    parser.add_argument("--output", default="resources/trajectories/ur5e_isaac_reference.npz")
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--screenshot")
    parser.add_argument("--max-arm-velocity", type=float, default=float("inf"))
    parser.add_argument("--max-arm-acceleration", type=float, default=float("inf"))
    return parser.parse_args()


def _numpy(tensor):
    return tensor.detach().cpu().numpy().copy()


def _capture(env, canonical_indices):
    joint_position = _numpy(env.arm_hand_dof_pos[0, canonical_indices])
    joint_velocity = _numpy(env.arm_hand_dof_vel[0, canonical_indices])
    joint_target = _numpy(env.cur_targets[0, canonical_indices])
    palm_position = _numpy(env.palm_center_pos[0])
    palm_state = _numpy(env._palm_state[0, 3:13])
    fingertips = _numpy(env.fingertip_pos_offset[0])
    object_state = _numpy(env.object_state[0])
    goal_position = _numpy(env.goal_pos[0])
    return {
        "joint_position": joint_position,
        "joint_velocity": joint_velocity,
        "joint_target": joint_target,
        "palm_position": palm_position,
        "palm_state": palm_state,
        "fingertip_position": fingertips,
        "object_state": object_state,
        "goal_position": goal_position,
        "lifted": _numpy(env.lifted_object[0]),
        "success": _numpy(env.successes[0]),
        "reward": _numpy(env.rew_buf[0]),
    }


def _rigid_body_parameters(env, actor_name):
    handle = env.gym.find_actor_handle(env.envs[0], actor_name)
    names = env.gym.get_actor_rigid_body_names(env.envs[0], handle)
    props = env.gym.get_actor_rigid_body_properties(env.envs[0], handle)
    mass = np.asarray([prop.mass for prop in props], dtype=np.float64)
    center = np.asarray([[prop.com.x, prop.com.y, prop.com.z] for prop in props], dtype=np.float64)
    inertia = np.asarray(
        [
            [
                [prop.inertia.x.x, prop.inertia.y.x, prop.inertia.z.x],
                [prop.inertia.x.y, prop.inertia.y.y, prop.inertia.z.y],
                [prop.inertia.x.z, prop.inertia.y.z, prop.inertia.z.z],
            ]
            for prop in props
        ],
        dtype=np.float64,
    )
    return np.asarray(names), mass, center, inertia


def main():
    args = parse_args()
    isaac_root = Path(args.isaac_root).resolve()
    if str(isaac_root) not in sys.path:
        sys.path.insert(0, str(isaac_root))

    import isaacgym  # noqa: F401  Must precede torch.
    import torch
    import isaacgymenvs

    from mujoco_sim.ur5e_sim2sim import model_sha256
    from onnx_deploy.policy_runner import PolicyRunner

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env = isaacgymenvs.make(
        seed=args.seed,
        task=args.task,
        num_envs=1,
        sim_device=args.sim_device,
        rl_device=args.rl_device,
        graphics_device_id=0 if args.viewer else -1,
        headless=not args.viewer,
        force_render=args.viewer,
    )
    policy = PolicyRunner(ROOT / args.model, args.provider)
    policy.reset()
    env_ids = torch.zeros(1, dtype=torch.long, device=env.device)
    env.reset_idx(env_ids)
    env.set_actor_root_state_tensor_indexed()
    env.gym.simulate(env.sim)
    env.gym.fetch_results(env.sim, True)
    env.progress_buf[:] = 0
    env.compute_observations()
    env.obs_buf[:, -1] = 0.0
    env.clamp_obs(env.obs_buf)

    canonical_indices = env.arm_dof_indices + env.gripper_dof_indices
    joint_order = tuple(
        (
            "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
            "finger_joint", "left_inner_finger_joint", "left_inner_knuckle_joint",
            "right_outer_knuckle_joint", "right_inner_finger_joint", "right_inner_knuckle_joint",
        )
    )
    rows = {
        "observation": [_numpy(env.obs_buf[0])],
        "action": [],
        "joint_position": [],
        "joint_velocity": [],
        "joint_target": [],
        "palm_position": [],
        "palm_state": [],
        "fingertip_position": [],
        "object_state": [],
        "goal_position": [],
        "lifted": [],
        "success": [],
        "reward": [],
    }
    first = _capture(env, canonical_indices)
    for key, value in first.items():
        rows[key].append(value)

    command_velocity = np.zeros(6, dtype=np.float64)
    policy_period = float(env.dt * env.control_freq_inv)
    action_scale = float(env.hand_dof_speed_scale)
    for _ in range(args.steps):
        action = policy.infer(rows["observation"][-1])[0]
        desired_velocity = np.clip(
            action_scale * action[:6], -args.max_arm_velocity, args.max_arm_velocity
        )
        max_velocity_delta = args.max_arm_acceleration * policy_period
        command_velocity += np.clip(
            desired_velocity - command_velocity, -max_velocity_delta, max_velocity_delta
        )
        action[:6] = command_velocity / action_scale
        observations, _, done, _ = env.step(
            torch.as_tensor(action[None, :], dtype=torch.float32, device=env.rl_device)
        )
        rows["action"].append(action.copy())
        rows["observation"].append(_numpy(observations["obs"][0]))
        state = _capture(env, canonical_indices)
        for key, value in state.items():
            rows[key].append(value)
        if args.viewer:
            sleep(policy_period)
        if bool(done[0]):
            break

    if args.viewer and args.screenshot and getattr(env, "viewer", None) is not None:
        screenshot = ROOT / args.screenshot
        screenshot.parent.mkdir(parents=True, exist_ok=True)
        env.gym.write_viewer_image_to_file(env.viewer, str(screenshot))
    payload = {key: np.asarray(value) for key, value in rows.items()}
    payload["action_target"] = payload["joint_target"][1:].copy()
    for key in (
        "joint_position", "joint_velocity", "joint_target", "palm_position",
        "palm_state", "fingertip_position", "object_state", "goal_position",
    ):
        payload["initial_{}".format(key)] = payload[key][0].copy()
    payload.update(
        {
            "format_version": np.asarray(1, dtype=np.int64),
            "policy_period": np.asarray(env.dt * env.control_freq_inv, dtype=np.float64),
            "physics_timestep": np.asarray(env.dt / env.sim_params.substeps, dtype=np.float64),
            "object_scale": _numpy(env.object_scales[0]),
            "joint_order": np.asarray(joint_order),
            "quaternion_order": np.asarray("xyzw"),
            "model_sha256": np.asarray(model_sha256(ROOT / args.model)),
            "seed": np.asarray(args.seed, dtype=np.int64),
            "task": np.asarray(args.task),
        }
    )
    for prefix, actor_name in (("robot", "ur5e_robotiq"), ("object", "object")):
        names, mass, center, inertia = _rigid_body_parameters(env, actor_name)
        payload[prefix + "_body_names"] = names
        payload[prefix + "_body_mass"] = mass
        payload[prefix + "_body_com"] = center
        payload[prefix + "_body_inertia"] = inertia
    if payload["action"].shape[0]:
        if float(np.max(np.abs(payload["action"][:, :6]))) < 0.01:
            raise RuntimeError("Smoke policy arm action never reaches the required 0.01 magnitude")
        if float(np.ptp(payload["action"][:, 6])) <= 1e-6:
            raise RuntimeError("Smoke policy gripper action is constant")
        if not np.any(np.abs(np.diff(payload["joint_target"], axis=0)) > 1e-8):
            raise RuntimeError("Joint targets did not change")
    for key, value in payload.items():
        if value.dtype.kind not in "SU" and not np.isfinite(value).all():
            raise RuntimeError("Isaac trajectory {} contains NaN or Inf".format(key))
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(output), **payload)
    close = getattr(env, "close", None)
    if close is not None:
        close()
    print("steps={} output={} model_sha256={}".format(
        payload["action"].shape[0], output, str(payload["model_sha256"])
    ))


if __name__ == "__main__":
    main()
