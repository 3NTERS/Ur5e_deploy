#!/usr/bin/env python3
"""Collect a deterministic single-environment Allegro-Rokae Isaac Gym trajectory."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isaac-root", default="/home/liang/Workspace/Isaacgym_cjlu/IsaacGymEnvs")
    parser.add_argument(
        "--model",
        default="resources/models/rokae_allegro/successful_best_seed0/rank01_maxsucc13/policy.onnx",
    )
    parser.add_argument("--provider", choices=("cuda", "cpu"), default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--sim-device", default="cuda:0")
    parser.add_argument("--rl-device", default="cuda:0")
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="open the Isaac Gym viewer while collecting the reference trajectory",
    )
    parser.add_argument(
        "--output",
        default="resources/trajectories/rokae_successful_rank01/isaac_seed0_reference.npz",
    )
    return parser.parse_args()


def _numpy(tensor):
    return tensor.detach().cpu().numpy().copy()


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _capture(env):
    return {
        "joint_position": _numpy(env.arm_hand_dof_pos[0]),
        "joint_velocity": _numpy(env.arm_hand_dof_vel[0]),
        "joint_target": _numpy(env.cur_targets[0]),
        "palm_position": _numpy(env.palm_center_pos[0]),
        "palm_state": _numpy(env._palm_state[0, 3:13]),
        "fingertip_position": _numpy(env.fingertip_pos_offset[0]),
        "object_state": _numpy(env.object_state[0]),
        "goal_position": _numpy(env.goal_pos[0]),
        "lifted": np.asarray(bool(env.lifted_object[0]), dtype=np.bool_),
        "successes": np.asarray(float(env.successes[0]), dtype=np.float32),
        "progress": np.asarray(int(env.progress_buf[0]), dtype=np.int64),
    }


def _shape_properties(env, actor_handle):
    properties = env.gym.get_actor_rigid_shape_properties(env.envs[0], actor_handle)
    return {
        "friction": np.asarray([value.friction for value in properties], dtype=np.float64),
        "restitution": np.asarray([value.restitution for value in properties], dtype=np.float64),
        "rolling_friction": np.asarray(
            [value.rolling_friction for value in properties], dtype=np.float64
        ),
        "torsion_friction": np.asarray(
            [value.torsion_friction for value in properties], dtype=np.float64
        ),
        "contact_offset": np.asarray(
            [value.contact_offset for value in properties], dtype=np.float64
        ),
        "rest_offset": np.asarray([value.rest_offset for value in properties], dtype=np.float64),
    }


def _inertia_matrix(value):
    return [
        [value.x.x, value.y.x, value.z.x],
        [value.x.y, value.y.y, value.z.y],
        [value.x.z, value.y.z, value.z.z],
    ]


def main():
    args = parse_args()
    isaac_root = Path(args.isaac_root).resolve()
    sys.path.insert(0, str(isaac_root))
    sys.path.insert(0, str(ROOT))

    import isaacgym  # noqa: F401  Must precede torch.
    import torch
    import isaacgymenvs

    from onnx_deploy.policy_runner import PolicyRunner

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env = isaacgymenvs.make(
        seed=args.seed,
        task="AllegroRokae",
        num_envs=1,
        sim_device=args.sim_device,
        rl_device=args.rl_device,
        graphics_device_id=0 if args.viewer else -1,
        headless=not args.viewer,
        force_render=args.viewer,
    )
    env.force_scale = 0.0
    env.rb_forces.zero_()
    policy_path = (ROOT / args.model).resolve()
    policy = PolicyRunner(policy_path, args.provider)
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

    joint_order = tuple(env.gym.get_actor_dof_names(env.envs[0], env.allegro_hands[0]))
    object_handle = env.gym.find_actor_handle(env.envs[0], "object")
    table_handle = env.gym.find_actor_handle(env.envs[0], "table_object")
    object_body = env.gym.get_actor_rigid_body_properties(env.envs[0], object_handle)[0]
    robot_bodies = env.gym.get_actor_rigid_body_properties(env.envs[0], env.allegro_hands[0])
    robot_body_names = env.gym.get_actor_rigid_body_names(env.envs[0], env.allegro_hands[0])
    object_shapes = _shape_properties(env, object_handle)
    robot_shapes = _shape_properties(env, env.allegro_hands[0])
    table_shapes = _shape_properties(env, table_handle)
    rows = {
        "observation": [_numpy(env.obs_buf[0])],
        "action": [],
        "reward": [],
        "joint_position": [],
        "joint_velocity": [],
        "joint_target": [],
        "palm_position": [],
        "palm_state": [],
        "fingertip_position": [],
        "object_state": [],
        "goal_position": [],
        "lifted": [],
        "successes": [],
        "progress": [],
    }
    initial = _capture(env)
    for key, value in initial.items():
        rows[key].append(value)

    for _ in range(args.steps):
        action = policy.infer(rows["observation"][-1])[0]
        observations, rewards, done, _ = env.step(
            torch.as_tensor(action[None, :], dtype=torch.float32, device=env.rl_device)
        )
        rows["action"].append(action.copy())
        rows["reward"].append(float(rewards[0]))
        rows["observation"].append(_numpy(observations["obs"][0]))
        state = _capture(env)
        for key, value in state.items():
            rows[key].append(value)
        if bool(done[0]):
            break

    payload = {key: np.asarray(value) for key, value in rows.items()}
    payload["action_target"] = payload["joint_target"][1:].copy()
    payload.update(
        {
            "format_version": np.asarray(1, dtype=np.int64),
            "policy_period": np.asarray(env.dt * env.control_freq_inv, dtype=np.float64),
            "physics_timestep": np.asarray(env.dt / env.sim_params.substeps, dtype=np.float64),
            "object_base_size": np.asarray(env.object_base_size, dtype=np.float64),
            "object_density": np.asarray(400.0, dtype=np.float64),
            "object_scale": _numpy(env.object_scales[0]),
            "object_initial_z": np.asarray(float(env.object_init_state[0, 2]), dtype=np.float64),
            "joint_order": np.asarray(joint_order),
            "quaternion_order": np.asarray("xyzw"),
            "model_sha256": np.asarray(_sha256(policy_path)),
            "seed": np.asarray(args.seed, dtype=np.int64),
            "external_force_scale": np.asarray(0.0, dtype=np.float64),
            "object_mass": np.asarray(object_body.mass, dtype=np.float64),
            "object_inertia": np.asarray(_inertia_matrix(object_body.inertia), dtype=np.float64),
            "robot_body_names": np.asarray(robot_body_names),
            "robot_body_mass": np.asarray([body.mass for body in robot_bodies], dtype=np.float64),
            "robot_body_inertia": np.asarray(
                [_inertia_matrix(body.inertia) for body in robot_bodies], dtype=np.float64
            ),
        }
    )
    for prefix, properties in (
        ("object_shape", object_shapes),
        ("robot_shape", robot_shapes),
        ("table_shape", table_shapes),
    ):
        for name, value in properties.items():
            payload[f"{prefix}_{name}"] = value
    for key, value in payload.items():
        if value.dtype.kind not in "SUb" and not np.isfinite(value).all():
            raise RuntimeError(f"Isaac trajectory {key} contains NaN or Inf")
    output = (ROOT / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(output), **payload)
    close = getattr(env, "close", None)
    if close is not None:
        close()
    print(
        "steps={} output={} object_scale={} model_sha256={}".format(
            payload["action"].shape[0], output, payload["object_scale"].tolist(), payload["model_sha256"]
        )
    )


if __name__ == "__main__":
    main()
