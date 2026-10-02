"""Replay an Isaac Gym Allegro-Rokae trajectory in MuJoCo and report drift."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from mujoco_sim.rokae_adapter import RokaeAdapter, matrix_to_xyzw, wxyz_to_xyzw


def _capture(adapter):
    qpos = adapter.data.qpos[adapter.qpos_addr].copy()
    qvel = adapter.data.qvel[adapter.dof_addr].copy()
    palm_linear, palm_angular = adapter._velocity(
        mujoco.mjtObj.mjOBJ_BODY, adapter.palm_body
    )
    object_linear, object_angular = adapter._velocity(
        mujoco.mjtObj.mjOBJ_BODY, adapter.object_body
    )
    object_state = np.concatenate(
        (
            adapter.data.xpos[adapter.object_body],
            wxyz_to_xyzw(adapter.data.xquat[adapter.object_body]),
            object_linear,
            object_angular,
        )
    )
    palm_state = np.concatenate(
        (
            matrix_to_xyzw(adapter.data.site_xmat[adapter.palm_site]),
            palm_linear,
            palm_angular,
        )
    )
    return {
        "joint_position": qpos,
        "joint_velocity": qvel,
        "joint_target": adapter.previous_target.copy(),
        "palm_position": adapter.data.site_xpos[adapter.palm_site].copy(),
        "palm_state": palm_state,
        "fingertip_position": adapter.data.site_xpos[adapter.tip_sites].copy(),
        "object_state": object_state,
        "goal_position": adapter.data.site_xpos[adapter.goal_site].copy(),
        "lifted": np.asarray(adapter.lifted, dtype=np.bool_),
        "successes": np.asarray(adapter.successes, dtype=np.float64),
        "progress": np.asarray(adapter.progress, dtype=np.int64),
    }


def load_reference(path):
    with np.load(path, allow_pickle=False) as payload:
        reference = {key: payload[key] for key in payload.files}
    required = {
        "observation",
        "action",
        "action_target",
        "joint_position",
        "joint_velocity",
        "joint_target",
        "palm_position",
        "palm_state",
        "fingertip_position",
        "object_state",
        "goal_position",
        "object_scale",
        "object_initial_z",
        "joint_order",
        "policy_period",
    }
    missing = sorted(required.difference(reference))
    if missing:
        raise ValueError(f"Reference is missing fields: {missing}")
    steps = reference["action"].shape[0]
    expected_shapes = {
        "action": (steps, 23),
        "action_target": (steps, 23),
        "observation": (steps + 1, 99),
        "joint_position": (steps + 1, 23),
        "joint_velocity": (steps + 1, 23),
        "joint_target": (steps + 1, 23),
        "palm_position": (steps + 1, 3),
        "palm_state": (steps + 1, 10),
        "fingertip_position": (steps + 1, 4, 3),
        "object_state": (steps + 1, 13),
        "goal_position": (steps + 1, 3),
        "object_scale": (3,),
        "joint_order": (23,),
    }
    for key, shape in expected_shapes.items():
        if reference[key].shape != shape:
            raise ValueError(f"Reference {key} has shape {reference[key].shape}, expected {shape}")
    for key, value in reference.items():
        if value.dtype.kind not in "SUb" and not np.isfinite(value).all():
            raise ValueError(f"Reference {key} contains NaN or Inf")
    return reference


def reference_initial_state(reference):
    return {
        "joint_position": reference["joint_position"][0],
        "joint_velocity": reference["joint_velocity"][0],
        "joint_target": reference["joint_target"][0],
        "object_state": reference["object_state"][0],
        "goal_position": reference["goal_position"][0],
        "object_scale": reference["object_scale"],
        "object_initial_z": float(reference["object_initial_z"]),
        "progress": int(reference.get("progress", np.asarray([0]))[0]),
        "successes": float(reference.get("successes", np.asarray([0.0]))[0]),
    }


def replay(adapter, reference):
    steps = reference["action"].shape[0]
    initial = reference_initial_state(reference)
    initial_observation = adapter.reset(initial=initial)
    adapter.closest_keypoint = float(reference["observation"][0, 90])
    adapter.closest_fingertips = reference["observation"][0, 91:95].astype(np.float64).copy()
    adapter.lifted = bool(reference["observation"][0, 95])

    rows = {key: [] for key in _capture(adapter)}
    observations = [initial_observation]
    targets = []
    for key, value in _capture(adapter).items():
        rows[key].append(value)
    for index, action in enumerate(reference["action"]):
        targets.append(adapter.apply_action(action))
        adapter.step_physics_to((index + 1) * adapter.policy_period)
        observations.append(adapter.observe())
        for key, value in _capture(adapter).items():
            rows[key].append(value)
    result = {key: np.asarray(value) for key, value in rows.items()}
    result["observation"] = np.asarray(observations, dtype=np.float32)
    result["action"] = reference["action"].copy()
    result["action_target"] = np.asarray(targets, dtype=np.float64)
    result["sim_time"] = np.arange(steps + 1, dtype=np.float64) * adapter.policy_period
    result["physics_timestep"] = np.asarray(adapter.model.opt.timestep, dtype=np.float64)
    result["object_mass"] = np.asarray(adapter.model.body_mass[adapter.object_body], dtype=np.float64)
    result["object_friction"] = adapter.model.geom_friction[adapter.object_geom].copy()
    result["object_contact_margin"] = np.asarray(
        adapter.model.geom_margin[adapter.object_geom], dtype=np.float64
    )
    if "robot_body_names" in reference:
        result["robot_body_names"] = reference["robot_body_names"].copy()
        result["robot_body_mass"] = np.asarray(
            [adapter.model.body_mass[adapter._body_id(str(name))] for name in reference["robot_body_names"]],
            dtype=np.float64,
        )
        result["robot_body_inertia"] = np.asarray(
            [adapter.model.body_inertia[adapter._body_id(str(name))] for name in reference["robot_body_names"]],
            dtype=np.float64,
        )
    return result


def _error(reference, actual):
    delta = np.asarray(actual, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    return {
        "rmse": float(np.sqrt(np.mean(delta * delta))),
        "max_abs": float(np.max(np.abs(delta))),
    }


def _per_component_error(reference, actual, names):
    delta = np.asarray(actual, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    return {
        name: {
            "rmse": float(np.sqrt(np.mean(delta[..., index] ** 2))),
            "max_abs": float(np.max(np.abs(delta[..., index]))),
        }
        for index, name in enumerate(names)
    }


def _event_summary(values):
    values = np.asarray(values, dtype=bool).reshape(-1)
    transitions = np.flatnonzero(values & ~np.r_[False, values[:-1]])
    return {
        "count": int(transitions.size),
        "first_step": int(transitions[0]) if transitions.size else None,
        "steps": transitions.tolist(),
    }


def _increment_events(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return np.flatnonzero(np.diff(values, prepend=values[0]) > 0.0).tolist()


def _invalid_principal_inertias(matrices):
    eigenvalues = np.linalg.eigvalsh(np.asarray(matrices, dtype=np.float64))
    return int(np.count_nonzero(eigenvalues[:, 2] > eigenvalues[:, 0] + eigenvalues[:, 1] + 1e-12))


def _quaternion_angle(reference, actual):
    reference = reference / np.linalg.norm(reference, axis=-1, keepdims=True)
    actual = actual / np.linalg.norm(actual, axis=-1, keepdims=True)
    dot = np.clip(np.abs(np.sum(reference * actual, axis=-1)), 0.0, 1.0)
    angle = 2.0 * np.arccos(dot)
    return {"rmse_rad": float(np.sqrt(np.mean(angle * angle))), "max_rad": float(np.max(angle))}


def alignment_report(reference, actual):
    first_lift = np.flatnonzero(reference.get("lifted", np.zeros(1, dtype=bool)))
    prefix_frames = int(first_lift[0]) if first_lift.size else reference["joint_position"].shape[0]
    prefix_frames = max(prefix_frames, 1)
    joint_names = reference["joint_order"].tolist()
    tip_names = ["pinky", "middle", "index", "thumb"]
    isaac_success_steps = _increment_events(reference.get("successes", np.zeros(1)))
    mujoco_success_steps = _increment_events(actual.get("successes", np.zeros(1)))
    report = {
        "steps": int(reference["action"].shape[0]),
        "policy_period": {
            "isaac": float(reference["policy_period"]),
            "mujoco": 0.01667,
        },
        "joint_position": _error(reference["joint_position"], actual["joint_position"]),
        "joint_position_per_joint": _per_component_error(
            reference["joint_position"], actual["joint_position"], joint_names
        ),
        "joint_velocity": _error(reference["joint_velocity"], actual["joint_velocity"]),
        "joint_velocity_per_joint": _per_component_error(
            reference["joint_velocity"], actual["joint_velocity"], joint_names
        ),
        "action_target": _error(reference["action_target"], actual["action_target"]),
        "palm_position": _error(reference["palm_position"], actual["palm_position"]),
        "palm_quaternion": _quaternion_angle(reference["palm_state"][:, :4], actual["palm_state"][:, :4]),
        "fingertip_position": _error(reference["fingertip_position"], actual["fingertip_position"]),
        "fingertip_position_per_finger": {
            name: _error(reference["fingertip_position"][:, index], actual["fingertip_position"][:, index])
            for index, name in enumerate(tip_names)
        },
        "object_position": _error(reference["object_state"][:, :3], actual["object_state"][:, :3]),
        "object_quaternion": _quaternion_angle(reference["object_state"][:, 3:7], actual["object_state"][:, 3:7]),
        "object_linear_velocity": _error(reference["object_state"][:, 7:10], actual["object_state"][:, 7:10]),
        "object_angular_velocity": _error(reference["object_state"][:, 10:13], actual["object_state"][:, 10:13]),
        "observation": _error(reference["observation"], actual["observation"]),
        "initial": {
            "palm_position": _error(reference["palm_position"][0], actual["palm_position"][0]),
            "fingertip_position": _error(reference["fingertip_position"][0], actual["fingertip_position"][0]),
            "object_position": _error(reference["object_state"][0, :3], actual["object_state"][0, :3]),
        },
        "pre_first_isaac_lift": {
            "frames": prefix_frames,
            "joint_position": _error(
                reference["joint_position"][:prefix_frames], actual["joint_position"][:prefix_frames]
            ),
            "joint_velocity": _error(
                reference["joint_velocity"][:prefix_frames], actual["joint_velocity"][:prefix_frames]
            ),
            "palm_position": _error(
                reference["palm_position"][:prefix_frames], actual["palm_position"][:prefix_frames]
            ),
            "fingertip_position": _error(
                reference["fingertip_position"][:prefix_frames],
                actual["fingertip_position"][:prefix_frames],
            ),
            "object_position": _error(
                reference["object_state"][:prefix_frames, :3],
                actual["object_state"][:prefix_frames, :3],
            ),
            "observation": _error(
                reference["observation"][:prefix_frames], actual["observation"][:prefix_frames]
            ),
        },
        "events": {
            "isaac_lift": _event_summary(reference.get("lifted", np.zeros(1, dtype=bool))),
            "mujoco_lift": _event_summary(actual["lifted"]),
            "isaac_success_steps": isaac_success_steps,
            "mujoco_success_steps": mujoco_success_steps,
            "success_step_differences": [
                int(mujoco - isaac)
                for isaac, mujoco in zip(isaac_success_steps, mujoco_success_steps)
            ],
        },
    }
    if "robot_body_inertia" in reference:
        isaac_eigenvalues = np.linalg.eigvalsh(reference["robot_body_inertia"])
        mujoco_inertia = actual["robot_body_inertia"]
        report["physics_contract"] = {
            "isaac_physics_timestep": float(reference["physics_timestep"]),
            "mujoco_physics_timestep": float(actual["physics_timestep"]),
            "isaac_object_mass": float(reference["object_mass"]),
            "mujoco_object_mass": float(actual["object_mass"]),
            "object_mass_abs_error": float(abs(reference["object_mass"] - actual["object_mass"])),
            "isaac_object_friction": float(reference["object_shape_friction"][0]),
            "mujoco_object_sliding_friction": float(actual["object_friction"][0]),
            "isaac_contact_offset": float(reference["object_shape_contact_offset"][0]),
            "mujoco_contact_margin": float(actual["object_contact_margin"]),
            "isaac_invalid_inertia_body_count": _invalid_principal_inertias(
                reference["robot_body_inertia"]
            ),
            "mujoco_invalid_inertia_body_count": int(
                np.count_nonzero(
                    mujoco_inertia[:, 2]
                    > mujoco_inertia[:, 0] + mujoco_inertia[:, 1] + 1e-12
                )
            ),
            "robot_principal_inertia": _error(isaac_eigenvalues, mujoco_inertia),
        }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--scene", default="resources/assets/scenes/allegro_rokae.xml")
    parser.add_argument(
        "--metadata",
        default="resources/models/rokae_allegro/successful_best_seed0/rank01_maxsucc13/policy.meta.yaml",
    )
    parser.add_argument(
        "--npz", default="resources/trajectories/rokae_successful_rank01/mujoco_replay.npz"
    )
    parser.add_argument(
        "--report", default="resources/trajectories/rokae_successful_rank01/alignment_report.json"
    )
    args = parser.parse_args()

    reference = load_reference(args.reference)
    adapter = RokaeAdapter(args.scene, args.metadata, seed=int(reference.get("seed", 0)))
    if tuple(reference["joint_order"].tolist()) != tuple(adapter.joint_names):
        raise ValueError("Isaac reference joint order does not match policy metadata")
    actual = replay(adapter, reference)
    for key, value in actual.items():
        if value.dtype.kind not in "SUb" and not np.isfinite(value).all():
            raise RuntimeError(f"MuJoCo replay {key} contains NaN or Inf")
    report = alignment_report(reference, actual)
    npz_path = Path(args.npz)
    report_path = Path(args.report)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(npz_path), **actual)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
