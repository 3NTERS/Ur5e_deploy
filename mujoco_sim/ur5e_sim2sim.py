"""Trajectory recording, validation, replay, and numeric alignment reporting."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


STATE_KEYS = (
    "joint_position", "joint_velocity", "joint_target", "palm_position",
    "palm_state", "fingertip_position", "object_state", "goal_position",
)


def model_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _append_state(rows, state):
    for key in STATE_KEYS:
        rows[key].append(np.asarray(state[key]).copy())


def run_trajectory(adapter, policy=None, actions=None, steps=600, viewer=None, initial=None):
    if (policy is None) == (actions is None):
        raise ValueError("Supply exactly one of policy or actions")
    if policy is not None:
        policy.reset()
    observation = adapter.reset(initial)
    rows = {key: [] for key in STATE_KEYS}
    rows.update({"observation": [observation.copy()], "action": [], "sim_time": [adapter.data.time]})
    _append_state(rows, adapter.snapshot())
    count = int(steps) if actions is None else len(actions)
    for step in range(count):
        if adapter.done:
            break
        action = policy.infer(observation)[0] if actions is None else actions[step]
        adapter.apply_action(action)
        adapter.step_physics_to((step + 1) * adapter.policy_period)
        observation = adapter.observe()
        rows["action"].append(np.asarray(action).copy())
        rows["observation"].append(observation.copy())
        rows["sim_time"].append(adapter.data.time)
        _append_state(rows, adapter.snapshot())
        if viewer is not None:
            viewer.sync()
    result = {key: np.asarray(value) for key, value in rows.items()}
    result["action_target"] = result["joint_target"][1:].copy()
    for key in STATE_KEYS:
        result["initial_{}".format(key)] = result[key][0].copy()
    result.update(
        {
            "format_version": np.asarray(1, dtype=np.int64),
            "policy_period": np.asarray(adapter.policy_period, dtype=np.float64),
            "physics_timestep": np.asarray(adapter.model.opt.timestep, dtype=np.float64),
            "object_scale": adapter.object_scale.astype(np.float64),
            "joint_order": np.asarray(adapter.joint_names),
            "quaternion_order": np.asarray("xyzw"),
        }
    )
    return result


def save_npz(path, trajectory, policy_model=None):
    payload = dict(trajectory)
    payload["model_sha256"] = np.asarray(model_sha256(policy_model) if policy_model else "replay")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(path), **payload)


def save_csv(path, trajectory):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    actions = trajectory["action"]
    targets = trajectory["action_target"]
    joints = trajectory["joint_position"][1:len(actions) + 1]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["step", "sim_time"]
            + ["action_{}".format(index) for index in range(7)]
            + ["target_{}".format(index) for index in range(12)]
            + ["q_{}".format(index) for index in range(12)]
        )
        for index in range(len(actions)):
            writer.writerow(
                [index, trajectory["sim_time"][index + 1]]
                + actions[index].tolist() + targets[index].tolist() + joints[index].tolist()
            )


def load_reference(path):
    with np.load(str(path), allow_pickle=False) as payload:
        reference = {key: payload[key] for key in payload.files}
    required = set(
        STATE_KEYS
        + ("observation", "action", "action_target", "policy_period", "object_scale",
           "format_version", "joint_order", "quaternion_order", "model_sha256")
        + tuple("initial_{}".format(key) for key in STATE_KEYS)
    )
    missing = sorted(required.difference(reference))
    if missing:
        raise ValueError("Reference is missing arrays: {}".format(missing))
    steps = reference["action"].shape[0]
    expected = {
        "action": (steps, 7),
        "action_target": (steps, 12),
        "observation": (steps + 1, 69),
        "joint_position": (steps + 1, 12),
        "joint_velocity": (steps + 1, 12),
        "joint_target": (steps + 1, 12),
        "palm_position": (steps + 1, 3),
        "palm_state": (steps + 1, 10),
        "fingertip_position": (steps + 1, 2, 3),
        "object_state": (steps + 1, 13),
        "goal_position": (steps + 1, 3),
        "object_scale": (3,),
        "joint_order": (12,),
    }
    for key in STATE_KEYS:
        expected["initial_{}".format(key)] = expected[key][1:]
    for key, shape in expected.items():
        if reference[key].shape != shape:
            raise ValueError("Reference {} has shape {}, expected {}".format(key, reference[key].shape, shape))
        if reference[key].dtype.kind not in "SU" and not np.isfinite(reference[key]).all():
            raise ValueError("Reference {} contains NaN or Inf".format(key))
    if not np.isfinite(float(reference["policy_period"])):
        raise ValueError("Reference policy period is not finite")
    if int(reference["format_version"]) != 1:
        raise ValueError("Unsupported reference format version {}".format(reference["format_version"]))
    if str(reference["quaternion_order"]) != "xyzw":
        raise ValueError("Reference quaternion order must be xyzw")
    return reference


def initial_from_reference(reference):
    return {
        "joint_position": reference["initial_joint_position"],
        "joint_velocity": reference["initial_joint_velocity"],
        "joint_target": reference["initial_joint_target"],
        "object_state": reference["initial_object_state"],
        "goal_position": reference["initial_goal_position"],
    }


def _error(reference, actual):
    difference = np.asarray(actual) - np.asarray(reference)
    if not np.isfinite(difference).all():
        raise ValueError("Alignment result contains NaN or Inf")
    return {
        "rmse": float(np.sqrt(np.mean(np.square(difference)))),
        "max_abs": float(np.max(np.abs(difference))),
    }


def _named_errors(reference, actual, names):
    return {name: _error(reference[..., index], actual[..., index]) for index, name in enumerate(names)}


def _quaternion_angle(reference, actual):
    reference = reference / np.linalg.norm(reference, axis=-1, keepdims=True)
    actual = actual / np.linalg.norm(actual, axis=-1, keepdims=True)
    dot = np.abs(np.sum(reference * actual, axis=-1))
    return 2.0 * np.arccos(np.clip(dot, 0.0, 1.0))


def _first_event(condition):
    indices = np.flatnonzero(condition)
    return None if indices.size == 0 else int(indices[0])


def _events(trajectory):
    position = trajectory["object_state"][:, :3]
    lift = _first_event(position[:, 2] - position[0, 2] > 0.10)
    master_target = trajectory["joint_target"][:, 6]
    if lift is None:
        release = None
    else:
        release_condition = master_target < 0.144
        release_condition[:lift + 1] = False
        release = _first_event(release_condition)
    distance = np.linalg.norm(position - trajectory["goal_position"], axis=1)
    return {"lift": lift, "release": release, "goal_range": _first_event(distance <= 0.1125)}


def alignment_report(reference, actual, joint_names):
    for key in STATE_KEYS + ("observation", "action", "action_target"):
        if reference[key].shape != actual[key].shape:
            raise ValueError("Trajectory shape mismatch for {}: {} vs {}".format(
                key, reference[key].shape, actual[key].shape
            ))
    ref_events, actual_events = _events(reference), _events(actual)
    events = {}
    for name in ref_events:
        ref_step, actual_step = ref_events[name], actual_events[name]
        events[name] = {
            "isaac_step": ref_step,
            "mujoco_step": actual_step,
            "step_delta": None if ref_step is None or actual_step is None else actual_step - ref_step,
        }
    object_angle = _quaternion_angle(reference["object_state"][:, 3:7], actual["object_state"][:, 3:7])
    palm_angle = _quaternion_angle(reference["palm_state"][:, :4], actual["palm_state"][:, :4])
    report = {
        "format_version": 1,
        "steps": int(reference["action"].shape[0]),
        "policy_period": {
            "isaac": float(reference["policy_period"]),
            "mujoco": float(actual["policy_period"]),
        },
        "joint_position": _named_errors(reference["joint_position"], actual["joint_position"], joint_names),
        "joint_velocity": _named_errors(reference["joint_velocity"], actual["joint_velocity"], joint_names),
        "action_target": _named_errors(reference["action_target"], actual["action_target"], joint_names),
        "palm": {
            "position": _error(reference["palm_position"], actual["palm_position"]),
            "quaternion_angle_rad": _error(np.zeros_like(palm_angle), palm_angle),
        },
        "fingertips": {
            "left_position": _error(reference["fingertip_position"][:, 0], actual["fingertip_position"][:, 0]),
            "right_position": _error(reference["fingertip_position"][:, 1], actual["fingertip_position"][:, 1]),
        },
        "object": {
            "position": _error(reference["object_state"][:, :3], actual["object_state"][:, :3]),
            "quaternion_angle_rad": _error(np.zeros_like(object_angle), object_angle),
            "linear_velocity": _error(reference["object_state"][:, 7:10], actual["object_state"][:, 7:10]),
            "angular_velocity": _error(reference["object_state"][:, 10:13], actual["object_state"][:, 10:13]),
        },
        "observation": _error(reference["observation"], actual["observation"]),
        "events": events,
    }
    json.dumps(report, allow_nan=False)
    return report


def save_report(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
