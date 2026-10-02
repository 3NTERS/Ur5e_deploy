#!/usr/bin/env python3
"""Replay the UR5e lift/gripper ONNX policy in MuJoCo."""

import argparse
import json
import math
import time
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort
import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = REPOSITORY_ROOT / "resources/assets/robots/ur5e_robotiq_2f85/ur5e_lift_gripper.xml"
DEFAULT_ONNX = REPOSITORY_ROOT / "resources/models/ur5e_lift_gripper/ur5e_lift_gripper.onnx"
DEFAULT_METADATA = REPOSITORY_ROOT / "resources/models/ur5e_lift_gripper/ur5e_lift_gripper.meta.yaml"
CONTROL_DT = 0.01667


def object_id(model, object_type, name):
    result = mujoco.mj_name2id(model, object_type, name)
    if result < 0:
        raise KeyError("MuJoCo object not found: {}".format(name))
    return result


def joint_addresses(model, names):
    qpos = []
    qvel = []
    for name in names:
        joint_id = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        qpos.append(int(model.jnt_qposadr[joint_id]))
        qvel.append(int(model.jnt_dofadr[joint_id]))
    return np.asarray(qpos), np.asarray(qvel)


def body_pose_velocity(model, data, body_name):
    body_id = object_id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    quaternion_wxyz = data.xquat[body_id]
    quaternion_xyzw = quaternion_wxyz[[1, 2, 3, 0]]
    velocity = np.zeros(6, dtype=np.float64)
    mujoco.mj_objectVelocity(
        model, data, mujoco.mjtObj.mjOBJ_BODY, body_id, velocity, 0
    )
    # mj_objectVelocity returns angular followed by linear velocity.
    return (
        data.xpos[body_id].copy(),
        quaternion_xyzw.copy(),
        velocity[3:6].copy(),
        velocity[0:3].copy(),
    )


def build_observation(model, data, metadata, qpos_indices, qvel_indices, progress):
    observation = np.zeros((1, 69), dtype=np.float32)
    qpos = data.qpos[qpos_indices]
    qvel = data.qvel[qvel_indices]
    limits = metadata["joint_limits"]
    lower = np.asarray([item["lower"] for item in limits], dtype=np.float64)
    upper = np.asarray([item["upper"] for item in limits], dtype=np.float64)
    observation[0, 0:12] = (2.0 * (qpos - lower) / (upper - lower) - 1.0).astype(np.float32)
    observation[0, 12:24] = qvel.astype(np.float32)

    _palm_position, palm_quaternion, palm_linear, palm_angular = body_pose_velocity(
        model, data, "robotiq_arg2f_base_link"
    )
    pinch_site = object_id(model, mujoco.mjtObj.mjOBJ_SITE, "robotiq_pinch")
    palm_center = data.site_xpos[pinch_site]
    observation[0, 24:27] = palm_center.astype(np.float32)
    observation[0, 27:37] = np.concatenate(
        (palm_quaternion, palm_linear, palm_angular)
    ).astype(np.float32)

    left_pad = object_id(model, mujoco.mjtObj.mjOBJ_BODY, "robotiq_left_inner_finger_pad")
    right_pad = object_id(model, mujoco.mjtObj.mjOBJ_BODY, "robotiq_right_inner_finger_pad")
    observation[0, 47:50] = (data.xpos[left_pad] - palm_center).astype(np.float32)
    observation[0, 50:53] = (data.xpos[right_pad] - palm_center).astype(np.float32)
    observation[0, 66] = np.float32(math.log(progress / 10.0 + 1.0))
    # Object, keypoint, grasp state, success count, and reward channels remain zero.
    return observation


def phase_name(step, metadata):
    for stage in metadata["sequence"]["stages"]:
        if stage["start"] <= step < stage["end_exclusive"]:
            return stage["name"]
    raise ValueError("Step {} is outside the configured episode".format(step))


def run(args):
    with Path(args.metadata).open("r", encoding="utf-8") as stream:
        metadata = yaml.safe_load(stream)
    model = mujoco.MjModel.from_xml_path(str(Path(args.model).resolve()))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    joint_names = metadata["joint_order"]
    qpos_indices, qvel_indices = joint_addresses(model, joint_names)
    actuator_names = [
        "shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"
    ]
    arm_actuators = [
        object_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in actuator_names
    ]
    gripper_actuator = object_id(
        model, mujoco.mjtObj.mjOBJ_ACTUATOR, "fingers_actuator"
    )
    session = ort.InferenceSession(
        str(Path(args.onnx).resolve()), providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    arm_target = data.qpos[qpos_indices[:6]].copy()
    arm_lower = np.asarray(
        [item["lower"] for item in metadata["joint_limits"][:6]], dtype=np.float64
    )
    arm_upper = np.asarray(
        [item["upper"] for item in metadata["joint_limits"][:6]], dtype=np.float64
    )
    master_upper = float(metadata["joint_limits"][6]["upper"])
    raw_master_upper = float(model.jnt_range[object_id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint"
    ), 1])

    pinch_site = object_id(model, mujoco.mjtObj.mjOBJ_SITE, "robotiq_pinch")
    initial_tcp_z = float(data.site_xpos[pinch_site, 2])
    max_tcp_lift = 0.0
    close_best_error = float("inf")
    final_open_best_error = float("inf")
    final_success_steps = 0
    raised_target = np.asarray(metadata["targets"]["raised_arm_rad"], dtype=np.float64)
    arm_tolerance = float(metadata["targets"]["arm_tolerance_rad"])
    gripper_tolerance = float(metadata["targets"]["gripper_tolerance_rad"])
    previous_phase = None

    viewer = None
    if not args.headless:
        from mujoco import viewer as mj_viewer
        viewer = mj_viewer.launch_passive(model, data)
        viewer.cam.distance = 1.8
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -20
        viewer.cam.lookat[:] = np.array([0.0, 0.0, 0.55])

    try:
        for step in range(int(metadata["sequence"]["episode_steps"])):
            wall_start = time.time()
            current_phase = phase_name(step, metadata)
            if current_phase != previous_phase:
                print("step={:03d} phase={}".format(step, current_phase))
                previous_phase = current_phase

            observation = build_observation(
                model, data, metadata, qpos_indices, qvel_indices, step
            )
            action = session.run([output_name], {input_name: observation})[0][0]
            action = np.clip(action, -1.0, 1.0)
            arm_target = np.clip(
                arm_target + CONTROL_DT * action[:6], arm_lower, arm_upper
            )
            master_target = 0.5 * (float(action[6]) + 1.0) * master_upper
            data.ctrl[arm_actuators] = arm_target
            data.ctrl[gripper_actuator] = master_target / raw_master_upper * 255.0

            control_end_time = (step + 1) * CONTROL_DT
            while data.time + 1e-12 < control_end_time:
                mujoco.mj_step(model, data)
            if viewer is not None:
                viewer.sync()
            if args.realtime:
                elapsed = time.time() - wall_start
                if elapsed < CONTROL_DT:
                    time.sleep(CONTROL_DT - elapsed)

            tcp_lift = float(data.site_xpos[pinch_site, 2]) - initial_tcp_z
            max_tcp_lift = max(max_tcp_lift, tcp_lift)
            master_position = float(data.qpos[qpos_indices[6]])
            if 30 <= step < 195:
                close_best_error = min(close_best_error, abs(master_position - master_upper))
            if step >= 195:
                final_open_best_error = min(final_open_best_error, abs(master_position))
                arm_error = float(np.max(np.abs(data.qpos[qpos_indices[:6]] - raised_target)))
                if arm_error <= arm_tolerance and abs(master_position) <= gripper_tolerance:
                    final_success_steps += 1
                else:
                    final_success_steps = 0
    finally:
        if viewer is not None:
            viewer.close()

    final_arm_error = float(np.max(np.abs(data.qpos[qpos_indices[:6]] - raised_target)))
    final_open_error = abs(float(data.qpos[qpos_indices[6]]))
    report = {
        "simulator": "mujoco",
        "steps": int(metadata["sequence"]["episode_steps"]),
        "success": bool(final_success_steps >= int(metadata["targets"]["consecutive_success_steps"])),
        "final_success_steps": int(final_success_steps),
        "close_best_error_rad": close_best_error,
        "final_open_best_error_rad": final_open_best_error,
        "final_open_error_rad": final_open_error,
        "final_arm_max_error_rad": final_arm_error,
        "max_tcp_lift_m": max_tcp_lift,
    }
    print("MUJOCO_LIFT_GRIPPER " + json.dumps(report, sort_keys=True))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--onnx", default=str(DEFAULT_ONNX))
    parser.add_argument("--metadata", default=str(DEFAULT_METADATA))
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--realtime", action="store_true", help="pace the replay at 60 Hz")
    args = parser.parse_args()
    report = run(args)
    if not report["success"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
