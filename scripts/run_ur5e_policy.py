#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from onnx_deploy.policy_runner import PolicyRunner
from ur5e_comm.deployment import DeploymentSession
from ur5e_comm.geometry import load_eye_on_base
from ur5e_comm.observation import (
    JOINT_ORDER,
    MujocoStateProjector,
    SafetyMonitor,
    Ur5eActionMapper,
    Ur5eObservationBuilder,
)
from ur5e_comm.robot import UR5eHardware
from ur5e_comm.vision import RealSenseCamera, YoloInitialObjectLocator


ROOT = Path(__file__).resolve().parents[1]


def resolve_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def validate_contract(metadata_path, policy):
    metadata = yaml.safe_load(resolve_path(metadata_path).read_text(encoding="utf-8"))
    io = metadata["io"]
    if int(io["observation_dim"]) != policy.observation_dim:
        raise RuntimeError("ONNX observation dimension does not match policy.meta.yaml")
    if int(io["action_dim"]) != policy.action_dim:
        raise RuntimeError("ONNX action dimension does not match policy.meta.yaml")
    if tuple(io["joint_order"]) != JOINT_ORDER:
        raise RuntimeError("policy.meta.yaml joint order does not match deployment code")


def main():
    parser = argparse.ArgumentParser(description="Deploy a Ur5eRobotiq ONNX policy with initial YOLO state")
    parser.add_argument("--config", default="resources/config/ur5e_deploy.yaml")
    parser.add_argument("--provider", choices=("cuda", "cpu"))
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--execute", action="store_true", help="enable real motion (also requires config allow_motion)")
    args = parser.parse_args()
    config = yaml.safe_load(resolve_path(args.config).read_text(encoding="utf-8"))
    policy_cfg = config["policy"]
    safety_cfg = config["safety"]
    if args.execute and not bool(safety_cfg.get("allow_motion", False)):
        raise RuntimeError("Refusing motion: set safety.allow_motion=true as well as passing --execute")
    if args.execute:
        confirmation = input("即将向真实 UR5e/Robotiq 发送动作。确认安全区域无人后输入 ARM：").strip()
        if confirmation != "ARM":
            raise RuntimeError("Motion arming cancelled")

    policy = PolicyRunner(resolve_path(policy_cfg["model"]), args.provider or policy_cfg["provider"])
    validate_contract(policy_cfg["metadata"], policy)
    period = float(policy_cfg["period"])
    camera_cfg = config["camera"]
    vision_cfg = config["vision"]
    robot_cfg = config["robot"]
    task_cfg = config["task"]
    base_to_camera = load_eye_on_base(resolve_path(camera_cfg["calibration"]))
    locator = YoloInitialObjectLocator(
        resolve_path(vision_cfg["weights"]),
        base_to_camera,
        vision_cfg.get("target_class"),
        vision_cfg["confidence"],
        vision_cfg["depth_radius"],
        vision_cfg["depth_min"],
        vision_cfg["depth_max"],
        vision_cfg.get("device"),
    )
    projector = MujocoStateProjector(
        ROOT / "resources/assets/robots/ur5e_robotiq_2f85/ur5e_robotiq_2f85.xml",
        task_cfg["palm_center_offset"],
    )
    observation = Ur5eObservationBuilder(
        projector,
        task_cfg["goal_position"],
        task_cfg["object_quaternion_xyzw"],
        task_cfg["object_scale"],
    )
    action_mapper = Ur5eActionMapper(
        period,
        10.0,
        safety_cfg["max_arm_step"],
        safety_cfg["arm_lower"],
        safety_cfg["arm_upper"],
    )
    safety = SafetyMonitor(
        safety_cfg["arm_lower"],
        safety_cfg["arm_upper"],
        safety_cfg["max_joint_velocity"],
        safety_cfg["max_target_error"],
        safety_cfg["max_state_age"],
        safety_cfg["workspace_min"],
        safety_cfg["workspace_max"],
        safety_cfg["object_position_min"],
        safety_cfg["object_position_max"],
    )
    steps = int(args.steps or policy_cfg["episode_steps"])
    output = resolve_path(config["output"]["directory"])
    print("模式：{}；每回合只在开始时锁定一次物体位置。".format("实机执行" if args.execute else "只读 dry-run"))
    with RealSenseCamera(
        camera_cfg["width"], camera_cfg["height"], camera_cfg["fps"], camera_cfg.get("serial")
    ) as camera, UR5eHardware(
        robot_cfg["host"],
        robot_cfg["gripper_port"],
        servo_period=robot_cfg["servo_period"],
        servo_speed=robot_cfg["servo_speed"],
        servo_acceleration=robot_cfg["servo_acceleration"],
        servo_lookahead=robot_cfg["servo_lookahead"],
        servo_gain=robot_cfg["servo_gain"],
        gripper_speed=robot_cfg["gripper_speed"],
        gripper_force=robot_cfg["gripper_force"],
        allow_motion=args.execute,
        activate_gripper=args.execute,
    ) as robot:
        session = DeploymentSession(
            camera,
            locator,
            robot,
            policy,
            observation,
            action_mapper,
            safety,
            output,
            args.execute,
            period,
            safety_cfg["max_policy_lag"],
        )
        for index in range(args.episodes):
            print("开始回合 {}/{}".format(index + 1, args.episodes))
            result = session.run_episode(steps, int(vision_cfg["detection_attempts"]))
            print("回合 {}：verdict={}，steps={}，轨迹={}".format(
                result.episode_id, result.verdict, result.steps, result.trajectory_path
            ))


if __name__ == "__main__":
    main()
