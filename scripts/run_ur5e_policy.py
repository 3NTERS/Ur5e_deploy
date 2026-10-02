#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from onnx_deploy.policy_runner import PolicyRunner
from ur5e_comm.deployment import DeploymentSession, request_smoke_verdict, request_verdict
from ur5e_comm.geometry import load_eye_on_base, load_eye_on_hand
from ur5e_comm.observation import (
    JOINT_ORDER,
    MujocoStateProjector,
    SafetyMonitor,
    Ur5eActionMapper,
    Ur5eObservationBuilder,
)
from ur5e_comm.robot import UR5eHardware
from ur5e_comm.vision import RealSenseCamera, YoloEyeOnBaseLocator, YoloInitialObjectLocator


def resolve_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_contract(metadata_path, policy, model_path=None):
    metadata = yaml.safe_load(resolve_path(metadata_path).read_text(encoding="utf-8"))
    io = metadata["io"]
    if int(io["observation_dim"]) != policy.observation_dim:
        raise RuntimeError("ONNX observation dimension does not match policy.meta.yaml")
    if int(io["action_dim"]) != policy.action_dim:
        raise RuntimeError("ONNX action dimension does not match policy.meta.yaml")
    if tuple(io["joint_order"]) != JOINT_ORDER:
        raise RuntimeError("policy.meta.yaml joint order does not match deployment code")
    if model_path is not None and metadata.get("onnx", {}).get("sha256") != _sha256(resolve_path(model_path)):
        raise RuntimeError("ONNX hash does not match policy.meta.yaml")
    inertia = metadata.get("inertia_audit")
    if metadata.get("deployment_class") == "smoke_only":
        if not inertia:
            raise RuntimeError("smoke_only policy metadata is missing inertia_audit")
        report_path = resolve_path(inertia["path"])
        if _sha256(report_path) != inertia.get("sha256"):
            raise RuntimeError("Inertia report hash does not match policy.meta.yaml")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if not bool(report.get("passed", False)):
            raise RuntimeError("Inertia audit did not pass")
    return metadata


def main():
    parser = argparse.ArgumentParser(description="Deploy a Ur5eRobotiq ONNX policy with initial YOLO state")
    parser.add_argument("--config", default="resources/config/ur5e_deploy.yaml")
    parser.add_argument("--provider", choices=("cuda", "cpu"))
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--execute", action="store_true", help="enable real motion (also requires config allow_motion)")
    parser.add_argument(
        "--allow-smoke-policy",
        action="store_true",
        help="explicitly allow a smoke_only policy to command hardware",
    )
    args = parser.parse_args()
    config = yaml.safe_load(resolve_path(args.config).read_text(encoding="utf-8"))
    policy_cfg = config["policy"]
    safety_cfg = config["safety"]
    if args.execute and not bool(safety_cfg.get("allow_motion", False)):
        raise RuntimeError("Refusing motion: set safety.allow_motion=true as well as passing --execute")
    policy = PolicyRunner(resolve_path(policy_cfg["model"]), args.provider or policy_cfg["provider"])
    metadata = validate_contract(policy_cfg["metadata"], policy, policy_cfg["model"])
    deployment_class = str(metadata.get("deployment_class", "production"))
    smoke_only = deployment_class == "smoke_only"
    smoke_cfg = config.get("smoke_policy", {})
    if args.execute and smoke_only and not args.allow_smoke_policy:
        raise RuntimeError("Refusing smoke_only motion without --allow-smoke-policy")
    if args.execute and smoke_only and args.steps is not None and int(args.steps) != 600:
        raise RuntimeError("Executed smoke_only episodes must run the fixed 600-step contract")
    if args.execute:
        confirmation = input("即将向真实 UR5e/Robotiq 发送动作。确认安全区域无人后输入 ARM：").strip()
        if confirmation != "ARM":
            raise RuntimeError("Motion arming cancelled")
    period = float(policy_cfg["period"])
    camera_cfg = config["camera"]
    tracking_camera_cfg = config["tracking_camera"]
    wrist_serial = camera_cfg.get("serial")
    tracking_serial = tracking_camera_cfg.get("serial")
    if not wrist_serial or not tracking_serial or str(wrist_serial) == str(tracking_serial):
        raise RuntimeError("camera.serial and tracking_camera.serial must be set to distinct devices")
    tracking_cfg = config["tracking"]
    vision_cfg = config["vision"]
    robot_cfg = config["robot"]
    task_cfg = config["task"]
    tcp_to_camera = load_eye_on_hand(
        resolve_path(camera_cfg["calibration"]),
        config["calibration"]["checkerboard"],
        camera_cfg.get("model"),
    )
    locator = YoloInitialObjectLocator(
        resolve_path(vision_cfg["weights"]),
        tcp_to_camera,
        vision_cfg.get("target_class"),
        vision_cfg["confidence"],
        vision_cfg["depth_radius"],
        vision_cfg["depth_min"],
        vision_cfg["depth_max"],
        vision_cfg["center_depth_offset_m"],
        vision_cfg.get("device"),
    )
    base_to_tracking_camera = load_eye_on_base(
        resolve_path(tracking_camera_cfg["calibration"]),
        config["calibration"]["checkerboard"],
        tracking_camera_cfg.get("model"),
        tracking_serial,
    )
    tracking_detector = YoloInitialObjectLocator(
        resolve_path(vision_cfg["weights"]),
        np.eye(4),
        vision_cfg.get("target_class"),
        vision_cfg["confidence"],
        vision_cfg["depth_radius"],
        vision_cfg["depth_min"],
        vision_cfg["depth_max"],
        vision_cfg["center_depth_offset_m"],
        vision_cfg.get("device"),
        model=locator.model,
    )
    tracking_locator = YoloEyeOnBaseLocator(base_to_tracking_camera, tracking_detector)
    projector = MujocoStateProjector(
        ROOT / "resources/assets/robots/ur5e_robotiq_2f85/ur5e_robotiq_2f85.xml",
        task_cfg["palm_center_offset"],
    )
    observation = Ur5eObservationBuilder(
        projector,
        task_cfg["goal_position"],
        task_cfg["object_quaternion_xyzw"],
        task_cfg["object_scale"],
        subtask="grasp" if smoke_only else "throw",
        goal_offset=smoke_cfg.get("goal_offset_m") if smoke_only else None,
    )
    max_arm_step = smoke_cfg["max_arm_step"] if smoke_only else safety_cfg["max_arm_step"]
    action_mapper = Ur5eActionMapper(
        period,
        10.0,
        max_arm_step,
        safety_cfg["arm_lower"],
        safety_cfg["arm_upper"],
        action_scale=smoke_cfg["arm_action_scale"] if smoke_only else 1.0,
    )
    safety = SafetyMonitor(
        safety_cfg["arm_lower"],
        safety_cfg["arm_upper"],
        smoke_cfg["max_joint_velocity"] if smoke_only else safety_cfg["max_joint_velocity"],
        smoke_cfg["max_target_error"] if smoke_only else safety_cfg["max_target_error"],
        safety_cfg["max_state_age"],
        safety_cfg["workspace_min"],
        safety_cfg["workspace_max"],
        safety_cfg["object_position_min"],
        safety_cfg["object_position_max"],
    )
    steps = 600 if args.execute and smoke_only else int(args.steps or policy_cfg["episode_steps"])
    output = resolve_path(config["output"]["directory"])
    print("模式：{}；腕部相机初始化，固定相机在线跟踪。".format("实机执行" if args.execute else "只读 dry-run"))
    with RealSenseCamera(
        camera_cfg["width"],
        camera_cfg["height"],
        camera_cfg["fps"],
        camera_cfg.get("serial"),
        camera_cfg.get("model"),
    ) as camera, RealSenseCamera(
        tracking_camera_cfg["width"],
        tracking_camera_cfg["height"],
        tracking_camera_cfg["fps"],
        tracking_camera_cfg.get("serial"),
        tracking_camera_cfg.get("model"),
    ) as tracking_camera, UR5eHardware(
        robot_cfg["host"],
        robot_cfg["gripper_port"],
        servo_period=robot_cfg["servo_period"],
        servo_speed=robot_cfg["servo_speed"],
        servo_acceleration=robot_cfg["servo_acceleration"],
        servo_lookahead=robot_cfg["servo_lookahead"],
        servo_gain=robot_cfg["servo_gain"],
        gripper_speed=smoke_cfg["gripper_speed"] if smoke_only else robot_cfg["gripper_speed"],
        gripper_force=smoke_cfg["gripper_force"] if smoke_only else robot_cfg["gripper_force"],
        rtde_receive_priority=robot_cfg["rtde_receive_priority"],
        rtde_control_priority=robot_cfg["rtde_control_priority"],
        servo_thread_priority=robot_cfg["servo_thread_priority"],
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
            camera_cfg["max_detection_sync_interval_s"],
            camera_cfg["max_detection_translation_delta_m"],
            camera_cfg["max_detection_rotation_delta_deg"],
            camera_cfg["max_detection_linear_speed_m_s"],
            camera_cfg["max_detection_angular_speed_rad_s"],
            home_joint_position=robot_cfg["home_joint_position"],
            home_speed=robot_cfg["home_speed"],
            home_acceleration=robot_cfg["home_acceleration"],
            home_tolerance=robot_cfg["home_tolerance"],
            tracking_camera=tracking_camera,
            tracking_locator=tracking_locator,
            cross_camera_max_delta_m=tracking_cfg["cross_camera_max_delta_m"],
            association_max_distance_m=tracking_cfg["association_max_distance_m"],
            position_alpha=tracking_cfg["position_alpha"],
            velocity_alpha=tracking_cfg["velocity_alpha"],
            prediction_horizon_s=tracking_cfg["prediction_horizon_s"],
            max_object_state_age_s=tracking_cfg["max_state_age_s"],
            verdict_provider=request_smoke_verdict if smoke_only else request_verdict,
            deployment_class=deployment_class,
        )
        for index in range(args.episodes):
            print("开始回合 {}/{}".format(index + 1, args.episodes))
            result = session.run_episode(steps, int(vision_cfg["detection_attempts"]))
            print("回合 {}：verdict={}，steps={}，轨迹={}".format(
                result.episode_id, result.verdict, result.steps, result.trajectory_path
            ))


if __name__ == "__main__":
    main()
