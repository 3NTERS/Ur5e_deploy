from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic, sleep

import numpy as np

from .geometry import average_transforms, invert_transform, rotation_angle_degrees, tcp_pose_to_transform


@dataclass(frozen=True)
class EpisodeResult:
    episode_id: str
    verdict: str
    steps: int
    trajectory_path: Path


def request_verdict(input_fn=input):
    while True:
        value = input("物体命中请输入 y，未命中请输入 x：").strip().lower()
        if value in ("y", "x"):
            return value
        print("只接受 y 或 x。")


class DeploymentSession:
    """One-object-lock per episode real-robot policy loop."""

    def __init__(
        self,
        camera,
        locator,
        robot,
        policy,
        observation_builder,
        action_mapper,
        safety_monitor,
        output_directory,
        execute=False,
        policy_period=0.01667,
        max_policy_lag=0.05,
        max_detection_sync_interval=0.25,
        max_detection_translation_delta=0.002,
        max_detection_rotation_delta_deg=0.5,
        max_detection_linear_speed=0.005,
        max_detection_angular_speed=0.02,
        verdict_provider=request_verdict,
    ):
        self.camera = camera
        self.locator = locator
        self.robot = robot
        self.policy = policy
        self.observation_builder = observation_builder
        self.action_mapper = action_mapper
        self.safety_monitor = safety_monitor
        self.output_directory = Path(output_directory)
        self.execute = bool(execute)
        self.policy_period = float(policy_period)
        self.max_policy_lag = float(max_policy_lag)
        self.max_detection_sync_interval = float(max_detection_sync_interval)
        self.max_detection_translation_delta = float(max_detection_translation_delta)
        self.max_detection_rotation_delta_deg = float(max_detection_rotation_delta_deg)
        self.max_detection_linear_speed = float(max_detection_linear_speed)
        self.max_detection_angular_speed = float(max_detection_angular_speed)
        self.verdict_provider = verdict_provider
        if policy.observation_dim != observation_builder.observation_dim:
            raise ValueError(
                "Policy expects {} observations, deployment builds {}".format(
                    policy.observation_dim, observation_builder.observation_dim
                )
            )
        if policy.action_dim != action_mapper.action_dim:
            raise ValueError(
                "Policy emits {} actions, deployment expects {}".format(
                    policy.action_dim, action_mapper.action_dim
                )
            )

    def _lock_initial_object(self, attempts):
        errors = []
        for _ in range(int(attempts)):
            before = self.robot.read()
            self.safety_monitor.check_state(before)
            frame = self.camera.read()
            after = self.robot.read()
            self.safety_monitor.check_state(after)
            before_transform = tcp_pose_to_transform(before.tcp_pose)
            after_transform = tcp_pose_to_transform(after.tcp_pose)
            relative = invert_transform(before_transform).dot(after_transform)
            elapsed = after.timestamp - before.timestamp
            translation_delta = float(np.linalg.norm(relative[:3, 3]))
            rotation_delta = rotation_angle_degrees(relative)
            linear_speed = max(
                float(np.linalg.norm(before.tcp_speed[:3])),
                float(np.linalg.norm(after.tcp_speed[:3])),
            )
            angular_speed = max(
                float(np.linalg.norm(before.tcp_speed[3:])),
                float(np.linalg.norm(after.tcp_speed[3:])),
            )
            unstable = (
                elapsed > self.max_detection_sync_interval
                or translation_delta > self.max_detection_translation_delta
                or rotation_delta > self.max_detection_rotation_delta_deg
                or linear_speed > self.max_detection_linear_speed
                or angular_speed > self.max_detection_angular_speed
            )
            if unstable:
                errors.append(
                    "unstable TCP during image: dt={:.3f}s, d={:.4f}m, r={:.2f}deg".format(
                        elapsed, translation_delta, rotation_delta
                    )
                )
                continue
            if not before.timestamp <= frame.timestamp <= after.timestamp:
                errors.append("camera timestamp was not bracketed by RTDE samples")
                continue
            base_to_tcp = average_transforms([before_transform, after_transform])
            try:
                return self.locator.locate(frame, base_to_tcp), after
            except RuntimeError as error:
                errors.append(str(error))
        detail = errors[-1] if errors else "no attempts were made"
        raise RuntimeError("Failed to lock initial object after {} frames: {}".format(attempts, detail))

    def run_episode(self, steps=600, detection_attempts=30):
        if int(detection_attempts) <= 0:
            raise ValueError("detection attempts must be positive")
        detection, initial_snapshot = self._lock_initial_object(detection_attempts)
        self.safety_monitor.check_initial_object(detection.position_base)
        self.policy.reset()
        self.observation_builder.begin_episode(detection.position_base)
        self.action_mapper.reset(initial_snapshot)
        if self.execute:
            self.robot.start_motion()
        episode_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        rows = {
            "observation": [],
            "action": [],
            "arm_target": [],
            "gripper_target": [],
            "joint_position": [],
            "joint_velocity": [],
            "tcp_pose": [],
            "tcp_speed": [],
            "gripper_position": [],
            "joint_current": [],
            "target_moment": [],
            "robot_mode": [],
            "safety_mode": [],
            "protective_stopped": [],
            "emergency_stopped": [],
            "palm_position": [],
            "fingertip_position": [],
            "monotonic_time": [],
        }
        started = monotonic()
        try:
            for step in range(int(steps)):
                snapshot = initial_snapshot if step == 0 else self.robot.read()
                self.safety_monitor.check_state(snapshot)
                observation, projected = self.observation_builder.build(snapshot)
                action = self.policy.infer(observation)[0]
                # Read-only runs evaluate safe one-step commands around the measured
                # state instead of pretending the stationary robot followed them.
                if not self.execute:
                    self.action_mapper.reset(snapshot)
                arm_target, gripper_target = self.action_mapper.map(action)
                self.safety_monitor.check_target(
                    snapshot,
                    arm_target,
                    gripper_target,
                    self.observation_builder.projector,
                )
                if self.execute:
                    self.robot.command(arm_target, gripper_target)
                rows["observation"].append(observation)
                rows["action"].append(action)
                rows["arm_target"].append(arm_target)
                rows["gripper_target"].append(gripper_target)
                rows["joint_position"].append(snapshot.joint_position)
                rows["joint_velocity"].append(snapshot.joint_velocity)
                rows["tcp_pose"].append(snapshot.tcp_pose)
                rows["tcp_speed"].append(snapshot.tcp_speed)
                rows["gripper_position"].append(snapshot.gripper_position)
                rows["joint_current"].append(snapshot.joint_current)
                rows["target_moment"].append(snapshot.target_moment)
                rows["robot_mode"].append(snapshot.robot_mode)
                rows["safety_mode"].append(snapshot.safety_mode)
                rows["protective_stopped"].append(snapshot.protective_stopped)
                rows["emergency_stopped"].append(snapshot.emergency_stopped)
                rows["palm_position"].append(projected.palm_center_position)
                rows["fingertip_position"].append(projected.fingertip_position)
                rows["monotonic_time"].append(snapshot.timestamp)
                self.observation_builder.advance()
                deadline = started + (step + 1) * self.policy_period
                remaining = deadline - monotonic()
                if remaining < -self.max_policy_lag:
                    raise RuntimeError(
                        "Policy loop missed its schedule by {:.3f}s".format(-remaining)
                    )
                if remaining > 0.0:
                    sleep(remaining)
        finally:
            if self.execute:
                self.robot.stop_motion()

        verdict = self.verdict_provider()
        self.output_directory.mkdir(parents=True, exist_ok=True)
        trajectory_path = self.output_directory / "episode_{}.npz".format(episode_id)
        arrays = {name: np.asarray(value) for name, value in rows.items()}
        arrays.update(
            {
                "initial_object_position_camera": detection.position_camera,
                "initial_object_position_base": detection.position_base,
                "initial_base_to_camera": detection.base_to_camera,
                "detection_center_pixel": detection.center_pixel,
                "detection_confidence": np.asarray(detection.confidence),
                "verdict": np.asarray(verdict),
                "executed": np.asarray(self.execute),
            }
        )
        np.savez_compressed(str(trajectory_path), **arrays)
        summary = {
            "episode_id": episode_id,
            "utc_time": datetime.now(timezone.utc).isoformat(),
            "verdict": verdict,
            "hit": verdict == "y",
            "executed": self.execute,
            "steps": len(rows["action"]),
            "trajectory": str(trajectory_path),
            "detection": {
                "class_id": detection.class_id,
                "class_name": detection.class_name,
                "confidence": detection.confidence,
                "center_pixel": detection.center_pixel.tolist(),
                "depth_m": detection.depth_m,
                "position_camera": detection.position_camera.tolist(),
                "position_base": detection.position_base.tolist(),
                "base_to_camera": detection.base_to_camera.tolist(),
            },
        }
        with (self.output_directory / "episodes.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(summary, ensure_ascii=False) + "\n")
        return EpisodeResult(episode_id, verdict, len(rows["action"]), trajectory_path)
