from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic, sleep

import numpy as np


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

    def run_episode(self, steps=600, detection_attempts=30):
        detection = self.locator.locate_from_camera(self.camera, attempts=detection_attempts)
        self.safety_monitor.check_initial_object(detection.position_base)
        initial_snapshot = self.robot.read()
        self.safety_monitor.check_state(initial_snapshot)
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
            },
        }
        with (self.output_directory / "episodes.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(summary, ensure_ascii=False) + "\n")
        return EpisodeResult(episode_id, verdict, len(rows["action"]), trajectory_path)
