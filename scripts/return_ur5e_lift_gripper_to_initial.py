#!/usr/bin/env python3
"""Slowly return a stationary UR5e to the fixed lift/gripper initial pose."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import monotonic, sleep

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ur5e_comm.robot import UR5eHardware


def resolve_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def vector(config, key, size=6):
    value = np.asarray(config[key], dtype=np.float64)
    if value.shape != (size,) or not np.isfinite(value).all():
        raise ValueError("{} must contain {} finite values".format(key, size))
    return value


def validate_stationary_start(snapshot, initial, return_config):
    if snapshot.protective_stopped or snapshot.emergency_stopped:
        raise RuntimeError("UR protective stop or emergency stop is active")

    max_velocity = float(np.max(np.abs(snapshot.joint_velocity)))
    velocity_limit = float(return_config["max_start_joint_velocity"])
    if max_velocity > velocity_limit:
        raise RuntimeError(
            "robot must be stationary before return; measured {:.6f} rad/s "
            "exceeds {:.6f} rad/s".format(max_velocity, velocity_limit)
        )

    start_lower = vector(return_config, "start_lower")
    start_upper = vector(return_config, "start_upper")
    if np.any(start_lower >= start_upper):
        raise RuntimeError("return start envelope is invalid")
    if np.any(snapshot.joint_position < start_lower) or np.any(
        snapshot.joint_position > start_upper
    ):
        raise RuntimeError(
            "current joints {} are outside the permitted return start envelope; "
            "use supervised manual recovery".format(
                snapshot.joint_position.tolist()
            )
        )

    max_distance = float(np.max(np.abs(snapshot.joint_position - initial)))
    distance_limit = float(return_config["max_joint_distance"])
    if max_distance > distance_limit:
        raise RuntimeError(
            "return distance {:.6f} rad exceeds {:.6f} rad; "
            "use supervised manual recovery".format(max_distance, distance_limit)
        )
    return max_velocity, max_distance


def open_empty_gripper(robot, robot_config, safety):
    threshold = int(safety["initial_gripper_open_max"])
    timeout_s = float(robot_config["gripper_open_timeout_s"])
    position = int(robot.gripper.get("POS"))
    if position <= threshold:
        return position

    robot.gripper.move(
        0,
        int(robot_config["gripper_speed"]),
        int(robot_config["gripper_force"]),
    )
    deadline = monotonic() + timeout_s
    while monotonic() < deadline:
        position = int(robot.gripper.get("POS"))
        if position <= threshold:
            return position
        sleep(0.05)
    robot.gripper.stop()
    raise TimeoutError(
        "empty gripper did not open to POS<={} within {:.1f}s".format(
            threshold, timeout_s
        )
    )


def run(args):
    config = yaml.safe_load(resolve_path(args.config).read_text(encoding="utf-8"))
    robot_config = config["robot"]
    safety = config["safety"]
    return_config = config["return_to_initial"]
    initial = vector(robot_config, "initial_joint_position")

    speed = float(return_config["joint_speed"])
    acceleration = float(return_config["joint_acceleration"])
    tolerance = float(return_config["position_tolerance"])
    if speed <= 0.0 or acceleration <= 0.0 or tolerance <= 0.0:
        raise RuntimeError("return speed, acceleration, and tolerance must be positive")
    if np.any(initial < vector(safety, "corridor_lower")) or np.any(
        initial > vector(safety, "corridor_upper")
    ):
        raise RuntimeError("fixed initial pose is outside the deployment corridor")
    if args.execute and not bool(safety.get("allow_motion", False)):
        raise RuntimeError(
            "refusing motion: set safety.allow_motion=true and pass --execute"
        )

    if args.execute:
        confirmation = input(
            "即将先打开空夹爪，再以 {:.2f} rad/s 缓慢回到固定初始位姿。"
            "确认夹爪内无物体、回程路径无障碍且急停可达后输入 RETURN：".format(
                speed
            )
        ).strip()
        if confirmation != "RETURN":
            raise RuntimeError("return motion cancelled")

    summary = {
        "executed": bool(args.execute),
        "target_joint_position": initial.tolist(),
        "joint_speed_rad_s": speed,
        "joint_acceleration_rad_s2": acceleration,
        "success": False,
    }

    with UR5eHardware(
        robot_config["host"],
        robot_config["gripper_port"],
        servo_speed=robot_config["servo_speed"],
        servo_acceleration=robot_config["servo_acceleration"],
        servo_period=robot_config["servo_period"],
        servo_lookahead=robot_config["servo_lookahead"],
        servo_gain=robot_config["servo_gain"],
        gripper_speed=robot_config["gripper_speed"],
        gripper_force=robot_config["gripper_force"],
        rtde_receive_priority=robot_config["rtde_receive_priority"],
        rtde_control_priority=robot_config["rtde_control_priority"],
        servo_thread_priority=robot_config["servo_thread_priority"],
        allow_motion=args.execute,
        activate_gripper=args.execute,
    ) as robot:
        start = robot.read()
        max_velocity, max_distance = validate_stationary_start(
            start, initial, return_config
        )
        summary.update(
            {
                "start_joint_position": start.joint_position.tolist(),
                "start_gripper_position": int(start.gripper_position),
                "start_max_joint_velocity_rad_s": max_velocity,
                "planned_max_joint_distance_rad": max_distance,
            }
        )

        if not args.execute:
            summary["ready"] = True
            print("LIFT_GRIPPER_RETURN " + json.dumps(summary, sort_keys=True))
            return summary

        summary["opened_gripper_position"] = open_empty_gripper(
            robot, robot_config, safety
        )
        stationary = robot.read()
        _, return_distance = validate_stationary_start(
            stationary, initial, return_config
        )

        if return_distance > tolerance:
            robot.home(
                initial,
                speed=speed,
                acceleration=acceleration,
                tolerance=tolerance,
            )

        final = robot.read()
        final_error = float(np.max(np.abs(final.joint_position - initial)))
        final_velocity = float(np.max(np.abs(final.joint_velocity)))
        if final_error > tolerance:
            raise RuntimeError(
                "final joint error {:.6f} rad exceeds {:.6f} rad".format(
                    final_error, tolerance
                )
            )
        if final_velocity > float(return_config["max_start_joint_velocity"]):
            raise RuntimeError(
                "robot did not settle after return; measured {:.6f} rad/s".format(
                    final_velocity
                )
            )
        summary.update(
            {
                "final_joint_position": final.joint_position.tolist(),
                "final_gripper_position": int(final.gripper_position),
                "final_joint_error_rad": final_error,
                "final_max_joint_velocity_rad_s": final_velocity,
                "success": True,
            }
        )

    print("LIFT_GRIPPER_RETURN " + json.dumps(summary, sort_keys=True))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="resources/config/ur5e_lift_gripper_deploy.yaml"
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="open the empty gripper and command the slow joint-space return",
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
