#!/usr/bin/env python3
"""Offline preflight for files, dependencies, policy I/O, and safety configuration."""

from __future__ import annotations

import argparse
import importlib
import os
import platform
from pathlib import Path
import resource
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ur5e_comm.geometry import load_eye_on_base, load_eye_on_hand
from ur5e_comm.yolo_compat import load_yolo_model


os.environ.setdefault("MPLCONFIGDIR", "/tmp/ur5e-deploy-matplotlib")
EXPECTED_HARDWARE_PACKAGES = {
    "ur-rtde": ("1.6.5", ("rtde_control", "rtde_receive")),
    "ultralytics": ("8.0.20", ("ultralytics",)),
    "pyrealsense2": ("2.55.1.6486", ("pyrealsense2",)),
}


def distribution_version(name):
    try:
        try:
            from importlib.metadata import version
        except ImportError:
            from importlib_metadata import version
        return version(name)
    except Exception:
        return None


def model_names(model):
    names = getattr(model, "names", None)
    if isinstance(names, dict):
        return [str(value) for _, value in sorted(names.items(), key=lambda item: int(item[0]))]
    if isinstance(names, (list, tuple)):
        return [str(value) for value in names]
    raise RuntimeError("weights do not expose YOLO class names")


def resolve_target_class(names, target):
    target = str(target)
    if target in names:
        return names.index(target)
    try:
        class_id = int(target)
    except ValueError as error:
        raise RuntimeError(
            "target class {!r} is absent; available classes: {}".format(target, names)
        ) from error
    if class_id < 0 or class_id >= len(names):
        raise RuntimeError(
            "target class ID {} is outside [0, {}]".format(class_id, len(names) - 1)
        )
    return class_id


def resolve_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="resources/config/ur5e_deploy.yaml")
    args = parser.parse_args()
    config_path = resolve_path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    failures = []

    def report(status, label, detail=""):
        print("[{:<7}] {}{}".format(status, label, ": " + detail if detail else ""))

    required_files = {
        "UR5e ONNX policy": config["policy"]["model"],
        "policy metadata": config["policy"]["metadata"],
        "YOLO weights": config["vision"]["weights"],
        "eye-on-hand calibration": config["camera"]["calibration"],
        "eye-on-base calibration": config["tracking_camera"]["calibration"],
        "kinematic MJCF": "resources/assets/robots/ur5e_robotiq_2f85/ur5e_robotiq_2f85.xml",
        "UR5e throw sim2sim scene": "resources/assets/robots/ur5e_robotiq_2f85/ur5e_robotiq_throw.xml",
    }
    for label, value in required_files.items():
        path = resolve_path(value)
        if path.is_file():
            report("PASS", label, str(path))
        else:
            report("MISSING", label, str(path))
            failures.append(label)

    for package, (expected, modules) in EXPECTED_HARDWARE_PACKAGES.items():
        actual = distribution_version(package)
        import_errors = []
        for module in modules:
            try:
                importlib.import_module(module)
            except Exception as error:
                import_errors.append("{}: {}".format(module, error))
        if import_errors:
            report("MISSING", "Python dependency {}".format(package), "; ".join(import_errors))
            failures.append(package)
        elif actual != expected:
            report("FAIL", "Python dependency {}".format(package), "expected {}, found {}".format(expected, actual))
            failures.append("{} version".format(package))
        else:
            report("PASS", "Python dependency {}".format(package), actual)
    try:
        importlib.import_module("cv2")
        report("PASS", "Python dependency opencv-python", distribution_version("opencv-python") or "unknown")
    except Exception as error:
        report("MISSING", "Python dependency opencv-python", str(error))
        failures.append("opencv-python")

    weights_path = resolve_path(config["vision"]["weights"])
    if weights_path.is_file() and "ultralytics" not in failures:
        try:
            names = model_names(load_yolo_model(weights_path))
            target_id = resolve_target_class(names, config["vision"]["target_class"])
            report(
                "PASS",
                "YOLO class contract",
                "target {}: {}; all classes {}".format(target_id, names[target_id], names),
            )
        except Exception as error:
            report("FAIL", "YOLO class contract", str(error))
            failures.append("YOLO class contract")

    calibration_path = resolve_path(config["camera"]["calibration"])
    if calibration_path.is_file():
        try:
            load_eye_on_hand(
                calibration_path,
                config["calibration"]["checkerboard"],
                config["camera"].get("model"),
            )
            report("PASS", "eye-on-hand convention", "T_tcp_camera, board and camera match")
        except Exception as error:
            report("FAIL", "eye-on-hand convention", str(error))
            failures.append("eye-on-hand convention")

    tracking_calibration_path = resolve_path(config["tracking_camera"]["calibration"])
    if tracking_calibration_path.is_file():
        try:
            load_eye_on_base(
                tracking_calibration_path,
                config["calibration"]["checkerboard"],
                config["tracking_camera"].get("model"),
                config["tracking_camera"].get("serial"),
            )
            report("PASS", "eye-on-base convention", "T_base_camera, board and camera match")
        except Exception as error:
            report("FAIL", "eye-on-base convention", str(error))
            failures.append("eye-on-base convention")

    board = config["calibration"]["checkerboard"]
    board_valid = (
        board.get("model") == "DFVision Q12-240-15"
        and int(board.get("columns", 0)) == 11
        and int(board.get("rows", 0)) == 8
        and abs(float(board.get("square_size_m", 0.0)) - 0.015) <= 1e-9
    )
    report(
        "PASS" if board_valid else "FAIL",
        "checkerboard contract",
        "DFVision Q12-240-15: 11x8 inner corners, 15 mm squares",
    )
    if not board_valid:
        failures.append("checkerboard contract")
    camera_valid = all(
        "d435i" in str(camera.get("model", "")).lower()
        for camera in (config["camera"], config["tracking_camera"])
    )
    report("PASS" if camera_valid else "FAIL", "dual-camera model contract", "two RealSense D435i")
    if not camera_valid:
        failures.append("dual-camera model contract")
    wrist_serial = config["camera"].get("serial")
    tracking_serial = config["tracking_camera"].get("serial")
    serials_valid = bool(wrist_serial and tracking_serial and str(wrist_serial) != str(tracking_serial))
    report(
        "PASS" if serials_valid else "FAIL",
        "dual-camera serial contract",
        "wrist={!r}, fixed={!r}".format(wrist_serial, tracking_serial),
    )
    if not serials_valid:
        failures.append("dual-camera serial contract")

    metadata_path = resolve_path(config["policy"]["metadata"])
    if metadata_path.is_file():
        metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
        io = metadata.get("io", {})
        valid_io = io.get("observation_dim") == 69 and io.get("action_dim") == 7
        report("PASS" if valid_io else "FAIL", "policy metadata I/O", "expected 69 -> 7")
        if not valid_io:
            failures.append("policy metadata I/O")

    robot = config["robot"]
    robot_host = str(robot.get("host", "")).strip()
    host_valid = bool(robot_host)
    report("PASS" if host_valid else "FAIL", "UR control-box host", robot_host or "missing")
    if not host_valid:
        failures.append("UR control-box host")
    try:
        gripper_port = int(robot.get("gripper_port"))
    except (TypeError, ValueError):
        gripper_port = None
    gripper_socket_valid = gripper_port == 63352
    report(
        "PASS" if gripper_socket_valid else "FAIL",
        "Robotiq control-box socket contract",
        "{}:{} via Robotiq Grippers URCap".format(robot_host or "<robot-host>", gripper_port),
    )
    if not gripper_socket_valid:
        failures.append("Robotiq control-box socket contract")
    priorities_valid = (
        int(robot.get("rtde_receive_priority", 0)) == 90
        and int(robot.get("rtde_control_priority", 0)) == 85
        and int(robot.get("servo_thread_priority", 0)) == 80
    )
    report(
        "PASS" if priorities_valid else "FAIL",
        "real-time priority contract",
        "RTDE receive/control and servo feeder: 90/85/80",
    )
    if not priorities_valid:
        failures.append("real-time priority contract")
    kernel_release = platform.release()
    lowlatency_valid = "lowlatency" in kernel_release
    report(
        "PASS" if lowlatency_valid else "FAIL",
        "soft real-time kernel",
        kernel_release,
    )
    if not lowlatency_valid:
        failures.append("soft real-time kernel")
    rtprio_soft, _ = resource.getrlimit(resource.RLIMIT_RTPRIO)
    rtprio_valid = rtprio_soft == resource.RLIM_INFINITY or rtprio_soft >= 90
    report(
        "PASS" if rtprio_valid else "FAIL",
        "real-time scheduling permission",
        "RLIMIT_RTPRIO={}".format(rtprio_soft),
    )
    if not rtprio_valid:
        failures.append("real-time scheduling permission")
    home_valid = (
        robot.get("home_joint_position") == [-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0]
        and float(robot.get("home_speed", 0.0)) == 0.25
        and float(robot.get("home_acceleration", 0.0)) == 0.5
        and float(robot.get("home_tolerance", 0.0)) == 0.02
    )
    report("PASS" if home_valid else "FAIL", "automatic home contract")
    if not home_valid:
        failures.append("automatic home contract")
    depth_offset_valid = abs(float(config["vision"].get("center_depth_offset_m", 0.0)) - 0.01) <= 1e-12
    report("PASS" if depth_offset_valid else "FAIL", "object center depth offset", "0.01 m")
    if not depth_offset_valid:
        failures.append("object center depth offset")

    safety = config["safety"]
    if safety.get("allow_motion", False):
        report("REVIEW", "real motion is enabled in configuration")
    else:
        report("PASS", "default is read-only (allow_motion=false)")
    report(
        "REVIEW",
        "PolyScope services",
        "Remote Control plus Dashboard, Primary/Secondary, Real-time and RTDE",
    )
    report(
        "REVIEW",
        "control-box gripper driver",
        "compatible Robotiq Grippers URCap installed, activated and serving port 63352",
    )
    report(
        "REVIEW",
        "control-box RS-485 wiring",
        "external cable, controller 24 V/0 V and original USB-RS485 adapter verified on site",
    )
    report("REVIEW", "goal position in UR base", str(config["task"]["goal_position"]))
    report("REVIEW", "joint/workspace limits", "must be measured for the physical cell")
    if failures:
        print("\nPreflight incomplete: {}".format(", ".join(failures)))
        raise SystemExit(2)
    print(
        "\nOffline preflight passed. Controller services, URCap socket, wiring, payload/TCP "
        "and physical values still require on-site validation."
    )


if __name__ == "__main__":
    main()
