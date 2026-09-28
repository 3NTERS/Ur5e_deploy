#!/usr/bin/env python3
"""Offline preflight for files, dependencies, policy I/O, and safety configuration."""

from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
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
        "eye-on-base calibration": config["camera"]["calibration"],
        "kinematic MJCF": "resources/assets/robots/ur5e_robotiq_2f85/ur5e_robotiq_2f85.xml",
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
            from ultralytics import YOLO
            names = model_names(YOLO(str(weights_path)))
            target = str(config["vision"]["target_class"])
            if names != [target]:
                raise RuntimeError("expected [{}], found {}".format(target, names))
            report("PASS", "YOLO class contract", "0: {}".format(target))
        except Exception as error:
            report("FAIL", "YOLO class contract", str(error))
            failures.append("YOLO class contract")

    metadata_path = resolve_path(config["policy"]["metadata"])
    if metadata_path.is_file():
        metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
        io = metadata.get("io", {})
        valid_io = io.get("observation_dim") == 69 and io.get("action_dim") == 7
        report("PASS" if valid_io else "FAIL", "policy metadata I/O", "expected 69 -> 7")
        if not valid_io:
            failures.append("policy metadata I/O")

    safety = config["safety"]
    if safety.get("allow_motion", False):
        report("REVIEW", "real motion is enabled in configuration")
    else:
        report("PASS", "default is read-only (allow_motion=false)")
    report("REVIEW", "robot host", str(config["robot"]["host"]))
    report("REVIEW", "goal position in UR base", str(config["task"]["goal_position"]))
    report("REVIEW", "joint/workspace limits", "must be measured for the physical cell")
    if failures:
        print("\nPreflight incomplete: {}".format(", ".join(failures)))
        raise SystemExit(2)
    print("\nOffline preflight passed. Hardware connectivity and physical values still require on-site validation.")


if __name__ == "__main__":
    main()
