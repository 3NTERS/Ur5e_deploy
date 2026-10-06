#!/usr/bin/env python3
"""Replay recorded real UR5e hover states in MuJoCo without robot I/O."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import sys
from time import monotonic, sleep

import mujoco
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mujoco_sim.ur5e_adapter import GRIPPER_MULTIPLIERS, Ur5eGraspAdapter


def resolve_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def latest_nonempty_trajectory(directory):
    for path in sorted(directory.glob("episode_*.npz"), reverse=True):
        with np.load(str(path), allow_pickle=False) as payload:
            if "progress" in payload and len(payload["progress"]) > 0:
                return path
    raise RuntimeError("No nonempty real hover trajectory found in {}".format(directory))


def load_trace(path):
    required = (
        "progress", "monotonic_timestamp", "joint_position",
        "gripper_position", "object_position", "palm_position",
    )
    with np.load(str(path), allow_pickle=False) as payload:
        missing = [key for key in required if key not in payload]
        if missing:
            raise RuntimeError("Real trajectory is missing {}".format(missing))
        trace = {key: np.asarray(payload[key]).copy() for key in required}
    count = len(trace["progress"])
    shapes = {
        "progress": (count,),
        "monotonic_timestamp": (count,),
        "joint_position": (count, 6),
        "gripper_position": (count,),
        "object_position": (count, 3),
        "palm_position": (count, 3),
    }
    if count == 0:
        raise RuntimeError("Real trajectory has no recorded steps")
    for key, shape in shapes.items():
        if trace[key].shape != shape or not np.isfinite(trace[key]).all():
            raise RuntimeError("{} must have finite shape {}".format(key, shape))
    if np.any(np.diff(trace["monotonic_timestamp"]) < 0):
        raise RuntimeError("Real trajectory timestamps must be nondecreasing")
    gripper = trace["gripper_position"]
    if np.any(gripper < 0) or np.any(gripper > 255):
        raise RuntimeError("Recorded gripper position must be in [0, 255]")
    return trace


def replay(args):
    config = yaml.safe_load(resolve_path(args.config).read_text(encoding="utf-8"))
    sim_config = yaml.safe_load(resolve_path(args.sim_config).read_text(encoding="utf-8"))
    trajectory = (
        resolve_path(args.trajectory) if args.trajectory else
        latest_nonempty_trajectory(resolve_path(config["output"]["directory"]))
    )
    if not trajectory.is_file():
        raise FileNotFoundError("Real trajectory does not exist: {}".format(trajectory))
    trace = load_trace(trajectory)
    if not np.isfinite(args.speed) or args.speed <= 0:
        raise RuntimeError("--speed must be positive")

    adapter = Ur5eGraspAdapter(
        resolve_path(config["observation"]["model"]),
        resolve_path(config["policy"]["metadata"]),
        simulation_config=sim_config,
    )
    model = adapter.model
    data = adapter.data
    object_address = adapter.object_qpos_addr
    model.geom_rgba[adapter.object_geom] = [0.05, 0.25, 1.0, 1.0]
    hover_height = float(config["safety"]["hover_height_m"])
    palm_errors = []
    frames_replayed = 0

    if args.headless:
        viewer_context = nullcontext(None)
    else:
        from mujoco import viewer as mj_viewer
        viewer_context = mj_viewer.launch_passive(model, data)

    with viewer_context as viewer:
        if viewer is not None:
            viewer.cam.lookat[:] = [-0.10, -0.50, 0.20]
            viewer.cam.distance = 1.2
        started = monotonic()
        timestamps = trace["monotonic_timestamp"]
        for index in range(len(trace["progress"])):
            if viewer is not None and not viewer.is_running():
                break
            if viewer is not None:
                deadline = started + (timestamps[index] - timestamps[0]) / args.speed
                remaining = deadline - monotonic()
                if remaining > 0:
                    sleep(remaining)

            master = float(trace["gripper_position"][index]) / 255.0 * 0.72
            data.qpos[adapter.qpos_addr[:6]] = trace["joint_position"][index]
            data.qpos[adapter.qpos_addr[6:]] = master * GRIPPER_MULTIPLIERS
            data.qpos[object_address:object_address + 3] = trace["object_position"][index]
            data.qpos[object_address + 3:object_address + 7] = [1.0, 0.0, 0.0, 0.0]
            model.site_pos[adapter.goal_site] = (
                trace["object_position"][index] + [0.0, 0.0, hover_height]
            )
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            palm_errors.append(float(np.linalg.norm(
                data.site_xpos[adapter.palm_site] - trace["palm_position"][index]
            )))
            frames_replayed += 1
            if viewer is not None:
                viewer.sync()
        if viewer is not None and not args.no_hold:
            while viewer.is_running():
                sleep(0.05)

    summary_path = trajectory.with_suffix(".json")
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("error"):
            print("recorded_episode_error={}".format(summary["error"]))
    print(
        "trajectory={} frames_replayed={} recorded_duration_s={:.3f} "
        "mean_palm_alignment_error_m={:.9g} max_palm_alignment_error_m={:.9g}".format(
            trajectory, frames_replayed,
            float(trace["monotonic_timestamp"][-1] - trace["monotonic_timestamp"][0]),
            float(np.mean(palm_errors)) if palm_errors else float("nan"),
            float(np.max(palm_errors)) if palm_errors else float("nan"),
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="resources/config/ur5e_hover_gripper_deploy.yaml")
    parser.add_argument("--sim-config", default="resources/config/ur5e_hover_gripper_sim2sim.yaml")
    parser.add_argument("--trajectory", help="Real episode NPZ; default: latest nonempty episode")
    parser.add_argument("--speed", type=float, default=1.0, help="Replay speed multiplier")
    parser.add_argument("--headless", action="store_true", help="Validate replay without opening a viewer")
    parser.add_argument("--no-hold", action="store_true", help="Close viewer at the last recorded frame")
    replay(parser.parse_args())


if __name__ == "__main__":
    main()
