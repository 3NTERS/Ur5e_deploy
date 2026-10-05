#!/usr/bin/env python3
"""Read-only wrist-camera localization preview in the hover MuJoCo scene.

The blue cube is the raw camera detection transformed into the robot base
frame. MuJoCo is only used for display; no policy or robot command is run.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from time import monotonic, sleep
import xml.etree.ElementTree as ET

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_ur5e_hover_gripper_policy import (
    SingleWristObjectTracker,
    resolve_path,
    validate_calibration,
)
from ur5e_comm.observation import GRIPPER_MULTIPLIERS, JOINT_ORDER
from ur5e_comm.robot import UR5eHardware
from ur5e_comm.vision import RealSenseCamera, YoloInitialObjectLocator


def joint_addresses(model, mujoco):
    addresses = []
    for name in JOINT_ORDER:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise RuntimeError("MuJoCo model is missing joint {}".format(name))
        addresses.append(int(model.jnt_qposadr[joint_id]))
    return addresses


def add_bounds_edges(worldbody, lower, upper, name, rgba):
    """Add twelve non-colliding capsule edges for a base-frame AABB."""
    corners = np.asarray([
        [x, y, z]
        for x in (lower[0], upper[0])
        for y in (lower[1], upper[1])
        for z in (lower[2], upper[2])
    ], dtype=np.float64)
    edges = [
        (first, second)
        for first in range(8)
        for second in range(first + 1, 8)
        if (first ^ second) in (1, 2, 4)
    ]
    for index, (first, second) in enumerate(edges):
        ET.SubElement(
            worldbody, "geom",
            name="preview_{}_bounds_{:02d}".format(name, index),
            type="capsule",
            fromto="{} {} {} {} {} {}".format(
                *("{:.9g}".format(value) for value in (
                    *corners[first], *corners[second],
                ))
            ),
            size="0.003",
            rgba=rgba,
            contype="0",
            conaffinity="0",
            group="0",
        )


def make_preview_model(scene, mujoco, object_lower, object_upper,
                       palm_lower, palm_upper):
    """Add visual-only bounds to an in-memory copy of the hover model."""
    root = ET.parse(str(scene)).getroot()
    for element in root.iter():
        asset_file = element.get("file")
        if asset_file:
            # The in-memory XML has no source directory for relative meshes.
            element.set("file", str((scene.parent / asset_file).resolve()))
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError("hover model has no worldbody")
    add_bounds_edges(worldbody, object_lower, object_upper,
                     "object", "1 0.05 0.05 1")
    add_bounds_edges(worldbody, palm_lower, palm_upper,
                     "palm", "0.72 0.08 0.95 1")
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="resources/config/ur5e_hover_gripper_deploy.yaml"
    )
    parser.add_argument(
        "--sim-config", default="resources/config/ur5e_hover_gripper_sim2sim.yaml"
    )
    parser.add_argument(
        "--object-base", nargs=3, type=float, metavar=("X", "Y", "Z"),
        help="Preview a saved base-frame position without connecting to hardware",
    )
    parser.add_argument("--no-camera-preview", action="store_true")
    args = parser.parse_args()

    import mujoco
    import mujoco.viewer

    config = yaml.safe_load(resolve_path(args.config).read_text(encoding="utf-8"))
    sim_config = yaml.safe_load(resolve_path(args.sim_config).read_text(encoding="utf-8"))
    nominal = np.asarray(sim_config["object_position_center_base_m"], dtype=float)
    lower = np.asarray(config["safety"]["object_position_min"], dtype=np.float64)
    upper = np.asarray(config["safety"]["object_position_max"], dtype=np.float64)
    palm_lower = np.asarray(config["safety"]["workspace_min"], dtype=np.float64)
    palm_upper = np.asarray(config["safety"]["workspace_max"], dtype=np.float64)
    for name, minimum, maximum in (
        ("object_position", lower, upper),
        ("workspace", palm_lower, palm_upper),
    ):
        if (
            minimum.shape != (3,) or maximum.shape != (3,)
            or not np.isfinite(minimum).all() or not np.isfinite(maximum).all()
            or np.any(minimum >= maximum)
        ):
            raise RuntimeError("safety.{}_min/max must form a finite 3D box".format(name))
    print(
        "red_object_bounds_base min={} max={}".format(
            lower.tolist(), upper.tolist(),
        ), flush=True,
    )
    print(
        "purple_palm_bounds_base min={} max={}".format(
            palm_lower.tolist(), palm_upper.tolist(),
        ), flush=True,
    )
    scene = resolve_path(config["observation"]["model"])
    model = make_preview_model(
        scene, mujoco, lower, upper, palm_lower, palm_upper,
    )
    data = mujoco.MjData(model)
    addresses = joint_addresses(model, mujoco)
    object_joint = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "object_freejoint"
    )
    object_geom = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "object_geom"
    )
    if object_joint < 0 or object_geom < 0:
        raise RuntimeError("Hover scene requires object_freejoint and object_geom")
    object_address = int(model.jnt_qposadr[object_joint])
    model.geom_rgba[object_geom] = [0.05, 0.25, 1.0, 0.0]
    data.qpos[object_address + 3:object_address + 7] = [1, 0, 0, 0]

    def show_object(position):
        position = np.asarray(position, dtype=float)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError("object base position must be three finite coordinates")
        data.qpos[object_address:object_address + 3] = position
        model.geom_rgba[object_geom] = [0.05, 0.25, 1.0, 1.0]
        delta = position - nominal
        print(
            "camera_object_base={} simulation_nominal_base={} delta={} "
            "xy_distance={:.4f}m inside_red_bounds={}".format(
                np.round(position, 5).tolist(), np.round(nominal, 5).tolist(),
                np.round(delta, 5).tolist(), float(np.linalg.norm(delta[:2])),
                bool(np.all(position >= lower) and np.all(position <= upper)),
            ), flush=True,
        )

    if args.object_base is not None:
        data.qpos[addresses[:6]] = config["robot"]["initial_joint_position"]
        show_object(args.object_base)
        mujoco.mj_forward(model, data)
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.lookat[:] = 0.5 * (lower + upper)
            viewer.cam.distance = 1.2
            while viewer.is_running():
                viewer.sync()
                sleep(0.05)
        return

    # This path deliberately uses only RTDE receive and Robotiq GET; no
    # RTDE control interface, gripper activation, policy, or robot command.
    camera_config = config["camera"]
    vision_config = config["vision"]
    robot_config = config["robot"]
    tracker_config = config["tracking"]
    tcp_to_camera = validate_calibration(config)
    locator = YoloInitialObjectLocator(
        resolve_path(vision_config["weights"]), tcp_to_camera,
        vision_config.get("target_class"), vision_config["confidence"],
        vision_config["depth_radius"], vision_config["depth_min"],
        vision_config["depth_max"], vision_config["center_depth_offset_m"],
        vision_config.get("device"),
    )
    camera = RealSenseCamera(
        camera_config["width"], camera_config["height"],
        camera_config["fps"], camera_config["serial"],
        camera_config.get("model"),
    )
    tracker = SingleWristObjectTracker(
        camera, locator, vision_config["inference_interval_s"],
        camera_config["max_frame_age_s"],
        camera_config["max_pose_sync_error_s"], tracker_config,
    )
    cv2 = None
    if not args.no_camera_preview:
        import cv2
    last_timestamp = None
    try:
        with UR5eHardware(
            robot_config["host"], robot_config["gripper_port"],
            allow_motion=False, activate_gripper=False,
            rtde_receive_priority=robot_config["rtde_receive_priority"],
        ) as robot:
            tracker.update_robot_pose(robot.read())
            tracker.start()
            print("Read-only preview: blue cube = raw camera detection in robot base frame; "
                  "close MuJoCo or press q in camera window to quit.", flush=True)
            with mujoco.viewer.launch_passive(model, data) as viewer:
                viewer.cam.lookat[:] = 0.5 * (lower + upper)
                viewer.cam.distance = 1.2
                while viewer.is_running():
                    snapshot = robot.read()
                    tracker.update_robot_pose(snapshot)
                    gripper = float(snapshot.gripper_position) / 255.0 * 0.72
                    data.qpos[addresses] = np.concatenate((
                        snapshot.joint_position, gripper * GRIPPER_MULTIPLIERS,
                    ))
                    preview = tracker.visualization_snapshot()
                    if preview is not None:
                        image, _, detection, _, _, error = preview
                        fresh = detection is not None and (
                            monotonic() - detection.timestamp
                            <= float(tracker_config["max_state_age_s"])
                        )
                        if not fresh:
                            model.geom_rgba[object_geom, 3] = 0.0
                        if fresh and (
                            last_timestamp is None or detection.timestamp != last_timestamp
                        ):
                            last_timestamp = detection.timestamp
                            show_object(detection.position_base)
                            print(
                                "pixel={} surface_depth={:.4f}m center_depth={:.4f}m "
                                "confidence={:.3f} age={:.3f}s".format(
                                    np.round(detection.center_pixel, 1).tolist(),
                                    detection.surface_depth_m, detection.depth_m,
                                    detection.confidence,
                                    monotonic() - detection.timestamp,
                                ), flush=True,
                            )
                        if cv2 is not None:
                            if fresh:
                                u, v = np.rint(detection.center_pixel).astype(int)
                                cv2.drawMarker(image, (u, v), (255, 0, 0),
                                               cv2.MARKER_CROSS, 24, 2)
                            if error:
                                cv2.putText(image, error[:80], (10, 25),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                            (0, 0, 255), 1)
                            cv2.imshow("UR5e hover wrist camera", image)
                            if cv2.waitKey(1) & 0xFF == ord("q"):
                                break
                    mujoco.mj_forward(model, data)
                    viewer.sync()
                    sleep(0.02)
    finally:
        tracker.stop()
        camera.close()
        if cv2 is not None:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
