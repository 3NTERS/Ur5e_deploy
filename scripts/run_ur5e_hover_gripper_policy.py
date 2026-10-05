#!/usr/bin/env python3
"""Run the 69-observation hover-gripper policy with one wrist RealSense."""

from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
from threading import Event, Lock, Thread
from time import monotonic, sleep

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from onnx_deploy.policy_runner import PolicyRunner
from ur5e_comm.geometry import load_eye_on_hand, tcp_pose_to_transform
from ur5e_comm.observation import (
    JOINT_ORDER,
    MujocoStateProjector,
    SafetyMonitor,
    Ur5eActionMapper,
    Ur5eObservationBuilder,
)
from ur5e_comm.robot import UR5eHardware
from ur5e_comm.vision import (
    RealSenseCamera,
    TrackedObjectState,
    YoloInitialObjectLocator,
)


def resolve_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def vector(mapping, name, size):
    result = np.asarray(mapping[name], dtype=np.float64)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise RuntimeError("{} must contain {} finite values".format(name, size))
    return result


class SingleWristObjectTracker:
    """Asynchronous RGB-D detector transformed with buffered RTDE TCP poses."""

    def __init__(
        self,
        camera,
        locator,
        inference_interval_s,
        max_frame_age_s,
        max_pose_sync_error_s,
        tracking_config,
    ):
        self.camera = camera
        self.locator = locator
        self.inference_interval_s = float(inference_interval_s)
        self.max_frame_age_s = float(max_frame_age_s)
        self.max_pose_sync_error_s = float(max_pose_sync_error_s)
        self.association_max_distance_m = float(
            tracking_config["association_max_distance_m"]
        )
        self.position_alpha = float(tracking_config["position_alpha"])
        self.velocity_alpha = float(tracking_config["velocity_alpha"])
        self.prediction_horizon_s = float(
            tracking_config["prediction_horizon_s"]
        )
        self.max_state_age_s = float(tracking_config["max_state_age_s"])
        self.sample_capacity = max(
            32, 2 * int(tracking_config["initial_measurements"])
        )
        if min(
            self.inference_interval_s,
            self.max_frame_age_s,
            self.max_pose_sync_error_s,
            self.association_max_distance_m,
            self.max_state_age_s,
        ) <= 0.0:
            raise ValueError("camera/tracker timing and distance limits must be positive")
        if not 0.0 < self.position_alpha <= 1.0:
            raise ValueError("tracking.position_alpha must be in (0, 1]")
        if not 0.0 < self.velocity_alpha <= 1.0:
            raise ValueError("tracking.velocity_alpha must be in (0, 1]")
        if self.prediction_horizon_s < 0.0:
            raise ValueError("tracking.prediction_horizon_s cannot be negative")

        self._lock = Lock()
        self._stop = Event()
        self._thread = None
        self._pose_history = deque(maxlen=256)
        self._accepted_positions = deque(maxlen=self.sample_capacity)
        self._frame_timestamp = None
        self._frame_error = None
        self._detection = None
        self._detection_error = None
        self._detection_frame = None
        self._detection_frame_timestamp = None
        self._position = None
        self._velocity = np.zeros(3, dtype=np.float64)
        self._measurement_timestamp = None
        self._measurement_version = 0
        self._reported_version = 0
        self._pose_sync_error_s = np.nan
        self._last_detection_attempt = -np.inf

    def update_robot_pose(self, snapshot):
        transform = tcp_pose_to_transform(snapshot.tcp_pose)
        with self._lock:
            self._pose_history.append((float(snapshot.timestamp), transform))

    def _pose_for_frame_locked(self, frame_timestamp):
        if not self._pose_history:
            raise RuntimeError("no buffered RTDE TCP pose")
        timestamp, transform = min(
            self._pose_history,
            key=lambda item: abs(item[0] - float(frame_timestamp)),
        )
        error = abs(float(timestamp) - float(frame_timestamp))
        if error > self.max_pose_sync_error_s:
            raise RuntimeError(
                "camera/RTDE pose sync error {:.4f}s exceeds {:.4f}s".format(
                    error, self.max_pose_sync_error_s
                )
            )
        return transform.copy(), error

    def _accept_locked(self, detection):
        measured = np.asarray(detection.position_base, dtype=np.float64)
        if measured.shape != (3,) or not np.isfinite(measured).all():
            raise RuntimeError("detected object position is invalid")
        timestamp = float(detection.timestamp)
        if self._position is None or self._measurement_timestamp is None:
            self._position = measured.copy()
            self._velocity[:] = 0.0
        else:
            dt = timestamp - float(self._measurement_timestamp)
            if dt <= 0.0:
                raise RuntimeError("object detection timestamps did not increase")
            if dt > self.max_state_age_s:
                self._position = measured.copy()
                self._velocity[:] = 0.0
            else:
                prediction_time = min(dt, self.prediction_horizon_s)
                reference = self._position + self._velocity * prediction_time
                distance = float(np.linalg.norm(measured - reference))
                if distance > self.association_max_distance_m:
                    raise RuntimeError(
                        "object association jump {:.4f}m exceeds {:.4f}m".format(
                            distance, self.association_max_distance_m
                        )
                    )
                previous_position = self._position.copy()
                filtered = (
                    self.position_alpha * measured
                    + (1.0 - self.position_alpha) * previous_position
                )
                raw_velocity = (filtered - previous_position) / dt
                self._velocity = (
                    self.velocity_alpha * raw_velocity
                    + (1.0 - self.velocity_alpha) * self._velocity
                )
                self._position = filtered
        self._measurement_timestamp = timestamp
        self._detection = detection
        self._detection_error = None
        self._accepted_positions.append(measured.copy())
        self._measurement_version += 1

    def _run(self):
        while not self._stop.is_set():
            try:
                frame = self.camera.read(timeout_ms=250)
                with self._lock:
                    self._frame_timestamp = float(frame.timestamp)
                    self._frame_error = None
                    run_detection = (
                        monotonic() - self._last_detection_attempt
                        >= self.inference_interval_s
                    )
                    if run_detection:
                        self._last_detection_attempt = monotonic()
                        base_to_tcp, pose_error = self._pose_for_frame_locked(
                            frame.timestamp
                        )
                if not run_detection:
                    continue
                detection = None
                detection_error = None
                try:
                    detection = self.locator.locate(frame, base_to_tcp)
                except Exception as error:
                    detection_error = str(error)
                with self._lock:
                    self._detection_frame = frame.color_bgr.copy()
                    self._detection_frame_timestamp = float(frame.timestamp)
                    self._pose_sync_error_s = float(pose_error)
                    if detection is not None:
                        try:
                            self._accept_locked(detection)
                        except Exception as error:
                            detection_error = str(error)
                    if detection_error:
                        self._detection_error = detection_error
            except Exception as error:
                with self._lock:
                    self._frame_error = str(error)
                sleep(0.01)

    def start(self):
        if self._thread is not None:
            raise RuntimeError("wrist object tracker is already running")
        self._stop.clear()
        self._thread = Thread(
            target=self._run, name="hover-wrist-object-tracker", daemon=True
        )
        self._thread.start()

    def stop(self):
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=2.0)
            if thread.is_alive():
                # A first YOLO inference can take several seconds. Closing the
                # camera unblocks any pending RealSense read before the final join.
                self.camera.close()
                thread.join(timeout=5.0)
            if thread.is_alive():
                raise RuntimeError("wrist object tracker did not stop")
        self._thread = None

    def state(self, now=None):
        now = monotonic() if now is None else float(now)
        with self._lock:
            frame_timestamp = self._frame_timestamp
            frame_error = self._frame_error
            detection = self._detection
            position = None if self._position is None else self._position.copy()
            velocity = self._velocity.copy()
            measurement_timestamp = self._measurement_timestamp
            last_error = self._detection_error
            predicted = (
                self._measurement_version == self._reported_version
                or last_error is not None
            )
            self._reported_version = self._measurement_version
        if frame_timestamp is None:
            raise RuntimeError("wrist camera has not produced a frame")
        frame_age = max(0.0, now - float(frame_timestamp))
        if frame_age > self.max_frame_age_s:
            detail = ": {}".format(frame_error) if frame_error else ""
            raise RuntimeError(
                "wrist camera frame is stale by {:.3f}s{}".format(frame_age, detail)
            )
        if detection is None or position is None or measurement_timestamp is None:
            detail = ": {}".format(last_error) if last_error else ""
            raise RuntimeError("wrist camera has no object state{}".format(detail))
        age = max(0.0, now - float(measurement_timestamp))
        if age > self.max_state_age_s:
            detail = ": {}".format(last_error) if last_error else ""
            raise RuntimeError(
                "object state is stale by {:.3f}s{}".format(age, detail)
            )
        prediction_time = min(age, self.prediction_horizon_s)
        return TrackedObjectState(
            position_base=position + velocity * prediction_time,
            linear_velocity_base=velocity,
            measurement_timestamp=float(measurement_timestamp),
            output_timestamp=now,
            confidence=float(detection.confidence),
            predicted=bool(predicted),
            center_pixel=detection.center_pixel.copy(),
            surface_depth_m=float(detection.surface_depth_m),
            center_depth_m=float(detection.depth_m),
        )

    def quality(self, sample_count):
        with self._lock:
            samples = list(self._accepted_positions)[-int(sample_count):]
            velocity = self._velocity.copy()
            version = self._measurement_version
            error = self._detection_error or self._frame_error or ""
            pose_sync_error = self._pose_sync_error_s
        if samples:
            values = np.stack(samples)
            median = np.median(values, axis=0)
            spread = float(np.max(np.linalg.norm(values - median, axis=1)))
        else:
            spread = np.inf
        return {
            "measurement_count": int(version),
            "sample_count": len(samples),
            "position_spread_m": spread,
            "speed_m_s": float(np.linalg.norm(velocity)),
            "pose_sync_error_s": float(pose_sync_error),
            "error": error,
        }

    def visualization_snapshot(self):
        now = monotonic()
        with self._lock:
            if self._detection_frame is None:
                return None
            image = self._detection_frame.copy()
            timestamp = float(self._detection_frame_timestamp)
            detection = self._detection
            error = self._detection_error or ""
            position = None if self._position is None else self._position.copy()
            velocity = self._velocity.copy()
            measurement_timestamp = self._measurement_timestamp
        if position is not None and measurement_timestamp is not None:
            age = max(0.0, now - float(measurement_timestamp))
            if age <= self.max_state_age_s:
                position += velocity * min(age, self.prediction_horizon_s)
            else:
                position = None
                error = error or "object state stale"
        return image, timestamp, detection, position, velocity, error


class CameraVisualizer:
    """Render detector and filtered base-frame state from the main thread."""

    def __init__(self, tracker, window_name, refresh_interval_s):
        self.tracker = tracker
        self.window_name = str(window_name)
        self.refresh_interval_s = float(refresh_interval_s)
        self.cv2 = None
        self.active = False
        self.error = None
        self._last_render = -np.inf

    def start(self):
        try:
            import cv2

            self.cv2 = cv2
            self.cv2.namedWindow(self.window_name, self.cv2.WINDOW_NORMAL)
            self.active = True
        except Exception as error:
            self.error = str(error)
            self.stop()
            raise RuntimeError("camera visualization failed: {}".format(error))

    def update(self, force=False):
        if not self.active:
            return
        try:
            now = monotonic()
            if force or now - self._last_render >= self.refresh_interval_s:
                snapshot = self.tracker.visualization_snapshot()
                if snapshot is not None:
                    image, timestamp, detection, position, velocity, error = snapshot
                    if detection is not None:
                        u, v = np.rint(detection.center_pixel).astype(int)
                        self.cv2.drawMarker(
                            image, (int(u), int(v)), (0, 220, 0),
                            markerType=self.cv2.MARKER_CROSS,
                            markerSize=32, thickness=2,
                        )
                    lines = []
                    if detection is None:
                        lines.append("NO TARGET: {}".format(error or "waiting"))
                    else:
                        lines.append(
                            "{} conf={:.3f} depth={:.3f}m".format(
                                detection.class_name,
                                float(detection.confidence),
                                float(detection.depth_m),
                            )
                        )
                    if position is not None:
                        lines.append(
                            "base xyz=[{:.3f}, {:.3f}, {:.3f}]m".format(*position)
                        )
                        lines.append(
                            "base v=[{:.3f}, {:.3f}, {:.3f}]m/s".format(*velocity)
                        )
                    elif error:
                        lines.append(error)
                    self.cv2.rectangle(
                        image, (0, 0), (image.shape[1], 88), (0, 0, 0), -1
                    )
                    for index, line in enumerate(lines[:3]):
                        self.cv2.putText(
                            image, line, (12, 24 + 27 * index),
                            self.cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                            (230, 230, 230), 1, self.cv2.LINE_AA,
                        )
                    self.cv2.putText(
                        image,
                        "age={:.1f}ms  Q/ESC closes view only".format(
                            1000.0 * max(0.0, now - timestamp)
                        ),
                        (12, image.shape[0] - 14),
                        self.cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                        (255, 255, 255), 1, self.cv2.LINE_AA,
                    )
                    self.cv2.imshow(self.window_name, image)
                    self._last_render = now
            key = self.cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27) or self.cv2.getWindowProperty(
                self.window_name, self.cv2.WND_PROP_VISIBLE
            ) < 1:
                self.stop()
        except Exception as error:
            self.error = str(error)
            self.stop()

    def stop(self):
        was_active = self.active
        self.active = False
        if was_active and self.cv2 is not None:
            try:
                self.cv2.destroyWindow(self.window_name)
                self.cv2.waitKey(1)
            except self.cv2.error:
                pass


def validate_contract(config, policy):
    policy_config = config["policy"]
    model_path = resolve_path(policy_config["model"])
    metadata_path = resolve_path(policy_config["metadata"])
    actual_hash = sha256(model_path)
    if actual_hash != str(policy_config["sha256"]):
        raise RuntimeError("hover ONNX SHA256 mismatch")
    metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("task") != "Ur5eRobotiqHoverGripper":
        raise RuntimeError("metadata task is not Ur5eRobotiqHoverGripper")
    io = metadata["io"]
    if int(io["observation_dim"]) != 69 or policy.observation_dim != 69:
        raise RuntimeError("hover policy must accept 69 observations")
    if int(io["action_dim"]) != 7 or policy.action_dim != 7:
        raise RuntimeError("hover policy must emit 7 actions")
    if tuple(io["joint_order"]) != JOINT_ORDER:
        raise RuntimeError("hover metadata joint order mismatch")
    if bool(metadata.get("network", {}).get("recurrent", True)):
        raise RuntimeError("hover real deployment requires the exported non-recurrent policy")
    contract = metadata["task_contract"]
    period = float(policy_config["period"])
    if abs(period - float(contract["policy_period_s"])) > 1e-9:
        raise RuntimeError("policy period does not match hover metadata")
    if int(policy_config["episode_steps"]) != int(contract["episode_steps"]):
        raise RuntimeError("episode length does not match hover metadata")
    if tuple(tuple(item) for item in contract["observation_zero_slices"]) != (
        (56, 59), (62, 66), (67, 69)
    ):
        raise RuntimeError("hover observation zero slices changed")
    signs = np.asarray(contract["simulation_to_robot_joint_sign"], dtype=np.float64)
    if not np.array_equal(signs, np.ones(6)):
        raise RuntimeError("hover real deployment does not allow legacy joint sign mapping")
    initial = vector(config["robot"], "initial_joint_position", 6)
    if not np.allclose(initial, contract["initial_arm_rad"], atol=1e-6, rtol=0.0):
        raise RuntimeError("configured initial joints do not match hover metadata")
    observation_config = config["observation"]
    observation_model = resolve_path(observation_config["model"])
    if sha256(observation_model) != str(observation_config["model_sha256"]):
        raise RuntimeError("hover MuJoCo model SHA256 mismatch")
    return metadata, actual_hash


def validate_calibration(config):
    camera_config = config["camera"]
    calibration_path = resolve_path(camera_config["calibration"])
    if not calibration_path.is_file():
        raise FileNotFoundError("eye-on-hand calibration does not exist")
    payload = yaml.safe_load(calibration_path.read_text(encoding="utf-8"))
    calibrated_serial = str(
        payload.get("metadata", {}).get("camera", {}).get("serial", "")
    )
    configured_serial = str(camera_config["serial"])
    if calibrated_serial and calibrated_serial != configured_serial:
        raise RuntimeError(
            "calibration camera serial {} does not match {}".format(
                calibrated_serial, configured_serial
            )
        )
    return load_eye_on_hand(
        calibration_path,
        config["calibration"]["checkerboard"],
        camera_config.get("model"),
    )


def require_initial_state(snapshot, initial, safety_config):
    error = float(np.max(np.abs(snapshot.joint_position - initial)))
    tolerance = float(safety_config["initial_joint_tolerance"])
    if error > tolerance:
        raise RuntimeError(
            "initial joint error {:.6f} rad exceeds {:.6f} rad".format(
                error, tolerance
            )
        )
    if int(snapshot.gripper_position) > int(
        safety_config["initial_gripper_open_max"]
    ):
        raise RuntimeError(
            "gripper must be open before progress 0; POS={}".format(
                snapshot.gripper_position
            )
        )


def wait_for_return_start(robot, initial, safety_monitor, return_config):
    deadline = monotonic() + float(return_config["settle_timeout_s"])
    velocity_limit = float(return_config["max_start_joint_velocity"])
    distance_limit = float(return_config["max_joint_distance"])
    while True:
        snapshot = robot.read()
        safety_monitor.check_state(snapshot)
        distance = float(np.max(np.abs(snapshot.joint_position - initial)))
        if distance > distance_limit:
            raise RuntimeError(
                "return start distance {:.6f} rad exceeds {:.6f} rad; "
                "use supervised manual recovery".format(distance, distance_limit)
            )
        velocity = float(np.max(np.abs(snapshot.joint_velocity)))
        if velocity <= velocity_limit:
            return snapshot
        if monotonic() >= deadline:
            raise RuntimeError(
                "arm did not settle before return: {:.6f} rad/s exceeds "
                "{:.6f} rad/s".format(velocity, velocity_limit)
            )
        sleep(0.05)


def validate_return_path(projector, start, initial, object_position,
                         safety_monitor, return_config):
    """Check a joint-space return against the configured MuJoCo scene."""
    count = int(return_config["path_samples"])
    if count < 2:
        raise RuntimeError("return_to_initial.path_samples must be >= 2")
    object_position = np.asarray(object_position, dtype=np.float64)
    if object_position.shape != (3,) or not np.isfinite(object_position).all():
        raise RuntimeError("return path requires a finite observed object position")
    mujoco = projector.mujoco
    model = projector.model
    data = projector.data
    object_joint = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "object_freejoint"
    )
    robot_root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    if object_joint < 0 or robot_root < 0:
        raise RuntimeError("hover model lacks object_freejoint or robot base")
    object_address = int(model.jnt_qposadr[object_joint])
    data.qpos[object_address:object_address + 3] = object_position
    data.qpos[object_address + 3:object_address + 7] = [1.0, 0.0, 0.0, 0.0]

    def robot_body(body):
        while body != 0:
            if body == robot_root:
                return True
            body = int(model.body_parentid[body])
        return False

    for index, fraction in enumerate(np.linspace(0.0, 1.0, count)):
        joint = (1.0 - fraction) * start + fraction * initial
        if np.any(joint < safety_monitor.lower) or np.any(joint > safety_monitor.upper):
            raise RuntimeError("return path waypoint {} leaves joint corridor".format(index))
        palm = projector.palm_position_for(joint, 0)
        if np.any(palm < safety_monitor.workspace_min) or np.any(
            palm > safety_monitor.workspace_max
        ):
            raise RuntimeError(
                "return path waypoint {} palm {} leaves workspace".format(
                    index, palm.tolist()
                )
            )
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            first = int(model.geom_bodyid[contact.geom1])
            second = int(model.geom_bodyid[contact.geom2])
            if robot_body(first) or robot_body(second):
                raise RuntimeError(
                    "return path waypoint {} has MuJoCo robot contact "
                    "between {} and {}".format(
                        index,
                        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, first),
                        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, second),
                    )
                )
    return count


def prepare_open_gripper(robot, robot_config, safety_config):
    threshold = int(safety_config["initial_gripper_open_max"])
    position = int(robot.gripper.get("POS"))
    if position <= threshold:
        print("gripper already open: POS={}".format(position))
        return position
    print("opening empty gripper before camera lock: POS={}".format(position))
    robot.gripper.move(
        0, int(robot_config["gripper_speed"]), int(robot_config["gripper_force"])
    )
    deadline = monotonic() + float(robot_config["gripper_open_timeout_s"])
    while monotonic() < deadline:
        position = int(robot.gripper.get("POS"))
        if position <= threshold:
            print("gripper open confirmed: POS={}".format(position))
            return position
        sleep(0.05)
    robot.gripper.stop()
    raise TimeoutError("Robotiq did not reach the configured open threshold")


def wait_for_initial_object(
    robot,
    tracker,
    visualizer,
    initial,
    safety_monitor,
    safety_config,
    tracking_config,
    timeout_s,
):
    required = int(tracking_config["initial_measurements"])
    deadline = monotonic() + float(timeout_s)
    next_report = monotonic()
    while monotonic() < deadline:
        snapshot = robot.read()
        safety_monitor.check_state(snapshot)
        require_initial_state(snapshot, initial, safety_config)
        tracker.update_robot_pose(snapshot)
        if visualizer is not None:
            visualizer.update()
        quality = tracker.quality(required)
        if monotonic() >= next_report:
            print(
                "camera_lock measurements={}/{} spread={:.4f}m speed={:.4f}m/s {}".format(
                    quality["sample_count"], required,
                    quality["position_spread_m"], quality["speed_m_s"],
                    quality["error"],
                )
            )
            next_report = monotonic() + 1.0
        if (
            quality["sample_count"] >= required
            and quality["position_spread_m"]
            <= float(tracking_config["initial_max_position_spread_m"])
            and quality["speed_m_s"]
            <= float(tracking_config["initial_max_speed_m_s"])
        ):
            state = tracker.state()
            return state, snapshot, quality
        sleep(0.01)
    quality = tracker.quality(required)
    raise RuntimeError(
        "initial object lock timed out: measurements={}/{} spread={:.4f}m "
        "speed={:.4f}m/s error={}".format(
            quality["sample_count"], required,
            quality["position_spread_m"], quality["speed_m_s"], quality["error"],
        )
    )


def save_result(output_directory, episode_id, rows, summary, save_object_csv):
    output_directory.mkdir(parents=True, exist_ok=True)
    trajectory_path = output_directory / "episode_{}.npz".format(episode_id)
    summary_path = output_directory / "episode_{}.json".format(episode_id)
    np.savez_compressed(
        str(trajectory_path),
        **{name: np.asarray(values) for name, values in rows.items()}
    )
    summary["trajectory"] = str(trajectory_path)
    object_csv_path = None
    if save_object_csv:
        object_csv_path = output_directory / "episode_{}.objects.csv".format(
            episode_id
        )
        if rows["progress"]:
            object_values = np.column_stack(
                (
                    np.asarray(rows["progress"], dtype=np.int64),
                    np.stack(rows["object_position"]),
                    np.stack(rows["object_linear_velocity"]),
                    np.asarray(rows["object_measurement_age_s"], dtype=np.float64),
                    np.asarray(rows["object_predicted"], dtype=np.int64),
                    np.asarray(rows["object_confidence"], dtype=np.float64),
                    np.asarray(rows["hover_distance_m"], dtype=np.float64),
                    np.asarray(rows["object_displacement_m"], dtype=np.float64),
                )
            )
        else:
            object_values = np.empty((0, 12), dtype=np.float64)
        np.savetxt(
            str(object_csv_path),
            object_values,
            delimiter=",",
            header=(
                "progress,position_base_x,position_base_y,position_base_z,"
                "velocity_base_x,velocity_base_y,velocity_base_z,"
                "measurement_age_s,predicted,confidence,hover_distance_m,"
                "object_displacement_m"
            ),
            comments="",
            fmt=["%d"] + ["%.9g"] * 7 + ["%d"] + ["%.9g"] * 3,
        )
        summary["object_csv"] = str(object_csv_path)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return trajectory_path, summary_path, object_csv_path


def run(args):
    config_path = resolve_path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    policy_config = config["policy"]
    robot_config = config["robot"]
    observation_config = config["observation"]
    camera_config = config["camera"]
    vision_config = config["vision"]
    tracking_config = config["tracking"]
    safety_config = config["safety"]
    output_config = config["output"]
    visualization_config = vision_config.get("visualization", {})
    return_config = config.get("return_to_initial", {})
    if bool(return_config.get("enabled", False)):
        for key in (
            "joint_speed", "joint_acceleration", "position_tolerance",
            "max_start_joint_velocity", "settle_timeout_s", "max_joint_distance",
        ):
            value = float(return_config[key])
            if not np.isfinite(value) or value <= 0.0:
                raise RuntimeError("return_to_initial.{} must be positive".format(key))
        if int(return_config["path_samples"]) < 2:
            raise RuntimeError("return_to_initial.path_samples must be >= 2")

    if args.execute and not bool(safety_config.get("allow_motion", False)):
        raise RuntimeError(
            "refusing motion: set safety.allow_motion=true and pass --execute"
        )
    if not camera_config.get("serial"):
        raise RuntimeError("camera.serial must contain the wrist D435i serial")
    if camera_config.get("enabled") is not True or vision_config.get("enabled") is not True:
        raise RuntimeError("hover deployment requires camera and vision enabled")
    if args.steps <= 0 or args.steps > int(policy_config["episode_steps"]):
        raise RuntimeError("--steps must be in [1, {}]".format(policy_config["episode_steps"]))

    policy = PolicyRunner(
        resolve_path(policy_config["model"]),
        args.provider or policy_config["provider"],
    )
    metadata, model_hash = validate_contract(config, policy)
    tcp_to_camera = validate_calibration(config)
    contract = metadata["task_contract"]
    period = float(policy_config["period"])
    initial = vector(robot_config, "initial_joint_position", 6)
    arm_lower = vector(safety_config, "arm_lower", 6)
    arm_upper = vector(safety_config, "arm_upper", 6)
    if np.any(arm_lower >= arm_upper) or np.any(initial < arm_lower) or np.any(initial > arm_upper):
        raise RuntimeError("invalid hover deployment joint corridor")

    projector = MujocoStateProjector(
        resolve_path(observation_config["model"]),
        observation_config["palm_center_offset"],
    )
    hover_height = float(safety_config["hover_height_m"])
    observation_builder = Ur5eObservationBuilder(
        projector,
        np.zeros(3, dtype=np.float64),
        observation_config["object_quaternion_xyzw"],
        observation_config["object_scale"],
        subtask="hover_gripper",
        goal_offset=np.asarray((0.0, 0.0, hover_height)),
    )
    action_mapper = Ur5eActionMapper(
        period=period,
        speed_scale=float(contract["arm_action_scale"]),
        max_arm_step=float(safety_config["max_arm_step"]),
        arm_lower=arm_lower,
        arm_upper=arm_upper,
        action_scale=1.0,
        max_arm_velocity=float(contract["max_command_joint_velocity_rad_s"]),
        max_arm_acceleration=float(
            contract["max_command_joint_acceleration_rad_s2"]
        ),
    )
    safety_monitor = SafetyMonitor(
        arm_lower,
        arm_upper,
        float(safety_config["max_joint_velocity"]),
        float(safety_config["max_target_error"]),
        float(safety_config["max_state_age_s"]),
        safety_config["workspace_min"],
        safety_config["workspace_max"],
        safety_config["object_position_min"],
        safety_config["object_position_max"],
    )
    locator = YoloInitialObjectLocator(
        resolve_path(vision_config["weights"]),
        tcp_to_camera,
        vision_config.get("target_class"),
        vision_config["confidence"],
        vision_config["depth_radius"],
        vision_config["depth_min"],
        vision_config["depth_max"],
        vision_config["center_depth_offset_m"],
        vision_config.get("device"),
    )

    if args.execute:
        confirmation = input(
            "即将使用腕部相机目标状态执行 hover 策略。确认夹爪为空、工作空间无人且急停可达后输入 ARM："
        ).strip()
        if confirmation != "ARM":
            raise RuntimeError("motion arming cancelled")

    episode_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    summary = {
        "episode_id": episode_id,
        "utc_time": datetime.now(timezone.utc).isoformat(),
        "task": "Ur5eRobotiqHoverGripper",
        "model_sha256": model_hash,
        "executed": bool(args.execute),
        "requested_steps": int(args.steps),
        "completed_steps": 0,
        "success": False,
        "error": None,
    }
    row_names = (
        "progress", "monotonic_timestamp", "observation", "action",
        "limited_action", "arm_target", "gripper_target",
        "joint_position", "joint_velocity", "tcp_pose", "tcp_speed",
        "gripper_position", "palm_position", "fingertip_position",
        "object_position", "object_linear_velocity",
        "object_measurement_age_s", "object_predicted", "object_confidence",
        "object_center_pixel", "object_surface_depth_m", "object_center_depth_m",
        "hover_distance_m", "object_displacement_m", "target_tracking_error_rad",
        "onnx_inference_time_ms", "cycle_duration_s",
    )
    rows = {name: [] for name in row_names}
    camera = None
    tracker = None
    visualizer = None
    motion_started = False
    success_steps = 0
    max_cycle_duration = 0.0
    initial_object_position = None
    target_error_cycles = 0

    try:
        camera = RealSenseCamera(
            camera_config["width"], camera_config["height"],
            camera_config["fps"], camera_config["serial"],
            camera_config.get("model"),
        )
        tracker = SingleWristObjectTracker(
            camera, locator, vision_config["inference_interval_s"],
            camera_config["max_frame_age_s"],
            camera_config["max_pose_sync_error_s"], tracking_config,
        )
        summary["camera"] = {
            "device_name": camera.device_name,
            "serial": camera.serial,
            "calibrated": True,
            "inference_interval_s": float(vision_config["inference_interval_s"]),
        }
        print(
            "wrist_camera={!r} serial={} calibrated=True; camera object state feeds ONNX".format(
                camera.device_name, camera.serial
            )
        )
        with UR5eHardware(
            robot_config["host"], robot_config["gripper_port"],
            servo_period=robot_config["servo_period"],
            servo_speed=robot_config["servo_speed"],
            servo_acceleration=robot_config["servo_acceleration"],
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
            if args.execute and bool(robot_config["prepare_gripper_open"]):
                prepare_open_gripper(robot, robot_config, safety_config)
            first_snapshot = robot.read()
            safety_monitor.check_state(first_snapshot)
            require_initial_state(first_snapshot, initial, safety_config)
            tracker.update_robot_pose(first_snapshot)
            tracker.start()
            if bool(visualization_config.get("enabled", False)):
                visualizer = CameraVisualizer(
                    tracker,
                    visualization_config.get(
                        "window_name", "UR5e hover wrist-camera diagnostics"
                    ),
                    visualization_config.get("refresh_interval_s", 0.05),
                )
                visualizer.start()
            initial_object, snapshot, lock_quality = wait_for_initial_object(
                robot, tracker, visualizer, initial, safety_monitor,
                safety_config, tracking_config,
                camera_config["initial_lock_timeout_s"],
            )
            safety_monitor.check_initial_object(initial_object.position_base)
            initial_object_position = initial_object.position_base.copy()
            summary["initial_object_lock"] = {
                "position_base": initial_object_position.tolist(),
                "linear_velocity_base": initial_object.linear_velocity_base.tolist(),
                "confidence": float(initial_object.confidence),
                "position_spread_m": float(lock_quality["position_spread_m"]),
                "speed_m_s": float(lock_quality["speed_m_s"]),
                "pose_sync_error_s": float(lock_quality["pose_sync_error_s"]),
            }
            print(
                "object_lock position_base={} velocity_base={}".format(
                    np.array2string(initial_object.position_base, precision=5),
                    np.array2string(initial_object.linear_velocity_base, precision=5),
                )
            )

            policy.reset()
            projector.reset()
            observation_builder.begin_episode(initial_object_position)
            action_mapper.reset(snapshot)
            # Refresh after YOLO warm-up and initial locking, immediately before motion.
            snapshot = robot.read()
            safety_monitor.check_state(snapshot)
            require_initial_state(snapshot, initial, safety_config)
            tracker.update_robot_pose(snapshot)
            if args.execute:
                robot.start_motion()
                motion_started = True

            started = monotonic()
            for progress in range(int(args.steps)):
                tick = monotonic()
                if progress > 0:
                    snapshot = robot.read()
                safety_monitor.check_state(snapshot)
                tracker.update_robot_pose(snapshot)
                object_state = tracker.state()
                object_speed = float(
                    np.linalg.norm(object_state.linear_velocity_base)
                )
                if object_speed > float(tracking_config["max_object_speed_m_s"]):
                    raise RuntimeError(
                        "estimated object speed {:.4f}m/s exceeds {:.4f}m/s".format(
                            object_speed,
                            float(tracking_config["max_object_speed_m_s"]),
                        )
                    )
                safety_monitor.check_object_position(object_state.position_base)
                observation, projected = observation_builder.build(
                    snapshot, object_state
                )
                if not (
                    np.all(observation[56:59] == 0.0)
                    and np.all(observation[62:66] == 0.0)
                    and np.all(observation[67:69] == 0.0)
                ):
                    raise RuntimeError("hover observation zero-slice contract violated")
                if not np.allclose(
                    observation[37:41],
                    observation_builder.object_quaternion,
                    atol=1e-6,
                    rtol=0.0,
                ):
                    raise RuntimeError("fixed object quaternion was not mapped into obs[37:41]")
                if not np.allclose(
                    observation[41:44],
                    object_state.linear_velocity_base,
                    atol=1e-6,
                    rtol=0.0,
                ):
                    raise RuntimeError("camera object velocity was not mapped into obs[41:44]")
                if not np.all(observation[44:47] == 0.0):
                    raise RuntimeError("object angular velocity in obs[44:47] must stay zero")
                expected_relative = (
                    object_state.position_base - projected.palm_center_position
                )
                if not np.allclose(
                    observation[53:56], expected_relative, atol=1e-6, rtol=0.0
                ):
                    raise RuntimeError("camera object position was not mapped into obs[53:56]")

                inference_started = monotonic()
                action = np.clip(policy.infer(observation)[0], -1.0, 1.0)
                inference_time_ms = (monotonic() - inference_started) * 1000.0
                if not args.execute:
                    action_mapper.reset(snapshot)
                arm_target, gripper_target = action_mapper.map(action)
                safety_monitor.check_target(
                    snapshot, arm_target, gripper_target, projector
                )
                target_error = float(
                    np.max(np.abs(snapshot.joint_position - arm_target))
                )
                if args.execute and target_error > float(safety_config["max_target_error"]):
                    target_error_cycles += 1
                else:
                    target_error_cycles = 0
                if target_error_cycles >= int(safety_config["max_target_error_cycles"]):
                    raise RuntimeError("joint target tracking watchdog expired")

                hover_target = object_state.position_base + np.asarray(
                    (0.0, 0.0, hover_height), dtype=np.float64
                )
                hover_distance = float(
                    np.linalg.norm(projected.palm_center_position - hover_target)
                )
                object_displacement = float(
                    np.linalg.norm(
                        object_state.position_base - initial_object_position
                    )
                )
                if (
                    hover_distance <= float(safety_config["success_radius_m"])
                    and object_displacement
                    <= float(safety_config["max_object_displacement_m"])
                ):
                    success_steps += 1
                else:
                    success_steps = 0

                if args.execute:
                    robot.command(arm_target, gripper_target)
                if visualizer is not None:
                    visualizer.update()

                values = {
                    "progress": progress,
                    "monotonic_timestamp": snapshot.timestamp,
                    "observation": observation,
                    "action": action,
                    "limited_action": action_mapper.last_limited_action.copy(),
                    "arm_target": arm_target,
                    "gripper_target": gripper_target,
                    "joint_position": snapshot.joint_position,
                    "joint_velocity": snapshot.joint_velocity,
                    "tcp_pose": snapshot.tcp_pose,
                    "tcp_speed": snapshot.tcp_speed,
                    "gripper_position": snapshot.gripper_position,
                    "palm_position": projected.palm_center_position,
                    "fingertip_position": projected.fingertip_position,
                    "object_position": object_state.position_base,
                    "object_linear_velocity": object_state.linear_velocity_base,
                    "object_measurement_age_s": object_state.measurement_age_s,
                    "object_predicted": object_state.predicted,
                    "object_confidence": object_state.confidence,
                    "object_center_pixel": object_state.center_pixel,
                    "object_surface_depth_m": object_state.surface_depth_m,
                    "object_center_depth_m": object_state.center_depth_m,
                    "hover_distance_m": hover_distance,
                    "object_displacement_m": object_displacement,
                    "target_tracking_error_rad": target_error,
                    "onnx_inference_time_ms": inference_time_ms,
                }
                for name, value in values.items():
                    rows[name].append(value)
                observation_builder.advance()
                summary["completed_steps"] = progress + 1

                cycle_duration = monotonic() - tick
                rows["cycle_duration_s"].append(cycle_duration)
                max_cycle_duration = max(max_cycle_duration, cycle_duration)
                if cycle_duration > float(safety_config["max_cycle_duration_s"]):
                    raise RuntimeError(
                        "policy cycle took {:.6f}s, exceeding {:.6f}s".format(
                            cycle_duration,
                            float(safety_config["max_cycle_duration_s"]),
                        )
                    )
                deadline = started + (progress + 1) * period
                remaining = deadline - monotonic()
                if remaining < -float(safety_config["max_policy_lag_s"]):
                    raise RuntimeError(
                        "policy loop missed schedule by {:.6f}s".format(-remaining)
                    )
                if remaining > 0.0:
                    sleep(remaining)

            summary["final_success_steps"] = int(success_steps)
            summary["success"] = bool(
                args.execute
                and args.steps == int(policy_config["episode_steps"])
                and success_steps >= int(safety_config["consecutive_success_steps"])
            )
            if (
                args.execute
                and args.steps == int(policy_config["episode_steps"])
                and bool(return_config.get("enabled", False))
            ):
                # Only a completed episode reaches this point. On any watchdog
                # error the exception path stops the servo and never returns.
                robot.stop_motion()
                motion_started = False
                return_summary = {
                    "offered": False, "confirmed": False, "executed": False,
                    "success": False,
                }
                summary["return_to_initial"] = return_summary
                start = wait_for_return_start(
                    robot, initial, safety_monitor, return_config
                )
                object_for_return = np.asarray(rows["object_position"][-1])
                return_summary["path_samples"] = validate_return_path(
                    projector, start.joint_position, initial,
                    object_for_return, safety_monitor, return_config,
                )
                return_summary["start_joint_position"] = (
                    start.joint_position.tolist()
                )
                return_summary["object_position_base"] = (
                    object_for_return.tolist()
                )
                return_summary["offered"] = True
                print(
                    "策略伺服已停止。回到初始位姿将打开空夹爪，并以 {:.2f} "
                    "rad/s 沿关节空间返回。确认夹爪无物体、整条回程路径无障碍、"
                    "现场人员已撤离且急停可达后输入 RETURN；直接回车则保持当前位置。"
                    .format(float(return_config["joint_speed"]))
                )
                try:
                    confirmation = input("RETURN: ").strip()
                except EOFError:
                    confirmation = ""
                if confirmation == "RETURN":
                    # Re-read and re-plan after the operator has confirmed; do
                    # not trust the position sampled before the input prompt.
                    start = wait_for_return_start(
                        robot, initial, safety_monitor, return_config
                    )
                    validate_return_path(
                        projector, start.joint_position, initial,
                        object_for_return, safety_monitor, return_config,
                    )
                    return_summary["confirmed"] = True
                    return_summary["opened_gripper_position"] = (
                        prepare_open_gripper(robot, robot_config, safety_config)
                    )
                    return_summary["executed"] = True
                    robot.home(
                        initial,
                        speed=float(return_config["joint_speed"]),
                        acceleration=float(return_config["joint_acceleration"]),
                        tolerance=float(return_config["position_tolerance"]),
                    )
                    final = wait_for_return_start(
                        robot, initial, safety_monitor, return_config
                    )
                    final_error = float(np.max(np.abs(
                        final.joint_position - initial
                    )))
                    if final_error > float(return_config["position_tolerance"]):
                        raise RuntimeError(
                            "return final joint error {:.6f} rad exceeds "
                            "{:.6f} rad".format(
                                final_error,
                                float(return_config["position_tolerance"]),
                            )
                        )
                    return_summary["final_joint_position"] = (
                        final.joint_position.tolist()
                    )
                    return_summary["final_joint_error_rad"] = final_error
                    return_summary["success"] = True
                else:
                    print("未确认 RETURN；机械臂保持当前位置，不自动回位。")
    except Exception as error:
        summary["error"] = str(error)
        raise
    finally:
        if motion_started:
            # UR5eHardware context cleanup also stops, but stop here before camera/UI cleanup.
            try:
                robot.stop_motion()
            except Exception as stop_error:
                summary["motion_stop_error"] = str(stop_error)
                summary["success"] = False
        if visualizer is not None:
            visualizer.stop()
            if visualizer.error:
                summary["visualization_error"] = visualizer.error
        if tracker is not None:
            try:
                tracker.stop()
            except Exception as tracker_error:
                summary["tracker_cleanup_error"] = str(tracker_error)
                summary["success"] = False
        if camera is not None:
            camera.close()
        summary["max_cycle_duration_s"] = float(max_cycle_duration)
        trajectory_path, summary_path, object_csv_path = save_result(
            resolve_path(output_config["directory"]), episode_id, rows, summary,
            bool(output_config.get("save_object_csv", True)),
        )
        print("trajectory={}".format(trajectory_path))
        print("summary={}".format(summary_path))
        if object_csv_path is not None:
            print("object_csv={}".format(object_csv_path))
        if bool(output_config.get("print_object_state", False)):
            for index, progress in enumerate(rows["progress"]):
                print(
                    "OBJECT_STATE progress={:03d} position_base={} velocity_base={} "
                    "age={:.4f}s predicted={}".format(
                        int(progress),
                        np.array2string(
                            np.asarray(rows["object_position"][index]), precision=6,
                            separator=",",
                        ),
                        np.array2string(
                            np.asarray(rows["object_linear_velocity"][index]),
                            precision=6, separator=",",
                        ),
                        float(rows["object_measurement_age_s"][index]),
                        bool(rows["object_predicted"][index]),
                    )
                )
        if bool(output_config.get("print_observation_69", False)):
            for progress, observation in zip(rows["progress"], rows["observation"]):
                print(
                    "OBSERVATION_69 progress={:03d} {}".format(
                        int(progress),
                        np.array2string(
                            np.asarray(observation), precision=6, separator=",",
                            max_line_width=10000,
                        ),
                    )
                )

    print("HOVER_GRIPPER_REAL " + json.dumps(summary, sort_keys=True))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="resources/config/ur5e_hover_gripper_deploy.yaml"
    )
    parser.add_argument("--provider", choices=("cpu", "cuda"))
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument(
        "--execute", action="store_true",
        help="send position targets to real hardware; also requires allow_motion=true",
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
