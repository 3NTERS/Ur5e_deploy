#!/usr/bin/env python3
"""Run the fixed 240-step Ur5eRobotiqLiftGripper policy on real hardware."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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
from ur5e_comm.observation import JOINT_ORDER, MujocoStateProjector
from ur5e_comm.robot import RobotSnapshot, UR5eHardware
from ur5e_comm.vision import RealSenseCamera, YoloInitialObjectLocator


ALLOWED_STOP_STEPS = (30, 75, 195, 240)
STAGE_NAMES = {
    0: "open_hold",
    30: "close_hold",
    75: "raise_closed",
    195: "raised_open",
}


class WristCameraMonitor:
    """Asynchronous single wrist-camera diagnostics, independent of policy input."""

    def __init__(
        self,
        camera,
        max_frame_age_s,
        detector=None,
        calibrated=False,
        detection_interval_s=0.20,
        object_tracking=None,
    ):
        self.camera = camera
        self.max_frame_age_s = float(max_frame_age_s)
        self.detector = detector
        self.calibrated = bool(calibrated)
        self.detection_interval_s = float(detection_interval_s)
        tracking = object_tracking or {}
        self.object_tracking_enabled = bool(tracking.get("enabled", False))
        self.association_max_distance_m = float(
            tracking.get("association_max_distance_m", 0.25)
        )
        self.position_alpha = float(tracking.get("position_alpha", 0.60))
        self.velocity_alpha = float(tracking.get("velocity_alpha", 0.40))
        self.prediction_horizon_s = float(
            tracking.get("prediction_horizon_s", 0.10)
        )
        self.object_max_state_age_s = float(tracking.get("max_state_age_s", 0.50))
        if self.max_frame_age_s <= 0.0:
            raise ValueError("camera max_frame_age_s must be positive")
        if self.detection_interval_s <= 0.0:
            raise ValueError("vision inference_interval_s must be positive")
        if not 0.0 < self.position_alpha <= 1.0:
            raise ValueError("vision object_tracking.position_alpha must be in (0, 1]")
        if not 0.0 < self.velocity_alpha <= 1.0:
            raise ValueError("vision object_tracking.velocity_alpha must be in (0, 1]")
        if (
            self.association_max_distance_m <= 0.0
            or self.prediction_horizon_s < 0.0
            or self.object_max_state_age_s <= 0.0
        ):
            raise ValueError("vision object_tracking limits are invalid")
        self._lock = Lock()
        self._stop = Event()
        self._first_frame = Event()
        self._thread = None
        self._frame_index = 0
        self._frame_timestamp = None
        self._frame_error = None
        self._detection = None
        self._detection_error = None
        self._base_to_tcp = None
        self._robot_timestamp = None
        self._detection_robot_timestamp = None
        self._last_detection_attempt = -np.inf
        self._detection_frame_bgr = None
        self._detection_frame_timestamp = None
        self._object_raw_position_base = None
        self._object_position_base = None
        self._object_linear_velocity_base = np.zeros(3, dtype=np.float64)
        self._object_measurement_timestamp = None
        self._object_confidence = np.nan
        self._object_tracking_error = None
        self._object_measurement_version = 0
        self._object_reported_version = 0

    def start(self):
        if self._thread is not None:
            raise RuntimeError("wrist camera monitor is already running")
        self._stop.clear()
        self._first_frame.clear()
        self._thread = Thread(target=self._run, name="wrist-camera", daemon=True)
        self._thread.start()

    def wait_for_first_frame(self, timeout_s):
        if not self._first_frame.wait(float(timeout_s)):
            with self._lock:
                detail = self._frame_error or "no frame received"
            raise RuntimeError("wrist camera startup timed out: {}".format(detail))

    def update_robot_pose(self, snapshot):
        transform = tcp_pose_to_transform(snapshot.tcp_pose)
        with self._lock:
            self._base_to_tcp = transform
            self._robot_timestamp = float(snapshot.timestamp)

    def _update_object_tracking_locked(self, detection, detection_error):
        if not self.object_tracking_enabled:
            return
        if detection is None:
            if detection_error:
                self._object_tracking_error = str(detection_error)
            return

        measured = np.asarray(detection.position_base, dtype=np.float64)
        timestamp = float(detection.timestamp)
        if self._object_position_base is None:
            self._object_raw_position_base = measured.copy()
            self._object_position_base = measured.copy()
            self._object_linear_velocity_base[:] = 0.0
            self._object_measurement_timestamp = timestamp
        else:
            dt = timestamp - float(self._object_measurement_timestamp)
            if dt <= 0.0:
                self._object_tracking_error = "non-increasing object timestamp"
                return
            if dt > self.object_max_state_age_s:
                self._object_raw_position_base = measured.copy()
                self._object_position_base = measured.copy()
                self._object_linear_velocity_base[:] = 0.0
                self._object_measurement_timestamp = timestamp
            else:
                prediction_time = min(dt, self.prediction_horizon_s)
                reference = (
                    self._object_position_base
                    + self._object_linear_velocity_base * prediction_time
                )
                association_distance = float(np.linalg.norm(measured - reference))
                if association_distance > self.association_max_distance_m:
                    self._object_tracking_error = (
                        "object association jump {:.4f}m exceeds {:.4f}m".format(
                            association_distance, self.association_max_distance_m
                        )
                    )
                    return
                previous_position = self._object_position_base.copy()
                filtered_position = (
                    self.position_alpha * measured
                    + (1.0 - self.position_alpha) * previous_position
                )
                raw_velocity = (filtered_position - previous_position) / dt
                self._object_linear_velocity_base = (
                    self.velocity_alpha * raw_velocity
                    + (1.0 - self.velocity_alpha)
                    * self._object_linear_velocity_base
                )
                self._object_raw_position_base = measured.copy()
                self._object_position_base = filtered_position
                self._object_measurement_timestamp = timestamp
        self._object_confidence = float(detection.confidence)
        self._object_tracking_error = None
        self._object_measurement_version += 1

    def warmup_detector(self):
        """Run lazy YOLO initialization before the real-time arm loop starts."""
        if self.detector is None:
            return None
        frame = self.camera.read(timeout_ms=5000)
        with self._lock:
            base_to_tcp = (
                None if self._base_to_tcp is None else self._base_to_tcp.copy()
            )
            robot_timestamp = self._robot_timestamp
        detection = None
        detection_error = None
        try:
            if self.calibrated:
                if base_to_tcp is None:
                    raise RuntimeError("waiting for synchronized RTDE TCP pose")
                detection = self.detector.locate(frame, base_to_tcp)
            else:
                detection = self.detector.locate_with_base_to_camera(
                    frame, np.eye(4, dtype=np.float64)
                )
        except Exception as error:
            detection_error = str(error)
        with self._lock:
            self._frame_index += 1
            self._frame_timestamp = float(frame.timestamp)
            self._frame_error = None
            self._detection = detection
            self._detection_error = detection_error
            self._detection_robot_timestamp = (
                robot_timestamp if detection is not None else None
            )
            self._detection_frame_bgr = frame.color_bgr.copy()
            self._detection_frame_timestamp = float(frame.timestamp)
            self._update_object_tracking_locked(detection, detection_error)
            self._last_detection_attempt = monotonic()
        return detection_error

    def _run(self):
        while not self._stop.is_set():
            try:
                frame = self.camera.read(timeout_ms=250)
                with self._lock:
                    self._frame_index += 1
                    self._frame_timestamp = float(frame.timestamp)
                    self._frame_error = None
                    base_to_tcp = (
                        None if self._base_to_tcp is None else self._base_to_tcp.copy()
                    )
                    robot_timestamp = self._robot_timestamp
                self._first_frame.set()

                run_detection = (
                    self.detector is not None
                    and monotonic() - self._last_detection_attempt
                    >= self.detection_interval_s
                )
                if run_detection:
                    detection = None
                    detection_error = None
                    try:
                        if self.calibrated:
                            if base_to_tcp is None:
                                raise RuntimeError("waiting for synchronized RTDE TCP pose")
                            detection = self.detector.locate(frame, base_to_tcp)
                        else:
                            detection = self.detector.locate_with_base_to_camera(
                                frame, np.eye(4, dtype=np.float64)
                            )
                    except Exception as error:
                        detection_error = str(error)
                    with self._lock:
                        self._detection = detection
                        self._detection_error = detection_error
                        self._detection_robot_timestamp = (
                            robot_timestamp if detection is not None else None
                        )
                        self._detection_frame_bgr = frame.color_bgr.copy()
                        self._detection_frame_timestamp = float(frame.timestamp)
                        self._update_object_tracking_locked(
                            detection, detection_error
                        )
                        self._last_detection_attempt = monotonic()
            except Exception as error:
                with self._lock:
                    self._frame_error = str(error)

    def state(self, now=None):
        now = monotonic() if now is None else float(now)
        with self._lock:
            frame_index = self._frame_index
            frame_timestamp = self._frame_timestamp
            frame_error = self._frame_error
            detection = self._detection
            detection_error = self._detection_error
            robot_timestamp = self._robot_timestamp
            detection_robot_timestamp = self._detection_robot_timestamp
            object_raw_position = (
                None
                if self._object_raw_position_base is None
                else self._object_raw_position_base.copy()
            )
            object_position = (
                None
                if self._object_position_base is None
                else self._object_position_base.copy()
            )
            object_velocity = self._object_linear_velocity_base.copy()
            object_measurement_timestamp = self._object_measurement_timestamp
            object_confidence = self._object_confidence
            object_tracking_error = self._object_tracking_error
            object_predicted = (
                self._object_measurement_version == self._object_reported_version
                or object_tracking_error is not None
            )
            self._object_reported_version = self._object_measurement_version
        if frame_timestamp is None:
            raise RuntimeError("wrist camera has not produced a frame")
        frame_age = max(0.0, now - frame_timestamp)
        if frame_age > self.max_frame_age_s:
            detail = ": {}".format(frame_error) if frame_error else ""
            raise RuntimeError(
                "wrist camera frame is stale by {:.3f}s{}".format(frame_age, detail)
            )
        if detection is None:
            position_camera = np.full(3, np.nan, dtype=np.float64)
            position_base = np.full(3, np.nan, dtype=np.float64)
            center_pixel = np.full(2, np.nan, dtype=np.float64)
            confidence = np.nan
            class_id = -1
            class_name = ""
            detection_timestamp = np.nan
        else:
            position_camera = detection.position_camera.copy()
            position_base = (
                detection.position_base.copy()
                if self.calibrated
                else np.full(3, np.nan, dtype=np.float64)
            )
            center_pixel = detection.center_pixel.copy()
            confidence = float(detection.confidence)
            class_id = int(detection.class_id)
            class_name = str(detection.class_name)
            detection_timestamp = float(detection.timestamp)
        if object_position is None or object_measurement_timestamp is None:
            object_tracking_valid = False
            object_measurement_age = np.nan
            object_raw_position = np.full(3, np.nan, dtype=np.float64)
            object_position = np.full(3, np.nan, dtype=np.float64)
            object_velocity = np.full(3, np.nan, dtype=np.float64)
            object_confidence = np.nan
        else:
            object_measurement_age = max(
                0.0, now - float(object_measurement_timestamp)
            )
            object_tracking_valid = (
                object_measurement_age <= self.object_max_state_age_s
            )
            if object_tracking_valid:
                prediction_time = min(
                    object_measurement_age, self.prediction_horizon_s
                )
                object_position = object_position + object_velocity * prediction_time
            else:
                object_raw_position = np.full(3, np.nan, dtype=np.float64)
                object_position = np.full(3, np.nan, dtype=np.float64)
                object_velocity = np.full(3, np.nan, dtype=np.float64)
                object_confidence = np.nan
        return {
            "frame_index": int(frame_index),
            "frame_timestamp": float(frame_timestamp),
            "frame_age_s": float(frame_age),
            "frame_error": frame_error or "",
            "calibrated": self.calibrated,
            "robot_timestamp": np.nan if robot_timestamp is None else float(robot_timestamp),
            "detection_robot_timestamp": (
                np.nan
                if detection_robot_timestamp is None
                else float(detection_robot_timestamp)
            ),
            "detection_pose_delta_s": (
                np.nan
                if detection_robot_timestamp is None or not np.isfinite(detection_timestamp)
                else float(detection_timestamp - detection_robot_timestamp)
            ),
            "detection_valid": detection is not None,
            "detection_error": detection_error or "",
            "detection_timestamp": detection_timestamp,
            "class_id": class_id,
            "class_name": class_name,
            "confidence": confidence,
            "center_pixel": center_pixel,
            "position_camera": position_camera,
            "position_base": position_base,
            "object_tracking_valid": bool(object_tracking_valid),
            "object_tracking_predicted": bool(object_predicted),
            "object_raw_position_base": object_raw_position,
            "object_position_base": object_position,
            "object_linear_velocity_base": object_velocity,
            "object_measurement_timestamp": (
                np.nan
                if object_measurement_timestamp is None
                else float(object_measurement_timestamp)
            ),
            "object_measurement_age_s": float(object_measurement_age),
            "object_tracking_confidence": float(object_confidence),
            "object_tracking_error": object_tracking_error or "",
        }

    def visualization_snapshot(self):
        now = monotonic()
        with self._lock:
            if self._detection_frame_bgr is None:
                return None
            tracked_position = (
                None
                if self._object_position_base is None
                else self._object_position_base.copy()
            )
            tracked_velocity = self._object_linear_velocity_base.copy()
            tracking_error = self._object_tracking_error or ""
            if (
                tracked_position is not None
                and self._object_measurement_timestamp is not None
            ):
                measurement_age = max(
                    0.0, now - float(self._object_measurement_timestamp)
                )
                if measurement_age <= self.object_max_state_age_s:
                    tracked_position += tracked_velocity * min(
                        measurement_age, self.prediction_horizon_s
                    )
                else:
                    tracked_position = None
                    tracked_velocity[:] = np.nan
                    tracking_error = tracking_error or (
                        "tracked object state stale by {:.3f}s".format(
                            measurement_age
                        )
                    )
            return (
                self._detection_frame_bgr.copy(),
                float(self._detection_frame_timestamp),
                self._detection,
                self._detection_error or "",
                tracked_position,
                tracked_velocity,
                tracking_error,
            )

    def stop(self):
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=0.5)
            if thread.is_alive():
                self.camera.close()
                thread.join(timeout=5.0)
            if thread.is_alive():
                raise RuntimeError("wrist camera thread did not stop")
        self._thread = None


class YoloDiagnosticVisualizer:
    """Display asynchronous YOLO results from the OpenCV-compatible main thread."""

    def __init__(self, monitor, window_name, refresh_interval_s):
        self.monitor = monitor
        self.window_name = str(window_name)
        self.refresh_interval_s = float(refresh_interval_s)
        if self.refresh_interval_s <= 0.0:
            raise ValueError("vision visualization refresh_interval_s must be positive")
        self.cv2 = None
        self.active = False
        self._last_render = -np.inf
        self.error = None

    def start(self):
        if self.active:
            raise RuntimeError("YOLO visualization is already running")
        self.error = None
        try:
            import cv2 as cv2_module

            self.cv2 = cv2_module
            self.cv2.namedWindow(self.window_name, self.cv2.WINDOW_NORMAL)
            snapshot = self.monitor.visualization_snapshot()
            if snapshot is not None:
                image = snapshot[0]
                self.cv2.resizeWindow(
                    self.window_name, image.shape[1], image.shape[0]
                )
            self.active = True
            self.update(force=True)
            if self.error:
                raise RuntimeError(self.error)
        except Exception as error:
            self.error = str(error)
            self.stop()
            raise RuntimeError("YOLO visualization failed: {}".format(error))

    def update(self, force=False):
        if not self.active:
            return
        try:
            now = monotonic()
            if force or now - self._last_render >= self.refresh_interval_s:
                snapshot = self.monitor.visualization_snapshot()
                if snapshot is not None:
                    (
                        image,
                        frame_timestamp,
                        detection,
                        detection_error,
                        tracked_position,
                        tracked_velocity,
                        tracking_error,
                    ) = snapshot
                    detected = detection is not None
                    color = (0, 210, 0) if detected else (0, 0, 255)

                    if detected:
                        u, v = np.rint(detection.center_pixel).astype(int)
                        self.cv2.drawMarker(
                            image,
                            (int(u), int(v)),
                            color,
                            markerType=self.cv2.MARKER_CROSS,
                            markerSize=36,
                            thickness=2,
                        )
                        self.cv2.circle(image, (int(u), int(v)), 22, color, 2)
                        status = "{}  conf={:.3f}  depth={:.3f}m".format(
                            detection.class_name,
                            float(detection.confidence),
                            float(detection.depth_m),
                        )
                        camera_xyz = "camera xyz=[{:.3f}, {:.3f}, {:.3f}]m".format(
                            *detection.position_camera
                        )
                        base_xyz = "base xyz=[{:.3f}, {:.3f}, {:.3f}]m".format(
                            *detection.position_base
                        )
                    else:
                        status = "NO YOLO TARGET"
                        camera_xyz = detection_error or "waiting for detection"
                        base_xyz = ""

                    if tracked_position is None:
                        tracked_xyz = tracking_error or "tracked base xyz unavailable"
                        velocity_xyz = "base velocity unavailable"
                    else:
                        tracked_xyz = "tracked xyz=[{:.3f}, {:.3f}, {:.3f}]m".format(
                            *tracked_position
                        )
                        velocity_xyz = "base v=[{:.3f}, {:.3f}, {:.3f}]m/s".format(
                            *tracked_velocity
                        )

                    self.cv2.rectangle(
                        image, (0, 0), (image.shape[1], 145), (0, 0, 0), -1
                    )
                    for index, line in enumerate(
                        [status, camera_xyz, base_xyz, tracked_xyz, velocity_xyz]
                    ):
                        if not line:
                            continue
                        self.cv2.putText(
                            image,
                            line,
                            (12, 22 + 27 * index),
                            self.cv2.FONT_HERSHEY_SIMPLEX,
                            0.50,
                            color if index == 0 else (230, 230, 230),
                            1,
                            self.cv2.LINE_AA,
                        )
                    age_ms = 1000.0 * max(0.0, now - frame_timestamp)
                    self.cv2.putText(
                        image,
                        "YOLO diagnostic  age={:.1f}ms  Q/ESC closes view only".format(
                            age_ms
                        ),
                        (12, image.shape[0] - 14),
                        self.cv2.FONT_HERSHEY_SIMPLEX,
                        0.50,
                        (255, 255, 255),
                        1,
                        self.cv2.LINE_AA,
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


def resolve_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def vector(config, key, size=6):
    value = np.asarray(config[key], dtype=np.float64)
    if value.shape != (size,) or not np.isfinite(value).all():
        raise ValueError("{} must contain {} finite values".format(key, size))
    return value


def snapshot_in_simulation_coordinates(snapshot, simulation_to_robot_sign):
    """Map measured robot joints into the Isaac/MuJoCo joint convention."""
    sign = np.asarray(simulation_to_robot_sign, dtype=np.float64)
    return RobotSnapshot(
        joint_position=snapshot.joint_position * sign,
        joint_velocity=snapshot.joint_velocity * sign,
        tcp_pose=snapshot.tcp_pose.copy(),
        tcp_speed=snapshot.tcp_speed.copy(),
        gripper_position=snapshot.gripper_position,
        timestamp=snapshot.timestamp,
        joint_current=snapshot.joint_current.copy(),
        target_moment=snapshot.target_moment.copy(),
        robot_mode=snapshot.robot_mode,
        safety_mode=snapshot.safety_mode,
        protective_stopped=snapshot.protective_stopped,
        emergency_stopped=snapshot.emergency_stopped,
    )


def create_wrist_camera_monitor(config):
    camera_cfg = config["camera"]
    if not bool(camera_cfg.get("enabled", False)):
        return None, None
    serial = camera_cfg.get("serial")
    if not serial:
        raise RuntimeError("camera.serial must be set for single wrist-camera deployment")

    calibration_path = resolve_path(camera_cfg["calibration"])
    calibrated = calibration_path.is_file()
    if bool(camera_cfg.get("require_calibration", False)) and not calibrated:
        raise FileNotFoundError(
            "required eye-on-hand calibration does not exist: {}".format(
                calibration_path
            )
        )
    tcp_to_camera = np.eye(4, dtype=np.float64)
    if calibrated:
        calibration_payload = yaml.safe_load(
            calibration_path.read_text(encoding="utf-8")
        )
        calibrated_serial = str(
            calibration_payload.get("metadata", {}).get("camera", {}).get("serial", "")
        )
        if calibrated_serial and calibrated_serial != str(serial):
            raise RuntimeError(
                "eye-on-hand calibration serial {} does not match configured camera {}".format(
                    calibrated_serial, serial
                )
            )
        tcp_to_camera = load_eye_on_hand(
            calibration_path,
            config["calibration"]["checkerboard"],
            camera_cfg.get("model"),
        )

    vision_cfg = config["vision"]
    object_tracking_cfg = vision_cfg.get("object_tracking", {})
    detector = None
    if bool(vision_cfg.get("enabled", False)):
        weights = resolve_path(vision_cfg["weights"])
        if not weights.is_file():
            raise FileNotFoundError("YOLO weights do not exist: {}".format(weights))
        detector = YoloInitialObjectLocator(
            weights,
            tcp_to_camera,
            vision_cfg.get("target_class"),
            vision_cfg["confidence"],
            vision_cfg["depth_radius"],
            vision_cfg["depth_min"],
            vision_cfg["depth_max"],
            vision_cfg["center_depth_offset_m"],
            vision_cfg.get("device"),
        )
    if bool(object_tracking_cfg.get("enabled", False)) and not calibrated:
        raise RuntimeError(
            "base-frame object tracking requires a valid eye-on-hand calibration"
        )

    camera = RealSenseCamera(
        camera_cfg["width"],
        camera_cfg["height"],
        camera_cfg["fps"],
        serial,
        camera_cfg.get("model"),
    )
    monitor = WristCameraMonitor(
        camera,
        camera_cfg["max_frame_age_s"],
        detector,
        calibrated,
        vision_cfg.get("inference_interval_s", 0.20),
        object_tracking_cfg,
    )
    return camera, monitor


def validate_contract(config, policy):
    policy_cfg = config["policy"]
    model_path = resolve_path(policy_cfg["model"])
    metadata_path = resolve_path(policy_cfg["metadata"])
    actual_hash = sha256(model_path)
    if actual_hash != str(policy_cfg["sha256"]):
        raise RuntimeError(
            "ONNX SHA256 mismatch: expected {}, got {}".format(
                policy_cfg["sha256"], actual_hash
            )
        )
    metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
    network = metadata.get("network", {})
    if metadata.get("task") != "Ur5eRobotiqLiftGripper":
        raise RuntimeError("metadata task is not Ur5eRobotiqLiftGripper")
    if int(network.get("observation_dim", -1)) != 69 or policy.observation_dim != 69:
        raise RuntimeError("lift/gripper policy must accept 69 observations")
    if int(network.get("action_dim", -1)) != 7 or policy.action_dim != 7:
        raise RuntimeError("lift/gripper policy must emit 7 actions")
    if bool(network.get("recurrent", True)):
        raise RuntimeError("lift/gripper deployment policy must be non-recurrent")
    if tuple(metadata.get("joint_order", ())) != JOINT_ORDER:
        raise RuntimeError("metadata joint order does not match deployment code")
    sequence = metadata.get("sequence", {})
    if int(sequence.get("episode_steps", -1)) != 240:
        raise RuntimeError("lift/gripper metadata must declare exactly 240 steps")
    if int(policy_cfg["episode_steps"]) != 240:
        raise RuntimeError("deployment configuration must use exactly 240 steps")
    period = float(policy_cfg["period"])
    if abs(period - 0.01667) > 1e-8:
        raise RuntimeError("deployment policy period must be 0.01667 seconds")
    return metadata, actual_hash


def build_observation(snapshot, projector, metadata, progress):
    if progress < 0 or progress >= 240:
        raise ValueError("progress must be in [0, 239]")
    state = projector.project(snapshot)
    limits = metadata["joint_limits"]
    lower = np.asarray([item["lower"] for item in limits], dtype=np.float64)
    upper = np.asarray([item["upper"] for item in limits], dtype=np.float64)
    if lower.shape != (12,) or upper.shape != (12,) or np.any(lower >= upper):
        raise RuntimeError("metadata contains invalid joint limits")

    observation = np.zeros(69, dtype=np.float32)
    observation[0:12] = (
        2.0 * (state.joint_position - lower) / (upper - lower) - 1.0
    ).astype(np.float32)
    observation[12:24] = state.joint_velocity.astype(np.float32)
    observation[24:27] = state.palm_center_position.astype(np.float32)
    observation[27:37] = state.palm_state.astype(np.float32)
    relative = state.fingertip_position - state.palm_center_position
    observation[47:53] = relative.reshape(-1).astype(np.float32)
    observation[66] = np.float32(math.log(progress / 10.0 + 1.0))
    # Object, keypoint, grasp state, success count, and reward fields stay zero.
    if not np.isfinite(observation).all():
        raise RuntimeError("observation contains NaN or Inf")
    return observation, state


def check_snapshot(snapshot, safety, now=None):
    now = monotonic() if now is None else float(now)
    if snapshot.protective_stopped or snapshot.emergency_stopped:
        raise RuntimeError("UR protective stop or emergency stop is active")
    if now - snapshot.timestamp > float(safety["max_state_age"]):
        raise RuntimeError("robot state is stale")
    max_velocity = float(np.max(np.abs(snapshot.joint_velocity)))
    if max_velocity > float(safety["max_joint_velocity"]):
        raise RuntimeError(
            "measured joint velocity {:.6f} exceeds safety limit".format(max_velocity)
        )


def require_initial_state(snapshot, initial, safety):
    error = float(np.max(np.abs(snapshot.joint_position - initial)))
    tolerance = float(safety["initial_joint_tolerance"])
    if error > tolerance:
        raise RuntimeError(
            "initial joint error {:.6f} rad exceeds {:.6f} rad".format(error, tolerance)
        )
    if int(snapshot.gripper_position) > int(safety["initial_gripper_open_max"]):
        raise RuntimeError(
            "gripper must be open before progress 0; POS={}".format(
                snapshot.gripper_position
            )
        )


def prepare_open_gripper(robot, robot_config, safety):
    """Open an empty gripper before progress 0 and wait for measured confirmation."""
    threshold = int(safety["initial_gripper_open_max"])
    timeout_s = float(robot_config["gripper_open_timeout_s"])
    if timeout_s <= 0.0:
        raise ValueError("gripper_open_timeout_s must be positive")
    initial_position = int(robot.gripper.get("POS"))
    if initial_position <= threshold:
        print("gripper already open: POS={}".format(initial_position))
        return initial_position
    print(
        "opening empty gripper before progress 0: POS={} -> <= {}".format(
            initial_position, threshold
        )
    )
    deadline = monotonic() + timeout_s
    try:
        robot.gripper.move(
            0,
            int(robot_config["gripper_speed"]),
            int(robot_config["gripper_force"]),
        )
        while monotonic() < deadline:
            position = int(robot.gripper.get("POS"))
            if position <= threshold:
                print("gripper open confirmed: POS={}".format(position))
                return position
            sleep(0.05)
    except Exception:
        try:
            robot.gripper.stop()
        finally:
            raise
    try:
        robot.gripper.stop()
    finally:
        raise TimeoutError(
            "gripper did not open to POS<={} within {:.1f}s".format(
                threshold, timeout_s
            )
        )


def append_row(
    rows,
    progress,
    snapshot,
    simulation_snapshot,
    observation,
    action,
    mapped_robot_action,
    policy_target,
    command_target,
    gripper_target,
    state,
    inference_time_ms,
    driver_status,
    target_tracking_error,
    camera_state,
):
    values = {
        "progress": progress,
        "monotonic_timestamp": snapshot.timestamp,
        "observation": observation,
        "policy_action": action,
        "mapped_robot_action": mapped_robot_action,
        "measured_joint_position": snapshot.joint_position,
        "measured_joint_velocity": snapshot.joint_velocity,
        "simulation_joint_position": simulation_snapshot.joint_position,
        "simulation_joint_velocity": simulation_snapshot.joint_velocity,
        "policy_joint_target": policy_target,
        "commanded_joint_target": command_target,
        "policy_command_backlog": float(
            np.max(np.abs(policy_target - command_target))
        ),
        "gripper_request": gripper_target,
        "onnx_inference_time_ms": inference_time_ms,
        "driver_status": driver_status,
        "target_tracking_error": target_tracking_error,
        "gripper_position": snapshot.gripper_position,
        "tcp_pose": snapshot.tcp_pose,
        "tcp_speed": snapshot.tcp_speed,
        "palm_position": state.palm_center_position,
        "robot_mode": snapshot.robot_mode,
        "robot_safety_state": snapshot.safety_mode,
        "safety_mode": snapshot.safety_mode,
        "protective_stopped": snapshot.protective_stopped,
        "emergency_stopped": snapshot.emergency_stopped,
        "camera_frame_index": camera_state["frame_index"],
        "camera_frame_timestamp": camera_state["frame_timestamp"],
        "camera_frame_age_s": camera_state["frame_age_s"],
        "camera_frame_error": camera_state["frame_error"],
        "camera_calibrated": camera_state["calibrated"],
        "camera_robot_timestamp": camera_state["robot_timestamp"],
        "camera_detection_robot_timestamp": camera_state["detection_robot_timestamp"],
        "camera_detection_pose_delta_s": camera_state["detection_pose_delta_s"],
        "camera_detection_valid": camera_state["detection_valid"],
        "camera_detection_error": camera_state["detection_error"],
        "camera_detection_timestamp": camera_state["detection_timestamp"],
        "camera_class_id": camera_state["class_id"],
        "camera_class_name": camera_state["class_name"],
        "camera_confidence": camera_state["confidence"],
        "camera_center_pixel": camera_state["center_pixel"],
        "camera_position_camera": camera_state["position_camera"],
        "camera_position_base": camera_state["position_base"],
        "object_tracking_valid": camera_state["object_tracking_valid"],
        "object_tracking_predicted": camera_state["object_tracking_predicted"],
        "object_raw_position_base": camera_state["object_raw_position_base"],
        "object_position_base": camera_state["object_position_base"],
        "object_linear_velocity_base": camera_state[
            "object_linear_velocity_base"
        ],
        "object_measurement_timestamp": camera_state[
            "object_measurement_timestamp"
        ],
        "object_measurement_age_s": camera_state["object_measurement_age_s"],
        "object_tracking_confidence": camera_state[
            "object_tracking_confidence"
        ],
        "object_tracking_error": camera_state["object_tracking_error"],
    }
    for name, value in values.items():
        rows[name].append(value)


def save_result(
    output_directory,
    rows,
    summary,
    save_observation_csv,
    save_object_tracking_csv,
):
    output_directory.mkdir(parents=True, exist_ok=True)
    episode_id = summary["episode_id"]
    trajectory_path = output_directory / "episode_{}.npz".format(episode_id)
    summary_path = output_directory / "episode_{}.json".format(episode_id)
    np.savez_compressed(
        str(trajectory_path),
        **{name: np.asarray(value) for name, value in rows.items()}
    )
    summary["trajectory"] = str(trajectory_path)
    observation_path = None
    if save_observation_csv:
        observation_path = output_directory / "episode_{}.observations.csv".format(
            episode_id
        )
        if rows["observation"]:
            observations = np.stack(rows["observation"]).astype(np.float64)
            progress = np.asarray(rows["progress"], dtype=np.int64).reshape(-1, 1)
        else:
            observations = np.empty((0, 69), dtype=np.float64)
            progress = np.empty((0, 1), dtype=np.int64)
        if observations.shape[1:] != (69,):
            raise RuntimeError("saved observations must have shape Nx69")
        csv_values = np.concatenate((progress, observations), axis=1)
        header = ",".join(
            ["progress"] + ["observation_{:02d}".format(i) for i in range(69)]
        )
        np.savetxt(
            str(observation_path),
            csv_values,
            delimiter=",",
            header=header,
            comments="",
            fmt=["%d"] + ["%.9g"] * 69,
        )
        summary["observation_csv"] = str(observation_path)
    object_tracking_path = None
    if save_object_tracking_csv:
        object_tracking_path = output_directory / "episode_{}.objects.csv".format(
            episode_id
        )
        if rows["progress"]:
            raw_position = np.stack(rows["object_raw_position_base"])
            tracked_position = np.stack(rows["object_position_base"])
            linear_velocity = np.stack(rows["object_linear_velocity_base"])
        else:
            raw_position = np.empty((0, 3), dtype=np.float64)
            tracked_position = np.empty((0, 3), dtype=np.float64)
            linear_velocity = np.empty((0, 3), dtype=np.float64)
        object_values = np.column_stack(
            (
                np.asarray(rows["progress"], dtype=np.int64),
                np.asarray(rows["object_tracking_valid"], dtype=np.int64),
                np.asarray(rows["object_tracking_predicted"], dtype=np.int64),
                np.asarray(rows["object_measurement_timestamp"], dtype=np.float64),
                np.asarray(rows["object_measurement_age_s"], dtype=np.float64),
                np.asarray(rows["object_tracking_confidence"], dtype=np.float64),
                raw_position,
                tracked_position,
                linear_velocity,
            )
        )
        object_header = ",".join(
            (
                "progress",
                "valid",
                "predicted",
                "measurement_timestamp",
                "measurement_age_s",
                "confidence",
                "raw_base_x",
                "raw_base_y",
                "raw_base_z",
                "tracked_base_x",
                "tracked_base_y",
                "tracked_base_z",
                "velocity_base_x",
                "velocity_base_y",
                "velocity_base_z",
            )
        )
        np.savetxt(
            str(object_tracking_path),
            object_values,
            delimiter=",",
            header=object_header,
            comments="",
            fmt=["%d", "%d", "%d"] + ["%.9g"] * 12,
        )
        summary["object_tracking_csv"] = str(object_tracking_path)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return trajectory_path, summary_path, observation_path, object_tracking_path


def run(args):
    config_path = resolve_path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    policy_cfg = config["policy"]
    robot_cfg = config["robot"]
    safety = config["safety"]
    camera_cfg = config["camera"]
    vision_cfg = config["vision"]
    output_cfg = config["output"]
    visualization_cfg = vision_cfg.get("visualization", {})
    object_tracking_cfg = vision_cfg.get("object_tracking", {})
    if camera_cfg.get("enabled") is not True:
        raise RuntimeError("single wrist-camera deployment requires camera.enabled=true")
    if not camera_cfg.get("serial"):
        raise RuntimeError("camera.serial must be filled with the wrist D435i serial number")
    if not isinstance(camera_cfg.get("require_calibration"), bool):
        raise RuntimeError("camera.require_calibration must be true or false")
    if not isinstance(vision_cfg.get("enabled"), bool):
        raise RuntimeError("vision.enabled must be true or false")
    if not isinstance(visualization_cfg.get("enabled", False), bool):
        raise RuntimeError("vision.visualization.enabled must be true or false")
    if bool(visualization_cfg.get("enabled", False)) and not vision_cfg["enabled"]:
        raise RuntimeError("YOLO visualization requires vision.enabled=true")
    if not isinstance(object_tracking_cfg.get("enabled", False), bool):
        raise RuntimeError("vision.object_tracking.enabled must be true or false")
    if bool(object_tracking_cfg.get("enabled", False)) and not vision_cfg["enabled"]:
        raise RuntimeError("object tracking requires vision.enabled=true")
    if not isinstance(output_cfg.get("save_observation_csv", True), bool):
        raise RuntimeError("output.save_observation_csv must be true or false")
    if not isinstance(output_cfg.get("print_observation_69", False), bool):
        raise RuntimeError("output.print_observation_69 must be true or false")
    if not isinstance(output_cfg.get("save_object_tracking_csv", True), bool):
        raise RuntimeError("output.save_object_tracking_csv must be true or false")
    if not isinstance(output_cfg.get("print_object_tracking", False), bool):
        raise RuntimeError("output.print_object_tracking must be true or false")
    if not isinstance(robot_cfg.get("prepare_gripper_open"), bool):
        raise RuntimeError("robot.prepare_gripper_open must be true or false")
    if args.execute and not bool(safety.get("allow_motion", False)):
        raise RuntimeError(
            "refusing motion: set safety.allow_motion=true and pass --execute"
        )
    if args.steps not in ALLOWED_STOP_STEPS:
        raise RuntimeError(
            "--steps must be one of {} so execution stops at a stage boundary".format(
                ALLOWED_STOP_STEPS
            )
        )

    policy = PolicyRunner(
        resolve_path(policy_cfg["model"]), args.provider or policy_cfg["provider"]
    )
    metadata, model_hash = validate_contract(config, policy)
    period = float(policy_cfg["period"])
    initial = vector(robot_cfg, "initial_joint_position")
    raised = vector(robot_cfg, "raised_joint_position")
    simulation_to_robot_sign = vector(
        robot_cfg, "simulation_to_robot_joint_sign"
    )
    if not np.all(np.isin(simulation_to_robot_sign, (-1.0, 1.0))):
        raise RuntimeError("simulation_to_robot_joint_sign entries must be +1 or -1")
    corridor_lower = vector(safety, "corridor_lower")
    corridor_upper = vector(safety, "corridor_upper")
    if np.any(corridor_lower >= corridor_upper):
        raise RuntimeError("deployment corridor is invalid")
    if np.any(initial < corridor_lower) or np.any(initial > corridor_upper):
        raise RuntimeError("initial target is outside the deployment corridor")
    if np.any(raised < corridor_lower) or np.any(raised > corridor_upper):
        raise RuntimeError("raised target is outside the deployment corridor")
    max_command_joint_velocity = float(safety["max_command_joint_velocity"])
    max_measured_joint_velocity = float(safety["max_joint_velocity"])
    if not 0.0 < max_command_joint_velocity <= max_measured_joint_velocity:
        raise RuntimeError(
            "max_command_joint_velocity must be positive and no greater than "
            "max_joint_velocity"
        )
    metadata_initial = np.asarray(metadata["targets"]["initial_arm_rad"], dtype=np.float64)
    metadata_raised = np.asarray(metadata["targets"]["raised_arm_rad"], dtype=np.float64)
    configured_lift_delta = (raised - initial) * simulation_to_robot_sign
    metadata_lift_delta = metadata_raised - metadata_initial
    if not np.allclose(
        configured_lift_delta,
        metadata_lift_delta,
        rtol=0.0,
        atol=1e-6,
    ):
        raise RuntimeError(
            "configured relative lift does not match the simulation metadata"
        )
    gripper_max_position = int(robot_cfg["gripper_max_position"])
    if not 1 <= gripper_max_position <= 230:
        raise RuntimeError("gripper_max_position must stay in the conservative range [1, 230]")

    if args.execute:
        confirmation = input(
            "即将打开空夹爪并向真实 UR5e/Robotiq 发送动作。"
            "确认夹爪内无物体、工作空间无人且急停可达后输入 ARM："
        ).strip()
        if confirmation != "ARM":
            raise RuntimeError("motion arming cancelled")

    projector = MujocoStateProjector(
        resolve_path(config["observation"]["model"]),
        config["observation"]["palm_center_offset"],
    )
    output_directory = resolve_path(output_cfg["directory"])
    rows = {
        name: []
        for name in (
            "progress",
            "monotonic_timestamp",
            "observation",
            "policy_action",
            "mapped_robot_action",
            "measured_joint_position",
            "measured_joint_velocity",
            "simulation_joint_position",
            "simulation_joint_velocity",
            "policy_joint_target",
            "commanded_joint_target",
            "policy_command_backlog",
            "gripper_request",
            "onnx_inference_time_ms",
            "policy_cycle_duration_s",
            "driver_status",
            "target_tracking_error",
            "gripper_position",
            "tcp_pose",
            "tcp_speed",
            "palm_position",
            "robot_mode",
            "robot_safety_state",
            "safety_mode",
            "protective_stopped",
            "emergency_stopped",
            "camera_frame_index",
            "camera_frame_timestamp",
            "camera_frame_age_s",
            "camera_frame_error",
            "camera_calibrated",
            "camera_robot_timestamp",
            "camera_detection_robot_timestamp",
            "camera_detection_pose_delta_s",
            "camera_detection_valid",
            "camera_detection_error",
            "camera_detection_timestamp",
            "camera_class_id",
            "camera_class_name",
            "camera_confidence",
            "camera_center_pixel",
            "camera_position_camera",
            "camera_position_base",
            "object_tracking_valid",
            "object_tracking_predicted",
            "object_raw_position_base",
            "object_position_base",
            "object_linear_velocity_base",
            "object_measurement_timestamp",
            "object_measurement_age_s",
            "object_tracking_confidence",
            "object_tracking_error",
        )
    }
    episode_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    summary = {
        "episode_id": episode_id,
        "utc_time": datetime.now(timezone.utc).isoformat(),
        "task": "Ur5eRobotiqLiftGripper",
        "model_sha256": model_hash,
        "executed": bool(args.execute),
        "requested_steps": int(args.steps),
        "completed_steps": 0,
        "success": False,
        "error": None,
        "simulation_to_robot_joint_sign": simulation_to_robot_sign.tolist(),
    }

    motion_started = False
    initial_tcp_z = None
    max_tcp_lift = 0.0
    target_error_cycles = 0
    final_success_steps = 0
    policy_target = initial.copy()
    command_target = initial.copy()
    max_policy_command_backlog = 0.0
    max_policy_cycle_duration = 0.0
    camera = None
    camera_monitor = None
    yolo_visualizer = None
    try:
        camera, camera_monitor = create_wrist_camera_monitor(config)
        summary["camera"] = {
            "device_name": camera.device_name,
            "serial": camera.serial,
            "calibrated": camera_monitor.calibrated,
            "vision_enabled": bool(config["vision"].get("enabled", False)),
            "vision_inference_interval_s": float(
                config["vision"].get("inference_interval_s", 0.20)
            ),
            "object_tracking_enabled": bool(
                config["vision"].get("object_tracking", {}).get("enabled", False)
            ),
        }
        print(
            "wrist_camera={!r} serial={} calibrated={} vision={}".format(
                camera.device_name,
                camera.serial,
                camera_monitor.calibrated,
                vision_cfg["enabled"],
            )
        )
        print("camera detections are diagnostic only; ONNX object fields remain zero")
        with UR5eHardware(
            robot_cfg["host"],
            robot_cfg["gripper_port"],
            servo_period=robot_cfg["servo_period"],
            servo_speed=robot_cfg["servo_speed"],
            servo_acceleration=robot_cfg["servo_acceleration"],
            servo_lookahead=robot_cfg["servo_lookahead"],
            servo_gain=robot_cfg["servo_gain"],
            gripper_speed=robot_cfg["gripper_speed"],
            gripper_force=robot_cfg["gripper_force"],
            rtde_receive_priority=robot_cfg["rtde_receive_priority"],
            rtde_control_priority=robot_cfg["rtde_control_priority"],
            servo_thread_priority=robot_cfg["servo_thread_priority"],
            allow_motion=args.execute,
            activate_gripper=args.execute,
        ) as robot:
            if args.execute and robot_cfg["prepare_gripper_open"]:
                summary["prepared_gripper_position"] = prepare_open_gripper(
                    robot, robot_cfg, safety
                )
            snapshot = robot.read()
            check_snapshot(snapshot, safety)
            require_initial_state(snapshot, initial, safety)
            camera_monitor.update_robot_pose(snapshot)

            warmup_started = monotonic()
            warmup_error = camera_monitor.warmup_detector()
            summary["camera"]["vision_warmup_s"] = float(
                monotonic() - warmup_started
            )
            summary["camera"]["vision_warmup_error"] = warmup_error
            if camera_monitor.detector is not None:
                if warmup_error:
                    print(
                        "vision warm-up complete; diagnostic detection unavailable: {}".format(
                            warmup_error
                        )
                    )
                else:
                    print("vision warm-up and initial diagnostic detection complete")

            camera_monitor.start()
            camera_monitor.wait_for_first_frame(
                config["camera"]["initial_frame_timeout_s"]
            )
            # YOLO warm-up may take seconds. Refresh the RTDE state immediately
            # before arming so progress 0 never reuses a stale snapshot.
            snapshot = robot.read()
            check_snapshot(snapshot, safety)
            require_initial_state(snapshot, initial, safety)
            camera_monitor.update_robot_pose(snapshot)
            camera_state = camera_monitor.state()
            if bool(visualization_cfg.get("enabled", False)):
                yolo_visualizer = YoloDiagnosticVisualizer(
                    camera_monitor,
                    visualization_cfg.get(
                        "window_name", "UR5e wrist YOLO diagnostics"
                    ),
                    visualization_cfg.get("refresh_interval_s", 0.05),
                )
                yolo_visualizer.start()
                summary["camera"]["yolo_visualization"] = True
            projector.reset()
            policy.reset()
            if args.execute:
                robot.start_motion()
                motion_started = True

            started = monotonic()
            for progress in range(args.steps):
                tick = monotonic()
                if progress in STAGE_NAMES:
                    print("progress={:03d} phase={}".format(progress, STAGE_NAMES[progress]))
                snapshot = snapshot if progress == 0 else robot.read()
                check_snapshot(snapshot, safety)
                camera_monitor.update_robot_pose(snapshot)
                camera_state = camera_monitor.state()
                if yolo_visualizer is not None:
                    yolo_visualizer.update()
                simulation_snapshot = snapshot_in_simulation_coordinates(
                    snapshot, simulation_to_robot_sign
                )
                observation, state = build_observation(
                    simulation_snapshot, projector, metadata, progress
                )
                if initial_tcp_z is None:
                    initial_tcp_z = float(state.palm_center_position[2])
                max_tcp_lift = max(
                    max_tcp_lift,
                    float(state.palm_center_position[2]) - initial_tcp_z,
                )
                inference_started = monotonic()
                action = np.clip(policy.infer(observation)[0], -1.0, 1.0)
                inference_time_ms = (monotonic() - inference_started) * 1000.0
                mapped_robot_action = action.astype(np.float64)
                mapped_robot_action[:6] *= simulation_to_robot_sign
                policy_target = np.clip(
                    policy_target + period * mapped_robot_action[:6],
                    corridor_lower,
                    corridor_upper,
                )
                max_command_step = max_command_joint_velocity * period
                command_target = np.clip(
                    command_target
                    + np.clip(
                        policy_target - command_target,
                        -max_command_step,
                        max_command_step,
                    ),
                    corridor_lower,
                    corridor_upper,
                )
                policy_command_backlog = float(
                    np.max(np.abs(policy_target - command_target))
                )
                max_policy_command_backlog = max(
                    max_policy_command_backlog, policy_command_backlog
                )
                if np.any(command_target < corridor_lower) or np.any(command_target > corridor_upper):
                    raise RuntimeError("commanded target left the deployment corridor")
                gripper_target = int(
                    np.clip(
                        round(
                            0.5
                            * float(gripper_max_position)
                            * (float(action[6]) + 1.0)
                        ),
                        0,
                        gripper_max_position,
                    )
                )

                target_error = float(
                    np.max(np.abs(snapshot.joint_position - command_target))
                )
                if args.execute and target_error > float(safety["max_target_error"]):
                    target_error_cycles += 1
                else:
                    target_error_cycles = 0
                if target_error_cycles >= int(safety["max_target_error_cycles"]):
                    raise RuntimeError(
                        "joint target error {:.6f} rad exceeded the watchdog for {} cycles".format(
                            target_error, target_error_cycles
                        )
                    )

                if progress >= 195:
                    arm_error = float(np.max(np.abs(snapshot.joint_position - raised)))
                    gripper_rad = float(snapshot.gripper_position) / 255.0 * 0.72
                    if (
                        arm_error <= float(safety["final_arm_tolerance"])
                        and gripper_rad <= float(safety["final_gripper_tolerance_rad"])
                    ):
                        final_success_steps += 1
                    else:
                        final_success_steps = 0

                if args.execute:
                    robot.command(command_target, gripper_target)
                append_row(
                    rows,
                    progress,
                    snapshot,
                    simulation_snapshot,
                    observation,
                    action,
                    mapped_robot_action,
                    policy_target.copy(),
                    command_target.copy(),
                    gripper_target,
                    state,
                    inference_time_ms,
                    "motion_active" if args.execute else "read_only",
                    target_error,
                    camera_state,
                )
                summary["completed_steps"] = progress + 1

                work_duration = monotonic() - tick
                rows["policy_cycle_duration_s"].append(work_duration)
                max_policy_cycle_duration = max(
                    max_policy_cycle_duration, work_duration
                )
                if work_duration > float(safety["max_cycle_duration"]):
                    raise RuntimeError(
                        "policy cycle took {:.6f}s, exceeding {:.6f}s".format(
                            work_duration, float(safety["max_cycle_duration"])
                        )
                    )
                deadline = started + (progress + 1) * period
                remaining = deadline - monotonic()
                if remaining < -float(safety["max_policy_lag"]):
                    raise RuntimeError(
                        "policy loop missed schedule by {:.6f}s".format(-remaining)
                    )
                if remaining > 0.0:
                    sleep(remaining)

            summary["final_success_steps"] = int(final_success_steps)
            summary["success"] = bool(
                args.execute
                and args.steps == 240
                and final_success_steps >= int(safety["consecutive_success_steps"])
            )
            summary["max_tcp_lift_m"] = float(max_tcp_lift)
    except Exception as error:
        summary["error"] = str(error)
        raise
    finally:
        # UR5eHardware's context manager performs the verified arm/gripper stop.
        if yolo_visualizer is not None:
            try:
                yolo_visualizer.stop()
                if yolo_visualizer.error:
                    summary["yolo_visualization_error"] = yolo_visualizer.error
            except Exception as visualization_error:
                summary["yolo_visualization_cleanup_error"] = str(
                    visualization_error
                )
                summary["success"] = False
        if camera_monitor is not None:
            try:
                camera_monitor.stop()
            except Exception as camera_error:
                summary["camera_cleanup_error"] = str(camera_error)
                summary["success"] = False
        if camera is not None:
            camera.close()
        summary["final_policy_joint_target"] = policy_target.tolist()
        summary["final_commanded_joint_target"] = command_target.tolist()
        summary["max_policy_command_backlog_rad"] = float(
            max_policy_command_backlog
        )
        summary["max_policy_cycle_duration_s"] = float(
            max_policy_cycle_duration
        )
        summary["motion_started"] = bool(motion_started)
        (
            trajectory_path,
            summary_path,
            observation_path,
            object_tracking_path,
        ) = save_result(
            output_directory,
            rows,
            summary,
            bool(output_cfg.get("save_observation_csv", True)),
            bool(output_cfg.get("save_object_tracking_csv", True)),
        )
        print("trajectory={}".format(trajectory_path))
        print("summary={}".format(summary_path))
        if observation_path is not None:
            print("observation_csv={}".format(observation_path))
        if object_tracking_path is not None:
            print("object_tracking_csv={}".format(object_tracking_path))
        if bool(output_cfg.get("print_observation_69", False)):
            for progress, observation in zip(rows["progress"], rows["observation"]):
                print(
                    "OBSERVATION_69 progress={:03d} {}".format(
                        int(progress),
                        np.array2string(
                            np.asarray(observation),
                            precision=6,
                            separator=",",
                            suppress_small=False,
                            max_line_width=10000,
                        ),
                    )
                )
        if bool(output_cfg.get("print_object_tracking", False)):
            for index, progress in enumerate(rows["progress"]):
                print(
                    "OBJECT_TRACK progress={:03d} valid={} predicted={} "
                    "position_base={} linear_velocity_base={}".format(
                        int(progress),
                        bool(rows["object_tracking_valid"][index]),
                        bool(rows["object_tracking_predicted"][index]),
                        np.array2string(
                            np.asarray(rows["object_position_base"][index]),
                            precision=6,
                            separator=",",
                            suppress_small=False,
                        ),
                        np.array2string(
                            np.asarray(rows["object_linear_velocity_base"][index]),
                            precision=6,
                            separator=",",
                            suppress_small=False,
                        ),
                    )
                )

    print("LIFT_GRIPPER_REAL " + json.dumps(summary, sort_keys=True))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="resources/config/ur5e_lift_gripper_deploy.yaml"
    )
    parser.add_argument("--provider", choices=("cpu", "cuda"))
    parser.add_argument(
        "--steps",
        type=int,
        choices=ALLOWED_STOP_STEPS,
        default=240,
        help="stop after a complete phase: 30, 75, 195, or 240 steps",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="send commands to real hardware (also requires safety.allow_motion=true)",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
