from __future__ import annotations

from dataclasses import dataclass, field
from threading import Event, Lock, Thread
from time import monotonic

import numpy as np

from .geometry import CameraIntrinsics, as_transform, deproject_pixel, transform_point
from .yolo_compat import load_yolo_model


@dataclass(frozen=True)
class RGBDFrame:
    color_bgr: np.ndarray
    depth_m: np.ndarray
    intrinsics: CameraIntrinsics
    timestamp: float


@dataclass(frozen=True)
class ObjectDetection:
    position_camera: np.ndarray
    position_base: np.ndarray
    center_pixel: np.ndarray
    depth_m: float
    confidence: float
    class_id: int
    class_name: str
    timestamp: float
    surface_depth_m: float = np.nan
    base_to_camera: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=np.float64))


@dataclass(frozen=True)
class TrackedObjectState:
    position_base: np.ndarray
    linear_velocity_base: np.ndarray
    measurement_timestamp: float
    output_timestamp: float
    confidence: float
    predicted: bool
    center_pixel: np.ndarray
    surface_depth_m: float
    center_depth_m: float

    @property
    def measurement_age_s(self):
        return self.output_timestamp - self.measurement_timestamp


class RealSenseCamera:
    """Aligned colour/depth source. Import is lazy so offline tests stay light."""

    def __init__(self, width=640, height=480, fps=30, serial=None, expected_model=None):
        try:
            import pyrealsense2 as rs
        except ImportError as error:
            raise RuntimeError("Install pyrealsense2 to use a RealSense camera") from error
        self.rs = rs
        self.pipeline = rs.pipeline()
        config = rs.config()
        if serial:
            config.enable_device(str(serial))
        config.enable_stream(rs.stream.color, int(width), int(height), rs.format.bgr8, int(fps))
        config.enable_stream(rs.stream.depth, int(width), int(height), rs.format.z16, int(fps))
        self.closed = True
        profile = self.pipeline.start(config)
        self.closed = False
        device = profile.get_device()
        self.device_name = str(device.get_info(rs.camera_info.name))
        self.serial = str(device.get_info(rs.camera_info.serial_number))
        if expected_model and str(expected_model).lower() not in self.device_name.lower():
            self.close()
            raise RuntimeError(
                "Expected RealSense {!r}, connected device is {!r}".format(
                    expected_model, self.device_name
                )
            )
        self.align = rs.align(rs.stream.color)
        sensor = device.first_depth_sensor()
        self.depth_scale = float(sensor.get_depth_scale())

    def read(self, timeout_ms=2000) -> RGBDFrame:
        if self.closed:
            raise RuntimeError("Camera is closed")
        frames = self.align.process(self.pipeline.wait_for_frames(int(timeout_ms)))
        color = frames.get_color_frame()
        depth = frames.get_depth_frame()
        if not color or not depth:
            raise RuntimeError("RealSense did not return aligned colour and depth frames")
        profile = color.profile.as_video_stream_profile()
        intr = profile.intrinsics
        camera = CameraIntrinsics(
            intr.fx,
            intr.fy,
            intr.ppx,
            intr.ppy,
            intr.width,
            intr.height,
            tuple(float(value) for value in intr.coeffs),
            str(intr.model).split(".")[-1],
        )
        return RGBDFrame(
            color_bgr=np.asanyarray(color.get_data()),
            depth_m=np.asanyarray(depth.get_data()).astype(np.float32) * self.depth_scale,
            intrinsics=camera,
            timestamp=monotonic(),
        )

    def close(self):
        if not self.closed:
            self.pipeline.stop()
            self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def robust_depth(depth_m, u, v, radius=3, minimum=0.05, maximum=3.0) -> float:
    depth = np.asarray(depth_m, dtype=np.float64)
    if depth.ndim != 2:
        raise ValueError("depth image must be HxW")
    u, v, radius = int(round(u)), int(round(v)), int(radius)
    x0, x1 = max(0, u - radius), min(depth.shape[1], u + radius + 1)
    y0, y1 = max(0, v - radius), min(depth.shape[0], v + radius + 1)
    values = depth[y0:y1, x0:x1]
    valid = values[np.isfinite(values) & (values >= minimum) & (values <= maximum)]
    if valid.size == 0:
        raise RuntimeError("No valid depth around detection centre ({}, {})".format(u, v))
    return float(np.median(valid))


class YoloInitialObjectLocator:
    """YOLO RGB-D locator for a camera rigidly attached to the TCP."""

    def __init__(
        self,
        weights,
        tcp_to_camera,
        target_class=None,
        confidence=0.5,
        depth_radius=3,
        depth_min=0.05,
        depth_max=3.0,
        center_depth_offset_m=0.01,
        device=None,
        model=None,
    ):
        if model is None:
            try:
                model = load_yolo_model(weights)
            except ImportError as error:
                raise RuntimeError("Install ultralytics to use YOLO object detection") from error
        self.model = model
        self.tcp_to_camera = as_transform(tcp_to_camera, "tcp_to_camera")
        self.target_class = target_class
        self.confidence = float(confidence)
        self.depth_radius = int(depth_radius)
        self.depth_min = float(depth_min)
        self.depth_max = float(depth_max)
        self.center_depth_offset_m = float(center_depth_offset_m)
        self.device = device

    def _candidates(self, frame):
        results = self.model.predict(
            source=frame.color_bgr,
            conf=self.confidence,
            device=self.device,
            verbose=False,
        )
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            raise RuntimeError("YOLO did not detect any object")
        result = results[0]
        names = getattr(result, "names", None)
        if names is None:
            names = getattr(self.model, "names", None)
        if names is None:
            raise RuntimeError("YOLO result does not expose class names")
        candidates = []
        for index in range(len(result.boxes)):
            class_id = int(result.boxes.cls[index].item())
            class_name = str(names[class_id])
            if self.target_class is not None and str(self.target_class) not in (str(class_id), class_name):
                continue
            confidence = float(result.boxes.conf[index].item())
            box = result.boxes.xyxy[index].detach().cpu().numpy()
            candidates.append((confidence, class_id, class_name, box))
        if not candidates:
            raise RuntimeError("YOLO did not detect target class {!r}".format(self.target_class))
        return candidates

    def locate_with_base_to_camera(
        self,
        frame,
        base_to_camera,
        reference_position=None,
        max_distance=None,
    ):
        base_to_camera = as_transform(base_to_camera, "base_to_camera")
        detections = []
        depth_errors = []
        for confidence, class_id, class_name, box in self._candidates(frame):
            u, v = 0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3])
            try:
                surface_depth = robust_depth(
                    frame.depth_m,
                    u,
                    v,
                    self.depth_radius,
                    self.depth_min,
                    self.depth_max,
                )
            except RuntimeError as error:
                depth_errors.append(str(error))
                continue
            center_depth = surface_depth + self.center_depth_offset_m
            camera_point = deproject_pixel((u, v), center_depth, frame.intrinsics)
            base_point = transform_point(base_to_camera, camera_point)
            detections.append(
                ObjectDetection(
                    position_camera=camera_point,
                    position_base=base_point,
                    center_pixel=np.array([u, v], dtype=np.float64),
                    depth_m=center_depth,
                    confidence=confidence,
                    class_id=class_id,
                    class_name=class_name,
                    timestamp=frame.timestamp,
                    surface_depth_m=surface_depth,
                    base_to_camera=base_to_camera,
                )
            )
        if not detections:
            detail = depth_errors[-1] if depth_errors else "no usable target detection"
            raise RuntimeError("YOLO target has no valid RGB-D position: {}".format(detail))
        if reference_position is None:
            return max(detections, key=lambda item: item.confidence)
        reference = np.asarray(reference_position, dtype=np.float64)
        selected = min(
            detections,
            key=lambda item: float(np.linalg.norm(item.position_base - reference)),
        )
        distance = float(np.linalg.norm(selected.position_base - reference))
        if max_distance is not None and distance > float(max_distance):
            raise RuntimeError(
                "nearest target is {:.4f}m from predicted position (limit {:.4f}m)".format(
                    distance, float(max_distance)
                )
            )
        return selected

    def locate(self, frame: RGBDFrame, base_to_tcp: np.ndarray) -> ObjectDetection:
        base_to_camera = as_transform(base_to_tcp, "base_to_tcp").dot(self.tcp_to_camera)
        return self.locate_with_base_to_camera(frame, base_to_camera)

    def locate_from_camera(self, camera, base_to_tcp, attempts=30) -> ObjectDetection:
        if int(attempts) <= 0:
            raise ValueError("detection attempts must be positive")
        errors = []
        for _ in range(int(attempts)):
            try:
                return self.locate(camera.read(), base_to_tcp)
            except RuntimeError as error:
                errors.append(str(error))
        raise RuntimeError("Failed to lock initial object after {} frames: {}".format(attempts, errors[-1]))


class YoloEyeOnBaseLocator:
    """RGB-D locator using one fixed ``T_base_camera`` transform."""

    def __init__(self, base_to_camera, detector):
        self.base_to_camera = as_transform(base_to_camera, "base_to_camera")
        self.detector = detector

    def locate(self, frame, reference_position=None, max_distance=None):
        return self.detector.locate_with_base_to_camera(
            frame,
            self.base_to_camera,
            reference_position,
            max_distance,
        )

    def locate_from_camera(
        self,
        camera,
        attempts=30,
        reference_position=None,
        max_distance=None,
    ):
        errors = []
        for _ in range(int(attempts)):
            try:
                return self.locate(camera.read(), reference_position, max_distance)
            except RuntimeError as error:
                errors.append(str(error))
        detail = errors[-1] if errors else "no attempts were made"
        raise RuntimeError(
            "Failed to lock eye-on-base object after {} frames: {}".format(attempts, detail)
        )


class EyeOnBaseObjectTracker:
    """Asynchronous fixed-camera tracker with filtered velocity and bounded prediction."""

    def __init__(
        self,
        camera,
        locator,
        association_max_distance_m=0.25,
        position_alpha=0.6,
        velocity_alpha=0.4,
        prediction_horizon_s=0.10,
        max_state_age_s=0.15,
    ):
        self.camera = camera
        self.locator = locator
        self.association_max_distance_m = float(association_max_distance_m)
        self.position_alpha = float(position_alpha)
        self.velocity_alpha = float(velocity_alpha)
        self.prediction_horizon_s = float(prediction_horizon_s)
        self.max_state_age_s = float(max_state_age_s)
        self._lock = Lock()
        self._stop = Event()
        self._thread = None
        self._detection = None
        self._position = None
        self._velocity = np.zeros(3, dtype=np.float64)
        self._last_error = None
        self._measurement_version = 0
        self._reported_measurement_version = 0

    def start(self, initial_detection):
        if self._thread is not None:
            raise RuntimeError("eye-on-base tracker is already running")
        with self._lock:
            self._detection = initial_detection
            self._position = np.asarray(initial_detection.position_base, dtype=np.float64).copy()
            self._velocity = np.zeros(3, dtype=np.float64)
            self._last_error = None
            self._measurement_version = 1
            self._reported_measurement_version = 0
        self._stop.clear()
        self._thread = Thread(target=self._run, name="eye-on-base-tracker", daemon=True)
        self._thread.start()

    def _reference(self, timestamp):
        with self._lock:
            elapsed = max(0.0, float(timestamp) - self._detection.timestamp)
            elapsed = min(elapsed, self.prediction_horizon_s)
            return self._position + self._velocity * elapsed

    def _accept(self, detection):
        with self._lock:
            previous_detection = self._detection
            previous_position = self._position.copy()
            dt = float(detection.timestamp - previous_detection.timestamp)
            if dt <= 0.0:
                return
            measured = np.asarray(detection.position_base, dtype=np.float64)
            filtered_position = (
                self.position_alpha * measured
                + (1.0 - self.position_alpha) * previous_position
            )
            raw_velocity = (filtered_position - previous_position) / dt
            self._velocity = (
                self.velocity_alpha * raw_velocity
                + (1.0 - self.velocity_alpha) * self._velocity
            )
            self._position = filtered_position
            self._detection = detection
            self._last_error = None
            self._measurement_version += 1

    def _run(self):
        while not self._stop.is_set():
            try:
                frame = self.camera.read()
                reference = self._reference(frame.timestamp)
                detection = self.locator.locate(
                    frame,
                    reference,
                    self.association_max_distance_m,
                )
                self._accept(detection)
            except Exception as error:
                with self._lock:
                    self._last_error = str(error)

    def state(self, now=None):
        now = monotonic() if now is None else float(now)
        with self._lock:
            if self._detection is None:
                raise RuntimeError("eye-on-base tracker has no initial state")
            detection = self._detection
            position = self._position.copy()
            velocity = self._velocity.copy()
            last_error = self._last_error
            predicted = (
                self._measurement_version == self._reported_measurement_version
                or last_error is not None
            )
            self._reported_measurement_version = self._measurement_version
        age = now - detection.timestamp
        if age < 0.0:
            age = 0.0
        if age > self.max_state_age_s:
            detail = ": {}".format(last_error) if last_error else ""
            raise RuntimeError(
                "Eye-on-base object state is stale by {:.3f}s{}".format(age, detail)
            )
        prediction_time = min(age, self.prediction_horizon_s)
        return TrackedObjectState(
            position_base=position + velocity * prediction_time,
            linear_velocity_base=velocity,
            measurement_timestamp=detection.timestamp,
            output_timestamp=now,
            confidence=detection.confidence,
            predicted=predicted,
            center_pixel=detection.center_pixel.copy(),
            surface_depth_m=detection.surface_depth_m,
            center_depth_m=detection.depth_m,
        )

    def stop(self):
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join()
        self._thread = None
