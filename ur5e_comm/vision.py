from __future__ import annotations

from dataclasses import dataclass, field
from time import monotonic

import numpy as np

from .geometry import CameraIntrinsics, as_transform, deproject_pixel, transform_point


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
    base_to_camera: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=np.float64))


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
    """Detect exactly once; callers retain the returned initial object state."""

    def __init__(
        self,
        weights,
        tcp_to_camera,
        target_class=None,
        confidence=0.5,
        depth_radius=3,
        depth_min=0.05,
        depth_max=3.0,
        device=None,
    ):
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise RuntimeError("Install ultralytics to use YOLO object detection") from error
        self.model = YOLO(str(weights))
        self.tcp_to_camera = as_transform(tcp_to_camera, "tcp_to_camera")
        self.target_class = target_class
        self.confidence = float(confidence)
        self.depth_radius = int(depth_radius)
        self.depth_min = float(depth_min)
        self.depth_max = float(depth_max)
        self.device = device

    def locate(self, frame: RGBDFrame, base_to_tcp: np.ndarray) -> ObjectDetection:
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
        confidence, class_id, class_name, box = max(candidates, key=lambda item: item[0])
        u, v = 0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3])
        depth = robust_depth(
            frame.depth_m,
            u,
            v,
            self.depth_radius,
            self.depth_min,
            self.depth_max,
        )
        camera_point = deproject_pixel((u, v), depth, frame.intrinsics)
        base_to_camera = as_transform(base_to_tcp, "base_to_tcp").dot(self.tcp_to_camera)
        base_point = transform_point(base_to_camera, camera_point)
        return ObjectDetection(
            position_camera=camera_point,
            position_base=base_point,
            center_pixel=np.array([u, v], dtype=np.float64),
            depth_m=depth,
            confidence=confidence,
            class_id=class_id,
            class_name=class_name,
            timestamp=frame.timestamp,
            base_to_camera=base_to_camera,
        )

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
