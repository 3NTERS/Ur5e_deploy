from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import yaml


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    distortion: Tuple[float, ...] = ()

    def __post_init__(self):
        values = (self.fx, self.fy, self.cx, self.cy)
        if not np.isfinite(values).all() or self.fx <= 0.0 or self.fy <= 0.0:
            raise ValueError("Camera intrinsics must be finite and focal lengths positive")
        if self.distortion and not np.isfinite(self.distortion).all():
            raise ValueError("Camera distortion coefficients must be finite")

    @property
    def matrix(self):
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )


def as_transform(value: Iterable[Iterable[float]], name: str = "transform") -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("{} must be a finite 4x4 matrix".format(name))
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise ValueError("{} must have homogeneous last row [0, 0, 0, 1]".format(name))
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T.dot(rotation), np.eye(3), atol=1e-5):
        raise ValueError("{} rotation is not orthonormal".format(name))
    if np.linalg.det(rotation) < 0.999:
        raise ValueError("{} rotation must be right handed".format(name))
    return matrix


def transform_point(transform: np.ndarray, point: Iterable[float]) -> np.ndarray:
    transform = as_transform(transform)
    point = np.asarray(point, dtype=np.float64)
    if point.shape != (3,) or not np.isfinite(point).all():
        raise ValueError("point must contain three finite values")
    return transform[:3, :3].dot(point) + transform[:3, 3]


def deproject_pixel(pixel: Tuple[float, float], depth_m: float, intrinsics: CameraIntrinsics) -> np.ndarray:
    if not np.isfinite(depth_m) or depth_m <= 0.0:
        raise ValueError("depth_m must be positive and finite")
    u, v = pixel
    if intrinsics.distortion and np.any(np.abs(intrinsics.distortion) > 1e-12):
        try:
            import cv2
        except ImportError as error:
            raise RuntimeError("OpenCV is required to correct camera distortion") from error
        normalized = cv2.undistortPoints(
            np.asarray([[[u, v]]], dtype=np.float64),
            intrinsics.matrix,
            np.asarray(intrinsics.distortion, dtype=np.float64),
        )[0, 0]
        return np.array([normalized[0] * depth_m, normalized[1] * depth_m, depth_m], dtype=np.float64)
    return np.array(
        [
            (float(u) - intrinsics.cx) * depth_m / intrinsics.fx,
            (float(v) - intrinsics.cy) * depth_m / intrinsics.fy,
            depth_m,
        ],
        dtype=np.float64,
    )


def invert_transform(transform: np.ndarray) -> np.ndarray:
    transform = as_transform(transform)
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = transform[:3, :3].T
    result[:3, 3] = -result[:3, :3].dot(transform[:3, 3])
    return result


def make_transform(rotation: np.ndarray, translation: Iterable[float]) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = np.asarray(rotation, dtype=np.float64)
    result[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return as_transform(result)


def average_transforms(transforms: Iterable[np.ndarray]) -> np.ndarray:
    values = [as_transform(item) for item in transforms]
    if not values:
        raise ValueError("At least one transform is required")
    rotations = np.stack([item[:3, :3] for item in values])
    u, _, vt = np.linalg.svd(rotations.mean(axis=0))
    rotation = u.dot(vt)
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u.dot(vt)
    translation = np.stack([item[:3, 3] for item in values]).mean(axis=0)
    return make_transform(rotation, translation)


def estimate_eye_on_base(
    qr_corners_px: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    qr_size_m: float,
    base_to_qr: np.ndarray,
):
    """Return T_base_camera and the solvePnP reprojection RMS in pixels.

    ``qr_corners_px`` must use OpenCV QRCodeDetector order: top-left,
    top-right, bottom-right, bottom-left. ``base_to_qr`` describes the centre
    of the printed code and fixes the QR coordinate convention in the cell.
    """
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("OpenCV is required for QR calibration") from error
    corners = np.asarray(qr_corners_px, dtype=np.float64).reshape(4, 2)
    if qr_size_m <= 0.0 or not np.isfinite(corners).all():
        raise ValueError("QR size and corners are invalid")
    half = 0.5 * float(qr_size_m)
    object_points = np.array(
        [[-half, half, 0.0], [half, half, 0.0], [half, -half, 0.0], [-half, -half, 0.0]],
        dtype=np.float64,
    )
    ok, rvec, tvec = cv2.solvePnP(
        object_points,
        corners,
        np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3),
        np.asarray(distortion, dtype=np.float64),
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if not ok:
        raise RuntimeError("solvePnP failed for the detected QR corners")
    camera_from_qr = make_transform(cv2.Rodrigues(rvec)[0], tvec.reshape(3))
    base_from_camera = as_transform(base_to_qr, "base_to_qr").dot(invert_transform(camera_from_qr))
    projected, _ = cv2.projectPoints(
        object_points,
        rvec,
        tvec,
        np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3),
        np.asarray(distortion, dtype=np.float64),
    )
    rms = float(np.sqrt(np.mean(np.square(projected.reshape(4, 2) - corners))))
    return as_transform(base_from_camera, "base_to_camera"), rms


def load_eye_on_base(path) -> np.ndarray:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "base_to_camera" not in payload:
        raise ValueError("Calibration file must contain base_to_camera")
    return as_transform(payload["base_to_camera"], "base_to_camera")


def save_eye_on_base(path, base_to_camera: np.ndarray, metadata=None) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "convention": "T_base_camera maps camera-frame metres into UR base frame",
        "base_to_camera": as_transform(base_to_camera).tolist(),
    }
    if metadata:
        payload["metadata"] = metadata
    destination.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
