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
    distortion_model: str = "none"

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


def undistort_pixels(pixels: np.ndarray, intrinsics: CameraIntrinsics) -> np.ndarray:
    """Convert RealSense stream pixels to normalized undistorted camera rays."""
    values = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
    if not np.isfinite(values).all():
        raise ValueError("pixels must be finite")
    model = str(intrinsics.distortion_model).lower().split(".")[-1]
    coefficients = np.zeros(5, dtype=np.float64)
    supplied = np.asarray(intrinsics.distortion, dtype=np.float64).reshape(-1)
    coefficients[:min(5, len(supplied))] = supplied[:5]
    if model in ("none", "distortion_none", "") or np.all(np.abs(coefficients) <= 1e-12):
        return np.column_stack((
            (values[:, 0] - intrinsics.cx) / intrinsics.fx,
            (values[:, 1] - intrinsics.cy) / intrinsics.fy,
        ))
    if model == "brown_conrady":
        try:
            import cv2
        except ImportError as error:
            raise RuntimeError("OpenCV is required to correct Brown-Conrady distortion") from error
        return cv2.undistortPoints(
            values.reshape(-1, 1, 2), intrinsics.matrix, coefficients
        ).reshape(-1, 2)
    if model == "inverse_brown_conrady":
        x = (values[:, 0] - intrinsics.cx) / intrinsics.fx
        y = (values[:, 1] - intrinsics.cy) / intrinsics.fy
        radius2 = x * x + y * y
        radial = 1.0 + coefficients[0] * radius2 + coefficients[1] * radius2 ** 2 + coefficients[4] * radius2 ** 3
        undistorted_x = x * radial + 2.0 * coefficients[2] * x * y + coefficients[3] * (
            radius2 + 2.0 * x * x
        )
        undistorted_y = y * radial + 2.0 * coefficients[3] * x * y + coefficients[2] * (
            radius2 + 2.0 * y * y
        )
        return np.column_stack((undistorted_x, undistorted_y))
    raise ValueError("unsupported RealSense distortion model: {}".format(intrinsics.distortion_model))


def deproject_pixel(pixel: Tuple[float, float], depth_m: float, intrinsics: CameraIntrinsics) -> np.ndarray:
    if not np.isfinite(depth_m) or depth_m <= 0.0:
        raise ValueError("depth_m must be positive and finite")
    normalized = undistort_pixels(np.asarray([pixel], dtype=np.float64), intrinsics)[0]
    return np.array([normalized[0] * depth_m, normalized[1] * depth_m, depth_m], dtype=np.float64)


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


def tcp_pose_to_transform(tcp_pose: Iterable[float]) -> np.ndarray:
    """Convert UR ``[x, y, z, rx, ry, rz]`` axis-angle pose to ``T_base_tcp``."""
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("OpenCV is required to convert a UR TCP pose") from error
    pose = np.asarray(tcp_pose, dtype=np.float64)
    if pose.shape != (6,) or not np.isfinite(pose).all():
        raise ValueError("tcp_pose must contain six finite values")
    rotation, _ = cv2.Rodrigues(pose[3:].reshape(3, 1))
    return make_transform(rotation, pose[:3])


def rotation_angle_degrees(transform: np.ndarray) -> float:
    rotation = as_transform(transform)[:3, :3]
    cosine = np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


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


def transform_spans(transforms: Iterable[np.ndarray]):
    """Return maximum pairwise translation (m) and rotation (deg)."""
    values = [as_transform(item) for item in transforms]
    if len(values) < 2:
        return 0.0, 0.0
    maximum_translation = 0.0
    maximum_rotation = 0.0
    for first_index, first in enumerate(values[:-1]):
        for second in values[first_index + 1:]:
            maximum_translation = max(
                maximum_translation,
                float(np.linalg.norm(first[:3, 3] - second[:3, 3])),
            )
            maximum_rotation = max(
                maximum_rotation,
                rotation_angle_degrees(invert_transform(first).dot(second)),
            )
    return maximum_translation, maximum_rotation


def checkerboard_object_points(columns: int, rows: int, square_size_m: float) -> np.ndarray:
    columns, rows = int(columns), int(rows)
    square_size_m = float(square_size_m)
    if columns < 2 or rows < 2 or square_size_m <= 0.0:
        raise ValueError("checkerboard columns/rows must be >=2 and square size positive")
    points = np.zeros((columns * rows, 3), dtype=np.float64)
    points[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2) * square_size_m
    return points


def estimate_checkerboard_pose(
    color_bgr: np.ndarray,
    intrinsics: CameraIntrinsics,
    columns: int,
    rows: int,
    square_size_m: float,
):
    """Return ``T_camera_board``, reprojection RMS, and detected inner corners."""
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("OpenCV is required for checkerboard calibration") from error
    image = np.asarray(color_bgr)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("color_bgr must be an HxWx3 image")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    flags = cv2.CALIB_CB_NORMALIZE_IMAGE
    if hasattr(cv2, "CALIB_CB_EXHAUSTIVE"):
        flags |= cv2.CALIB_CB_EXHAUSTIVE
    if hasattr(cv2, "CALIB_CB_ACCURACY"):
        flags |= cv2.CALIB_CB_ACCURACY
    found, corners = cv2.findChessboardCornersSB(gray, (int(columns), int(rows)), flags=flags)
    if not found or corners is None:
        raise RuntimeError("checkerboard {}x{} inner corners not found".format(columns, rows))
    corners = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
    object_points = checkerboard_object_points(columns, rows, square_size_m)
    normalized_corners = undistort_pixels(corners, intrinsics)
    normalized_matrix = np.eye(3, dtype=np.float64)
    no_distortion = np.zeros(5, dtype=np.float64)
    ok, rotation_vector, translation = cv2.solvePnP(
        object_points,
        normalized_corners,
        normalized_matrix,
        no_distortion,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        raise RuntimeError("solvePnP failed for checkerboard corners")
    projected, _ = cv2.projectPoints(
        object_points, rotation_vector, translation, normalized_matrix, no_distortion
    )
    normalized_error = projected.reshape(-1, 2) - normalized_corners
    pixel_error = normalized_error * np.asarray([intrinsics.fx, intrinsics.fy])
    rms = float(np.sqrt(np.mean(np.square(pixel_error))))
    rotation, _ = cv2.Rodrigues(rotation_vector)
    return make_transform(rotation, translation.reshape(3)), rms, corners


def calibrate_eye_on_hand(base_to_tcp, camera_to_board, method="park") -> np.ndarray:
    """Solve eye-on-hand calibration and return ``T_tcp_camera``."""
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("OpenCV is required for hand-eye calibration") from error
    gripper_poses = [as_transform(item, "base_to_tcp") for item in base_to_tcp]
    board_poses = [as_transform(item, "camera_to_board") for item in camera_to_board]
    if len(gripper_poses) != len(board_poses) or len(gripper_poses) < 3:
        raise ValueError("hand-eye calibration requires at least three paired poses")
    methods = {
        "tsai": cv2.CALIB_HAND_EYE_TSAI,
        "park": cv2.CALIB_HAND_EYE_PARK,
        "horaud": cv2.CALIB_HAND_EYE_HORAUD,
        "andreff": cv2.CALIB_HAND_EYE_ANDREFF,
        "daniilidis": cv2.CALIB_HAND_EYE_DANIILIDIS,
    }
    method_name = str(method).lower()
    if method_name not in methods:
        raise ValueError("unknown hand-eye method {!r}".format(method))
    rotation, translation = cv2.calibrateHandEye(
        [item[:3, :3] for item in gripper_poses],
        [item[:3, 3] for item in gripper_poses],
        [item[:3, :3] for item in board_poses],
        [item[:3, 3] for item in board_poses],
        method=methods[method_name],
    )
    return make_transform(rotation, np.asarray(translation).reshape(3))


def eye_on_hand_residuals(base_to_tcp, camera_to_board, tcp_to_camera):
    """Measure how constant the fixed board pose is across hand-eye samples."""
    tcp_to_camera = as_transform(tcp_to_camera, "tcp_to_camera")
    board_in_base = [
        as_transform(base_pose).dot(tcp_to_camera).dot(as_transform(board_pose))
        for base_pose, board_pose in zip(base_to_tcp, camera_to_board)
    ]
    if not board_in_base:
        raise ValueError("at least one paired hand-eye pose is required")
    mean_board = average_transforms(board_in_base)
    translation_errors = np.asarray(
        [np.linalg.norm(item[:3, 3] - mean_board[:3, 3]) for item in board_in_base],
        dtype=np.float64,
    )
    rotation_errors = np.asarray(
        [rotation_angle_degrees(invert_transform(mean_board).dot(item)) for item in board_in_base],
        dtype=np.float64,
    )
    return mean_board, translation_errors, rotation_errors


def robust_calibrate_eye_on_hand(
    base_to_tcp,
    camera_to_board,
    method="park",
    max_translation_error_m=0.01,
    max_rotation_error_deg=2.0,
    minimum_inliers=10,
):
    """Solve, reject inconsistent board poses once, and refine hand-eye calibration."""
    base_poses = [as_transform(item, "base_to_tcp") for item in base_to_tcp]
    board_poses = [as_transform(item, "camera_to_board") for item in camera_to_board]
    if len(base_poses) != len(board_poses):
        raise ValueError("base_to_tcp and camera_to_board sample counts differ")
    minimum_inliers = int(minimum_inliers)
    if minimum_inliers < 3 or len(base_poses) < minimum_inliers:
        raise ValueError("not enough samples for the requested minimum inliers")
    estimate = calibrate_eye_on_hand(base_poses, board_poses, method)
    _, translation_errors, rotation_errors = eye_on_hand_residuals(
        base_poses, board_poses, estimate
    )
    inliers = np.logical_and(
        translation_errors <= float(max_translation_error_m),
        rotation_errors <= float(max_rotation_error_deg),
    )
    if int(inliers.sum()) < minimum_inliers:
        raise RuntimeError(
            "hand-eye consistency left only {}/{} inliers".format(int(inliers.sum()), len(inliers))
        )
    estimate = calibrate_eye_on_hand(
        [pose for pose, keep in zip(base_poses, inliers) if keep],
        [pose for pose, keep in zip(board_poses, inliers) if keep],
        method,
    )
    mean_board, translation_errors, rotation_errors = eye_on_hand_residuals(
        base_poses, board_poses, estimate
    )
    final_inliers = np.logical_and(
        translation_errors <= float(max_translation_error_m),
        rotation_errors <= float(max_rotation_error_deg),
    )
    if int(final_inliers.sum()) < minimum_inliers:
        raise RuntimeError(
            "refined hand-eye consistency left only {}/{} inliers".format(
                int(final_inliers.sum()), len(final_inliers)
            )
        )
    if not np.array_equal(final_inliers, inliers):
        estimate = calibrate_eye_on_hand(
            [pose for pose, keep in zip(base_poses, final_inliers) if keep],
            [pose for pose, keep in zip(board_poses, final_inliers) if keep],
            method,
        )
        mean_board, translation_errors, rotation_errors = eye_on_hand_residuals(
            base_poses, board_poses, estimate
        )
    return estimate, final_inliers, mean_board, translation_errors, rotation_errors


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


def load_eye_on_hand(path, expected_checkerboard=None, expected_camera_model=None) -> np.ndarray:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("calibration_type") != "eye_on_hand":
        raise ValueError("Calibration file must declare calibration_type: eye_on_hand")
    if "tcp_to_camera" not in payload:
        raise ValueError("Eye-on-hand calibration must contain tcp_to_camera")
    metadata = payload.get("metadata", {})
    if expected_checkerboard is not None:
        actual = metadata.get("checkerboard", {})
        for key in ("model", "columns", "rows"):
            if str(actual.get(key)) != str(expected_checkerboard.get(key)):
                raise ValueError(
                    "Calibration checkerboard {} mismatch: expected {!r}, found {!r}".format(
                        key, expected_checkerboard.get(key), actual.get(key)
                    )
                )
        if not np.isclose(
            float(actual.get("square_size_m", np.nan)),
            float(expected_checkerboard["square_size_m"]),
            rtol=0.0,
            atol=1e-9,
        ):
            raise ValueError("Calibration checkerboard square_size_m does not match configuration")
    if expected_camera_model is not None:
        actual_model = str(metadata.get("camera", {}).get("model", ""))
        if str(expected_camera_model).lower() not in actual_model.lower():
            raise ValueError(
                "Calibration camera mismatch: expected {!r}, found {!r}".format(
                    expected_camera_model, actual_model
                )
            )
    return as_transform(payload["tcp_to_camera"], "tcp_to_camera")


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


def save_eye_on_hand(path, tcp_to_camera: np.ndarray, metadata=None) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 2,
        "calibration_type": "eye_on_hand",
        "convention": "T_tcp_camera maps RealSense color optical-frame metres into the configured UR TCP frame",
        "tcp_to_camera": as_transform(tcp_to_camera, "tcp_to_camera").tolist(),
    }
    if metadata:
        payload["metadata"] = metadata
    destination.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
