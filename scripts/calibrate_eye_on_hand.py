#!/usr/bin/env python3
"""Calibrate a TCP-mounted RealSense against a fixed checkerboard."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
from time import monotonic

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ur5e_comm.geometry import (
    average_transforms,
    estimate_checkerboard_pose,
    invert_transform,
    robust_calibrate_eye_on_hand,
    rotation_angle_degrees,
    save_eye_on_hand,
    tcp_pose_to_transform,
    transform_spans,
)
from ur5e_comm.vision import RealSenseCamera


def resolve_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def capture_motion_metrics(before_pose, after_pose, before_speed, after_speed, elapsed):
    before = tcp_pose_to_transform(before_pose)
    after = tcp_pose_to_transform(after_pose)
    relative = invert_transform(before).dot(after)
    return {
        "elapsed_s": float(elapsed),
        "translation_delta_m": float(np.linalg.norm(relative[:3, 3])),
        "rotation_delta_deg": rotation_angle_degrees(relative),
        "linear_speed_m_s": float(max(np.linalg.norm(before_speed[:3]), np.linalg.norm(after_speed[:3]))),
        "angular_speed_rad_s": float(max(np.linalg.norm(before_speed[3:]), np.linalg.norm(after_speed[3:]))),
    }


def validate_stationary(metrics, config):
    limits = {
        "elapsed_s": float(config["max_capture_interval_s"]),
        "translation_delta_m": float(config["max_capture_translation_delta_m"]),
        "rotation_delta_deg": float(config["max_capture_rotation_delta_deg"]),
        "linear_speed_m_s": float(config["max_tcp_linear_speed_m_s"]),
        "angular_speed_rad_s": float(config["max_tcp_angular_speed_rad_s"]),
    }
    failures = [
        "{}={:.6g}>{:.6g}".format(name, metrics[name], limit)
        for name, limit in limits.items()
        if metrics[name] > limit
    ]
    if failures:
        raise RuntimeError("robot/camera pair was not stationary: " + ", ".join(failures))


def sufficiently_distinct(candidate, accepted, minimum_translation, minimum_rotation):
    for previous in accepted:
        relative = invert_transform(previous).dot(candidate)
        translation = float(np.linalg.norm(relative[:3, 3]))
        rotation = rotation_angle_degrees(relative)
        if translation < minimum_translation and rotation < minimum_rotation:
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="resources/config/ur5e_deploy.yaml")
    parser.add_argument("--output", help="override camera.calibration output path")
    parser.add_argument("--samples", type=int, help="override calibration.samples")
    parser.add_argument("--method", choices=("tsai", "park", "horaud", "andreff", "daniilidis"))
    args = parser.parse_args()

    config = yaml.safe_load(resolve_path(args.config).read_text(encoding="utf-8"))
    camera_config = config["camera"]
    calibration = config["calibration"]
    if calibration.get("type") != "eye_on_hand":
        raise ValueError("calibration.type must be eye_on_hand")
    board = calibration["checkerboard"]
    pattern_size = np.asarray(board.get("pattern_size_m", []), dtype=np.float64)
    calculated_pattern_size = np.asarray(
        [
            (int(board["columns"]) + 1) * float(board["square_size_m"]),
            (int(board["rows"]) + 1) * float(board["square_size_m"]),
        ]
    )
    if pattern_size.shape != (2,) or not np.allclose(
        pattern_size, calculated_pattern_size, rtol=0.0, atol=1e-9
    ):
        raise ValueError(
            "checkerboard pattern_size_m must equal (inner corners + 1) * square_size_m"
        )
    samples_required = int(args.samples or calibration["samples"])
    minimum_inliers = int(calibration["minimum_inliers"])
    if samples_required < minimum_inliers:
        raise ValueError("samples must be >= calibration.minimum_inliers")
    output = resolve_path(args.output or camera_config["calibration"])
    method = args.method or calibration["method"]
    session = resolve_path(calibration["session_directory"]) / datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S.%fZ"
    )
    session.mkdir(parents=True, exist_ok=False)

    try:
        import cv2
        import rtde_receive
    except ImportError as error:
        raise RuntimeError("Install requirements-hardware.txt before calibration") from error

    receiver = rtde_receive.RTDEReceiveInterface(str(config["robot"]["host"]))
    base_to_tcp = []
    camera_to_board = []
    reprojection_errors = []
    capture_metrics = []
    intrinsics_reference = None
    distortion_reference = None
    distortion_model_reference = None
    print("Eye-on-hand 标定不会发送任何机械臂运动命令。")
    print(
        "固定棋盘格不动；使用示教器移动相机。棋盘格为 {}x{} 内角点，方格 {:.3f} mm。".format(
            board["columns"], board["rows"], 1000.0 * float(board["square_size_m"])
        )
    )
    try:
        with RealSenseCamera(
            camera_config["width"],
            camera_config["height"],
            camera_config["fps"],
            camera_config.get("serial"),
            camera_config.get("model"),
        ) as camera:
            print("已连接相机：{}，序列号 {}".format(camera.device_name, camera.serial))
            while len(base_to_tcp) < samples_required:
                command = input(
                    "姿态 {}/{}：确认机械臂静止且棋盘完整可见后按回车；输入 q 退出：".format(
                        len(base_to_tcp) + 1, samples_required
                    )
                ).strip().lower()
                if command == "q":
                    raise RuntimeError("calibration cancelled after {} samples".format(len(base_to_tcp)))
                before_pose = np.asarray(receiver.getActualTCPPose(), dtype=np.float64)
                before_speed = np.asarray(receiver.getActualTCPSpeed(), dtype=np.float64)
                started = monotonic()
                frame = camera.read()
                after_pose = np.asarray(receiver.getActualTCPPose(), dtype=np.float64)
                after_speed = np.asarray(receiver.getActualTCPSpeed(), dtype=np.float64)
                metrics = capture_motion_metrics(
                    before_pose, after_pose, before_speed, after_speed, monotonic() - started
                )
                try:
                    validate_stationary(metrics, calibration)
                    camera_board, rms, corners = estimate_checkerboard_pose(
                        frame.color_bgr,
                        frame.intrinsics,
                        int(board["columns"]),
                        int(board["rows"]),
                        float(board["square_size_m"]),
                    )
                    if rms > float(calibration["max_reprojection_error_px"]):
                        raise RuntimeError(
                            "reprojection RMS {:.3f}px exceeds {:.3f}px".format(
                                rms, float(calibration["max_reprojection_error_px"])
                            )
                        )
                    tcp_transform = average_transforms(
                        [tcp_pose_to_transform(before_pose), tcp_pose_to_transform(after_pose)]
                    )
                    if not sufficiently_distinct(
                        tcp_transform,
                        base_to_tcp,
                        float(calibration["minimum_sample_translation_m"]),
                        float(calibration["minimum_sample_rotation_deg"]),
                    ):
                        raise RuntimeError("pose is too similar to an accepted sample")
                    if intrinsics_reference is None:
                        intrinsics_reference = frame.intrinsics.matrix
                        distortion_reference = np.asarray(frame.intrinsics.distortion, dtype=np.float64)
                        distortion_model_reference = frame.intrinsics.distortion_model
                    elif (
                        not np.allclose(intrinsics_reference, frame.intrinsics.matrix, atol=1e-9)
                        or distortion_model_reference != frame.intrinsics.distortion_model
                        or not np.allclose(
                            distortion_reference,
                            np.asarray(frame.intrinsics.distortion, dtype=np.float64),
                            atol=1e-12,
                        )
                    ):
                        raise RuntimeError("RealSense color intrinsics changed during calibration")
                except RuntimeError as error:
                    print("样本拒绝：{}".format(error))
                    continue

                base_to_tcp.append(tcp_transform)
                camera_to_board.append(camera_board)
                reprojection_errors.append(rms)
                capture_metrics.append(metrics)
                annotated = frame.color_bgr.copy()
                cv2.drawChessboardCorners(
                    annotated,
                    (int(board["columns"]), int(board["rows"])),
                    corners.astype(np.float32).reshape(-1, 1, 2),
                    True,
                )
                image_path = session / "sample_{:03d}.png".format(len(base_to_tcp))
                if not cv2.imwrite(str(image_path), annotated):
                    raise RuntimeError("failed to save calibration image {}".format(image_path))
                print(
                    "已接受 {}/{}：重投影 RMS={:.3f}px，图像={}".format(
                        len(base_to_tcp), samples_required, rms, image_path
                    )
                )
    finally:
        disconnect = getattr(receiver, "disconnect", None)
        if disconnect is not None:
            disconnect()

    sample_payload = {
        "base_to_tcp": np.stack(base_to_tcp),
        "camera_to_board": np.stack(camera_to_board),
        "reprojection_error_px": np.asarray(reprojection_errors),
        "capture_inlier": np.ones(len(base_to_tcp), dtype=bool),
        "camera_matrix": intrinsics_reference,
        "distortion": distortion_reference,
        "distortion_model": np.asarray(distortion_model_reference),
        "capture_elapsed_s": np.asarray([item["elapsed_s"] for item in capture_metrics]),
        "capture_translation_delta_m": np.asarray(
            [item["translation_delta_m"] for item in capture_metrics]
        ),
        "capture_rotation_delta_deg": np.asarray(
            [item["rotation_delta_deg"] for item in capture_metrics]
        ),
        "capture_linear_speed_m_s": np.asarray(
            [item["linear_speed_m_s"] for item in capture_metrics]
        ),
        "capture_angular_speed_rad_s": np.asarray(
            [item["angular_speed_rad_s"] for item in capture_metrics]
        ),
    }
    # Persist accepted raw pairs before solving so a failed quality gate remains diagnosable.
    np.savez_compressed(str(session / "samples.npz"), **sample_payload)

    translation_span, rotation_span = transform_spans(base_to_tcp)
    if translation_span < float(calibration["minimum_translation_span_m"]):
        raise RuntimeError("TCP translation span {:.4f}m is insufficient".format(translation_span))
    if rotation_span < float(calibration["minimum_rotation_span_deg"]):
        raise RuntimeError("TCP rotation span {:.2f}deg is insufficient".format(rotation_span))

    tcp_to_camera, inliers, base_to_board, translation_errors, rotation_errors = (
        robust_calibrate_eye_on_hand(
            base_to_tcp,
            camera_to_board,
            method,
            float(calibration["max_board_translation_error_m"]),
            float(calibration["max_board_rotation_error_deg"]),
            minimum_inliers,
        )
    )
    sample_payload["capture_inlier"] = inliers
    np.savez_compressed(str(session / "samples.npz"), **sample_payload)
    metadata = {
        "method": method,
        "configured_tcp_warning": "Changing the active UR TCP invalidates this calibration",
        "checkerboard": {
            "model": str(board.get("model", "unspecified")),
            "columns": int(board["columns"]),
            "rows": int(board["rows"]),
            "square_size_m": float(board["square_size_m"]),
            "pattern_size_m": [float(value) for value in board.get("pattern_size_m", [])],
            "overall_size_m": [float(value) for value in board.get("overall_size_m", [])],
        },
        "camera": {
            "model": camera.device_name,
            "serial": camera.serial,
            "configured_model": camera_config.get("model"),
        },
        "sample_count": len(base_to_tcp),
        "inlier_count": int(inliers.sum()),
        "session_directory": str(session),
        "mean_reprojection_error_px": float(np.mean(reprojection_errors)),
        "max_reprojection_error_px": float(np.max(reprojection_errors)),
        "tcp_translation_span_m": translation_span,
        "tcp_rotation_span_deg": rotation_span,
        "mean_board_translation_error_m": float(np.mean(translation_errors[inliers])),
        "max_board_translation_error_m": float(np.max(translation_errors[inliers])),
        "mean_board_rotation_error_deg": float(np.mean(rotation_errors[inliers])),
        "max_board_rotation_error_deg": float(np.max(rotation_errors[inliers])),
        "base_to_board_validation": base_to_board.tolist(),
    }
    save_eye_on_hand(output, tcp_to_camera, metadata)
    (session / "summary.yaml").write_text(
        yaml.safe_dump(metadata, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    print("标定已保存到 {}".format(output))
    print("T_tcp_camera=\n{}".format(np.array2string(tcp_to_camera, precision=8)))
    print(
        "内点 {}/{}，板位姿残差 max={:.4f}m/{:.2f}deg".format(
            int(inliers.sum()), len(inliers),
            float(np.max(translation_errors[inliers])),
            float(np.max(rotation_errors[inliers])),
        )
    )


if __name__ == "__main__":
    main()
