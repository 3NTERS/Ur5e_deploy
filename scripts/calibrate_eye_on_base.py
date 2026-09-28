#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

from ur5e_comm.geometry import average_transforms, estimate_eye_on_base, save_eye_on_base
from ur5e_comm.vision import RealSenseCamera


ROOT = Path(__file__).resolve().parents[1]


def resolve_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def main():
    parser = argparse.ArgumentParser(description="Calibrate a fixed eye-on-base camera from a QR code")
    parser.add_argument("--config", default="resources/config/ur5e_deploy.yaml")
    parser.add_argument("--output", help="override camera.calibration output path")
    args = parser.parse_args()
    config = yaml.safe_load(resolve_path(args.config).read_text(encoding="utf-8"))
    camera_cfg = config["camera"]
    calibration_cfg = config["calibration"]
    output = resolve_path(args.output or camera_cfg["calibration"])
    base_to_qr = np.asarray(calibration_cfg["base_to_qr"], dtype=np.float64)
    samples = int(calibration_cfg["samples"])
    maximum_error = float(calibration_cfg["max_reprojection_error_px"])

    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("Install requirements-hardware.txt before calibration") from error
    detector = cv2.QRCodeDetector()
    transforms = []
    errors = []
    print("保持二维码固定并完整出现在画面中；正在采集 {} 个有效样本。".format(samples))
    with RealSenseCamera(
        camera_cfg["width"],
        camera_cfg["height"],
        camera_cfg["fps"],
        camera_cfg.get("serial"),
    ) as camera:
        attempts = 0
        while len(transforms) < samples and attempts < samples * 20:
            attempts += 1
            frame = camera.read()
            _, corners, _ = detector.detectAndDecode(frame.color_bgr)
            if corners is None:
                continue
            transform, rms = estimate_eye_on_base(
                corners,
                frame.intrinsics.matrix,
                np.asarray(frame.intrinsics.distortion, dtype=np.float64),
                float(calibration_cfg["qr_size_m"]),
                base_to_qr,
            )
            if rms > maximum_error:
                continue
            transforms.append(transform)
            errors.append(rms)
            print("有效样本 {}/{}，重投影 RMS={:.3f}px".format(len(transforms), samples, rms))
    if len(transforms) < samples:
        raise RuntimeError("只得到 {}/{} 个有效标定样本".format(len(transforms), samples))
    base_to_camera = average_transforms(transforms)
    translations = np.stack([item[:3, 3] for item in transforms])
    translation_std = translations.std(axis=0)
    metadata = {
        "sample_count": samples,
        "qr_size_m": float(calibration_cfg["qr_size_m"]),
        "mean_reprojection_error_px": float(np.mean(errors)),
        "max_reprojection_error_px": float(np.max(errors)),
        "translation_std_m": translation_std.tolist(),
        "base_to_qr": base_to_qr.tolist(),
    }
    save_eye_on_base(output, base_to_camera, metadata)
    print("标定已保存到 {}".format(output))
    print("T_base_camera=\n{}".format(np.array2string(base_to_camera, precision=8)))
    print("平移标准差(m): {}".format(translation_std.tolist()))


if __name__ == "__main__":
    main()
