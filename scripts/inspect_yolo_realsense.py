#!/usr/bin/env python3
"""Print multi-class YOLO boxes and aligned RealSense centre depths without using the robot."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from time import monotonic

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ur5e_comm.geometry import deproject_pixel
from ur5e_comm.vision import RealSenseCamera, robust_depth
from ur5e_comm.yolo_compat import load_yolo_model


def resolve_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def class_names(result, model):
    names = getattr(result, "names", None)
    if names is None:
        names = getattr(model, "names", None)
    if isinstance(names, dict):
        return {int(key): str(value) for key, value in names.items()}
    if isinstance(names, (list, tuple)):
        return {index: str(value) for index, value in enumerate(names)}
    raise RuntimeError("YOLO weights do not expose class names")


def is_target(class_id, class_name, target):
    return target is None or str(target) in (str(class_id), class_name)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="resources/config/ur5e_deploy.yaml")
    parser.add_argument("--weights", help="override vision.weights")
    parser.add_argument("--target-class", help="override vision.target_class by name or ID")
    parser.add_argument("--confidence", type=float, help="override vision.confidence")
    parser.add_argument("--device", help="CUDA index such as 0, or cpu")
    parser.add_argument(
        "--frames",
        type=int,
        help="number of frames; default 1 in console mode and unlimited in --view mode",
    )
    parser.add_argument(
        "--view",
        action="store_true",
        help="show a live annotated stream; press q or Esc to quit",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        help="terminal output interval; default 1 frame in console mode, 30 in --view mode",
    )
    return parser.parse_args()


def class_color(class_id):
    """Return a stable, bright BGR color for one class ID."""
    palette = (
        (60, 220, 60),
        (255, 160, 40),
        (60, 180, 255),
        (220, 80, 220),
        (255, 100, 100),
        (80, 240, 240),
    )
    return palette[int(class_id) % len(palette)]


def draw_detection(image, row, selected):
    import cv2

    x1, y1, x2, y2 = [int(round(value)) for value in row["xyxy"]]
    u, v = [int(round(value)) for value in row["center"]]
    color = class_color(row["class_id"])
    thickness = 3 if selected else 2
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    cv2.drawMarker(image, (u, v), color, cv2.MARKER_CROSS, 14, 2)
    depth = row["center_depth"]
    depth_label = (
        "depth invalid" if depth is None else "surf {:.3f} / center {:.3f}m".format(
            row["surface_depth"], depth
        )
    )
    target_label = " TARGET" if selected else ""
    label = "{} {:.2f} {}{}".format(
        row["class_name"], row["confidence"], depth_label, target_label
    )
    (text_width, text_height), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
    )
    text_top = max(0, y1 - text_height - baseline - 6)
    cv2.rectangle(
        image,
        (x1, text_top),
        (x1 + text_width + 6, text_top + text_height + baseline + 6),
        color,
        -1,
    )
    cv2.putText(
        image,
        label,
        (x1 + 3, text_top + text_height + 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )


def main():
    args = parse_args()
    frame_limit = args.frames if args.frames is not None else (0 if args.view else 1)
    if frame_limit < 0 or (frame_limit == 0 and not args.view):
        raise ValueError("--frames must be positive, or 0 together with --view")
    print_every = args.print_every if args.print_every is not None else (30 if args.view else 1)
    if print_every <= 0:
        raise ValueError("--print-every must be positive")
    config = yaml.safe_load(resolve_path(args.config).read_text(encoding="utf-8"))
    camera_config = config["camera"]
    vision = config["vision"]
    weights = resolve_path(args.weights or vision["weights"])
    if not weights.is_file():
        raise FileNotFoundError("YOLO weights do not exist: {}".format(weights))
    target = args.target_class if args.target_class is not None else vision.get("target_class")
    confidence = args.confidence if args.confidence is not None else float(vision["confidence"])
    device = args.device if args.device is not None else vision.get("device")

    model = load_yolo_model(weights)
    if args.view:
        import cv2
    try:
        with RealSenseCamera(
            camera_config["width"],
            camera_config["height"],
            camera_config["fps"],
            camera_config.get("serial"),
            camera_config.get("model"),
        ) as camera:
            print("camera={!r} serial={} weights={}".format(camera.device_name, camera.serial, weights))
            print("target_class={!r} confidence={:.3f} device={!r}".format(target, confidence, device))
            if args.view:
                print("实时窗口已启动；按 q 或 Esc 退出。")
            frame_index = 0
            previous_time = monotonic()
            while frame_limit == 0 or frame_index < frame_limit:
                frame_index += 1
                frame = camera.read()
                predictions = model.predict(
                    source=frame.color_bgr,
                    conf=confidence,
                    device=device,
                    verbose=False,
                )
                result = predictions[0]
                names = class_names(result, model)
                should_print = frame_index == 1 or frame_index % print_every == 0
                if should_print:
                    print("\nframe={} classes={}".format(frame_index, names))
                boxes = result.boxes
                if boxes is None or len(boxes) == 0:
                    if should_print:
                        print("  no detections")
                    rows = []
                else:
                    rows = []
                    for index in range(len(boxes)):
                        class_id = int(boxes.cls[index].item())
                        class_name = names[class_id]
                        score = float(boxes.conf[index].item())
                        xyxy = boxes.xyxy[index].detach().cpu().numpy().astype(np.float64)
                        u = float((xyxy[0] + xyxy[2]) * 0.5)
                        v = float((xyxy[1] + xyxy[3]) * 0.5)
                        surface_depth = None
                        center_depth = None
                        try:
                            surface_depth = robust_depth(
                                frame.depth_m,
                                u,
                                v,
                                int(vision["depth_radius"]),
                                float(vision["depth_min"]),
                                float(vision["depth_max"]),
                            )
                            center_depth = surface_depth + float(vision["center_depth_offset_m"])
                            point = deproject_pixel((u, v), center_depth, frame.intrinsics)
                            depth_text = (
                                "surface_depth={:.4f}m center_depth={:.4f}m "
                                "center_camera_xyz=[{:.4f}, {:.4f}, {:.4f}]m"
                            ).format(
                                surface_depth, center_depth, point[0], point[1], point[2]
                            )
                        except (RuntimeError, ValueError) as error:
                            depth_text = "depth=INVALID ({})".format(error)
                        row = {
                            "index": index,
                            "class_id": class_id,
                            "class_name": class_name,
                            "confidence": score,
                            "xyxy": xyxy,
                            "center": (u, v),
                            "target": is_target(class_id, class_name, target),
                            "surface_depth": surface_depth,
                            "center_depth": center_depth,
                            "depth_text": depth_text,
                        }
                        rows.append(row)

                target_rows = [row for row in rows if row["target"]]
                selected = max(target_rows, key=lambda row: row["confidence"]) if target_rows else None
                if should_print:
                    for row in rows:
                        marker = " [DEPLOYMENT TARGET]" if row is selected else ""
                        box = row["xyxy"]
                        print(
                            "  #{:02d} id={} name={!r} conf={:.3f} "
                            "xyxy=[{:.1f}, {:.1f}, {:.1f}, {:.1f}] center=({:.1f}, {:.1f}) {}{}".format(
                                row["index"],
                                row["class_id"],
                                row["class_name"],
                                row["confidence"],
                                box[0], box[1], box[2], box[3],
                                row["center"][0], row["center"][1],
                                row["depth_text"],
                                marker,
                            )
                        )
                    if selected is None:
                        print("  target {!r} was not detected".format(target))

                if args.view:
                    annotated = frame.color_bgr.copy()
                    for row in rows:
                        draw_detection(annotated, row, row is selected)
                    current_time = monotonic()
                    fps = 1.0 / max(current_time - previous_time, 1e-9)
                    previous_time = current_time
                    cv2.putText(
                        annotated,
                        "FPS {:.1f} | q/Esc quit".format(fps),
                        (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
                    window_name = "D435i YOLO RGB-D"
                    cv2.imshow(window_name, annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27) or cv2.getWindowProperty(
                        window_name, cv2.WND_PROP_VISIBLE
                    ) < 1:
                        break
    finally:
        if args.view:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
