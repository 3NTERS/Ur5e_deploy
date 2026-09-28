#!/usr/bin/env python3
"""Run YOLO prediction on one image and save an annotated result."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = (
    ROOT
    / "resources"
    / "training"
    / "yolo"
    / "object_yolov8n"
    / "weights"
    / "best.pt"
)
DEFAULT_OUTPUT_DIRECTORY = ROOT / "tests" / "predictions"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict one image with a trained YOLO model."
    )
    parser.add_argument("--image", type=Path, required=True, help="input image path")
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="trained .pt weights (default: current object_yolov8n best.pt)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help="directory for the annotated image",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="inference image size")
    parser.add_argument("--device", default="0", help="CUDA device index or cpu")
    return parser.parse_args()


def class_name(names, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def main() -> None:
    args = parse_args()
    image_path = args.image.expanduser().resolve()
    model_path = args.model.expanduser().resolve()
    output_directory = args.output_dir.expanduser().resolve()

    if not image_path.is_file():
        raise FileNotFoundError(f"Input image does not exist: {image_path}")
    if not model_path.is_file():
        raise FileNotFoundError(f"Model weights do not exist: {model_path}")
    if not 0.0 <= args.conf <= 1.0:
        raise ValueError("--conf must be between 0 and 1")
    if not 0.0 <= args.iou <= 1.0:
        raise ValueError("--iou must be between 0 and 1")
    if args.imgsz <= 0:
        raise ValueError("--imgsz must be positive")

    model = YOLO(str(model_path))
    results = model.predict(
        source=str(image_path),
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=str(args.device),
        save=False,
        verbose=False,
    )
    if not results:
        raise RuntimeError("YOLO returned no prediction result")

    result = results[0]
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / f"{image_path.stem}_pred.jpg"
    result.save(filename=str(output_path))

    boxes = result.boxes
    detection_count = 0 if boxes is None else len(boxes)
    print(f"Image: {image_path}")
    print(f"Model: {model_path}")
    print(f"Detections: {detection_count}")

    if boxes is not None:
        for index, box in enumerate(boxes, 1):
            class_id = int(box.cls.item())
            confidence = float(box.conf.item())
            x1, y1, x2, y2 = (float(value) for value in box.xyxy[0].tolist())
            print(
                f"{index}: class={class_name(result.names, class_id)} "
                f"id={class_id} confidence={confidence:.4f} "
                f"xyxy=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})"
            )

    print(f"Annotated image: {output_path}")


if __name__ == "__main__":
    main()
