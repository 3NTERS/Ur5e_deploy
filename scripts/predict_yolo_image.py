#!/usr/bin/env python3
"""Run YOLO prediction on images and save annotated results."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ur5e_comm.yolo_compat import load_yolo_model


DEFAULT_MODEL = (
    ROOT
    / "resources"
    / "models"
    / "vision"
    / "object_yolo.pt"
)
DEFAULT_OUTPUT_DIRECTORY = ROOT / "resources" / "predictions"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def resolve_path(path: Path) -> Path:
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def collect_images(path: Path) -> list[Path]:
    path = resolve_path(path)
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError("Input image or directory does not exist: {}".format(path))
    images = sorted(
        item for item in path.iterdir()
        if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise ValueError("Input directory contains no supported images: {}".format(path))
    return images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict one image or every image in a directory."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="input image or directory",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="trained .pt weights",
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
    image_paths = collect_images(args.input)
    model_path = resolve_path(args.model)
    output_directory = resolve_path(args.output_dir)

    if not model_path.is_file():
        raise FileNotFoundError(f"Model weights do not exist: {model_path}")
    if not 0.0 <= args.conf <= 1.0:
        raise ValueError("--conf must be between 0 and 1")
    if not 0.0 <= args.iou <= 1.0:
        raise ValueError("--iou must be between 0 and 1")
    if args.imgsz <= 0:
        raise ValueError("--imgsz must be positive")

    model = load_yolo_model(model_path)
    output_directory.mkdir(parents=True, exist_ok=True)
    import cv2

    for image_path in image_paths:
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
        output_path = output_directory / f"{image_path.stem}_pred.jpg"
        boxes = result.boxes
        detection_count = 0 if boxes is None else len(boxes)
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError("Failed to read image: {}".format(image_path))
        print(f"Image: {image_path}")
        print(f"Model: {model_path}")
        print(f"Detections: {detection_count}")

        if boxes is not None:
            for index, box in enumerate(boxes, 1):
                class_id = int(box.cls.item())
                confidence = float(box.conf.item())
                x1, y1, x2, y2 = (float(value) for value in box.xyxy[0].tolist())
                label = "{} {:.2f}".format(
                    class_name(model.names, class_id), confidence
                )
                cv2.rectangle(
                    image,
                    (int(round(x1)), int(round(y1))),
                    (int(round(x2)), int(round(y2))),
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    image,
                    label,
                    (int(round(x1)), max(15, int(round(y1)) - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
                print(
                    f"{index}: class={class_name(model.names, class_id)} "
                    f"id={class_id} confidence={confidence:.4f} "
                    f"xyxy=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})"
                )

        if not cv2.imwrite(str(output_path), image):
            raise RuntimeError("Failed to write annotated image: {}".format(output_path))
        print(f"Annotated image: {output_path}")


if __name__ == "__main__":
    main()
