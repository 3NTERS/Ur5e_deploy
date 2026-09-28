#!/usr/bin/env python3
"""Train and optionally publish the single-class object detector."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ur5e_comm.yolo_training import TrainingOptions, run_training  # noqa: E402


def repo_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Ultralytics YOLO Detection data.yaml")
    parser.add_argument("--model", default="yolov8n.pt", help="pretrained .pt or model .yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0", help="CUDA index such as 0, or cpu")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", help="path to a YOLO run's weights/last.pt")
    parser.add_argument("--project", default="resources/training/yolo")
    parser.add_argument("--name", default="object_yolov8n")
    parser.add_argument("--publish", action="store_true", help="publish best.pt for deployment")
    parser.add_argument("--output", default="resources/models/vision/object_yolo.pt")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    options = TrainingOptions(
        data=Path(args.data), model=args.model, epochs=args.epochs, imgsz=args.imgsz,
        batch=args.batch, device=args.device, workers=args.workers, seed=args.seed,
        resume=Path(args.resume) if args.resume else None,
        project=repo_path(args.project), name=args.name, publish=args.publish,
        output=repo_path(args.output),
    )
    result = run_training(options)
    print("Best weights: {}".format(result["best"]))
    print("Training metadata: {}".format(result["metadata"]))
    if result["published_weights"]:
        print("Published weights: {}".format(result["published_weights"]))
        print("Published metadata: {}".format(result["published_metadata"]))
    else:
        print("Not published. Review validation metrics and rerun with --publish when accepted.")


if __name__ == "__main__":
    main()
