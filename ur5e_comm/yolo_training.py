"""Validated single-class YOLO training and explicit deployment publishing."""

from __future__ import annotations

import hashlib
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class DatasetSpec:
    source: Path
    root: Path
    train_directory: Path
    val_directory: Path
    train_images: tuple
    val_images: tuple


@dataclass(frozen=True)
class TrainingOptions:
    data: Path
    model: str = "yolov8n.pt"
    epochs: int = 100
    imgsz: int = 640
    batch: int = 16
    device: str = "0"
    workers: int = 4
    seed: int = 42
    resume: Path = None
    project: Path = ROOT / "resources/training/yolo"
    name: str = "object_yolov8n"
    publish: bool = False
    output: Path = ROOT / "resources/models/vision/object_yolo.pt"


def _normalise_names(names):
    if isinstance(names, list):
        result = list(names)
    elif isinstance(names, dict):
        try:
            indexed = {int(key): str(value) for key, value in names.items()}
        except (TypeError, ValueError) as error:
            raise ValueError("dataset names keys must be integer class IDs") from error
        if sorted(indexed) != list(range(len(indexed))):
            raise ValueError("dataset class IDs must be contiguous and start at 0")
        result = [indexed[index] for index in range(len(indexed))]
    else:
        raise ValueError("dataset names must be a list or mapping")
    if result != ["object"]:
        raise ValueError("dataset must contain exactly one class: 0: object")
    return result


def _split_directory(config, data_path, split):
    value = config.get(split)
    if not isinstance(value, str) or not value.strip():
        raise ValueError("dataset {!r} must be a single image directory".format(split))
    root_value = config.get("path")
    if root_value is None:
        root = data_path.parent
    else:
        root = Path(str(root_value)).expanduser()
        if not root.is_absolute():
            root = data_path.parent / root
    root = root.resolve()
    directory = Path(value).expanduser()
    if not directory.is_absolute():
        directory = root / directory
    return root, directory.resolve()


def _label_directory(root, image_directory):
    try:
        relative = image_directory.relative_to(root)
    except ValueError as error:
        raise ValueError("image directory must be inside dataset path: {}".format(image_directory)) from error
    parts = list(relative.parts)
    try:
        index = parts.index("images")
    except ValueError as error:
        raise ValueError("split path must use the standard images/<split> layout") from error
    parts[index] = "labels"
    return root.joinpath(*parts)


def _validate_label(label_path):
    rows = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("label file is empty: {}".format(label_path))
    for line_number, row in enumerate(rows, 1):
        fields = row.split()
        if len(fields) != 5:
            raise ValueError("{}:{} must contain class x_center y_center width height".format(label_path, line_number))
        try:
            class_value = float(fields[0])
            values = [float(value) for value in fields[1:]]
        except ValueError as error:
            raise ValueError("{}:{} contains a non-numeric value".format(label_path, line_number)) from error
        if class_value != 0.0 or not class_value.is_integer():
            raise ValueError("{}:{} class ID must be 0 (object)".format(label_path, line_number))
        x_center, y_center, width, height = values
        if not (0.0 <= x_center <= 1.0 and 0.0 <= y_center <= 1.0):
            raise ValueError("{}:{} box centre must be normalized to [0, 1]".format(label_path, line_number))
        if not (0.0 < width <= 1.0 and 0.0 < height <= 1.0):
            raise ValueError("{}:{} box size must be normalized to (0, 1]".format(label_path, line_number))
        tolerance = 1e-6
        if (x_center - width / 2.0 < -tolerance or x_center + width / 2.0 > 1.0 + tolerance or
                y_center - height / 2.0 < -tolerance or y_center + height / 2.0 > 1.0 + tolerance):
            raise ValueError("{}:{} box extends outside the image".format(label_path, line_number))


def _validate_split(root, directory, split):
    if not directory.is_dir():
        raise ValueError("{} image directory does not exist: {}".format(split, directory))
    images = tuple(sorted(path for path in directory.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES))
    if not images:
        raise ValueError("{} image directory contains no supported images: {}".format(split, directory))
    labels = _label_directory(root, directory)
    if not labels.is_dir():
        raise ValueError("{} label directory does not exist: {}".format(split, labels))
    for image in images:
        relative = image.relative_to(directory).with_suffix(".txt")
        label = labels / relative
        if not label.is_file():
            raise ValueError("missing YOLO label for {}: {}".format(image, label))
        _validate_label(label)
    return images


def validate_dataset(data):
    """Validate the strict one-class X-AnyLabeling YOLO export contract."""
    data_path = Path(data).expanduser().resolve()
    if not data_path.is_file():
        raise ValueError("dataset YAML does not exist: {}".format(data_path))
    config = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("dataset YAML must contain a mapping")
    _normalise_names(config.get("names"))
    train_root, train_directory = _split_directory(config, data_path, "train")
    val_root, val_directory = _split_directory(config, data_path, "val")
    if train_root != val_root:
        raise ValueError("train and val must share one dataset path")
    train_images = _validate_split(train_root, train_directory, "train")
    val_images = _validate_split(val_root, val_directory, "val")
    overlap = set(train_images).intersection(val_images)
    if overlap:
        raise ValueError("train and val contain the same image paths")
    return DatasetSpec(data_path, train_root, train_directory, val_directory, train_images, val_images)


def _relative_or_absolute(path, root):
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def write_resolved_manifest(spec, project):
    """Write a persistent absolute-root manifest so resume does not depend on cwd."""
    project = Path(project).expanduser().resolve()
    manifest_directory = project / "_datasets"
    manifest_directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "path": str(spec.root),
        "train": _relative_or_absolute(spec.train_directory, spec.root),
        "val": _relative_or_absolute(spec.val_directory, spec.root),
        "names": {0: "object"},
    }
    digest = hashlib.sha256((str(spec.source) + yaml.safe_dump(payload, sort_keys=True)).encode("utf-8")).hexdigest()[:12]
    destination = manifest_directory / "{}-{}.yaml".format(spec.source.stem, digest)
    _atomic_text(destination, yaml.safe_dump(payload, sort_keys=False, allow_unicode=True))
    return destination


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_version(distribution):
    try:
        try:
            from importlib.metadata import version
        except ImportError:
            from importlib_metadata import version
        return version(distribution)
    except Exception:
        return "unknown"


def _prepare_ultralytics_cache(project):
    """Keep plotting caches local and prevent Ultralytics' first-run font download."""
    cache = Path(project) / "_cache"
    matplotlib_cache = cache / "matplotlib"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))


def _ensure_ascii_font():
    # Ultralytics 8.0.20 downloads Arial.ttf during dataset validation. The
    # detector has one ASCII class name, so a locally installed sans font is
    # an equivalent offline substitute.
    from ultralytics.yolo.utils import USER_CONFIG_DIR

    destination = Path(USER_CONFIG_DIR) / "Arial.ttf"
    if destination.is_file():
        return destination
    candidates = (
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"),
        Path("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"),
    )
    source = next((candidate for candidate in candidates if candidate.is_file()), None)
    if source is None:
        raise RuntimeError("no local TrueType font found; install fonts-dejavu-core")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(source), str(destination))
    return destination


def _plain(value):
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _model_names(model):
    names = getattr(model, "names", None)
    if isinstance(names, dict):
        return [str(names[index]) for index in sorted(names)]
    if isinstance(names, (list, tuple)):
        return [str(name) for name in names]
    raise RuntimeError("trained YOLO weights do not expose class names")


def _atomic_text(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(str(temporary), str(path))


def _atomic_copy(source, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    shutil.copy2(str(source), str(temporary))
    os.replace(str(temporary), str(destination))


def run_training(options, yolo_factory=None):
    """Train, validate, smoke-predict, and optionally publish deployable weights."""
    if options.epochs <= 0 or options.imgsz <= 0 or options.batch <= 0 or options.workers < 0:
        raise ValueError("epochs/imgsz/batch must be positive and workers must be non-negative")
    spec = validate_dataset(options.data)
    project = Path(options.project).expanduser().resolve()
    manifest = write_resolved_manifest(spec, project)
    if yolo_factory is None:
        _prepare_ultralytics_cache(project)
        from ultralytics import YOLO
        _ensure_ascii_font()
        yolo_factory = YOLO

    resume = Path(options.resume).expanduser().resolve() if options.resume else None
    if resume is not None:
        if not resume.is_file():
            raise ValueError("resume checkpoint does not exist: {}".format(resume))
        if not (resume.parent.parent / "args.yaml").is_file():
            raise ValueError("resume checkpoint must be inside a YOLO run with args.yaml")
    model = yolo_factory(str(resume) if resume else options.model)
    train_arguments = {
        "data": str(manifest),
        "epochs": int(options.epochs),
        "imgsz": int(options.imgsz),
        "batch": int(options.batch),
        "device": str(options.device),
        "workers": int(options.workers),
        "seed": int(options.seed),
        "project": str(project),
        "name": str(options.name),
    }
    if resume is not None:
        train_arguments["resume"] = True
    model.train(**train_arguments)
    trainer = getattr(model, "trainer", None)
    best = Path(getattr(trainer, "best", "")) if trainer is not None else Path()
    if not best.is_file():
        raise RuntimeError("training completed without a best.pt checkpoint")

    metrics = _plain(getattr(trainer, "metrics", {}) or {})
    run_directory = Path(getattr(trainer, "save_dir", best.parent.parent))
    best_model = yolo_factory(str(best))
    if _model_names(best_model) != ["object"]:
        raise RuntimeError("trained weights must contain exactly one class named object")
    best_model.val(
        data=str(manifest), imgsz=int(options.imgsz), batch=int(options.batch),
        device=str(options.device), workers=int(options.workers), verbose=True,
        project=str(run_directory), name="validation", exist_ok=True,
    )
    prediction = best_model.predict(
        source=str(spec.val_images[0]), imgsz=int(options.imgsz),
        device=str(options.device), save=False, verbose=False,
    )
    if prediction is None:
        raise RuntimeError("YOLO smoke prediction returned no result object")

    metadata = {
        "format_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_dataset": str(spec.source),
        "resolved_dataset": str(manifest),
        "dataset": {
            "class_names": ["object"],
            "train_images": len(spec.train_images),
            "val_images": len(spec.val_images),
        },
        "training": {
            "model": str(resume) if resume else options.model,
            "arguments": _plain(train_arguments),
            "resumed": resume is not None,
        },
        "metrics": metrics,
        "best_weights": str(best.resolve()),
        "sha256": _sha256(best),
        "packages": {
            "ultralytics": _package_version("ultralytics"),
            "torch": _package_version("torch"),
            "torchvision": _package_version("torchvision"),
            "numpy": _package_version("numpy"),
        },
    }
    run_metadata = run_directory / "training_metadata.yaml"
    _atomic_text(run_metadata, yaml.safe_dump(metadata, sort_keys=False, allow_unicode=True))

    published_weights = None
    published_metadata = None
    if options.publish:
        published_weights = Path(options.output).expanduser().resolve()
        published_metadata = published_weights.with_name(published_weights.stem + ".meta.yaml")
        _atomic_copy(best, published_weights)
        published_payload = dict(metadata)
        published_payload["published_weights"] = str(published_weights)
        published_payload["sha256"] = _sha256(published_weights)
        _atomic_text(published_metadata, yaml.safe_dump(published_payload, sort_keys=False, allow_unicode=True))

    return {
        "best": best,
        "metadata": run_metadata,
        "published_weights": published_weights,
        "published_metadata": published_metadata,
        "metrics": metrics,
    }
