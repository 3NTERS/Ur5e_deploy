"""Compatibility helpers for loading newer YOLOv8 checkpoints in Ultralytics 8.0.20."""

from __future__ import annotations

import os
import sys


NEWER_MODULE_NAMES = ("block", "conv", "head", "transformer")


def install_checkpoint_module_aliases():
    """Map newer split-module pickle paths to the 8.0.20 monolithic module.

    Ultralytics 8.0.20 exposes ``ultralytics.nn.modules`` as one Python file.
    Newer YOLOv8 checkpoints pickle common layers from submodules such as
    ``ultralytics.nn.modules.conv``. The classes used by the supported YOLOv8n
    checkpoint still exist in the old module, so aliases let torch unpickle it
    without changing the pinned Python 3.7 runtime.
    """
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    import ultralytics.nn.modules as legacy_modules

    if not hasattr(legacy_modules, "__path__"):
        for name in NEWER_MODULE_NAMES:
            sys.modules.setdefault(
                "ultralytics.nn.modules.{}".format(name), legacy_modules
            )

    # If Ultralytics was imported before this helper, update its cached flag as
    # well. A missing Python module is not a package requirement and must never
    # trigger a deployment-time pip invocation.
    try:
        import ultralytics.yolo.utils as yolo_utils
        import ultralytics.yolo.utils.checks as yolo_checks

        yolo_utils.AUTOINSTALL = False
        yolo_checks.AUTOINSTALL = False
    except (ImportError, AttributeError):
        pass


def load_yolo_model(weights):
    """Load PT/ONNX weights with deterministic, offline-compatible behavior."""
    install_checkpoint_module_aliases()
    from ultralytics import YOLO

    try:
        return YOLO(str(weights))
    except (AttributeError, ModuleNotFoundError) as error:
        raise RuntimeError(
            "YOLO weights use layers unavailable in pinned Ultralytics 8.0.20. "
            "Export the model to ONNX in its training environment, or retrain "
            "with the deployment version. Original error: {}".format(error)
        ) from error
