import tempfile
import types
import unittest
from pathlib import Path

import yaml

from ur5e_comm.yolo_training import TrainingOptions, run_training, validate_dataset


def make_dataset(root, names=None, label="0 0.5 0.5 0.4 0.4\n"):
    root = Path(root)
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        (root / "images" / split / "{}.jpg".format(split)).write_bytes(b"fake image")
        (root / "labels" / split / "{}.txt".format(split)).write_text(label, encoding="utf-8")
    data = root / "data.yaml"
    data.write_text(
        yaml.safe_dump({
            "path": str(root),
            "train": "images/train",
            "val": "images/val",
            "names": {0: "object"} if names is None else names,
        }),
        encoding="utf-8",
    )
    return data


class FakeYoloFactory:
    def __init__(self, run_directory):
        self.run_directory = Path(run_directory)
        self.instances = []

    def __call__(self, model):
        instance = FakeYolo(model, self.run_directory)
        self.instances.append(instance)
        return instance


class FakeYolo:
    names = {0: "object"}

    def __init__(self, model, run_directory):
        self.model = model
        self.run_directory = run_directory
        self.trainer = None
        self.train_arguments = None
        self.val_arguments = None
        self.predict_arguments = None

    def train(self, **kwargs):
        self.train_arguments = kwargs
        best = self.run_directory / "weights" / "best.pt"
        best.parent.mkdir(parents=True, exist_ok=True)
        best.write_bytes(b"fake yolo weights")
        self.trainer = types.SimpleNamespace(
            best=best,
            save_dir=self.run_directory,
            metrics={"metrics/mAP50(B)": 0.875},
        )

    def val(self, **kwargs):
        self.val_arguments = kwargs

    def predict(self, **kwargs):
        self.predict_arguments = kwargs
        return []


class TestDatasetValidation(unittest.TestCase):
    def test_accepts_single_object_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            spec = validate_dataset(make_dataset(directory))
            self.assertEqual(len(spec.train_images), 1)
            self.assertEqual(len(spec.val_images), 1)

    def test_rejects_wrong_class_name(self):
        with tempfile.TemporaryDirectory() as directory:
            data = make_dataset(directory, names={0: "ball"})
            with self.assertRaisesRegex(ValueError, "exactly one class"):
                validate_dataset(data)

    def test_rejects_bbox_outside_image(self):
        with tempfile.TemporaryDirectory() as directory:
            data = make_dataset(directory, label="0 0.9 0.5 0.4 0.4\n")
            with self.assertRaisesRegex(ValueError, "outside the image"):
                validate_dataset(data)

    def test_rejects_missing_label(self):
        with tempfile.TemporaryDirectory() as directory:
            data = make_dataset(directory)
            (Path(directory) / "labels" / "val" / "val.txt").unlink()
            with self.assertRaisesRegex(ValueError, "missing YOLO label"):
                validate_dataset(data)


class TestTrainingWorkflow(unittest.TestCase):
    def test_trains_validates_and_publishes_explicitly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = make_dataset(root / "dataset")
            output = root / "published" / "object_yolo.pt"
            factory = FakeYoloFactory(root / "runs" / "object")
            options = TrainingOptions(
                data=data,
                epochs=3,
                imgsz=320,
                batch=2,
                device="cpu",
                workers=0,
                seed=7,
                project=root / "runs",
                publish=True,
                output=output,
            )
            result = run_training(options, yolo_factory=factory)

            self.assertEqual(factory.instances[0].train_arguments["epochs"], 3)
            self.assertEqual(factory.instances[0].train_arguments["seed"], 7)
            self.assertIsNotNone(factory.instances[1].val_arguments)
            self.assertEqual(factory.instances[1].val_arguments["name"], "validation")
            self.assertEqual(
                Path(factory.instances[1].val_arguments["project"]), root / "runs" / "object"
            )
            self.assertEqual(Path(factory.instances[1].predict_arguments["source"]).name, "val.jpg")
            self.assertEqual(output.read_bytes(), b"fake yolo weights")
            self.assertTrue(result["published_metadata"].is_file())
            metadata = yaml.safe_load(result["published_metadata"].read_text(encoding="utf-8"))
            self.assertEqual(metadata["dataset"]["class_names"], ["object"])
            self.assertEqual(metadata["metrics"]["metrics/mAP50(B)"], 0.875)
            self.assertEqual(len(metadata["sha256"]), 64)

    def test_does_not_publish_without_flag(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = make_dataset(root / "dataset")
            output = root / "published" / "object_yolo.pt"
            factory = FakeYoloFactory(root / "runs" / "object")
            result = run_training(
                TrainingOptions(data=data, project=root / "runs", output=output),
                yolo_factory=factory,
            )
            self.assertFalse(output.exists())
            self.assertIsNone(result["published_weights"])
            self.assertTrue(result["metadata"].is_file())


if __name__ == "__main__":
    unittest.main()
