import os
import tempfile
import threading
import unittest
import types
from unittest import mock
from pathlib import Path
from time import monotonic, sleep

import numpy as np

from ur5e_comm.deployment import DeploymentSession
from ur5e_comm.geometry import (
    CameraIntrinsics,
    average_transforms,
    calibrate_eye_on_base,
    calibrate_eye_on_hand,
    deproject_pixel,
    eye_on_base_residuals,
    eye_on_hand_residuals,
    estimate_checkerboard_pose,
    estimate_eye_on_base,
    invert_transform,
    load_eye_on_hand,
    load_eye_on_base,
    make_transform,
    robust_calibrate_eye_on_base,
    robust_calibrate_eye_on_hand,
    save_eye_on_hand,
    save_eye_on_base,
    tcp_pose_to_transform,
    transform_point,
    transform_spans,
)
from ur5e_comm.observation import (
    ProjectedRobotState,
    MujocoStateProjector,
    SafetyMonitor,
    TRAINING_LOWER,
    TRAINING_UPPER,
    Ur5eActionMapper,
    Ur5eObservationBuilder,
)
from ur5e_comm.robot import RobotSnapshot, UR5eHardware
from ur5e_comm.vision import (
    EyeOnBaseObjectTracker,
    ObjectDetection,
    TrackedObjectState,
    YoloEyeOnBaseLocator,
    robust_depth,
)
from ur5e_comm.vision import RGBDFrame, YoloInitialObjectLocator


def snapshot():
    return RobotSnapshot(
        joint_position=np.array([-1.57, -1.57, 1.57, -1.57, -1.57, 0.0]),
        joint_velocity=np.zeros(6),
        tcp_pose=np.zeros(6),
        tcp_speed=np.zeros(6),
        gripper_position=0,
        timestamp=monotonic(),
    )


class FakeProjector:
    def reset(self):
        pass

    def project(self, state):
        master = state.gripper_position / 255.0 * 0.72
        qpos = np.concatenate((state.joint_position, [master, -master, master, master, -master, master]))
        return ProjectedRobotState(
            qpos,
            np.concatenate((state.joint_velocity, np.zeros(6))),
            np.array([0.5, 0.0, 0.4]),
            np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([[0.55, -0.02, 0.35], [0.55, 0.02, 0.35]]),
        )

    def palm_position_for(self, arm_position, gripper_position):
        return np.array([0.5, 0.0, 0.4])


class TestGeometry(unittest.TestCase):
    def test_deprojection_and_transform(self):
        intrinsics = CameraIntrinsics(100.0, 100.0, 50.0, 40.0, 100, 80)
        point = deproject_pixel((60.0, 20.0), 2.0, intrinsics)
        np.testing.assert_allclose(point, [0.2, -0.4, 2.0])
        transform = np.eye(4)
        transform[:3, 3] = [1.0, 2.0, 3.0]
        transformed = transform_point(transform, point)
        np.testing.assert_allclose(transform_point(invert_transform(transform), transformed), point)

    def test_inverse_brown_conrady_deprojection(self):
        coefficients = (0.1, -0.02, 0.003, -0.004, 0.005)
        intrinsics = CameraIntrinsics(
            500.0,
            510.0,
            320.0,
            240.0,
            640,
            480,
            coefficients,
            "inverse_brown_conrady",
        )
        pixel = (420.0, 180.0)
        x = (pixel[0] - intrinsics.cx) / intrinsics.fx
        y = (pixel[1] - intrinsics.cy) / intrinsics.fy
        radius2 = x * x + y * y
        radial = 1.0 + coefficients[0] * radius2 + coefficients[1] * radius2 ** 2 + coefficients[4] * radius2 ** 3
        expected_x = x * radial + 2.0 * coefficients[2] * x * y + coefficients[3] * (
            radius2 + 2.0 * x * x
        )
        expected_y = y * radial + 2.0 * coefficients[3] * x * y + coefficients[2] * (
            radius2 + 2.0 * y * y
        )
        np.testing.assert_allclose(
            deproject_pixel(pixel, 0.75, intrinsics),
            [expected_x * 0.75, expected_y * 0.75, 0.75],
            atol=1e-12,
        )

    def test_average_transforms(self):
        first, second = np.eye(4), np.eye(4)
        first[:3, 3] = [0.0, 1.0, 2.0]
        second[:3, 3] = [2.0, 3.0, 4.0]
        result = average_transforms([first, second])
        np.testing.assert_allclose(result[:3, 3], [1.0, 2.0, 3.0])
        np.testing.assert_allclose(result[:3, :3], np.eye(3))

    def test_robust_depth_ignores_holes(self):
        depth = np.zeros((7, 7), dtype=np.float32)
        depth[2:5, 2:5] = 0.7
        depth[3, 3] = np.nan
        self.assertAlmostEqual(robust_depth(depth, 3, 3, radius=1), 0.7, places=6)

    def test_eye_on_base_recovery_from_synthetic_qr(self):
        try:
            import cv2
        except ImportError:
            self.skipTest("opencv is not installed")
        size = 0.08
        half = size / 2.0
        qr_points = np.array(
            [[-half, half, 0.0], [half, half, 0.0], [half, -half, 0.0], [-half, -half, 0.0]]
        )
        camera_matrix = np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]])
        rotation_vector = np.array([[0.1], [-0.2], [0.05]])
        translation = np.array([[0.02], [-0.01], [0.7]])
        pixels, _ = cv2.projectPoints(qr_points, rotation_vector, translation, camera_matrix, np.zeros(5))
        camera_from_qr = make_transform(cv2.Rodrigues(rotation_vector)[0], translation.reshape(3))
        base_from_qr = np.eye(4)
        base_from_qr[:3, 3] = [0.5, 0.1, 0.2]
        expected = base_from_qr.dot(invert_transform(camera_from_qr))
        actual, rms = estimate_eye_on_base(pixels, camera_matrix, np.zeros(5), size, base_from_qr)
        np.testing.assert_allclose(actual, expected, atol=1e-8)
        self.assertLess(rms, 1e-8)

    def test_ur_tcp_axis_angle_conversion(self):
        pose = np.array([0.4, -0.2, 0.3, 0.0, 0.0, np.pi / 2.0])
        transform = tcp_pose_to_transform(pose)
        np.testing.assert_allclose(transform[:3, 3], pose[:3])
        np.testing.assert_allclose(
            transform[:3, :3],
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            atol=1e-8,
        )

    def test_eye_on_hand_recovery_and_file_contract(self):
        try:
            import cv2
        except ImportError:
            self.skipTest("opencv is not installed")

        def transform(rotation_vector, translation):
            return make_transform(cv2.Rodrigues(np.asarray(rotation_vector, dtype=np.float64))[0], translation)

        expected = transform([0.12, -0.08, 0.05], [0.04, -0.03, 0.11])
        base_to_board = transform([0.05, 0.02, -0.03], [0.62, 0.08, 0.14])
        base_to_tcp = []
        for index in range(12):
            base_to_tcp.append(transform(
                [
                    0.15 * np.sin(index),
                    0.18 * np.cos(index * 0.7),
                    0.12 * np.sin(index * 0.4),
                ],
                [
                    0.35 + 0.04 * np.sin(index * 0.8),
                    -0.2 + 0.05 * np.cos(index * 0.5),
                    0.3 + 0.03 * np.sin(index * 0.3),
                ],
            ))
        camera_to_board = [
            invert_transform(base_pose.dot(expected)).dot(base_to_board)
            for base_pose in base_to_tcp
        ]
        actual = calibrate_eye_on_hand(base_to_tcp, camera_to_board, "park")
        np.testing.assert_allclose(actual, expected, atol=1e-7)
        _, translation_error, rotation_error = eye_on_hand_residuals(
            base_to_tcp, camera_to_board, actual
        )
        self.assertLess(float(translation_error.max()), 1e-8)
        self.assertLess(float(rotation_error.max()), 1e-4)
        translation_span, rotation_span = transform_spans(base_to_tcp)
        self.assertGreater(translation_span, 0.05)
        self.assertGreater(rotation_span, 10.0)

        refined, inliers, _, _, _ = robust_calibrate_eye_on_hand(
            base_to_tcp, camera_to_board, minimum_inliers=10
        )
        self.assertTrue(inliers.all())
        np.testing.assert_allclose(refined, expected, atol=1e-7)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "eye_on_hand.yaml"
            board = {
                "model": "DFVision Q12-240-15",
                "columns": 11,
                "rows": 8,
                "square_size_m": 0.015,
            }
            save_eye_on_hand(
                path,
                actual,
                {
                    "sample_count": 12,
                    "checkerboard": board,
                    "camera": {"model": "Intel RealSense D435I"},
                },
            )
            np.testing.assert_allclose(load_eye_on_hand(path), expected, atol=1e-7)
            np.testing.assert_allclose(
                load_eye_on_hand(path, board, "D435i"), expected, atol=1e-7
            )
            with self.assertRaisesRegex(ValueError, "square_size_m"):
                load_eye_on_hand(path, dict(board, square_size_m=0.025), "D435i")

    def test_eye_on_base_robot_world_recovery_and_file_contract(self):
        try:
            import cv2
        except ImportError:
            self.skipTest("opencv is not installed")

        def transform(rotation_vector, translation):
            rotation = cv2.Rodrigues(np.asarray(rotation_vector, dtype=np.float64))[0]
            return make_transform(rotation, translation)

        expected_camera = transform([0.2, -0.1, 0.08], [0.75, -0.3, 0.9])
        expected_board = transform([-0.1, 0.12, 0.04], [0.02, 0.01, 0.18])
        base_to_tcp = [
            transform(
                [0.25 * np.sin(index), 0.2 * np.cos(index * 0.6), 0.18 * np.sin(index * 0.4)],
                [
                    0.4 + 0.08 * np.sin(index * 0.7),
                    -0.1 + 0.07 * np.cos(index * 0.5),
                    0.35 + 0.05 * np.sin(index * 0.3),
                ],
            )
            for index in range(16)
        ]
        camera_to_board = [
            invert_transform(expected_camera).dot(pose).dot(expected_board)
            for pose in base_to_tcp
        ]
        actual_camera, actual_board = calibrate_eye_on_base(
            base_to_tcp, camera_to_board, "shah"
        )
        np.testing.assert_allclose(actual_camera, expected_camera, atol=1e-6)
        np.testing.assert_allclose(actual_board, expected_board, atol=1e-6)
        translation_error, rotation_error = eye_on_base_residuals(
            base_to_tcp, camera_to_board, actual_camera, actual_board
        )
        self.assertLess(float(translation_error.max()), 1e-6)
        self.assertLess(float(rotation_error.max()), 1e-4)
        refined = robust_calibrate_eye_on_base(
            base_to_tcp, camera_to_board, minimum_inliers=12
        )
        np.testing.assert_allclose(refined[0], expected_camera, atol=1e-6)
        self.assertTrue(refined[2].all())
        corrupted = [item.copy() for item in camera_to_board]
        corrupted[-1][:3, 3] += [0.08, -0.05, 0.04]
        robust = robust_calibrate_eye_on_base(
            base_to_tcp, corrupted, minimum_inliers=12
        )
        self.assertFalse(bool(robust[2][-1]))
        np.testing.assert_allclose(robust[0], expected_camera, atol=1e-5)

        board = {
            "model": "DFVision Q12-240-15",
            "columns": 11,
            "rows": 8,
            "square_size_m": 0.015,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "eye_on_base.yaml"
            save_eye_on_base(
                path,
                actual_camera,
                {
                    "checkerboard": board,
                    "camera": {"model": "Intel RealSense D435I", "serial": "fixed-123"},
                },
            )
            np.testing.assert_allclose(
                load_eye_on_base(path, board, "D435i", "fixed-123"),
                expected_camera,
                atol=1e-6,
            )
            with self.assertRaisesRegex(ValueError, "serial mismatch"):
                load_eye_on_base(path, board, "D435i", "different-camera")

    def test_checkerboard_detection_and_pose(self):
        try:
            import cv2
        except ImportError:
            self.skipTest("opencv is not installed")
        image = np.full((480, 640, 3), 255, dtype=np.uint8)
        origin_x, origin_y, square_pixels = 140, 100, 30
        for row in range(9):
            for column in range(12):
                if (row + column) % 2 == 0:
                    top_left = (
                        origin_x + column * square_pixels,
                        origin_y + row * square_pixels,
                    )
                    bottom_right = (
                        top_left[0] + square_pixels,
                        top_left[1] + square_pixels,
                    )
                    cv2.rectangle(image, top_left, bottom_right, (0, 0, 0), -1)
        intrinsics = CameraIntrinsics(600.0, 600.0, 320.0, 240.0, 640, 480)
        camera_to_board, rms, corners = estimate_checkerboard_pose(
            image, intrinsics, 11, 8, 0.015
        )
        self.assertEqual(corners.shape, (88, 2))
        self.assertLess(rms, 0.2)
        self.assertGreater(camera_to_board[2, 3], 0.0)


class TestPinnedUltralyticsCompatibility(unittest.TestCase):
    def test_locator_filters_multiple_classes_and_uses_model_names(self):
        import torch

        class FakeBoxes:
            cls = torch.tensor([1.0, 0.0])
            conf = torch.tensor([0.99, 0.9])
            xyxy = torch.tensor([[0.0, 0.0, 2.0, 2.0], [1.0, 1.0, 3.0, 3.0]])

            def __len__(self):
                return 2

        class FakeYOLO:
            names = {0: "object", 1: "distractor"}

            def __init__(self, weights):
                self.weights = weights

            def predict(self, **kwargs):
                # Ultralytics 8.0.20 Results has boxes but no names attribute.
                return [types.SimpleNamespace(boxes=FakeBoxes())]

        tcp_to_camera = np.eye(4)
        tcp_to_camera[:3, 3] = [0.1, 0.0, 0.0]
        with mock.patch(
            "ur5e_comm.vision.load_yolo_model", return_value=FakeYOLO("weights.pt")
        ):
            locator = YoloInitialObjectLocator(
                "weights.pt", tcp_to_camera, target_class="object", depth_radius=0,
                center_depth_offset_m=0.01,
            )
        frame = RGBDFrame(
            color_bgr=np.zeros((5, 5, 3), dtype=np.uint8),
            depth_m=np.ones((5, 5), dtype=np.float32),
            intrinsics=CameraIntrinsics(1.0, 1.0, 2.0, 2.0, 5, 5),
            timestamp=1.0,
        )
        base_to_tcp = np.eye(4)
        base_to_tcp[:3, 3] = [0.2, 0.0, 0.0]
        detection = locator.locate(frame, base_to_tcp)
        self.assertEqual(detection.class_name, "object")
        self.assertEqual(detection.class_id, 0)
        self.assertAlmostEqual(detection.confidence, 0.9, places=6)
        np.testing.assert_allclose(detection.center_pixel, [2.0, 2.0])
        self.assertAlmostEqual(detection.surface_depth_m, 1.0)
        self.assertAlmostEqual(detection.depth_m, 1.01)
        np.testing.assert_allclose(detection.position_camera, [0.0, 0.0, 1.01])
        np.testing.assert_allclose(detection.position_base, [0.3, 0.0, 1.01])
        np.testing.assert_allclose(detection.base_to_camera[:3, 3], [0.3, 0.0, 0.0])

    def test_depth_offset_follows_original_camera_ray(self):
        import torch

        class FakeBoxes:
            cls = torch.tensor([0.0])
            conf = torch.tensor([0.9])
            xyxy = torch.tensor([[3.0, 1.0, 3.0, 1.0]])

            def __len__(self):
                return 1

        model = mock.Mock()
        model.names = {0: "object"}
        model.predict.return_value = [types.SimpleNamespace(boxes=FakeBoxes())]
        with mock.patch("ur5e_comm.vision.load_yolo_model", return_value=model):
            locator = YoloInitialObjectLocator(
                "weights.pt", np.eye(4), depth_radius=0, center_depth_offset_m=0.01
            )
        frame = RGBDFrame(
            np.zeros((5, 5, 3), dtype=np.uint8),
            np.ones((5, 5), dtype=np.float32),
            CameraIntrinsics(1.0, 1.0, 2.0, 2.0, 5, 5),
            1.0,
        )
        detection = locator.locate(frame, np.eye(4))
        np.testing.assert_allclose(detection.position_camera, [1.01, -1.01, 1.01])

    def test_eye_on_base_locator_associates_nearest_3d_target(self):
        import torch

        class FakeBoxes:
            cls = torch.tensor([0.0, 0.0])
            conf = torch.tensor([0.95, 0.80])
            xyxy = torch.tensor([[1.0, 2.0, 1.0, 2.0], [3.0, 2.0, 3.0, 2.0]])

            def __len__(self):
                return 2

        model = mock.Mock()
        model.names = {0: "object"}
        model.predict.return_value = [types.SimpleNamespace(boxes=FakeBoxes())]
        detector = YoloInitialObjectLocator(
            "weights.pt", np.eye(4), target_class="object", depth_radius=0,
            center_depth_offset_m=0.0, model=model,
        )
        locator = YoloEyeOnBaseLocator(np.eye(4), detector)
        frame = RGBDFrame(
            np.zeros((5, 5, 3), dtype=np.uint8),
            np.ones((5, 5), dtype=np.float32),
            CameraIntrinsics(1.0, 1.0, 2.0, 2.0, 5, 5),
            1.0,
        )
        detection = locator.locate(frame, [0.9, 0.0, 1.0], 0.2)
        np.testing.assert_allclose(detection.position_base, [1.0, 0.0, 1.0])
        self.assertAlmostEqual(detection.confidence, 0.80, places=6)


class TestPolicyContract(unittest.TestCase):
    def test_mujoco_projector_reconstructs_all_robot_dofs(self):
        try:
            import mujoco  # noqa: F401
        except ImportError:
            self.skipTest("mujoco is not installed")
        model = Path(__file__).resolve().parents[1] / (
            "resources/assets/robots/ur5e_robotiq_2f85/ur5e_robotiq_2f85.xml"
        )
        projector = MujocoStateProjector(model)
        state = projector.project(snapshot())
        self.assertEqual(state.joint_position.shape, (12,))
        self.assertEqual(state.joint_velocity.shape, (12,))
        self.assertEqual(state.palm_state.shape, (10,))
        self.assertEqual(state.fingertip_position.shape, (2, 3))
        self.assertTrue(np.isfinite(state.palm_center_position).all())
        np.testing.assert_allclose(state.joint_position[:6], snapshot().joint_position)

    def test_observation_layout_and_live_object_state(self):
        builder = Ur5eObservationBuilder(FakeProjector(), [0.8, 0.3, 0.05])
        builder.begin_episode([0.6, 0.0, 0.2])
        observation, _ = builder.build(snapshot())
        self.assertEqual(observation.shape, (69,))
        expected_q = FakeProjector().project(snapshot()).joint_position
        expected_scaled = 2.0 * (expected_q - TRAINING_LOWER) / (TRAINING_UPPER - TRAINING_LOWER) - 1.0
        np.testing.assert_allclose(observation[:12], expected_scaled, atol=1e-6)
        np.testing.assert_allclose(observation[41:47], 0.0)  # object lin/ang velocity
        np.testing.assert_allclose(observation[53:56], [0.1, 0.0, -0.2])
        np.testing.assert_allclose(observation[56:59], [-0.2, -0.3, 0.15])
        self.assertEqual(float(observation[65]), 0.0)  # no inferred lift after initial lock
        self.assertAlmostEqual(float(observation[68]), 0.01, places=6)
        live = TrackedObjectState(
            np.array([0.6, 0.0, 0.32]),
            np.array([0.1, -0.2, 0.3]),
            1.0,
            1.0,
            0.9,
            False,
            np.array([2.0, 2.0]),
            0.5,
            0.51,
        )
        live_observation, _ = builder.build(snapshot(), live)
        np.testing.assert_allclose(live_observation[41:44], live.linear_velocity_base)
        np.testing.assert_allclose(live_observation[44:47], 0.0)
        np.testing.assert_allclose(live_observation[53:56], [0.1, 0.0, -0.08])
        self.assertEqual(float(live_observation[65]), 0.0)
        lifted_observation, _ = builder.build(snapshot(), live)
        self.assertEqual(float(lifted_observation[65]), 1.0)

    def test_action_mapping(self):
        state = snapshot()
        mapper = Ur5eActionMapper(max_arm_step=0.05, arm_lower=[-3] * 6, arm_upper=[3] * 6)
        mapper.reset(state)
        target, gripper = mapper.map(np.ones(7))
        np.testing.assert_allclose(target, state.joint_position + 0.05)
        self.assertEqual(gripper, 255)
        target, gripper = mapper.map(-np.ones(7))
        np.testing.assert_allclose(target, state.joint_position)
        self.assertEqual(gripper, 0)

    def test_smoke_action_scale_and_grasp_goal(self):
        state = snapshot()
        mapper = Ur5eActionMapper(
            period=0.01667,
            speed_scale=10.0,
            max_arm_step=0.008,
            arm_lower=[-3] * 6,
            arm_upper=[3] * 6,
            action_scale=0.05,
        )
        mapper.reset(state)
        raw = np.array([0.5] * 6 + [1.0])
        target, gripper = mapper.map(raw)
        np.testing.assert_allclose(target - state.joint_position, 10.0 * 0.01667 * 0.05 * 0.5)
        np.testing.assert_allclose(mapper.last_limited_action[:6], 0.025)
        self.assertEqual(mapper.last_limited_action[6], 1.0)
        self.assertEqual(gripper, 255)

        builder = Ur5eObservationBuilder(
            FakeProjector(),
            [0.8, 0.3, 0.05],
            subtask="grasp",
            goal_offset=[0.0, 0.0, 0.12],
        )
        builder.begin_episode([0.6, 0.0, 0.2])
        observation, _ = builder.build(state)
        np.testing.assert_allclose(builder.goal_position, [0.6, 0.0, 0.32])
        np.testing.assert_allclose(observation[56:59], [0.0, 0.0, -0.12])


class TestEyeOnBaseTracker(unittest.TestCase):
    @staticmethod
    def detection(position, timestamp):
        position = np.asarray(position, dtype=np.float64)
        return ObjectDetection(
            position.copy(),
            position.copy(),
            np.array([10.0, 20.0]),
            0.51,
            0.9,
            0,
            "object",
            timestamp,
            surface_depth_m=0.50,
        )

    def test_filter_prediction_and_stale_failure(self):
        tracker = EyeOnBaseObjectTracker(None, None)
        tracker._detection = self.detection([0.0, 0.0, 0.0], 1.0)
        tracker._position = np.zeros(3)
        tracker._accept(self.detection([1.0, 0.0, 0.0], 1.1))
        np.testing.assert_allclose(tracker._position, [0.6, 0.0, 0.0])
        np.testing.assert_allclose(tracker._velocity, [2.4, 0.0, 0.0])
        measured = tracker.state(1.15)
        np.testing.assert_allclose(measured.position_base, [0.72, 0.0, 0.0])
        self.assertFalse(measured.predicted)
        predicted = tracker.state(1.16)
        np.testing.assert_allclose(predicted.position_base, [0.744, 0.0, 0.0])
        self.assertTrue(predicted.predicted)
        with self.assertRaisesRegex(RuntimeError, "stale"):
            tracker.state(1.26)


class FakeCamera:
    def read(self):
        return RGBDFrame(
            color_bgr=np.zeros((5, 5, 3), dtype=np.uint8),
            depth_m=np.ones((5, 5), dtype=np.float32),
            intrinsics=CameraIntrinsics(1.0, 1.0, 2.0, 2.0, 5, 5),
            timestamp=monotonic(),
        )


class FakeLocator:
    def locate(self, frame, base_to_tcp):
        return ObjectDetection(
            np.array([0.0, 0.0, 0.5]),
            np.array([0.6, 0.0, 0.2]),
            np.array([320.0, 240.0]),
            0.5,
            0.9,
            0,
            "object",
            frame.timestamp,
            surface_depth_m=0.49,
        )

    def locate_from_camera(self, camera, attempts, reference_position, max_distance):
        return self.locate(camera.read(), np.eye(4))


class FakeTracker:
    def __init__(self, camera, locator, *args):
        self.detection = None
        self.stopped = False

    def start(self, detection):
        self.detection = detection

    def state(self):
        now = monotonic()
        return TrackedObjectState(
            self.detection.position_base.copy(),
            np.zeros(3),
            now,
            now,
            self.detection.confidence,
            False,
            self.detection.center_pixel.copy(),
            self.detection.surface_depth_m,
            self.detection.depth_m,
        )

    def stop(self):
        self.stopped = True


class FakeRobot:
    def __init__(self):
        self.commands = []
        self.starts = 0
        self.stops = 0
        self.homes = []
        self.events = []

    def read(self):
        self.events.append("read")
        return snapshot()

    def home(self, joint_position, speed, acceleration, tolerance):
        self.events.append("home")
        self.homes.append((np.asarray(joint_position), speed, acceleration, tolerance))

    def command(self, arm_target, gripper):
        self.commands.append((arm_target, gripper))

    def start_motion(self):
        self.starts += 1

    def stop_motion(self):
        self.stops += 1


class FakePolicy:
    observation_dim = 69
    action_dim = 7

    def reset(self):
        pass

    def infer(self, observation):
        return np.zeros((1, 7), dtype=np.float32)


class TestDeploymentSession(unittest.TestCase):
    def test_read_only_episode_records_human_verdict(self):
        projector = FakeProjector()
        builder = Ur5eObservationBuilder(projector, [0.8, 0.3, 0.05])
        mapper = Ur5eActionMapper(max_arm_step=0.05, arm_lower=[-3] * 6, arm_upper=[3] * 6)
        safety = SafetyMonitor(
            [-3] * 6,
            [3] * 6,
            workspace_min=[0.0, -1.0, 0.0],
            workspace_max=[1.0, 1.0, 1.0],
        )
        robot = FakeRobot()
        with tempfile.TemporaryDirectory() as directory:
            session = DeploymentSession(
                FakeCamera(), FakeLocator(), robot, FakePolicy(), builder, mapper, safety,
                directory, execute=False, policy_period=0.00001, verdict_provider=lambda: "y",
                tracking_camera=FakeCamera(), tracking_locator=FakeLocator(),
                tracker_factory=FakeTracker,
            )
            result = session.run_episode(steps=3)
            self.assertEqual(result.verdict, "y")
            self.assertEqual(result.steps, 3)
            self.assertTrue(result.trajectory_path.is_file())
            self.assertEqual(robot.commands, [])
            payload = np.load(str(result.trajectory_path))
            self.assertEqual(payload["observation"].shape, (3, 69))
            self.assertEqual(str(payload["verdict"]), "y")
            self.assertTrue((Path(directory) / "episodes.jsonl").is_file())
            self.assertEqual(robot.homes, [])

    def test_execute_episode_starts_and_stops_servo_lifecycle(self):
        projector = FakeProjector()
        builder = Ur5eObservationBuilder(projector, [0.8, 0.3, 0.05])
        mapper = Ur5eActionMapper(max_arm_step=0.05, arm_lower=[-3] * 6, arm_upper=[3] * 6)
        safety = SafetyMonitor(
            [-3] * 6,
            [3] * 6,
            workspace_min=[0.0, -1.0, 0.0],
            workspace_max=[1.0, 1.0, 1.0],
        )
        robot = FakeRobot()
        events = robot.events

        class OrderedCamera(FakeCamera):
            def read(self):
                events.append("detect")
                return super().read()

        with tempfile.TemporaryDirectory() as directory:
            session = DeploymentSession(
                OrderedCamera(), FakeLocator(), robot, FakePolicy(), builder, mapper, safety,
                directory, execute=True, policy_period=0.00001, max_policy_lag=1.0,
                verdict_provider=lambda: "x",
                home_confirmation_provider=lambda: events.append("confirm"),
                tracking_camera=FakeCamera(), tracking_locator=FakeLocator(),
                tracker_factory=FakeTracker,
            )
            session.run_episode(steps=2)
        self.assertEqual(robot.starts, 1)
        self.assertEqual(robot.stops, 1)
        self.assertEqual(len(robot.commands), 2)
        self.assertEqual(len(robot.homes), 1)
        self.assertEqual(robot.events[:4], ["confirm", "home", "read", "detect"])

    def test_dry_run_rejects_manual_state_without_writing(self):
        robot = FakeRobot()
        robot.read = lambda: RobotSnapshot(
            joint_position=np.zeros(6),
            joint_velocity=np.zeros(6),
            tcp_pose=np.zeros(6),
            tcp_speed=np.zeros(6),
            gripper_position=10,
            timestamp=monotonic(),
        )
        builder = Ur5eObservationBuilder(FakeProjector(), [0.8, 0.3, 0.05])
        mapper = Ur5eActionMapper(max_arm_step=0.05, arm_lower=[-3] * 6, arm_upper=[3] * 6)
        safety = SafetyMonitor([-3] * 6, [3] * 6)
        with tempfile.TemporaryDirectory() as directory:
            session = DeploymentSession(
                FakeCamera(), FakeLocator(), robot, FakePolicy(), builder, mapper, safety,
                directory, execute=False, verdict_provider=lambda: "x",
            )
            with self.assertRaisesRegex(RuntimeError, "manual home"):
                session.run_episode(steps=1)
        self.assertEqual(robot.homes, [])
        self.assertEqual(robot.commands, [])

    def test_cross_camera_mismatch_never_starts_servo(self):
        class MismatchLocator(FakeLocator):
            def locate_from_camera(self, camera, attempts, reference_position, max_distance):
                detection = super().locate_from_camera(
                    camera, attempts, reference_position, max_distance
                )
                return ObjectDetection(
                    detection.position_camera,
                    detection.position_base + np.array([0.10, 0.0, 0.0]),
                    detection.center_pixel,
                    detection.depth_m,
                    detection.confidence,
                    detection.class_id,
                    detection.class_name,
                    detection.timestamp,
                    detection.surface_depth_m,
                    detection.base_to_camera,
                )

        robot = FakeRobot()
        builder = Ur5eObservationBuilder(FakeProjector(), [0.8, 0.3, 0.05])
        mapper = Ur5eActionMapper(max_arm_step=0.05, arm_lower=[-3] * 6, arm_upper=[3] * 6)
        safety = SafetyMonitor([-3] * 6, [3] * 6)
        with tempfile.TemporaryDirectory() as directory:
            session = DeploymentSession(
                FakeCamera(), FakeLocator(), robot, FakePolicy(), builder, mapper, safety,
                directory, execute=True, home_confirmation_provider=lambda: None,
                tracking_camera=FakeCamera(), tracking_locator=MismatchLocator(),
                tracker_factory=FakeTracker,
            )
            with self.assertRaisesRegex(RuntimeError, "differ"):
                session.run_episode(steps=1)
        self.assertEqual(robot.starts, 0)
        self.assertEqual(robot.commands, [])

    def test_stale_tracking_stops_before_policy_command(self):
        class StaleTracker(FakeTracker):
            def state(self):
                raise RuntimeError("Eye-on-base object state is stale by 0.200s")

        robot = FakeRobot()
        builder = Ur5eObservationBuilder(FakeProjector(), [0.8, 0.3, 0.05])
        mapper = Ur5eActionMapper(max_arm_step=0.05, arm_lower=[-3] * 6, arm_upper=[3] * 6)
        safety = SafetyMonitor([-3] * 6, [3] * 6)
        with tempfile.TemporaryDirectory() as directory:
            session = DeploymentSession(
                FakeCamera(), FakeLocator(), robot, FakePolicy(), builder, mapper, safety,
                directory, execute=True, home_confirmation_provider=lambda: None,
                tracking_camera=FakeCamera(), tracking_locator=FakeLocator(),
                tracker_factory=StaleTracker,
            )
            with self.assertRaisesRegex(RuntimeError, "stale"):
                session.run_episode(steps=1)
        self.assertEqual(robot.starts, 0)
        self.assertEqual(robot.stops, 0)
        self.assertEqual(robot.commands, [])

    def test_policy_timeout_stops_before_next_command(self):
        class SlowPolicy(FakePolicy):
            def infer(self, observation):
                sleep(0.01)
                return super().infer(observation)

        robot = FakeRobot()
        builder = Ur5eObservationBuilder(FakeProjector(), [0.8, 0.3, 0.05])
        mapper = Ur5eActionMapper(max_arm_step=0.05, arm_lower=[-3] * 6, arm_upper=[3] * 6)
        safety = SafetyMonitor([-3] * 6, [3] * 6)
        with tempfile.TemporaryDirectory() as directory:
            session = DeploymentSession(
                FakeCamera(), FakeLocator(), robot, SlowPolicy(), builder, mapper, safety,
                directory, execute=True, policy_period=0.00001, max_policy_lag=0.001,
                home_confirmation_provider=lambda: None,
                tracking_camera=FakeCamera(), tracking_locator=FakeLocator(),
                tracker_factory=FakeTracker,
            )
            with self.assertRaisesRegex(RuntimeError, "missed its schedule"):
                session.run_episode(steps=1)
        self.assertEqual(robot.starts, 1)
        self.assertEqual(robot.stops, 1)
        self.assertEqual(robot.commands, [])


class TestHardwareReadOnlyMode(unittest.TestCase):
    def test_home_opens_gripper_then_moves_and_checks_final_state(self):
        events = []
        hardware = object.__new__(UR5eHardware)
        hardware.allow_motion = True
        hardware.gripper_speed = 255
        hardware.gripper_force = 100
        hardware.gripper = mock.Mock()
        hardware.gripper.move.side_effect = lambda *args: events.append("gripper")
        hardware.gripper.get.side_effect = [20, 5]
        hardware.control = mock.Mock()
        hardware.control.moveJ.side_effect = lambda *args: events.append("moveJ") or True
        hardware.receive = mock.Mock()
        hardware.receive.getActualQd.return_value = [0.0] * 6
        target = [-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0]
        hardware.receive.getActualQ.return_value = target
        with mock.patch("ur5e_comm.robot.sleep"):
            actual = hardware.home(target, 0.25, 0.5, 0.02)
        self.assertEqual(events, ["gripper", "moveJ"])
        hardware.control.moveJ.assert_called_once_with(target, 0.25, 0.5, False)
        np.testing.assert_allclose(actual, target)

    def test_home_stops_on_movej_failure_or_joint_error(self):
        hardware = object.__new__(UR5eHardware)
        hardware.allow_motion = True
        hardware.gripper_speed = 255
        hardware.gripper_force = 100
        hardware.gripper = mock.Mock()
        hardware.gripper.get.return_value = 0
        hardware.control = mock.Mock()
        hardware.receive = mock.Mock()
        hardware.receive.getActualQd.return_value = [0.0] * 6
        target = np.zeros(6)
        hardware.control.moveJ.return_value = False
        with self.assertRaisesRegex(RuntimeError, "moveJ rejected"):
            hardware.home(target)
        hardware.control.moveJ.return_value = True
        hardware.receive.getActualQ.return_value = [0.03] * 6
        with self.assertRaisesRegex(RuntimeError, "exceeds"):
            hardware.home(target, tolerance=0.02)

    def test_read_only_mode_never_constructs_control_or_writes_gripper(self):
        class FakeReceive:
            def __init__(self, host, rt_priority=0):
                self.host = host
                self.rt_priority = rt_priority

            def disconnect(self):
                pass

        receive_module = types.SimpleNamespace(RTDEReceiveInterface=FakeReceive)
        fake_gripper = mock.Mock()
        with mock.patch.dict("sys.modules", {"rtde_receive": receive_module}):
            with mock.patch("ur5e_comm.robot.RobotiqSocket", return_value=fake_gripper):
                hardware = UR5eHardware("127.0.0.1", allow_motion=False, activate_gripper=True)
                self.assertIsNone(hardware.control)
                self.assertEqual(hardware.receive.rt_priority, 90)
                hardware.stop_motion()
                hardware.close()
        fake_gripper.activate.assert_not_called()
        fake_gripper.stop.assert_not_called()
        fake_gripper.set.assert_not_called()
        fake_gripper.close.assert_called_once()

    def test_rtde_and_servo_threads_use_configured_fifo_priorities(self):
        class FakeReceive:
            def __init__(self, host, rt_priority=0):
                self.rt_priority = rt_priority

            def disconnect(self):
                pass

        class FakeControl:
            def __init__(self, host, rt_priority=0):
                self.rt_priority = rt_priority

            def stopScript(self):
                pass

        modules = {
            "rtde_receive": types.SimpleNamespace(RTDEReceiveInterface=FakeReceive),
            "rtde_control": types.SimpleNamespace(RTDEControlInterface=FakeControl),
        }
        fake_gripper = mock.Mock()
        with mock.patch.dict("sys.modules", modules):
            with mock.patch("ur5e_comm.robot.RobotiqSocket", return_value=fake_gripper):
                hardware = UR5eHardware(
                    "127.0.0.1", allow_motion=True, activate_gripper=False
                )
                self.assertEqual(hardware.receive.rt_priority, 90)
                self.assertEqual(hardware.control.rt_priority, 85)
                self.assertEqual(hardware.servo_thread_priority, 80)
                hardware.close()

        hardware = object.__new__(UR5eHardware)
        hardware.servo_thread_priority = 80
        hardware._servo_stop_event = threading.Event()
        hardware._target_lock = threading.Lock()
        hardware._arm_target = np.zeros(6)
        hardware.servo_speed = 0.5
        hardware.servo_acceleration = 1.0
        hardware.servo_period = 0.002
        hardware.servo_lookahead = 0.1
        hardware.servo_gain = 300
        hardware._servo_error = None
        hardware.control = mock.Mock()
        hardware.control.initPeriod.return_value = object()
        hardware.control.servoJ.return_value = True
        hardware.control.waitPeriod.side_effect = lambda started: hardware._servo_stop_event.set()
        with mock.patch("ur5e_comm.robot.os.sched_setscheduler") as scheduler:
            hardware._servo_loop()
        scheduler.assert_called_once_with(0, os.SCHED_FIFO, os.sched_param(80))
        self.assertIsNone(hardware._servo_error)


if __name__ == "__main__":
    unittest.main()
    eye_on_base_residuals,
    robust_calibrate_eye_on_base,
