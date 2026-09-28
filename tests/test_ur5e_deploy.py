import tempfile
import unittest
import types
from unittest import mock
from pathlib import Path
from time import monotonic

import numpy as np

from ur5e_comm.deployment import DeploymentSession
from ur5e_comm.geometry import (
    CameraIntrinsics,
    average_transforms,
    calibrate_eye_on_hand,
    deproject_pixel,
    eye_on_hand_residuals,
    estimate_checkerboard_pose,
    estimate_eye_on_base,
    invert_transform,
    load_eye_on_hand,
    make_transform,
    robust_calibrate_eye_on_hand,
    save_eye_on_hand,
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
from ur5e_comm.vision import ObjectDetection, robust_depth
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
    def test_locator_uses_model_names_when_old_results_have_no_names(self):
        import torch

        class FakeBoxes:
            cls = torch.tensor([0.0])
            conf = torch.tensor([0.9])
            xyxy = torch.tensor([[1.0, 1.0, 3.0, 3.0]])

            def __len__(self):
                return 1

        class FakeYOLO:
            names = {0: "object"}

            def __init__(self, weights):
                self.weights = weights

            def predict(self, **kwargs):
                # Ultralytics 8.0.20 Results has boxes but no names attribute.
                return [types.SimpleNamespace(boxes=FakeBoxes())]

        module = types.SimpleNamespace(YOLO=FakeYOLO)
        tcp_to_camera = np.eye(4)
        tcp_to_camera[:3, 3] = [0.1, 0.0, 0.0]
        with mock.patch.dict("sys.modules", {"ultralytics": module}):
            locator = YoloInitialObjectLocator(
                "weights.pt", tcp_to_camera, target_class="object", depth_radius=0
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
        np.testing.assert_allclose(detection.center_pixel, [2.0, 2.0])
        np.testing.assert_allclose(detection.position_base, [0.3, 0.0, 1.0])
        np.testing.assert_allclose(detection.base_to_camera[:3, 3], [0.3, 0.0, 0.0])


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

    def test_observation_layout_and_frozen_object_state(self):
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
        self.assertEqual(float(observation[68]), 0.0)  # no online reward

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
        )


class FakeRobot:
    def __init__(self):
        self.commands = []
        self.starts = 0
        self.stops = 0

    def read(self):
        return snapshot()

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
        with tempfile.TemporaryDirectory() as directory:
            session = DeploymentSession(
                FakeCamera(), FakeLocator(), robot, FakePolicy(), builder, mapper, safety,
                directory, execute=True, policy_period=0.00001, max_policy_lag=1.0,
                verdict_provider=lambda: "x",
            )
            session.run_episode(steps=2)
        self.assertEqual(robot.starts, 1)
        self.assertEqual(robot.stops, 1)
        self.assertEqual(len(robot.commands), 2)


class TestHardwareReadOnlyMode(unittest.TestCase):
    def test_read_only_mode_never_constructs_control_or_writes_gripper(self):
        class FakeReceive:
            def __init__(self, host):
                self.host = host

            def disconnect(self):
                pass

        receive_module = types.SimpleNamespace(RTDEReceiveInterface=FakeReceive)
        fake_gripper = mock.Mock()
        with mock.patch.dict("sys.modules", {"rtde_receive": receive_module}):
            with mock.patch("ur5e_comm.robot.RobotiqSocket", return_value=fake_gripper):
                hardware = UR5eHardware("127.0.0.1", allow_motion=False, activate_gripper=True)
                self.assertIsNone(hardware.control)
                hardware.stop_motion()
                hardware.close()
        fake_gripper.activate.assert_not_called()
        fake_gripper.stop.assert_not_called()
        fake_gripper.set.assert_not_called()
        fake_gripper.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
