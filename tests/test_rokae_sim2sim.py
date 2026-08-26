import tempfile
import unittest
from pathlib import Path

import mujoco
import numpy as np

from mujoco_sim.episode import EpisodeRunner
from mujoco_sim.rokae_adapter import RokaeAdapter, wxyz_to_xyzw
from onnx_deploy.policy_runner import PolicyRunner


ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "resources/assets/scenes/allegro_rokae.xml"
MODEL = ROOT / "resources/models/rokae_allegro/policy.onnx"
META = ROOT / "resources/models/rokae_allegro/policy.meta.yaml"


class TestRokaeModel(unittest.TestCase):
    def setUp(self):
        self.adapter = RokaeAdapter(SCENE, META, seed=7)

    def test_model_joint_and_actuator_contract(self):
        model = self.adapter.model
        robot_joints = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index)
            for index in range(model.njnt - 1)
        ]
        actuators = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, index)
            for index in range(model.nu)
        ]
        self.assertEqual(robot_joints, self.adapter.joint_names)
        self.assertEqual(actuators, [f"{name}_position" for name in self.adapter.joint_names])
        self.assertEqual(model.nu, 23)

    def test_observation_contract(self):
        observation = self.adapter.reset()
        self.assertEqual(observation.shape, (99,))
        self.assertEqual(observation.dtype, np.float32)
        self.assertTrue(np.isfinite(observation).all())
        expected_scaled = 2.0 * (self.adapter.default_qpos - self.adapter.lower) / (
            self.adapter.upper - self.adapter.lower
        ) - 1.0
        np.testing.assert_allclose(observation[0:23], expected_scaled, atol=1e-6)
        np.testing.assert_allclose(observation[23:46], 0.0, atol=1e-6)
        self.assertEqual(observation[69:81].shape, (12,))
        self.assertEqual(observation[91:95].shape, (4,))

    def test_quaternion_order(self):
        np.testing.assert_array_equal(wxyz_to_xyzw([1, 2, 3, 4]), [2, 3, 4, 1])

    def test_action_mapping_and_limits(self):
        self.adapter.reset()
        initial = self.adapter.previous_target.copy()
        zero_target = self.adapter.apply_action(np.zeros(23))
        np.testing.assert_allclose(zero_target[:7], initial[:7])
        np.testing.assert_allclose(
            zero_target[7:], (self.adapter.lower[7:] + self.adapter.upper[7:]) / 2.0
        )
        upper = self.adapter.apply_action(np.ones(23))
        self.assertTrue(np.all(upper <= self.adapter.upper))
        lower = self.adapter.apply_action(-np.ones(23))
        self.assertTrue(np.all(lower >= self.adapter.lower))


class TestPolicyAndEpisode(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.policy = PolicyRunner(MODEL, provider="cpu")

    def test_lstm_state_reset_and_update(self):
        self.policy.reset()
        self.assertEqual(self.policy.state["hidden_state"].shape, (1, 1, 768))
        self.assertEqual(self.policy.state["cell_state"].shape, (1, 1, 768))
        self.assertTrue(np.all(self.policy.state["hidden_state"] == 0))
        action = self.policy.infer(np.zeros(99, dtype=np.float32))
        self.assertEqual(action.shape, (1, 23))
        self.assertTrue(np.isfinite(action).all())
        self.assertFalse(np.all(self.policy.state["hidden_state"] == 0))
        self.policy.reset()
        self.assertTrue(np.all(self.policy.state["hidden_state"] == 0))

    def test_headless_episode_and_trajectory(self):
        adapter = RokaeAdapter(SCENE, META, seed=11)
        with tempfile.TemporaryDirectory() as directory:
            result = EpisodeRunner(adapter, self.policy).run(
                steps=600,
                npz_path=Path(directory) / "episode.npz",
                csv_path=Path(directory) / "episode.csv",
            )
            trajectory = result["trajectory"]
            self.assertGreater(result["steps"], 0)
            self.assertLessEqual(result["steps"], 600)
            self.assertTrue(np.isfinite(trajectory["observation"]).all())
            self.assertTrue(np.isfinite(trajectory["action"]).all())
            self.assertTrue(np.all(trajectory["target"] <= adapter.upper + 1e-6))
            self.assertTrue(np.all(trajectory["target"] >= adapter.lower - 1e-6))
            self.assertTrue((Path(directory) / "episode.npz").exists())
            self.assertTrue((Path(directory) / "episode.csv").exists())


if __name__ == "__main__":
    unittest.main()
