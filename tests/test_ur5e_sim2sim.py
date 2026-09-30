import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from mujoco_sim.ur5e_adapter import GRIPPER_MULTIPLIERS, Ur5eThrowAdapter
from mujoco_sim.build_ur5e_throw_scene import build
from mujoco_sim.ur5e_sim2sim import (
    alignment_report,
    initial_from_reference,
    load_reference,
    run_trajectory,
    save_npz,
)


ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "resources/assets/robots/ur5e_robotiq_2f85/ur5e_robotiq_throw.xml"
METADATA = ROOT / "resources/models/ur5e_robotiq/policy.meta.yaml"
POLICY = ROOT / "resources/models/ur5e_robotiq/policy.onnx"


class ZeroPolicy:
    observation_dim = 69
    action_dim = 7

    def reset(self):
        pass

    def infer(self, observation):
        return np.zeros((1, 7), dtype=np.float32)


def all_finite(value):
    if isinstance(value, dict):
        return all(all_finite(item) for item in value.values())
    return value is None or isinstance(value, (str, int)) or np.isfinite(value)


class TestUr5eSim2Sim(unittest.TestCase):
    def adapter(self, seed=0):
        return Ur5eThrowAdapter(SCENE, METADATA, seed=seed)

    def test_scene_and_policy_contract(self):
        adapter = self.adapter()
        self.assertEqual(adapter.model.nq, 19)
        self.assertEqual(adapter.model.nv, 18)
        self.assertEqual(adapter.model.nu, 7)
        observation = adapter.reset()
        self.assertEqual(observation.shape, (69,))
        self.assertEqual(adapter.model.njnt, 13)  # 12 robot hinges plus the object free joint.
        self.assertEqual(tuple(adapter.joint_names), tuple(adapter.metadata["io"]["joint_order"]))
        np.testing.assert_allclose(adapter.model.key_qpos[0, :6], adapter.home)
        self.assertAlmostEqual(adapter.model.body_mass[adapter.object_body], 567.0 * 0.04 ** 3)
        np.testing.assert_allclose(adapter.previous_target[:6], adapter.data.qpos[adapter.qpos_addr[:6]])
        np.testing.assert_allclose(observation[37:41], adapter.snapshot()["object_state"][3:7])
        self.assertAlmostEqual(adapter.model.opt.timestep, 0.002)
        self.assertAlmostEqual(adapter.policy_period, 0.01667)

    def test_scene_builder_supports_an_explicit_output(self):
        base = ROOT / "resources/assets/robots/ur5e_robotiq_2f85/ur5e_robotiq_2f85.xml"
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "throw.xml"
            self.assertEqual(build(base, output), output.resolve())
            adapter = Ur5eThrowAdapter(output, METADATA)
            self.assertEqual(adapter.model.nu, 7)

    def test_action_mapping_matches_training_master_joint(self):
        adapter = self.adapter()
        adapter.reset()
        initial = adapter.previous_target.copy()
        action = np.asarray([1.0, -1.0, 0.5, 0.0, -0.5, 1.0, 1.0])
        target = adapter.apply_action(action)
        expected_arm = np.clip(
            initial[:6] + 10.0 * adapter.policy_period * action[:6],
            adapter.lower[:6], adapter.upper[:6],
        )
        np.testing.assert_allclose(target[:6], expected_arm)
        np.testing.assert_allclose(target[6:], 0.72 * GRIPPER_MULTIPLIERS)
        self.assertAlmostEqual(adapter.data.ctrl[adapter.gripper_actuator_id], 255.0)

    def test_fixed_seed_is_reproducible(self):
        first = run_trajectory(self.adapter(seed=9), policy=ZeroPolicy(), steps=5)
        second = run_trajectory(self.adapter(seed=9), policy=ZeroPolicy(), steps=5)
        for key in ("observation", "action", "joint_position", "object_state", "goal_position"):
            np.testing.assert_allclose(first[key], second[key], atol=1e-12)

    def test_reference_replays_and_produces_finite_report(self):
        reference = run_trajectory(self.adapter(seed=4), policy=ZeroPolicy(), steps=6)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.npz"
            save_npz(path, reference)
            loaded = load_reference(path)
            replay = run_trajectory(
                self.adapter(seed=99),
                actions=loaded["action"],
                initial=initial_from_reference(loaded),
            )
        report = alignment_report(loaded, replay, self.adapter().joint_names)
        self.assertTrue(all_finite(report))
        self.assertEqual(report["steps"], 6)
        self.assertEqual(report["joint_position"]["shoulder_pan_joint"]["rmse"], 0.0)
        json.dumps(report, allow_nan=False)

    def test_reference_rejects_wrong_dimensions_and_nonfinite_values(self):
        trajectory = run_trajectory(self.adapter(seed=2), policy=ZeroPolicy(), steps=2)
        with tempfile.TemporaryDirectory() as directory:
            bad_shape = dict(trajectory)
            bad_shape["action"] = np.zeros((2, 6))
            shape_path = Path(directory) / "bad_shape.npz"
            save_npz(shape_path, bad_shape)
            with self.assertRaisesRegex(ValueError, "action has shape"):
                load_reference(shape_path)

            nonfinite = dict(trajectory)
            nonfinite["observation"] = trajectory["observation"].copy()
            nonfinite["observation"][0, 0] = np.nan
            nonfinite_path = Path(directory) / "nonfinite.npz"
            save_npz(nonfinite_path, nonfinite)
            with self.assertRaisesRegex(ValueError, "NaN or Inf"):
                load_reference(nonfinite_path)

    @unittest.skipUnless(POLICY.is_file(), "policy.onnx is not committed")
    def test_real_policy_contract(self):
        from onnx_deploy.policy_runner import PolicyRunner
        policy = PolicyRunner(POLICY, "cpu")
        result = run_trajectory(self.adapter(), policy=policy, steps=2)
        self.assertEqual(result["action"].shape, (2, 7))


if __name__ == "__main__":
    unittest.main()
