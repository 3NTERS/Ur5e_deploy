import os
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from mujoco_sim.build_ur5e_robotiq_2f85_mjcf import (
    ACTUATORS,
    ARM_JOINTS,
    GRIPPER_EQUALITIES,
    GRIPPER_JOINTS,
)


ROOT = Path(__file__).resolve().parents[1]
ROBOT_ROOT = ROOT / "resources/assets/robots/ur5e_robotiq_2f85"
MODEL_PATH = ROBOT_ROOT / "ur5e_robotiq_2f85.xml"
GRIPPER_PATH = (
    ROBOT_ROOT / "components/robotiq_2f_85_gripper_visualization/2f85.xml"
)


def _names(model, object_type, count):
    return [mujoco.mj_id2name(model, object_type, index) for index in range(count)]


def _relative_pose(data, parent_id, child_id):
    parent_rotation = data.xmat[parent_id].reshape(3, 3)
    child_rotation = data.xmat[child_id].reshape(3, 3)
    position = parent_rotation.T.dot(data.xpos[child_id] - data.xpos[parent_id])
    rotation = parent_rotation.T.dot(child_rotation)
    return position, rotation


class TestUr5eRobotiqAssets(unittest.TestCase):
    def setUp(self):
        self.model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        self.data = mujoco.MjData(self.model)

    def test_joint_and_actuator_contract(self):
        joints = _names(self.model, mujoco.mjtObj.mjOBJ_JOINT, self.model.njnt)
        actuators = _names(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, self.model.nu)
        self.assertEqual(joints, list(ARM_JOINTS + GRIPPER_JOINTS))
        self.assertEqual(actuators, list(ACTUATORS))
        self.assertEqual((self.model.nq, self.model.nv, self.model.nu, self.model.neq), (12, 12, 7, 5))
        self.assertFalse(np.any(self.model.jnt_type == mujoco.mjtJoint.mjJNT_FREE))
        np.testing.assert_allclose(self.model.actuator_ctrlrange[-1], [0.0, 255.0])
        gripper = mujoco.MjModel.from_xml_path(str(GRIPPER_PATH))
        self.assertEqual((gripper.nq, gripper.nv, gripper.nu, gripper.neq), (6, 6, 1, 5))
        expected_ranges = (
            [0.0, 0.8],
            [-0.8757, 0.0],
            [0.0, 0.8757],
            [0.0, 0.81],
            [-0.8757, 0.0],
            [0.0, 0.8757],
        )
        for name, expected in zip(GRIPPER_JOINTS, expected_ranges):
            joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            np.testing.assert_allclose(self.model.jnt_range[joint], expected)

    def test_names_are_unique(self):
        collections = (
            (mujoco.mjtObj.mjOBJ_BODY, self.model.nbody),
            (mujoco.mjtObj.mjOBJ_JOINT, self.model.njnt),
            (mujoco.mjtObj.mjOBJ_ACTUATOR, self.model.nu),
            (mujoco.mjtObj.mjOBJ_MESH, self.model.nmesh),
            (mujoco.mjtObj.mjOBJ_MATERIAL, self.model.nmat),
            (mujoco.mjtObj.mjOBJ_SITE, self.model.nsite),
            (mujoco.mjtObj.mjOBJ_TENDON, self.model.ntendon),
            (mujoco.mjtObj.mjOBJ_EQUALITY, self.model.neq),
        )
        for object_type, count in collections:
            names = [name for name in _names(self.model, object_type, count) if name is not None]
            self.assertEqual(len(names), len(set(names)))

    def test_all_meshes_are_repository_relative(self):
        tree = ET.parse(MODEL_PATH)
        for mesh in tree.findall("./asset/mesh"):
            mesh_path = Path(mesh.get("file"))
            self.assertFalse(mesh_path.is_absolute())
            resolved = (ROBOT_ROOT / mesh_path).resolve()
            self.assertEqual(os.path.commonpath((str(ROBOT_ROOT.resolve()), str(resolved))), str(ROBOT_ROOT.resolve()))
            self.assertTrue(resolved.is_file(), str(resolved))
        self.assertNotIn("MuJoCo-UR5e-with-Robotiq", MODEL_PATH.read_text(encoding="utf-8"))
        self.assertNotIn("components/robotiq_2f85", MODEL_PATH.read_text(encoding="utf-8"))

    def test_gripper_is_rigidly_mounted(self):
        wrist = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "wrist_3_link")
        gripper = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "robotiq_arg2f_base_link"
        )
        mujoco.mj_forward(self.model, self.data)
        initial_position, initial_rotation = _relative_pose(self.data, wrist, gripper)
        self.data.qpos[:6] = [0.4, -1.0, 1.2, -0.7, 0.6, -0.2]
        mujoco.mj_forward(self.model, self.data)
        moved_position, moved_rotation = _relative_pose(self.data, wrist, gripper)
        np.testing.assert_allclose(moved_position, initial_position, atol=1e-12)
        np.testing.assert_allclose(moved_rotation, initial_rotation, atol=1e-12)

    def test_gripper_body_joint_topology(self):
        for side in ("right", "left"):
            body_ids = {}
            for link in (
                "outer_knuckle",
                "outer_finger",
                "inner_finger",
                "inner_finger_pad",
                "inner_knuckle",
            ):
                body = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_BODY, f"robotiq_{side}_{link}"
                )
                body_ids[link] = body
            for moving_link in ("outer_knuckle", "inner_finger", "inner_knuckle"):
                body = body_ids[moving_link]
                self.assertEqual(self.model.body_jntnum[body], 1)
                joint = self.model.body_jntadr[body]
                self.assertEqual(self.model.jnt_type[joint], mujoco.mjtJoint.mjJNT_HINGE)
            for fixed_link in ("outer_finger", "inner_finger_pad"):
                self.assertEqual(self.model.body_jntnum[body_ids[fixed_link]], 0)
            self.assertEqual(
                self.model.body_parentid[body_ids["outer_finger"]],
                body_ids["outer_knuckle"],
            )
            self.assertEqual(
                self.model.body_parentid[body_ids["inner_finger"]],
                body_ids["outer_finger"],
            )
            np.testing.assert_allclose(
                self.model.body_pos[body_ids["inner_finger"]], [0.0, 0.0061, 0.0471]
            )
            base = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, "robotiq_arg2f_base_link"
            )
            self.assertEqual(
                self.model.body_parentid[body_ids["outer_knuckle"]], base
            )
            self.assertEqual(
                self.model.body_parentid[body_ids["inner_knuckle"]], base
            )
            self.assertEqual(
                self.model.body_parentid[body_ids["inner_finger_pad"]],
                body_ids["inner_finger"],
            )
        for fixed_body in ("robotiq_mount_frame", "robotiq_arg2f_base_link"):
            body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, fixed_body)
            self.assertEqual(self.model.body_jntnum[body], 0)

    def test_component_joints_have_explicit_viewer_attributes(self):
        tree = ET.parse(GRIPPER_PATH)
        joints = tree.findall(".//worldbody//joint")
        self.assertEqual(len(joints), len(GRIPPER_JOINTS))
        for joint in joints:
            self.assertEqual(joint.get("type"), "hinge", joint.get("name"))
            self.assertEqual(joint.get("pos"), "0 0 0", joint.get("name"))
            self.assertEqual(joint.get("axis"), "1 0 0", joint.get("name"))

    def test_actuator_force_is_distributed_to_all_mimic_joints(self):
        tree = ET.parse(GRIPPER_PATH)
        tendon = tree.find("./tendon/fixed[@name='finger_coupling']")
        self.assertIsNotNone(tendon)
        entries = {
            entry.get("joint"): float(entry.get("coef"))
            for entry in tendon.findall("joint")
        }
        self.assertEqual(set(entries), set(GRIPPER_JOINTS))
        self.assertGreater(entries["finger_joint"], 0.0)
        for name in ("left_inner_knuckle_joint", "right_outer_knuckle_joint", "right_inner_knuckle_joint"):
            self.assertGreater(entries[name], 0.0)
        for name in ("left_inner_finger_joint", "right_inner_finger_joint"):
            self.assertLess(entries[name], 0.0)
        actuator = tree.find("./actuator/general[@name='fingers_actuator']")
        self.assertEqual(actuator.get("tendon"), "finger_coupling")
        self.assertIsNone(actuator.get("joint"))
        data = mujoco.MjData(self.model)
        data.ctrl[-1] = 255.0
        mujoco.mj_forward(self.model, data)
        expected_moment = np.array([1.0, -1.0, 1.0, 1.0, -1.0, 1.0]) / 6.0
        np.testing.assert_allclose(
            data.actuator_moment[-1, -len(GRIPPER_JOINTS):],
            expected_moment,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            data.qfrc_actuator[-len(GRIPPER_JOINTS):],
            expected_moment * 30.0,
            atol=1e-8,
        )

    def test_pad_remains_fixed_to_follower(self):
        follower = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "robotiq_right_inner_finger"
        )
        mujoco.mj_forward(self.model, self.data)
        fixed_pairs = ((follower, mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "robotiq_right_inner_finger_pad"
        )),)
        initial_poses = [_relative_pose(self.data, parent, child) for parent, child in fixed_pairs]
        follower_joint = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "right_inner_finger_joint"
        )
        self.data.qpos[self.model.jnt_qposadr[follower_joint]] = -0.5
        mujoco.mj_forward(self.model, self.data)
        for (parent, child), (initial_position, initial_rotation) in zip(fixed_pairs, initial_poses):
            moved_position, moved_rotation = _relative_pose(self.data, parent, child)
            np.testing.assert_allclose(moved_position, initial_position, atol=1e-12)
            np.testing.assert_allclose(moved_rotation, initial_rotation, atol=1e-12)

    def test_external_force_on_finger_joint_moves_mimics(self):
        data = mujoco.MjData(self.model)
        joint_addresses = [
            self.model.jnt_qposadr[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            ]
            for name in GRIPPER_JOINTS
        ]
        finger_dof = self.model.jnt_dofadr[
            mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint"
            )
        ]
        for _ in range(1000):
            data.qfrc_applied[finger_dof] = 5.0
            mujoco.mj_step(self.model, data)
        q = data.qpos[joint_addresses]
        self.assertGreater(q[0], 0.04)
        np.testing.assert_allclose(q[[2, 3, 5]], q[0], atol=1e-3)
        np.testing.assert_allclose(q[[1, 4]], -q[0], atol=1e-3)

    def test_mimic_constraint_names(self):
        equalities = _names(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, self.model.neq)
        self.assertEqual(
            equalities,
            list(GRIPPER_EQUALITIES),
        )

    def test_official_mimic_constraints_are_joint_equalities(self):
        for equality_name in GRIPPER_EQUALITIES:
            equality = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_EQUALITY, equality_name
            )
            self.assertEqual(self.model.eq_type[equality], mujoco.mjtEq.mjEQ_JOINT)

    def test_open_close_dynamics(self):
        keyframe = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        right_pad = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "robotiq_right_inner_finger_pad"
        )
        left_pad = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "robotiq_left_inner_finger_pad"
        )
        joint_addresses = [
            self.model.jnt_qposadr[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            ]
            for name in GRIPPER_JOINTS
        ]

        def settle(control):
            data = mujoco.MjData(self.model)
            mujoco.mj_resetDataKeyframe(self.model, data, keyframe)
            data.ctrl[-1] = control
            for _ in range(1000):
                mujoco.mj_step(self.model, data)
            for values in (data.qpos, data.qvel, data.ctrl, data.xpos, data.efc_pos):
                self.assertTrue(np.isfinite(values).all())
            q = data.qpos[joint_addresses]
            np.testing.assert_allclose(q[[2, 3, 5]], q[0], atol=2e-4)
            np.testing.assert_allclose(q[[1, 4]], -q[0], atol=2e-4)
            self.assertLess(np.max(np.abs(data.qvel[joint_addresses])), 1e-3)
            equality_rows = data.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY
            self.assertLess(np.max(np.abs(data.efc_pos[equality_rows])), 2e-4)
            return np.linalg.norm(data.xpos[right_pad] - data.xpos[left_pad])

        open_gap = settle(0.0)
        closed_gap = settle(255.0)
        self.assertGreater(open_gap, 0.09)
        self.assertLess(closed_gap, 0.07)
        self.assertGreater(open_gap - closed_gap, 0.03)


if __name__ == "__main__":
    unittest.main()
