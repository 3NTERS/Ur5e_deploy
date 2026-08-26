"""Build a fixed-base UR5e with a statically mounted Robotiq 2F85."""

import argparse
import copy
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco

from mujoco_sim.build_rokae_mjcf import _indent


ARM_JOINTS = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)
GRIPPER_JOINTS = (
    "finger_joint",
    "left_inner_finger_joint",
    "left_inner_knuckle_joint",
    "right_outer_knuckle_joint",
    "right_inner_finger_joint",
    "right_inner_knuckle_joint",
)
ACTUATORS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow",
    "wrist_1",
    "wrist_2",
    "wrist_3",
    "fingers_actuator",
)
GRIPPER_EQUALITIES = (
    "left_inner_finger_mimic",
    "left_inner_knuckle_mimic",
    "right_outer_knuckle_mimic",
    "right_inner_finger_mimic",
    "right_inner_knuckle_mimic",
)


def _copy_children(source, target):
    for child in source:
        target.append(copy.deepcopy(child))


def _rewrite_assets(source, target, directory, prefix=None):
    mesh_names = {}
    material_names = {}
    for element in source:
        copied = copy.deepcopy(element)
        if copied.tag == "mesh":
            old_name = copied.get("name", Path(copied.get("file")).stem)
            new_name = "{}_{}".format(prefix, old_name) if prefix else old_name
            copied.set("name", new_name)
            copied.set("file", "{}/{}".format(directory, copied.get("file")))
            mesh_names[old_name] = new_name
        elif copied.tag == "material" and prefix:
            old_name = copied.get("name")
            new_name = "{}_{}".format(prefix, old_name)
            copied.set("name", new_name)
            material_names[old_name] = new_name
        target.append(copied)
    return mesh_names, material_names


def _rewrite_gripper_names(root, mesh_names, material_names):
    for element in root.iter():
        if element.tag == "body" and element.get("name"):
            element.set("name", "robotiq_{}".format(element.get("name")))
        if element.tag == "site" and element.get("name"):
            element.set("name", "robotiq_{}".format(element.get("name")))
        if element.tag == "geom" and element.get("name"):
            element.set("name", "robotiq_{}".format(element.get("name")))
        if element.get("mesh") in mesh_names:
            element.set("mesh", mesh_names[element.get("mesh")])
        if element.get("material") in material_names:
            element.set("material", material_names[element.get("material")])
        for attribute in ("body1", "body2"):
            if element.get(attribute):
                element.set(attribute, "robotiq_{}".format(element.get(attribute)))


def _namespace_gripper_classes(root):
    for element in root.iter():
        for attribute in ("class", "childclass"):
            if element.get(attribute):
                element.set(attribute, "robotiq_{}".format(element.get(attribute)))


def _find_body(root, name):
    body = root.find(".//body[@name='{}']".format(name))
    if body is None:
        raise ValueError("Missing body {}".format(name))
    return body


def build(component_root, output):
    component_root = Path(component_root).resolve()
    output = Path(output).resolve()
    arm = ET.parse(component_root / "ur5e/ur5e.xml").getroot()
    gripper = ET.parse(
        component_root / "robotiq_2f_85_gripper_visualization/2f85.xml"
    ).getroot()
    _namespace_gripper_classes(gripper)

    root = ET.Element("mujoco", model="ur5e_robotiq_2f85")
    ET.SubElement(root, "compiler", angle="radian", autolimits="true")
    ET.SubElement(
        root,
        "option",
        timestep="0.002",
        integrator="implicitfast",
        cone="elliptic",
        impratio="10",
    )

    defaults = ET.SubElement(root, "default")
    _copy_children(arm.find("default"), defaults)
    _copy_children(gripper.find("default"), defaults)

    assets = ET.SubElement(root, "asset")
    _rewrite_assets(arm.find("asset"), assets, "components/ur5e/assets")
    gripper_meshes, gripper_materials = _rewrite_assets(
        gripper.find("asset"),
        assets,
        "components/robotiq_2f_85_gripper_visualization/meshes",
        prefix="robotiq",
    )

    worldbody = ET.SubElement(root, "worldbody")
    arm_worldbody = arm.find("worldbody")
    _copy_children(arm_worldbody, worldbody)
    wrist = _find_body(worldbody, "wrist_3_link")
    attachment = wrist.find("site[@name='attachment_site']")
    if attachment is None:
        raise ValueError("UR5e component is missing attachment_site")
    mount = ET.SubElement(
        wrist,
        "body",
        name="robotiq_mount_frame",
        pos=attachment.get("pos", "0 0 0"),
        quat=attachment.get("quat", "1 0 0 0"),
    )
    gripper_body = copy.deepcopy(gripper.find("worldbody/body"))
    freejoint = gripper_body.find("joint[@type='free']")
    if freejoint is not None:
        gripper_body.remove(freejoint)
    _rewrite_gripper_names(gripper_body, gripper_meshes, gripper_materials)
    mount.append(gripper_body)

    contact = ET.SubElement(root, "contact")
    gripper_contact = copy.deepcopy(gripper.find("contact"))
    _rewrite_gripper_names(gripper_contact, gripper_meshes, gripper_materials)
    _copy_children(gripper_contact, contact)

    gripper_tendon = gripper.find("tendon")
    if gripper_tendon is not None:
        tendon = ET.SubElement(root, "tendon")
        _copy_children(gripper_tendon, tendon)

    equality = ET.SubElement(root, "equality")
    gripper_equality = copy.deepcopy(gripper.find("equality"))
    _rewrite_gripper_names(gripper_equality, gripper_meshes, gripper_materials)
    _copy_children(gripper_equality, equality)

    actuator = ET.SubElement(root, "actuator")
    _copy_children(arm.find("actuator"), actuator)
    _copy_children(gripper.find("actuator"), actuator)

    keyframe = ET.SubElement(root, "keyframe")
    home = arm.find("keyframe/key[@name='home']")
    arm_home_qpos = home.get("qpos").split()
    arm_home_ctrl = home.get("ctrl").split()
    ET.SubElement(
        keyframe,
        "key",
        name="home",
        qpos=" ".join(arm_home_qpos + ["0"] * len(GRIPPER_JOINTS)),
        ctrl=" ".join(arm_home_ctrl + ["0"]),
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    _indent(root)
    ET.ElementTree(root).write(output, encoding="unicode", xml_declaration=True)
    model = mujoco.MjModel.from_xml_path(str(output))
    joint_names = tuple(
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index)
        for index in range(model.njnt)
    )
    actuator_names = tuple(
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, index)
        for index in range(model.nu)
    )
    if joint_names != ARM_JOINTS + GRIPPER_JOINTS:
        raise RuntimeError("Unexpected joint order: {}".format(joint_names))
    if actuator_names != ACTUATORS:
        raise RuntimeError("Unexpected actuator order: {}".format(actuator_names))
    equality_names = tuple(
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_EQUALITY, index)
        for index in range(model.neq)
    )
    if equality_names != GRIPPER_EQUALITIES:
        raise RuntimeError("Unexpected gripper constraints: {}".format(equality_names))
    if any(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE):
        raise RuntimeError("Combined robot must not contain a free joint")
    gripper_base_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "robotiq_arg2f_base_link"
    )
    for side in ("left", "right"):
        body_ids = {
            link: mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, "robotiq_{}_{}".format(side, link)
            )
            for link in ("outer_knuckle", "outer_finger", "inner_finger", "inner_finger_pad", "inner_knuckle")
        }
        for moving_link in ("outer_knuckle", "inner_finger", "inner_knuckle"):
            if model.body_jntnum[body_ids[moving_link]] != 1:
                raise RuntimeError("{} {} must contain one hinge".format(side, moving_link))
        for fixed_link in ("outer_finger", "inner_finger_pad"):
            if model.body_jntnum[body_ids[fixed_link]] != 0:
                raise RuntimeError("{} {} must be fixed".format(side, fixed_link))
        expected_parents = {
            "outer_knuckle": gripper_base_id,
            "outer_finger": body_ids["outer_knuckle"],
            "inner_finger": body_ids["outer_finger"],
            "inner_finger_pad": body_ids["inner_finger"],
            "inner_knuckle": gripper_base_id,
        }
        for link, expected_parent in expected_parents.items():
            if model.body_parentid[body_ids[link]] != expected_parent:
                raise RuntimeError("Unexpected {} {} parent".format(side, link))
    print("Wrote {} (nq={}, nv={}, nu={}, neq={})".format(output, model.nq, model.nv, model.nu, model.neq))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--components",
        default="resources/assets/robots/ur5e_robotiq_2f85/components",
    )
    parser.add_argument(
        "--output",
        default="resources/assets/robots/ur5e_robotiq_2f85/ur5e_robotiq_2f85.xml",
    )
    args = parser.parse_args()
    build(args.components, args.output)


if __name__ == "__main__":
    main()
