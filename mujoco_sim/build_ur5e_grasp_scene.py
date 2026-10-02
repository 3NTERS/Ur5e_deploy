"""Build the fixed UR5e+2F-85 smoke-grasp MuJoCo scene."""

import argparse
import copy
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco

from mujoco_sim.build_rokae_mjcf import _indent


ROBOT_JOINTS = 12
ACTUATORS = 7


def _values(values):
    return " ".join(str(float(value)) for value in values)


def build(base_scene, output, object_scale=(1.0, 1.0, 1.0)):
    base_scene = Path(base_scene).resolve()
    output = Path(output).resolve()
    root = copy.deepcopy(ET.parse(str(base_scene)).getroot())
    root.set("model", "ur5e_robotiq_grasp")
    for mesh in root.find("asset").findall("mesh"):
        source = (base_scene.parent / mesh.get("file")).resolve()
        mesh.set("file", os.path.relpath(str(source), str(output.parent)))

    world = root.find("worldbody")
    ET.SubElement(
        world, "geom", name="floor", type="plane", size="2 2 0.05",
        rgba="0.18 0.18 0.18 1", friction="1 0.005 0.0001",
    )
    table = ET.SubElement(world, "body", name="table", pos="0.55 0 -0.10")
    ET.SubElement(
        table, "geom", name="table_geom", type="box", size="0.2375 0.2 0.15",
        rgba="0.45 0.35 0.25 1", friction="1 0.005 0.0001",
    )
    scale = tuple(float(value) for value in object_scale)
    object_body = ET.SubElement(world, "body", name="object", pos="0.55 0 0.15")
    ET.SubElement(object_body, "freejoint", name="object_freejoint")
    ET.SubElement(
        object_body, "geom", name="object_geom", type="box",
        size=_values(0.02 * value for value in scale), density="567",
        rgba="0.9 0.25 0.1 1", friction="0.9 0.005 0.0001",
    )
    ET.SubElement(
        world, "site", name="goal", pos="0.55 0 0.27", type="sphere",
        size="0.012", rgba="0 1 0 0.5",
    )
    home = root.find("keyframe").find("key")
    home.set("qpos", home.get("qpos") + " 0.55 0 0.15 1 0 0 0")
    output.parent.mkdir(parents=True, exist_ok=True)
    _indent(root)
    ET.ElementTree(root).write(str(output), encoding="unicode", xml_declaration=True)
    model = mujoco.MjModel.from_xml_path(str(output))
    if model.nq != ROBOT_JOINTS + 7 or model.nv != ROBOT_JOINTS + 6 or model.nu != ACTUATORS:
        raise RuntimeError(
            "Unexpected grasp scene dimensions: nq={}, nv={}, nu={}".format(
                model.nq, model.nv, model.nu
            )
        )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-scene",
        default="resources/assets/robots/ur5e_robotiq_2f85/ur5e_robotiq_2f85.xml",
    )
    parser.add_argument(
        "--output",
        default="resources/assets/robots/ur5e_robotiq_2f85/ur5e_robotiq_grasp.xml",
    )
    parser.add_argument("--object-scale", nargs=3, type=float, default=(1.0, 1.0, 1.0))
    args = parser.parse_args()
    print("Wrote {}".format(build(args.base_scene, args.output, args.object_scale)))


if __name__ == "__main__":
    main()
