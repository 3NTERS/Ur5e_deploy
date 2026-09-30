"""Add the Ur5eRobotiq throw task geometry to the validated robot MJCF."""

import argparse
import copy
import math
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco

from mujoco_sim.build_rokae_mjcf import _indent


ROBOT_JOINTS = 12
ACTUATORS = 7
HOME = (-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0)


def _values(values):
    return " ".join(str(float(value)) for value in values)


def build(base_scene, output, object_scale=(1.0, 1.0, 1.0)):
    base_scene = Path(base_scene).resolve()
    output = Path(output).resolve()
    root = copy.deepcopy(ET.parse(str(base_scene)).getroot())
    root.set("model", "ur5e_robotiq_throw")

    asset = root.find("asset")
    for mesh in asset.findall("mesh"):
        source = (base_scene.parent / mesh.get("file")).resolve()
        mesh.set("file", os.path.relpath(str(source), str(output.parent)))
    bucket_mesh = base_scene.parent / "bucket.obj"
    ET.SubElement(
        asset,
        "mesh",
        name="training_bucket",
        file=os.path.relpath(str(bucket_mesh), str(output.parent)),
    )
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

    bucket = ET.SubElement(world, "body", name="bucket", pos="0.80 0.35 0")
    ET.SubElement(
        bucket, "geom", name="bucket_visual", type="mesh", mesh="training_bucket",
        contype="0", conaffinity="0", rgba="0.2 0.4 0.9 0.55",
    )
    ET.SubElement(
        bucket, "geom", name="bucket_bottom", type="cylinder", pos="0 0 0.006",
        size="0.077 0.006", rgba="0.2 0.4 0.9 0.3", friction="0.8 0.005 0.0001",
    )
    for index in range(12):
        angle = 2.0 * math.pi * index / 12.0
        radius = 0.094
        ET.SubElement(
            bucket, "geom", name="bucket_wall_{}".format(index), type="box",
            pos=_values((radius * math.cos(angle), radius * math.sin(angle), 0.10)),
            euler=_values((0.0, 0.0, angle)), size="0.031 0.006 0.10",
            rgba="0.2 0.4 0.9 0.2", friction="0.8 0.005 0.0001",
        )
    ET.SubElement(
        world, "site", name="goal", pos="0.80 0.35 0.05", type="sphere",
        size="0.012", rgba="0 1 0 0.5",
    )

    home = root.find("keyframe/key[@name='home']")
    home.set("qpos", home.get("qpos") + " 0.55 0 0.15 1 0 0 0")
    output.parent.mkdir(parents=True, exist_ok=True)
    _indent(root)
    ET.ElementTree(root).write(str(output), encoding="unicode", xml_declaration=True)

    model = mujoco.MjModel.from_xml_path(str(output))
    if model.nq != ROBOT_JOINTS + 7 or model.nv != ROBOT_JOINTS + 6 or model.nu != ACTUATORS:
        raise RuntimeError(
            "Unexpected throw scene dimensions: nq={}, nv={}, nu={}".format(
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
        default="resources/assets/robots/ur5e_robotiq_2f85/ur5e_robotiq_throw.xml",
    )
    parser.add_argument("--object-scale", nargs=3, type=float, default=(1.0, 1.0, 1.0))
    args = parser.parse_args()
    output = build(args.base_scene, args.output, args.object_scale)
    print("Wrote {}".format(output))


if __name__ == "__main__":
    main()
