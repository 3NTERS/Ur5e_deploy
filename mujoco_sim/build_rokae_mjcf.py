"""Build a self-contained Allegro-Rokae MJCF from the training URDF."""

import argparse
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco


JOINTS = [
    *(f"xmate_joint_{index}" for index in range(1, 8)),
    *(f"jmf{index}" for index in range(1, 5)),
    *(f"jpf{index}" for index in range(1, 5)),
    *(f"jif{index}" for index in range(1, 5)),
    *(f"jth{index}" for index in range(1, 5)),
]


def _indent(element, level=0):
    """Python 3.7-compatible equivalent of ElementTree.indent()."""
    prefix = "\n" + level * "  "
    child_prefix = "\n" + (level + 1) * "  "
    if len(element):
        if not element.text or not element.text.strip():
            element.text = child_prefix
        for child in element:
            _indent(child, level + 1)
            if not child.tail or not child.tail.strip():
                child.tail = child_prefix
        element[-1].tail = prefix


def _find_body(root, name):
    body = root.find(f".//body[@name='{name}']")
    if body is None:
        raise ValueError(f"Converted URDF is missing body {name}")
    return body


def build(source, output):
    source = Path(source).resolve()
    output = Path(output).resolve()
    asset_root = source.parent
    with tempfile.TemporaryDirectory() as directory:
        staging_root = Path(directory)
        staged = staging_root / "robot.urdf"
        # MuJoCo 2.3's URDF importer resolves meshes by basename beside the
        # URDF, even when the URDF contains an absolute path. Symlinks keep
        # conversion compatible without copying or modifying source assets.
        for mesh_dir in (asset_root / "meshes/xmate", asset_root / "meshes/allegro"):
            for mesh_path in mesh_dir.glob("*"):
                link = staging_root / mesh_path.name
                if not link.exists():
                    link.symlink_to(mesh_path)
        text = source.read_text(encoding="utf-8")
        text = text.replace(
            "package://xMatePro7_description/meshes/visual/", str(asset_root / "meshes/xmate") + "/"
        ).replace(
            "package://xMatePro7_description/meshes/collision/", str(asset_root / "meshes/xmate") + "/"
        ).replace(
            "package://allegro_hand_model/meshes/", str(asset_root / "meshes/allegro") + "/"
        )
        text = text.replace(
            "</robot>",
            '<mujoco><compiler balanceinertia="true" discardvisual="false"/></mujoco></robot>',
        )
        staged.write_text(text, encoding="utf-8")
        model = mujoco.MjModel.from_xml_path(str(staged))
        converted = Path(directory) / "robot.xml"
        mujoco.mj_saveLastXML(str(converted), model)
        tree = ET.parse(converted)

    root = tree.getroot()
    root.set("model", "allegro_rokae_sim2sim")
    compiler = root.find("compiler")
    compiler.set("meshdir", ".")
    for mesh in root.findall("./asset/mesh"):
        basename = Path(mesh.get("file")).name
        family = "xmate" if basename.startswith("xMate") else "allegro"
        mesh.set("file", f"../robots/rokae_allegro/meshes/{family}/{basename}")

    option = root.find("option")
    if option is None:
        option = ET.SubElement(root, "option")
    option.set("timestep", "0.002")
    option.set("gravity", "0 0 -9.81")
    option.set("integrator", "implicitfast")

    default = ET.SubElement(root, "default")
    ET.SubElement(default, "joint", damping="5", armature="0")
    ET.SubElement(default, "geom", friction="1 0.005 0.0001", condim="4")

    world = root.find("worldbody")
    for body in world.iter("body"):
        base_freejoint = body.find("./joint[@name='xMatePro7_base_free_joint']")
        if base_freejoint is not None:
            body.remove(base_freejoint)
            break
    else:
        raise ValueError("Converted URDF is missing xMatePro7_base_free_joint")
    ET.SubElement(world, "light", pos="0 0 3", directional="true")
    ET.SubElement(world, "geom", name="ground", type="plane", size="3 3 0.1", rgba="0.2 0.3 0.4 1")
    table = ET.SubElement(world, "body", name="table", pos="0.8 0 0.38")
    ET.SubElement(table, "geom", type="box", size="0.2 0.225 0.15", rgba="0.6 0.4 0.25 1")
    obj = ET.SubElement(world, "body", name="object", pos="0.75 0 0.555")
    ET.SubElement(obj, "freejoint", name="object_freejoint")
    ET.SubElement(obj, "geom", name="object_geom", type="box", size="0.025 0.025 0.025", mass="0.125", rgba="0.8 0.2 0.2 1")
    bucket = ET.SubElement(world, "body", name="bucket", pos="1.0 0.6 0.45")
    ET.SubElement(bucket, "geom", name="bucket_bottom", type="box", size="0.14 0.14 0.01", rgba="0.2 0.5 0.8 1")
    for name, pos, size in (
        ("bucket_left", "0 0.14 0.12", "0.14 0.01 0.12"),
        ("bucket_right", "0 -0.14 0.12", "0.14 0.01 0.12"),
        ("bucket_front", "0.14 0 0.12", "0.01 0.14 0.12"),
        ("bucket_back", "-0.14 0 0.12", "0.01 0.14 0.12"),
    ):
        ET.SubElement(bucket, "geom", name=name, type="box", pos=pos, size=size, rgba="0.2 0.5 0.8 1")
    ET.SubElement(bucket, "site", name="goal", pos="0 0 0.05", size="0.02", rgba="0 1 0 0.5")

    # The fixed hand base is fused into link 7 by MuJoCo's URDF compiler.
    ET.SubElement(
        _find_body(root, "xMatePro7_link7"),
        "site",
        name="palm_center",
        pos="-0.07 0 0.307",
        quat="0.707388 0 0 -0.706825",
        size="0.008",
    )
    offsets = {"pf4": "0.035 0 0", "mf4": "0.035 0 0", "if4": "0.035 0 0", "th4": "0 0.035 0"}
    for body_name, pos in offsets.items():
        ET.SubElement(_find_body(root, body_name), "site", name=f"{body_name}_tip", pos=pos, size="0.006")

    actuator = ET.SubElement(root, "actuator")
    for index, joint in enumerate(JOINTS):
        joint_element = root.find(f".//joint[@name='{joint}']")
        if joint_element is None:
            raise ValueError(f"Converted URDF is missing joint {joint}")
        joint_element.set("damping", "5")
        effort = "300" if index < 7 else "10"
        kp = "140" if index < 7 else "40"
        ET.SubElement(
            actuator,
            "position",
            name=f"{joint}_position",
            joint=joint,
            kp=kp,
            ctrllimited="true",
            ctrlrange=joint_element.get("range"),
            forcelimited="true",
            forcerange=f"-{effort} {effort}",
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    _indent(root)
    tree.write(output, encoding="unicode", xml_declaration=True)
    checked = mujoco.MjModel.from_xml_path(str(output))
    if checked.njnt != 24 or checked.nu != 23:  # 23 robot joints plus object freejoint.
        raise RuntimeError(f"Unexpected generated model dimensions: njnt={checked.njnt}, nu={checked.nu}")
    print(f"Wrote {output} (nq={checked.nq}, nv={checked.nv}, nu={checked.nu})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="resources/assets/robots/rokae_allegro/allegro_rokae.urdf")
    parser.add_argument("--output", default="resources/assets/scenes/allegro_rokae.xml")
    args = parser.parse_args()
    build(args.source, args.output)


if __name__ == "__main__":
    main()
