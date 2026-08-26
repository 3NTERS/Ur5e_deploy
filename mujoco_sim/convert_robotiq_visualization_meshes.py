"""Convert the official Robotiq visualization DAE meshes for MuJoCo 2.3.6."""

import argparse
from pathlib import Path

import trimesh


MESH_NAMES = (
    "robotiq_arg2f_85_base_link",
    "robotiq_arg2f_85_outer_knuckle",
    "robotiq_arg2f_85_outer_finger",
    "robotiq_arg2f_85_inner_knuckle",
    "robotiq_arg2f_85_inner_finger",
)


def convert(package_root):
    package_root = Path(package_root).resolve()
    mesh_root = package_root / "meshes"
    for group in ("visual", "collision"):
        output_dir = mesh_root / "mujoco" / group
        output_dir.mkdir(parents=True, exist_ok=True)
        for name in MESH_NAMES:
            source = mesh_root / group / (name + ".dae")
            if not source.is_file():
                if group == "collision" and name == "robotiq_arg2f_85_base_link":
                    continue
                raise FileNotFoundError(source)
            scene = trimesh.load(str(source), force="scene")
            mesh = trimesh.util.concatenate(tuple(scene.dump()))
            # MJCF assigns materials per geom; discard DAE textures so each OBJ is
            # self-contained and conversion does not emit shared MTL/PNG sidecars.
            mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh)
            output = output_dir / (name + ".obj")
            mesh.export(str(output))
            print("Wrote {} (vertices={}, faces={})".format(output, len(mesh.vertices), len(mesh.faces)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--package-root",
        default=(
            "resources/assets/robots/ur5e_robotiq_2f85/components/"
            "robotiq_2f_85_gripper_visualization"
        ),
    )
    args = parser.parse_args()
    convert(args.package_root)


if __name__ == "__main__":
    main()
