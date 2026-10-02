#!/usr/bin/env python3
"""Audit UR5e+2F85 rigid-body inertias before training or deployment."""

import argparse
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ISAAC_ROOT = Path("/home/liang/Workspace/Isaacgym_cjlu/IsaacGymEnvs")
URDF = ISAAC_ROOT / "assets/ur5e_robotiq/ur5e_robotiq_2f85_isaacgym.urdf"
MJCF = ROOT / "resources/assets/robots/ur5e_robotiq_2f85/ur5e_robotiq_2f85.xml"

BODY_MAP = {
    "base_link": "base",
    "shoulder_link": "shoulder_link",
    "upper_arm_link": "upper_arm_link",
    "forearm_link": "forearm_link",
    "wrist_1_link": "wrist_1_link",
    "wrist_2_link": "wrist_2_link",
    "wrist_3_link": "wrist_3_link",
    "robotiq_arg2f_base_link": "robotiq_arg2f_base_link",
    "left_outer_knuckle": "robotiq_left_outer_knuckle",
    "left_outer_finger": "robotiq_left_outer_finger",
    "left_inner_finger": "robotiq_left_inner_finger",
    "left_inner_finger_pad": "robotiq_left_inner_finger_pad",
    "left_inner_knuckle": "robotiq_left_inner_knuckle",
    "right_outer_knuckle": "robotiq_right_outer_knuckle",
    "right_outer_finger": "robotiq_right_outer_finger",
    "right_inner_finger": "robotiq_right_inner_finger",
    "right_inner_finger_pad": "robotiq_right_inner_finger_pad",
    "right_inner_knuckle": "robotiq_right_inner_knuckle",
}


def _vector(text, length=3):
    value = np.asarray([float(item) for item in text.split()], dtype=np.float64)
    if value.shape != (length,):
        raise ValueError("Expected {} values, got {}".format(length, text))
    return value


def _valid(mass, matrix):
    eigenvalues = np.linalg.eigvalsh(matrix)
    margin = eigenvalues[0] + eigenvalues[1] - eigenvalues[2]
    return bool(mass > 0.0 and eigenvalues[0] > 0.0 and margin >= -1e-12), eigenvalues, margin


def _urdf_bodies(path):
    root = ET.parse(str(path)).getroot()
    bodies = {}
    missing = []
    for name in BODY_MAP:
        link = root.find("./link[@name='{}']".format(name))
        inertial = None if link is None else link.find("inertial")
        if inertial is None:
            missing.append(name)
            continue
        mass = float(inertial.find("mass").get("value"))
        origin = inertial.find("origin")
        com = np.zeros(3) if origin is None else _vector(origin.get("xyz", "0 0 0"))
        values = inertial.find("inertia").attrib
        matrix = np.asarray(
            [
                [values["ixx"], values["ixy"], values["ixz"]],
                [values["ixy"], values["iyy"], values["iyz"]],
                [values["ixz"], values["iyz"], values["izz"]],
            ],
            dtype=np.float64,
        )
        ok, eigenvalues, margin = _valid(mass, matrix)
        bodies[name] = {
            "mass": mass,
            "com": com,
            "eigenvalues": eigenvalues,
            "triangle_margin": margin,
            "valid": ok,
        }
    return bodies, missing


def _mujoco_bodies(path):
    import mujoco

    root = ET.parse(str(path)).getroot()
    compiler = root.find("compiler")
    if compiler is not None and compiler.get("balanceinertia", "false").lower() == "true":
        raise RuntimeError("MuJoCo balanceinertia must not be enabled")
    model = mujoco.MjModel.from_xml_path(str(path))
    bodies = {}
    for urdf_name, mjcf_name in BODY_MAP.items():
        index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, mjcf_name)
        if index < 0:
            raise RuntimeError("MuJoCo body is missing: {}".format(mjcf_name))
        eigenvalues = np.sort(model.body_inertia[index].copy())
        bodies[urdf_name] = {
            "mass": float(model.body_mass[index]),
            "com": model.body_ipos[index].copy(),
            "eigenvalues": eigenvalues,
            "triangle_margin": float(eigenvalues[0] + eigenvalues[1] - eigenvalues[2]),
            "valid": bool(
                model.body_mass[index] > 0.0
                and eigenvalues[0] > 0.0
                and eigenvalues[0] + eigenvalues[1] >= eigenvalues[2] - 1e-12
            ),
        }
    return bodies


def _physx_bodies(isaac_root):
    if str(isaac_root) not in sys.path:
        sys.path.insert(0, str(isaac_root))
    import isaacgym  # noqa: F401
    import isaacgymenvs

    env = isaacgymenvs.make(
        seed=0,
        task="Ur5eRobotiqGrasp",
        num_envs=1,
        sim_device="cuda:0",
        rl_device="cuda:0",
        graphics_device_id=-1,
        headless=True,
        force_render=False,
    )
    names = env.gym.get_actor_rigid_body_names(env.envs[0], env.allegro_hands[0])
    props = env.gym.get_actor_rigid_body_properties(env.envs[0], env.allegro_hands[0])
    by_name = dict(zip(names, props))
    bodies = {}
    for name in BODY_MAP:
        prop = by_name[name]
        matrix = np.asarray(
            [
                [prop.inertia.x.x, prop.inertia.y.x, prop.inertia.z.x],
                [prop.inertia.x.y, prop.inertia.y.y, prop.inertia.z.y],
                [prop.inertia.x.z, prop.inertia.y.z, prop.inertia.z.z],
            ],
            dtype=np.float64,
        )
        ok, eigenvalues, margin = _valid(float(prop.mass), matrix)
        bodies[name] = {
            "mass": float(prop.mass),
            "eigenvalues": eigenvalues,
            "triangle_margin": margin,
            "valid": ok,
        }
    env.gym.destroy_sim(env.sim)
    return bodies


def _serializable(bodies):
    return {
        name: {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in body.items()
        }
        for name, body in bodies.items()
    }


def _compare(reference, actual, include_com):
    failures = []
    differences = {}
    for name in BODY_MAP:
        item = {
            "mass_abs": abs(actual[name]["mass"] - reference[name]["mass"]),
            "eigenvalues_max_abs": float(
                np.max(np.abs(actual[name]["eigenvalues"] - reference[name]["eigenvalues"]))
            ),
        }
        if include_com:
            item["com_max_abs"] = float(
                np.max(np.abs(actual[name]["com"] - reference[name]["com"]))
            )
        mass_ok = np.isclose(actual[name]["mass"], reference[name]["mass"], rtol=1e-5, atol=1e-9)
        inertia_ok = np.allclose(
            actual[name]["eigenvalues"], reference[name]["eigenvalues"], rtol=1e-5, atol=1e-10
        )
        com_ok = not include_com or np.allclose(
            actual[name]["com"], reference[name]["com"], rtol=1e-5, atol=1e-8
        )
        item["matched"] = bool(mass_ok and inertia_ok and com_ok)
        if not item["matched"]:
            failures.append(name)
        differences[name] = item
    return differences, failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isaac-root", type=Path, default=ISAAC_ROOT)
    parser.add_argument("--urdf", type=Path, default=URDF)
    parser.add_argument("--mjcf", type=Path, default=MJCF)
    parser.add_argument("--runtime-isaac", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "resources/reports/ur5e_inertia_audit.json")
    args = parser.parse_args()

    urdf, missing = _urdf_bodies(args.urdf)
    mujoco_bodies = _mujoco_bodies(args.mjcf)
    static_difference, static_failures = _compare(urdf, mujoco_bodies, include_com=True)
    failures = list(missing) + [name for name, body in urdf.items() if not body["valid"]]
    failures += [name for name, body in mujoco_bodies.items() if not body["valid"]]
    failures += static_failures
    payload = {
        "format_version": 1,
        "urdf": str(args.urdf.resolve()),
        "mjcf": str(args.mjcf.resolve()),
        "mapped_body_count": len(BODY_MAP),
        "urdf_missing": missing,
        "urdf_bodies": _serializable(urdf),
        "mujoco_bodies": _serializable(mujoco_bodies),
        "urdf_mujoco_difference": static_difference,
    }
    if args.runtime_isaac:
        physx = _physx_bodies(args.isaac_root.resolve())
        runtime_difference, runtime_failures = _compare(urdf, physx, include_com=False)
        payload["physx_bodies"] = _serializable(physx)
        payload["urdf_physx_difference"] = runtime_difference
        failures += [name for name, body in physx.items() if not body["valid"]]
        failures += runtime_failures
    payload["failures"] = sorted(set(failures))
    payload["passed"] = not payload["failures"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("passed={} bodies={} report={}".format(payload["passed"], len(BODY_MAP), args.output.resolve()))
    if not payload["passed"]:
        raise SystemExit("Inertia audit failed: {}".format(", ".join(payload["failures"])))


if __name__ == "__main__":
    main()
