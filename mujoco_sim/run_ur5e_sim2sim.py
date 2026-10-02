#!/usr/bin/env python3
"""Run UR5e+2F85 in MuJoCo, or replay and compare an Isaac Gym reference."""

import argparse
from contextlib import nullcontext

from mujoco_sim.ur5e_adapter import Ur5eGraspAdapter
from mujoco_sim.ur5e_sim2sim import (
    alignment_report,
    initial_from_reference,
    load_reference,
    run_trajectory,
    save_csv,
    save_npz,
    save_report,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default="resources/assets/robots/ur5e_robotiq_2f85/ur5e_robotiq_grasp.xml")
    parser.add_argument("--model", default="resources/models/ur5e_robotiq/policy.onnx")
    parser.add_argument("--metadata", default="resources/models/ur5e_robotiq/policy.meta.yaml")
    parser.add_argument("--provider", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reference", help="Isaac Gym NPZ; replays its actions instead of ONNX inference")
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--npz", default="resources/trajectories/ur5e_grasp_mujoco.npz")
    parser.add_argument("--csv", default="resources/trajectories/ur5e_grasp_mujoco.csv")
    parser.add_argument("--report", default="resources/trajectories/ur5e_grasp_alignment.json")
    args = parser.parse_args()

    reference = load_reference(args.reference) if args.reference else None
    scale = reference["object_scale"] if reference is not None else (1.0, 1.0, 1.0)
    adapter = Ur5eGraspAdapter(args.scene, args.metadata, args.seed, scale)
    if reference is not None and tuple(reference["joint_order"].tolist()) != adapter.joint_names:
        raise ValueError("Isaac reference joint order does not match policy metadata")
    import mujoco.viewer
    context = mujoco.viewer.launch_passive(adapter.model, adapter.data) if args.viewer else nullcontext()
    with context as viewer:
        if reference is None:
            from onnx_deploy.policy_runner import PolicyRunner
            policy = PolicyRunner(args.model, args.provider)
            trajectory = run_trajectory(adapter, policy=policy, steps=args.steps, viewer=viewer)
            policy_model = args.model
        else:
            trajectory = run_trajectory(
                adapter,
                actions=reference["action"],
                viewer=viewer,
                initial=initial_from_reference(reference),
            )
            policy_model = None
    save_npz(args.npz, trajectory, policy_model)
    save_csv(args.csv, trajectory)
    if reference is not None:
        report = alignment_report(reference, trajectory, adapter.joint_names)
        save_report(args.report, report)
        print("alignment_report={}".format(args.report))
    print("steps={} final_sim_time={:.5f} npz={} csv={}".format(
        trajectory["action"].shape[0], adapter.data.time, args.npz, args.csv
    ))


if __name__ == "__main__":
    main()
