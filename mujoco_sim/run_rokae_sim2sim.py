import argparse
from contextlib import nullcontext

import numpy as np

import mujoco.viewer

from mujoco_sim.episode import EpisodeRunner
from mujoco_sim.rokae_alignment import load_reference, reference_initial_state
from mujoco_sim.rokae_adapter import RokaeAdapter
from onnx_deploy.policy_runner import PolicyRunner


def main():
    parser = argparse.ArgumentParser(description="Run Allegro-Rokae LSTM policy in MuJoCo")
    parser.add_argument("--scene", default="resources/assets/scenes/allegro_rokae.xml")
    parser.add_argument(
        "--model",
        default="resources/models/rokae_allegro/successful_best_seed0/rank01_maxsucc13/policy.onnx",
    )
    parser.add_argument(
        "--metadata",
        default="resources/models/rokae_allegro/successful_best_seed0/rank01_maxsucc13/policy.meta.yaml",
    )
    parser.add_argument(
        "--reference-initial",
        help="initialize MuJoCo from the first state of an Isaac reference NPZ",
    )
    parser.add_argument("--provider", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument(
        "--playback-speed",
        type=float,
        default=1.0,
        help="viewer playback rate relative to real time (for example, 0.25 is 4x slower)",
    )
    parser.add_argument("--npz", default="resources/trajectories/rokae_episode.npz")
    parser.add_argument("--csv", default="resources/trajectories/rokae_episode.csv")
    args = parser.parse_args()
    if args.playback_speed <= 0.0:
        parser.error("--playback-speed must be positive")
    adapter = RokaeAdapter(args.scene, args.metadata, args.seed)
    policy = PolicyRunner(args.model, args.provider)
    initial = None
    if args.reference_initial:
        reference = load_reference(args.reference_initial)
        if tuple(reference["joint_order"].tolist()) != tuple(adapter.joint_names):
            raise ValueError("Isaac reference joint order does not match policy metadata")
        initial = reference_initial_state(reference)
    context = mujoco.viewer.launch_passive(adapter.model, adapter.data) if args.viewer else nullcontext()
    with context as viewer:
        result = EpisodeRunner(adapter, policy).run(
            args.steps,
            viewer,
            args.npz,
            args.csv,
            initial=initial,
            playback_speed=args.playback_speed,
        )
    observation = result["trajectory"]["observation"]
    lifted_steps = int(np.count_nonzero(observation[:, 95] > 0.5)) if len(observation) else 0
    goal_steps = int(
        np.count_nonzero(np.linalg.norm(observation[:, 84:87], axis=1) <= 0.1125)
    ) if len(observation) else 0
    print(
        f"steps={result['steps']} elapsed={result['elapsed_seconds']:.3f}s "
        f"provider={policy.provider} final_sim_time={adapter.data.time:.5f} "
        f"lifted_steps={lifted_steps} goal_steps={goal_steps}"
    )


if __name__ == "__main__":
    main()
