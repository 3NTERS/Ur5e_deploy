import argparse
from contextlib import nullcontext

import mujoco.viewer

from mujoco_sim.episode import EpisodeRunner
from mujoco_sim.rokae_adapter import RokaeAdapter
from onnx_deploy.policy_runner import PolicyRunner


def main():
    parser = argparse.ArgumentParser(description="Run Allegro-Rokae LSTM policy in MuJoCo")
    parser.add_argument("--scene", default="resources/assets/scenes/allegro_rokae.xml")
    parser.add_argument("--model", default="resources/models/rokae_allegro/policy.onnx")
    parser.add_argument("--metadata", default="resources/models/rokae_allegro/policy.meta.yaml")
    parser.add_argument("--provider", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--npz", default="resources/trajectories/rokae_episode.npz")
    parser.add_argument("--csv", default="resources/trajectories/rokae_episode.csv")
    args = parser.parse_args()
    adapter = RokaeAdapter(args.scene, args.metadata, args.seed)
    policy = PolicyRunner(args.model, args.provider)
    context = mujoco.viewer.launch_passive(adapter.model, adapter.data) if args.viewer else nullcontext()
    with context as viewer:
        result = EpisodeRunner(adapter, policy).run(args.steps, viewer, args.npz, args.csv)
    print(
        f"steps={result['steps']} elapsed={result['elapsed_seconds']:.3f}s "
        f"provider={policy.provider} final_sim_time={adapter.data.time:.5f}"
    )


if __name__ == "__main__":
    main()
