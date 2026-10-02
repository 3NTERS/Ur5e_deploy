#!/usr/bin/env bash
set -euo pipefail

deploy_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(dirname "${deploy_dir}")"
isaacgym_root="/home/liang/Workspace/Isaacgym_cjlu/IsaacGymEnvs"
rlgpu_root="/home/liang/miniconda3/envs/rlgpu"
python_bin="${rlgpu_root}/bin/python3.7"
checkpoint="${repository_root}/resources/models/ur5e_lift_gripper/ur5e_lift_gripper_distilled.pth"

export LD_LIBRARY_PATH="${rlgpu_root}/lib:${rlgpu_root}/lib/python3.7/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
cd "${isaacgym_root}"

exec "${python_bin}" isaacgymenvs/train.py \
  task=Ur5eRobotiqLiftGripper \
  test=True headless=False num_envs=1 \
  checkpoint="${checkpoint}" \
  task.env.evaluationEpisodes=1 \
  train.params.config.player.games_num=1 \
  train.params.config.player.deterministic=True
