#!/usr/bin/env bash
set -euo pipefail

DEPLOY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISAAC_ROOT="${ISAAC_ROOT:-/home/liang/Workspace/Isaacgym_cjlu/IsaacGymEnvs}"
OUTPUT_ROOT="${ISAAC_ROOT}/train_dir/ur5e_grasp_smoke_seed0"

cd "${DEPLOY_ROOT}"
PYTHONPATH="${ISAAC_ROOT}" conda run --no-capture-output -n rlgpu \
  python scripts/check_ur5e_inertias.py --isaac-root "${ISAAC_ROOT}" --runtime-isaac

mkdir -p "${OUTPUT_ROOT}"
cd "${OUTPUT_ROOT}"
RESTORE_CHECKPOINT=""
for MILESTONE in 10 20 30; do
  RESTORE_ARGUMENT=()
  if [[ -n "${RESTORE_CHECKPOINT}" ]]; then
    RESTORE_ARGUMENT+=("checkpoint=${RESTORE_CHECKPOINT}")
  fi
  PYTHONPATH="${ISAAC_ROOT}" conda run --no-capture-output -n rlgpu \
    python -m isaacgymenvs.train \
    task=Ur5eRobotiqGrasp \
    train=Ur5eRobotiqGraspLSTMPPO \
    experiment=ur5e_grasp_smoke_seed0 \
    seed=0 \
    torch_deterministic=True \
    num_envs=256 \
    max_iterations="${MILESTONE}" \
    headless=True \
    force_render=False \
    task.env.randomizeObjectDimensions=False \
    train.params.config.horizon_length=16 \
    train.params.config.seq_length=16 \
    train.params.config.minibatch_size=4096 \
    train.params.config.mini_epochs=2 \
    train.params.config.save_best_after=0 \
    train.params.config.save_frequency=10 \
    "${RESTORE_ARGUMENT[@]}"

  SAVED_CHECKPOINT="$(find "${OUTPUT_ROOT}/runs" -type f -name "last_ur5e_grasp_smoke_seed0_ep_${MILESTONE}_rew_*.pth" -print | sort | tail -n 1)"
  if [[ -z "${SAVED_CHECKPOINT}" ]]; then
    echo "Training completed without an epoch-${MILESTONE} checkpoint" >&2
    exit 1
  fi
  RESTORE_CHECKPOINT="${OUTPUT_ROOT}/ur5e_grasp_smoke_epoch${MILESTONE}.pth"
  cp "${SAVED_CHECKPOINT}" "${RESTORE_CHECKPOINT}"
  echo "Stable checkpoint: ${RESTORE_CHECKPOINT}"
done
