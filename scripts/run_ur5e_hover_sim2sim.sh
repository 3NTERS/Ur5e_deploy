#!/usr/bin/env bash
set -euo pipefail

deploy_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(dirname "${deploy_dir}")"
cd "${repository_root}"

rlgpu_prefix="$(conda run -n rlgpu python -c 'import sys; print(sys.prefix)')"
rlgpu_site="${rlgpu_prefix}/lib/python3.7/site-packages"
cuda_libs="${rlgpu_site}/nvidia/cublas/lib:${rlgpu_site}/nvidia/cudnn/lib:${rlgpu_site}/nvidia/cuda_runtime/lib:${rlgpu_site}/nvidia/curand/lib:${rlgpu_site}/nvidia/cufft/lib:${rlgpu_site}/torch/lib"
export LD_LIBRARY_PATH="${cuda_libs}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

exec conda run --no-capture-output -n rlgpu \
  python scripts/run_ur5e_sim2sim.py \
  --scene resources/assets/robots/ur5e_robotiq_2f85/ur5e_hover_gripper.xml \
  --model resources/models/ur5e_hover_gripper/policy.onnx \
  --metadata resources/models/ur5e_hover_gripper/policy.meta.yaml \
  --hover-config resources/config/ur5e_hover_gripper_sim2sim.yaml \
  --provider cuda \
  --steps 300 \
  --npz resources/trajectories/ur5e_hover_mujoco.npz \
  --csv resources/trajectories/ur5e_hover_mujoco.csv \
  --report resources/trajectories/ur5e_hover_alignment.json \
  "$@"
