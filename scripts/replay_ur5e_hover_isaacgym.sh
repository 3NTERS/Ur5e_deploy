#!/usr/bin/env bash
# 仅用于在isaacgym中查看策略动作
set -euo pipefail

deploy_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(dirname "${deploy_dir}")"
cd "${repository_root}"

exec ./scripts/export_ur5e_isaac_reference.sh \
  --task Ur5eRobotiqHoverGripper \
  --model resources/models/ur5e_hover_gripper/policy.onnx \
  --provider cuda \
  --steps 300 \
  --viewer \
  --view-only \
  "$@"
