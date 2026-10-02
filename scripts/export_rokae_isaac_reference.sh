#!/usr/bin/env bash
set -euo pipefail

rlgpu_prefix="$(conda run -n rlgpu python -c 'import sys; print(sys.prefix)')"
rlgpu_site="${rlgpu_prefix}/lib/python3.7/site-packages"
cuda_libs="${rlgpu_site}/nvidia/cublas/lib:${rlgpu_site}/nvidia/cudnn/lib:${rlgpu_site}/nvidia/cuda_runtime/lib:${rlgpu_site}/nvidia/curand/lib:${rlgpu_site}/nvidia/cufft/lib:${rlgpu_site}/torch/lib"
export LD_LIBRARY_PATH="${cuda_libs}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

exec conda run --no-capture-output -n rlgpu \
  python scripts/export_rokae_isaac_reference.py "$@"
