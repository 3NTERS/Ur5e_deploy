#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIRECTORY="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd -- "${SCRIPT_DIRECTORY}/.." && pwd)"
WHEEL_DIRECTORY="${1:-${REPOSITORY_ROOT}/resources/wheels/rlgpu-py37-linux-x86_64}"
PYPI_MIRROR_URL="${PYPI_MIRROR_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
DOWNLOAD_JOBS="${DOWNLOAD_JOBS:-6}"

mkdir -p "${WHEEL_DIRECTORY}"

download_one() {
  python -m pip download \
    --no-deps \
    --dest "${WHEEL_DIRECTORY}" \
    --index-url "${PYPI_MIRROR_URL}" \
    "$1"
}
export -f download_one
export WHEEL_DIRECTORY PYPI_MIRROR_URL

sed -e '/^[[:space:]]*#/d' -e '/^[[:space:]]*$/d' \
  "${REPOSITORY_ROOT}/requirements-hardware-download.txt" \
  | xargs -r -n 1 -P "${DOWNLOAD_JOBS}" bash -c 'download_one "$1"' _

echo "Hardware packages downloaded to ${WHEEL_DIRECTORY}"
echo "Install with:"
echo "python -m pip install --no-index --find-links ${WHEEL_DIRECTORY} -r ${REPOSITORY_ROOT}/requirements-hardware.txt"
