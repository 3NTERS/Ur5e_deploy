#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIRECTORY="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_TARGET="$(cd -- "${SCRIPT_DIRECTORY}/.." && pwd)"
CHECKSUM_FILE="${SCRIPT_DIRECTORY}/SHA256SUMS"
OVERWRITE=false
TARGET_ROOT=""

usage() {
  echo "Usage: $0 [--overwrite] [repository-root]"
  echo ""
  echo "Without --overwrite, restoration stops before extracting if any target file exists."
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --overwrite)
      OVERWRITE=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    -* )
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      if [[ -n "${TARGET_ROOT}" ]]; then
        echo "Only one repository root may be specified." >&2
        exit 2
      fi
      TARGET_ROOT="$1"
      shift
      ;;
  esac
done

TARGET_ROOT="${TARGET_ROOT:-${DEFAULT_TARGET}}"
TARGET_ROOT="$(cd -- "${TARGET_ROOT}" && pwd)"

if [[ ! -f "${TARGET_ROOT}/README.md" || ! -d "${TARGET_ROOT}/resources" ]]; then
  echo "Target does not look like the Ur5e_deploy repository: ${TARGET_ROOT}" >&2
  exit 1
fi
if [[ ! -f "${CHECKSUM_FILE}" ]]; then
  echo "Missing checksum manifest: ${CHECKSUM_FILE}" >&2
  exit 1
fi

(
  cd "${SCRIPT_DIRECTORY}"
  sha256sum --check SHA256SUMS
)

mapfile -t ARCHIVES < <(awk '{print $2}' "${CHECKSUM_FILE}")
if [[ ${#ARCHIVES[@]} -eq 0 ]]; then
  echo "Checksum manifest contains no archives." >&2
  exit 1
fi

conflicts=0
for relative_archive in "${ARCHIVES[@]}"; do
  archive="${SCRIPT_DIRECTORY}/${relative_archive}"
  if [[ ! -f "${archive}" ]]; then
    echo "Missing archive: ${archive}" >&2
    exit 1
  fi
  while IFS= read -r entry; do
    case "${entry}" in
      resources/*) ;;
      *)
        echo "Unsafe archive entry outside resources/: ${entry}" >&2
        exit 1
        ;;
    esac
    case "/${entry}/" in
      */../*)
        echo "Unsafe parent traversal in archive: ${entry}" >&2
        exit 1
        ;;
    esac
    if [[ "${OVERWRITE}" == false && -e "${TARGET_ROOT}/${entry}" ]]; then
      echo "Existing target: ${entry}" >&2
      conflicts=$((conflicts + 1))
    fi
  done < <(tar --list --gzip --file="${archive}")
done

if [[ ${conflicts} -gt 0 ]]; then
  echo "Restore cancelled: ${conflicts} target files already exist." >&2
  echo "Review them first, or rerun with --overwrite." >&2
  exit 1
fi

for relative_archive in "${ARCHIVES[@]}"; do
  archive="${SCRIPT_DIRECTORY}/${relative_archive}"
  if [[ "${OVERWRITE}" == true ]]; then
    tar --extract --gzip --overwrite --file="${archive}" --directory="${TARGET_ROOT}"
  else
    tar --extract --gzip --file="${archive}" --directory="${TARGET_ROOT}"
  fi
  echo "restored ${relative_archive}"
done

echo "Restore complete: ${TARGET_ROOT}"
