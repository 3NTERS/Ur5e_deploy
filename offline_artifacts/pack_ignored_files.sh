#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIRECTORY="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd -- "${SCRIPT_DIRECTORY}/.." && pwd)"
ARCHIVE_DIRECTORY="${SCRIPT_DIRECTORY}/archives"
CHECKSUM_FILE="${SCRIPT_DIRECTORY}/SHA256SUMS"
TEMP_DIRECTORY="$(mktemp -d "${TMPDIR:-/tmp}/ur5e-offline-pack.XXXXXX")"
trap 'rm -rf -- "${TEMP_DIRECTORY}"' EXIT

RESOURCE_GROUPS=(models calibration trajectories training predictions wheels)
mkdir -p "${TEMP_DIRECTORY}/archives" "${ARCHIVE_DIRECTORY}"

for group in "${RESOURCE_GROUPS[@]}"; do
  source_directory="resources/${group}"
  file_list="${TEMP_DIRECTORY}/${group}.files"
  archive_name="resources_${group}_ignored.tar.gz"

  git -C "${REPOSITORY_ROOT}" ls-files \
    --others --ignored --exclude-standard -z -- "${source_directory}" \
    > "${file_list}"

  if [[ ! -s "${file_list}" ]]; then
    echo "skip ${source_directory}: no ignored files"
    continue
  fi

  tar --create --gzip \
    --file="${TEMP_DIRECTORY}/archives/${archive_name}" \
    --directory="${REPOSITORY_ROOT}" \
    --null --files-from="${file_list}"
  echo "packed ${source_directory} -> archives/${archive_name}"
done

if ! compgen -G "${TEMP_DIRECTORY}/archives/*.tar.gz" > /dev/null; then
  echo "No ignored resource files were found; nothing was packaged." >&2
  exit 1
fi

(
  cd "${TEMP_DIRECTORY}"
  sha256sum archives/*.tar.gz > SHA256SUMS
)

for archive in "${TEMP_DIRECTORY}"/archives/*.tar.gz; do
  mv -- "${archive}" "${ARCHIVE_DIRECTORY}/"
done
mv -- "${TEMP_DIRECTORY}/SHA256SUMS" "${CHECKSUM_FILE}"

echo "archives=${ARCHIVE_DIRECTORY}"
echo "checksums=${CHECKSUM_FILE}"
