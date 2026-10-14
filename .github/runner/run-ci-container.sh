#!/usr/bin/env bash
set -euo pipefail

source_dir="${1:-${GITHUB_WORKSPACE:-$(pwd)}}"
ci_image="${AVELANG_CI_IMAGE:-avelang-ci:rocm7.2.2}"

docker image inspect "${ci_image}" >/dev/null

docker run --rm \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --cap-drop=ALL \
  --security-opt=no-new-privileges:true \
  --pids-limit=4096 \
  --shm-size=16g \
  --volume "${source_dir}:/workspace/avelang:ro" \
  "${ci_image}" \
  bash -lc '
    cp -a /workspace/avelang /tmp/avelang
    cd /tmp/avelang
    bash .github/runner/ci.sh
  '
