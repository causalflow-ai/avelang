#!/usr/bin/env bash
set -euo pipefail

token="${RUNNER_TOKEN:?RUNNER_TOKEN must be set}"

until docker info >/dev/null 2>&1; do
  echo "Waiting for the DinD daemon..."
  sleep 2
done

cleanup() {
  ./config.sh remove --unattended --token "${token}" || true
}
trap cleanup EXIT INT TERM

./config.sh \
  --unattended \
  --replace \
  --url "${RUNNER_URL:?RUNNER_URL must be set}" \
  --token "${token}" \
  --name "${RUNNER_NAME:-mi300-dind}" \
  --labels "${RUNNER_LABELS:-mi300,rocm72,dind}" \
  --work /home/runner/_work

./run.sh
