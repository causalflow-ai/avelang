#!/usr/bin/env bash
set -euo pipefail

token_file="${RUNNER_TOKEN_FILE:-/run/secrets/runner-token}"

if [[ ! -r "${token_file}" ]]; then
  echo "Runner registration token is missing: ${token_file}" >&2
  exit 1
fi

until docker info >/dev/null 2>&1; do
  echo "Waiting for the DinD daemon..."
  sleep 2
done

cleanup() {
  ./config.sh remove --unattended --token "$(tr -d '\r\n' < "${token_file}")" || true
}
trap cleanup EXIT INT TERM

./config.sh \
  --unattended \
  --replace \
  --url "${RUNNER_URL:?RUNNER_URL must be set}" \
  --token "$(tr -d '\r\n' < "${token_file}")" \
  --name "${RUNNER_NAME:-mi300-dind}" \
  --labels "${RUNNER_LABELS:-mi300,rocm72,dind}" \
  --work /home/runner/_work

./run.sh
