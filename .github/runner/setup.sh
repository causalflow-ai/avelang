#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

if [[ -z "${RUNNER_TOKEN:-}" ]]; then
  echo "Pass a fresh repository runner registration token in RUNNER_TOKEN." >&2
  exit 1
fi

docker compose --file compose.yml build runner
docker compose --file compose.yml up --detach dind

for attempt in $(seq 1 60); do
  if docker compose --file compose.yml exec --no-TTY dind docker info >/dev/null 2>&1; then
    break
  fi
  if [[ "${attempt}" -eq 60 ]]; then
    echo "DinD daemon did not become ready within 120 seconds." >&2
    docker compose --file compose.yml logs dind >&2
    exit 1
  fi
  sleep 2
done

docker compose --file compose.yml exec dind mkdir -p /runner-context
docker compose --file compose.yml cp Dockerfile.avelang-ci dind:/runner-context/Dockerfile.avelang-ci
docker compose --file compose.yml exec dind \
  docker build \
    --file /runner-context/Dockerfile.avelang-ci \
    --tag avelang-ci:rocm7.2.2 \
    /runner-context
docker compose --file compose.yml up --detach --force-recreate runner

for attempt in $(seq 1 60); do
  runner_logs="$(docker compose --file compose.yml logs --no-color runner 2>&1 || true)"
  if [[ "${runner_logs}" == *"Listening for Jobs"* ]]; then
    echo "GitHub Actions runner is listening for jobs."
    exit 0
  fi
  if [[ "${runner_logs}" == *"Runner registration token is missing"* ]] \
      || [[ "${runner_logs}" == *"Response status code does not indicate success"* ]]; then
    echo "GitHub Actions runner registration failed." >&2
    printf '%s\n' "${runner_logs}" >&2
    exit 1
  fi
  if [[ "${attempt}" -eq 60 ]]; then
    echo "GitHub Actions runner did not become ready within 120 seconds." >&2
    printf '%s\n' "${runner_logs}" >&2
    exit 1
  fi
  sleep 2
done
