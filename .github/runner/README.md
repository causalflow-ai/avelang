# MI300 self-hosted runner

This deployment keeps the GitHub runner away from the host Docker socket. The
unprivileged runner talks to a dedicated Docker-in-Docker daemon on a private
Compose network. The daemon port is not published to the host. Only that daemon
is privileged, and only `/dev/kfd`, `/dev/dri`, and the runner work volume are
forwarded into it. Test containers drop all Linux capabilities and enable
`no-new-privileges`.

The DinD boundary reduces host exposure but is not a VM boundary. A privileged
DinD daemon and GPU device access still carry kernel-level risk. The workflow is
therefore limited to trusted `master` pushes and manual dispatches. Do not enable
automatic execution for fork pull requests.

## Triggering CI

Every push to upstream `master` runs MI300 CI automatically.

Upstream collaborators can validate a feature branch before merging:

1. Push the branch to the upstream repository.
2. Open `Actions > MI300 ROCm CI > Run workflow`.
3. Select the upstream feature branch and click `Run workflow`.

Manual dispatch intentionally supports upstream branches only. Do not accept an
arbitrary repository URL or commit SHA as workflow input: that would allow
untrusted fork code to execute on the self-hosted GPU machine. Fork pull requests
require maintainer review before their code is copied to an upstream branch and
manually dispatched.

## Bootstrap

Run these commands on the MI300 host, where Docker Compose and the AMD device
nodes are available:

```bash
cd .github/runner
printf '%s' '<fresh repository runner registration token>' > runner-token
./setup.sh
rm runner-token
```

Generate the short-lived token from the upstream repository's
`Settings > Actions > Runners > New self-hosted runner` page immediately before
bootstrap. Never commit it. The runner image uses GitHub Actions runner 2.334.0.

The first setup builds `avelang-ci:rocm7.2.2`. This is intentionally expensive:
it compiles the pinned ROCm LLVM commit used by the current machine environment.
Subsequent runs reuse that local DinD image.

## Operations

Inspect runner state:

```bash
docker compose --file .github/runner/compose.yml ps
docker compose --file .github/runner/compose.yml logs --follow runner
```

Stop the deployment:

```bash
docker compose --file .github/runner/compose.yml down
```

To rebuild the CI image after changing `Dockerfile.avelang-ci`, copy the runner
directory into the DinD daemon and rebuild:

```bash
docker compose --file .github/runner/compose.yml exec dind mkdir -p /runner-context
docker compose --file .github/runner/compose.yml cp \
  .github/runner/Dockerfile.avelang-ci dind:/runner-context/Dockerfile.avelang-ci
docker compose --file .github/runner/compose.yml exec dind \
  docker build --file /runner-context/Dockerfile.avelang-ci \
  --tag avelang-ci:rocm7.2.2 /runner-context
```
