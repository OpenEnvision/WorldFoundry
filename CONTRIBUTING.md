# Contributing

Thanks for helping improve WorldFoundry. Keep changes scoped, document public
behavior, and prefer manifests, task YAML, runtime profiles, and public CLI
commands over private scripts.

This file owns the repository's code and pull-request conventions. The
[detailed maintainer guide](docs/fumadocs/content/docs/maintainers/contributing.mdx)
covers architecture, generated docs, and release evidence. Participants follow
the [Code of Conduct](CODE_OF_CONDUCT.md). Report potential vulnerabilities
privately using the [Security Policy](SECURITY.md), rather than public issues.

## Agent Skills

Task-specific workflows live in [skills/index.md](skills/index.md), using the
vendor-neutral `skills/<name>/SKILL.md` format. They complement these rules and
the maintained architecture and validation docs. Client discovery is opt-in;
see the [maintainer guide](docs/fumadocs/content/docs/maintainers/contributing.mdx#agent-skills)
for local setup. Keep skills linked to current contracts when changing a workflow.

## Development Environment

WorldFoundry is GPU-first. Use the same CUDA environment for development,
inference, Studio, and evaluation. Start with Linux, an NVIDIA GPU, and a driver
supporting CUDA 12.8; individual model pages specify GPU architecture and VRAM.

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/OpenEnvision/WorldFoundry.git
cd WorldFoundry
bash scripts/setup/bootstrap_worldfoundry.sh
source tmp/worldfoundry_unified_env.sh
conda activate "${WORLDFOUNDRY_UNIFIED_ENV_PREFIX}"
python -m pip install -e ".[dev,test]"
pre-commit install
```

Model demo videos are served from GitHub CDN; docs development does not need
`git lfs pull`. Checkpoints, datasets, caches, credentials, and generated run
artifacts stay outside git. Use a model-specific environment only for a
documented ABI or simulator conflict.

## Coding Conventions

- Python 3.10+. Annotate new functions, methods, and public request/result contracts.
- Ruff owns formatting and import order; ty checks Python types. Run pre-commit
  on staged files. The `Python quality` PR check validates changed Python files
  with the repository's configured upstream-code exclusions.
- Prefer small, named functions and existing helpers over duplicated code or new
  abstractions. Comments explain ownership, constraints, or why a choice matters.
- Give public APIs concise docstrings describing inputs, outputs, and relevant
  exceptions. Keep shapes, units, camera conventions, and device assumptions explicit.
- Keep catalog discovery, CLI help, and package initializers lazy. Importing a
  control-plane module must not load checkpoints or initialize CUDA.
- Validate inputs before expensive model loading. Report failures with the model
  ID and relevant stage; release workers, streams, and temporary resources on exit.
- Put focused regressions under `tests/`, mirroring the affected module. Mark
  weight-free unit tests with `unit` and CUDA-dependent tests with `gpu`.
  Include each automated regression in the relevant Make/CI test selection.
  Manual inference and credential/download-heavy scripts belong in `tests/manual/`
  or `tests/stream/`, outside automatic pytest collection.
- Preserve upstream copyright and license notices. New first-party Python files
  carry SPDX copyright and `Apache-2.0` license headers; record adapted code's
  source and applicable terms in the catalog or central third-party notices.

## Developer Certificate Of Origin

Contributions are accepted under the [Apache License 2.0](LICENSE), subject to
retained upstream terms. Every contribution commit must have a
[Developer Certificate of Origin](https://developercertificate.org/) sign-off
certifying the contributor's right to submit the work.

Use `[type]Description` for commit subjects and PR titles, with no space after
the closing bracket. Use a meaningful type such as `fix`, `feat`, `refactor`,
`test`, `docs`, or `chore`, followed by an English imperative description.

```bash
git commit --signoff -m "[fix]Correct model session cleanup"
```

Git adds `Signed-off-by: Your Name <your@email.com>` using your configured name
and email. The `DCO` PR check validates every commit introduced by the PR.

## Pull Requests And Review

1. Fork the repository and create a focused branch from `main`.
2. Keep the change to one pipeline, benchmark path, public surface, or docs area.
   Discuss new integrations and substantial architecture changes in an issue first.
3. Add the relevant regression and update paired English/Chinese docs when public
   behavior changes. Declare GPU, checkpoint, API, simulator, and gated-data needs.
4. Stage changes and run `pre-commit run`, `make lint`, and `make test-infer` in
   the GPU development environment. Run the affected CUDA/real-checkpoint gates
   for inference changes and `bash scripts/docs/build.sh` for documentation changes.
5. Sign off each commit and run the committed-change checks:

   ```bash
   make pr-check BASE_REF="$(git merge-base origin/main HEAD)"
   ```

6. Open a PR using the [PR template](.github/PULL_REQUEST_TEMPLATE.md). Include
   exact commands, results, resource requirements, and relevant artifacts.

[CODEOWNERS](.github/CODEOWNERS) defines review responsibility across the
repository. Merging requires passing applicable CI checks and at least one
approving maintainer review, including the affected code owner. Use squash
merge with a descriptive PR title and retain contributor DCO trailers in the
squash commit. Resolve review conversations before merging.

Repository administrators configure `main` rulesets to require pull requests,
code-owner approval, resolved conversations, and the `DCO`, `Python quality`,
`public-surface`, `cpu-tests`, `inference-tensors`, `geometry-impact`,
`inference-replay`, and `packaging-license-gate` checks. Documentation changes
also need the applicable `docs-build` result. Hosted CPU checks are fast
weight-free validation; inference changes additionally require the existing
trusted GPU evidence gates.

## Change Checklists

### Add Or Update A Model

- Update `worldfoundry/data/models/catalog/` and, when needed, `worldfoundry/data/models/runtime/profiles/`.
- Declare aliases, source status, integration status, auth, checkpoint refs, license, and blockers.
- Add or reuse a public adapter target and a minimal validation command before claiming `integrated`.
- Keep GPU, checkpoint, authentication, and download requirements explicit in the model's quickstart.
- Verify with `worldfoundry-eval zoo models --json` and the smallest documented model smoke run.

### Add Or Update A Benchmark

- Update `worldfoundry/data/benchmarks/catalog/` and task YAML under `worldfoundry/data/benchmarks/tasks/external/` when applicable.
- Declare dataset refs, auth, unsafe/gated data, official repo, simulator, metrics, and blockers.
- Add a contract adapter or normalizer before claiming `contract_ready`.
- Add official runtime evidence before claiming leaderboard validity.
- Verify with `worldfoundry-eval zoo benchmarks --json` and the documented readiness checks.

### Add Or Update A Metric

- Add deterministic validation examples for metric correctness.
- Document whether higher is better, output units, primary metric status, and leaderboard mapping.
- Ensure scorecards expose the metric under stable keys.

### Add Or Update Docs

- Use the GPU bootstrap as the default first-run path; state hardware, asset, and authentication requirements.
- Link complex environment, official runtime, and API paths to reference docs.
- Run the docs build and docs checks.

### Add Or Update A CLI Argument

- Update `docs/fumadocs/content/docs/reference/cli.mdx` when public behavior changes.
- Preserve JSON output compatibility or document the breaking change.
- Run `make docs-check` and document help, error handling, and no-heavy-import behavior.

## Core Responsibility Boundaries

Keep `worldfoundry/core` model-neutral. Place camera geometry in `core.geometry`,
inference execution and process-local caches in `core.execution`, logging and timing
in `core.observability`, and media codecs and streaming processing in `core.media`.
Use the existing `nn`, `attention`, `io`, `model_loading/checkpoints`, and other domain packages
for their respective primitives. The root is reserved for the lazy public facade
and cross-cutting contracts, registries, and input normalization.

Import implementation modules directly or use the existing public facade; do not
add forwarding modules at abandoned paths. Package initializers must preserve
lightweight control-plane imports. Model-specific orchestration and components
belong in `synthesis` and `base_models`, respectively.

## Synthesis Is Infer-Only

`worldfoundry/synthesis/**` packages upstream **inference** runtimes only. Do not add:

- shell launch scripts: framework adapters call Python entrypoints or component APIs directly
- unused placeholder modules or leaf packages left after moving implementations; import canonical
  components directly instead of keeping unused re-export shells in `synthesis`
- copies of networks, VAEs, text encoders, schedulers, or third-party models already owned by
  `worldfoundry/base_models/`; reuse those implementations, including
  `worldfoundry/base_models/diffusion_model/`. Keep checkpoint-specific differences as small
  variants there, and keep model orchestration and input/output adaptation in `synthesis`.

- training launchers, dataset builders, finetune scripts, or benchmark-only tooling
- demo media, generated outputs, checkpoints, or downloaded datasets
- notebooks, README files, or other documentation inside the synthesis tree.
  The only `README.md` allowed in this repository is the root file; put
  operational notes in fumadocs instead.

Keep local demo assets in `testcase/` and reference them through explicit input paths; they are not part of the public source distribution.

For each retained file or helper, identify its inference caller, configured target, checkpoint-loading
contract, or public API role. After migrations, remove abandoned entrypoints and their private helpers
together. Check lazy imports, subprocess workers, registries, and serialized checkpoint classes before
deleting code; an unused-import report alone is not evidence that a runtime component is unnecessary.

Before release or large runtime imports, run the public quality gates:

```bash
make lint
make docs-check
```

The public distribution includes inference, evaluation, and the source tests.
Model weights, test media and generated replay evidence stay outside git.
Run the fast, weight-free gate inside the active GPU development environment:

```bash
make test-infer
make cli-entrypoint-check
make cli-check
make packaging-check
```

When model manifests or homepage recipes change, refresh and verify the
checked-in recipe data:

```bash
npm --prefix docs/fumadocs run models:generate
npm --prefix docs/fumadocs run models:check
```


For inference changes, replay the affected real-checkpoint cases and pass
`make test-geometry` against an accepted reference before publishing. See the
[validation guide](docs/fumadocs/content/docs/reference/validation.mdx) for fixed
assets, environments, numerical comparison, and source-hash verification.
