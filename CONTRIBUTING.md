# Contributing

Thanks for helping improve WorldFoundry. Keep changes scoped, document public
behavior, and prefer manifests, task YAML, runtime profiles, and public CLI
commands over private scripts.

The detailed maintainer guide lives in
[`docs/fumadocs/content/docs/maintainers/contributing.mdx`](docs/fumadocs/content/docs/maintainers/contributing.mdx).

Model demo videos are served from GitHub CDN; docs development does not need `git lfs pull`.

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/OpenEnvision/WorldFoundry.git
```

## Before Opening A Pull Request

- Keep the change scoped to one pipeline, benchmark path, CLI surface, or docs area.
- Do not commit checkpoints, generated videos, local caches, API keys, tokens, or credentials.
- State any GPU, API quota, checkpoint, simulator, official repo, or gated-data requirement.
- Run the public validation commands relevant to the changed surface.
- Run `bash scripts/docs/build.sh --skip-bootstrap` when docs or README links change.
- Include scorecard, preflight, or validation evidence for any readiness claim.

## Change Checklists

### Add Or Update A Model

- Update `worldfoundry/data/models/catalog/` and, when needed, `worldfoundry/data/models/runtime/profiles/`.
- Declare aliases, source status, integration status, auth, checkpoint refs, license, and blockers.
- Add or reuse a public adapter target and a minimal validation command before claiming `integrated`.
- Keep API/GPU/checkpoint requirements out of quickstart defaults.
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

- Keep first-run docs CPU-only, no-download, no-token, and no-GPU.
- Link complex environment, official runtime, and API paths to reference docs.
- Run the docs build and docs checks.

### Add Or Update A CLI Argument

- Update `docs/fumadocs/content/docs/reference/cli.mdx` when public behavior changes.
- Preserve JSON output compatibility or document the breaking change.
- Run `make docs-check` and document help, error handling, and no-heavy-import behavior.

## Core Responsibility Boundaries

Keep `worldfoundry/core` model-neutral. Place camera geometry in `core.geometry`,
inference execution and process-local caches in `core.execution`, logging and timing
in `core.observability`, and streaming video post-processing in `core.video`.
Use the existing `nn`, `attention`, `io`, `checkpoint`, and other domain packages
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

The public distribution includes inference and evaluation. Training sources and
local test suites stay in the development repository. For a CPU smoke run:

```bash
python -m pip install -e .
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
