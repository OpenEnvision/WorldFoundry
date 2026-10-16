---
name: worldfoundry-repository-maintenance
description: >
  Use this skill when cleaning obsolete WorldFoundry files, organizing tests,
  updating bilingual docs or repository policy, reviewing source packaging and
  licenses, self-reviewing a scoped change, or preparing a PR or release.
  Guides scoped maintenance, generated documentation, contribution checks,
  and release evidence.
license: Apache-2.0
compatibility: >
  Requires a WorldFoundry checkout, Git, and development tools.
  Fumadocs changes also require Node.js and npm.
---

# Maintain the WorldFoundry Repository

Run commands from the WorldFoundry repository root. Follow
[repository guidance](../../AGENTS.md) and
[contribution rules](../../CONTRIBUTING.md). Read the relevant
[maintainer guide](../../docs/fumadocs/content/docs/maintainers/contributing.mdx)
sections when changing documentation generation or release workflows.

## Preserve Ownership and Scope

Inspect the working tree before editing and preserve unrelated changes. Use the
unified GPU development environment. Identify each retained runtime file's
caller, configured target, checkpoint contract, or public API role before
removing it; an unused-import report alone is not runtime evidence.

Keep synthesis infer-only and reuse canonical base-model components. Remove
abandoned private implementations with their obsolete tests and entrypoints,
while preserving failures that expose supported integration defects.

## Maintain the Published Surfaces

- Keep `README.md` at the repository root. Human operational documentation
  belongs in paired English/Chinese Fumadocs pages; task-specific agent
  workflows belong in `skills/<name>/SKILL.md`.
- Organize tests by owning module under `tests/`. Use behavioral names and the
  `unit` or `gpu` markers defined by CONTRIBUTING.md. Keep download-, API-, and
  checkpoint-heavy manual scripts under `tests/manual/` or `tests/stream/`.
  Add automated regressions to the relevant Make/CI selection.
- Keep generated recipe/API metadata consistent with its source. Use the
  existing generators when the task changes those sources; preserve authored
  pages and unrelated generated-file edits.
- Preserve upstream copyright and license files. Record provenance in
  `THIRD-PARTY-NOTICES` and the owning catalog. Check source and wheel inclusion
  with the existing packaging gate; checkpoints, demo media, credentials,
  caches, and generated run artifacts stay outside Git.
- Keep repository skills concise, indexed in `skills/index.md`, and linked to
  canonical docs. Update their workflows when an owning command or contract
  changes; local client configuration remains opt-in.

## Validate the Changed Surface

Choose the relevant checks, rather than running unrelated expensive stages:

```bash
pre-commit run
make lint
make test-infer
make cli-entrypoint-check
make packaging-check
```

For maintained documentation changes, use `bash scripts/docs/build.sh` and
`make docs-check`. When model manifests or homepage recipes change, refresh
with `npm --prefix docs/fumadocs run models:generate`, then verify with
`npm --prefix docs/fumadocs run models:check`.

Inference behavior changes also require
[inference validation](../worldfoundry-inference-validation/SKILL.md).
Weight-free checks cannot replace CUDA or real-checkpoint evidence.

## Review a Scoped Change

For a review request, report findings without editing unless fixes are also
requested. Inspect the intended committed diff and relevant staged, unstaged,
and untracked changes against current CONTRIBUTING.md and owning contracts.
For each actual issue, give `path:line`, observed impact, and the rule or call
path that establishes it; distinguish blockers from non-blocking suggestions.
Trace runtime, registry, and checkpoint callers before declaring code dead.
State plainly when the scoped change is correct.

## Prepare the PR

Use the [PR template](../../.github/PULL_REQUEST_TEMPLATE.md). Record scoped
changes, executed checks, resource requirements, artifacts, and remaining
blockers. Sign off each contribution commit with `git commit --signoff`, then
check the scoped diff with `git diff --check`, and run applicable Ruff and ty
checks for changed Python files. Follow CODEOWNERS review and squash-merge
conventions from the maintainer guide.
