---
name: Model integration request
about: Propose adding or updating a world-model method, checkpoint family, or pipeline backend
title: "[Model Integration]: "
labels: ["model-integration", "needs-triage"]
assignees: ""
---

## Summary

<!-- Required: what model should WorldFoundry support, and what user workflow would it unlock? -->

## Model Details

<!-- Required: provide canonical references and implementation details. -->

- Model name:
- Paper / project / repository:
- License:
- Maintainer or upstream contact:
- Expected task family:
- Proposed WorldFoundry pipeline entrypoint:

## Checkpoint / API Key Needs

<!-- Required: list checkpoint size, download source, cache expectations, API provider, and env vars. Do not paste secrets. -->

- Checkpoints / weights:
- Required environment variables:
- API providers:
- Local cache assumptions:
- Redistribution constraints:

## Runtime Profile

<!-- Required: describe install and hardware requirements. -->

- Suggested profile:
- Python / CUDA constraints:
- GPU memory estimate:
- Minimum GPU / architecture and tested device:
- Known upstream dependency risks:

## Benchmark And Validation Plan

<!-- Required: identify how maintainers should validate the integration. -->

| Area | Proposed command or evidence | Required? |
| --- | --- | --- |
| Real inference validation |  | Yes / No |
| Streaming or multi-turn path |  | Yes / No |
| Benchmark runner / metric path |  | Yes / No |
| Deterministic regression |  | Yes / No |
| Docs build |  | Yes / No |

## Preflight / Scorecard Evidence

<!-- Required if available: provide command output paths. Use "Not available yet" for proposals. -->

- `worldfoundry-eval prepare` or env-check report:
- Runner demo or validation scorecard:
- Expected `leaderboard_valid` value:
- Known blockers before integration:

## Sample Artifact Evidence

<!-- Required if any output exists: attach examples from the upstream model or local experiments. -->

- Artifact path or link:
- What it demonstrates:
- Known limitations:

## Integration Scope

<!-- Required: mark all expected work areas. -->

- [ ] New `worldfoundry/pipelines/` entrypoint
- [ ] New or updated runtime backend
- [ ] New install profile or setup script
- [ ] New fixtures under `data/test_cases/`
- [ ] New docs
- [ ] New tests or focused validation scripts

## Additional Context

<!-- Optional: upstream issues, forks, compatibility notes, or prior experiments. -->
