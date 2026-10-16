## Summary

Describe the change, its user-visible behavior, and the issue it resolves.

Fixes #

## Scope

- Affected model / benchmark / runtime profile:
- Public entrypoint or output schema:
- Compatibility and resource impact:

## Runtime And Assets

For inference or evaluation changes, include the environment and assets needed
for reproduction. Keep credentials out of the PR.

- GPU / VRAM, driver, Python, Torch and CUDA:
- Checkpoint / dataset source and revision:
- Required environment variables / API provider / simulator:

## Validation

List the commands relevant to this change. Record failed or unexecuted stages
with their reasons.

| Command | Result | Artifact / Notes |
| --- | --- | --- |
| | | |

For model execution changes, link the affected real-checkpoint replay and a
representative output. For benchmark or readiness changes, link the scorecard
and explain any change to `leaderboard_valid`. For docs changes, include the
build result and affected English/Chinese pages.

## Known Limitations

## Contribution Requirements

- [ ] The PR addresses one focused change and includes a regression test where applicable.
- [ ] Public behavior and paired English/Chinese docs are updated.
- [ ] Upstream attribution and license records are retained where applicable.
- [ ] No credentials, checkpoints, caches, or generated run artifacts are committed.
- [ ] Every contribution commit has a DCO sign-off (`git commit --signoff`).
