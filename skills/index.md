# WorldFoundry Agent Skills

Task-specific workflows use the [Agent Skills format](https://agentskills.io/specification).
Each directory contains a `SKILL.md` with `name`, `description`, and Markdown
instructions. Skills are first-party source documentation under
[Apache-2.0](../LICENSE), included in source distributions.

## Available Skills

| Skill | Use when |
| --- | --- |
| [Model integration](worldfoundry-model-integration/SKILL.md) | Adding or migrating a model, checkpoint conversion, component parity, recipe, binding, or runtime profile |
| [Benchmark integration](worldfoundry-benchmark-integration/SKILL.md) | Adding benchmark manifests, task materialization, official runners, metrics, or scorecard normalization |
| [Inference optimization](worldfoundry-inference-optimization/SKILL.md) | Profiling and changing attention, compilation, warmup, caches, offload, decoding, or streaming latency |
| [Inference validation](worldfoundry-inference-validation/SKILL.md) | Comparing component numerics, locating first divergence, running checkpoint replays, or reviewing inference evidence |
| [Repository maintenance](worldfoundry-repository-maintenance/SKILL.md) | Updating docs, tests, packaging, licensing, scoped self-review, or release readiness |

## Using Skills

Load the relevant `SKILL.md` explicitly, or configure your client to discover
this directory. Client setup and selective linking are documented in the
[maintainer guide](https://openenvision.github.io/WorldFoundry/docs/maintainers/contributing#agent-skills).
Local client configuration is opt-in and remains outside Git.

[AGENTS.md](../AGENTS.md) and [CONTRIBUTING.md](../CONTRIBUTING.md) own repository
rules. Skills describe task workflows; the linked Fumadocs pages own detailed
architecture, environment setup, and validation contracts.

## Workflow References

Model integration and numerical validation draw on the component-first parity
and checkpoint-conversion workflows in
[Diffusers](https://github.com/huggingface/diffusers) and
[FastVideo](https://github.com/hao-ai-lab/FastVideo). Optimization draws on the
executed-backend, profiling, and lifecycle workflows in
[SGLang diffusion](https://github.com/sgl-project/sglang),
[LightX2V](https://github.com/ModelTC/LightX2V), and
[vLLM-Omni](https://github.com/vllm-project/vllm-omni). Scoped self-review follows
the report-first approach used by Diffusers.

WorldFoundry owns the implementation boundaries, commands, and acceptance
contracts. Detailed parity and execution-lifecycle references load only for
the relevant task; basic model integration, performance, and serving
reliability remain separate evidence scopes.

## Adding a Skill

Follow the official [specification](https://agentskills.io/specification) and
[authoring best practices](https://agentskills.io/skill-creation/best-practices).
Use a lowercase, hyphenated `worldfoundry-` name matching its directory
(1–64 characters). `name` and `description` are the required frontmatter fields;
descriptions must be non-empty and at most 1024 characters. State the user's
task and when to activate the skill, using "Use this skill when…".

Repository skills declare `license: Apache-2.0`. Use the optional `compatibility`
field for checkout and environment requirements. Keep each `SKILL.md` below
500 lines. Load detailed references only when relevant; resolve file references
relative to the skill directory and state the working directory for commands.
Update this index when adding or removing a skill.

Validate formatting with the official `skills-ref validate <skill-directory>`
reference tool and run `python -m pytest -q tests/core/test_agent_skills.py`
from the repository root. Evaluate activation and task outcomes separately,
following the [maintainer guide](https://openenvision.github.io/WorldFoundry/docs/maintainers/contributing#agent-skills).
