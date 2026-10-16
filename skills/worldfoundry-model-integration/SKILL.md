---
name: worldfoundry-model-integration
description: >
  Use this skill when adding or migrating a model into WorldFoundry, adding a
  model variant, porting checkpoint loading, or making a model available through
  CLI, Studio, or evaluation. Guides canonical component reuse, checkpoint
  conversion, component parity, input operators, bindings, and runtime evidence.
license: Apache-2.0
compatibility: >
  Requires a WorldFoundry checkout and its prepared Python environment.
  Hardware, credentials, and assets depend on the selected runtime profile.
---

# Integrate a WorldFoundry Model

Run commands from the WorldFoundry repository root. Follow
[contribution rules](../../CONTRIBUTING.md). Read
[model runtime architecture](../../docs/fumadocs/content/docs/maintainers/architecture/model-runtime.mdx)
when selecting pipeline/operator ownership, and
[native diffusion](../../docs/fumadocs/content/docs/maintainers/architecture/native-diffusion.mdx)
when changing diffusion components or recipes.

## Establish the Contract

Identify the upstream release and license, checkpoint layout, model family,
required inputs, expected artifacts, hardware requirements, and reference
inference command. Choose the closest existing in-tree integration by runtime
contract rather than by model name. Pin the reference revision and checkpoint
variant. Start with one reproducible task and its real input/output tensors;
extend to other tasks after that route passes. Prepare the unified GPU
environment from CONTRIBUTING.md; use a dedicated environment only for a
documented conflict.

## Establish Component Parity

Define deterministic comparisons for the required denoiser, encoders, latent
codec, and scheduler before composing the public pipeline. Include reused
components whose checkpoint or normalization semantics differ. Exercise the
same source weights and inputs on both sides, then cover WorldFoundry's public
loading path separately from direct component construction.

Use [inference validation](../worldfoundry-inference-validation/SKILL.md#compare-components-and-locate-numerical-drift)
for conversion checks and first-divergence diagnosis. Component parity must
execute real forwards and compare tensor values; shapes and successful loading
alone do not establish parity.

## Implement in the Owning Layer

1. Reuse canonical network, encoder, autoencoder, scheduler, and loading
   components. Local diffusion composition belongs in
   `worldfoundry/base_models/diffusion_model/recipes/`; checkpoint mapping and
   denoiser adaptation belong in `models/denoisers/`. Bind preprocessing and
   latent preparation to the existing encoder and initializer contracts. New
   shared sampling strategies belong in `runners/`, not a family pipeline.
   Generic attention, placement, IO, and distributed primitives belong in
   `worldfoundry/core/`.
2. Keep model-specific inference orchestration in `worldfoundry/synthesis/`.
   Upstream frameworks are migration references, not alternate diffusion
   runtime backends. Port the required inference slice and retain attribution.
3. Bind input validation and preprocessing through `worldfoundry/operators/`
   and the public entrypoint through `worldfoundry/pipelines/`. Preserve the
   shared request, result, and artifact contracts. Implement streaming only
   when the integration supports it.
4. Add the model ID and aliases to `worldfoundry/data/models/catalog/`, the
   route to `models/bindings/pipelines/`, and requirements to
   `models/runtime/profiles/` and `models/runtime/environments/`. Keep presets
   in `models/runtime/configs/`. Use the existing binding resolver.
5. Declare checkpoint roles, acquisition metadata, credentials, and blockers.
   Checkpoints and demo media stay outside Git; local demo inputs use explicit
   paths under `testcase/`. Import and discovery must remain weight-free.

## Verify the Public Route

Replace `<model-id>` with the declared catalog ID:

```bash
worldfoundry-eval zoo model-show --model-id <model-id> --include-manifest --json
worldfoundry-eval zoo model-download --model-id <model-id> --check-local --json
worldfoundry-eval run <model-id> --model-status
worldfoundry-eval run <model-id> --print-config --json
```

Add focused tests for resolution, loading options, supported inputs, request
forwarding, and observable output contracts through the resolved public loader.
Include component and checkpoint-mapping regressions in the relevant Make
selection; direct construction alone does not verify production loading.

Use [inference validation](../worldfoundry-inference-validation/SKILL.md) for
real-weight acceptance. A resolved route, mock, or plan is not generation
parity. Basic model integration and Studio deployment are separate acceptance
scopes: serving claims additionally need request isolation, cancellation, and
worker-reuse evidence for the advertised route. Keep readiness tied to the
specific tasks, variants, and hardware actually validated; record unexecuted
combinations as not tested in the evidence, using existing manifest statuses.

## Document the Integration

Maintain paired model pages under
`docs/fumadocs/content/docs/guides/supported-models/`. Record the public command,
input and artifact contracts, environment, and measured evidence. Refresh model
recipe data with `npm --prefix docs/fumadocs run models:generate` and verify it
with `npm --prefix docs/fumadocs run models:check` when manifests change.
