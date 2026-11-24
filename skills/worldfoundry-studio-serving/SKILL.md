---
name: worldfoundry-studio-serving
description: >
  Use this skill when making a model work in WorldFoundry Studio or the
  Workspace, adding streaming, interactive, or realtime support to a pipeline,
  or debugging Studio jobs: a model missing from the catalog, inputs that never
  reach the model, blank previews, broken INIT/STREAM/RESET, resident worker
  reuse, cancellation, GPU leases, timeouts, or realtime cadence, dropped
  frames, and latency. Covers catalog discovery, runtime drivers,
  pipeline-owned stream state, native realtime methods, declared video chunks,
  artifact materialization, worker lifecycle, and serving evidence. Basic
  model integration is worldfoundry-model-integration; kernel speedups are
  worldfoundry-inference-optimization.
license: Apache-2.0
compatibility: >
  Requires a WorldFoundry checkout with the unified environment. Realtime runs
  and replays need a CUDA GPU and local weights; browser replays also need
  Node.js and Playwright Chromium.
---

# Serve Models in WorldFoundry Studio

Studio is a catalog-driven shell over the same pipeline and artifact contracts
as the CLI and evaluation. It discovers entries, stages inputs, drives the
pipeline through a runtime driver, writes a `RunRecord`, and picks previews.
Model semantics stay in the pipeline and operator; sessions, workers, leases,
and transport stay in Studio. Run commands from the repository root with the
unified environment active.
[Studio internals](../../docs/fumadocs/content/docs/maintainers/architecture/studio-internals.mdx)
owns the file map and lifecycle; the [Studio guide](../../docs/fumadocs/content/docs/guides/studio.mdx)
owns launch flags and environment variables.

## Ownership

| Concern | Owner |
| --- | --- |
| Input validation, controls, camera and action parsing | The model's operator |
| Generation and interactive state (memory, caches, scenes) | The pipeline and its memory module |
| Catalog discovery: `discover_catalog()`, `find_entry()` | `worldfoundry.studio.inference.catalog` |
| Input staging, runtime drivers, `StudioManager`, `RunRecord`, previews | `worldfoundry/studio/inference/execution.py` |
| GPU admission, resident workers, cancellation, deadlines | `worldfoundry/studio/inference/dispatch.py` |
| Realtime input resampling, frame queues, WebRTC | `worldfoundry/studio/serving/realtime/` |
| Post-decode processing such as super-resolution | `worldfoundry/core/media/processing/postprocess.py` |

Studio inference modules never import the Workspace, serving, or
visualization packages; `tests/studio/test_module_boundaries.py` enforces it.

## Launch and reproduce

```bash
bash scripts/workspace/run_workspace.sh --host 127.0.0.1 --port 7870 --max-jobs 8
python -m worldfoundry.studio.cli <model-id> --frontend world --device cuda:0
python -m worldfoundry.studio.workspace_job infer --model-id <model-id> \
  --prompt "<prompt>" --output-dir tmp/studio/<model-id> --device cuda
```

The Workspace serves many models; a `wf-studio` process locks one model at
launch. `workspace_job infer` is the same job path the Workspace submits and
reproduces a UI job from the shell.

## Workflow

### 1. Prove the public route first

`wf run <model-id>` must work before Studio work starts (see
[model integration](../worldfoundry-model-integration/SKILL.md)). Studio adds
session behavior; it does not fix model loading or numerics.

### 2. Confirm catalog discovery

```bash
python -c "from worldfoundry.studio.inference.catalog import find_entry; print(find_entry('<model-id>'))"
```

Discovery parses pipeline modules statically, without importing model
stacks. The entry's `call_params`, `stream_params`, and `load_params` come from
the `__call__`, `stream`, and `from_pretrained` signatures, and `runtime_kind`
selects the driver. An entry can be listed yet fail at LOAD when its runtime
imports or checkpoints are missing in the current environment.

### 3. Choose the interaction contract

| Contract | Implement | Reference |
| --- | --- | --- |
| One-shot | `__call__` returning durable artifacts | Most video pipelines |
| Pipeline-owned interactive state | `stream_state_ready()`, optional `init_stream(...)`, `stream(...)`, `reset_memory()` | `worldfoundry/pipelines/worldfm/pipeline_worldfm.py` |
| Native realtime session | `prepare_realtime()`, `configure_realtime(...)`, `stream_realtime(...)`, `realtime_next_output_frames()`, `reset_realtime()` | `worldfoundry/pipelines/matrix_game/pipeline_matrix_game_2.py`, `worldfoundry/pipelines/forcing/pipeline_native_forcing.py` |
| Reconstruct then render | A specialized driver such as `TwoStage3DGSRuntimeDriver` | `worldfoundry/studio/inference/execution.py` |

- `init_stream(...)` prepares state through the pipeline's input contract
  without generating a chunk, and preserves the prior state if preparation
  fails. Studio forwards inputs and never seeds or overwrites model-owned memory.
- Reset calls `reset_realtime()`, otherwise `reset_memory()`, otherwise the
  frame-memory reset; release caches but keep weights.
- For realtime cadence, frame policies, and control ACKs, read
  [realtime contract](references/realtime-contract.md).

### 4. Declare outputs

Return artifact paths that outlive the process. For decoded RGB, return a
`video_chunk` built from `VideoChunk` with explicit `layout` and
`value_range`; transport and export use that declaration instead of guessing
from pixel values. `video_chunk: None` means no RGB output, and non-output
distributed ranks return no frames.

### 5. Respect worker lifecycle

- Managed CUDA runs acquire a device lease before launch; resident workers
  keep it until their process group stops. Controllers sharing GPUs must use
  the same `WORLDFOUNDRY_GPU_LEASE_DIR`.
- `WORLDFOUNDRY_STUDIO_INFERENCE_TIMEOUT` is one deadline covering GPU queueing,
  worker allocation, execution, and output draining; `0` disables it.
- Cancellation and failure stop the owned process group before the lease is
  released. Weight pipelines are cached; API clients are request-scoped so
  credentials never leak between requests.
- Cached pipelines serve many requests, so per-request RNG, latents, scheduler
  history, and caches must reset between them.

### 6. Collect serving evidence

```bash
python -m pytest -q tests/studio/test_dispatch_lifecycle.py \
  tests/studio/test_pipeline_owned_stream_state.py \
  tests/studio/test_stream_init.py tests/studio/test_stream_step.py
make test-infer-tensors
make test-realtime-client
```

Output declarations are covered by `tests/pipelines/test_native_output_contracts.py`,
`tests/pipelines/test_non_native_output_contracts.py`, and
`tests/core/test_video_chunk_transport.py`. Serving claims additionally need
A/B/A isolation, a failed request followed by reuse, cancellation, worker
reuse, simultaneous sessions, and the advertised rank topology. Injected
cancellation and deterministic interleaving prove bounded cleanup, not live
CUDA concurrency. Checkpoint-backed browser replay and the realtime
performance gate are described in [realtime contract](references/realtime-contract.md).

## Debug path

| Symptom | Inspect |
| --- | --- |
| Model missing from the UI | `discover_catalog()` output, aliases, hidden IDs, pipeline file naming |
| Wrong variant or device | Launch flags, variant resolution, `PipelineContext` |
| Inputs never reach the model | `prepare_inputs()`, `PreparedInputs`, driver call kwargs, signature names |
| Blank preview | Artifact keys returned by the pipeline, `collect_artifact_paths()`, `pick_preview_assets()` |
| INIT/STREAM broken | `BaseRuntimeDriver.run_init()`/`run_continue()`, `stream_state_ready()`, memory state |
| Recent run will not reload | `RunRecord.to_manifest()`, `materialize_run()`, drifted artifact paths |
| LOAD fails with `cv2`, `python3.x`, or `libssl` errors | Workspace started outside the unified env, or a copied conda env |

## Gotchas

- Hide no supported input in `**kwargs`: Studio forms list only parameters
  named in the static signatures.
- Fix the pipeline's artifact contract before widening
  `WORLDFOUNDRY_STUDIO_ARTIFACT_SCAN_MODE` or conversion helpers. Staged inputs
  never satisfy required generated-output kinds.
- Re-seeding every realtime chunk repeats diffusion noise and produces visible
  temporal beats; a rollout owns one continuous RNG stream.
- Transports never infer latent strides or change segment counts; the model
  owns cadence through its `RealtimeSpec`.
- A plain SSH tunnel carries signaling only. Browsers that cannot reach the
  GPU host need a TURN relay in `WORLDFOUNDRY_REALTIME_ICE_SERVERS_JSON`.
- Model integration and Studio deployment are separate acceptance scopes; a
  passing `wf run` says nothing about reuse, cancellation, or latency.
- Public demo media come only from a run of the matching official
  configuration through the in-tree Studio path, with its manifest retained.

## Done criteria

- The entry resolves through `find_entry()` with the intended runtime kind,
  parameters, and defaults.
- INIT, STREAM, RESET, failure, and retry behave per the chosen contract, with
  A/B/A isolation on a cached pipeline.
- Outputs declare layout and value range; previews select generated
  artifacts only.
- Weight-free Studio tests pass; realtime claims carry replay and timing
  evidence with hardware and network conditions stated.
