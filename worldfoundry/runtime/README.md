# `worldfoundry.runtime`

Execution-environment preparation: which conda environment a model needs, which CUDA tier
the host supports, which GPUs to use, where checkpoints are cached, and how long-running jobs
are tracked. This package decides *where and how* a model runs, not *what* it computes.

## Used by

- `scripts/setup/bootstrap_worldfoundry.sh` and `scripts/setup/model_env_install.sh` (environment profiles)
- `scripts/inference/prepare_model_infer.sh` and `wf zoo model-download --check-local` (asset checks)
- `wf run` and the Studio workspace (devices, jobs, compile cache)

## Layout

| File | Purpose |
| --- | --- |
| `conda.py`, `env.py` | Resolve and activate per-model conda profiles and environment variables |
| `cuda_tiers.py` | CUDA tier detection (`cu128`, …) and wheel-index selection |
| `platforms/` | Hardware/platform detection and provider abstraction |
| `device_pool.py` | GPU allocation across concurrent workers |
| `assets.py`, `local_checkpoint_cache.py` | Checkpoint discovery and Hugging Face / HFD cache resolution |
| `compile_cache.py`, `performance.py` | Re-export of the core `torch.compile` cache API; performance profiles |
| `jobs.py`, `job_worker.py` | Durable job records, cancellation, and worker processes |
| `in_tree_cli.py`, `official_inference.py` | Bridges for in-tree runtimes launched as subprocesses |
| `inference_catalog.py` | Inference-spec registry and lookups (`get_model_inference_spec`, `model_inference_spec`), the generic fallback spec, and asset-gating policy |
| `inference_specs/` | Curated inference specs, one module per model family (`lingbot_world.py`, `cosmos3.py`, `matrix_game4.py`, …), plus shared field/fixture helpers in `_common.py` |
| `interactive_inference_catalog.py` | Shared builders for interactive (keyboard/camera) world-model specs |
| `probes.py`, `realtime_regression.py` | Runtime probes and realtime latency regression checks |

Environment details: [Environments](https://openenvision.github.io/WorldFoundry/docs/reference/environments).

## Adding an inference spec

Add a module under `inference_specs/` that builds the family's `ModelInferenceSpec`, then
register it in `_MODEL_INFERENCE_SPECS` in `inference_catalog.py`. Families without a
curated spec fall back to `generic_model_inference_spec`.
