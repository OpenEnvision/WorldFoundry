# WorldFoundry

[![Python](https://img.shields.io/badge/python-3.10%20--%203.13-blue)](pyproject.toml)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)
[![CLI](https://img.shields.io/badge/CLI-worldfoundry--eval-purple)](https://openenvision.github.io/WorldFoundry/docs/reference/cli)
[![Docs](https://img.shields.io/badge/docs-Fumadocs-orange)](https://openenvision.github.io/WorldFoundry/docs)

WorldFoundry is an open-source infrastructure for world models: a shared stack for in-tree runners, local asset staging, inference (TUI / CLI / Studio), and benchmark evaluation across video generation, 3D/4D representation, embodied action, and interactive worlds.

> ⚠️ This repository is still under active development. We will keep updating it regularly. Feel free to open an issue if you encounter any problem.

Day-one workflow:

1. **Environment + assets** — bootstrap conda, stage checkpoints and datasets outside git.
2. **Inference** — generate and inspect artifacts via TUI, CLI, scripts, or Studio.
3. **Evaluation** — score only after artifacts match the benchmark layout; use scorecards for readiness claims.

## 🤝 Community

Join the **WorldFoundry Community** on Slack, [Discord](https://discord.gg/ybUQMDA4x), or WeChat for discussions, announcements, technical support, and the latest project updates.

<p align="center">

<a href="https://join.slack.com/t/worldfoundrycommunity/shared_invite/zt-43nbi9fw4-okYiELzZHp0_1UPa3dh3bQ">
  <img src="https://img.shields.io/badge/Slack-Join%20Workspace-4A154B?logo=slack&logoColor=white" alt="Join Slack">
</a>

<a href="https://github.com/OpenEnvision/WorldFoundry/issues">
  <img src="https://img.shields.io/badge/GitHub-Issues-181717?logo=github" alt="GitHub Issues">
</a>

</p>

<p align="center">
  <strong>WeChat Community</strong>
</p>

<table align="center">
  <tr>
    <td align="center"><strong>Group 1 — Full</strong></td>
    <td align="center"><strong>Group 2 — Full</strong></td>
    <td align="center"><strong>Group 3 — Open</strong></td>
  </tr>
  <tr>
    <td align="center"><img src="docs/fumadocs/public/wechat_qr.png" alt="WorldFoundry WeChat Group 1" width="280"></td>
    <td align="center"><img src="docs/fumadocs/public/wechat_qr_2.png" alt="WorldFoundry WeChat Group 2" width="280"></td>
    <td align="center"><img src="docs/fumadocs/public/wechat_qr_3.png" alt="WorldFoundry WeChat Group 3" width="280"></td>
  </tr>
</table>

<p align="center">
  <em>Groups 1 and 2 are full. Scan the Group 3 QR code to join.<br>
  QR codes are updated periodically if they expire.</em>
</p>

## 📰 News

- **[2026-07-12]** 🔥 **WorldFoundry reached 100+ stars on its very first day!** Thanks to the community for the incredible support and encouragement. More exciting updates are coming!
- **[2026-07-11]** 🎉 **WorldFoundry is officially open-sourced.** We welcome ⭐ stars, bug reports, feature requests, and pull requests from the community!


![WorldFoundry teaser](docs/fumadocs/public/teaser.png)

## Links

- [Documentation](https://openenvision.github.io/WorldFoundry/docs)
- [Design and architecture](https://openenvision.github.io/WorldFoundry/docs/overview/design)
- [Quickstart](https://openenvision.github.io/WorldFoundry/docs/quickstart)
- [Environment reference](https://openenvision.github.io/WorldFoundry/docs/reference/environments)
- [Local asset preparation](https://openenvision.github.io/WorldFoundry/docs/guides/local-assets)
- [TUI](https://openenvision.github.io/WorldFoundry/docs/guides/tui)
- [Inference guide](https://openenvision.github.io/WorldFoundry/docs/guides/inference)
- [Studio guide](https://openenvision.github.io/WorldFoundry/docs/guides/studio)
- [CLI reference](https://openenvision.github.io/WorldFoundry/docs/reference/cli)
- [Python API reference](https://openenvision.github.io/WorldFoundry/docs/api-reference)
- [Supported models](https://openenvision.github.io/WorldFoundry/docs/guides/supported-models)
- [Benchmark hub](https://openenvision.github.io/WorldFoundry/docs/evaluation/benchmark-hub)
- [Contributing](CONTRIBUTING.md)

## What WorldFoundry Provides

| Surface | Purpose | Entry point |
| --- | --- | --- |
| Model zoo | Catalogs video, world, 3D/4D, VLA/VA/WAM, hosted API, and metadata-only model entries. | [`worldfoundry/data/models/catalog`](worldfoundry/data/models/catalog) |
| In-tree runtimes | Keeps model architecture and inference adapters inside `worldfoundry`; checkpoints stay in local/Hugging Face caches. | [`worldfoundry/synthesis`](worldfoundry/synthesis), [`worldfoundry/pipelines`](worldfoundry/pipelines) |
| TUI | Interactive model/benchmark picker that prints runnable CLI commands. | `worldfoundry-eval tui` / `worldfoundry-tui` |
| Studio workspace | Browser UI for inference jobs, model-specific parameters, and artifact review. | [`worldfoundry.studio.serving.workspace`](worldfoundry/studio/serving/workspace.py) |
| Benchmark zoo | Catalogs benchmark manifests, required assets, official runner constraints, and readiness states. | [`worldfoundry/data/benchmarks/catalog`](worldfoundry/data/benchmarks/catalog) |
| Evaluation runner | Runs model × benchmark cells, imports existing outputs, and writes normalized scorecards. | [`worldfoundry/evaluation`](worldfoundry/evaluation) |
| Tests | Public CPU inference gate and extended module suites. | [`tests`](tests), [test guide](docs/fumadocs/content/docs/reference/validation.mdx) |
| Docs | Bilingual Fumadocs site with setup, inference, evaluation, Studio, and maintainer guides. | [`docs/fumadocs`](docs/fumadocs) |

## From Clone To First Run

Choose the install track that matches the work you are doing:

| Track | Use it for | Install |
| --- | --- | --- |
| **Lightweight / CPU** | Catalog and CLI inspection, TUI use, docs work, and CPU release checks | An editable pip install with only the required extras, for example `python -m pip install -e ".[tui]"` |
| **GPU runtime** | CUDA inference, Studio model execution, and GPU-backed benchmark runners | `bash scripts/setup/bootstrap_worldfoundry.sh` |

Optional extras declare Python dependencies but do not select a CUDA-specific
PyTorch wheel index. Do not treat a bare pip install of a GPU-coupled extra as
the supported CUDA setup; use the bootstrap/conda track instead. Optional native
kernels must be built inside the exact target PyTorch environment.

For GPU work, start with the unified environment and use a dedicated
environment only when a model profile documents an ABI or simulator conflict.
The full day-one path lives in the
[Quickstart](https://openenvision.github.io/WorldFoundry/docs/quickstart).

Model demo videos are served from GitHub CDN; docs development does not need `git lfs pull`.

```bash
# Recommended: skip LFS smudge for a much faster clone
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/OpenEnvision/WorldFoundry.git

cd WorldFoundry

bash scripts/setup/bootstrap_worldfoundry.sh
source tmp/worldfoundry_unified_env.sh
conda activate "${WORLDFOUNDRY_UNIFIED_ENV_PREFIX}"
```

To download local demo video binaries later, install Git LFS and run `git lfs pull` inside the clone.

Checkpoints, datasets, evaluator weights, API keys, and generated artifacts are **not** in git.
See [Local asset preparation](https://openenvision.github.io/WorldFoundry/docs/guides/local-assets) for cache layout, Hugging Face downloads, non-HF aliases, and benchmark assets.

On modern CUDA 12.8 hosts the installer resolves `worldfoundry-unified-cu128`. Pin a wheel tier only when the host requires it:

```bash
bash scripts/setup/bootstrap_worldfoundry.sh --cuda cu124
bash scripts/setup/bootstrap_worldfoundry.sh --cuda cu121
```

Keep datasets and checkpoints outside the repository on shared machines:

```bash
bash scripts/setup/bootstrap_worldfoundry.sh \
  --home /path/to/worldfoundry-home \
  --data-root /path/to/worldfoundry-data \
  --model-root /path/to/worldfoundry-models \
  --artifact-root /path/to/worldfoundry-artifacts
```

Hugging Face models use native Hub loading (`from_pretrained`, `snapshot_download`, `HF_HOME` / `HF_HUB_CACHE`, and `HF_TOKEN` for gated assets). `WORLDFOUNDRY_CKPT_DIR` remains for non-HF checkpoints and compatibility aliases.

Some VLA/action policies need a documented model-specific environment (for example OpenVLA-OFT / CogACT). Embodied simulator benchmarks follow the Docker VLA harness pattern — see the [environment reference](https://openenvision.github.io/WorldFoundry/docs/reference/environments).

After the environment is active:

```bash
worldfoundry-eval --help
worldfoundry-eval zoo models --json
worldfoundry-eval zoo benchmarks --json
```

### Interactive first path (TUI)

```bash
python -m pip install -e ".[tui]"
worldfoundry-eval tui
# or: worldfoundry-tui
```

The TUI reads the same catalogs as the CLI and can print a runnable command before anything expensive runs:

```bash
worldfoundry-eval tui \
  --model-id <model-id> \
  --benchmark-id <benchmark-id> \
  --print-command
```

### Scripted first model run

Prepare assets, then launch a small demo. A common starter is `matrix-game-2` (public HF repo `Skywork/Matrix-Game-2.0`):

```bash
bash scripts/inference/prepare_model_infer.sh matrix-game-2 --download
worldfoundry-eval zoo model-download --model-id matrix-game-2 --check-local --json

bash scripts/inference/run_infer.sh --category navigation-video --model matrix-game-2 \
  --output-dir tmp/matrix_game2_first_run
```

If weights already live in a shared checkpoint tree, link them instead of copying:

```bash
bash scripts/setup/link_hf_checkpoints.sh \
  --ckpt-dir "${WORLDFOUNDRY_CKPT_DIR}" \
  --hfd-root "${WORLDFOUNDRY_HFD_ROOT}" \
  --hf-hub-cache "${HF_HUB_CACHE}" \
  --default-world
```

## Run Inference

Prefer the TUI or the documented inference helpers once assets are staged:

```bash
bash scripts/inference/run_infer.sh --category navigation-video --model matrix-game-2

conda run -p "${WORLDFOUNDRY_UNIFIED_ENV_PREFIX}" \
  bash scripts/inference/run_infer.sh --category navigation-video --model matrix-game-2

bash scripts/inference/run_infer.sh --category video --model <model-id>
bash scripts/inference/run_infer.sh --category three_d_four_d --model <model-id>
```

CLI-shaped inference (same contract as Studio jobs):

```bash
python -m worldfoundry.studio.workspace_job infer \
  --model-id <model-id> \
  --prompt "a cinematic scene, high quality" \
  --output-dir tmp/worldfoundry_infer/<model-id> \
  --device cuda
```

Each successful run should write media, logs, and manifest metadata under the output directory. Treat a file as demo evidence only after visual check and matching runtime-profile assumptions. Details: [Inference guide](https://openenvision.github.io/WorldFoundry/docs/guides/inference).

### Opt-in inference acceleration

Acceleration is request-scoped. Pipelines record the requested backend, the effective backend, and any fallback. The exact path stays the default. The [inference optimization truth matrix](docs/fumadocs/content/docs/guides/inference-optimization.mdx) shows which options have a CPU contract or GPU certification, and which remain pending or rejected.

The following example shows available controls, not a certified combination. Check model-specific compatibility and quality gates before enabling approximate options.

```python
from worldfoundry.pipelines.wan.pipeline_wan_2p2 import Wan2p2Pipeline

pipe = Wan2p2Pipeline.from_pretrained(
    model_path="/checkpoints/wan22",
    device="cuda",
    offload_mode="resident",           # aliases: fast/none; async-block when VRAM is tight
    attention_backend="flash2",       # flash2/flash3/sage/sage3/xformers
    quantization={"mode": "fp8"},     # runtime report proves kernel vs dense fallback
    fuse_qkv=True,
    fused_rope=True,
    static_cross_kv=True,
    torch_compile=True,
    teacache=True,                      # lossy and explicit
    vae_decode_autocast="bf16",
    vae_spatial_tiling=True,
    vae_temporal_chunk_size=4,
)
```

**Acceleration options and checks**

- FlashAttention and xFormers are intended as numerically equivalent providers; check the effective backend in the run report.
- SageAttention, FP8/NVFP4, STA/VSA/VMoBA, pinned LightX2V sparse lanes, TeaCache, and token pruning are approximate. They must pass the target model's quality budget.
- Missing packages, unsupported GPUs, or unsupported shapes either fall back to a reported exact path or fail closed.
- Wan2.2 TI2V-5B uses the resident preset by default in Studio on the target H100 path. `block` / `async-block` overlaps the next layer's H2D copy with compute, and is effective only with async copies, no synchronous rescue copies, and at most two CUDA-resident layers.
- Quantization plus block offload is rejected. Use resident quantization or dense async offload.

**Check a run before citing a speedup**

```bash
# Sequence-parallel reference, then the multi-rank candidate.
torchrun --nproc_per_node=1 -m benchmarks.inference.wan22_multigpu_e2e \
  --checkpoint /checkpoints/Wan2.2-TI2V-5B \
  --output-dir benchmarks/results/wan22-sp1 \
  --fused-rope --rope-precision fp64 \
  --save-reference-latents \
  --warmup-runs 1 --measured-runs 3

torchrun --nproc_per_node=4 -m benchmarks.inference.wan22_multigpu_e2e \
  --checkpoint /checkpoints/Wan2.2-TI2V-5B \
  --output-dir benchmarks/results/wan22-sp4 \
  --sp-degree 4 \
  --fused-rope --rope-precision fp64 \
  --warmup-runs 1 --measured-runs 3 \
  --reference-latents benchmarks/results/wan22-sp1/final-latents.pt \
  --reference-video benchmarks/results/wan22-sp1/output.mp4 \
  --fail-on-fallback --profile-collectives

# Adapter overhead must stay within 3% of the direct provider.
python -m benchmarks.operators.attention_adapter_parity \
  --backend flash2 --strict --out benchmarks/results
```

The checked-in VMoBA profile is only valid for the default TI2V grid `(31, 22, 40)`. Other frame or spatial shapes need a retuned profile; bad geometry fails closed. Framework A/B numbers need schema v4 on one physical GPU in `Exclusive_Process` mode. Shared H100s in `Default` mode fail that gate, so current timings stay diagnostic. Gate details live in the truth matrix linked above.

## Launch Studio Workspace

Studio is the preferred UI for release validation: model-specific forms, job status, preview media, and artifact links in one place. Start it from the same unified env used for inference:

```bash
source tmp/worldfoundry_unified_env.sh
conda activate "${WORLDFOUNDRY_UNIFIED_ENV_PREFIX}"

bash scripts/workspace/run_workspace.sh \
  --host 127.0.0.1 \
  --port 7870 \
  --max-jobs 8
```

Open `http://127.0.0.1:7870/`. If `python`, **LOAD**, or **START** fails with a missing interpreter, `cv2`, or `libssl`/`libcrypto` error, recreate or verify the env and restart:

```bash
bash scripts/setup/bootstrap_worldfoundry.sh --verify-only
source tmp/worldfoundry_unified_env.sh
bash scripts/workspace/run_workspace.sh
```

Configure jobs in **Create Job**; optional shared defaults can use `WORLDFOUNDRY_STUDIO_SETTINGS_FILE`. Expensive runtime checks and preview builders are opt-in via `WORLDFOUNDRY_STUDIO_*` — see the [Studio guide](https://openenvision.github.io/WorldFoundry/docs/guides/studio).

Use the **Visualizers** tab as the browser entrypoint for local preview services (World realtime, Spark, Viser, Rerun, Embodied bridge). On a remote machine, forward port `7870` plus any viewer ports you launch.

For a single-model Studio process:

```bash
worldfoundry-studio
```

## Run Evaluation

Run evaluation through a runnable benchmark path. Use `official-run` when the evaluator can execute locally; use `official-validation` when you already have official-shaped result files to import.

```bash
worldfoundry-eval run \
  --model matrix-game-2 \
  --benchmark vbench \
  --mode official-run \
  --output-dir tmp/hello_world_run \
  --json
```

Inspect:

- `run_manifest.json`: selected model, benchmark/task metadata, timestamps, and output paths.
- `results.jsonl`: per-sample generation records and artifact metadata.
- `metrics/summary.json`: aggregate metrics and failed/skipped sample counts.
- `scorecard.json`: readiness, leaderboard eligibility, metric values, and blockers.

For model and benchmark discovery:

```bash
worldfoundry-eval tasks list
worldfoundry-eval zoo models --json
worldfoundry-eval zoo benchmarks --json
worldfoundry-eval zoo model-show --model-id <model-id> --include-manifest --json
worldfoundry-eval zoo benchmark-show --benchmark-id <benchmark-id> --include-spec --json
```

For existing official-shaped benchmark outputs:

```bash
worldfoundry-eval zoo benchmark-run \
  --benchmark-id vbench \
  --mode official-validation \
  --official-results-path <official_results.json> \
  --generated-artifact-dir <generated_videos> \
  --output-dir tmp/benchmark_zoo/official_validation/vbench \
  --json
```

For existing generated outputs:

```bash
worldfoundry-eval evaluate \
  --results-path tmp/results.jsonl \
  --output-dir tmp/worldfoundry_evaluate \
  --metric artifact_count \
  --required-artifact video \
  --json
```

For the formal benchmark inventory, review the expanded plan first:

```bash
worldfoundry-eval prepare \
  --all-benchmarks \
  --output-dir tmp/worldfoundry_all_benchmarks_plan \
  --json

worldfoundry-eval run \
  --all-benchmarks \
  --model <model-zoo-id> \
  --prepare \
  --data-root cache/worldfoundry/data/hfd_datasets \
  --plan-only \
  --output-dir tmp/worldfoundry_all_benchmarks_plan \
  --json
```

Use the integrity commands before claiming benchmark support:

```bash
worldfoundry-eval zoo benchmarks --json
worldfoundry-eval run --plan-only --json
```

For release audits, use public CLI surfaces only:

```bash
worldfoundry-eval validate-artifact tmp/benchmark-run/scorecard.json \
  --kind scorecard --check-artifacts --json
```

Contract runs, normalizer-only imports, partial dataset runs, and missing-official-runner checks are not leaderboard evidence. A public claim needs the full official data/runtime path and a scorecard whose eligibility fields explicitly support the claim.

## Documentation Site

Run the docs locally:

```bash
cd docs/fumadocs
npm ci
npm run dev -- --port 8014
```

Build the static docs from the repository root:

```bash
bash scripts/docs/build.sh
```

The docs app serves English routes under `/docs` and Chinese routes under `/zh/docs`.

## Development Checks

Use these checks before opening a PR or marking a model/benchmark ready:

```bash
source tmp/worldfoundry_unified_env.sh
conda activate "${WORLDFOUNDRY_UNIFIED_ENV_PREFIX}"

PYTHONPATH=. python -m compileall -q worldfoundry scripts
make lint
make packaging-check
make docs-check
bash scripts/docs/build.sh --skip-bootstrap

worldfoundry-eval zoo model-download --model-id <model-id> --check-local --json
worldfoundry-eval zoo benchmark-download --benchmark-id <benchmark-id> --check-local --json
worldfoundry-eval run --plan-only --fail-on-overclaim --fail-on-stale --json
```

When adding or changing a model:

1. Port required inference code into `worldfoundry`; do not depend on a cloned external repo at runtime.
2. Keep official repositories only as provenance or parity references.
3. Declare checkpoints, runtime variables, and environment assumptions in the model catalog/runtime profile.
4. Run the smallest official-style demo and visually inspect the artifact.
5. Record evidence in the docs before promoting readiness.

## Repository Layout

```text
WorldFoundry
├─ docs/fumadocs                         # Documentation site, teaser, screenshots, and demo media
├─ requirements                          # Unified and optional dependency presets
├─ scripts
│  ├─ inference                          # User-facing inference entrypoints
│  ├─ setup                              # Conda setup wrappers
│  ├─ workspace                          # Studio / Workspace launch helpers
│  └─ docs                               # Documentation build wrapper
├─ worldfoundry
│  ├─ core                               # Shared contracts and reusable runtime abstractions
│  ├─ data                               # Model/benchmark catalogs, runtime profiles, fixtures
│  ├─ evaluation                         # Runner, tasks, metrics, scorecards, reports
│  ├─ operators                          # Input validation, preprocessing, interaction handling
│  ├─ pipelines                          # User-facing pipeline wrappers
│  ├─ representations                    # 3D/4D and spatial representation outputs
│  ├─ runtime                            # Runtime paths, assets, jobs, and probes
│  ├─ studio                             # Workspace and Studio frontends
│  └─ synthesis                          # In-tree model synthesis/action-generation runtimes
├─ thirdparty                            # Reviewed vendored/native dependencies
└─ tools                                 # Maintenance and asset utilities
```

## Citation

If you use WorldFoundry or its benchmark/model integrations in research, cite this repository and the upstream methods, checkpoints, datasets, and benchmarks that your run depends on. A formal paper citation will be added when the technical report is released.

## License

WorldFoundry-authored code is licensed under the [Apache License 2.0](LICENSE), except where a file or component carries a different license notice. Vendored and adapted upstream components remain subject to their original licenses.

## Acknowledgment

WorldFoundry integrates and wraps a large set of upstream world-model, video-generation, perception, reconstruction, and embodied-action projects. See the method-specific runtime profiles and the docs appendix for upstream project pointers and licenses.

We also thank the following open-source projects for their model, runtime, and evaluation work:

- [FastVideo](https://github.com/hao-ai-lab/FastVideo) — a unified inference and post-training framework for accelerated video generation
- [OpenWorldLib](https://github.com/OpenDCAI/OpenWorldLib) — a unified codebase for advanced world models
- [VLA Evaluation Harness](https://github.com/allenai/vla-evaluation-harness) — one framework to evaluate VLA models on robot simulation benchmarks
