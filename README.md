<p align="center">
  <img src="assets/logo.png" width="720" alt="WorldFoundry"/>
</p>

<h1 align="center">
  WorldFoundry: Unified World Model<br>Inference &amp; Evaluation Infrastructure
</h1>

<p align="center">
  <a href="https://openenvision.github.io/index.html"><b>OpenEnvision</b></a>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2611.10771"><img src="https://img.shields.io/badge/arXiv-2611.10771-B31B1B?style=for-the-badge&logo=arxiv&logoColor=white" alt="arXiv"/></a>
  <a href="https://github.com/OpenEnvision/WorldFoundry"><img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white" alt="GitHub"/></a>
  <a href="https://openenvision.github.io/WorldFoundry/docs"><img src="https://img.shields.io/badge/Docs-4B8BBE?style=for-the-badge&logo=readthedocs&logoColor=white" alt="Docs"/></a>
  <a href="https://openenvision.github.io/WorldFoundry/zh/docs"><img src="https://img.shields.io/badge/中文文档-07C160?style=for-the-badge" alt="中文文档"/></a>
  <a href="https://openenvision.github.io/WorldFoundry/docs/guides/supported-models"><img src="https://img.shields.io/badge/Models-FF6F00?style=for-the-badge&logo=google-chrome&logoColor=white" alt="Models"/></a>
  <a href="https://openenvision.github.io/WorldFoundry/docs/evaluation/benchmark-hub"><img src="https://img.shields.io/badge/Benchmarks-32CD32?style=for-the-badge&logo=google-chrome&logoColor=white" alt="Benchmarks"/></a>
  <a href="https://discord.gg/ybUQMDA4x"><img src="https://img.shields.io/badge/Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white" alt="Discord"/></a>
  <a href="https://github.com/OpenEnvision/WorldFoundry/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/OpenEnvision/WorldFoundry/ci.yml?branch=main&style=for-the-badge&label=CI" alt="CI"/></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/Python-3.10--3.13-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache--2.0-yellow?style=for-the-badge" alt="Apache-2.0"/></a>
</p>

<p align="center"><em>One environment, one catalog, one CLI, and one browser workspace for video, interactive-world, 3D/4D, and embodied-action models.</em></p>

---

<p align="center"><b>TL;DR</b> — WorldFoundry runs 270+ world models (video generation, interactive worlds, 3D/4D reconstruction, VLA policies) through one CLI, one Python API, and one browser workspace, and scores their outputs on integrated benchmarks with auditable scorecards.</p>

## 📰 News
- **[2026-11-24]** 🚀 **WorldFoundry now supports more models, including HY-Nomas, π-H3, and π4!** We also standardized the codebase more thoroughly and removed redundant code, so it is now cleaner and more consistent. Thank you all for your patience!
- **[2026-11-10]** 🔥 **The WorldFoundry technical report is now available on [arXiv](https://arxiv.org/abs/2611.10771)!** We welcome your support with a GitHub Star and a [Hugging Face Upvote](https://huggingface.co/papers/2611.10771). Thank you for your support!
- **[2026-07-12]** 🔥 **WorldFoundry reached 100+ stars on its very first day!** Thanks to the community for the incredible support and encouragement. More exciting updates are coming!
- **[2026-07-11]** 🎉 **WorldFoundry is officially open-sourced.** We welcome ⭐ stars, bug reports, feature requests, and pull requests from the community!

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
    <td align="center"><img src="assets/wechat_qr.png" alt="WorldFoundry WeChat Group 1" width="280"></td>
    <td align="center"><img src="assets/wechat_qr_2.png" alt="WorldFoundry WeChat Group 2" width="280"></td>
    <td align="center"><img src="assets/wechat_qr_3.png" alt="WorldFoundry WeChat Group 3" width="280"></td>
  </tr>
</table>

<p align="center">
  <em>Groups 1 and 2 are full. Scan the Group 3 QR code to join.<br>
  QR codes are updated periodically if they expire.</em>
</p>

## Highlights

| Goal | Entry point |
| --- | --- |
| Browse models, benchmarks, checkpoints, and readiness | `wf zoo models` / `zoo benchmarks` |
| Generate a video, world, 3D asset, or action rollout | `wf run <model-id>` |
| Submit jobs and review outputs in a browser | `bash scripts/workspace/run_workspace.sh` → `http://127.0.0.1:7870` |
| Build a command interactively in the terminal | `wf tui` |
| Call a model from your own code | `from worldfoundry.pipelines... import <Pipeline>` |
| Score generated outputs on a benchmark | `wf score` / `wf run --model ... --benchmark ...` |
| Let an LLM agent run inference and evaluation | `wf mcp` |

All of these surfaces read the same catalogs and call the same pipelines, so a command
that works in one also works in the others.

## 💻 System Requirements

- **Platform:** Linux with an NVIDIA CUDA-capable GPU.
- **Driver:** CUDA 12.8+ support for the default unified environment.
- **GPU architecture / VRAM:** model-specific; `wf zoo model-show --model-id <id>` and the model page list it.
- **Tools:** Git and Conda, or Docker with the NVIDIA Container Toolkit ([Docker guide](docker/README.md)).

Catalog browsing, `--help`, `--model-status`, `--print-config`, and `--plan-only` never load
weights, so they also work on a CPU-only machine.

## 🚀 Quick Start

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/OpenEnvision/WorldFoundry.git
cd WorldFoundry

# Creates the unified CUDA conda env and writes tmp/worldfoundry_unified_env.sh
bash scripts/setup/bootstrap_worldfoundry.sh
```

**Every new shell** must load the environment before running any WorldFoundry command:

```bash
source tmp/worldfoundry_unified_env.sh
conda activate "${WORLDFOUNDRY_UNIFIED_ENV_PREFIX}"
wf --help   # sanity check
```

`wf` is the WorldFoundry CLI; `wf-studio` launches a single-model Studio frontend.

Common bootstrap options:

```bash
# Keep caches, checkpoints, data, and outputs on a large/shared disk
bash scripts/setup/bootstrap_worldfoundry.sh \
  --home /path/to/worldfoundry-home \
  --data-root /path/to/worldfoundry-data \
  --model-root /path/to/worldfoundry-models \
  --artifact-root /path/to/worldfoundry-artifacts

# Verify an existing install without changing it
bash scripts/setup/bootstrap_worldfoundry.sh --verify-only

# Skip building flash-attn (e.g. no compiler toolchain)
bash scripts/setup/bootstrap_worldfoundry.sh --skip-flash-attn

# Also install a model's dedicated env and print its checkpoint plan
bash scripts/setup/bootstrap_worldfoundry.sh --with-model matrix-game-2 --prepare-model matrix-game-2
```

Most models run in the unified environment. A few declare a dedicated environment for a
pinned torch/CUDA ABI, JAX/TensorFlow, or a simulator; list and install them with:

```bash
bash scripts/setup/model_env_install.sh --list
bash scripts/setup/model_env_install.sh --model <model-id>
```

See the [environment guide](https://openenvision.github.io/WorldFoundry/docs/reference/environments) for the full mapping.

## 🧭 Key Concepts

- **Model ID** — a stable catalog name such as `wan2.1-t2v-1.3b`, `self-forcing`, or `matrix-game-2`. Aliases are accepted interactively; use canonical IDs in scripts.
- **Task profile** — a model's preset input/output contract (e.g. `t2v`, `image-to-video`, `interactive-video`). Pass it with `--pipeline.task-profile`; the model page and `run <id> --help` list valid values.
- **Readiness** — `runnable_runner` models execute directly; `runner_candidate` and `listed_only` models can be inspected but stop before loading weights and explain why.
- **Assets** — checkpoints, datasets, metric weights, and API keys are never committed. They live in the Hugging Face cache or under `WORLDFOUNDRY_HOME` (see *Paths and Environment Variables* below).
- **Inference vs. evaluation** — inference produces artifacts; evaluation scores artifacts against a benchmark protocol. Generate and inspect first, then score.

## 🎮 Usage

The examples below use `wan2.1-t2v-1.3b` (Wan2.1 text-to-video, 832×480, 81 frames, ~8 GB VRAM).
Replace it with any model ID from the catalog.

### 1. Find a model

```bash
wf zoo models                                   # all models
wf zoo models --integration-status integrated   # integrated only
wf zoo models --source-status open_source       # exclude hosted APIs

wf zoo model-show --model-id wan2.1-t2v-1.3b --include-manifest --json
```

The `needs` column tells you what a model requires before it can run: `checkpoint`,
`auth` (gated Hugging Face repo), or `api-key` (hosted API such as Veo3, Kling, Sora2).
Per-model pages with inputs, defaults, and VRAM live under
[Models](https://openenvision.github.io/WorldFoundry/docs/guides/supported-models).

### 2. Prepare checkpoints

```bash
# Check environment and local cache, print what is missing (downloads nothing)
bash scripts/inference/prepare_model_infer.sh wan2.1-t2v-1.3b

# Download missing public checkpoints into the Hugging Face cache
bash scripts/inference/prepare_model_infer.sh wan2.1-t2v-1.3b --download

# Re-check at any time
wf zoo model-download --model-id wan2.1-t2v-1.3b --check-local --json
```

For gated checkpoints, accept the terms on Hugging Face and export a token first:

```bash
export HF_TOKEN=<your-huggingface-token>
```

If the weights already exist on shared storage, link them into the expected cache layout
instead of copying:

```bash
bash scripts/setup/link_hf_checkpoints.sh \
  --ckpt-dir /path/to/checkpoints \
  --hf-hub-cache "${HF_HUB_CACHE}" \
  --default-world
```

Details and per-model layouts: [Local assets](https://openenvision.github.io/WorldFoundry/docs/guides/local-assets).

### 3. Run inference from the CLI

`wf run <model-id>` is the canonical inference command. Inspect the resolved
configuration first — neither command loads weights:

```bash
wf run wan2.1-t2v-1.3b --help            # typed flags for this model
wf run wan2.1-t2v-1.3b --print-config --json
```

Then run with only the overrides you need:

```bash
wf run wan2.1-t2v-1.3b \
  --pipeline.prompt "A paper boat drifting down a forest stream, cinematic lighting" \
  --pipeline.seed 42 \
  --output-dir tmp/worldfoundry_infer/wan2.1-t2v-1.3b
```

Flags are grouped by prefix:

| Prefix | Controls | Examples |
| --- | --- | --- |
| `--pipeline.*` | Request inputs and generation settings | `prompt`, `negative-prompt`, `frames`, `height`, `width`, `steps`, `guidance-scale`, `seed`, `task-profile` |
| `--pipeline.load.*` | Checkpoint and model loading | `checkpoint-path`, `model-path`, `torch-dtype`, `offload-mode`, `attention-backend`, `compile`, `quantization` |
| `--runtime.*` | Execution device and runner | `device` |

Available fields differ per model — always check `run <model-id> --help`. Image-, video-,
camera-, and action-conditioned models expose their extra inputs (input image, camera
trajectory, action sequence such as `--pipeline.interactions`) as additional `--pipeline.*`
fields on that help page.

Generated media is written to `<output-dir>/generated/`; see [Outputs](#-outputs).

### 4. Use the browser workspace (Studio)

```bash
bash scripts/workspace/run_workspace.sh --host 127.0.0.1 --port 7870
```

Open **http://127.0.0.1:7870/**. On a remote GPU machine, forward the port first:

```bash
ssh -L 7870:127.0.0.1:7870 user@gpu-host
```

In the workspace you can:

- **Create Job** — pick a model and variant, fill in the model-specific form (prompt, image, camera, actions), toggle runtime options such as CPU offload or `torch.compile`, and submit.
- **Inference jobs** — follow status and logs, stop jobs.
- **Gallery** — review generated videos, images, and 3D outputs.
- **Visualizers** — launch point-cloud, Rerun, and interactive-world viewers embedded in the page.
- **Evaluation** — run benchmark scoring (e.g. VBench) on a folder of generated videos.

Useful launch options: `--max-jobs N` (concurrent workers, default 8), `--ckpt-dir PATH`,
`--data-dir PATH`. Interactive real-time models (WASD/IJKL control) stream frames over
WebRTC from the same workspace. Full guide: [Studio](https://openenvision.github.io/WorldFoundry/docs/guides/studio) ·
[Interactive inference](https://openenvision.github.io/WorldFoundry/docs/guides/interactive-inference).

### 5. Use the terminal UI

```bash
wf tui
```

The TUI lists models and benchmarks with readiness indicators, lets you fill inputs, and
launches the run. To only print the equivalent CLI command:

```bash
wf tui --model-id <model-id> --benchmark-id <benchmark-id> --print-command
```

### 6. Call a pipeline from Python

Each model exposes a pipeline class with `from_pretrained(...)` and `__call__(...)`:

```python
from worldfoundry.pipelines.wan.pipeline_wan_2p1_t2v import Wan2p1T2VPipeline

pipe = Wan2p1T2VPipeline.from_pretrained(device="cuda")
result = pipe(
    "A paper boat drifting down a forest stream, cinematic lighting",
    num_frames=81,
    num_inference_steps=50,
    seed=42,
    output_path="tmp/wan21_t2v.mp4",
    return_dict=True,
)
print(result["artifact_path"])
```

Common call arguments are `prompt`, `images` / `image_path`, `video` / `video_path`,
`interactions`, `negative_prompt`, `height`, `width`, `num_frames`, `num_inference_steps`,
`guidance_scale`, `seed`, and `output_path`. Model-specific arguments (camera trajectories,
reference images, control videos) are documented on each model page. Pipeline classes live
under `worldfoundry/pipelines/<family>/`.

### 7. Evaluate on benchmarks

A benchmark run needs benchmark assets, candidate artifacts, and a runner mode. Start by
listing benchmarks and preparing their assets:

```bash
wf zoo benchmarks --ready-now
wf zoo benchmark-show --benchmark-id vbench --include-spec --json

python scripts/setup/prepare_benchmark_assets.py --benchmark-id vbench --json
python scripts/setup/prepare_benchmark_assets.py --benchmark-id vbench \
  --write-env "${WORLDFOUNDRY_HOME}/vbench.env" --create-dirs
source "${WORLDFOUNDRY_HOME}/vbench.env"
```

Then pick the workflow that matches what you have:

```bash
# A. Score videos you already generated (anywhere) with a benchmark's protocol
wf score \
  --benchmark vbench \
  --artifacts /path/to/videos \
  --output-dir tmp/score/vbench \
  --plan-only --json            # drop --plan-only once it reports ready: true

# B. Generate with an in-tree model and score it in one command
wf run \
  --model wan2.1-t2v-1.3b \
  --benchmark vbench \
  --mode official-run \
  --output-dir tmp/vbench/wan2.1-t2v-1.3b \
  --json

# C. Plan a model × benchmark matrix before spending GPU time
wf run --model <model-id> --all-benchmarks --plan-only \
  --output-dir tmp/worldfoundry_plan --json

# D. Import results produced by a benchmark's official code
wf zoo benchmark-run \
  --benchmark-id <benchmark-id> \
  --mode official-validation \
  --official-results-path /path/to/official-results.json \
  --generated-artifact-dir /path/to/generated-artifacts \
  --output-dir tmp/<benchmark-id>/official-validation \
  --json
```

Repeat `--model` / `--benchmark` to build a matrix, use `--suite <suite-id>` for a named
preset, and add `--resume` to skip completed cells.

Runner modes:

| Mode | Meaning |
| --- | --- |
| `official-run` | Compute scores from generated artifacts with the benchmark's evaluator. |
| `official-validation` | Import official-shaped results produced elsewhere. |
| `normalizer` | Convert provided result files into WorldFoundry format without running the scorer. |
| `contract` | Exercise wiring only; not a score. |

Read `scorecard.json` after every run: `score_valid`, `leaderboard_valid`, and `blockers`
tell you whether the number is a complete, comparable score. Per-benchmark data layouts and
commands are in the [Benchmark Hub](https://openenvision.github.io/WorldFoundry/docs/evaluation/benchmark-hub).

### 8. Drive WorldFoundry from an agent (MCP)

```bash
python -m pip install -e ".[mcp]"
wf mcp                                          # stdio transport
wf mcp --transport streamable-http --port 8000  # HTTP at http://127.0.0.1:8000/mcp
```

The server exposes catalog discovery, run lifecycle (`preview_run`, `evaluate`, `get_run_result`, …),
and Studio job tools (`submit_studio_inference`, `wait_for_studio_job`, …); call `server_info` first
for the recommended workflow. Studio tools talk to
the workspace at `WORLDFOUNDRY_STUDIO_WORKSPACE_URL` (default `http://127.0.0.1:7870`).

## 📦 Outputs

Commands that take `--output-dir` write a self-describing run directory:

```text
<output-dir>/
├── generated/              # videos, images, 3D assets, rollouts (direct model runs)
├── run_manifest.json       # model, variant, parameters, paths, timings, status
├── requests.jsonl          # per-sample inputs
├── results.jsonl           # per-sample results and artifact paths
├── artifacts.jsonl         # index of generated artifacts
├── metrics/summary.json    # aggregate metrics and failure counts
├── scorecard.json          # metrics, eligibility, blockers
└── logs/<run-id>/events.jsonl
```

Multiple runs can be indexed and compared:

```bash
wf index-runs tmp/ --json
wf compare-runs <run-dir-a> <run-dir-b>
```

Exit code `0` means the command completed; `1` means a validation/metric/sample failure;
`2` means invalid input or a missing file, environment, or asset.

## ⚙️ Paths and Environment Variables

`tmp/worldfoundry_unified_env.sh` (written by bootstrap) sets these; override them before
sourcing or pass the matching bootstrap option.

| Variable | Purpose | Default |
| --- | --- | --- |
| `WORLDFOUNDRY_HOME` | Runtime root for caches, data, models, artifacts | `~/.cache/worldfoundry` |
| `HF_HOME` / `HF_HUB_CACHE` | Hugging Face checkpoint cache | `$WORLDFOUNDRY_HOME/huggingface` |
| `WORLDFOUNDRY_CKPT_DIR` | Non-HF checkpoints and upstream-style aliases | `$WORLDFOUNDRY_HOME/models/checkpoints` |
| `WORLDFOUNDRY_DATA_DIR` | Benchmark datasets | `$WORLDFOUNDRY_HOME/data` |
| `WORLDFOUNDRY_ARTIFACT_DIR` | Generated artifacts | `$WORLDFOUNDRY_HOME/artifacts` |
| `WORLDFOUNDRY_UNIFIED_ENV_PREFIX` | Unified conda environment prefix | set by bootstrap |
| `HF_TOKEN` | Access to gated Hugging Face repos | unset |
| `WORLDFOUNDRY_LOG_LEVEL` | `DEBUG` … `CRITICAL` (same as `--log-level`) | `INFO` |

## 🛠️ Troubleshooting

- **`wf: command not found`** — the environment is not loaded in this shell. Run `source tmp/worldfoundry_unified_env.sh && conda activate "${WORLDFOUNDRY_UNIFIED_ENV_PREFIX}"`, or use `python -m worldfoundry ...`.
- **Model stops before loading with a readiness message** — run `wf run <model-id> --model-status` and `bash scripts/inference/prepare_model_infer.sh <model-id>` to see missing checkpoints, auth, or environments.
- **Missing `cv2`, `python3.x`, or `libssl` in Studio jobs** — start the workspace from the unified environment, and never copy a conda env between machines; repair with `bash scripts/setup/bootstrap_worldfoundry.sh --verify-only`.
- **Out of GPU memory** — try `--pipeline.load.offload-mode`, fewer `--pipeline.frames`, or a smaller variant; the model page lists the tested VRAM.
- **Benchmark run reports blockers** — open `scorecard.json`; most blockers are missing benchmark assets or an artifact layout that does not match the benchmark's naming.
- **Need the full traceback** — add `-v` (`wf -v run ...`) or `--log-level DEBUG`.

## 🔌 Add Your Model or Benchmark

WorldFoundry integrates models in-tree through one contract, so a new model works in the CLI,
Studio, MCP, and every compatible benchmark at once.

| Type | Input | Examples |
| --- | --- | --- |
| **Text / image to video** | Prompt, optional first frame | Wan2.1/2.2, HunyuanVideo, Cosmos, CogVideoX |
| **Camera-conditioned** | First frame + camera trajectory | GEN3C, Matrix-Game 3.5, CameraCtrl, Wan-Fun-Cam |
| **Interactive world** | First frame + keyboard/mouse actions, streamed | Matrix-Game 2, LingBot-World, HY-WorldPlay |
| **3D / 4D** | Images or video | DUSt3R, VGGT, Depth Anything 3, pixelSplat |
| **Embodied action (VLA)** | Observation + instruction | OpenVLA, GR00T, DreamZero |
| **Hosted API** | Prompt / image + API key | Veo3, Sora2, Kling, Runway, Luma |

**Model steps**

1. Add catalog metadata in `worldfoundry/data/models/catalog/<category>/<model>.yaml` and a runtime profile in `worldfoundry/data/models/runtime/profiles/`.
2. Implement a pipeline in `worldfoundry/pipelines/<family>/` and its operator in `worldfoundry/operators/`; reuse the native diffusion kernel in `worldfoundry/base_models/diffusion_model/` instead of copying sampling code.
3. Bind the model ID to the pipeline under `worldfoundry/data/models/bindings/`.
4. Check it without weights (`wf run <model-id> --model-status`), then on GPU.

**Benchmark steps:** add a catalog entry under `worldfoundry/data/benchmarks/catalog/`, a runner under
`worldfoundry/evals/tasks/execution/runners/`, and any new metrics under
`worldfoundry/evals/tasks/metrics/`.

Guides: [Add a model](https://openenvision.github.io/WorldFoundry/docs/guides/add-model) ·
[Add a benchmark](https://openenvision.github.io/WorldFoundry/docs/guides/add-benchmark) ·
[package map](worldfoundry/README.md).

## 📚 Documentation

Browse the documentation in [English](https://openenvision.github.io/WorldFoundry/docs)
or [中文](https://openenvision.github.io/WorldFoundry/zh/docs).

| Task | Guide |
| --- | --- |
| Clone-to-first-run walkthrough | [Quickstart](https://openenvision.github.io/WorldFoundry/docs/quickstart) |
| Choose a model or benchmark | [Models](https://openenvision.github.io/WorldFoundry/docs/guides/supported-models) · [Benchmark Hub](https://openenvision.github.io/WorldFoundry/docs/evaluation/benchmark-hub) |
| Prepare environments and assets | [Environments](https://openenvision.github.io/WorldFoundry/docs/reference/environments) · [Local assets](https://openenvision.github.io/WorldFoundry/docs/guides/local-assets) |
| Run inference from the terminal | [Inference](https://openenvision.github.io/WorldFoundry/docs/guides/inference) · [CLI](https://openenvision.github.io/WorldFoundry/docs/reference/cli) · [TUI](https://openenvision.github.io/WorldFoundry/docs/guides/tui) |
| Use the browser workspace | [Studio](https://openenvision.github.io/WorldFoundry/docs/guides/studio) · [Interactive inference](https://openenvision.github.io/WorldFoundry/docs/guides/interactive-inference) |
| Tune inference speed and memory | [Inference optimization](https://openenvision.github.io/WorldFoundry/docs/guides/inference-optimization) |
| Score generated outputs | [Evaluation](https://openenvision.github.io/WorldFoundry/docs/evaluation) |
| Add a model or benchmark | [Add a model](https://openenvision.github.io/WorldFoundry/docs/guides/add-model) · [Add a benchmark](https://openenvision.github.io/WorldFoundry/docs/guides/add-benchmark) |

## 🤗 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for GPU development setup, coding conventions,
tests, DCO sign-off, and the PR workflow.

- [Architecture](https://openenvision.github.io/WorldFoundry/docs/maintainers/architecture) — module boundaries and execution flow.
- [Validation](https://openenvision.github.io/WorldFoundry/docs/reference/validation) — fast checks, CUDA contracts, and real-checkpoint evidence.
- [Agent Skills](skills/index.md) — inference, evaluation, integration, serving, optimization, validation, review, and maintenance workflows.

Community standards and private vulnerability reporting are covered by the
[Code of Conduct](CODE_OF_CONDUCT.md) and [Security Policy](SECURITY.md).

## 📝 Citation

If you use WorldFoundry in your research, please cite our [paper](https://arxiv.org/abs/2611.10771) together with the upstream
methods, checkpoints, datasets, and benchmarks your work depends on:

```bibtex
@misc{tian2026worldfoundry,
  title         = {WorldFoundry: Towards Unified Infrastructure for World Intelligence},
  author        = {Juanxi Tian and Jiadong Hou and Baiang Li and Xin Lin and Zhengbo Jiao and Weijia Zhang and Yeying Jin and Truong Nguyen and Haitian Zheng and Zhifei Zhang and Jianfei Yang and Ming-Hsuan Yang and Felix Heide and Wei Xiong and Enze Xie and Wanhua Li},
  year          = {2026},
  eprint        = {2611.10771},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2611.10771}
}
```

## 📄 License

WorldFoundry-authored code is licensed under the [Apache License 2.0](LICENSE), except where a file or component carries a different license notice. Vendored and adapted upstream components remain subject to their original licenses, collected in [THIRD-PARTY-NOTICES](THIRD-PARTY-NOTICES).

## 🙏 Acknowledgement

WorldFoundry builds on upstream world-model, video-generation, reconstruction, and embodied-action projects.
See the model pages and [upstream license guide](https://openenvision.github.io/WorldFoundry/docs/reference/vendored-licenses) for attribution.

We also thank:

- [FastVideo](https://github.com/hao-ai-lab/FastVideo) — a unified inference and post-training framework for accelerated video generation
- [OpenWorldLib](https://github.com/OpenDCAI/OpenWorldLib) — a unified codebase for advanced world models
- [VLA Evaluation Harness](https://github.com/allenai/vla-evaluation-harness) — one framework to evaluate VLA models on robot simulation benchmarks

---

**WorldFoundry** · [Paper](https://arxiv.org/abs/2611.10771) · [GitHub](https://github.com/OpenEnvision/WorldFoundry) · [Documentation](https://openenvision.github.io/WorldFoundry/docs) · [中文文档](https://openenvision.github.io/WorldFoundry/zh/docs) · [Discord](https://discord.gg/ybUQMDA4x)
