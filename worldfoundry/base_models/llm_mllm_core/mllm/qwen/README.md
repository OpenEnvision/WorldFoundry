# `mllm/qwen`

Qwen-family components. Some are benchmark judges, others are text encoders or policy backbones
for generation models.

## Contents

| Path | What it is | Used by |
| --- | --- | --- |
| `qwen_vl_utils.py` | Qwen-VL video/image preprocessing (`process_vision_info`) | Every Qwen2.5-VL / Qwen3-VL caller below, and the Wan prompt expander |
| `wbench_visual_plausibility.py` | PAVRM visual-plausibility scorer (Qwen3-VL) | WBench — Visual Plausibility (`worldfoundry/evals/tasks/execution/runners/wbench/runtime/wbench/src/metrics/physical/visual_plausibility.py`) |
| *(loaded with `transformers`)* | Qwen2.5-VL / Qwen3-VL judges | Physical-AI-Bench — PAI-Bench-G VQA accuracy, Qwen2.5-VL-72B-Instruct (`physical_ai_bench/.../judges.py`); EWMBench — scene captions, Qwen2.5-VL (`ewmbench/.../caption.py`); VideoVerse — local VLM judge, Qwen2.5-VL (`videoverse/videoverse_local_vlm.py`); WorldOlympiad — QwenVL judge, Qwen3-VL-8B-Instruct |
| `cosmos_reason1/` | Cosmos-Reason1 (Qwen2.5-VL based) inference | Cosmos-Predict2.5 text encoder (`diffusion_model/models/encoders/cosmos2p5`) |
| `qwen_image/` | Qwen-Image DiT, VAE, text encoder, ControlNet, panorama pipeline | HunyuanWorld 2.0 (`worldfoundry/pipelines/hunyuan_world/pipeline_hy_world_2p0.py`) |
| `qwen3_backbone.py` | Qwen3 language backbone | GR00T policy (`synthesis/action_generation/gr00t`) |
| `beingh/` | Qwen2 / Qwen3 NaViT backbones and tokenizers | Being-H0.5 policy (`synthesis/action_generation/being_h05`) |
| `qwenpi_policy.py` | Qwen-π / π0 inference policy and a standalone policy server (`QwenPiServer`) | Run as a script; no in-repo importer |

## Usage

Benchmark judges are staged through the capability registry:

```bash
wf models assets --capability qwen3_vl_8b_instruct       # WorldOlympiad judge
wf models assets --capability qwen2_5_vl_72b_instruct    # PAI-Bench-G VQA judge
wf models assets --capability wbench_pavrm_qwen3vl       # WBench visual plausibility
```

Qwen-π policies read `WORLDFOUNDRY_QWEN_PI_MODEL_PATH` and `WORLDFOUNDRY_QWEN_PI_BASE_MODEL_PATH`.
