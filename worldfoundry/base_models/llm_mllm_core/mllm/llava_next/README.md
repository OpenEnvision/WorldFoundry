# `mllm/llava_next`

In-tree LLaVA-NeXT / LLaVA-OneVision / LLaVA-Video inference package (`llava/`): model builders,
conversation templates, and video preprocessing. No training code.

## Used by

| Benchmark | Checkpoint | Metrics (dimensions) | Caller |
| --- | --- | --- | --- |
| VBench-2.0 | `lmms-lab/LLaVA-Video-7B-Qwen2` | Complex Landscape, Complex Plot, Composition, Dynamic Attribute, Dynamic Spatial Relationship, Human Clothes, Human Interaction, Material, Mechanics, Motion Order Understanding, Motion Rationality, Thermotics | `worldfoundry/evals/tasks/execution/runners/vbench_2_0/runtime/vbench2/` |
| T2V-CompBench | `liuhaotian/llava-v1.6-34b` | Consistent Attribute, Dynamic Attribute, Action Binding, Object Interaction | `worldfoundry/evals/tasks/execution/runners/t2v_compbench/runtime/t2v_compbench/mllm_metrics/` |
| T2VWorldBench | `liuhaotian/llava-v1.6-34b` | MLLM judge scores | `worldfoundry/evals/tasks/execution/runners/t2vworldbench/runtime/t2vworldbench/eval.py` |
| VQAScore (t2v_metrics) | LLaVA-OneVision / LLaVA-Video backends | VQAScore | [`perception_core/video_text/vqa_score`](../../../perception_core/video_text/vqa_score/) |

## Usage

Import from `worldfoundry.base_models.llm_mllm_core.mllm.llava_next.llava` (for example
`llava.model.builder.load_pretrained_model`). The VBench-2.0 checkpoint is part of
`wf models assets --capability vbench_metric_checkpoint_assets` (asset
`vbench2_llava_video_qwen2_model`); T2V-CompBench and T2VWorldBench load
`llava-v1.6-34b` through the Hugging Face cache.
