# ViCLIP

ViCLIP video–text model (InternVid-10M-FLT).

## Used by

| Benchmark / consumer | Metric | Caller |
| --- | --- | --- |
| VBench | `overall_consistency`, `temporal_style` | `worldfoundry/evals/tasks/execution/runners/vbench/runtime/vbench/` |
| VBench++ trustworthiness | Culture fairness | `worldfoundry/evals/tasks/execution/runners/vbench_plus_plus/runtime/vbench2_beta_trustworthiness/culture_fairness.py` |
| MiraBench | Text–video consistency | `worldfoundry/evals/tasks/execution/runners/mirabench/runtime/mirabench/evaluation/text_video_consistency.py` |
| DEVIL | Text alignment | `worldfoundry/evals/tasks/execution/runners/devil_dynamics/runtime/official/viclip/` |

## Usage

`from worldfoundry.base_models.perception_core.video_text.viclip import ViCLIP, SimpleTokenizer, checkpoint_path`. Weights: VBench asset `vbench_viclip_checkpoint` (capability `vbench_metric_checkpoint_assets`); `WORLDFOUNDRY_VICLIP_BPE_PATH` overrides the tokenizer vocabulary.
