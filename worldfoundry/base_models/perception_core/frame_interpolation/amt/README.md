# AMT

AMT frame interpolation (AMT-S by default; AMT-L/G and IFRNet configs available).

## Used by

| Benchmark / consumer | Metric | Caller |
| --- | --- | --- |
| VBench, VBench++ (I2V, trustworthiness) | `motion_smoothness` | `worldfoundry/evals/tasks/execution/runners/vbench/runtime/vbench/motion_smoothness.py` |
| DEVIL, HarnessEval-W, MiraBench, WBench | `motion_smoothness` | each runner's `motion_smoothness.py` |
| Open-Sora-Plan (generation) | Frame interpolation | `worldfoundry/synthesis/visual_generation/open_sora_plan/` |

## Usage

`checkpoint_path()` / `config_path()` resolve the weights and configs (`worldfoundry/data/models/runtime/configs/amt/`). Weights: VBench asset `vbench_amt_s_checkpoint`. `WORLDFOUNDRY_AMT_OUTPUT_OFFLOAD` keeps intermediate frames on CPU.
