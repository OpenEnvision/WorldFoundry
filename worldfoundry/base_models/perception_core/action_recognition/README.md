# `perception_core/action_recognition`

Action classifiers and latent-action encoders that turn videos into action labels or embeddings.

| Path | Model | Used by (benchmark → metric) |
| --- | --- | --- |
| [`umt/`](umt/README.md) | UMT-L/16 Kinetics-400 action classifier | VBench → `human_action` (`worldfoundry/evals/tasks/execution/runners/vbench/runtime/vbench/human_action.py`) |
| [`latent_action/`](latent_action/README.md) | Latent-action encoders (SigLIP2 / DINO backbones) | LARYBench → action classification and regression (`worldfoundry/evals/tasks/execution/runners/larybench/models.py`, `classification_model.py`) |

## Usage

- `umt` weights are the VBench asset `vbench_umt_checkpoint` (capability
  `vbench_metric_checkpoint_assets`).
- `latent_action` reads the SigLIP2 backbone from `WORLDFOUNDRY_SIGLIP2_MODEL_DIR`; LARYBench builds the
  encoders through `worldfoundry/evals/tasks/execution/runners/larybench/models.py`.

Stage weights with `wf models assets --capability <id>` (add `--execute-downloads` to fetch them); `wf models assets --list` shows every id.
