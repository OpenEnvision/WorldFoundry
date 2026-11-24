# Latent-action encoders

Encoders that embed robot/egocentric video into latent actions (SigLIP2 and DINO backbones).

## Used by

| Benchmark / consumer | Metric | Caller |
| --- | --- | --- |
| LARYBench | Action classification (`top1_accuracy`) and regression | `worldfoundry/evals/tasks/execution/runners/larybench/models.py`, `classification_model.py` |

## Usage

LARYBench builds the encoders through its runner. The SigLIP2 backbone is read from `WORLDFOUNDRY_SIGLIP2_MODEL_DIR`.
