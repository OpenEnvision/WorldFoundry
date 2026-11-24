# `mllm/umt`

UMT (Unmasked Teacher) video–text model: video encoder, text encoder, and retrieval/ITM heads.

## Used by

| Benchmark | Metric | Caller |
| --- | --- | --- |
| ChronoMagic-Bench | MTScore | `worldfoundry/evals/tasks/execution/runners/chronomagic_bench/runtime/chronomagic_bench/MTScore/configs/utils.py` |
| t2v_metrics scorers | UMT ITM and CLIP-style scores | `worldfoundry/evals/tasks/execution/runners/_scorers/itm_score/`, `worldfoundry/evals/tasks/execution/runners/_scorers/clip_score/` |

It is also the backbone that [`internvideo2`](../internvideo2/) builds on. VBench's Human Action
dimension uses the separate UMT classifier in
[`perception_core/action_recognition/umt`](../../../perception_core/action_recognition/umt/).

## Usage

`evaluation` exposes the inference entry points and `download_umt` resolves the checkpoint;
benchmarks call them through their runners.
