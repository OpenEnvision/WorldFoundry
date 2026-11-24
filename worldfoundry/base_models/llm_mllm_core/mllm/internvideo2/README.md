# `mllm/internvideo2`

InternVideo2 video–text encoder (vision tower, text encoder, and the stage-2 retrieval head), used
as a video–text similarity and matching model by evaluation code.

## Used by

| Benchmark | Metric | Caller |
| --- | --- | --- |
| ChronoMagic-Bench | MTScore (metamorphic amplitude) | `worldfoundry/evals/tasks/execution/runners/chronomagic_bench/runtime/chronomagic_bench/MTScore/` |
| PhyGenBench | PhyGenEval video-level physics score | `worldfoundry/evals/tasks/execution/runners/phygenbench/runtime/phygenbench/PhyGenEval/video/MTScore/InternVideo_physical.py` |
| GenAI-Bench / t2v_metrics | InternVideo2 ITM and CLIP-style scores | `worldfoundry/evals/tasks/execution/runners/_scorers/itm_score/`, `worldfoundry/evals/tasks/execution/runners/_scorers/clip_score/` |

## Usage

`evaluation` exposes the inference entry points and `download_internvideo2` resolves the
checkpoint. Benchmarks reach it through their runners; for t2v_metrics the ITM model is staged as
`wf models assets --capability t2v_metrics_internvideo2_itm`. The UMT backbone it shares lives in
[`umt`](../umt/).
