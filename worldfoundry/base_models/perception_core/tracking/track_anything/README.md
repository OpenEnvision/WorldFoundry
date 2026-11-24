# Track-Anything

Track-Anything: Grounding DINO detection, SAM masks, and DeAOT propagation.

## Used by

| Benchmark / consumer | Metric | Caller |
| --- | --- | --- |
| ViPE | Dynamic-object masks | `three_dimensions/general_3d/vipe/pipeline/processors.py` |
| `track-anything-prior` model | Geometry prior for generation | `worldfoundry/synthesis/visual_generation/geometry_priors.py` |

## Usage

Not a benchmark metric. Needs capabilities `grounding_dino`, `sam_v1`, and `aot_deaot_l`.
