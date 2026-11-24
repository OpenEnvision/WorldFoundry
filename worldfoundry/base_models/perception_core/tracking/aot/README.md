# DeAOT

Checkpoint paths for the DeAOT-L (R50) object tracker.

## Used by

| Benchmark / consumer | Metric | Caller |
| --- | --- | --- |
| Track-Anything | Mask propagation | `../track_anything/aot_tracker.py` |

## Usage

Stage the weights with `wf models assets --capability aot_deaot_l --execute-downloads`. Override: `WORLDFOUNDRY_DEAOT_R50_CKPT`.
