# Grounding DINO

Grounding DINO Swin-T (OGC) open-vocabulary detector with its BERT text encoder.

## Used by

| Benchmark / consumer | Metric | Caller |
| --- | --- | --- |
| T2V-CompBench | Numeracy, spatial relationships, motion binding | `worldfoundry/evals/tasks/execution/runners/t2v_compbench/runtime/t2v_compbench/grounded_sam_metrics/` |
| VMBench | Perceptible amplitude score, temporal coherence score | `worldfoundry/evals/tasks/execution/runners/vmbench/runtime/official/` |
| WorldScore | Motion accuracy, object detection | `worldscore/.../metric_impls/` |
| Physical-AI-Bench | PAI-Bench-C `seg_m_iou`, `seg_recall` (through Grounded-SAM2) | `../../segment/grounded_segment_anything/` |

## Usage

```python
from worldfoundry.base_models.perception_core.detection.grounding_dino.paths import checkpoint_path, config_path
from worldfoundry.base_models.perception_core.detection.grounding_dino.util.inference import Model

detector = Model(model_config_path=str(config_path()), model_checkpoint_path=str(checkpoint_path()))
detections, phrases = detector.predict_with_caption(image=bgr_frame, caption="a red car")
```

Capability `grounding_dino`; overrides `WORLDFOUNDRY_GROUNDING_DINO_CKPT` and `WORLDFOUNDRY_GROUNDING_DINO_TEXT_ENCODER_DIR`.
