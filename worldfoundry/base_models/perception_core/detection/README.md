# `perception_core/detection`

Open-vocabulary object detectors: given a text phrase, return boxes in a frame.

| Path | Model | Used by (benchmark → metric) |
| --- | --- | --- |
| [`grounding_dino/`](grounding_dino/README.md) | Grounding DINO (Swin-T OGC) | T2V-CompBench → numeracy, spatial relationships, motion binding; VMBench → perceptible amplitude score, temporal coherence score; WorldScore → motion accuracy, object detection; Physical-AI-Bench → PAI-Bench-C `seg_m_iou` / `seg_recall` (through Grounded-SAM2) |
| [`yolo_world/`](yolo_world/README.md) | YOLO-World detector | VBench-2.0 → human anatomy (through `human_anatomy/vit_detector`) |

Grounding DINO is also the detector inside `segment/grounded_segment_anything` and
`tracking/track_anything`.

## Usage

```python
from worldfoundry.base_models.perception_core.detection.grounding_dino.util.inference import Model
from worldfoundry.base_models.perception_core.detection.grounding_dino.paths import checkpoint_path, config_path

detector = Model(model_config_path=str(config_path()), model_checkpoint_path=str(checkpoint_path()))
detections, phrases = detector.predict_with_caption(image=bgr_frame, caption="a red car")
```

- Grounding DINO: capability `grounding_dino`; overrides `WORLDFOUNDRY_GROUNDING_DINO_CKPT` and
  `WORLDFOUNDRY_GROUNDING_DINO_TEXT_ENCODER_DIR` (BERT text encoder).
- YOLO-World: VBench-2.0 asset `vbench2_yolo_world_checkpoint` (capability
  `vbench_metric_checkpoint_assets`).

Stage weights with `wf models assets --capability <id>` (add `--execute-downloads` to fetch them); `wf models assets --list` shows every id.
