# `mllm/perceptionlm`

Meta Perception Language Model (PLM) inference runtime: model, tokenizer, and video/image
transforms.

## Used by

| Consumer | Metric | Caller |
| --- | --- | --- |
| VQAScore (t2v_metrics) | VQAScore with a PerceptionLM backend | [`perception_core/video_text/vqa_score`](../../../perception_core/video_text/vqa_score/) `models/vqascore_models/perceptionlm_model.py` |

## Usage

Selected as a VQAScore backend (`get_score_model(...)` in the t2v_metrics scorers); checkpoints
come from the Hugging Face cache. No benchmark selects it by default.
