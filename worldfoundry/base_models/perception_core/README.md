# `worldfoundry.base_models.perception_core`

Inference-only perception models shared across WorldFoundry. Benchmark metrics are their
main consumer — detection, segmentation, tracking, optical flow, and quality models turn
generated videos into scores — and model pipelines reuse them for preprocessing (masks,
tracks, captions, embeddings).

## Layout

| Path | Models |
| --- | --- |
| [`general_perception/`](general_perception/README.md) | Feature backbones and embeddings: DINO, DINOv2, DINOv3, OpenAI CLIP, OpenCLIP, UniCeption, Qwen3-VL entity features |
| [`detection/`](detection/README.md) | Open-vocabulary detection: Grounding DINO, YOLO-World |
| [`segment/`](segment/README.md) | SAM v1, SAM 2, SAM 3, RepViT-SAM, Grounded-SAM, scene segmentation |
| [`tracking/`](tracking/README.md) | CoTracker, DOT, AOT, Track-Anything |
| [`optical_flow/`](optical_flow/README.md) | RAFT, SEA-RAFT, FlowFormer++ |
| [`video_quality/`](video_quality/README.md) | DOVER, FAST-VQA, LPIPS, DreamSim, HPSv3, TransNetV2, FETV StyleGAN-V |
| [`video_text/`](video_text/README.md) | Video–text alignment: ViCLIP, VQAScore, BLIP-2 features, FETV BLIP |
| [`captioning/`](captioning/README.md) | Tag2Text, GRiT |
| [`action_recognition/`](action_recognition/README.md) | UMT, latent-action models |
| [`frame_interpolation/`](frame_interpolation/README.md) | AMT, VFIMamba |
| [`face/`](face/README.md) | ArcFace, RetinaFace |
| [`human_anatomy/`](human_anatomy/README.md) | ViT-based body detector |
| [`safety/`](safety/README.md) | NudeNet |
| [`visualization/`](visualization/README.md) | Colormaps and rendering helpers for perception outputs |

Each folder's README lists its models, the benchmarks and metrics that use them, and how to
stage their weights.

## Usage

These models are normally reached indirectly: through a benchmark metric
(`wf metric list`, `wf score ...`) or a pipeline that needs preprocessing. When importing
one directly, keep the import local to the call site — perception stacks are heavy, and the
package is designed so that importing `worldfoundry` does not pull them in.

## Conventions

- Inference only: keep transforms needed at inference time; no training datasets, losses, or experiment scripts.
- Checkpoints resolve through the Hugging Face cache or `WORLDFOUNDRY_CKPT_DIR`, not machine-local paths.
- Retain upstream attribution and license notices in vendored code.
