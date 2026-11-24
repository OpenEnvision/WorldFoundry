# UniWorld-View inference runtime

Adapted from PKU-YuanGroup/UniWorld-View, revision `660212c48bff7c91a2ea5c7d5f3785af581100e2`
(Apache License 2.0; see root `THIRD-PARTY-NOTICES`). `configs/`, `model/`, `utils/`, `demo.py`
and `inference.py` come from the repository root; `painter.py` comes from `extern/sam2/`.
Imports are package-qualified, and the retained code follows the repository's formatting rules.

The package contains the inference entrypoint, UniView's Diffusers WanVACEPipeline backend,
camera controls, artifact handling and the warp, mesh and hybrid renderers. It retains only
module definitions and methods reachable from the package entrypoint, including dynamic targets.

## Shared components

- SAM 2 uses `base_models/perception_core/segment/sam2` and its packaged
  `configs/sam2/sam2_hiera_l.yaml`. The canonical builder qualifies the configuration's targets
  before Hydra instantiation.
- MoGe-2 uses `base_models/three_dimensions/depth/moge/variants/uniworld_view.py`. The variant
  preserves the pinned revision's truncated token grid and explicit-matmul attention fallback;
  xFormers remains the preferred path when installed. Encoder, heads and checkpoint loading are
  inherited from canonical MoGe. The DINOv2 backbone belongs to
  `base_models/perception_core/general_perception/dinov2`.
- STream3R uses `base_models/three_dimensions/point_clouds/stream3r`, corresponding to the
  upstream submodule revision `7bdca8cf` (S-Lab License 1.0).

The pipeline initializes STream3R for dynamic view, forwards the requested seed to diffusion,
and writes final single-view video at the requested FPS. All weights are read from explicitly
resolved local paths. There are no Gradio entrypoints, training modules or alternate foundation
model implementations in this package.
