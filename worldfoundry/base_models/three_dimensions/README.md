# `worldfoundry.base_models.three_dimensions`

Inference-only 3D/4D models: monocular and video depth, multi-view reconstruction, point
clouds and Gaussian splats, and camera tracking / SLAM. Three kinds of consumers use them:

- **3D/4D model integrations** — catalog models such as VGGT, CUT3R, Depth Anything 3, or pixelSplat run directly from here.
- **World-model pipelines** — camera-controlled generators (Lyra, HunyuanWorld, …) use depth and pose priors for conditioning and reconstruction.
- **Benchmark metrics** — 3D-consistency and camera-accuracy metrics reconstruct generated videos.

## Layout

| Path | Contents |
| --- | --- |
| `depth/` | Depth estimation behind a shared adapter (`base.py`, `adapter.py`, `alignment.py`): Depth Anything, Depth Pro, Metric3D, MiDaS, MoGe, UniDepth, Video Depth Anything, PriorDA, ViGeo |
| `general_3d/` | Multi-view and dynamic reconstruction: MASt3R, MapAnything, GeoCalib, ViPE, MVDiffusion, LaGeR-NVS, Uni3C |
| `point_clouds/` | Feed-forward point maps and splats: VGGT, VGGT-Omega, Pi3, CUT3R, Stream3R, LoGeR, Infinite-VGGT, LingBot-Map, FlashWorld, Lyra, HunyuanWorld-Mirror / Voyager, WorldFM, SHARP, pixelSplat, Gaussian splatting; plus PLY IO, projection, depth warping |
| `slam/` | DROID-SLAM and MegaSaM camera tracking (including a resident variant) |
| `three_d_four_d/` | `ThreeDFourDRuntimeSynthesis` — runs 3D/4D models as isolated subprocess runtimes with per-run plans and artifacts, plus PyTorch3D compatibility shims |

## Usage

Run an integrated 3D/4D model like any other model:

```bash
wf zoo models --integration-status integrated      # see the three_d_four_d entries
wf run vggt --help
wf run depth-anything-v3 --output-dir tmp/da3
```

Generated point clouds, splats, and camera trajectories open in the Studio workspace's
point-cloud and Rerun viewers.

## Conventions

- Inference only: keep geometry helpers needed at inference; no training datasets, losses, or experiment scripts.
- Reuse camera, pose, trajectory, and warping utilities from `worldfoundry/core/geometry` instead of re-implementing them per model.
- Checkpoints resolve through the Hugging Face cache or `WORLDFOUNDRY_CKPT_DIR`; retain upstream attribution in vendored code.
