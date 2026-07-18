# Third-Party License Notes

This file summarizes license terms for third-party and adapted code in WorldFoundry.
It is not a substitute for the full upstream license text retained in each component.

## `LARYBench` (adapted in-tree runtime)

- local_paths: `worldfoundry/evaluation/tasks/execution/runners/larybench`, `worldfoundry/base_models/perception_core/action_recognition/latent_action`, `worldfoundry/data/benchmarks/assets/larybench`
- upstream_url: `https://github.com/meituan-longcat/LARYBench`
- upstream_revision: `e7feaf1b72921ee2c34e489adb0f45faf356ecee`
- license: MIT
- license_file: `thirdparty/licenses/LARYBench-LICENSE`
- commercial_restrictions: none stated in the retained MIT license.
- purpose: Latent-action extraction plus semantic classification and physical-action regression evaluation.
- modifications: Ported to WorldFoundry paths, shared model components, runtime contracts, artifact scorecards, and bounded execution helpers.

## `diff-gaussian-rasterization` (modified fork)

- local_path: `thirdparty/diff-gaussian-rasterization`
- upstream_url: `https://github.com/graphdeco/diff-gaussian-rasterization`
- license: Inria non-commercial research license (Gaussian-Splatting License)
- license_summary: Free for research and evaluation use; commercial use requires explicit consent from Inria. Derivative works must retain the same use limitation. See the upstream `LICENSE.md` for full terms.
- commercial_restrictions: **Non-commercial only** — research and evaluation use permitted; commercial use prohibited without prior written consent from Inria (`stip-sophia.transfert@inria.fr`).
- purpose: CUDA rasterization extension for 3D Gaussian Splatting, modified to output depth alongside color and radii.
- modifications: See `MODIFICATIONS.md`.

## `depth-diff-gaussian-rasterization-min` (modified fork)

- local_path: `thirdparty/depth-diff-gaussian-rasterization-min`
- upstream_url: `https://github.com/graphdeco/diff-gaussian-rasterization`
- license: Inria non-commercial research license (same as upstream diff-gaussian-rasterization)
- license_summary: Same terms as `diff-gaussian-rasterization` above — non-commercial research use only.
- commercial_restrictions: **Non-commercial only** — same as `diff-gaussian-rasterization`.
- purpose: Depth-focused variant of diff-gaussian-rasterization, outputting depth, median depth, and final opacity.
- modifications: See `MODIFICATIONS.md`.

## `simple-knn` (modified fork)

- local_path: `thirdparty/simple-knn`
- upstream_url: `https://github.com/camenduru/simple-knn`
- license: Inria non-commercial research license (Gaussian-Splatting License, same terms as diff-gaussian-rasterization)
- license_summary: Free for research and evaluation use; commercial use requires explicit consent from Inria. Source headers identify Inria GRAPHDECO copyright.
- commercial_restrictions: **Non-commercial only** — research and evaluation use permitted; commercial use prohibited without prior consent.
- purpose: CUDA extension for average nearest-neighbor distance over 3D points.
- modifications: See `MODIFICATIONS.md`.

## `gsplat`

- local_path: `thirdparty/gsplat`
- upstream_url: `https://github.com/nerfstudio-project/gsplat.git`
- license: Apache-2.0
- license_file: `thirdparty/gsplat/LICENSE`
- commercial_restrictions: none stated in the retained Apache-2.0 license.
- purpose: CUDA accelerated Gaussian splatting rasterization with Python bindings.

### Nested Code In `gsplat`

- component: `glm`
- local_path: `thirdparty/gsplat/gsplat/cuda/csrc/third_party/glm`
- upstream_url: `https://github.com/g-truc/glm.git`
- license: MIT or Happy Bunny License
- license_file: `thirdparty/gsplat/gsplat/cuda/csrc/third_party/glm/copying.txt`
- commercial_restrictions: none stated in the MIT license; Happy Bunny License includes a non-binding military-use note.
- purpose: C++ mathematics header dependency used by the CUDA extension.
