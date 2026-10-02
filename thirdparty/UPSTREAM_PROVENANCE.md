# Third-Party Upstream Provenance

This file records the local provenance review for vendored code under `thirdparty/`.
Entries are based only on files present in this repository.

## `simple-knn` (modified fork)

- upstream_url: `https://github.com/camenduru/simple-knn`
- local_path: `thirdparty/simple-knn`
- fork_status: **modified** — not the unmodified upstream. See `MODIFICATIONS.md` for details.
- modifications: Build configuration and CUDA kernels adapted for WorldFoundry integration and compatibility with the depth-modified `diff-gaussian-rasterization` fork.
- evidence: Source headers identify Inria GRAPHDECO copyright.
- purpose: CUDA extension for average nearest-neighbor distance over 3D points.
- license_summary: see root `THIRD-PARTY-NOTICES`.

## `diff-gaussian-rasterization` (modified fork — depth and opt-in auxiliary outputs)

- upstream_url: `https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/`
- upstream_repo: `https://github.com/graphdeco/diff-gaussian-rasterization`
- local_path: `thirdparty/diff-gaussian-rasterization`
- fork_status: **modified** — not the unmodified upstream. See `MODIFICATIONS.md` for details.
- modifications: The default forward pass returns `depth` alongside `color` and
  `radii`; an explicit Python opt-in additionally returns `median_depth` and
  `final_opacity`. The backward pass continues to propagate canonical color and
  depth gradients, and the upstream low-pass covariance filter remains enabled.
- evidence: Source headers identify Inria GRAPHDECO copyright; `MODIFICATIONS.md` documents changes.
- purpose: CUDA rasterization extension for 3D Gaussian Splatting with depth
  rendering, used by pixelSplat and by WonderWorld's opt-in visibility outputs.
- nested_third_party:
  - `third_party/stbi_image_write.h`: single-header image writer from stb.
  - `third_party/glm`: listed in `.gitmodules` as `https://github.com/g-truc/glm.git`; GLM source files are present in the current tree.
- license_summary: see root `THIRD-PARTY-NOTICES`.

## `gsplat`

- upstream_url: `https://github.com/nerfstudio-project/gsplat.git`
- local_path: `thirdparty/gsplat`
- source_commit: `b5392febf6047655c18db17693636cd21bbe58c0`
- evidence: shallow clone HEAD recorded in `.worldfoundry_upstream_commit`; license terms are recorded in root `THIRD-PARTY-NOTICES`.
- purpose: CUDA accelerated Gaussian splatting rasterization with Python bindings.
- nested_third_party:
  - `gsplat/cuda/csrc/third_party/glm`: listed in upstream `.gitmodules` as `https://github.com/g-truc/glm.git`.
- license_summary: see root `THIRD-PARTY-NOTICES`.

## `SageAttention` (distribution quarantined)

- upstream_repo: `https://github.com/thu-ml/SageAttention`
- local_path: `thirdparty/SageAttention`
- source_revision: **not recorded in the imported snapshot**
- evidence: `setup.py` identifies the SageAttention team, upstream URL, and an Apache-2.0 declaration; source headers identify the 2024/2025 SageAttention team.
- provenance_gap: The imported tree does not retain the upstream `LICENSE` or `README.md`, so an exact source revision and complete license bundle cannot be audited locally.
- distribution_policy: Excluded from source distributions by `MANIFEST.in` until the exact revision and full upstream license are restored.
