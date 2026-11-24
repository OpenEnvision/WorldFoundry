# Third-Party Upstream Provenance

This file is the index for `thirdparty/`. Each component is brought in one way: as a
pinned Git submodule (unmodified upstream), as vendored source (with any changes recorded
in its `MODIFICATIONS.md`), or not at all (provided by the user's environment). License
texts are collected in the root `THIRD-PARTY-NOTICES`.

| Path | How it is included | Upstream | Pinned revision | Modified | License |
| --- | --- | --- | --- | --- | --- |
| `matrix-game` | Git submodule (optional, shallow) | `SkyworkAI/Matrix-Game` | `5446525c80da2a4a2f87801addbbc7faa8d2486e` | no | Apache-2.0 |
| `gsplat` | Vendored source | `nerfstudio-project/gsplat` | `b5392febf6097655c18db17693636cd21bbe58c0` | no | Apache-2.0 |
| `fastvideo-kernel` | Vendored source; CUTLASS / ThunderKittens headers fetched by `fetch_headers.sh` | `hao-ai-lab/FastVideo` (`fastvideo-kernel/`) | `1b2b2a0161bc6b3b80158d1fa6380a051c6530c7` | yes | Apache-2.0 |

## `matrix-game` (optional Git submodule)

- upstream_url: `https://github.com/SkyworkAI/Matrix-Game.git`
- local_path: `thirdparty/matrix-game`
- source_commit: `5446525c80da2a4a2f87801addbbc7faa8d2486e`
- fork_status: **unmodified** — the gitlink fixes the complete upstream repository.
- fetch: `git submodule update --init thirdparty/matrix-game` (shallow, as declared in `.gitmodules`).
- purpose: Optional native game source under `Matrix-Game-4/game/`; WorldFoundry's Python renderer never imports this checkout.
- assets: Downloaded separately through the upstream `releases/game-assets.json` manifest; not stored in Git or Python distributions.
- license_summary: Apache-2.0, with nested dependency licenses retained upstream; see root `THIRD-PARTY-NOTICES`.

## `gsplat` (vendored, unmodified)

- upstream_url: `https://github.com/nerfstudio-project/gsplat.git`
- local_path: `thirdparty/gsplat`
- source_commit: `b5392febf6097655c18db17693636cd21bbe58c0`
- evidence: shallow clone HEAD recorded in `.worldfoundry_upstream_commit`; license terms are recorded in root `THIRD-PARTY-NOTICES`.
- purpose: CUDA accelerated Gaussian splatting rasterization with Python bindings.
- nested_third_party:
  - `gsplat/cuda/csrc/third_party/glm`: vendored in-tree from `https://github.com/g-truc/glm.git`. The upstream `gsplat/.gitmodules` that declares it is kept verbatim but is inactive here, because `gsplat` itself is vendored rather than a submodule.
- license_summary: see root `THIRD-PARTY-NOTICES`.

## `fastvideo-kernel` (vendored, modified)

- upstream_url: `https://github.com/hao-ai-lab/FastVideo.git` (path `fastvideo-kernel/`)
- local_path: `thirdparty/fastvideo-kernel`
- source_commit: `1b2b2a0161bc6b3b80158d1fa6380a051c6530c7`
- fork_status: **modified** — see `MODIFICATIONS.md`.
- build_dependencies: CUTLASS 4.3.0 (`e67e63c331d6e4b729047c95cf6b92c8454cba89`, BSD-3-Clause) and ThunderKittens (`6c27e28c8115d1839d9eeeb530913c184a75fc87`, MIT), fetched into the git-ignored `include/` by `fetch_headers.sh`.
- purpose: Hopper sparse attention (STA / VSA) for the optional approximate-attention lane in `worldfoundry/base_models/diffusion_model/optimizations/approximate_attention.py`.
- license_summary: Apache-2.0; `LICENSE` links to the root `THIRD-PARTY-NOTICES`, which retains the verbatim text.

## Not distributed in this repository

The following directories are git-ignored. Code that uses them expects the package to be
installed in the active environment; it is not shipped in Git or Python distributions.
Their license records remain in the root `THIRD-PARTY-NOTICES` as retained text.

- `diff-gaussian-rasterization` — depth-enabled Gaussian rasterization fork (Inria non-commercial license) used by pixelSplat; provide it in the pixelSplat environment.
- `simple-knn` — Inria GRAPHDECO nearest-neighbor CUDA extension (non-commercial license).
- `SageAttention` — `https://github.com/thu-ml/SageAttention`; install the `sageattention` package where an optional SageAttention backend is wanted.
- `flash-attention` — optional local FlashAttention checkout; not third-party source of this repository.
