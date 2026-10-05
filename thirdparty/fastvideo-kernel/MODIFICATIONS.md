# fastvideo-kernel — WorldFoundry vendoring & modifications

Vendored from **hao-ai-lab/FastVideo** `fastvideo-kernel/`.

- upstream_repo: `https://github.com/hao-ai-lab/FastVideo.git`
- upstream_commit: `1b2b2a0161bc6b3b80158d1fa6380a051c6530c7`
- upstream_path: `fastvideo-kernel/`
- license: Apache-2.0 (see `LICENSE`)
- purpose: Hopper (sm_90a) sparse attention — Sliding Tile Attention (STA) and
  Video Sparse Attention (VSA) — consumed by
  `worldfoundry/base_models/diffusion_model/optimizations/approximate_attention.py`
  as the optional, opt-in approximate self-attention lane. When this package is
  absent the approximate lane falls back to exact attention.

## Bundled third-party headers (git submodules, not vendored in-tree)

The 136 MiB of CUTLASS + ThunderKittens headers are NOT committed here. They are
declared as git submodules pinned to the exact upstream-tested commits, mounted
at the paths the kernel's `CMakeLists.txt` expects (`include/cutlass`,
`include/tk`):

| submodule | path | url | commit |
|---|---|---|---|
| CUTLASS 4.3.0 | `include/cutlass` | `https://github.com/NVIDIA/cutlass.git` | `e67e63c331d6e4b729047c95cf6b92c8454cba89` |
| ThunderKittens | `include/tk` | `https://github.com/HazyResearch/ThunderKittens.git` | `6c27e28c8115d1839d9eeeb530913c184a75fc87` |

Fetch before building:

```
git submodule update --init --recursive thirdparty/fastvideo-kernel/include/cutlass thirdparty/fastvideo-kernel/include/tk
```

CUTLASS is BSD-3-Clause; ThunderKittens is MIT. See each submodule's upstream license.

## Modifications (vs upstream)

1. **torch 2.5 `custom_op` schema compatibility** —
   `python/fastvideo_kernel/block_sparse_attn.py`: return annotations on
   `@torch.library.custom_op`-decorated functions changed from
   `typing.Tuple[...]` to the builtin `tuple[...]`. torch 2.5's
   `torch._library.infer_schema` cannot resolve `typing.Tuple` as a type and
   raises `ValueError: Unsupported type annotation Tuple[...]` at import time.
   The builtin `tuple[...]` is what `infer_schema` expects; behaviour is
   otherwise identical. This is the only functional change.

2. **Bundled headers de-vendored to submodules** — the in-repo `include/cutlass`
   and `include/tk` header trees were removed and replaced by the pinned
   submodules above, to keep the WorldFoundry repository small.

## Build (Hopper / sm_90a)

Built and verified against WorldFoundry's `worldfm` env (torch 2.5.1+cu121,
Python 3.10, H100) with the CUDA 12.6 toolkit + devtoolset-11 gcc:

```
source /opt/rh/devtoolset-11/enable
export CUDA_HOME=/usr/local/cuda-12.6 PATH=$CUDA_HOME/bin:$PATH
export TORCH_CUDA_ARCH_LIST="9.0a"
pip install ./thirdparty/fastvideo-kernel --no-build-isolation --no-deps
```

Verified: `import fastvideo_kernel` exposes `sliding_tile_attention`,
`video_sparse_attn`, `video_sparse_attn_bshd`; VSA H100 forward runs; and the
WorldFoundry approximate-attention lane reports `effective_kernel="vsa"` with
sparse calls (no fallback).
