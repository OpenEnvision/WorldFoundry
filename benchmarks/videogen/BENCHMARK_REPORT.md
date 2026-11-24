# WorldFoundry VideoGen & World Models — H200 Inference Acceleration Benchmarks

This document reports objective, reproducible inference acceleration benchmarks for WorldFoundry across **15 foundation models** (11 flagship video generation models + 4 real-time autoregressive / interactive world models), alongside a candid comparison with peer open-source and vendor inference frameworks.

## Methodology & Hardware Disclosures

- **Hardware & Software Stack**: Bare-metal **NVIDIA H200 SXM (141 GiB HBM3e, SM90, NVSwitch)**, PyTorch `2.11.0+cu128`, CUDA 12.8, Triton 3.6.0, Hopper FlashAttention-3 `3.0.1` (`flash_attn_interface`), SM90 CUTLASS `sol_attn`, and in-tree Hopper TMA `PISA`.
- **Contention Transparency**:
  - **Zero-Contention Clean (`Clean`)**: Measured on dedicated GPUs with zero competing processes (`external_gpu_contention = false`). Covers **11 of 15 models** plus 1× H200 `LTX-2.5`.
  - **Paired Same-Session (`Paired`)**: On `Wan2.2-TI2V-5B`, `Wan2.2-T2V-A14B`, `HunyuanVideo-13B`, and `2× H200 LTX-2.5 CFG=2`, a background training job shared the host (~33.5 GiB/GPU). For those runs, baseline `L0` and accelerated tiers were timed back-to-back on the exact same GPU(s). We report **paired relative speedups (`L0 / L_opt`)** and omit contended absolute seconds from exclusive latency claims (empirical cross-checks on `SANA-Video` `2.862×` contended vs `2.836×` clean and `LingBot-Video` `3.059×` contended vs `3.069×` clean confirm `<1%` ratio variance). Functional correctness, CUDA Graph capture/replay, zero-fallback receipts, PSNR/SSIM, and visual audits are unaffected by GPU sharing.
- **Cross-Architecture Comparison Caveat**: Peer publications span different GPUs (Blackwell **B200 SM100 with native NVFP4**, **H200 SXM**, **H100 SXM**), resolutions, step counts, and approximation budgets. Where NVIDIA **SOL Engine** leverages Blackwell NVFP4, WorldFoundry runs **Hopper SM90 Rowwise FP8 W8A8 with full FP32 accumulation (`use_fast_accum=False`)** on H200. SM100 NVFP4 weight packing and dispatch are implemented and unit-tested, pending physical B200 benchmarking.
- **Zero-Collapse Quality Gate**: Every reported tier requires auditable zero-fallback runtime receipts (`fallbacks: []`, `capture_failed: 0`, `dense_fallback_calls: 0`) and direct visual inspection of rendered RGB frames. Aggressive settings that introduce visual artifacts or late-frame drift are explicitly rejected (see [Quality Engineering & Rejected Configurations](#quality-engineering--rejected-configurations)).

---

## 1. Foundation Video Generation Models (11 Models on NVIDIA H200)

| # | Model (Pretrained Checkpoint) | Workload | Reference Stack (SOL Engine Published Ref) | WorldFoundry H200 Stack & Measured Result | Regime | Quality & Receipts |
| ---: | --- | --- | --- | --- | --- | --- |
| 1 | **Cosmos3-Super (64B, 64L)** | `704×1280×33f`, 35 steps | TeaCache + step-selective NVFP4 (`~2.27×`) | **1× H200**: `77.94s → 33.02s` (**`2.36×`**); bitwise-exact `und` static KV cache (4,352 layers skipped, `-56.2 GiB` VRAM `122.8→66.6 GiB`) + step-selective FP8 + FA3 GQA + TeaCache(0.18) + Piecewise Graph | Clean | `fp8_und_kv (2/2)`: `20.35 dB / SSIM 0.9286`; `sol_balanced`: `16.42 dB / 0.8255`; `fallbacks=[]` |
| 2 | **LTX-2.3 (22B A+V Distilled)** | `768×1280×65f`, 8+3 steps | Fusion + cache + PISA + NVFP4 + token-prune (`~2.38×`) | **1× H200**: `4.648s` (W1 L0) / `4.481s` (W3 L0) → **`1.974s` (`2.35×` vs W1 / `2.27×` vs W3 / `2.51×` DiT)** w/ FP8+PISA(`d=0.10`)+**FBCache(0.16)** (default); **`1.782s` (`2.61×` vs W1 / `2.52×` vs W3 / `2.86×` DiT)** w/ **TeaCache(0.15)** (no pruning) | Clean | Dual T2V + real I2V (`frame0_psnr=39.19 dB`), PISA `90/90` zero fallbacks; FBCache `audio_rms_ratio=0.839` (TeaCache `0.55`) |
| 3 | **SANA-Video (2B)** | `480×832×81f`, 50 steps, CFG 6 | EasyCache + fusion + compile (`~2.77×`) | **1× H200**: `55.76s → 27.19s` (`2.05×` lossless fusion+FA3+compile, `27.99 dB`) → **`19.66s` (`2.84×`)** w/ EasyCache(0.08) + compiled FA3 (`80/80` verified) | Clean | `27.43 dB / SSIM 0.9891`; `fallbacks=[]` |
| 4 | **Wan2.2-TI2V-5B** | `704×1280×121f`, 50 steps | EasyCache + fusion + compile (`~2.89×`) | **1× H200**: Pure BF16 EasyCache(0.24, `d=2`) + PISA(0.15) + Piecewise Graph + compiled VideoVAE38 (**`3.01×` E2E / `3.04×` DiT**, VAE `2.72×`); Rowwise FP8 **`3.06×` E2E / `3.09×` DiT**; no-PISA FP8 `compile_fullgraph` **`2.87×` E2E / `2.89×` DiT** | Paired | Sharp visual grids; real TI2V I2V (`per-token`) verified; `fallbacks=[]` |
| 5 | **Wan2.2-T2V-A14B (Dual-Expert 28B)** | `480×832×81f`, 40 steps | Fusion + EasyCache + PISA (`~2.17×`) | **1× H200**: PISA + EasyCache (expert-boundary reset) + compiled VideoVAE_ (**`2.29×` E2E / `2.29×` DiT**, VAE `2.96×`); **2× H200 `CFG=2`**: **`4.27×` vs 1× L0 / `2.17×` vs 2× L0** | Paired | `SSIM 0.711`, `expert_boundary_resets=1`, zero cross-expert bleed; `fallbacks=[]` |
| 6 | **LingBot-Video (1.3B)** | `480×832×121f`, 40 steps, CFG 3 | Fusion + refiner PISA + EasyCache (`~2.60×`) | **1× H200**: `114.59s → 86.94s` (`1.32×` exact, `27.44 dB`) → `67.33s` (`1.70×` PISA, `20.11 dB`) → **`37.03s` (`3.10×` E2E / `3.22×` DiT)** w/ fusion + PISA + EasyCache + Piecewise Graph | Clean | `19.39 dB / SSIM 0.9207`; `fallbacks=[]` |
| 7 | **HunyuanVideo-13B (20D+40S)** | `720×1280×65f`, 50 steps | Fusion + TeaCache + Sol-Attn (`~5.03×`) | **1× H200**: Sol-Attn(`tau=1.0`, `1×8×8`) + TeaCache + FP8 + Piecewise Graph (**`5.25×` DiT**; conservative `4.33×` at `25.98 dB / 0.9431`); BF16 Tiled VAE `1.29×` (`51.8 dB`); **2× H200 `SP=2`**: **`8.59×`** vs 1× L0 | Paired | `22.51 dB / SSIM 0.9097`, `sp-sync` TeaCache consensus, `0.56s` clean SP teardown; `fallbacks=[]` |
| 8 | **Wan2.1-T2V-14B** | `480p` & `720p`, 81f, 50 steps | Fusion + EasyCache + Sol-Attn (`~3.48×`) | **4× H200**: 480p `82.00s` (4× FA3 exact, **94.4% eff.**) → **`22.50s` (`3.64×` vs 4× FA3 / `13.86×` vs 1× L0 `311.81s`)**; 720p `316.90s` → **`79.02s` (`4.01×` vs 4× exact)**; 1× H200 `EasyCache.run_stack + Sol-Attn + Piecewise Graph` SM90 functional-verified (`72` replays, `28` skips, `0` eager) | Clean (4×) | Crisp 480p/720p & 1× EasyCache grids, FP64 Triton RoPE (`13.63×`), `use_fast_accum=False`; `fallbacks=[]` |
| 9 | **LTX-2.5 (22B Dev & Distilled)** | `768×1280×65f`, 24-step CFG=3 / 8+3 | Parallel + FBCache + fusion (`~4.68×`) | **1× H200 (Clean)**: `dev 24-step` `30.38s → 13.34s` (**`2.28×` E2E / `2.34×` DiT**), `distilled 8+3` `3.68s → 2.29s` (`1.61×`); **2× H200 `CFG=2` (Paired)**: FBCache(0.14) + FP8 + Sol-Attn + Graph + compiled VAE **`5.46×` E2E / `5.33×` DiT** (`39.29 GiB/GPU`) | Clean (1×) / Paired (2×) | `26.26 dB / SSIM 0.9907` (2× CFG=2), `zero_fallbacks_verified=True`, `cfg_steps=24` |
| 10 | **MiniMax-H3 (31B FL2VA)** | `512p 4s` & `768p 5s` | Context parallel + fusion + Sol-Attn + FBCache (`~3.95×`) | 36L VideoVAE **`10.97×`** (`22.01s→2.01s`, `58.36 dB`); **1× H200**: 768p 20s **`38.64s` (`3.22×`)**, 512p 50s **`20.44s` (`3.60×`)**; **2× H200 `SP=2`**: 768p 20s **`20.97s` (`5.93×`)**, 768p 50s **`41.88s` (`7.42×`)**, 512p CFG=2 **`14.36s` (`4.20×`)** | Clean | Rank0/Rank1 max diff `0.0`, `kv_splits=4`, keyframe `fl2va` `25.53 dB / 0.931 SSIM`; `fallbacks=[]` |
| 11 | **MiniMax-H3 Super Acceleration** | `5s` & `10s` `768p` | 4-step H3 draft + 3-step LTX-2.5 refine (`22.2×` 5s / `27.7×` 10s) | **1× H200**: **5s 768p** `361.41s → 8.92s` E2E (**`40.54×`**); **10s 768p** `1,054.84s → 19.82s` E2E (**`53.22×`**); stereo audio MP4 muxing + `offload_between_stages=True` (`57.14 GiB`, `cos=1.000000`) | Clean | Warm E2E uses cached text embeddings; 5s baseline ran all 50 full steps (`361.41s`), 10s baseline uses measured 2-step × 25 + full measured 10s VAE (`95.44s`) |

---

## 2. Real-Time Autoregressive & Interactive World Models (Zero-Contention H200)

| Model Family | Workload | Official / Peer Reference | WorldFoundry H200 Result | Speedup | Parity & Graph Receipts |
| --- | --- | --- | --- | ---: | --- |
| **Matrix-Game-2.0** | `universal` (69f) / `templerun` (117f) | Official: ~25 FPS (H100) | **Lossless BF16 default**: **`57.72 FPS`** (`67.03 dB`) / **`60.52 FPS`** (`65.88 dB`, DiT latent `PSNR=inf`); **w/ FA3**: **`67.55 FPS`** steady | **2.3×–2.7× vs official** | `capture=1, replay=40–50, eager=0, recompiles=0, fallbacks=[]` |
| **LongLive-1.3B** | 81f / 153f / 105f 3-prompt switch | Official paper: 24.8 FPS (FP8) | **Lossless BF16 default (`O(1)` 4 segs)**: 81f **`27.17 FPS`** (`PSNR=inf` eager / `55.84 dB` compiled), 153f **`26.72 FPS`**, 3-prompt **`25.61 FPS`**; **w/ FA3**: 81f **`30.09–30.16 FPS`** (`2.69s`) | **Beats official FP8 in lossless BF16** (`+21%` w/ FA3) | **`O(1)` 4 graph segments forever**, `0` prompt-switch invalidations, `fallbacks=[]` |
| **Self-Forcing 1.3B** | 81f / 153f streaming | Official: ~17 FPS (H100); FastVideo/SGLang: ~20–24 FPS | **Lossless BF16 + FA3 + compile + overlap**: 81f **`27.90 FPS` E2E** (`TTFT 313.0 ms`), 153f **`28.11 FPS` E2E / `28.16 FPS` steady** (`TTFT 324.5 ms`) | **3.27×** vs eager | **`O(1)` 5 graph segments forever**, `320` replays, overlap bitwise-equal to serial |
| **HY-WorldPlay (34 GB)** | Bi 12-lat (45f) & AR 4-step 8ctx | Eager: `2,853 ms` Bi / `701 ms` AR | **Bi 12-lat**: `1,000 ms` 1× BF16 (`2.85×` exact) → **`603.4 ms`** 2× SP=2 Full FP8 (**`4.73×`**); **AR 8ctx**: `277.4 ms` 1× (`14.42 FPS`) → **`191.9 ms`** 2× SP=2 (**`20.84 FPS / 3.65×`**) | **3.65×–4.73×** | RGB VAE verified: BF16+FA3 `54.90 dB / 0.9999`, FP8+FA3 `38.10 dB`, Full FP8 `37.06 dB`, FP8+Sol `36.05 dB` |

---

## 3. Objective Cross-Framework Comparison

| Dimension | WorldFoundry (H200 SM90) | SOL Engine (NVIDIA) | SGLang-Diffusion | LightX2V | FastVideo | FlashDreams |
| --- | --- | --- | --- | --- | --- | --- |
| **Model Breadth in One Stack** | **15 models** (11 flagship video + 4 streaming world models) | 11 flagship video models | Wan2.1/2.2, Hunyuan, LTX, FastWan | Wan2.1/2.2, Hunyuan, Qwen-Image | Wan, Hunyuan, FastH3, Self-Forcing | Streaming world models |
| **Wan2.1-14B (4× GPU)** | **82.00s** exact (**94.4%** eff.); **22.50s** 480p (**3.64×** vs 4× / **13.86×** vs 1×); **79.02s** 720p (**4.01×**) | `~3.48×` algorithmic | ~85–95s 4× H100 exact; ~30–35s cached; Breakable CUDA Graph | Strong FP8/INT8 + LightVAE | ~2.4–2.9× VSA/STA | N/A |
| **Blackwell SM100 NVFP4** | Weight packing & dispatch implemented; **unbenchmarked on physical B200** | **Leads**: Hardware-validated B200 NVFP4 | Emerging SM100 support | FP8/INT8 focus | BF16/FP8 focus | BF16/FP8 |
| **Multi-Tenant Serving** | Single-request, multi-GPU & streaming optimized; **continuous multi-tenant HTTP serving not yet at SGLang parity** | Engine-focused | **Leads**: Production OpenAI-compatible serving & batching | Serving workers | Inference CLI | Streaming runtime |
| **Streaming World Models (Lossless BF16)** | **Leads**: `Matrix-Game-2.0` **67.55 FPS**, `LongLive` **30.16 FPS (`O(1)` 4-seg)**, `Self-Forcing` **28.16 FPS (`O(1)` 5-seg)** | Not primary focus | Limited coverage | Limited coverage | Strong Self-Forcing (~20–24 FPS) | Specialized streaming runtime |
| **Zero-Collapse Auditability** | **Leads**: Zero-fallback JSON receipts, FP32-accum FP8, FP64 Triton RoPE, explicit blocked presets | Research recipes | Production telemetry | Modular plugins | Training + inference co-design | Task-specific |

### Where Peer Frameworks Currently Lead
1. **Blackwell B200 NVFP4**: SOL Engine provides hardware-validated B200 NVFP4 benchmarks; WorldFoundry matches/exceeds the target speedups on H200 using SM90 FP8 W8A8 + kernel fusion + sparse attention + caching, while SM100 NVFP4 hardware validation awaits B200 silicon.
2. **Multi-Tenant HTTP Serving**: SGLang-Diffusion leads in continuous batching, request scheduling, and OpenAI-compatible serving APIs.
3. **Trained Latent-to-Latent Cascading**: Our `MiniMax-H3 Super Acceleration` (`40.54×` 5s / `53.22×` 10s) hands off Stage-1 draft to Stage-2 LTX-2.5 via decoded-frame upscale + LTX VAE encode (`1.72s` at 5s); a trained direct latent-space adapter could remove that handoff cost.

---

## 4. Quality Engineering & Rejected Configurations

1. **Rejected — LTX-2.3 Distilled Stage-2 Token Pruning**: Pruning 20–25% of Stage-2 tokens on 3-step distilled LTX-2.3 (`1.60s / 2.91×`, `15.4 dB`) caused visible rainbow blotches in sky and foliage. Token pruning is now automatically bypassed on short-step distilled runs (`min_steps_to_enable=6`); unpruned `PISA + FBCache` (`2.35×`, `audio_rms_ratio=0.839`) and `PISA + TeaCache` (`2.61×`) are artifact-free.
2. **Rejected — HunyuanVideo-13B `tau=1.25`**: Reached `5.84×` but dropped PSNR to `20.80 dB / SSIM 0.888` with late-frame subject drift; replaced with `tau=1.0 + dense_single=(0,-2,-1)` (`5.25×`, `22.51 dB / SSIM 0.9097`).
3. **Fixed — Hopper FP8 `use_fast_accum=True` (FP22) Residual Drift**: Enabled full FP32 WGMMA accumulation (`use_fast_accum=False`) across all 40–50L deep DiTs (`Wan2.1`, `Wan2.2`, `MiniMax-H3`, `Cosmos3`, `HunyuanVideo-13B`), eliminating `+13.6%` norm drift at `<3%` latency cost.
4. **Fixed — FP32 RoPE & Untiled Sparse Attention**: Replaced FP32 RoPE with fused FP64 Triton RoPE (`13.63×` vs `complex128`, `+1.34 dB`), and enforced 3D tiling `tile_3d=(1,8,8)` + dense edge blocks `(0,1,-1)` + dense warmup steps (`+11.23 dB` on LTX-2.3).
5. **Enforced — Lossless BF16 Default on 1.3B Autoregressive World Models**: Preserves `51–67 dB` / `PSNR=inf` without autoregressive FP8 drift while running at **28–67.5 FPS**.

---

## 5. Reproducing Benchmarks

```bash
# Example: Wan2.1-T2V-14B 4x H200 480p SOL profile with zero-fallback receipts
torchrun --nproc_per_node=4 -m benchmarks.videogen.runner \
  --model wan2.1-t2v-14b --profile sol --attention-backend flash_attention_3 \
  --height 480 --width 832 --frames 81 --steps 50 --repeats 2 \
  --out wan14b_480p_sol.json

# Example: Wan2.1-T2V-14B 1x H200 EasyCache + Sol-Attn + Piecewise CUDA Graph
python -m benchmarks.videogen.runner \
  --model wan2.1-t2v-14b --profile sol --attention-backend flash_attention_3 \
  --easycache 0.15 --easycache-stride 64 \
  --height 480 --width 832 --frames 81 --steps 50 --repeats 2 \
  --out wan14b_1gpu_easycache.json
```
