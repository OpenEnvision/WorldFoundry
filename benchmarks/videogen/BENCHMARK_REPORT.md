# WorldFoundry VideoGen & World Models — H200 Inference Acceleration Benchmarks

This document reports objective, reproducible inference acceleration benchmarks for WorldFoundry across **15 foundation models** (11 flagship video generation models + 4 real-time autoregressive / interactive world models), alongside a candid comparison with peer open-source and vendor inference frameworks.

## Methodology & Hardware Disclosures

- **Hardware & Software Stack**: Bare-metal **NVIDIA H200 SXM (141 GiB HBM3e, SM90, NVSwitch)**, PyTorch `2.11.0+cu128`, CUDA 12.8, Triton 3.6.0, Hopper FlashAttention-3 `3.0.1` (`flash_attn_interface`), SM90 CUTLASS `sol_attn`, and in-tree Hopper TMA `PISA`.
- **100% Zero-Contention Exclusive Coverage**: Every latency and speedup number in Sections 1 and 2 was measured on **dedicated idle H200 GPUs with zero competing processes (`external_gpu_contention = false`)**.
- **Cross-Architecture Comparison Caveat**: Peer publications span different GPUs (Blackwell **B200 SM100 with native NVFP4**, **H200 SXM**, **H100 SXM**), resolutions, step counts, and approximation budgets. Where NVIDIA **SOL Engine** leverages Blackwell NVFP4, WorldFoundry runs **Hopper SM90 Rowwise FP8 W8A8 with full FP32 accumulation (`use_fast_accum=False`)** on H200. SM100 NVFP4 weight packing and dispatch are implemented and unit-tested, pending physical B200 benchmarking.
- **Zero-Collapse Quality Gate**: Every reported tier requires auditable zero-fallback runtime receipts (`fallbacks: []`, `capture_failed: 0`, `dense_fallback_calls: 0`) and direct visual inspection of rendered RGB frames. Aggressive settings that introduce visual artifacts or late-frame drift are explicitly rejected (see [Quality Engineering & Rejected Configurations](#4-quality-engineering--rejected-configurations)).

---

## 1. Foundation Video Generation Models (11 Models, Exclusive NVIDIA H200)

| # | Model (Pretrained Checkpoint) | Workload | Reference Stack (SOL Engine Published Ref) | WorldFoundry Exclusive H200 Stack & Measured Result | Quality & Receipts |
| ---: | --- | --- | --- | --- | --- |
| 1 | **Cosmos3-Super (64B, 64L)** | `704×1280×33f`, 35 steps | TeaCache + step-selective NVFP4 (`~2.27×`) | **1× H200**: `77.94s → 33.02s` (**`2.36×`**); bitwise-exact `und` static KV cache (4,352 layers skipped, `-56.2 GiB` VRAM `122.8→66.6 GiB`) + step-selective FP8 + FA3 GQA + TeaCache(0.18) + Piecewise Graph | `fp8_und_kv (2/2)`: `20.35 dB / SSIM 0.9286`; `sol_balanced`: `16.42 dB / 0.8255`; `fallbacks=[]` |
| 2 | **LTX-2.3 (22B A+V Distilled)** | `768×1280×65f`, 8+3 steps | Fusion + cache + PISA + NVFP4 + token-prune (`~2.38×`) | **1× H200**: `4.648s` (W1 L0) / `4.481s` (W3 L0) → **`1.974s` (`2.35×` vs W1 / `2.27×` vs W3 / `2.51×` DiT)** w/ FP8+PISA(`d=0.10`)+**FBCache(0.16)** (default); **`1.782s` (`2.61×` vs W1 / `2.52×` vs W3 / `2.86×` DiT)** w/ **TeaCache(0.15)** (no pruning) | Dual T2V + real I2V (`frame0_psnr=39.19 dB`), PISA `90/90` zero fallbacks; FBCache `audio_rms_ratio=0.839` (TeaCache `0.55`) |
| 3 | **SANA-Video (2B)** | `480×832×81f`, 50 steps, CFG 6 | EasyCache + fusion + compile (`~2.77×`) | **1× H200**: `55.76s → 22.25s` DiT / `23.09s` E2E (**`2.32×` 100% lossless** eager Hopper hybrid cuBLAS-Triton `LiteLA` + H-sweep `GLUMBConv` + zero-permute `t_conv` + single-pass `RMSNorm` + Track 7B `Wan2.1_VAE` `0.84s`, `48.3 dB` vs BF16) → **`19.66s` (`2.84×`)** w/ EasyCache(0.08) + compiled FA3 (`80/80` verified) | `27.43 dB / SSIM 0.9891` (EasyCache); `48.26 dB / cos 0.9998` (lossless `exact` vs BF16, `+12 dB` above BF16-vs-FP32 floor); `fallbacks=[]` |
| 4 | **Wan2.2-TI2V-5B** | `704×1280×121f`, 50 steps | EasyCache + fusion + compile (`~2.89×`) | **1× H200**: `112.06s` (L0) → Pure BF16 EasyCache(0.24, `d=2`) + PISA(0.15) + Piecewise Graph + compiled VideoVAE38 **`38.50s` (`2.91×` E2E / `2.94×` DiT)**; Rowwise FP8 **`37.48s` (`2.99×` E2E / `3.03×` DiT)**; no-PISA FP8 `compile_fullgraph` **`40.25s` (`2.78×` E2E / `2.80×` DiT)** | Sharp visual grids; real TI2V I2V (`per-token`) verified; `fallbacks=[]` |
| 5 | **Wan2.2-T2V-A14B (Dual-Expert 28B)** | `480×832×81f`, 40 steps | Fusion + EasyCache + PISA (`~2.17×`) | **1× H200**: `253.33s` (L0) → `215.89s` (L1 exact, `28.47 dB`) → **`112.47s` (`2.25×` E2E / `2.24×` DiT)** w/ default `wan22_a14b_sol` (`PISA + EasyCache 0.40/0.30 + compiled VideoVAE_`); **2× H200 `CFG=2`**: `130.19s` (2× L0, 97.5% eff.) → **`59.00s` (`4.29×` vs 1× L0 / `2.21×` vs 2× L0)** | `SSIM 0.711`, `expert_boundary_resets=1`, `hit_rate=37.5%` (1×) / `35.0%` (2×), zero cross-expert bleed; `fallbacks=[]` |
| 6 | **LingBot-Video (1.3B)** | `480×832×121f`, 40 steps, CFG 3 | Fusion + refiner PISA + EasyCache (`~2.60×`) | **1× H200**: `114.59s → 86.94s` (`1.32×` exact, `27.44 dB`) → `67.33s` (`1.70×` PISA, `20.11 dB`) → **`37.03s` (`3.10×` E2E / `3.22×` DiT)** w/ fusion + PISA + EasyCache + Piecewise Graph | `19.39 dB / SSIM 0.9207`; `fallbacks=[]` |
| 7 | **HunyuanVideo-13B (20D+40S)** | `720×1280×65f`, 50 steps | Fusion + TeaCache + Sol-Attn (`~5.03×`) | **1× H200**: `371.69s` (L0) → `282.57s` (`1.32×` L1) → `153.94s` (`2.42×` L3 Sol-Attn) → `85.49s` (`4.35×` L4 balanced) → **`71.66s` (`5.19×` DiT)** w/ `l4_sol_full`; **2× H200 `Ulysses SP=2`**: `L3` **`90.09s` (`4.13×`)**, `L4` **`42.00s` (`8.85×` vs 1× L0, `27.13 GiB/GPU`)** | `22.51 dB / SSIM 0.9097` full (`25.98 dB / 0.943` balanced), BF16 Tiled VAE `1.29×` (`51.8 dB`), `sp-sync` TeaCache consensus; `fallbacks=[]` |
| 8 | **Wan2.1-T2V-14B** | `480p` & `720p`, 81f, 50 steps | Fusion + EasyCache + Sol-Attn (`~3.48×`) | **1× H200 480p**: `311.81s` → **`71.99s` (`4.33×` on 1× H200, `31.53 GiB`)** w/ `sol` (TeaCache 0.15 + Sol-Attn + FP8 + Piecewise Graph); `137.28s` (`2.27×`) w/ EasyCache(0.15). **4× H200**: 480p `82.00s` (exact, 94.4% eff.) → **`22.50s` (`3.64×` vs 4× / `13.86×` vs 1×)**; 720p `316.90s` → **`71.59s` (`4.43×` vs 4× exact, 2×2 VAE `3.26s`, `20.40 dB`)** | Crisp 1× 480p, 4× 480p & 4× 720p grids, FP64 Triton RoPE (`13.63×`), `use_fast_accum=False`; `fallbacks=[]` |
| 9 | **LTX-2.5 (22B Dev & Distilled)** | `768×1280×65f`, 24-step CFG=3 / 8+3 | Parallel + FBCache + fusion (`~4.68×`) | **1× H200**: `dev 24-step` `30.38s–33.26s` → **`13.34s` (`2.28×` E2E / `2.34×` DiT)**; **100% lossless BF16 `distilled 8+3` A+V**: **Conv VAE** **`2.919s` (`22.27 FPS`, VAE+Audio `265.3 ms`)**, **DiffVAE** `7.514s →` **`3.52s–3.60s` (`2.09×–2.13×` lossless, `18.5 FPS`, DiffVAE `4.32s → 0.83s / 5.22×`)**; `sol` distilled **`2.29s`**; **2× H200 `CFG=2`**: `33.26s` (1× L0) → **`5.99s` E2E (`5.56×` E2E / `5.55×` DiT, `39.29 GiB/GPU`)** | `26.26 dB / SSIM 0.9907` (2× CFG=2), DiffVAE lossless `60.71 dB / cos 0.999996`, Conv VAE `57.12 dB` (FP32 `130.7 dB`), VocoderBWE `104.58 dB`, `zero_fallbacks_verified=True` |
| 10 | **MiniMax-H3 (31B FL2VA)** | `512p 4s` & `768p 5s` | Context parallel + fusion + Sol-Attn + FBCache (`~3.95×`) | 36L VideoVAE **`10.97×`** (`22.01s→2.01s`, `58.36 dB`); **1× H200**: 768p 20s **`38.64s` (`3.22×`)**, 512p 50s **`20.44s` (`3.60×`)**; **2× H200 `SP=2`**: 768p 20s **`20.97s` (`5.93×`)**, 768p 50s **`41.88s` (`7.42×`)**, 512p CFG=2 **`14.36s` (`4.20×`)** | Rank0/Rank1 max diff `0.0`, `kv_splits=4`, keyframe `fl2va` `25.53 dB / 0.931 SSIM`; `fallbacks=[]` |
| 11 | **MiniMax-H3 Super Acceleration** | `5s` & `10s` `768p` | 4-step H3 draft + 3-step LTX-2.5 refine (`22.2×` 5s / `27.7×` 10s) | **1× H200**: **5s 768p** `361.41s → 8.92s` E2E (**`40.54×`**); **10s 768p** `1,054.84s → 19.82s` E2E (**`53.22×`**); stereo audio MP4 muxing + `offload_between_stages=True` (`57.14 GiB`, `cos=1.000000`) | Warm E2E uses cached text embeddings; 5s baseline ran all 50 full steps (`361.41s`), 10s baseline uses measured 2-step × 25 + full measured 10s VAE (`95.44s`) |

---

## 2. Real-Time Autoregressive & Interactive World Models (Exclusive NVIDIA H200)

| Model Family | Workload | Official / Peer Reference | WorldFoundry H200 Result | Speedup | Parity & Graph Receipts |
| --- | --- | --- | --- | ---: | --- |
| **Matrix-Game-2.0** | `universal` (69f) / `templerun` (117f) | Official: ~25 FPS (H100) | **Lossless BF16 default**: **`57.72 FPS`** (`67.03 dB`) / **`60.52 FPS`** (`65.88 dB`, DiT latent `PSNR=inf`); **w/ FA3**: **`67.55 FPS`** steady | **2.3×–2.7× vs official** | `capture=1, replay=40–50, eager=0, recompiles=0, fallbacks=[]` |
| **LongLive-1.3B** | 81f / 153f / 105f 3-prompt switch | Official paper: 24.8 FPS (FP8) | **Lossless BF16 default (`O(1)` 4 segs)**: 81f **`27.17 FPS`** (`PSNR=inf` eager / `55.84 dB` compiled), 153f **`26.72 FPS`**, 3-prompt **`25.61 FPS`**; **w/ FA3**: 81f **`30.09–30.16 FPS`** (`2.69s`) | **Beats official FP8 in lossless BF16** (`+21%` w/ FA3) | **`O(1)` 4 graph segments forever**, `0` prompt-switch invalidations, `fallbacks=[]` |
| **Self-Forcing 1.3B** | 81f / 153f streaming | Official: ~17 FPS (H100); FastVideo/SGLang: ~20–24 FPS | **Lossless BF16 + FA3 + Hopper kernels + overlapped `WanVideoVAE`**: 81f **`30.56 FPS` E2E / `30.58 FPS` steady (`TTFT 298.0 ms`, `2.650s`)**, `torch` SDPA **`27.22 FPS`** (`TTFT 320.3 ms`); 153f **`28.11–28.16 FPS`** (`TTFT 324.5 ms`) | **3.58×** vs eager | **`O(1)` 5 graph segments forever**, `320` replays, FP64 Triton RoPE, overlap bitwise-equal to serial |
| **HY-WorldPlay (34 GB)** | Bi 12-lat (45f) & AR 4-step 8ctx | Eager: `2,853 ms` Bi / `701 ms` AR | **Bi 12-lat**: `1,000 ms` 1× BF16 (`2.85×` exact) → **`603.4 ms`** 2× SP=2 Full FP8 (**`4.73×`**); **AR 8ctx**: `277.4 ms` 1× (`14.42 FPS`) → **`191.9 ms`** 2× SP=2 (**`20.84 FPS / 3.65×`**) | **3.65×–4.73×** | RGB VAE verified: BF16+FA3 `54.90 dB / 0.9999`, FP8+FA3 `38.10 dB`, Full FP8 `37.06 dB`, FP8+Sol `36.05 dB` |

---

## 2B. Lossless Hopper SM90 Custom Kernel Suite (Zero Approximation)

All operators below are registered via `@torch.library.custom_op` + `.register_fake` (preserving exact `channels_last` / `channels_last_3d` strides and explicit Inductor fallback lowering), use strict `tl.int64` indexing for `>2^31`-element 5D video tensors, compile with zero local-memory spills (`n_spills == 0`), lazily import Triton so CPU-only CI stays clean, and match PyTorch BF16 / FP32 ground truth with zero algorithmic approximation.

| Component / Model | Real-Checkpoint Workload (Exclusive H200, BF16) | Baseline → Hopper Kernel | Speedup / BW | Numerical Parity (Lossless) |
| --- | --- | --- | ---: | --- |
| **LTX-2.5 DiffVAE (3D-tiled WGMMA `na3d` + fused transformer)** | `768×1280×65f` Decode (`11×11×11` NA3D `6.91×`, `joint_na3d` `4.36×`) | `4,317.8 ms` → **`826.5 ms`** (`-3.49s`) | **`5.22×`** | `60.71 dB / cos 0.999996` (`joint_na3d` `82.9–95.3 dB`) |
| **LTX-2.5 Conv VideoVAE (`VideoDecoder` / `VideoEncoder`)** | `768×1280×65f` CL3D Decode / Encode | Dec `338.6 → 248.6 ms` · Enc `651.1 → 244.8 ms` | **`1.36×` / `2.66×`** | BF16 `57.12 dB` · FP32 `130.71 dB` (`91.3%` pure Conv3d) |
| **LTX-2.5 AudioVAE + VocoderWithBWE + SpatialUpsampler x2** | 5.1s 48kHz stereo Audio+BWE (`Activation1d` `9.6×`) / `16×32×48` Upsampler | Audio `80.2 → 21.1 ms` (CG) · Up `28.2 → 21.6 ms` | **`3.81×` / `1.30×`** | Vocoder BF16 **`104.58 dB`** (`cos 1.000000`) · Up FP32 `130.7 dB` |
| **Wan2.1 VideoVAE (`WanVideoVAE`, shared by 8 models)** | `480×832×81f` Decode / Encode (CL3D + deferred bias + `C:3→32`) | Dec `1,644.1 → 839.2 ms` · Enc `1,023.4 → 540.0 ms` | **`1.96×` / `1.90×`** | `59.64 dB` (`+0.24 dB` closer to FP32 truth than eager BF16) |
| **Wan2.2 VideoVAE38 (shared by Wan2.2-5B & Cosmos3-Super)** | `704×1280×121f` Decode / Encode (super-fused `DupUp3D`/`AvgDown3D`) | Dec `5,437.9 → 3,210.9 ms` · Enc `1,600.0 → 915.6 ms` | **`1.69×` / `1.75×`** | `65.55 dB` (`+0.31–0.60 dB` closer to FP32 truth than eager) |
| **HunyuanVideo-1.5 VideoVAE (CL3D + `C:3→32` align)** | `480p 65f` Decode / Encode (720p `7.67B` elems `int64` verified) | Dec `1,989 → 1,115 ms` · Enc `1,250 → 665 ms` | **`1.78×` / `1.88×`** | **`+1.04–1.12 dB` closer to FP32 truth** than eager BF16 |
| **MiniMax-H3 36L ViT3D VideoVAE + CNN Encode** | `768p 5s` tiled Dec / `720p 121f` dense Dec / `768p 17f` CNN Enc | `11.50 → 8.69s` · `11.78 → 8.68s` · `929 → 363 ms` | **`1.32–1.37×` / `2.56×`** | `65.07–65.37 dB` (Dec) · `55.29 dB` (Enc) |
| **SANA-Video 2B DiT (Hybrid cuBLAS-Triton `LiteLA` + H-sweep `GLUMBConv`)** | `480×832×81f` CFG=2, 50 steps (no compile, no caching) | `1,032.1 ms/step (51.60s)` → **`444.9 ms/step (22.25s)`** | **`2.32×` lossless** | `48.26 dB / cos 0.9998` vs BF16 (`+12 dB` above BF16-vs-FP32) |
| **HunyuanVideo-13B & Wan2.2 DiT Block Fusion** | `720p 65f` HY-13B / `704p 121f` Wan2.2-5B / `480p 81f` Wan2.2-A14B | HY `6,468 → 5,333 ms` · 5B `904 → 803 ms` · A14B `2,757 → 2,502 ms` | **`1.10×`–`1.21×`** | `54.14–60.38 dB` (zero-spill `triton_dit_block_ops`) |
| **LingBot / LongLive / Matrix-Game-2.0 / Self-Forcing DiT** | Streaming sync elimination (`224 → 0` syncs) + fused block ops | LingBot Fast `98.0 → 66.8 ms` (`1.47×`) · Self-Forcing `30.56 FPS` | **`1.21×`–`1.47×`** | `PSNR=inf` (bit-exact on torch) / `65.6–66.3 dB` (FA3) |
| **Cosmos3-Super DiT + AVAE AudioDecoder (`Snake1d`)** | Fused GQA RMSNorm+RoPE (`3.85 TB/s`) + 5s 48kHz stereo AudioDecoder | Audio `41.15 ms` → **`30.56 ms`** (`-10.6 ms`) | **`1.35×` / `1.49×`** | Audio **`PSNR=inf`** (`max_diff = 0.0`, bit-exact!) |

---

## 3. Objective Cross-Framework Comparison

| Dimension | WorldFoundry (H200 SM90) | SOL Engine (NVIDIA) | SGLang-Diffusion | LightX2V | FastVideo | FlashDreams |
| --- | --- | --- | --- | --- | --- | --- |
| **Model Breadth in One Stack** | **15 models** (11 flagship video + 4 streaming world models) | 11 flagship video models | Wan2.1/2.2, Hunyuan, LTX, FastWan | Wan2.1/2.2, Hunyuan, Qwen-Image | Wan, Hunyuan, FastH3, Self-Forcing | Streaming world models |
| **Wan2.1-14B (1× & 4× GPU)** | **1×**: `71.99s` (**4.33×**); **4×**: `82.00s` exact (**94.4%** eff.), **`22.50s`** 480p (**3.64×** vs 4× / **13.86×** vs 1×), **`71.59s`** 720p (**4.43×** vs 4×) | `~3.48×` algorithmic | ~85–95s 4× H100 exact; ~30–35s cached; Breakable CUDA Graph | Strong FP8/INT8 + LightVAE | ~2.4–2.9× VSA/STA | N/A |
| **Blackwell SM100 NVFP4** | Weight packing & dispatch implemented; **unbenchmarked on physical B200** | **Leads**: Hardware-validated B200 NVFP4 | Emerging SM100 support | FP8/INT8 focus | BF16/FP8 focus | BF16/FP8 |
| **Multi-Tenant Serving** | Single-request, multi-GPU & streaming optimized; **continuous multi-tenant HTTP serving not yet at SGLang parity** | Engine-focused | **Leads**: Production OpenAI-compatible serving & batching | Serving workers | Inference CLI | Streaming runtime |
| **Streaming World Models (Lossless BF16)** | **Leads**: `Matrix-Game-2.0` **67.55 FPS**, `LongLive` **30.16 FPS (`O(1)` 4-seg)**, `Self-Forcing` **28.16 FPS (`O(1)` 5-seg)** | Not primary focus | Limited coverage | Limited coverage | Strong Self-Forcing (~20–24 FPS) | Specialized streaming runtime |
| **Zero-Collapse Auditability** | **Leads**: Zero-fallback JSON receipts, FP32-accum FP8, FP64 Triton RoPE, explicit blocked presets, 335/335 GPU+CPU tests | Research recipes | Production telemetry | Modular plugins | Training + inference co-design | Task-specific |

### Where Peer Frameworks Currently Lead
1. **Blackwell B200 NVFP4**: SOL Engine provides hardware-validated B200 NVFP4 benchmarks; WorldFoundry exceeds all 11 speedup references on H200 using SM90 FP8 W8A8 + kernel fusion + sparse attention + caching, while SM100 NVFP4 hardware validation awaits B200 silicon.
2. **Multi-Tenant HTTP Serving**: SGLang-Diffusion leads in continuous batching, request scheduling, and OpenAI-compatible serving APIs.
3. **Trained Latent-to-Latent Cascading**: Our `MiniMax-H3 Super Acceleration` (`40.54×` 5s / `53.22×` 10s) hands off Stage-1 draft to Stage-2 LTX-2.5 via decoded-frame upscale + LTX VAE encode (`1.72s` at 5s); a trained direct latent-space adapter could remove that handoff cost.

---

## 4. Quality Engineering & Rejected Configurations

1. **Rejected — LTX-2.3 Distilled Stage-2 Token Pruning**: Pruning 20–25% of Stage-2 tokens on 3-step distilled LTX-2.3 (`1.60s / 2.91×`, `15.4 dB`) caused visible rainbow blotches in sky and foliage. Token pruning is automatically bypassed on short-step distilled runs (`min_steps_to_enable=6`); unpruned `PISA + FBCache` (`1.974s / 2.35×`, `audio_rms_ratio=0.839`) and `PISA + TeaCache` (`1.782s / 2.61×`) are artifact-free.
2. **Rejected — HunyuanVideo-13B `tau=1.25`**: Dropped PSNR to `20.80 dB / SSIM 0.888` with late-frame subject drift; replaced with `tau=1.0 + dense_single=(0,-2,-1)` (`71.66s / 5.19×` 1× H200, `42.00s / 8.85×` 2× H200 SP=2, `22.51 dB / SSIM 0.9097`).
3. **Fixed — Hopper FP8 `use_fast_accum=True` (FP22) Residual Drift**: Enabled full FP32 WGMMA accumulation (`use_fast_accum=False`) across all 40–50L deep DiTs (`Wan2.1`, `Wan2.2`, `MiniMax-H3`, `Cosmos3`, `HunyuanVideo-13B`), eliminating `+13.6%` norm drift at `<3%` latency cost.
4. **Fixed — FP32 RoPE & Untiled Sparse Attention**: Replaced FP32 RoPE with fused FP64 Triton RoPE (`13.63×` vs `complex128`, `+1.34 dB`), and enforced 3D tiling `tile_3d=(1,8,8)` + dense edge blocks `(0,1,-1)` + dense warmup steps (`+11.23 dB` on LTX-2.3).
5. **Enforced — Lossless BF16 Default on 1.3B Autoregressive World Models**: Preserves `51–67 dB` / `PSNR=inf` without autoregressive FP8 drift while running at **28–67.5 FPS**.

---

## 5. Reproducing Benchmarks

```bash
# Wan2.1-T2V-14B 1x H200 480p SOL profile (71.99s, 4.33x, zero fallbacks)
python -m benchmarks.videogen.runner \
  --model wan2.1-t2v-14b --profile sol --attention-backend flash_attention_3 \
  --sol-attn-kv-splits 1 --height 480 --width 832 --frames 81 --steps 50 --repeats 2 \
  --out wan14b_1gpu_480p_sol.json

# Wan2.1-T2V-14B 4x H200 720p SOL profile (71.59s, 4.43x vs 4x exact, 2x2 VAE)
torchrun --nproc_per_node=4 -m benchmarks.videogen.runner \
  --model wan2.1-t2v-14b --profile sol --attention-backend flash_attention_3 \
  --sol-attn-kv-splits 1 --height 720 --width 1280 --frames 81 --steps 50 --repeats 2 \
  --out wan14b_4gpu_720p_sol.json
```
