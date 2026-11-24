# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 OpenEnvision
"""Weight-free component parity and strict checkpoint-loading tests for FastFrames."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch
from safetensors.torch import save_file

from worldfoundry.base_models.diffusion_model.models.denoisers.fastframes import (
    build_fastframes_decoder,
    build_fastframes_dit,
    build_fastframes_refiner,
    load_fastframes_components,
    load_fastframes_state_dict,
)
from worldfoundry.base_models.diffusion_model.models.networks.wan.variants.fastframes import (
    FASTFRAMES_WAN_CONFIGS,
    BridgedPixelDecoder,
    FastFramesCausalWanModel,
    FastFramesFastOps,
    FastFramesWanConfig,
    KVRefinementNetwork,
    PixelDecoder,
    align_frames,
    allocate_fastframes_kv_cache,
    fastframes_attention,
    unpatchify_tokens,
)
from worldfoundry.base_models.diffusion_model.models.networks.wan.variants.forcing.model import (
    WanLayerNorm,
    WanRMSNorm,
)
from worldfoundry.base_models.diffusion_model.schedulers.fastframes import (
    FF_L,
    FF_S,
    FastFramesFlowMatchScheduler,
    resolve_fastframes_variant,
)

pytestmark = pytest.mark.unit

UPSTREAM_ROOT = Path(__file__).resolve().parents[2] / "tmp" / "upstream_refs" / "ml-fastframes"


def _import_upstream_fastframes() -> dict[str, ModuleType] | None:
    if not (UPSTREAM_ROOT / "fastframes").is_dir():
        return None
    root_str = str(UPSTREAM_ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return {
        "kv": importlib.import_module("fastframes.models.kv_refinement"),
        "dec": importlib.import_module("fastframes.models.pixel_decoder"),
        "pipe": importlib.import_module("fastframes.pipeline"),
        "attn": importlib.import_module("fastframes.wan.attention"),
        "causal": importlib.import_module("fastframes.wan.causal_model"),
        "model": importlib.import_module("fastframes.wan.model"),
        "sched": importlib.import_module("fastframes.wan.scheduler"),
    }


def _tiny_wan_config() -> FastFramesWanConfig:
    return FastFramesWanConfig(
        dim=48,
        ffn_dim=96,
        num_heads=4,
        num_layers=2,
        patch_size=(1, 2, 2),
        text_len=8,
        in_dim=4,
        freq_dim=16,
        text_dim=32,
        out_dim=4,
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
    )


def _randomize_non_zero_params(module: torch.nn.Module, seed: int) -> None:
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for param in module.parameters():
            param.copy_(torch.randn(param.shape, generator=gen, dtype=torch.float32) * 0.05)


def test_canonical_wan_primitives_reused_and_official_configs() -> None:
    assert FASTFRAMES_WAN_CONFIGS["Wan2.1-T2V-1.3B"].dim == 1536
    assert FASTFRAMES_WAN_CONFIGS["Wan2.1-T2V-1.3B"].num_layers == 30
    assert FASTFRAMES_WAN_CONFIGS["Wan2.1-T2V-14B"].dim == 5120
    assert FASTFRAMES_WAN_CONFIGS["Wan2.1-T2V-14B"].num_layers == 40

    cfg = _tiny_wan_config()
    dit = FastFramesCausalWanModel(cfg)
    block0 = dit.blocks[0]
    self_attn = getattr(block0, "self_attn")
    assert isinstance(getattr(block0, "norm1"), WanLayerNorm)
    assert isinstance(getattr(self_attn, "norm_q"), WanRMSNorm)
    assert isinstance(getattr(self_attn, "norm_k"), WanRMSNorm)
    assert isinstance(dit.head.norm, WanLayerNorm)

    dec = PixelDecoder(dim=32, n_low_blocks=1, n_high_blocks=1, ffn_mult=2)
    assert isinstance(dec.norm, WanLayerNorm)

    assert resolve_fastframes_variant("ff-s") is FF_S
    assert resolve_fastframes_variant("fastframes-s") is FF_S
    assert resolve_fastframes_variant("ff-l") is FF_L
    assert resolve_fastframes_variant("fastframes-l") is FF_L
    with pytest.raises(ValueError, match="Unsupported FastFrames variant"):
        resolve_fastframes_variant("fastframes-xl")


def test_scheduler_exact_parity_with_upstream() -> None:
    wf_sched = FastFramesFlowMatchScheduler(shift=5.0)
    assert wf_sched.sigmas.shape == (1000,)
    assert wf_sched.timesteps.shape == (1000,)
    assert float(wf_sched.sigmas[0]) == pytest.approx(1.0)
    assert float(wf_sched.timesteps[0]) == pytest.approx(1000.0)

    gen = torch.Generator().manual_seed(101)
    x0 = torch.randn(2, 4, 8, 8, generator=gen, dtype=torch.bfloat16)
    noise = torch.randn(2, 4, 8, 8, generator=gen, dtype=torch.bfloat16)
    flow = torch.randn(2, 4, 8, 8, generator=gen, dtype=torch.bfloat16)

    up = _import_upstream_fastframes()
    up_sched = up["sched"].FlowMatchScheduler(shift=5.0) if up is not None else None
    if up_sched is not None:
        assert torch.equal(wf_sched.sigmas, up_sched.sigmas)
        assert torch.equal(wf_sched.timesteps, up_sched.timesteps)

    for ts in (1000.0, 937.5, 833.3333129882812, 625.0, 937.0, 833.0):
        t = torch.tensor([ts, ts], dtype=torch.float32)
        wf_xt = wf_sched.add_noise(x0, noise, t)
        wf_pred_x0 = wf_sched.flow_to_x0(flow, wf_xt, t)
        wf_back_flow = wf_sched.x0_to_flow(wf_pred_x0, wf_xt, t)
        wf_step = wf_sched.step(flow, t, wf_xt)
        torch.testing.assert_close(wf_back_flow.float(), flow.float(), atol=5e-2, rtol=5e-2)
        assert wf_step.shape == x0.shape and torch.isfinite(wf_step).all()

        if up_sched is not None:
            up_xt = up_sched.add_noise(x0, noise, t)
            assert torch.equal(wf_xt, up_xt)
            up_pred_x0 = up_sched.flow_to_x0(flow, up_xt, t)
            assert torch.equal(wf_pred_x0, up_pred_x0)
            up_back_flow = up_sched.x0_to_flow(up_pred_x0, up_xt, t)
            assert torch.equal(wf_back_flow, up_back_flow)
            up_step = up_sched.step(flow, t, up_xt)
            assert torch.equal(wf_step, up_step)


def test_kv_refinement_identity_renorm_and_upstream_parity() -> None:
    # Zero-initialized network preserves per-head K norms and acts as identity on K, V.
    fresh = KVRefinementNetwork(
        dim=48, num_heads=4, head_dim=12, num_blocks=2, d_a=32, n_stem=2, stem_n_heads=4, freq_dim=16
    ).eval()
    gen = torch.Generator().manual_seed(17)
    stem_tokens = torch.randn(1, 6, 48, generator=gen)
    nk = torch.randn(2, 6, 48, generator=gen)
    nv = torch.randn(2, 6, 48, generator=gen)
    with torch.no_grad():
        emb = fresh.embed_timestep(833.3333129882812, 1, torch.device("cpu"), torch.float32)
        phi = fresh.compute_phi(stem_tokens, emb)
        k_id, v_id = fresh.repair_all(nk, nv, phi, emb)
    torch.testing.assert_close(k_id, nk, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(v_id, nv, atol=1e-6, rtol=1e-6)

    # Non-zero weights: strict per-head K norm preservation and exact parity with upstream when available.
    wf_net = KVRefinementNetwork(
        dim=48, num_heads=4, head_dim=12, num_blocks=2, d_a=32, n_stem=2, stem_n_heads=4, freq_dim=16
    ).eval()
    _randomize_non_zero_params(wf_net, seed=23)

    with torch.no_grad():
        wf_e = wf_net.embed_timestep(833.3333129882812, 1, torch.device("cpu"), torch.float32)
        wf_phi = wf_net.compute_phi(stem_tokens, wf_e)
        wf_k, wf_v = wf_net.repair_all(nk, nv, wf_phi, wf_e)

        up = _import_upstream_fastframes()
        if up is not None:
            up_net = (
                up["kv"]
                .KVRefinementNetwork(
                    dim=48, num_heads=4, head_dim=12, num_blocks=2, d_a=32, n_stem=2, stem_n_heads=4, freq_dim=16
                )
                .eval()
            )
            up_net.load_state_dict(wf_net.state_dict(), strict=True)
            up_e = up_net.embed_timestep(833.3333129882812, 1, torch.device("cpu"), torch.float32)
            torch.testing.assert_close(wf_e, up_e, atol=0.0, rtol=0.0)
            up_phi = up_net.compute_phi(stem_tokens, up_e)
            torch.testing.assert_close(wf_phi, up_phi, atol=0.0, rtol=0.0)
            up_k, up_v = up_net.repair_all(nk, nv, up_phi, up_e)
            torch.testing.assert_close(wf_k, up_k, atol=0.0, rtol=0.0)
            torch.testing.assert_close(wf_v, up_v, atol=0.0, rtol=0.0)

    # Verify post-RoPE per-head L2 norm preservation on K.
    ref_norm = nk.reshape(2, 6, 4, 12).norm(dim=-1)
    out_norm = wf_k.reshape(2, 6, 4, 12).norm(dim=-1)
    torch.testing.assert_close(out_norm, ref_norm, atol=1e-4, rtol=1e-4)
    assert not torch.allclose(wf_v, nv)


def test_pixel_decoders_exact_parity_with_upstream() -> None:
    up = _import_upstream_fastframes()
    gen = torch.Generator().manual_seed(31)
    grid = torch.tensor([[2, 2, 2]], dtype=torch.long)

    # Unbridged PixelDecoder (FastFrames-S architecture)
    wf_dec = PixelDecoder(dim=32, pixel_patch_size=(4, 16, 16), n_low_blocks=2, n_high_blocks=1, ffn_mult=2).eval()
    _randomize_non_zero_params(wf_dec, seed=37)

    x = torch.randn(1, 8, 32, generator=gen)
    e_mod = torch.randn(1, 2, 1, 32, generator=gen)
    with torch.no_grad():
        wf_px = wf_dec(x, e_mod, grid)
    assert wf_px.shape == (1, 8, 3, 32, 32)
    assert torch.isfinite(wf_px).all()
    assert align_frames(wf_px, 5).shape == (1, 5, 3, 32, 32)

    if up is not None:
        up_dec = (
            up["dec"]
            .PixelDecoder(dim=32, pixel_patch_size=(4, 16, 16), n_low_blocks=2, n_high_blocks=1, ffn_mult=2)
            .eval()
        )
        up_dec.load_state_dict(wf_dec.state_dict(), strict=True)
        with torch.no_grad():
            up_px = up_dec(x, e_mod, grid)
        assert torch.equal(wf_px, up_px)
        assert torch.equal(align_frames(wf_px, 5), up["dec"].align_frames(up_px, 5))

    # BridgedPixelDecoder (FastFrames-L architecture) including deterministic bridge seed init
    wf_bridged = BridgedPixelDecoder(in_dim=64, dim=32, n_low_blocks=2, n_high_blocks=1, ffn_mult=2).eval()
    wf_bridged_repeat = BridgedPixelDecoder(in_dim=64, dim=32, n_low_blocks=2, n_high_blocks=1, ffn_mult=2).eval()
    assert torch.equal(wf_bridged.bridge.weight, wf_bridged_repeat.bridge.weight)
    if up is not None:
        up_bridged = up["dec"].BridgedPixelDecoder(in_dim=64, dim=32).eval()
        assert torch.equal(wf_bridged.bridge.weight, up_bridged.bridge.weight)

    # Full bridged forward parity with matching block widths (default ffn_mult=4).
    wf_bridged_full = BridgedPixelDecoder(in_dim=48, dim=24).eval()
    _randomize_non_zero_params(wf_bridged_full, seed=41)
    xb = torch.randn(1, 8, 48, generator=gen)
    eb = torch.randn(1, 2, 1, 48, generator=gen)
    with torch.no_grad():
        wf_bpx = wf_bridged_full(xb, eb, grid)
    assert wf_bpx.shape == (1, 8, 3, 32, 32) and torch.isfinite(wf_bpx).all()
    if up is not None:
        up_bridged_full = up["dec"].BridgedPixelDecoder(in_dim=48, dim=24).eval()
        up_bridged_full.load_state_dict(wf_bridged_full.state_dict(), strict=True)
        with torch.no_grad():
            up_bpx = up_bridged_full(xb, eb, grid)
        assert torch.equal(wf_bpx, up_bpx)

    # Unpatchify helper check
    tokens = torch.randn(1, 2, 4, 4 * 4 * 4 * 3, generator=gen)
    unpatched = unpatchify_tokens(tokens, grid, (4, 4, 4), 3)
    assert unpatched.shape == (1, 8, 3, 8, 8)
    if up is not None:
        assert torch.equal(unpatched, up["dec"].unpatchify_tokens(tokens, grid, (4, 4, 4), 3))


def test_causal_dit_forward_frame_standard_and_fastops_parity(monkeypatch: pytest.MonkeyPatch) -> None:
    up = _import_upstream_fastframes()
    cfg = _tiny_wan_config()
    wf_dit = FastFramesCausalWanModel(cfg).eval()
    _randomize_non_zero_params(wf_dit, seed=53)
    wf_dit_bf16 = wf_dit.to(dtype=torch.bfloat16)

    up_dit_bf16 = None
    if up is not None:
        up_cfg = up["model"].WanConfig(
            dim=cfg.dim,
            ffn_dim=cfg.ffn_dim,
            num_heads=cfg.num_heads,
            num_layers=cfg.num_layers,
            patch_size=cfg.patch_size,
            text_len=cfg.text_len,
            in_dim=cfg.in_dim,
            freq_dim=cfg.freq_dim,
            text_dim=cfg.text_dim,
            out_dim=cfg.out_dim,
            qk_norm=cfg.qk_norm,
            cross_attn_norm=cfg.cross_attn_norm,
            eps=cfg.eps,
        )
        up_dit = up["causal"].CausalWanModel(up_cfg).eval()
        up_dit.load_state_dict(wf_dit.state_dict(), strict=True)
        orig_up_attn = up["attn"].attention
        monkeypatch.setattr(up["causal"], "attention", lambda q, k, v, **kw: orig_up_attn(q, k, v, backend="auto"))
        monkeypatch.setattr(up["model"], "cross_attention", lambda q, k, v: orig_up_attn(q, k, v, backend="auto"))
        up_dit_bf16 = up_dit.to(dtype=torch.bfloat16)

    gen = torch.Generator().manual_seed(59)
    prompt_embeds = torch.randn(1, 5, cfg.text_dim, generator=gen, dtype=torch.bfloat16)
    latents = torch.randn(1, 3, cfg.in_dim, 4, 6, generator=gen, dtype=torch.bfloat16)
    frame_tokens = (4 // 2) * (6 // 2)

    with torch.no_grad():
        wf_tc = wf_dit_bf16.text_cache(prompt_embeds)
        up_tc = up_dit_bf16.text_cache(prompt_embeds) if up_dit_bf16 is not None else None
        if up_tc is not None:
            for wf_entry, up_entry in zip(wf_tc, up_tc, strict=True):
                assert torch.equal(wf_entry["k"], up_entry["k"])
                assert torch.equal(wf_entry["v"], up_entry["v"])

        # Standard path (fast=None)
        wf_kv = allocate_fastframes_kv_cache(wf_dit_bf16, 3, frame_tokens, dtype=torch.bfloat16, device="cpu")
        up_kv = (
            up["causal"].allocate_kv_cache(up_dit_bf16, 3, frame_tokens, dtype=torch.bfloat16, device="cpu")
            if up is not None and up_dit_bf16 is not None
            else None
        )
        t = torch.tensor([[833.3333]], dtype=torch.float32)
        wf_e0, wf_eh = wf_dit_bf16.time_conditioning(t, torch.bfloat16)
        up_e0, up_eh = up_dit_bf16.time_conditioning(t, torch.bfloat16) if up_dit_bf16 is not None else (None, None)
        if up_e0 is not None and up_eh is not None:
            assert torch.equal(wf_e0, up_e0)
            assert torch.equal(wf_eh, up_eh)

        for frame in range(3):
            wf_flow, wf_feat = wf_dit_bf16.forward_frame(
                latents[:, frame : frame + 1], wf_e0, wf_eh, wf_tc, wf_kv, frame, fast=None
            )
            if (
                up_dit_bf16 is not None
                and up_e0 is not None
                and up_eh is not None
                and up_tc is not None
                and up_kv is not None
            ):
                up_flow, up_feat = up_dit_bf16.forward_frame(
                    latents[:, frame : frame + 1], up_e0, up_eh, up_tc, up_kv, frame, fast=None
                )
                assert torch.equal(wf_flow, up_flow)
                assert torch.equal(wf_feat, up_feat)
        if up_kv is not None:
            for wf_slot, up_slot in zip(wf_kv, up_kv, strict=True):
                assert torch.equal(wf_slot["k"], up_slot["k"])
                assert torch.equal(wf_slot["v"], up_slot["v"])

        # FastOps path (fused QKV + FP32 real RoPE + GEMM patch embed)
        wf_fast = FastFramesFastOps(wf_dit_bf16, 4, 6, 3)
        wf_kv_fast = allocate_fastframes_kv_cache(wf_dit_bf16, 3, frame_tokens, dtype=torch.bfloat16, device="cpu")
        up_fast = up["causal"].FastOps(up_dit_bf16, 4, 6, 3) if up is not None and up_dit_bf16 is not None else None
        up_kv_fast = (
            up["causal"].allocate_kv_cache(up_dit_bf16, 3, frame_tokens, dtype=torch.bfloat16, device="cpu")
            if up is not None and up_dit_bf16 is not None
            else None
        )

        for frame in range(3):
            wf_flow_f, wf_feat_f = wf_dit_bf16.forward_frame(
                latents[:, frame : frame + 1], wf_e0, wf_eh, wf_tc, wf_kv_fast, frame, fast=wf_fast
            )
            if (
                up_dit_bf16 is not None
                and up_e0 is not None
                and up_eh is not None
                and up_tc is not None
                and up_kv_fast is not None
                and up_fast is not None
            ):
                up_flow_f, up_feat_f = up_dit_bf16.forward_frame(
                    latents[:, frame : frame + 1], up_e0, up_eh, up_tc, up_kv_fast, frame, fast=up_fast
                )
                assert torch.equal(wf_flow_f, up_flow_f)
                assert torch.equal(wf_feat_f, up_feat_f)

        # FastOps numerically agrees closely with standard path (small FP32-vs-FP64 RoPE / BF16 rounding diff).
        torch.testing.assert_close(wf_flow_f.float(), wf_flow.float(), atol=5e-2, rtol=5e-2)
        torch.testing.assert_close(wf_feat_f.float(), wf_feat.float(), atol=5e-2, rtol=5e-2)

    # Verify graceful CPU SDPA fallback when backend="cudnn" is explicitly requested on CPU.
    q = torch.randn(1, 6, 4, 12, dtype=torch.float32)
    out = fastframes_attention(q, q, q, backend="cudnn")
    assert out.shape == q.shape and out.dtype == q.dtype and torch.isfinite(out).all()


def test_strict_checkpoint_loading_pt_safetensors_sharded_and_bridged(tmp_path: Path) -> None:
    cfg = _tiny_wan_config()

    dit_src = FastFramesCausalWanModel(cfg).eval()
    _randomize_non_zero_params(dit_src, seed=71)
    ref_src = KVRefinementNetwork(
        dim=cfg.dim,
        num_heads=cfg.num_heads,
        head_dim=cfg.dim // cfg.num_heads,
        num_blocks=cfg.num_layers,
        d_a=32,
        n_stem=2,
        stem_n_heads=4,
        stem_mlp_mult=4,
        head_mlp_mult=2,
        freq_dim=16,
    ).eval()
    _randomize_non_zero_params(ref_src, seed=73)
    dec_s_src = PixelDecoder(dim=cfg.dim, n_low_blocks=2, n_high_blocks=1, ffn_mult=2).eval()
    _randomize_non_zero_params(dec_s_src, seed=79)
    dec_l_src = BridgedPixelDecoder(in_dim=cfg.dim, dim=24, n_low_blocks=2, n_high_blocks=1, ffn_mult=2).eval()
    _randomize_non_zero_params(dec_l_src, seed=83)

    # 1. FF-S style: .pt with nested generator_ema + FSDP prefixes, .safetensors refiner & unbridged pixel_head.* decoder
    dit_pt = tmp_path / "causal-forcing++" / "framewise-2step.pt"
    dit_pt.parent.mkdir(parents=True, exist_ok=True)
    wrapped_dit = {
        (f"model._fsdp_wrapped_module.{k}" if i % 2 == 0 else f"model.{k}"): v.clone()
        for i, (k, v) in enumerate(dit_src.state_dict().items())
    }
    torch.save({"generator_ema": wrapped_dit}, dit_pt)

    ref_sf = tmp_path / "fastframes-s" / "kv_refiner.safetensors"
    ref_sf.parent.mkdir(parents=True, exist_ok=True)
    save_file(ref_src.state_dict(), str(ref_sf))

    dec_s_sf = tmp_path / "fastframes-s" / "pixel_decoder.safetensors"
    save_file({f"pixel_head.{k}": v.clone() for k, v in dec_s_src.state_dict().items()}, str(dec_s_sf))

    comps_s = load_fastframes_components(
        variant="fastframes-s",
        checkpoint_dir=tmp_path,
        wan_config=cfg,
        device="cpu",
        dit_dtype=torch.float32,
        decoder_dtype=torch.float32,
        load_text_encoder=False,
        stem_n_heads=4,
    )
    assert isinstance(comps_s.decoder, PixelDecoder)
    for k, v in dit_src.state_dict().items():
        assert torch.equal(comps_s.dit.state_dict()[k], v)
    for k, v in ref_src.state_dict().items():
        assert torch.equal(comps_s.refiner.state_dict()[k], v)
    for k, v in dec_s_src.state_dict().items():
        assert torch.equal(comps_s.decoder.state_dict()[k], v)

    # 2. FF-L style: sharded .safetensors.index.json DiT, .pt nested "adapter" refiner, bridged pixel_decoder.safetensors
    shard_dir = tmp_path / "fastframes-l"
    shard_dir.mkdir(parents=True, exist_ok=True)
    dit_items = list(dit_src.state_dict().items())
    mid = len(dit_items) // 2
    shard0 = dict(dit_items[:mid])
    shard1 = dict(dit_items[mid:])
    save_file(shard0, str(shard_dir / "dit-00001-of-00002.safetensors"))
    save_file(shard1, str(shard_dir / "dit-00002-of-00002.safetensors"))
    index_path = shard_dir / "dit.safetensors.index.json"
    weight_map = {
        **{k: "dit-00001-of-00002.safetensors" for k in shard0},
        **{k: "dit-00002-of-00002.safetensors" for k in shard1},
    }
    index_path.write_text(json.dumps({"weight_map": weight_map}), encoding="utf-8")

    ref_pt = shard_dir / "kv_refiner.pt"
    torch.save({"adapter": {f"_fsdp_wrapped_module.{k}": v.clone() for k, v in ref_src.state_dict().items()}}, ref_pt)

    dec_l_sf = shard_dir / "pixel_decoder.safetensors"
    save_file(dec_l_src.state_dict(), str(dec_l_sf))

    comps_l = load_fastframes_components(
        variant="fastframes-l",
        dit_path=index_path,
        refiner_path=ref_pt,
        decoder_path=dec_l_sf,
        wan_config=cfg,
        device="cpu",
        dit_dtype=torch.float32,
        decoder_dtype=torch.float32,
        load_text_encoder=False,
        stem_n_heads=4,
    )
    assert isinstance(comps_l.decoder, BridgedPixelDecoder)
    for k, v in dit_src.state_dict().items():
        assert torch.equal(comps_l.dit.state_dict()[k], v)
    for k, v in ref_src.state_dict().items():
        assert torch.equal(comps_l.refiner.state_dict()[k], v)
    for k, v in dec_l_src.state_dict().items():
        assert torch.equal(comps_l.decoder.state_dict()[k], v)

    # 3. Strictness checks: missing or unexpected keys must fail loudly.
    bad_state = dict(dit_src.state_dict())
    bad_state.pop("head.head.bias")
    with pytest.raises(RuntimeError, match="Missing key"):
        build_fastframes_dit(cfg, bad_state, dtype=torch.float32, device="cpu")

    extra_ref = dict(ref_src.state_dict())
    extra_ref["unexpected_param"] = torch.zeros(1)
    with pytest.raises(RuntimeError, match="Unexpected key"):
        build_fastframes_refiner(cfg, extra_ref, dtype=torch.float32, device="cpu", stem_n_heads=4)

    bad_bridge = dict(dec_l_src.state_dict())
    bad_bridge["bridge.weight"] = torch.zeros(24, cfg.dim + 8)
    with pytest.raises(ValueError, match="does not match wan_dim"):
        build_fastframes_decoder(cfg.dim, bad_bridge, dtype=torch.float32, device="cpu")

    with pytest.raises(KeyError, match="missing_key"):
        load_fastframes_state_dict(dit_pt, key="missing_key")
