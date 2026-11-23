# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 OpenEnvision
"""Weight-free pipeline, rollout parity, validation, MP4 export, and catalog tests for FastFrames."""

from __future__ import annotations

import importlib
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import torch
import yaml
from safetensors.torch import save_file

from worldfoundry.base_models.diffusion_model.models.networks.wan.variants.fastframes import (
    BridgedPixelDecoder,
    FastFramesCausalWanModel,
    FastFramesWanConfig,
    KVRefinementNetwork,
    PixelDecoder,
)
from worldfoundry.base_models.diffusion_model.schedulers.fastframes import (
    FF_L,
    FF_S,
    FastFramesVariantConfig,
)
from worldfoundry.core.media.processing.postprocess import VideoChunk
from worldfoundry.operators.fastframes_operator import FastFramesOperator
from worldfoundry.pipelines.fastframes import FastFramesPipeline, pixels_to_uint8
from worldfoundry.runtime.inference_specs.fastframes import FASTFRAMES_INFERENCE_SPEC

pytestmark = pytest.mark.unit

UPSTREAM_ROOT = Path(__file__).resolve().parents[2] / "tmp" / "upstream_refs" / "ml-fastframes"


def _import_upstream() -> dict[str, ModuleType] | None:
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
    }


def _tiny_wan_config(*, in_dim: int = 16) -> FastFramesWanConfig:
    return FastFramesWanConfig(
        dim=48,
        ffn_dim=96,
        num_heads=4,
        num_layers=2,
        patch_size=(1, 2, 2),
        text_len=8,
        in_dim=in_dim,
        freq_dim=16,
        text_dim=32,
        out_dim=in_dim,
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
    )


def _init_params(module: torch.nn.Module, seed: int) -> None:
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for param in module.parameters():
            param.copy_(torch.randn(param.shape, generator=gen, dtype=torch.float32) * 0.04)


class _DeterministicFakeTextEncoder:
    def __init__(self, text_dim: int = 32, seq_len: int = 6) -> None:
        self.text_dim = text_dim
        self.seq_len = seq_len
        self.calls: list[list[str]] = []

    def __call__(self, prompts: list[str]) -> torch.Tensor:
        self.calls.append(list(prompts))
        seed = sum(ord(ch) * (i + 1) for i, ch in enumerate(prompts[0])) + 1
        gen = torch.Generator().manual_seed(seed)
        return torch.randn(len(prompts), self.seq_len, self.text_dim, generator=gen, dtype=torch.bfloat16)


def _make_tiny_pipeline(
    *,
    variant_cfg: FastFramesVariantConfig = FF_S,
    bridged: bool = False,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cpu",
    seed_offset: int = 0,
) -> tuple[FastFramesPipeline, _DeterministicFakeTextEncoder, FastFramesWanConfig]:
    wan_cfg = _tiny_wan_config(in_dim=16)
    dit = FastFramesCausalWanModel(wan_cfg).eval()
    _init_params(dit, 100 + seed_offset)
    dit = dit.to(device=device, dtype=dtype)

    refiner = KVRefinementNetwork(
        dim=wan_cfg.dim,
        num_heads=wan_cfg.num_heads,
        head_dim=wan_cfg.dim // wan_cfg.num_heads,
        num_blocks=wan_cfg.num_layers,
        d_a=32,
        n_stem=2,
        stem_n_heads=4,
        stem_mlp_mult=2,
        head_mlp_mult=2,
        freq_dim=16,
    ).eval()
    _init_params(refiner, 200 + seed_offset)
    refiner = refiner.to(device=device, dtype=dtype)

    if bridged:
        decoder: PixelDecoder | BridgedPixelDecoder = BridgedPixelDecoder(
            in_dim=wan_cfg.dim, dim=24, n_low_blocks=1, n_high_blocks=1, ffn_mult=2
        ).eval()
    else:
        decoder = PixelDecoder(dim=wan_cfg.dim, n_low_blocks=1, n_high_blocks=1, ffn_mult=2).eval()
    _init_params(decoder, 300 + seed_offset)
    decoder = decoder.to(device=device, dtype=variant_cfg.decoder_dtype)

    text_encoder = _DeterministicFakeTextEncoder(text_dim=wan_cfg.text_dim)
    tiny_variant = replace(
        variant_cfg,
        num_frames=3,
        latent_size=(4, 4),
        num_pixel_frames=9,
    )
    pipe = FastFramesPipeline(
        config=tiny_variant,
        dit=dit,
        refiner=refiner,
        decoder=decoder,
        text_encoder=text_encoder,
        device=device,
    )
    return pipe, text_encoder, wan_cfg


@pytest.mark.parametrize("context", ["refiner", "dit"])
@pytest.mark.parametrize("recipe_name", ["ff-s", "ff-l"])
def test_rollout_and_decode_exact_parity_with_upstream(
    monkeypatch: pytest.MonkeyPatch,
    recipe_name: str,
    context: str,
) -> None:
    base_cfg = FF_S if recipe_name == "ff-s" else FF_L
    bridged = recipe_name == "ff-l"
    wf_pipe, text_enc, wan_cfg = _make_tiny_pipeline(
        variant_cfg=base_cfg,
        bridged=bridged,
        dtype=torch.bfloat16,
        device="cpu",
    )
    assert wf_pipe.dit is not None
    assert wf_pipe.refiner is not None

    prompt_embeds = text_enc(["A corgi surfing a wave at sunset"])
    wf_lat, wf_feat, wf_mod = wf_pipe.rollout(prompt_embeds, seed=7, context=context)
    wf_px = wf_pipe.decode(wf_feat, wf_mod)
    assert wf_lat.shape == (1, 3, wan_cfg.in_dim, 4, 4)
    assert wf_px.shape == (1, 9, 3, 32, 32)
    assert torch.isfinite(wf_lat).all() and torch.isfinite(wf_px).all()
    wf_u8 = pixels_to_uint8(wf_px[0])
    assert wf_u8.dtype == torch.uint8 and wf_u8.shape == (9, 32, 32, 3)

    up = _import_upstream()
    if up is None:
        return

    # Patch upstream attention to auto on CPU (upstream hardcodes cuDNN SDPA even on CPU).
    orig_up_attn = up["attn"].attention
    monkeypatch.setattr(up["causal"], "attention", lambda q, k, v, **kw: orig_up_attn(q, k, v, backend="auto"))
    monkeypatch.setattr(up["model"], "cross_attention", lambda q, k, v: orig_up_attn(q, k, v, backend="auto"))

    # Build weight-shared upstream FastFrames instance.
    up_wan_cfg = up["model"].WanConfig(
        dim=wan_cfg.dim,
        ffn_dim=wan_cfg.ffn_dim,
        num_heads=wan_cfg.num_heads,
        num_layers=wan_cfg.num_layers,
        patch_size=wan_cfg.patch_size,
        text_len=wan_cfg.text_len,
        in_dim=wan_cfg.in_dim,
        freq_dim=wan_cfg.freq_dim,
        text_dim=wan_cfg.text_dim,
        out_dim=wan_cfg.out_dim,
        qk_norm=wan_cfg.qk_norm,
        cross_attn_norm=wan_cfg.cross_attn_norm,
        eps=wan_cfg.eps,
    )
    up_dit = up["causal"].CausalWanModel(up_wan_cfg).eval()
    up_dit.load_state_dict(wf_pipe.dit.state_dict(), strict=True)
    up_dit = up_dit.to(dtype=torch.bfloat16)

    up_ref = (
        up["kv"]
        .KVRefinementNetwork(
            dim=wan_cfg.dim,
            num_heads=wan_cfg.num_heads,
            head_dim=wan_cfg.dim // wan_cfg.num_heads,
            num_blocks=wan_cfg.num_layers,
            d_a=32,
            n_stem=2,
            stem_n_heads=4,
            stem_mlp_mult=2,
            head_mlp_mult=2,
            freq_dim=16,
        )
        .eval()
    )
    up_ref.load_state_dict(wf_pipe.refiner.state_dict(), strict=True)
    up_ref = up_ref.to(dtype=torch.bfloat16)

    up_cfg = up["pipe"].FastFramesConfig(
        name=wf_pipe.config.name,
        wan_model=wf_pipe.config.wan_model,
        first_frame_timesteps=wf_pipe.config.first_frame_timesteps,
        timesteps=wf_pipe.config.timesteps,
        refine_timestep=wf_pipe.config.refine_timestep,
        noise=wf_pipe.config.noise,
        autocast=False,  # CPU execution
        fast_ops=wf_pipe.config.fast_ops,
        decoder_dtype=wf_pipe.config.decoder_dtype,
        contiguous_stem=wf_pipe.config.contiguous_stem,
        allow_tf32=wf_pipe.config.allow_tf32,
        num_frames=3,
        latent_size=(4, 4),
        num_pixel_frames=9,
    )
    up_ff = up["pipe"].FastFrames(up_cfg, up_dit, up_ref, wf_pipe.decoder, text_encoder=None)

    up_lat, up_feat, up_mod = up_ff.rollout(prompt_embeds, seed=7, context=context)
    assert torch.equal(wf_lat, up_lat)
    assert torch.equal(wf_feat, up_feat)
    assert torch.equal(wf_mod, up_mod)
    assert torch.equal(wf_u8, up["pipe"].to_uint8(wf_px[0]))


def test_pipeline_context_modes_seed_repeatability_and_video_chunk() -> None:
    pipe, text_enc, _ = _make_tiny_pipeline(variant_cfg=FF_S, bridged=False, dtype=torch.bfloat16, device="cpu")

    out_refiner_a = pipe(
        prompt="A golden retriever on a sunlit beach",
        seed=7,
        context="refiner",
        num_frames=9,
        height=32,
        width=32,
        return_latents=True,
        return_dict=True,
    )
    out_other_seed = pipe(
        prompt="A golden retriever on a sunlit beach",
        seed=13,
        context="refiner",
        num_frames=9,
        height=32,
        width=32,
        return_latents=True,
        return_dict=True,
    )
    out_refiner_a2 = pipe(
        prompt="A golden retriever on a sunlit beach",
        seed=7,
        context="refiner",
        num_frames=9,
        height=32,
        width=32,
        return_latents=True,
        return_dict=True,
    )
    out_dit = pipe(
        prompt="A golden retriever on a sunlit beach",
        seed=7,
        context="dit",
        num_frames=9,
        height=32,
        width=32,
        return_latents=True,
        return_dict=True,
    )

    assert len(text_enc.calls) == 4
    assert out_refiner_a["model_id"] == "fastframes"
    assert out_refiner_a["variant"] == "fastframes-s"
    assert out_refiner_a["fps"] == 16
    assert out_refiner_a["video"].dtype == np.uint8
    assert out_refiner_a["video"].shape == (9, 32, 32, 3)

    chunk = out_refiner_a["video_chunk"]
    assert isinstance(chunk, VideoChunk)
    assert chunk.layout == "thwc"
    assert chunk.value_range == "0,255"
    assert chunk.frame_count == 9
    assert np.array_equal(chunk.frames, out_refiner_a["video"])

    # A/B/A seed repeatability
    assert np.array_equal(out_refiner_a["video"], out_refiner_a2["video"])
    assert torch.equal(out_refiner_a["latents"], out_refiner_a2["latents"])
    assert not torch.equal(out_refiner_a["latents"], out_other_seed["latents"])

    # Context="refiner" vs context="dit": frame 0 matches (both run clean t=0 DiT), frame 2 diverges after frame 1 KV repair!
    assert torch.equal(out_refiner_a["latents"][:, :2], out_dit["latents"][:, :2])
    assert not torch.equal(out_refiner_a["latents"][:, 2:], out_dit["latents"][:, 2:])


def test_pipeline_mp4_export_and_return_dict_false(tmp_path: Path) -> None:
    pipe, _, _ = _make_tiny_pipeline(variant_cfg=FF_S, bridged=False, dtype=torch.bfloat16, device="cpu")

    explicit_mp4 = tmp_path / "corgi.mp4"
    res = pipe(
        prompt="A corgi surfing a wave at sunset",
        seed=0,
        context="refiner",
        num_frames=5,
        height=32,
        width=32,
        output_path=explicit_mp4,
        return_dict=True,
    )
    assert res["artifact_path"] == str(explicit_mp4)
    assert res["output_path"] == str(explicit_mp4)
    assert explicit_mp4.is_file() and explicit_mp4.stat().st_size > 0

    out_dir = tmp_path / "dir_out"
    returned_path = pipe(
        prompt="A corgi surfing a wave at sunset",
        seed=3,
        context="refiner",
        num_frames=5,
        height=32,
        width=32,
        output_dir=out_dir,
        return_dict=False,
    )
    assert returned_path == str(out_dir / "fastframes-s_seed3.mp4")
    assert Path(returned_path).is_file() and Path(returned_path).stat().st_size > 0


def test_operator_and_pipeline_input_validation() -> None:
    operator = FastFramesOperator()
    operator.get_interaction("  A sunlit forest  ")
    assert operator.process_interaction() == {"prompt": "A sunlit forest"}
    operator.delete_last_interaction()

    pipe, _, _ = _make_tiny_pipeline(variant_cfg=FF_S, bridged=False, dtype=torch.bfloat16, device="cpu")

    with pytest.raises(ValueError, match="non-empty"):
        pipe(prompt="   ", num_frames=5, height=32, width=32)
    with pytest.raises(ValueError, match="context"):
        pipe(prompt="valid", context="invalid", num_frames=5, height=32, width=32)
    with pytest.raises(ValueError, match="non-negative"):
        pipe(prompt="valid", seed=-1, num_frames=5, height=32, width=32)
    with pytest.raises(TypeError, match="integer"):
        pipe(prompt="valid", seed=True, num_frames=5, height=32, width=32)
    with pytest.raises(ValueError, match="4k\\+1"):
        pipe(prompt="valid", num_frames=6, height=32, width=32)
    with pytest.raises(ValueError, match="multiples of 16"):
        pipe(prompt="valid", num_frames=5, height=24, width=32)
    with pytest.raises(ValueError, match="noise mode"):
        pipe(prompt="valid", noise="gaussian", num_frames=5, height=32, width=32)
    with pytest.raises(ValueError, match="image"):
        pipe(prompt="valid", images="frame.png", num_frames=5, height=32, width=32)
    with pytest.raises(ValueError, match="video"):
        pipe(prompt="valid", video="clip.mp4", num_frames=5, height=32, width=32)
    with pytest.raises(ValueError, match="without reloading"):
        pipe(prompt="valid", variant="fastframes-l", num_frames=5, height=32, width=32)
    with pytest.raises(TypeError, match="Unexpected"):
        pipe(prompt="valid", unknown_flag=1, num_frames=5, height=32, width=32)


def test_pipeline_from_pretrained_loads_checkpoint_dir(tmp_path: Path) -> None:
    wan_cfg = _tiny_wan_config(in_dim=16)
    dit = FastFramesCausalWanModel(wan_cfg).eval()
    refiner = KVRefinementNetwork(
        dim=wan_cfg.dim,
        num_heads=wan_cfg.num_heads,
        head_dim=wan_cfg.dim // wan_cfg.num_heads,
        num_blocks=wan_cfg.num_layers,
        d_a=32,
        n_stem=2,
        stem_n_heads=4,
        stem_mlp_mult=2,
        head_mlp_mult=2,
        freq_dim=16,
    ).eval()
    decoder = PixelDecoder(dim=wan_cfg.dim, n_low_blocks=1, n_high_blocks=1, ffn_mult=2).eval()

    dit_pt = tmp_path / "causal-forcing++" / "framewise-2step.pt"
    dit_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"generator_ema": dit.state_dict()}, dit_pt)

    ff_dir = tmp_path / "fastframes-s"
    ff_dir.mkdir(parents=True, exist_ok=True)
    save_file(refiner.state_dict(), str(ff_dir / "kv_refiner.safetensors"))
    save_file(decoder.state_dict(), str(ff_dir / "pixel_decoder.safetensors"))

    pipe = FastFramesPipeline.from_pretrained(
        model_path=tmp_path,
        variant="ff-s",
        device="cpu",
        wan_config=wan_cfg,
        dit_dtype=torch.bfloat16,
        decoder_dtype=torch.float32,
        load_text_encoder=False,
        stem_n_heads=4,
        configure_backends=False,
    )
    assert pipe.MODEL_ID == "fastframes"
    assert pipe.variant == "fastframes-s"
    embeds = _DeterministicFakeTextEncoder(text_dim=wan_cfg.text_dim)(["A sunset wave"])
    res = pipe(prompt_embeds=embeds, seed=0, num_frames=5, height=32, width=32, return_dict=True)
    assert res["video"].shape == (5, 32, 32, 3)
    assert res["video"].dtype == np.uint8


def test_catalog_binding_profile_environment_and_inference_spec() -> None:
    from worldfoundry.zoo.bindings import resolve_pipeline_binding
    from worldfoundry.zoo.catalog.manifest import (
        load_model_catalog_manifest,
        load_model_catalog_manifests,
    )

    repo_root = Path(__file__).resolve().parents[2]
    catalog_path = repo_root / "worldfoundry" / "data" / "models" / "catalog" / "world_models" / "fastframes.yaml"
    profile_path = repo_root / "worldfoundry" / "data" / "models" / "runtime" / "profiles" / "fastframes.yaml"
    env_path = repo_root / "worldfoundry" / "data" / "models" / "runtime" / "environments" / "video" / "fastframes.yaml"

    manifest = load_model_catalog_manifest(catalog_path)
    assert manifest.model_id == "fastframes"
    all_ids = {entry.model_id for entry in load_model_catalog_manifests()}
    assert "fastframes" in all_ids

    for alias in ("fastframes", "fastframes-s", "fastframes-l", "ff-s", "ff-l"):
        binding = resolve_pipeline_binding(alias)
        assert binding.binding_id == "fastframes"
        assert binding.pipeline_target == "worldfoundry.pipelines.fastframes.pipeline_fastframes:FastFramesPipeline"

    profile = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    assert profile["model_id"] == "fastframes"
    assert profile["execution"]["pipeline_binding"] == "fastframes"
    assert profile["execution"]["defaults"]["fps"] == 16

    env_spec = yaml.safe_load(env_path.read_text(encoding="utf-8"))
    assert env_spec["model_id"] == "fastframes"
    assert env_spec["env_name"] == "worldfoundry-unified-cu128"

    assert FASTFRAMES_INFERENCE_SPEC.model_family_id == "fastframes"
    variant_ids = {variant.variant_id for variant in FASTFRAMES_INFERENCE_SPEC.variants}
    assert variant_ids == {"fastframes-s", "fastframes-l"}
    assert FASTFRAMES_INFERENCE_SPEC.default_variant_id == "fastframes-s"
    assert FASTFRAMES_INFERENCE_SPEC.default_task_id == "text-to-video"


@pytest.mark.gpu
@pytest.mark.parametrize("recipe_name", ["ff-s", "ff-l"])
def test_cuda_bf16_cudnn_sdpa_parity_with_unpatched_upstream(recipe_name: str) -> None:
    assert torch.cuda.is_available(), "CUDA required for GPU cuDNN SDPA parity test"
    base_cfg = FF_S if recipe_name == "ff-s" else FF_L
    bridged = recipe_name == "ff-l"
    device = "cuda:0"

    # Real Wan head_dim=128 so cuDNN SDPA executes directly against unpatched upstream.
    wan_cfg = FastFramesWanConfig(
        dim=256,
        ffn_dim=512,
        num_heads=2,
        num_layers=2,
        patch_size=(1, 2, 2),
        text_len=8,
        in_dim=16,
        freq_dim=64,
        text_dim=64,
        out_dim=16,
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
    )
    wf_dit = FastFramesCausalWanModel(wan_cfg).eval()
    _init_params(wf_dit, 401)
    wf_dit = wf_dit.to(device=device, dtype=torch.bfloat16)

    wf_ref = KVRefinementNetwork(
        dim=wan_cfg.dim,
        num_heads=wan_cfg.num_heads,
        head_dim=wan_cfg.dim // wan_cfg.num_heads,
        num_blocks=wan_cfg.num_layers,
        d_a=128,
        n_stem=2,
        stem_n_heads=4,
        stem_mlp_mult=2,
        head_mlp_mult=2,
        freq_dim=64,
    ).eval()
    _init_params(wf_ref, 402)
    wf_ref = wf_ref.to(device=device, dtype=torch.bfloat16)

    if bridged:
        wf_dec: PixelDecoder | BridgedPixelDecoder = BridgedPixelDecoder(
            in_dim=wan_cfg.dim, dim=64, n_low_blocks=1, n_high_blocks=1, ffn_mult=2
        ).eval()
    else:
        wf_dec = PixelDecoder(dim=wan_cfg.dim, n_low_blocks=1, n_high_blocks=1, ffn_mult=2).eval()
    _init_params(wf_dec, 403)
    wf_dec = wf_dec.to(device=device, dtype=base_cfg.decoder_dtype)

    tiny_cfg = replace(base_cfg, num_frames=3, latent_size=(4, 4), num_pixel_frames=9)
    wf_pipe = FastFramesPipeline(
        config=tiny_cfg,
        dit=wf_dit,
        refiner=wf_ref,
        decoder=wf_dec,
        text_encoder=None,
        device=device,
    )

    gen = torch.Generator(device=device).manual_seed(409)
    prompt_embeds = torch.randn(1, 6, wan_cfg.text_dim, generator=gen, device=device, dtype=torch.bfloat16)

    wf_lat, wf_feat, wf_mod = wf_pipe.rollout(prompt_embeds, seed=5, context="refiner")
    wf_px = wf_pipe.decode(wf_feat, wf_mod)
    assert wf_px.shape == (1, 9, 3, 32, 32) and torch.isfinite(wf_px).all()

    up = _import_upstream()
    if up is None:
        return

    up_wan_cfg = up["model"].WanConfig(
        dim=wan_cfg.dim,
        ffn_dim=wan_cfg.ffn_dim,
        num_heads=wan_cfg.num_heads,
        num_layers=wan_cfg.num_layers,
        patch_size=wan_cfg.patch_size,
        text_len=wan_cfg.text_len,
        in_dim=wan_cfg.in_dim,
        freq_dim=wan_cfg.freq_dim,
        text_dim=wan_cfg.text_dim,
        out_dim=wan_cfg.out_dim,
        qk_norm=wan_cfg.qk_norm,
        cross_attn_norm=wan_cfg.cross_attn_norm,
        eps=wan_cfg.eps,
    )
    up_dit = up["causal"].CausalWanModel(up_wan_cfg).eval()
    up_dit.load_state_dict({k: v.cpu() for k, v in wf_dit.state_dict().items()}, strict=True)
    up_dit = up_dit.to(device=device, dtype=torch.bfloat16)

    up_ref = (
        up["kv"]
        .KVRefinementNetwork(
            dim=wan_cfg.dim,
            num_heads=wan_cfg.num_heads,
            head_dim=wan_cfg.dim // wan_cfg.num_heads,
            num_blocks=wan_cfg.num_layers,
            d_a=128,
            n_stem=2,
            stem_n_heads=4,
            stem_mlp_mult=2,
            head_mlp_mult=2,
            freq_dim=64,
        )
        .eval()
    )
    up_ref.load_state_dict({k: v.cpu() for k, v in wf_ref.state_dict().items()}, strict=True)
    up_ref = up_ref.to(device=device, dtype=torch.bfloat16)

    up_cfg = up["pipe"].FastFramesConfig(
        name=tiny_cfg.name,
        wan_model=tiny_cfg.wan_model,
        first_frame_timesteps=tiny_cfg.first_frame_timesteps,
        timesteps=tiny_cfg.timesteps,
        refine_timestep=tiny_cfg.refine_timestep,
        noise=tiny_cfg.noise,
        autocast=tiny_cfg.autocast,
        fast_ops=tiny_cfg.fast_ops,
        decoder_dtype=tiny_cfg.decoder_dtype,
        contiguous_stem=tiny_cfg.contiguous_stem,
        allow_tf32=tiny_cfg.allow_tf32,
        num_frames=3,
        latent_size=(4, 4),
        num_pixel_frames=9,
    )
    up_ff = up["pipe"].FastFrames(up_cfg, up_dit, up_ref, wf_dec, text_encoder=None)

    up_lat, up_feat, up_mod = up_ff.rollout(prompt_embeds, seed=5, context="refiner")
    assert torch.equal(wf_lat, up_lat)
    assert torch.equal(wf_feat, up_feat)
    assert torch.equal(wf_mod, up_mod)

    up_px = up_ff.decode(up_feat, up_mod)
    assert torch.equal(wf_px, up_px)
    assert torch.equal(pixels_to_uint8(wf_px[0]), up["pipe"].to_uint8(up_px[0]))
