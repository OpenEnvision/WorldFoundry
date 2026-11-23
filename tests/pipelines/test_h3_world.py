"""Weight-free and CUDA-parity contracts for H3-World on the canonical MiniMax-H3 runtime."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image
from safetensors.torch import save_file

from worldfoundry.base_models.diffusion_model.models.networks.minimax_h3 import (
    MiniMaxH3DiTArchConfig,
    MiniMaxH3DiTModel,
)
from worldfoundry.base_models.diffusion_model.models.networks.minimax_h3.loading import (
    convert_minimax_h3_dit_state_dict,
)
from worldfoundry.base_models.diffusion_model.models.networks.minimax_h3.variants.h3_world import (
    H3WorldActionMaskSpec,
    build_h3_world_sdpa_mask,
    h3_world_directed_attention_context,
    install_h3_world_directed_attention,
    load_h3_world_adapter,
)
from worldfoundry.pipelines.h3_world import pipeline_h3_world as h3_world_module
from worldfoundry.pipelines.h3_world.action_script import (
    H3_WORLD_ACTION_PRESETS,
    H3_WORLD_KEYS9,
    annotate_from_keys9,
    null_action_script,
    resolve_action_script,
)
from worldfoundry.pipelines.h3_world.pipeline_h3_world import H3WorldPipeline
from worldfoundry.pipelines.minimax._minimax_h3 import (
    MiniMaxH3DenoiseBranch,
    minimax_h3_packed_sequence,
)


def _tiny_config(*, num_layers: int = 1) -> MiniMaxH3DiTArchConfig:
    return MiniMaxH3DiTArchConfig(
        num_layers=num_layers,
        token_refiner_num_layers=1,
        hidden_size=256,
        num_attention_heads=2,
        attention_head_dim=128,
        ffn_hidden_size=512,
        latents_dim=24,
        audio_latents_dim=32,
        patch_size=(1, 2, 2),
        text_dim=64,
        timestep_input_dim=256,
        time_embed_hidden_size=256,
        time_embed_dim=128,
        adaln_out_features=18 * 256,
        final_adaln_out_features=2 * 256,
        rope_inv_freq_len=16,
    )


def _tiny_fp32_model(*, num_layers: int = 1, seed: int = 7) -> MiniMaxH3DiTModel:
    torch.manual_seed(seed)
    model = MiniMaxH3DiTModel(_tiny_config(num_layers=num_layers)).float().eval()
    inv_freq = model.rope.get_buffer("inv_freq")
    inv_freq.copy_(
        1.0
        / (
            10000.0
            ** (torch.arange(0, model.arch.rope_inv_freq_len, dtype=torch.float32) / model.arch.rope_inv_freq_len)
        )
    )
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "adaln_proj" not in name and "time_embedder" not in name:
                param.mul_(0.25)
    return model


@pytest.mark.unit
def test_action_presets_and_key_purification_match_upstream_rules() -> None:
    expected = {
        "still": "the man stands still, camera holds steady",
        "forward": "the man walks forward, camera follows him",
        "back": "the man walks backward, camera follows him",
        "strafe-left": "the man strafes left, camera follows him",
        "strafe-right": "the man strafes right, camera follows him",
        "tilt-down": "the man stands still, camera tilts down",
        "tilt-up": "the man stands still, camera tilts up",
        "pan-left": "the man stands still, camera pans left slowly",
        "pan-right": "the man stands still, camera pans right slowly",
        "pan-left-fast": "the man stands still, camera pans left sharply",
        "pan-right-fast": "the man stands still, camera pans right sharply",
    }
    assert set(H3_WORLD_ACTION_PRESETS) == set(expected)
    for preset, sentence in expected.items():
        script = resolve_action_script(latent_t=3, action_preset=preset)
        assert script == [sentence] * 3

    # Mutually exclusive keys cancel cleanly; compound keys preserve W/S/A/D and pan/tilt order.
    k9 = np.zeros((2, len(H3_WORLD_KEYS9)), dtype=np.int64)
    k9[0, H3_WORLD_KEYS9.index("W")] = 1
    k9[0, H3_WORLD_KEYS9.index("S")] = 1
    k9[0, H3_WORLD_KEYS9.index("J")] = 1
    k9[0, H3_WORLD_KEYS9.index("L")] = 1
    k9[1, H3_WORLD_KEYS9.index("W")] = 1
    k9[1, H3_WORLD_KEYS9.index("A")] = 1
    k9[1, H3_WORLD_KEYS9.index("J")] = 1
    k9[1, H3_WORLD_KEYS9.index("K")] = 1
    k9[1, H3_WORLD_KEYS9.index("F")] = 1
    assert annotate_from_keys9(k9) == [
        "the man stands still, camera holds steady",
        "the man walks forward and strafes left, camera pans left sharply and tilts up",
    ]
    assert null_action_script(2) == ["the man stands still, camera holds steady"] * 2


@pytest.mark.unit
def test_resolve_action_script_supports_schedules_and_rejects_conflicts() -> None:
    scheduled = resolve_action_script(
        latent_t=4,
        interactions=[
            {"preset": "pan-left-fast", "latents": 2},
            {"keys": ["W", "L", "F"], "latents": 2},
        ],
    )
    assert scheduled == [
        "the man stands still, camera pans left sharply",
        "the man stands still, camera pans left sharply",
        "the man walks forward, camera pans right sharply",
        "the man walks forward, camera pans right sharply",
    ]
    assert (
        resolve_action_script(latent_t=2, interactions="W+D")
        == ["the man walks forward and strafes right, camera follows him"] * 2
    )

    with pytest.raises(ValueError, match="at most one"):
        resolve_action_script(latent_t=2, action_preset="forward", interactions="back")
    with pytest.raises(ValueError, match="cover 3 latents"):
        resolve_action_script(latent_t=4, interactions=[{"preset": "forward", "latents": 3}])


@pytest.mark.unit
def test_h3_world_lora_reorders_grouped_qkv_to_match_diffsynth_math(tmp_path: Path) -> None:
    torch.manual_seed(11)
    cfg = _tiny_config(num_layers=1)
    model = MiniMaxH3DiTModel(cfg).float().eval()

    num_heads, head_dim, hidden = cfg.num_attention_heads, cfg.attention_head_dim, cfg.hidden_size
    inner_dim = num_heads * head_dim
    rank = 4

    # Construct grouped-QKV base weights as released in MiniMax-H3 FL2VA (and loaded by DiffSynth).
    w_qkv_grouped = torch.randn(3 * inner_dim, hidden, dtype=torch.float32) * 0.1
    w_out = torch.randn(hidden, inner_dim, dtype=torch.float32) * 0.1
    w_ref_qkv_grouped = torch.randn(3 * inner_dim, hidden, dtype=torch.float32) * 0.1

    converted_base = convert_minimax_h3_dit_state_dict(
        {
            "blocks.0.attn.qkv_proj.weight": w_qkv_grouped,
            "blocks.0.attn.out_proj.weight": w_out,
            "token_refiner.blocks.0.attn.qkv_proj.weight": w_ref_qkv_grouped,
        },
        cfg,
    )
    block0_qkv = cast(nn.Linear, model.get_submodule("blocks.0.attn.qkv_proj"))
    block0_out = cast(nn.Linear, model.get_submodule("blocks.0.attn.out_proj"))
    refiner0_qkv = cast(nn.Linear, model.get_submodule("token_refiner.blocks.0.attn.qkv_proj"))
    with torch.no_grad():
        block0_qkv.weight.copy_(converted_base["blocks.0.attn.qkv_proj.weight"])
        block0_out.weight.copy_(converted_base["blocks.0.attn.out_proj.weight"])
        refiner0_qkv.weight.copy_(converted_base["token_refiner.blocks.0.attn.qkv_proj.weight"])

    lora_qkv_a = torch.randn(rank, hidden, dtype=torch.float32) * 0.1
    lora_qkv_b = torch.randn(3 * inner_dim, rank, dtype=torch.float32) * 0.1
    lora_out_a = torch.randn(rank, inner_dim, dtype=torch.float32) * 0.1
    lora_out_b = torch.randn(hidden, rank, dtype=torch.float32) * 0.1
    lora_ref_a = torch.randn(rank, hidden, dtype=torch.float32) * 0.1
    lora_ref_b = torch.randn(3 * inner_dim, rank, dtype=torch.float32) * 0.1

    adapter_path = tmp_path / "step-10000.safetensors"
    save_file(
        {
            "blocks.0.attn.qkv_proj.lora_A.default.weight": lora_qkv_a,
            "blocks.0.attn.qkv_proj.lora_B.default.weight": lora_qkv_b,
            "blocks.0.attn.out_proj.lora_A.default.weight": lora_out_a,
            "blocks.0.attn.out_proj.lora_B.default.weight": lora_out_b,
            "token_refiner.blocks.0.attn.qkv_proj.lora_A.default.weight": lora_ref_a,
            "token_refiner.blocks.0.attn.qkv_proj.lora_B.default.weight": lora_ref_b,
        },
        str(adapter_path),
    )

    report = load_h3_world_adapter(model, adapter_path, rank=rank)
    assert len(report["merged_targets"]) == 3
    assert report["alpha"] == rank

    # Compare WorldFoundry split-[q_all, k_all, v_all] projections against DiffSynth grouped-[H, 3, D] math.
    x = torch.randn(7, hidden, dtype=torch.float32)
    ref_qkv = (x @ (w_qkv_grouped + lora_qkv_b @ lora_qkv_a).T).view(7, num_heads, 3, head_dim)
    ref_q, ref_k, ref_v = ref_qkv[:, :, 0, :], ref_qkv[:, :, 1, :], ref_qkv[:, :, 2, :]

    wf_qkv = block0_qkv(x)
    wf_q, wf_k, wf_v = (part.view(7, num_heads, head_dim) for part in wf_qkv.split(inner_dim, dim=-1))
    torch.testing.assert_close(wf_q, ref_q, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(wf_k, ref_k, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(wf_v, ref_v, atol=1e-5, rtol=1e-5)

    attn_hidden = torch.randn(7, inner_dim, dtype=torch.float32)
    ref_out = attn_hidden @ (w_out + lora_out_b @ lora_out_a).T
    torch.testing.assert_close(block0_out(attn_hidden), ref_out, atol=1e-5, rtol=1e-5)

    # Also verify non-LoRA keys are rejected.
    bad_path = tmp_path / "bad.safetensors"
    save_file({"blocks.0.mlp.fc1.weight": torch.zeros(4, 4)}, str(bad_path))
    with pytest.raises(ValueError, match="Unsupported H3-World adapter tensor"):
        load_h3_world_adapter(model, bad_path, rank=rank)


@pytest.mark.unit
def test_mirrored_rope_offsets_and_segmented_token_refiner_isolation() -> None:
    head_len = 24
    spans = [(24, 28), (28, 33)]
    text_len = 33
    packed = minimax_h3_packed_sequence(
        text_len=text_len,
        latent_t=2,
        latent_h=4,
        latent_w=4,
        audio_t=4,
        include_keyframe_cond=True,
        keyframe_frame_indices=(0,),
        frame_count=5,
        action_text_spans=spans,
    )
    assert int(packed["cu_seqlens"][1]) == text_len + 4 + 8 + 8  # 53
    assert packed["seq_len"] == 64

    # Mirrored offset invariant: every frame k sees its own annotation at the same constant negative delta.
    g_t = packed["img_position_ids"][:, 0]
    v0 = int(packed["action_video_start"])
    fr = int(packed["action_frame_rows"])
    deltas = [float(g_t[v0 + k * fr]) - float(g_t[lo]) for k, (lo, _) in enumerate(spans)]
    assert deltas[0] == pytest.approx(deltas[1], abs=1e-12)
    assert deltas[0] > 0.0
    assert float(g_t[spans[-1][1] - 1]) < float(text_len)

    # Token Refiner isolation: perturbing action sentence 0 must NOT leak into head or sentence 1.
    model = _tiny_fp32_model(num_layers=1)
    branch = MiniMaxH3DenoiseBranch(
        packed=packed,
        text_embeddings=torch.randn(text_len, model.arch.text_dim),
        token_tags=packed["token_tags"],
        device=torch.device("cpu"),
    )
    refiner_cu = branch.static_kwargs["refiner_packed_seq_params"]["cu_seqlens_q"]
    assert refiner_cu.tolist() == [0, head_len, 28, 33]

    embeds_a = torch.randn(text_len, model.arch.text_dim)
    embeds_b = embeds_a.clone()
    embeds_b[24:28] += 3.0
    with torch.no_grad():
        ref_a = model.refine_prompt_embeds(embeds_a, refiner_cu, device=torch.device("cpu"))
        ref_b = model.refine_prompt_embeds(embeds_b, refiner_cu, device=torch.device("cpu"))

    assert not torch.allclose(ref_a[24:28], ref_b[24:28])
    assert torch.equal(ref_a[:head_len], ref_b[:head_len])
    assert torch.equal(ref_a[28:33], ref_b[28:33])


@pytest.mark.unit
def test_directed_attention_cut_vertex_and_padding_isolation() -> None:
    model = _tiny_fp32_model(num_layers=1, seed=21)
    state = install_h3_world_directed_attention(model, use_flex=False)

    spans = [(24, 28), (28, 33)]
    text_len = 33
    packed = minimax_h3_packed_sequence(
        text_len=text_len,
        latent_t=2,
        latent_h=4,
        latent_w=4,
        audio_t=4,
        include_keyframe_cond=True,
        keyframe_frame_indices=(0,),
        frame_count=5,
        action_text_spans=spans,
    )
    device = torch.device("cpu")
    embeds_a = torch.randn(text_len, model.arch.text_dim)
    embeds_b = embeds_a.clone()
    embeds_b[24:28] += 5.0  # Perturb ONLY action annotation 0.

    branch_a = MiniMaxH3DenoiseBranch(
        packed=packed, text_embeddings=embeds_a, token_tags=packed["token_tags"], device=device
    )
    branch_b = MiniMaxH3DenoiseBranch(
        packed=packed, text_embeddings=embeds_b, token_tags=packed["token_tags"], device=device
    )
    plan_a = branch_a.prepare_timestep_plan(
        video_timesteps=[0.5], audio_timesteps=[0.5], imgvid_cond_noise_aug=0.999, audio_ref_cond_noise_aug=1.0
    )
    plan_b = branch_b.prepare_timestep_plan(
        video_timesteps=[0.5], audio_timesteps=[0.5], imgvid_cond_noise_aug=0.999, audio_ref_cond_noise_aug=1.0
    )
    video_rows = torch.randn(int(packed["img_pos"].shape[0]), 96)
    audio_rows = torch.randn(int(packed["audio_pos"].shape[0]), 32)
    spec = H3WorldActionMaskSpec(
        action_text_rows=packed["action_text_rows"],
        video_start=int(packed["action_video_start"]),
        frame_rows=int(packed["action_frame_rows"]),
    )

    with torch.no_grad(), h3_world_directed_attention_context(model, spec):
        vid_a, aud_a = model(
            **branch_a.forward_kwargs(video_rows=video_rows, audio_rows=audio_rows, step_timesteps=plan_a[0])
        )
        vid_b, aud_b = model(
            **branch_b.forward_kwargs(video_rows=video_rows, audio_rows=audio_rows, step_timesteps=plan_b[0])
        )

    assert state.spec is None  # context cleaned up cleanly
    fr = int(packed["action_frame_rows"])
    assert torch.isfinite(vid_a).all() and torch.isfinite(aud_a).all()
    # Frame 0 sees annotation 0 -> changes; frame 1 and audio must NOT see annotation 0 in a 1-layer DiT.
    assert float((vid_a[:fr] - vid_b[:fr]).abs().max()) > 1e-5
    assert float((vid_a[fr : 2 * fr] - vid_b[fr : 2 * fr]).abs().max()) < 1e-7
    assert float((aud_a - aud_b).abs().max()) < 1e-7

    # Context restores state even if an exception occurs inside the forward pass.
    with pytest.raises(RuntimeError, match="boom"):
        with h3_world_directed_attention_context(model, spec):
            assert state.spec is spec
            raise RuntimeError("boom")
    assert state.spec is None

    # Verify directed mask edges on the live segment:
    # annotation 0 reads head/cond/own video frame 0, but NOT annotation 1 or video frame 1.
    live_len = int(packed["cu_seqlens"][1])
    sdpa_mask = build_h3_world_sdpa_mask(spec, start=0, length=live_len, device=device)
    assert sdpa_mask is not None
    mask = sdpa_mask[0, 0]
    v0 = int(packed["action_video_start"])
    assert bool(mask[24:28, :24].all())  # annotation 0 reads head
    assert bool(mask[24:28, v0 : v0 + fr].all())  # annotation 0 reads own frame 0
    assert not bool(mask[24:28, 28:33].any())  # annotation 0 cannot read annotation 1
    assert not bool(mask[24:28, v0 + fr : v0 + 2 * fr].any())  # annotation 0 cannot read frame 1
    assert not bool(mask[:24, 24:33].any())  # head cannot read annotations
    assert bool(mask[v0 : v0 + fr, 24:28].all())  # frame 0 reads annotation 0
    assert not bool(mask[v0 : v0 + fr, 28:33].any())  # frame 0 cannot read annotation 1
    # Alignment-pad segment [live_len, seq_len) has no annotations and needs no directed mask.
    assert (
        build_h3_world_sdpa_mask(spec, start=live_len, length=int(packed["seq_len"]) - live_len, device=device) is None
    )


@pytest.mark.unit
def test_h3_world_from_pretrained_delegates_to_canonical_fl2va_and_installs_hooks(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, Any] = {}
    fake_transformer = _tiny_fp32_model(num_layers=1)

    def fake_load(cls, *, model_path, required_components, device, model_id, component_devices=None):
        del cls, model_path, device, model_id, component_devices
        captured["base_options"] = dict(required_components)
        return SimpleNamespace(
            transformer=fake_transformer,
            video_vae=None,
            audio_vae=None,
            text_encoder=None,
            tokenizer=None,
            device="cpu",
            component_devices={"transformer": "cpu"},
        )

    monkeypatch.setattr(h3_world_module.NativeMiniMaxH3Pipeline, "from_pretrained", classmethod(fake_load))
    monkeypatch.setattr(
        h3_world_module,
        "load_h3_world_adapter",
        lambda model, path, **kw: captured.update(adapter=(model, Path(path), kw)),
    )
    monkeypatch.setattr(H3WorldPipeline, "_load_processor", staticmethod(lambda _root: object()))

    adapter_file = tmp_path / "step-10000.safetensors"
    adapter_file.write_bytes(b"")
    pipe = H3WorldPipeline.from_pretrained(
        base_model_path=tmp_path,
        adapter_path=adapter_file,
        use_flex_attention=False,
    )

    assert captured["base_options"]["partition"] == "FL2VA"
    assert captured["adapter"][0] is fake_transformer
    assert captured["adapter"][1] == adapter_file
    assert captured["adapter"][2]["rank"] == 32
    assert hasattr(fake_transformer, "_h3_world_attention_state")
    assert pipe.MODEL_ID == "h3-world"


@pytest.mark.unit
def test_h3_world_pipeline_end_to_end_dedup_cfg_and_validation() -> None:
    model = _tiny_fp32_model(num_layers=1, seed=31)
    install_h3_world_directed_attention(model, use_flex=False)

    encoded_lines: list[str] = []

    class FakeTextEncoder:
        def encode_presentation(self, prompt, *, tokenizer, processor, images=()):
            del tokenizer, processor
            # Return head length >= 24 so mirrored RoPE offset fits latent_t=2 cleanly.
            length = 24 + len(images)
            seed = sum(ord(ch) for ch in prompt) + 1
            gen = torch.Generator().manual_seed(seed)
            hidden = torch.randn(1, length, model.arch.text_dim, generator=gen)
            tags = torch.ones(length, dtype=torch.long)
            tags[:2] = 0
            return hidden, tags

        def encode_ids(self, input_ids):
            seed = int(input_ids.sum().item()) + 17
            gen = torch.Generator().manual_seed(seed)
            return torch.randn(int(input_ids.shape[0]), model.arch.text_dim, generator=gen)

    class FakeTokenizer:
        def __call__(self, text, *, return_tensors="pt", add_special_tokens=False):
            del return_tensors, add_special_tokens
            encoded_lines.append(text)
            tokens = [10 + (ord(ch) % 50) for ch in text[:5]]
            return {"input_ids": torch.tensor([tokens], dtype=torch.long)}

    class FakeVideoVAE:
        sglang_config = SimpleNamespace(arch_config=SimpleNamespace(latents_mean=[0.0] * 24, latents_std=[1.0] * 24))

        def encode_images(self, images, *, deterministic=False):
            assert deterministic is True
            w, h = images[0].size
            return [torch.zeros(24, 1, h // 16, w // 16, dtype=torch.float32)]

    pipe = H3WorldPipeline(
        transformer=model,
        video_vae=FakeVideoVAE(),
        audio_vae=None,
        text_encoder=FakeTextEncoder(),
        tokenizer=FakeTokenizer(),
        processor=object(),
        device="cpu",
        use_flex_attention=False,
    )
    frame = Image.fromarray(np.full((64, 64, 3), 120, dtype=np.uint8))

    out_cfg1 = pipe(
        prompt="A man in a yellow shirt stands in a garage.",
        images=frame,
        action_preset="forward",
        height=64,
        width=64,
        num_frames=5,
        num_inference_steps=2,
        cfg_scale=1.0,
        seed=2,
        return_latent_rows=True,
        return_dict=True,
    )
    # Two identical latent clauses must be encoded only once via the dedup cache.
    assert encoded_lines == ["the man walks forward, camera follows him"]
    assert out_cfg1["model_id"] == "h3-world"
    assert out_cfg1["action_script"] == ["the man walks forward, camera follows him"] * 2
    assert torch.isfinite(out_cfg1["video_rows"]).all()

    out_cfg2 = pipe(
        prompt="A man in a yellow shirt stands in a garage.",
        images=frame,
        action_preset="forward",
        height=64,
        width=64,
        num_frames=5,
        num_inference_steps=2,
        cfg_scale=2.0,
        seed=2,
        return_latent_rows=True,
        return_dict=True,
    )
    assert torch.isfinite(out_cfg2["video_rows"]).all()
    assert not torch.allclose(out_cfg1["video_rows"], out_cfg2["video_rows"])

    with pytest.raises(ValueError, match="17k\\+5"):
        pipe(prompt="scene", images=frame, num_frames=100)
    with pytest.raises(ValueError, match="multiples of 32"):
        pipe(prompt="scene", images=frame, height=50, width=64, num_frames=5)
    with pytest.raises(ValueError, match="first-frame"):
        pipe(prompt="scene", images=None, num_frames=5)


@pytest.mark.unit
def test_h3_world_catalog_binding_and_inference_spec_resolve() -> None:
    from worldfoundry.runtime.inference_catalog import get_model_inference_spec
    from worldfoundry.zoo.bindings import resolve_pipeline_binding
    from worldfoundry.zoo.catalog import load_model_zoo_registry

    registry = load_model_zoo_registry()
    for alias in ("h3-world", "h3_world", "minimax-h3-world"):
        entry = registry.get(alias)
        assert entry.model_id == "h3-world"
        binding = resolve_pipeline_binding(alias)
        assert binding.pipeline_target == "worldfoundry.pipelines.h3_world.pipeline_h3_world:H3WorldPipeline"
        spec = get_model_inference_spec(alias)
        assert spec is not None
        assert spec.model_family_id == "h3-world"


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for flex_attention parity")
def test_h3_world_cuda_flex_attention_matches_sdpa_and_updates_buffers_between_masks() -> None:
    device = torch.device("cuda:0")
    torch.manual_seed(42)

    # Realistic head_dim=128, padded length with two distinct annotation layouts on the SAME length.
    total, num_heads, head_dim = 256, 4, 128
    cu = torch.tensor([0, 224, 256], dtype=torch.int32, device=device)
    q = torch.randn(total, num_heads, head_dim, device=device, dtype=torch.bfloat16) * 0.2
    k = torch.randn(total, num_heads, head_dim, device=device, dtype=torch.bfloat16) * 0.2
    v = torch.randn(total, num_heads, head_dim, device=device, dtype=torch.bfloat16) * 0.2
    scale = head_dim**-0.5

    spec_a = H3WorldActionMaskSpec(
        action_text_rows=torch.tensor([[40, 48], [48, 56]], dtype=torch.long),
        video_start=120,
        frame_rows=40,
    )
    spec_b = H3WorldActionMaskSpec(
        action_text_rows=torch.tensor([[48, 56], [40, 48]], dtype=torch.long),
        video_start=120,
        frame_rows=40,
    )

    from worldfoundry.base_models.diffusion_model.models.networks.minimax_h3.variants.h3_world import (
        H3WorldAttentionState,
        H3WorldDirectedAttention,
    )

    flex_state = H3WorldAttentionState(use_flex=True)
    sdpa_state = H3WorldAttentionState(use_flex=False)
    flex_attn = H3WorldDirectedAttention(flex_state)
    sdpa_attn = H3WorldDirectedAttention(sdpa_state)

    outputs_flex = []
    outputs_sdpa = []
    for spec in (spec_a, spec_b):
        flex_state.set_spec(spec)
        sdpa_state.set_spec(spec)
        with torch.no_grad():
            out_f = flex_attn(q, k, v, cu_seqlens=cu, softmax_scale=scale)
            out_s = sdpa_attn(q, k, v, cu_seqlens=cu, softmax_scale=scale)
        torch.testing.assert_close(out_f[:200], out_s[:200], atol=2e-2, rtol=2e-2)
        outputs_flex.append(out_f[:200].clone())
        outputs_sdpa.append(out_s[:200].clone())

    # Swapping the two annotation spans on the exact same length must change frame outputs (no stale closure).
    assert float((outputs_flex[0][120:200] - outputs_flex[1][120:200]).abs().max()) > 1e-3
    torch.testing.assert_close(
        outputs_flex[0] - outputs_flex[1], outputs_sdpa[0] - outputs_sdpa[1], atol=2e-2, rtol=2e-2
    )
