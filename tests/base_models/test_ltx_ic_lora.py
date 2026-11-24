# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""IC-LoRA checkpoint fusion and reference-video contracts on canonical LTX components."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from safetensors.torch import save_file

from worldfoundry.base_models.diffusion_model.components import ComponentBuildContext, ComponentKey, ComponentKind
from worldfoundry.base_models.diffusion_model.contracts import DiffusionRequest, ModalityState, MultiModalDenoiserInput
from worldfoundry.base_models.diffusion_model.loaders import CheckpointSpec
from worldfoundry.base_models.diffusion_model.models.denoisers.ltx import (
    LTXAVTransformerModule,
    build_ltx_joint_denoiser,
)
from worldfoundry.base_models.diffusion_model.models.denoisers.ltx_lora import (
    convert_ltx_ic_lora_state_dict,
    prepare_ltx_ic_lora_state_dict,
)
from worldfoundry.base_models.diffusion_model.models.initializers import ltx as initializer_module
from worldfoundry.base_models.diffusion_model.models.representations.ltx.conditioning import (
    add_ltx_noise,
    append_ltx_reference_video,
)
from worldfoundry.base_models.diffusion_model.models.representations.ltx.patchifiers import VideoLatentPatchifier
from worldfoundry.base_models.diffusion_model.models.representations.ltx.tools import VideoLatentTools
from worldfoundry.base_models.diffusion_model.models.representations.ltx.types import VideoLatentShape
from worldfoundry.base_models.diffusion_model.optimizations import RuntimePolicy, parse_offload_policy
from worldfoundry.base_models.diffusion_model.recipes import default_native_diffusion_registry
from worldfoundry.base_models.diffusion_model.recipes.ltx import ltx2_v2v_recipe, ltx23_v2v_recipe
from worldfoundry.base_models.diffusion_model.runners import ExecutionBuildContext
from worldfoundry.base_models.diffusion_model.runners.strategies import build_joint_multistage_strategy

pytestmark = pytest.mark.unit


def _config() -> dict[str, Any]:
    return {
        "transformer": {
            "dropout": 0.0,
            "attention_bias": True,
            "num_vector_embeds": None,
            "activation_fn": "gelu-approximate",
            "num_embeds_ada_norm": 1000,
            "use_linear_projection": False,
            "only_cross_attention": False,
            "cross_attention_norm": True,
            "double_self_attention": False,
            "upcast_attention": False,
            "standardization_norm": "rms_norm",
            "norm_elementwise_affine": False,
            "qk_norm": "rms_norm",
            "positional_embedding_type": "rope",
            "use_audio_video_cross_attention": True,
            "share_ff": False,
            "av_cross_ada_norm": True,
            "use_middle_indices_grid": True,
            "num_attention_heads": 2,
            "attention_head_dim": 8,
            "audio_num_attention_heads": 2,
            "audio_attention_head_dim": 4,
            "in_channels": 4,
            "out_channels": 4,
            "audio_in_channels": 4,
            "audio_out_channels": 4,
            "num_layers": 1,
            "cross_attention_dim": 16,
            "audio_cross_attention_dim": 8,
            "caption_proj_before_connector": True,
            "cross_attention_adaln": True,
            "apply_gated_attention": True,
        }
    }


def _weights(seed: int = 21) -> tuple[LTXAVTransformerModule, dict[str, torch.Tensor]]:
    module = LTXAVTransformerModule(_config(), split_checkpoint=True).eval()
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for value in module.state_dict(keep_vars=True).values():
            value.copy_(torch.randn(value.shape, generator=generator) * 0.05)
    lora = {}
    for stem in ("attn1.to_q", "attn1.to_k", "ff.net.0.proj", "audio_attn1.to_q"):
        name = f"transformer_blocks.0.{stem}"
        out_dim, in_dim = module.velocity_model.get_submodule(name).weight.shape
        lora[f"diffusion_model.{name}.lora_A.weight"] = torch.randn(2, in_dim, generator=generator) * 0.1
        lora[f"diffusion_model.{name}.lora_B.weight"] = torch.randn(out_dim, 2, generator=generator) * 0.1
    return module, lora


def _input() -> MultiModalDenoiserInput:
    generator = torch.Generator().manual_seed(9)
    states = {}
    for name, axes in (("video", 3), ("audio", 1)):
        latent = torch.randn(1, 4, 4, generator=generator)
        positions = torch.arange(4).float()[None, None, :, None].expand(1, axes, 4, 2).clone()
        positions[..., 1] += 1
        states[name] = ModalityState(latent, torch.ones(1, 4, 1), positions, torch.zeros_like(latent))
    return MultiModalDenoiserInput(
        modalities=states,
        timestep=torch.tensor(0.7),
        conditioning={
            "video_context": torch.randn(1, 5, 16, generator=generator),
            "audio_context": torch.randn(1, 5, 8, generator=generator),
        },
        step_index=0,
        total_steps=1,
    )


def test_ic_lora_mapping_and_lazy_fusion_preserve_clean_base() -> None:
    module, lora = _weights()
    base = module.state_dict()
    clean = {key: value.clone() for key, value in base.items()}
    mapped = convert_ltx_ic_lora_state_dict(lora)
    fused = prepare_ltx_ic_lora_state_dict(module, base, lora=mapped, strength=0.7, device=torch.device("cpu"))
    assert len(mapped) == len(lora) == 8
    assert set(fused) == set(base)
    key = "velocity_model.transformer_blocks.0.attn1.to_q.weight"
    stem = key.removesuffix(".weight")
    expected = (mapped[f"{stem}.lora_B.weight"].bfloat16() * 0.7) @ mapped[f"{stem}.lora_A.weight"].bfloat16()
    expected.add_(base[key])
    torch.testing.assert_close(fused[key], expected.float(), rtol=0, atol=0)
    for name in base:
        torch.testing.assert_close(base[name], clean[name], rtol=0, atol=0)
    assert not torch.equal(fused[key], clean[key])


@pytest.mark.parametrize("invalid", ["unknown-key", "missing-pair", "wrong-shape", "absent-target"])
def test_ic_lora_rejects_unmapped_or_incompatible_released_keys(invalid: str) -> None:
    module, lora = _weights()
    key = next(iter(lora))
    if invalid == "unknown-key":
        lora["unknown.weight"] = torch.ones(1)
    elif invalid == "missing-pair":
        lora.pop(key)
    elif invalid == "wrong-shape":
        lora[key] = torch.ones(2, 1)
    else:
        lora = {key.replace("transformer_blocks.0.", "transformer_blocks.7."): value for key, value in lora.items()}
    with pytest.raises(ValueError, match="LTX IC-LoRA"):
        prepare_ltx_ic_lora_state_dict(
            module,
            module.state_dict(),
            lora=convert_ltx_ic_lora_state_dict(lora),
            strength=1.0,
            device=torch.device("cpu"),
        )


@pytest.mark.parametrize("offload", ["none", "disk"])
def test_ic_lora_public_builder_loads_adapted_and_clean_weights_independently(tmp_path: Path, offload: str) -> None:
    module, lora = _weights()
    checkpoint = tmp_path / "base.safetensors"
    adapter = tmp_path / "ic-lora.safetensors"
    save_file(
        {f"model.diffusion_model.{key}": value for key, value in module.velocity_model.state_dict().items()},
        str(checkpoint),
        metadata={"config": json.dumps(_config())},
    )
    save_file(lora, str(adapter))
    context = ComponentBuildContext(
        model_id="ltx-2-v2v",
        key=ComponentKey(ComponentKind.DENOISER),
        policy=RuntimePolicy(device="cpu", dtype=torch.float32, offload=parse_offload_policy(offload)),
        checkpoints={"weights": CheckpointSpec(source=checkpoint), "ic_lora": CheckpointSpec(source=adapter)},
        component_options={"split_checkpoint": True, "precision": "weights", "lora_strength": 0.7},
    )
    adapted = build_ltx_joint_denoiser(context)
    clean = build_ltx_joint_denoiser(replace(context, checkpoints={"weights": context.require_checkpoint("weights")}))
    expected_weights = prepare_ltx_ic_lora_state_dict(
        module,
        module.state_dict(),
        lora=convert_ltx_ic_lora_state_dict(lora),
        strength=0.7,
        device=torch.device("cpu"),
    )
    from worldfoundry.base_models.diffusion_model.models.denoisers.ltx import LTXJointDenoiser

    with torch.no_grad():
        clean_expected = LTXJointDenoiser(module, compute_dtype=torch.float32, precision="weights")(_input()).samples
        clean_actual = clean(_input()).samples
        module.load_state_dict(expected_weights)
        adapted_expected = LTXJointDenoiser(module, compute_dtype=torch.float32, precision="weights")(_input()).samples
        adapted_actual = adapted(_input()).samples
    for name in ("video", "audio"):
        torch.testing.assert_close(clean_actual[name], clean_expected[name], rtol=0, atol=0)
        torch.testing.assert_close(adapted_actual[name], adapted_expected[name], rtol=0, atol=0)
    assert not torch.equal(clean_actual["video"], adapted_actual["video"])


@pytest.mark.parametrize("strength", [0.0, 0.4, 1.0])
@pytest.mark.parametrize("temporal_scale", [1, 4])
def test_reference_tokens_preserve_target_and_align_scaled_coordinates(strength: float, temporal_scale: int) -> None:
    tools = VideoLatentTools(VideoLatentPatchifier(1), VideoLatentShape(1, 4, 5, 4, 6), fps=24.0)
    state = tools.create_initial_state("cpu", torch.float32)
    encoded = torch.arange(1 * 4 * 2 * 2 * 3).reshape(1, 4, 2, 2, 3).float() / 16
    reference = append_ltx_reference_video(
        state, tools, encoded, strength=strength, downscale_factor=2, temporal_scale_factor=temporal_scale
    )
    count = state.latent.shape[1]
    assert reference.latent.shape[1] == count + 12
    torch.testing.assert_close(reference.latent[:, :count], state.latent, rtol=0, atol=0)
    torch.testing.assert_close(reference.clean_latent[:, count:], tools.patchifier.patchify(encoded), rtol=0, atol=0)
    assert torch.all(reference.denoise_mask[:, count:] == 1 - strength)
    ref_positions = reference.positions[:, :, count:]
    torch.testing.assert_close(ref_positions[:, 1:].amax(dim=(2, 3)), torch.tensor([[128.0, 192.0]]), rtol=0, atol=0)
    first_bounds = (
        torch.tensor([0.0, 1.0]) / (24.0 / temporal_scale) - (temporal_scale - 1) * torch.tensor(1.0 / 24.0)
    ).clamp_min(0)
    torch.testing.assert_close(ref_positions[0, 0, 0], first_bounds, rtol=0, atol=0)
    noisy = add_ltx_noise(reference, scale=1.0, generator=torch.Generator().manual_seed(12))
    if strength == 1.0:
        torch.testing.assert_close(noisy.latent[:, count:], reference.clean_latent[:, count:], rtol=0, atol=0)


@pytest.mark.parametrize("factory", [ltx2_v2v_recipe, ltx23_v2v_recipe])
@pytest.mark.parametrize("skip", [False, True])
def test_ic_lora_recipes_bind_only_stage_one_to_the_adapter(factory: Any, skip: bool) -> None:
    recipe = factory(skip_stage_2=skip)
    roles = {str(component.key): component for component in recipe.components}
    assert roles["denoiser:main"].checkpoints == {"weights": "model", "ic_lora": "ic_lora"}
    assert roles["latent_initializer:main"].checkpoints["ic_lora"] == "ic_lora"
    assert roles["decoder:main"].options["spatial_divisor"] == (2 if skip else 1)
    assert recipe.execution.options["stage_steps"] == ((8,) if skip else (8, 3))
    if skip:
        assert "denoiser:stage-2" not in roles
        assert "processor" not in recipe.execution.bindings
    else:
        assert roles["denoiser:stage-2"].checkpoints == {"weights": "model"}
        assert recipe.execution.bindings["denoiser-2"] == roles["denoiser:stage-2"].key
    assert default_native_diffusion_registry().resolve(recipe.model_id).model_id == recipe.model_id


def test_multistage_strategy_rejects_incomplete_stage_binding() -> None:
    recipe = ltx2_v2v_recipe()
    bindings = dict(recipe.execution.bindings)
    bindings.pop("denoiser-2")
    recipe = replace(recipe, execution=replace(recipe.execution, bindings=bindings))
    with pytest.raises(ValueError, match="denoiser-.*bindings must match"):
        build_joint_multistage_strategy(
            ExecutionBuildContext(recipe=recipe, components={}, policy=RuntimePolicy(), extensions=())
        )


def test_reference_initializer_encodes_only_stage_one_and_reads_adapter_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    class Encoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = torch.nn.Module()
            self.encoder.per_channel_statistics = torch.nn.Module()
            self.encoder.per_channel_statistics.register_buffer("mean-of-means", torch.zeros(128))

        def forward(self, pixels: torch.Tensor) -> torch.Tensor:
            calls.append(tuple(pixels.shape))
            return torch.ones(1, 128, 2, 1, 1)

    def load_reference(path: str, **kwargs: Any) -> torch.Tensor:
        assert path == "pose.mp4"
        assert kwargs["height"] == kwargs["width"] == 32
        assert kwargs["temporal_scale_factor"] == 4
        return torch.zeros(1, 3, 9, 32, 32)

    path = tmp_path / "weights.safetensors"
    save_file({}, str(path), metadata={"reference_downscale_factor": "2", "reference_temporal_scale_factor": "4"})
    monkeypatch.setattr(initializer_module, "load_ltx_video_encoder", lambda context: Encoder())
    monkeypatch.setattr(initializer_module, "load_ltx_reference_video", load_reference)
    initializer = initializer_module.build_ltx_multistage_latent_initializer(
        ComponentBuildContext(
            model_id="ltx-2-v2v",
            key=ComponentKey(ComponentKind.LATENT_INITIALIZER),
            policy=RuntimePolicy(device="cpu", dtype=torch.float32),
            checkpoints={"weights": CheckpointSpec(path), "ic_lora": CheckpointSpec(path)},
            component_options={"split_checkpoint": True},
        )
    )
    request = DiffusionRequest(prompt="A dancer.", height=128, width=128, num_frames=9, inputs={"video": "pose.mp4"})
    states = []
    for stage_index in (0, 1):
        states.append(
            initializer.initialize_stage(
                request,
                stage_index=stage_index,
                previous_latents={},
                noise_scale=1.0,
                generator=torch.Generator().manual_seed(42),
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
        )
    assert calls == [(1, 3, 9, 32, 32)]
    assert states[0]["video"].latent.shape == (1, 10, 128)
    assert states[1]["video"].latent.shape == (1, 32, 128)
    assert torch.all(states[0]["video"].denoise_mask[:, -2:] == 0)
    assert torch.all(states[1]["video"].denoise_mask == 1)


@pytest.mark.parametrize("layout", ["directory", "single-file"])
def test_gemma_projector_loading_uses_resolved_weight_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, layout: str
) -> None:
    from worldfoundry.base_models.diffusion_model.models.encoders.ltx import component as encoder_module

    transformer = tmp_path / "transformer.safetensors"
    save_file({}, str(transformer), metadata={"config": json.dumps({"transformer": {}})})
    gemma_root = tmp_path / "gemma"
    gemma_root.mkdir()
    shards = tuple(gemma_root / f"model-{index}.safetensors" for index in (1, 2))
    for path in shards:
        save_file({}, str(path))
    weight_paths = tuple(str(path) for path in shards) if layout == "directory" else (str(shards[0]),)
    assets = SimpleNamespace(
        config_dict={"model_type": "gemma3" if layout == "directory" else "gemma4_unified"},
        weight_paths=weight_paths,
    )
    gemma = object()
    monkeypatch.setattr(encoder_module, "_load_split_gemma", lambda checkpoint, policy: (gemma, assets))
    loaded: list[CheckpointSpec] = []

    def load(self: Any, spec: Any, checkpoint: CheckpointSpec, policy: RuntimePolicy) -> object:
        loaded.append(checkpoint)
        return object()

    monkeypatch.setattr(encoder_module.NativeModuleLoader, "load", load)
    context = ComponentBuildContext(
        model_id="ltx-2-v2v",
        key=ComponentKey(ComponentKind.CONDITIONER),
        policy=RuntimePolicy(device="cpu", dtype=torch.float32),
        checkpoints={
            "weights": CheckpointSpec(source=transformer),
            "text_encoder": CheckpointSpec(source=gemma_root if layout == "directory" else shards[0]),
        },
    )
    conditioner = encoder_module.build_ltx_gemma_prompt_conditioner(context)
    assert conditioner.gemma is gemma
    assert len(loaded) == 1
    source = loaded[0].source
    assert isinstance(source, tuple)
    assert source == (str(transformer), *weight_paths)
    assert all(Path(path).is_file() for path in source)
