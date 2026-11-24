# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Echo's loaders bind native LTX graphs without changing noise or padding conventions."""

import copy
import importlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors.torch import save_file

from worldfoundry.base_models.diffusion_model.models.networks.ltx.modality import Modality
from worldfoundry.base_models.diffusion_model.models.networks.ltx.rope import LTXRopeType
from worldfoundry.base_models.diffusion_model.models.networks.ltx.transformer import BasicAVTransformerBlock
from worldfoundry.base_models.diffusion_model.models.networks.ltx.variants.joyai_echo.model import (
    EchoCrossNoisePreprocessor,
    JoyAIEchoModel,
)
from worldfoundry.synthesis.visual_generation.joyai_echo.ltx_attention_policy import (
    AttentionFunction,
    build_attention_ops,
)

pytestmark = pytest.mark.unit

_ROOT = "worldfoundry.synthesis.visual_generation.joyai_echo.echo_longvideo_runtime.ltx_core"


def _echo_config(**updates: Any) -> dict[str, Any]:
    transformer = {
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
        "attention_type": "pytorch",
        "connector_num_attention_heads": 2,
        "connector_attention_head_dim": 8,
        "connector_num_layers": 1,
        "audio_connector_num_attention_heads": 2,
        "audio_connector_attention_head_dim": 4,
    }
    transformer.update(updates)
    return {"transformer": transformer}


def _modalities() -> tuple[Modality, Modality]:
    grid = torch.tensor([[0, h, w] for h in range(2) for w in range(2)]).T
    video = Modality(
        latent=torch.randn(1, 4, 4),
        sigma=torch.tensor([0.5]),
        timesteps=torch.tensor([[[0.0], [0.5], [0.5], [0.5]]]),
        positions=torch.stack([grid, grid + 1], dim=-1)[None].float(),
        context=torch.randn(1, 3, 16),
    )
    audio = Modality(
        latent=torch.randn(1, 3, 4),
        sigma=torch.tensor([0.75]),
        timesteps=torch.tensor([[[0.0], [0.75], [0.75]]]),
        positions=torch.stack([torch.arange(3), torch.arange(3) + 1], dim=-1)[None, None].float(),
        context=torch.randn(1, 3, 8),
    )
    return video, audio


@pytest.mark.parametrize("with_audio", [False, True])
def test_echo_longvideo_checkpoint_loads_native_ltx_graph(tmp_path: Path, with_audio: bool) -> None:
    configs = importlib.import_module(f"{_ROOT}.model.transformer.model_configurator")
    builder = importlib.import_module(f"{_ROOT}.loader.single_gpu_model_builder").SingleGPUModelBuilder
    config = _echo_config()
    original = copy.deepcopy(config)
    factory = configs.LTXModelConfigurator if with_audio else configs.LTXVideoOnlyModelConfigurator
    reference = factory.from_config(config).eval()
    assert config == original
    assert type(reference) is JoyAIEchoModel
    assert reference.rope_type is LTXRopeType.INTERLEAVED
    assert reference.use_middle_indices_grid
    block = reference.transformer_blocks[0]
    assert type(block) is BasicAVTransformerBlock
    assert block.ada_zero_function.backend == "torch"
    assert block.attn1.attention_function is AttentionFunction.PYTORCH
    assert block.attn1.masked_attention_function is AttentionFunction.PYTORCH
    assert reference.prompt_adaln_single is not None
    assert reference.keyframes_abs_pos_embedding is None
    with torch.no_grad():
        for parameter in reference.parameters():
            parameter.uniform_(-0.05, 0.05)
    weights = tmp_path / "transformer.safetensors"
    save_file(
        {f"model.diffusion_model.{key}": value for key, value in reference.state_dict().items()},
        weights,
        metadata={"config": json.dumps(config)},
    )
    loaded = builder(
        model_path=str(weights),
        model_class_configurator=factory,
        model_sd_ops=configs.LTXV_MODEL_COMFY_RENAMING_MAP,
    ).build(device=torch.device("cpu"), dtype=torch.float32)
    assert type(loaded) is JoyAIEchoModel
    torch.testing.assert_close(loaded.state_dict(), reference.state_dict(), rtol=0, atol=0)
    video, audio = _modalities()
    with torch.no_grad():
        torch.testing.assert_close(
            loaded(video, audio if with_audio else None, None),
            reference(video, audio if with_audio else None, None),
            rtol=0,
            atol=0,
        )


def test_echo_longvideo_prequant_fp8_loader_matches_native_linear_paths(tmp_path: Path) -> None:
    configs = importlib.import_module(f"{_ROOT}.model.transformer.model_configurator")
    builder = importlib.import_module(f"{_ROOT}.loader.single_gpu_model_builder").SingleGPUModelBuilder
    quant = importlib.import_module(f"{_ROOT.removesuffix('.ltx_core')}.ltx_distillation.quantization")
    config = _echo_config()
    reference = configs.LTXModelConfigurator.from_config(config).to(dtype=torch.bfloat16)
    with torch.no_grad():
        for parameter in reference.parameters():
            parameter.uniform_(-0.05, 0.05)
    state = {key: value.detach().clone() for key, value in reference.state_dict().items()}
    name = "transformer_blocks.0.attn1.to_q"
    state[f"{name}.weight"] = state[f"{name}.weight"].to(torch.float8_e4m3fn)
    state[f"{name}.weight_scale"] = torch.tensor(0.125, dtype=torch.float32)
    weights = tmp_path / "fp8.safetensors"
    save_file(
        {f"model.diffusion_model.{key}": value for key, value in state.items()},
        weights,
        metadata={"config": json.dumps(config)},
    )
    info = quant.inspect_prequant_fp8_checkpoint(weights)
    assert info.module_names == (name,)
    policy = quant.build_prequant_fp8_policy(info)
    loaded = builder(
        model_path=str(weights),
        model_class_configurator=configs.LTXModelConfigurator,
        model_sd_ops=configs.LTXV_MODEL_COMFY_RENAMING_MAP,
        module_ops=policy.module_ops,
    ).build(device=torch.device("cpu"))
    assert isinstance(loaded.get_submodule(name), quant.PrequantFP8ScaledMMLinear)
    actual = loaded.state_dict()
    assert actual.keys() == state.keys()
    for key, expected in state.items():
        assert actual[key].dtype == expected.dtype
        torch.testing.assert_close(actual[key].float(), expected.float(), rtol=0, atol=0)


def test_echo_cross_noise_preprocessor_uses_cross_sigma_for_both_embeddings() -> None:
    configs = importlib.import_module(f"{_ROOT}.model.transformer.model_configurator")
    model = configs.LTXModelConfigurator.from_config(_echo_config())
    model.refresh_preprocessor_bindings()
    assert isinstance(model.video_args_preprocessor, EchoCrossNoisePreprocessor)
    assert model.video_args_preprocessor.simple_preprocessor.patchify_proj is model.patchify_proj
    video, audio = _modalities()
    with torch.no_grad():
        prepared = model.video_args_preprocessor.prepare(video, audio)
        scale, _ = model.av_ca_video_scale_shift_adaln_single(audio.sigma * 1000, hidden_dtype=video.latent.dtype)
        gate, _ = model.av_ca_a2v_gate_adaln_single(audio.sigma * 1000 * (1 / 1000), hidden_dtype=video.latent.dtype)
    torch.testing.assert_close(prepared.cross_scale_shift_timestep, scale[:, None], rtol=0, atol=0)
    torch.testing.assert_close(prepared.cross_gate_timestep, gate[:, None], rtol=0, atol=0)


def test_echo_attention_policy_preserves_live_sdpa_context(monkeypatch: pytest.MonkeyPatch) -> None:
    from torch.nn.attention import SDPBackend, sdpa_kernel

    import worldfoundry.synthesis.visual_generation.joyai_echo.ltx_attention_policy as policy

    original = torch.nn.functional.scaled_dot_product_attention
    observations = []

    def record(*args: Any, **kwargs: Any) -> torch.Tensor:
        observations.append((torch.backends.cuda.flash_sdp_enabled(), torch.backends.cuda.math_sdp_enabled()))
        return original(*args, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", record)
    ops = build_attention_ops(AttentionFunction.PYTORCH)
    q, k, v = (torch.randn(1, 3, 8) for _ in range(3))
    with sdpa_kernel(SDPBackend.MATH):
        result = ops.attention_function(q, k, v, 2)
        q_heads, k_heads, v_heads = (t.view(1, 3, 2, 4).transpose(1, 2) for t in (q, k, v))
        expected = original(q_heads, k_heads, v_heads)
    torch.testing.assert_close(result, expected.transpose(1, 2).reshape(1, 3, 8), rtol=0, atol=0)
    assert observations == [(False, True)]
    monkeypatch.setattr(policy, "_xformers_available", lambda: True)
    defaults = build_attention_ops()
    assert defaults.attention_function is AttentionFunction.XFORMERS
    assert defaults.masked_attention_function is AttentionFunction.XFORMERS
    with pytest.raises(RuntimeError, match="FlashAttention3"):
        AttentionFunction.FLASH_ATTENTION_3(q, k, v, 2)


def test_echo_connector_preserves_compaction_and_register_free_token_order() -> None:
    from worldfoundry.base_models.diffusion_model.models.encoders.ltx.embeddings_processor import (
        convert_to_additive_mask,
    )
    from worldfoundry.base_models.diffusion_model.models.encoders.ltx.variants.joyai_echo import (
        EchoEmbeddings1DConnector,
        EchoEmbeddingsProcessor,
    )

    configs = importlib.import_module(f"{_ROOT}.text_encoders.gemma.embeddings_connector")
    connector = configs.Embeddings1DConnectorConfigurator.from_config(_echo_config(connector_num_layers=0))
    assert type(connector) is EchoEmbeddings1DConnector
    assert connector.rope_type is LTXRopeType.INTERLEAVED
    processor = EchoEmbeddingsProcessor(video_connector=connector)
    features = torch.randn(2, 128, 16)
    counts = torch.tensor([2, 3])
    mask = convert_to_additive_mask(torch.arange(128)[None] >= (128 - counts[:, None]), torch.float32)
    expected = connector.learnable_registers.detach().float()[None].expand(2, -1, -1).clone()
    for index, count in enumerate(counts.tolist()):
        expected[index, :count] = features[index, -count:]
    with torch.no_grad():
        encoded, audio, binary = processor.create_embeddings(features, None, mask)
    assert audio is None
    torch.testing.assert_close(encoded, torch.nn.functional.rms_norm(expected, (16,), eps=1e-6), rtol=0, atol=0)
    assert torch.all(binary == 1)
    connector.num_learnable_registers = None
    with torch.no_grad():
        encoded, _, binary = processor.create_embeddings(features, None, mask)
    expected = torch.nn.functional.rms_norm(features, (16,), eps=1e-6)
    torch.testing.assert_close(encoded, expected, rtol=0, atol=0)
    assert torch.all(binary == 1)


_WM_ROOT = "worldfoundry.synthesis.visual_generation.joyai_echo.echo_wm_runtime"


@pytest.mark.parametrize("with_action", [False, True])
def test_echo_wm_ledger_loads_native_ltx_graph_and_ucpe(tmp_path: Path, with_action: bool) -> None:
    from worldfoundry.base_models.diffusion_model.models.networks.ltx.attention import Attention
    from worldfoundry.base_models.diffusion_model.models.networks.ltx.variants.joyai_echo.transformer import (
        ActionBlockConfig,
        EchoWMTransformerBlock,
    )
    from worldfoundry.base_models.diffusion_model.models.networks.ltx.variants.joyai_echo.transformer_args import (
        EchoWMArgsPreprocessor,
        EchoWMAVArgsPreprocessor,
    )
    from worldfoundry.base_models.diffusion_model.models.networks.ltx.variants.joyai_echo.wm import EchoWMModel

    configs = importlib.import_module(f"{_WM_ROOT}.ltx_core.model.transformer.model_configurator")
    ledger_type = importlib.import_module(f"{_WM_ROOT}.ltx_pipelines.utils.model_ledger").ModelLedger
    config = _echo_config()
    original_config = copy.deepcopy(config)
    reference = configs.LTXModelConfigurator.from_config(config).eval()
    action = None
    if with_action:
        action = ActionBlockConfig(
            block_indices=[0],
            ucpe_attn_dim=16,
            ucpe_num_heads=1,
            ucpe_patches_x=2,
            ucpe_patches_y=2,
            ucpe_image_width=64,
            ucpe_image_height=64,
        )
        action.enabled = True
        reference.enable_action_conditioning(action)
    with torch.no_grad():
        for parameter in reference.parameters():
            parameter.uniform_(-0.05, 0.05)
    weights = tmp_path / "wm.safetensors"
    save_file(
        {f"model.diffusion_model.{key}": value for key, value in reference.state_dict().items()},
        weights,
        metadata={"config": json.dumps(config)},
    )
    loaded = ledger_type(dtype=torch.float32, device=torch.device("cpu"), checkpoint_path=str(weights)).transformer(
        action_config=action
    )
    model = loaded.velocity_model
    assert type(model) is EchoWMModel
    assert config == original_config
    assert type(model.transformer_blocks[0]) is EchoWMTransformerBlock
    assert isinstance(model.transformer_blocks[0], BasicAVTransformerBlock)
    assert isinstance(model.transformer_blocks[0].attn1, Attention)
    assert model.transformer_blocks[0].action_ucpe_enabled is with_action
    assert model.rope_type is LTXRopeType.INTERLEAVED
    model.refresh_preprocessor_bindings()
    assert type(model.video_args_preprocessor) is EchoWMAVArgsPreprocessor
    assert type(model.video_args_preprocessor.simple_preprocessor) is EchoWMArgsPreprocessor
    assert model.video_args_preprocessor.simple_preprocessor.patchify_proj is model.patchify_proj
    torch.testing.assert_close(model.state_dict(), reference.state_dict(), rtol=0, atol=0)
    video, audio = _modalities()
    condition = (
        {"ucpe_viewmats": torch.eye(4)[None, None], "ucpe_Ks": torch.eye(3)[None, None]} if with_action else None
    )
    with torch.no_grad():
        expected = reference(video, audio, None, action_cond=condition)
        torch.testing.assert_close(model(video, audio, None, action_cond=condition), expected, rtol=0, atol=0)
        expected_x0 = tuple(
            modality.latent - value * modality.timesteps
            for modality, value in zip((video, audio), expected, strict=True)
        )
        torch.testing.assert_close(loaded(video, audio, None, action_cond=condition), expected_x0, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_echo_wm_causal_rollout_keeps_bounded_caches_request_local(
    dtype: torch.dtype, monkeypatch: pytest.MonkeyPatch
) -> None:
    from worldfoundry.base_models.diffusion_model.models.networks.ltx.variants.joyai_echo.transformer import (
        ActionBlockConfig,
    )

    configs = importlib.import_module(f"{_WM_ROOT}.ltx_core.model.transformer.model_configurator")
    caches_module = importlib.import_module(f"{_WM_ROOT}.ltx_causal.cache")
    wrapper_module = importlib.import_module(f"{_WM_ROOT}.ltx_causal.causal_wrapper")
    rollout = importlib.import_module(f"{_WM_ROOT}.ltx_causal.rollout").causal_rollout
    model = configs.LTXModelConfigurator.from_config(_echo_config()).eval()
    model.enable_action_conditioning(
        ActionBlockConfig(
            enabled=True,
            block_indices=[0],
            ucpe_attn_dim=16,
            ucpe_num_heads=1,
            ucpe_patches_x=1,
            ucpe_patches_y=1,
            ucpe_image_width=32,
            ucpe_image_height=32,
        )
    )
    model.to(dtype=dtype)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.uniform_(-0.05, 0.05)
    wrapper = wrapper_module.CausalModelWrapper(
        model,
        patches_per_frame=1,
        cache=caches_module.CausalCacheConfig(video_local_attn_size=4, video_sink_size=1),
    )
    allocate = wrapper.init_caches
    captured: list[list[dict[str, Any]]] = []

    def record_caches(**kwargs: Any) -> list[dict[str, Any]]:
        caches = allocate(**kwargs)
        captured.append(caches)
        return caches

    monkeypatch.setattr(wrapper, "init_caches", record_caches)
    frames, audio_frames = 7, 52
    grid = torch.stack([torch.arange(frames), torch.zeros(frames), torch.zeros(frames)])
    viewmats = torch.eye(4)[None, None].repeat(1, frames, 1, 1)
    viewmats[:, :, 0, 3] = torch.arange(frames) * 0.5
    kwargs: dict[str, Any] = {
        "wrapper": wrapper,
        "clean_video": torch.zeros(1, frames, 4, dtype=dtype),
        "clean_audio": torch.zeros(1, audio_frames, 4, dtype=dtype),
        "video_positions": torch.stack([grid, grid + 1], dim=-1)[None],
        "audio_positions": torch.stack([torch.arange(audio_frames), torch.arange(audio_frames) + 1], dim=-1)[
            None, None
        ].float(),
        "video_context": torch.randn(1, 3, 16, dtype=dtype),
        "audio_context": torch.randn(1, 3, 8, dtype=dtype),
        "context_mask": None,
        "action_cond": {"ucpe_viewmats": viewmats, "ucpe_Ks": torch.eye(3)[None, None].repeat(1, frames, 1, 1)},
        "timesteps": (1000,),
    }
    first = tuple(t.clone() for t in rollout(**kwargs, seed=17))
    first_keys = captured[0][0]["video_self"]["k"].clone()
    second = rollout(**kwargs, seed=29)
    repeated = rollout(**kwargs, seed=17)
    assert not torch.equal(first[0], second[0])
    torch.testing.assert_close(first, repeated, rtol=0, atol=0)
    torch.testing.assert_close(first_keys, captured[0][0]["video_self"]["k"], rtol=0, atol=0)
    for layer in (request[0] for request in captured):
        torch.testing.assert_close(layer["video_self"]["positions"], torch.tensor([0, 4, 5, 6]))
        torch.testing.assert_close(layer["audio_self"]["positions"], torch.tensor([0, 1, *range(27, 52)]))
        assert layer["video_text"]["is_init"] and layer["audio_text"]["is_init"]
        assert layer["video_ucpe"]["bounded_anchor_translation"]
    assert len({request[0]["video_self"]["k"].data_ptr() for request in captured}) == 3


def test_echo_wm_v2a_gradient_scale_preserves_forward_values() -> None:
    from worldfoundry.base_models.diffusion_model.models.representations.ltx.echo_wm import EchoWMModality

    configs = importlib.import_module(f"{_WM_ROOT}.ltx_core.model.transformer.model_configurator")
    model = configs.LTXModelConfigurator.from_config(_echo_config()).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.uniform_(-0.05, 0.05)
    video, audio = _modalities()
    outputs, gradients = [], []
    for scale in (1.0, 0.25, 0.0):
        vi = replace(EchoWMModality(**vars(video)), latent=video.latent.detach().clone().requires_grad_(True))
        au = replace(EchoWMModality(**vars(audio)), v2a_grad_scale=scale)
        _, value = model(vi, au, None)
        value.square().mean().backward()
        outputs.append(value.detach().clone())
        gradients.append(vi.latent.grad)
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
    torch.testing.assert_close(outputs[0], outputs[2], rtol=0, atol=0)
    assert gradients[0] is not None and gradients[1] is not None
    torch.testing.assert_close(gradients[0] * 0.25, gradients[1], rtol=0, atol=0)
    assert gradients[2] is not None
    torch.testing.assert_close(gradients[2], torch.zeros_like(gradients[0]), rtol=0, atol=0)


@pytest.mark.parametrize(
    ("valid_indices", "copied_tokens", "zero_indices"),
    [
        ((126, 127), ((0, 126), (1, 127)), ()),
        ((0, 1), (), (126, 127)),
        ((1, 127), ((0, 1),), (126,)),
    ],
    ids=["left", "right", "gapped"],
)
def test_echo_wm_connector_preserves_boolean_selection_and_flipped_mask(
    valid_indices: tuple[int, ...], copied_tokens: tuple[tuple[int, int], ...], zero_indices: tuple[int, ...]
) -> None:
    from worldfoundry.base_models.diffusion_model.models.encoders.ltx.embeddings_processor import (
        convert_to_additive_mask,
    )
    from worldfoundry.base_models.diffusion_model.models.encoders.ltx.variants.echo_wm import (
        EchoWMEmbeddings1DConnector,
    )

    configs = importlib.import_module(f"{_WM_ROOT}.ltx_core.text_encoders.gemma.embeddings_connector")
    connector = configs.Embeddings1DConnectorConfigurator.from_config(_echo_config(connector_num_layers=0))
    assert type(connector) is EchoWMEmbeddings1DConnector
    features = torch.randn(1, 128, 16)
    mask = torch.zeros(1, 128, dtype=torch.bool)
    mask[0, list(valid_indices)] = True
    expected = connector.learnable_registers.detach().float()[None].clone()
    for destination, source in copied_tokens:
        expected[:, destination] = features[:, source]
    expected[:, list(zero_indices)] = 0
    with torch.no_grad():
        encoded, actual_mask = connector(features, convert_to_additive_mask(mask, torch.float32))
    torch.testing.assert_close(encoded, torch.nn.functional.rms_norm(expected, (16,), eps=1e-6), rtol=0, atol=0)
    assert torch.all(actual_mask == 0)
