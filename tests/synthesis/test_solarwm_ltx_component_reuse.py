# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""SolarWM loads and executes native LTX graphs with its own runtime policy."""

from pathlib import Path
from typing import Any

import pytest
import torch
import torch.utils._pytree as pytree

pytestmark = pytest.mark.unit


def _solar_ltx_metadata(**updates: Any) -> dict[str, Any]:
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
        "ff_bias": False,
        "audio_ff_bias": False,
        "use_prompt_adaln_single": False,
        "use_keyframes_abs_pos_embedding": True,
        "connector_num_attention_heads": 2,
        "connector_attention_head_dim": 8,
        "connector_num_layers": 1,
        "audio_connector_num_attention_heads": 2,
        "audio_connector_attention_head_dim": 4,
        "connector_ff_bias": False,
    }
    transformer.update(updates)
    return {"config": {"transformer": transformer}}


@pytest.mark.parametrize("with_audio", [False, True])
def test_solarwm_ltx_checkpoint_loads_the_shared_graph(tmp_path: Path, with_audio: bool) -> None:
    import copy
    import json

    from safetensors.torch import save_file

    from worldfoundry.base_models.diffusion_model.models.networks.ltx.modality import Modality
    from worldfoundry.base_models.diffusion_model.models.networks.ltx.rope import LTXRopeType
    from worldfoundry.base_models.diffusion_model.models.networks.ltx.transformer import BasicAVTransformerBlock
    from worldfoundry.base_models.diffusion_model.models.networks.ltx.variants.solarwm import SolarWMModel
    from worldfoundry.synthesis.visual_generation.solarwm.solarwm_runtime.ltx_core.loader.attention_policy import (
        automatic_attention,
        automatic_masked_attention,
    )
    from worldfoundry.synthesis.visual_generation.solarwm.solarwm_runtime.ltx_core.loader.single_gpu_model_builder import (
        SingleGPUModelBuilder,
    )
    from worldfoundry.synthesis.visual_generation.solarwm.solarwm_runtime.ltx_core.model.transformer import (
        model_configurator,
    )

    metadata = _solar_ltx_metadata()
    original_metadata = copy.deepcopy(metadata)
    factory = (
        model_configurator.LTXModelConfigurator if with_audio else model_configurator.LTXVideoOnlyModelConfigurator
    )
    reference = factory.from_metadata(metadata).eval()
    assert metadata == original_metadata
    assert type(reference) is SolarWMModel
    assert reference.rope_type is LTXRopeType.SPLIT
    assert reference.use_middle_indices_grid
    assert reference.supports_keyframes_abs_pos_embedding
    assert reference.prompt_adaln_single is None
    block = reference.transformer_blocks[0]
    assert type(block) is BasicAVTransformerBlock
    assert block.ada_zero_function.backend == "torch"
    assert block.attn1.attention_function is automatic_attention()
    assert block.attn1.masked_attention_function is automatic_masked_attention()
    assert all(m.bias is None for m in block.ff.modules() if isinstance(m, torch.nn.Linear))
    with torch.no_grad():
        for parameter in reference.parameters():
            parameter.uniform_(-0.05, 0.05)
    state = {f"model.diffusion_model.{key}": value for key, value in reference.state_dict().items()}
    weights = tmp_path / "transformer.safetensors"
    save_file(state, weights, metadata={"config": json.dumps(metadata["config"])})
    loaded = SingleGPUModelBuilder(
        model_path=str(weights),
        model_class_configurator=factory,
        model_sd_ops=model_configurator.LTXV_MODEL_COMFY_RENAMING_MAP,
    ).build(device=torch.device("cpu"), dtype=torch.float32)
    assert type(loaded) is SolarWMModel
    torch.testing.assert_close(loaded.state_dict(), reference.state_dict(), rtol=0, atol=0)
    keyframes = loaded.keyframes_abs_pos_embedding
    loaded.enable_keyframes_abs_pos_embedding()
    assert loaded.keyframes_abs_pos_embedding is keyframes
    grid = torch.tensor([[0, h, w] for h in range(2) for w in range(2)]).T
    video = Modality(
        latent=torch.randn(1, 4, 4),
        sigma=torch.tensor([0.5]),
        timesteps=torch.full((1, 4, 1), 0.5),
        positions=torch.stack([grid, grid + 1], dim=-1)[None].float(),
        context=torch.randn(1, 3, 16),
        keyframes_mask=torch.tensor([[[1.0], [0.0], [0.0], [0.0]]]),
    )
    audio = (
        Modality(
            latent=torch.randn(1, 3, 4),
            sigma=torch.tensor([0.5]),
            timesteps=torch.full((1, 3, 1), 0.5),
            positions=torch.stack([torch.arange(3), torch.arange(3) + 1], dim=-1)[None, None].float(),
            context=torch.randn(1, 3, 8),
        )
        if with_audio
        else None
    )
    with torch.no_grad():
        actual = loaded(video, audio, None)
        expected = reference(video, audio, None)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("required", ["cross_attention_norm", "use_middle_indices_grid"])
def test_solarwm_video_configurator_retains_required_ltx2_fields(required: str) -> None:
    from worldfoundry.synthesis.visual_generation.solarwm.solarwm_runtime.ltx_core.model.transformer.model_configurator import (
        LTXVideoOnlyModelConfigurator,
    )

    metadata = _solar_ltx_metadata()
    del metadata["config"]["transformer"][required]
    with pytest.raises(ValueError, match=required):
        LTXVideoOnlyModelConfigurator.from_metadata(metadata)


def test_solarwm_keyframe_activation_materializes_only_missing_embeddings() -> None:
    from worldfoundry.synthesis.visual_generation.solarwm.solarwm_runtime.ltx_core.model.transformer.model_configurator import (
        LTXVideoOnlyModelConfigurator,
    )

    model = LTXVideoOnlyModelConfigurator.from_metadata(_solar_ltx_metadata(use_keyframes_abs_pos_embedding=False))
    assert not model.supports_keyframes_abs_pos_embedding
    model.enable_keyframes_abs_pos_embedding()
    embedding = model.keyframes_abs_pos_embedding
    assert model.supports_keyframes_abs_pos_embedding
    assert torch.count_nonzero(embedding) == 0
    with torch.no_grad():
        embedding.fill_(0.25)
    model.enable_keyframes_abs_pos_embedding()
    assert model.keyframes_abs_pos_embedding is embedding
    torch.testing.assert_close(embedding, torch.full_like(embedding, 0.25), rtol=0, atol=0)
    model.keyframes_abs_pos_embedding = torch.nn.Parameter(torch.empty_like(embedding, device="meta"))
    assert not model.supports_keyframes_abs_pos_embedding
    model.enable_keyframes_abs_pos_embedding()
    assert model.supports_keyframes_abs_pos_embedding
    assert torch.count_nonzero(model.keyframes_abs_pos_embedding) == 0


def test_solarwm_connectors_share_the_graph_and_keep_loader_attention_policy() -> None:
    from worldfoundry.base_models.diffusion_model.models.encoders.ltx.embeddings_connector import (
        Embeddings1DConnector,
    )
    from worldfoundry.base_models.diffusion_model.models.encoders.ltx.embeddings_processor import (
        EmbeddingsProcessor,
        convert_to_additive_mask,
    )
    from worldfoundry.synthesis.visual_generation.solarwm.solarwm_runtime.ltx_core.loader.attention_policy import (
        automatic_attention,
    )
    from worldfoundry.synthesis.visual_generation.solarwm.solarwm_runtime.ltx_core.text_encoders.gemma.embeddings_connector import (
        AudioEmbeddings1DConnectorConfigurator,
        Embeddings1DConnectorConfigurator,
    )

    metadata = _solar_ltx_metadata()
    video = Embeddings1DConnectorConfigurator.from_metadata(metadata)
    audio = AudioEmbeddings1DConnectorConfigurator.from_metadata(metadata)
    for connector in (video, audio):
        assert type(connector) is Embeddings1DConnector
        block = connector.transformer_1d_blocks[0]
        assert block.attn1.attention_function is automatic_attention()
        assert all(m.bias is None for m in block.ff.modules() if isinstance(m, torch.nn.Linear))
    processor = EmbeddingsProcessor(video_connector=video, audio_connector=audio).eval()
    video_features = torch.randn(1, 128, 16)
    audio_features = torch.randn(1, 128, 8)
    mask = convert_to_additive_mask(torch.arange(128)[None] >= 64, torch.float32)
    with torch.no_grad():
        encoded_video, encoded_audio, encoded_mask = processor.create_embeddings(video_features, audio_features, mask)
    assert encoded_video.shape == video_features.shape
    assert encoded_audio is not None
    assert encoded_audio.shape == audio_features.shape
    assert encoded_mask.shape == (1, 128)


def test_solarwm_camera_attention_accepts_native_control_and_cache_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    from worldfoundry.base_models.diffusion_model.models.networks.ltx.attention import Attention
    from worldfoundry.synthesis.visual_generation.solarwm.solarwm_runtime.ltx_core.loader.transformer_ops import (
        build_attention_ops,
    )
    from worldfoundry.synthesis.visual_generation.solarwm.solarwm_runtime.solarwm.backends.ltx25 import (
        torch_distributed,
        torch_model,
    )

    monkeypatch.setattr(torch_distributed, "_STATE", None)
    torch_distributed.initialize({"distributed": {"sequence_parallel_size": 1, "world_size": 1, "local_world_size": 1}})
    attention = torch_model.SolarLTX25VideoSelfAttention(query_dim=32, heads=2, dim_head=16, ops=build_attention_ops())
    attention.camera_translation_transform = "linear"
    features = torch.randn(1, 4, 32)
    with torch.no_grad():
        actual = attention(features, camera_control=None, cache=None)
        expected = Attention.forward(attention, features)
        cameras = torch.eye(4)[None, None].expand(1, 4, -1, -1)
        intrinsics = torch.eye(3)[None, None].expand(1, 4, -1, -1)
        transformed = attention(features, camera_viewmats=cameras, camera_k=intrinsics, camera_control=None, cache=None)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert transformed.shape == actual.shape
    assert torch.isfinite(transformed).all()


@pytest.mark.skipif(
    not hasattr(pytree, "register_dataclass"),
    reason="Solar CUDA-graph compilation requires Torch's dataclass pytree API",
)
def test_solarwm_native_compilation_retains_per_capture_buffer_ownership() -> None:
    from worldfoundry.base_models.diffusion_model.optimizations.ltx.compile import (
        CompilationConfig,
        CompiledBlockPerturbationsProcessor,
    )
    from worldfoundry.synthesis.visual_generation.solarwm.solarwm_runtime.ltx_core.model.transformer.compiling import (
        build_compile_transformer_op,
    )
    from worldfoundry.synthesis.visual_generation.solarwm.solarwm_runtime.ltx_core.model.transformer.cudagraph_capture import (
        CudaGraphRunner,
    )
    from worldfoundry.synthesis.visual_generation.solarwm.solarwm_runtime.ltx_core.model.transformer.model_configurator import (
        LTXModelConfigurator,
    )

    model = LTXModelConfigurator.from_metadata(_solar_ltx_metadata())
    config = CompilationConfig(backend="eager", capture=True)
    op = build_compile_transformer_op(config)
    assert op.matcher(model)
    assert op.mutator(model) is model
    assert type(model._process_transformer_blocks) is CudaGraphRunner
    assert type(model.block_input_processor) is CompiledBlockPerturbationsProcessor
    assert model._process_transformer_blocks._captures == {}
    assert all(hasattr(block, "_orig_mod") for block in model.transformer_blocks)
    assert config.max_video_tokens == config.max_audio_tokens == 0
