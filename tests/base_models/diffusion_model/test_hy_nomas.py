"""Small real Hy-Nomas forwards, shared-component reuse, and isolated rollout state."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

import pytest
import torch

from worldfoundry.base_models.diffusion_model.contracts import Conditioning, DiffusionRequest, SamplingConfig
from worldfoundry.base_models.diffusion_model.models.autoencoders.pixelumm import PixelUMMDecoder, patchify, unpatchify
from worldfoundry.base_models.diffusion_model.models.autoencoders.pixelumm_component import PixelUMMLatentDecoder
from worldfoundry.base_models.diffusion_model.models.denoisers.hy_nomas import (
    HyNomasDenoiser,
    convert_hy_nomas_checkpoint,
)
from worldfoundry.base_models.diffusion_model.models.encoders.hy_nomas import action_windows
from worldfoundry.base_models.diffusion_model.models.initializers.hy_nomas import HyNomasLatentInitializer
from worldfoundry.base_models.diffusion_model.models.networks.hy_nomas import (
    HyNomasCameraAttention,
    HyNomasTransformer3DModel,
    HyNomasTransformerBlock,
)
from worldfoundry.base_models.diffusion_model.recipes.registry import default_native_diffusion_registry
from worldfoundry.base_models.diffusion_model.runners.base import RunnerComponents
from worldfoundry.base_models.diffusion_model.runners.segmented import SegmentedAutoregressiveRunner
from worldfoundry.base_models.diffusion_model.schedulers.wan import WanFlowMatchEulerScheduler
from worldfoundry.core.attention.cache.kv_quantization import QuantizedKVStore
from worldfoundry.core.nn import DiTModulation, RMSNorm, TransformerMLP
from worldfoundry.operators.hy_nomas_operator import realtime_camera

DIT_CONFIG: dict[str, Any] = dict(
    num_layers=1,
    num_attention_heads=2,
    attention_head_dim=8,
    text_embed_dim=8,
    rope_axes_dim=(2, 2, 4),
    camera_attention_dim=8,
    camera_num_attention_heads=1,
    camera_rope_axes_dim=(2, 2, 4),
)
CODEC_CONFIG: dict[str, Any] = dict(
    hidden_size=16, intermediate_size=32, num_layers=1, num_heads=2, num_kv_heads=1, head_dim=8
)


class TinyConditioner:
    def encode(self, request: DiffusionRequest, *, device: torch.device, dtype: torch.dtype) -> Conditioning:
        prompts = cast(list[str], request.inputs.get("segments") or list(request.prompts))
        return Conditioning(
            positive={
                "segments": [
                    {
                        "encoder_hidden_states": torch.full((1, 3, 8), float(index), device=device, dtype=dtype),
                        "encoder_attention_mask": torch.tensor([[1, 1, 0]], device=device),
                    }
                    for index, _ in enumerate(prompts)
                ]
            },
            negative={"encoder_hidden_states": torch.zeros(1, 3, 8, device=device, dtype=dtype)},
        )


@pytest.fixture
def runner() -> SegmentedAutoregressiveRunner:
    torch.manual_seed(13)
    model = HyNomasTransformer3DModel(**DIT_CONFIG).eval()
    decoder = PixelUMMDecoder(**CODEC_CONFIG).eval()
    return SegmentedAutoregressiveRunner(
        model_id="hy-nomas",
        components=RunnerComponents(
            denoiser=HyNomasDenoiser(model),
            conditioner=TinyConditioner(),
            latent_initializer=HyNomasLatentInitializer(),
            scheduler=WanFlowMatchEulerScheduler(shift=5),
            decoder=PixelUMMLatentDecoder(decoder),
        ),
    )


def request(**inputs: Any) -> DiffusionRequest:
    return DiffusionRequest(
        prompt="A landscape",
        height=32,
        width=32,
        num_frames=8,
        sampling=SamplingConfig(num_inference_steps=2, seed=17),
        inputs=inputs,
    )


def test_recipe_is_lazy_native_component_graph() -> None:
    registry = default_native_diffusion_registry()
    recipe = registry.resolve("hy-nomas")
    assert recipe.execution.strategy == "segment-autoregressive"
    assert len(recipe.components) == 5
    assert registry.resolve("tencent/Hy-Nomas") is recipe
    low = registry.resolve("hy-nomas-low-memory")
    assert all(component.options["low_memory"] for component in low.components)


def test_checkpoint_projection_conversion_retains_exact_tensors() -> None:
    value = torch.randn(4, 4)
    result = convert_hy_nomas_checkpoint({"double_blocks.0.img_attn_q.weight": value})
    assert result["double_blocks.0.attn.img_attn_q.weight"] is value
    with pytest.raises(ValueError, match="duplicate"):
        convert_hy_nomas_checkpoint(
            {"double_blocks.0.img_attn_q.weight": value, "double_blocks.0.attn.img_attn_q.weight": value}
        )


def test_transformer_reuses_canonical_layers(runner: SegmentedAutoregressiveRunner) -> None:
    block = cast(HyNomasTransformerBlock, cast(HyNomasDenoiser, runner.components.denoiser).model.double_blocks[0])
    assert isinstance(block.img_mod, DiTModulation)
    assert isinstance(block.img_mlp, TransformerMLP)
    assert isinstance(block.attn.img_attn_q_norm, RMSNorm)


@pytest.mark.parametrize("guidance", [1.0, 2.0])
def test_real_rollout_ab_a_and_request_isolation(runner: SegmentedAutoregressiveRunner, guidance: float) -> None:
    req = replace(
        request(segments=["A landscape", "A turn"]),
        sampling=SamplingConfig(num_inference_steps=2, seed=17, guidance_scale=guidance),
    )
    first = runner.run(req)
    changed = runner.run(replace(req, sampling=replace(req.sampling, seed=18)))
    repeated = runner.run(req)
    assert first.sample.shape == (1, 3, 8, 32, 32)
    assert torch.isfinite(first.sample).all()
    assert not torch.equal(first.latents, changed.latents)
    torch.testing.assert_close(first.sample, repeated.sample, rtol=0, atol=0)
    assert first.metadata["frame_offset"] == 2


@pytest.mark.parametrize("quantization", ["none", "int8"])
def test_session_continuation_compressed_cache_and_failure_reset(
    runner: SegmentedAutoregressiveRunner,
    quantization: str,
) -> None:
    req = request(kv_cache_quantization=quantization)
    session = runner.new_session(req)
    runner.continue_session(req, session)
    assert isinstance(session.positive[0]["store"], QuantizedKVStore)
    assert session.positive[0]["store"].length == 5
    runner.continue_session(req, session)
    assert session.frame_offset == 4
    assert session.positive[0]["store"].length == 10
    with pytest.raises(ValueError, match="shape"):
        runner.continue_session(replace(req, inputs={"camera_extrinsics": torch.eye(4)}), session)
    assert session.failed and not session.positive and session.decoder is None
    with pytest.raises(RuntimeError, match="reset"):
        runner.continue_session(req, session)
    assert runner.run(req).sample.shape[2] == 8


def test_session_owner_geometry_and_history_bound(runner: SegmentedAutoregressiveRunner) -> None:
    req = request(max_cache_segments=1)
    session = runner.new_session(req)
    runner.continue_session(req, session)
    runner.continue_session(req, session)
    assert session.positive[0]["store"].length == 5
    assert session.frame_offset == 4
    with pytest.raises(ValueError, match="geometry"):
        runner.continue_session(replace(req, width=64), session)
    session.owner = object()
    with pytest.raises(ValueError, match="runner"):
        runner.continue_session(req, session)


def test_camera_condition_changes_real_dit_forward(runner: SegmentedAutoregressiveRunner) -> None:
    model = cast(HyNomasDenoiser, runner.components.denoiser).model
    with torch.no_grad():
        for action in model.action_modules:
            cast(HyNomasCameraAttention, action).proj_keyboard.weight.fill_(0.01)
    sample = torch.randn(1, 32, 1, 1, 1)
    text = torch.randn(1, 3, 8)
    poses = torch.eye(4).repeat(1, 4, 1, 1)
    moved = poses.clone()
    moved[:, :, 0, 3] = 1
    still = model(sample, text, torch.tensor([10.0]), camera_extrinsics=poses, return_dict=False)[0]
    changed = model(sample, text, torch.tensor([10.0]), camera_extrinsics=moved, return_dict=False)[0]
    assert not torch.equal(still, changed)


def test_pixelumm_tubelet_round_trip_and_streaming_prefix() -> None:
    pixels = torch.randn(1, 3, 8, 32, 32)
    torch.testing.assert_close(unpatchify(patchify(pixels), (2, 2, 2)), pixels)
    model = PixelUMMDecoder(**CODEC_CONFIG, temporal_window_latents=2, temporal_overlap_latents=1).eval()
    latents = torch.randn(1, 32, 3, 1, 1)
    with torch.inference_mode():
        reference = cast(tuple[torch.Tensor], model.decode(latents, return_dict=False))[0]
        stream = model.create_streaming_decode_session(retain_history=False)
        chunks = [stream.push(latents[:, :, :2]), stream.push(latents[:, :, 2:]), stream.finish()]
        actual = torch.cat([chunk for chunk in chunks if chunk.numel()], dim=2)
    torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-5)


def test_initializer_pads_720p_and_exact_pixel_length() -> None:
    req = replace(request(), height=720, width=1280, num_frames=7)
    latent = HyNomasLatentInitializer().initialize(
        req, generator=torch.Generator().manual_seed(1), device=torch.device("cpu"), dtype=torch.float32
    )
    assert latent.shape == (1, 32, 2, 23, 40)


def test_camera_reuses_shared_integrator_and_consumes_short_segment() -> None:
    still, _ = realtime_camera([], 4)
    moved, pose = realtime_camera(
        [], 4, segments=[{"duration": 1 / 24, "keys": ["forward"]}, {"duration": 3 / 24, "keys": []}]
    )
    assert not torch.equal(still, moved)
    continued, _ = realtime_camera([], 4, pose=pose)
    torch.testing.assert_close(continued[0, 0], torch.as_tensor(pose))


def test_structured_action_windows_reject_overlap() -> None:
    prompt = '{"segments":[{"contained_actions":[{"start_time":0,"end_time":1},{"start_time":0.5,"end_time":2}]}]}'
    with pytest.raises(ValueError, match="overlap"):
        action_windows(prompt, 24, 24)


def test_real_qwen_backbone_chat_crop_and_negative_sequence(monkeypatch: pytest.MonkeyPatch) -> None:
    from transformers import BatchEncoding
    from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLTextConfig

    from worldfoundry.base_models.diffusion_model.models.encoders.cosmos2p5.component import Cosmos25TextBackbone
    from worldfoundry.base_models.diffusion_model.models.encoders.hy_nomas import HyNomasConditioner

    class Tokenizer:
        def __call__(self, text: str, **kwargs: Any) -> dict[str, list[int]]:
            return {"input_ids": [3, 4]}

        def apply_chat_template(self, messages: Any, **kwargs: Any) -> Any:
            ids = [1, 2, 3, 4, 6, 7]
            length = kwargs.get("max_length", len(ids))
            mask = [1] * min(length, len(ids)) + [0] * max(length - len(ids), 0)
            ids = (ids + [0] * length)[:length]
            return BatchEncoding({"input_ids": torch.tensor([ids]), "attention_mask": torch.tensor([mask])})

    config = Qwen2_5_VLTextConfig(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=2,
        vocab_size=32,
        rope_scaling={"rope_type": "default", "mrope_section": [1, 1, 0]},
    )
    model = Cosmos25TextBackbone(config).eval()
    encoder = HyNomasConditioner(model, Tokenizer(), length=4)
    encoded = encoder.encode_text("A scene", device=torch.device("cpu"), dtype=torch.float32)
    assert encoded["encoder_hidden_states"].shape == (1, 4, 8)
    assert encoded["encoder_attention_mask"].tolist() == [[1, 1, 0, 0]]
    seen = []
    original = encoder.encode_text

    def capture(prompt: str, **kwargs: Any) -> dict[str, Any]:
        seen.append(prompt)
        return original(prompt, **kwargs)

    monkeypatch.setattr(encoder, "encode_text", capture)
    encoder.encode(
        replace(request(), negative_prompt=["blurry"], sampling=SamplingConfig(guidance_scale=2)),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert seen == ["A landscape", "blurry"]


@pytest.mark.parametrize("role", ["denoiser", "decoder"])
def test_low_memory_preset_uses_shared_quantization(monkeypatch: pytest.MonkeyPatch, role: str) -> None:
    from worldfoundry.base_models.diffusion_model.components import ComponentBuildContext, ComponentKey, ComponentKind
    from worldfoundry.base_models.diffusion_model.loaders import CheckpointSpec, NativeModuleLoader
    from worldfoundry.base_models.diffusion_model.models.autoencoders.pixelumm_component import build_pixelumm_decoder
    from worldfoundry.base_models.diffusion_model.models.denoisers.hy_nomas import build_hy_nomas_denoiser
    from worldfoundry.base_models.diffusion_model.optimizations import RuntimePolicy

    policies = []

    def load(self: Any, spec: Any, checkpoint: Any, policy: Any) -> Any:
        policies.append(policy)
        return spec.module_class(**spec.config)

    monkeypatch.setattr(NativeModuleLoader, "load", load)
    context = ComponentBuildContext(
        model_id="hy-nomas-low-memory",
        key=ComponentKey(ComponentKind(role)),
        policy=RuntimePolicy(device="cpu"),
        checkpoints={"weights": CheckpointSpec(source="/unused")},
        component_options={"low_memory": True, "config": DIT_CONFIG if role == "denoiser" else CODEC_CONFIG},
    )
    component = (build_hy_nomas_denoiser if role == "denoiser" else build_pixelumm_decoder)(context)
    assert component.low_memory
    assert policies[0].device == torch.device("cpu")
    assert policies[0].quantization.mode.value == ("int8" if role == "denoiser" else "fp8")
    assert policies[0].quantization.options["keep_dense_fallback"] is False
    if role == "denoiser":
        assert policies[0].quantization.options["group_size"] == 128
        assert policies[0].quantization.options["use_kernel"] is False


def test_failed_cache_commit_preserves_other_session(
    runner: SegmentedAutoregressiveRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    req = request()
    failed, other = runner.new_session(req), runner.new_session(req)
    runner.continue_session(req, other)
    other_key = other.positive[0]["store"].materialize()[0].clone()
    model = cast(HyNomasDenoiser, runner.components.denoiser).model
    original = model.forward
    failure = RuntimeError("cache commit failed")

    def forward(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        if kwargs.get("update_cache") and kwargs.get("kv_cache") is failed.positive:
            raise failure
        return result

    monkeypatch.setattr(model, "forward", forward)
    with pytest.raises(RuntimeError) as caught:
        runner.continue_session(req, failed)
    assert caught.value is failure
    assert failed.failed and not failed.positive and not failed.negative
    torch.testing.assert_close(other.positive[0]["store"].materialize()[0], other_key, rtol=0, atol=0)
    assert not other.failed and other.frame_offset == 2


def test_qwen_factory_resolves_optional_import_before_meta_load(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import json
    from types import SimpleNamespace

    from transformers import AutoTokenizer

    from worldfoundry.base_models.diffusion_model.components import ComponentBuildContext, ComponentKey, ComponentKind
    from worldfoundry.base_models.diffusion_model.loaders import CheckpointSpec
    from worldfoundry.base_models.diffusion_model.models.encoders import hy_nomas as module
    from worldfoundry.base_models.diffusion_model.optimizations import RuntimePolicy

    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen2_5_vl"}))
    monkeypatch.setattr(module.NativeCheckpointResolver, "materialize", lambda *args: SimpleNamespace(root=tmp_path))
    monkeypatch.setattr(module, "_reason1_config", lambda checkpoint: {"config": SimpleNamespace(hidden_size=3584)})
    resolved = []

    def resolve(name: str) -> type:
        assert torch.get_default_device().type == "cpu"
        resolved.append(name)
        return object

    def load(*args: Any, **kwargs: Any) -> Any:
        assert resolved == ["Qwen2_5_VLTextModel"]
        rotary = SimpleNamespace(
            inv_freq=torch.ones(2, dtype=torch.bfloat16),
            config=None,
            rope_init_fn=lambda config, device: (torch.ones(2, device=device, dtype=torch.float32), 1.0),
        )
        return SimpleNamespace(model=SimpleNamespace(rotary_emb=rotary))

    monkeypatch.setattr(module, "_qwen_class", resolve)
    monkeypatch.setattr(module.NativeModuleLoader, "load", load)
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *args, **kwargs: object())
    context = ComponentBuildContext(
        model_id="hy-nomas",
        key=ComponentKey(ComponentKind.CONDITIONER),
        policy=RuntimePolicy(device="cpu"),
        checkpoints={"weights": CheckpointSpec(source=str(tmp_path))},
    )
    encoder = module.build_hy_nomas_conditioner(context)
    assert encoder.model.model.rotary_emb.inv_freq.dtype == torch.float32


def test_text_rotary_restores_fp32_inference_state() -> None:
    from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLTextConfig

    from worldfoundry.base_models.diffusion_model.models.encoders.cosmos2p5.component import Cosmos25TextBackbone
    from worldfoundry.base_models.diffusion_model.models.encoders.hy_nomas import _restore_text_rotary

    config = Qwen2_5_VLTextConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        vocab_size=32,
        rope_scaling={"rope_type": "default", "mrope_section": [4, 6, 6]},
    )
    model = Cosmos25TextBackbone(config).eval()
    expected = model.model.rotary_emb.inv_freq.clone()
    model.to(dtype=torch.bfloat16)
    _restore_text_rotary(model)
    model.to("cpu")
    assert model.model.rotary_emb.inv_freq.dtype == torch.float32
    assert model.model.rotary_emb.original_inv_freq is model.model.rotary_emb.inv_freq
    torch.testing.assert_close(model.model.rotary_emb.inv_freq, expected, rtol=0, atol=0)
