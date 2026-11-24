"""CUDA-only native Hy-Nomas INT8/FP8 execution contracts; no released weights."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors.torch import save_file

from worldfoundry.base_models.diffusion_model import NativeDiffusionPipeline
from worldfoundry.base_models.diffusion_model.components import ComponentBuildContext, ComponentKey, ComponentKind
from worldfoundry.base_models.diffusion_model.contracts import Conditioning, DenoiserInput
from worldfoundry.base_models.diffusion_model.loaders import CheckpointSpec
from worldfoundry.base_models.diffusion_model.models.autoencoders.pixelumm import PixelUMMDecoder
from worldfoundry.base_models.diffusion_model.models.autoencoders.pixelumm_component import build_pixelumm_decoder
from worldfoundry.base_models.diffusion_model.models.denoisers.hy_nomas import build_hy_nomas_denoiser
from worldfoundry.base_models.diffusion_model.models.networks.hy_nomas import HyNomasTransformer3DModel
from worldfoundry.base_models.diffusion_model.optimizations import OffloadPolicy, RuntimePolicy
from worldfoundry.base_models.diffusion_model.recipes.hy_nomas import hy_nomas_low_memory_recipe, hy_nomas_recipe
from worldfoundry.base_models.diffusion_model.recipes.registry import NativeDiffusionRegistry
from worldfoundry.core.acceleration.quantization.linear import Float8Linear, WeightOnlyLinear
from worldfoundry.pipelines.hy_nomas.pipeline_hy_nomas import HyNomasPipeline

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


@torch.inference_mode()
def test_native_hy_nomas_quantized_components_execute(tmp_path: Path) -> None:
    if torch.cuda.get_device_capability() < (8, 9):
        pytest.skip("native FP8 execution requires SM89 or newer")
    dit: dict[str, Any] = dict(
        num_layers=1,
        num_attention_heads=8,
        attention_head_dim=128,
        text_embed_dim=128,
        rope_axes_dim=(16, 56, 56),
        camera_attention_dim=128,
        camera_num_attention_heads=1,
        camera_rope_axes_dim=(16, 56, 56),
    )
    codec: dict[str, Any] = dict(
        hidden_size=1024,
        intermediate_size=3072,
        num_layers=1,
        num_heads=8,
        num_kv_heads=2,
        head_dim=128,
        temporal_window_latents=2,
        temporal_overlap_latents=1,
    )
    save_file(HyNomasTransformer3DModel(**dit).state_dict(), tmp_path / "dit.safetensors")
    save_file(PixelUMMDecoder(**codec).state_dict(), tmp_path / "decoder.safetensors")
    policy = RuntimePolicy(device="cuda", dtype=torch.bfloat16)

    def context(kind: ComponentKind, path: str, config: dict[str, Any]) -> ComponentBuildContext:
        return ComponentBuildContext(
            model_id="hy-nomas-low-memory",
            key=ComponentKey(kind),
            policy=policy,
            checkpoints={"weights": CheckpointSpec(source=str(tmp_path / path))},
            component_options={"low_memory": True, "config": config},
        )

    denoiser = build_hy_nomas_denoiser(context(ComponentKind.DENOISER, "dit.safetensors", dit))
    denoiser.model.to("cuda")
    assert any(isinstance(layer, WeightOnlyLinear) for layer in denoiser.model.modules())
    value = denoiser(
        DenoiserInput(
            latents=torch.randn(1, 32, 2, 2, 2, device="cuda", dtype=torch.bfloat16),
            timestep=torch.tensor(1000, device="cuda"),
            next_timestep=torch.tensor(0, device="cuda"),
            step_index=0,
            total_steps=1,
            conditioning={"encoder_hidden_states": torch.randn(1, 4, 128, device="cuda", dtype=torch.bfloat16)},
        )
    ).sample
    assert value.shape == (1, 32, 2, 2, 2) and torch.isfinite(value).all()
    denoiser.model.to("cpu")
    decoder = build_pixelumm_decoder(context(ComponentKind.DECODER, "decoder.safetensors", codec))
    decoder.model.to("cuda")
    layers = [layer for layer in decoder.model.modules() if isinstance(layer, Float8Linear)]
    assert layers and all(layer.weight is None for layer in layers)
    pixels = decoder.model.decode(value, return_dict=False)
    assert isinstance(pixels, tuple)
    assert pixels[0].shape == (1, 3, 8, 64, 64) and torch.isfinite(pixels[0]).all()
    assert sum(layer.low_precision_kernel_calls for layer in layers) > 0


class _TextFixture:
    def encode(self, request: Any, *, device: torch.device, dtype: torch.dtype) -> Conditioning:
        prompts = request.inputs.get("segments") or list(request.prompts)
        tokens = torch.linspace(-1, 1, 4 * 128, device=device, dtype=dtype).reshape(1, 4, 128)
        return Conditioning(
            positive={
                "segments": [
                    {
                        "encoder_hidden_states": tokens.clone(),
                        "encoder_attention_mask": torch.tensor([[1, 1, 1, 0]], device=device),
                    }
                    for _ in prompts
                ]
            },
            negative={"encoder_hidden_states": torch.zeros_like(tokens)},
        )


@pytest.fixture(scope="module", params=[False, True], ids=["bf16", "int8-fp8"])
def gpu_pipeline(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> HyNomasPipeline:
    if request.param and torch.cuda.get_device_capability() < (8, 9):
        pytest.skip("native FP8 execution requires SM89 or newer")
    torch.manual_seed(13)
    root = tmp_path_factory.mktemp("hy-nomas-cuda")
    dit: dict[str, Any] = dict(
        num_layers=1,
        num_attention_heads=8,
        attention_head_dim=128,
        text_embed_dim=128,
        rope_axes_dim=(16, 56, 56),
        camera_attention_dim=128,
        camera_num_attention_heads=1,
        camera_rope_axes_dim=(16, 56, 56),
    )
    codec: dict[str, Any] = dict(
        hidden_size=1024,
        intermediate_size=3072,
        num_layers=1,
        num_heads=8,
        num_kv_heads=2,
        head_dim=128,
        temporal_window_latents=2,
        temporal_overlap_latents=1,
    )
    save_file(HyNomasTransformer3DModel(**dit).state_dict(), root / "dit.safetensors")
    save_file(PixelUMMDecoder(**codec).state_dict(), root / "decoder.safetensors")
    recipe = hy_nomas_low_memory_recipe() if request.param else hy_nomas_recipe()
    components = tuple(
        replace(component, factory=lambda context: _TextFixture())
        if component.key.kind is ComponentKind.CONDITIONER
        else component
        for component in recipe.components
    )
    registry = NativeDiffusionRegistry()
    registry.register(replace(recipe, components=components))
    native = NativeDiffusionPipeline.from_registry(
        registry,
        recipe.model_id,
        policy=RuntimePolicy(device="cuda", dtype=torch.bfloat16, offload=OffloadPolicy()),
        checkpoint_overrides={"transformer": str(root / "dit.safetensors"), "vae": str(root / "decoder.safetensors")},
        component_options={"denoiser:main": {"config": dit}, "decoder:main": {"config": codec}},
    )
    return HyNomasPipeline(native_pipeline=native, device="cuda")


@torch.inference_mode()
def test_public_cuda_generation_camera_and_repeatability(gpu_pipeline: HyNomasPipeline) -> None:
    camera = torch.eye(4).repeat(1, 8, 1, 1)
    moved = camera.clone()
    moved[:, :, 0, 3] = torch.linspace(0, 1, 8)
    options: dict[str, Any] = dict(
        prompt=["A scene", "A turn"],
        segment_latent_lengths=[1, 1],
        height=64,
        width=64,
        num_frames=8,
        num_inference_steps=2,
        guidance_scale=2,
        seed=17,
        output_type="pt",
        return_dict=True,
    )
    first = gpu_pipeline(**options, camera_extrinsics=camera)["video"].cpu()
    changed = gpu_pipeline(**options, camera_extrinsics=moved)["video"].cpu()
    repeated = gpu_pipeline(**options, camera_extrinsics=camera)["video"].cpu()
    assert first.shape == (1, 3, 8, 64, 64) and torch.isfinite(first).all()
    assert not torch.equal(first, changed)
    torch.testing.assert_close(first, repeated, rtol=0, atol=0)
    parts = gpu_pipeline.native_pipeline.runner.components
    low_memory = gpu_pipeline.native_pipeline.model_id == "hy-nomas-low-memory"
    expected_device = "cpu" if low_memory else "cuda"
    assert next(parts.denoiser.model.parameters()).device.type == expected_device
    assert parts.decoder.model.device.type == expected_device
    if low_memory:
        weights = [layer for layer in parts.denoiser.model.modules() if isinstance(layer, WeightOnlyLinear)]
        fp8 = [layer for layer in parts.decoder.model.modules() if isinstance(layer, Float8Linear)]
        assert weights and all(layer.weight is None for layer in weights)
        assert sum(layer.packed_weight_calls for layer in weights) > 0
        assert fp8 and all(layer.weight is None for layer in fp8)
        assert sum(layer.low_precision_kernel_calls for layer in fp8) > 0


@torch.inference_mode()
def test_resident_cuda_failure_isolation_and_reset(
    gpu_pipeline: HyNomasPipeline,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gpu_pipeline.configure_realtime(prompt="A scene", height=64, width=64, seed=42)
    first = gpu_pipeline.stream_realtime(["forward"])
    assert first is not None
    first_frames = first["frames"].frames.clone()
    owner = gpu_pipeline._realtime_state
    assert owner is not None
    runner = gpu_pipeline.native_pipeline.runner
    inputs: dict[str, Any] = (
        {"kv_cache_quantization": "int8"} if gpu_pipeline.native_pipeline.model_id.endswith("low-memory") else {}
    )
    request, _ = gpu_pipeline._prepare_call(prompt="An independent scene", height=64, width=64, num_frames=8, **inputs)
    other = runner.new_session(request)
    runner.continue_session(request, other)
    other_key = other.positive[0]["store"].materialize()[0].clone()
    original = runner.components.denoiser.model.forward
    failure = RuntimeError("cache commit failed")

    def forward(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        if kwargs.get("update_cache") and kwargs.get("kv_cache") is owner.positive:
            raise failure
        return result

    with monkeypatch.context() as patch:
        patch.setattr(runner.components.denoiser.model, "forward", forward)
        with pytest.raises(RuntimeError) as caught:
            gpu_pipeline.stream_realtime(["left"])
    assert caught.value is failure and owner.failed
    assert not owner.positive and not owner.negative and owner.decoder is None
    assert not other.failed
    torch.testing.assert_close(other.positive[0]["store"].materialize()[0], other_key, rtol=0, atol=0)
    gpu_pipeline.reset_realtime()
    gpu_pipeline.configure_realtime(prompt="A scene", height=64, width=64, seed=42)
    repeated = gpu_pipeline.stream_realtime(["forward"])
    assert repeated is not None
    torch.testing.assert_close(first_frames, repeated["frames"].frames, rtol=0, atol=0)
    gpu_pipeline.reset_realtime()


@torch.inference_mode()
def test_hy_nomas_cuda_qwen_text_load_matches_hf(tmp_path: Path) -> None:
    from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLTextConfig
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLTextModel

    from worldfoundry.base_models.diffusion_model.loaders import ModuleLoadSpec, NativeModuleLoader
    from worldfoundry.base_models.diffusion_model.models.encoders.cosmos2p5.component import Cosmos25TextBackbone
    from worldfoundry.base_models.diffusion_model.models.encoders.hy_nomas import _restore_text_rotary

    config = Qwen2_5_VLTextConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=2,
        vocab_size=32,
        rope_scaling={"rope_type": "default", "mrope_section": [4, 6, 6]},
    )
    Qwen2_5_VLTextModel(config).eval().save_pretrained(tmp_path, safe_serialization=True)
    reference = (
        Qwen2_5_VLTextModel.from_pretrained(
            tmp_path,
            dtype=torch.bfloat16,
            local_files_only=True,
            attn_implementation="sdpa",
        )
        .to("cuda")
        .eval()
    )
    native = NativeModuleLoader().load(
        ModuleLoadSpec(
            module_class=Cosmos25TextBackbone,
            config={"config": config},
            state_dict_converter=lambda state: {"model." + name: value for name, value in state.items()},
        ),
        CheckpointSpec(source=str(tmp_path / "model.safetensors")),
        RuntimePolicy(device="cuda", dtype=torch.bfloat16, offload=OffloadPolicy()),
    )
    assert isinstance(native, Cosmos25TextBackbone)
    _restore_text_rotary(native)
    native.to("cpu").to("cuda")
    assert native.model.rotary_emb.inv_freq.dtype == torch.float32
    torch.testing.assert_close(native.model.rotary_emb.inv_freq, reference.rotary_emb.inv_freq, rtol=0, atol=0)
    ids = torch.arange(1280, device="cuda").remainder(32).reshape(1, -1)
    expected = reference(input_ids=ids, output_hidden_states=True, use_cache=False).hidden_states[-3]
    actual = native.model(input_ids=ids, output_hidden_states=True, use_cache=False).hidden_states[-3]
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
