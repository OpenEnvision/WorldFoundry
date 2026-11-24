"""Public Hy-Nomas loading and Studio routing with real small diffusion components."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors.torch import save_file

from worldfoundry.base_models.diffusion_model import NativeDiffusionPipeline
from worldfoundry.base_models.diffusion_model.contracts import Conditioning
from worldfoundry.base_models.diffusion_model.models.autoencoders.pixelumm import PixelUMMDecoder
from worldfoundry.base_models.diffusion_model.models.networks.hy_nomas import HyNomasTransformer3DModel
from worldfoundry.base_models.diffusion_model.optimizations import OffloadPolicy, RuntimePolicy
from worldfoundry.base_models.diffusion_model.recipes.hy_nomas import hy_nomas_recipe
from worldfoundry.base_models.diffusion_model.recipes.registry import NativeDiffusionRegistry
from worldfoundry.core.execution.realtime.contracts import RealtimeSpec
from worldfoundry.pipelines.hy_nomas.pipeline_hy_nomas import HyNomasPipeline
from worldfoundry.studio.inference.catalog import find_entry


class _TextFixture:
    def encode(self, request: Any, *, device: torch.device, dtype: torch.dtype) -> Conditioning:
        return Conditioning(
            positive={
                "segments": [
                    {"encoder_hidden_states": torch.ones(1, 3, 8, device=device, dtype=dtype)}
                    for _ in (request.inputs.get("segments") or [request.prompt])
                ]
            }
        )


@pytest.fixture
def pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> HyNomasPipeline:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    config: dict[str, Any] = dict(
        num_layers=1,
        num_attention_heads=2,
        attention_head_dim=8,
        text_embed_dim=8,
        rope_axes_dim=(2, 2, 4),
        camera_attention_dim=8,
        camera_num_attention_heads=1,
        camera_rope_axes_dim=(2, 2, 4),
    )
    codec: dict[str, Any] = dict(
        hidden_size=16,
        intermediate_size=32,
        num_layers=1,
        num_heads=2,
        num_kv_heads=1,
        head_dim=8,
    )
    save_file(HyNomasTransformer3DModel(**config).state_dict(), tmp_path / "dit.safetensors")
    save_file(PixelUMMDecoder(**codec).state_dict(), tmp_path / "decoder.safetensors")
    recipe = hy_nomas_recipe()
    components = tuple(
        replace(component, factory=lambda context: _TextFixture())
        if component.key.kind.value == "conditioner"
        else component
        for component in recipe.components
    )
    registry = NativeDiffusionRegistry()
    registry.register(replace(recipe, components=components))

    def load(model_id: str, **kwargs: Any) -> NativeDiffusionPipeline:
        return NativeDiffusionPipeline.from_registry(
            registry,
            model_id,
            policy=RuntimePolicy(device="cpu", dtype=torch.float32, offload=OffloadPolicy()),
            checkpoint_overrides={
                "transformer": str(tmp_path / "dit.safetensors"),
                "vae": str(tmp_path / "decoder.safetensors"),
            },
            component_options={"denoiser:main": {"config": config}, "decoder:main": {"config": codec}},
        )

    monkeypatch.setattr(NativeDiffusionPipeline, "from_pretrained", load)
    return HyNomasPipeline.from_pretrained(device="cpu")


def test_public_loading_and_ordered_prompts(pipeline: HyNomasPipeline) -> None:
    result = pipeline(
        prompt=["A scene", "A turn"],
        height=32,
        width=32,
        num_frames=8,
        num_inference_steps=1,
        output_type="pt",
        return_dict=True,
    )
    assert result["video"].shape == (1, 3, 8, 32, 32)
    assert torch.isfinite(result["video"]).all()


def test_public_explicit_checkpoint_roles() -> None:
    result = HyNomasPipeline._checkpoint_overrides("/unused/model", {"text_encoder_path": "/unused/text"})
    assert result == {"transformer": "/unused/model", "vae": "/unused/model", "text-encoder": "/unused/text"}


def test_loading_low_memory_selects_native_recipe(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def load(model_id: str, **kwargs: Any) -> Any:
        calls.append((model_id, kwargs))
        return type("Native", (), {"model_id": model_id})()

    monkeypatch.setattr(NativeDiffusionPipeline, "from_pretrained", load)
    pipe = HyNomasPipeline.from_pretrained(device="cpu", low_memory=True)
    assert calls[0][0] == "hy-nomas-low-memory"
    assert pipe.native_pipeline.model_id == "hy-nomas-low-memory"


def test_resident_session_prompt_control_and_reset(pipeline: HyNomasPipeline) -> None:
    payload = pipeline.configure_realtime(prompt="A scene", height=32, width=32)
    spec = RealtimeSpec.from_payload(payload["realtime_spec"])
    assert spec.first_input_frames == 40
    assert spec.first_chunk_frames == 4 and spec.steady_chunk_frames == 20
    result = pipeline.stream_realtime(["forward"], prompt="A turn")
    assert result is not None and result["frames"].frame_count == spec.first_chunk_frames
    first = pipeline._realtime_state
    assert first is not None
    assert first.frame_offset == 10
    next_chunk = pipeline.stream_realtime([])
    assert next_chunk is not None and next_chunk["frames"].frame_count == spec.steady_chunk_frames
    assert first.frame_offset == 15
    pipeline.reset_realtime()
    assert pipeline._realtime_state is None and pipeline._camera_pose is None
    pipeline.configure_realtime(prompt="A new scene", height=32, width=32)
    assert pipeline._realtime_state is not first
    assert pipeline._realtime_state is not None
    assert pipeline._realtime_state.frame_offset == 0


def test_studio_discovery_routes_attention_only_to_loading() -> None:
    entry = find_entry("hy-nomas")
    assert entry is not None
    assert entry.supports_from_pretrained and entry.supports_stream
    assert "camera_extrinsics" in entry.call_params
    assert "low_memory" in entry.load_params
    assert "attention_backend" not in entry.call_params
    assert "realtime_segments" in entry.stream_params


def test_low_memory_resident_session_starts_with_compressed_cache(pipeline: HyNomasPipeline) -> None:
    pipeline.native_pipeline.runner.model_id = "hy-nomas-low-memory"
    pipeline.configure_realtime(prompt="A scene", height=32, width=32)
    assert pipeline._realtime_state is not None
    store = pipeline._realtime_state.positive[0]["store"]
    assert store.key_config.dtype == "int8"
