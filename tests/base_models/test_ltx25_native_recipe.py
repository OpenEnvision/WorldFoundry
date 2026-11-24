# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Native split-recipe, projected schedule and request-conditioning contracts."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest
import torch
from PIL import Image

from worldfoundry.base_models.diffusion_model.contracts import (
    Conditioning,
    DiffusionRequest,
    ModalityState,
    SamplingConfig,
)
from worldfoundry.base_models.diffusion_model.models.encoders.ltx.component import LTXGemmaPromptConditioner
from worldfoundry.base_models.diffusion_model.models.initializers.ltx import enhance_ltx_prompt
from worldfoundry.base_models.diffusion_model.recipes import default_native_diffusion_registry
from worldfoundry.base_models.diffusion_model.schedulers.ltx import (
    LTXProjectedAncestralEulerScheduler,
    LTXProjectedEulerScheduler,
)


def test_ltx25_recipe_is_registered_with_canonical_split_components() -> None:
    registry = default_native_diffusion_registry()
    recipe = registry.resolve("ltx-2.5")
    assert all(registry.resolve(alias) is recipe for alias in ("ltx25", "ltx2.5", "ltx-2.5-distilled"))
    assert len(recipe.components) == 7
    assert len(recipe.checkpoints) == 6
    assert recipe.execution.strategy == "joint-multistage"
    assert recipe.execution.options == {"stage_steps": (8, 3), "step_noise_seed_offset": 10000}
    assert all(component.factory.__module__.startswith("worldfoundry.base_models.") for component in recipe.components)
    assert all(value.repo_id == "Lightricks/LTX-2.5" for value in recipe.checkpoints.values())
    assert {value.revision for value in recipe.checkpoints.values()} == {"bf86adedf518142442575d1ce2e767b7d01c8c76"}


def _state(dtype: torch.dtype) -> ModalityState:
    return ModalityState(
        latent=torch.tensor([[[0.5, 1.0], [1.5, 2.0]]], dtype=dtype),
        clean_latent=torch.full((1, 2, 2), 0.25, dtype=dtype),
        denoise_mask=torch.tensor([[[0.0], [1.0]]], dtype=dtype),
        positions=torch.zeros(1, 3, 2, 2),
        keyframes_mask=torch.ones(1, 2, 1),
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_projected_euler_preserves_prediction_projection_and_terminal_rounding(dtype: torch.dtype) -> None:
    scheduler = LTXProjectedEulerScheduler((1.0, 0.5, 0.0))
    steps = scheduler.schedule(SamplingConfig(num_inference_steps=2), device=torch.device("cpu"), dtype=dtype)
    state = _state(dtype)
    prediction = torch.full_like(state.latent, 0.75)
    projected = prediction * state.denoise_mask + state.clean_latent * (1 - state.denoise_mask)
    result = scheduler.step_state(prediction, steps[0], state, generator=torch.Generator())
    expected = (state.latent.float() + (state.latent - projected).float() * -0.5).to(dtype)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    terminal = scheduler.step_state(prediction, steps[-1], state, generator=torch.Generator())
    torch.testing.assert_close(terminal, projected, rtol=0, atol=0)


def test_projected_ancestral_euler_uses_independent_noise_and_reprojects_frozen_tokens() -> None:
    state = _state(torch.float32)
    scheduler = LTXProjectedAncestralEulerScheduler((1.0, 0.5, 0.0))
    step = scheduler.schedule(SamplingConfig(num_inference_steps=2), device=torch.device("cpu"), dtype=torch.float32)[0]
    prediction = torch.ones_like(state.latent)
    actual = scheduler.step_state(prediction, step, state, generator=torch.Generator().manual_seed(10042))
    projected = prediction * state.denoise_mask + state.clean_latent * (1 - state.denoise_mask)
    noise = torch.randn(state.latent.shape, generator=torch.Generator().manual_seed(10042))
    expected = (0.5 / 0.75) * (0.25 * state.latent + 0.75 * projected)
    coefficient = torch.tensor(0.5**2 - 0.25**2 * 0.5**2 / 0.75**2).sqrt()
    expected = expected + noise * coefficient
    expected = expected * state.denoise_mask + state.clean_latent * (1 - state.denoise_mask)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual[:, :1], state.clean_latent[:, :1], rtol=0, atol=0)


@pytest.mark.parametrize(
    ("seconds", "bounds", "fps", "frames"),
    [(2.3, (1, 6), 24, 49), (0.1, (1.1, 6), 24, 33), (20, (1, 6), 24, 137), (2.5, (1, 6), 25, 57)],
)
def test_auto_duration_clamps_and_snaps_on_the_temporal_grid(seconds, bounds, fps, frames) -> None:
    head = Mock(return_value=torch.tensor([seconds]))
    conditioner = LTXGemmaPromptConditioner(Mock(), Mock(), duration_head=head)
    request = DiffusionRequest(prompt="A fox.", inputs={"auto_duration": bounds, "frame_rate": fps})
    contexts = Conditioning(positive={"video_context": torch.ones(1, 2, 3), "audio_context": torch.ones(1, 2, 2)})
    resolved = conditioner.resolve_request(request, contexts)
    assert resolved.num_frames == frames
    assert request.num_frames == 1


def test_generated_keyframes_require_checkpoint_position_embedding_support() -> None:
    conditioner = LTXGemmaPromptConditioner(Mock(), Mock())
    with pytest.raises(ValueError, match="keyframe position embeddings"):
        conditioner.encode(
            DiffusionRequest(prompt="A fox.", inputs={"num_generated_keyframes": 1}),
            device=torch.device("cpu"),
            dtype=torch.float32,
        )


def test_enhanced_prompt_cleaning_and_image_long_side_resize(tmp_path: Path) -> None:
    encoder = Mock()
    encoder.enhance_t2v.return_value = "123: ‘Fox’—sunset\u00a0light"
    assert enhance_ltx_prompt(encoder, "A fox.", []) == "Fox'-sunset light"
    encoder.enhance_t2v.assert_called_once_with("A fox.", seed=42)
    image = tmp_path / "reference.png"
    Image.new("RGB", (48, 32), color=(51, 102, 153)).save(image)
    encoder.enhance_i2v.return_value = "  “Ocean”"
    assert enhance_ltx_prompt(encoder, "Ocean.", [{"path": str(image)}]) == 'Ocean"'
    args, kwargs = encoder.enhance_i2v.call_args
    assert args[0] == "Ocean."
    assert args[1].shape == (597, 896, 3)
    assert args[1].dtype is torch.uint8
    assert kwargs == {"seed": 42}


def test_modality_clone_keeps_keyframe_storage_independent() -> None:
    state = _state(torch.float32)
    cloned = state.clone()
    assert cloned.keyframes_mask is not None and state.keyframes_mask is not None
    cloned.keyframes_mask.zero_()
    assert torch.count_nonzero(state.keyframes_mask) == 2


@pytest.mark.parametrize(
    ("offload_mode", "expected_offload"),
    [("component", True), ("none", False)],
)
def test_split_gemma_falls_back_to_native_gemma4_unified_when_transformers_lacks_it(
    monkeypatch,
    tmp_path: Path,
    offload_mode: str,
    expected_offload: bool,
) -> None:
    import builtins
    import types

    from worldfoundry.base_models.diffusion_model.loaders import CheckpointSpec, MaterializedCheckpoint
    from worldfoundry.base_models.diffusion_model.models.encoders.ltx import component as encoder_component
    from worldfoundry.base_models.diffusion_model.models.encoders.ltx import gemma as gemma_mod
    from worldfoundry.base_models.diffusion_model.models.encoders.ltx import gemma4_unified
    from worldfoundry.base_models.diffusion_model.optimizations import OffloadMode, OffloadPolicy, RuntimePolicy

    ckpt_file = tmp_path / "gemma4.safetensors"
    ckpt_file.write_bytes(b"")
    fake_assets = types.SimpleNamespace(config_dict={"model_type": "gemma4_unified"})
    sentinel_wrapper = torch.nn.Linear(2, 2)
    seen_calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        encoder_component.NativeCheckpointResolver,
        "materialize",
        lambda self, spec: MaterializedCheckpoint(root=tmp_path, paths=(ckpt_file,)),
    )
    monkeypatch.setattr(gemma_mod.GemmaAssets, "load", classmethod(lambda cls, src: fake_assets))

    def fake_load_native(path: Path, *, device, dtype, offload_to_cpu: bool = False):
        seen_calls.append({"path": path, "device": device, "dtype": dtype, "offload_to_cpu": offload_to_cpu})
        return sentinel_wrapper, fake_assets

    monkeypatch.setattr(gemma4_unified, "load_native_gemma4_unified_encoder", fake_load_native)

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("transformers.models.gemma4_unified"):
            raise ModuleNotFoundError("No module named 'transformers.models.gemma4_unified'", name=name)
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    policy = RuntimePolicy(
        device="cpu",
        dtype=torch.float32,
        offload=OffloadPolicy(mode=OffloadMode(offload_mode)),
    )
    module, assets = encoder_component._load_split_gemma(CheckpointSpec(source=str(ckpt_file)), policy)
    assert module is sentinel_wrapper
    assert assets is fake_assets
    assert seen_calls == [
        {
            "path": ckpt_file,
            "device": torch.device("cpu"),
            "dtype": torch.float32,
            "offload_to_cpu": expected_offload,
        }
    ]


def test_native_gemma4_unified_wrapper_cpu_offload_returns_backbone_to_cpu(monkeypatch) -> None:
    from worldfoundry.base_models.diffusion_model.models.encoders.ltx.gemma4_unified import (
        NativeGemma4UnifiedEncoderWrapper,
        NativeGemma4UnifiedTextBackbone,
    )

    text_cfg = {
        "hidden_size": 16,
        "intermediate_size": 32,
        "vocab_size": 64,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "num_global_key_value_heads": 1,
        "head_dim": 8,
        "global_head_dim": 8,
        "attention_k_eq_v": True,
        "sliding_window": 4,
        "layer_types": ["sliding_attention", "full_attention"],
    }
    backbone = NativeGemma4UnifiedTextBackbone(text_cfg).eval()

    class _FakeTok:
        def tokenize_with_weights(self, text: str) -> dict[str, list[tuple[int, int]]]:
            del text
            return {"gemma": [(1, 1), (2, 1), (3, 1)]}

    target_dev = torch.device("meta")
    moves: list[str] = []
    real_to = backbone.to

    def tracked_to(*args, **kwargs):
        dev = args[0] if args else kwargs.get("device")
        if dev is not None:
            moves.append(str(dev))
        if str(dev) == "meta":
            return backbone
        return real_to(*args, **kwargs)

    monkeypatch.setattr(backbone, "to", tracked_to)
    wrapper = NativeGemma4UnifiedEncoderWrapper(
        backbone,
        _FakeTok(),
        execution_device=torch.device("cpu"),
        offload_to_cpu=True,
    )
    assert wrapper.execution_device == torch.device("cpu")

    # Exercise offloading branch cleanly without requiring CUDA by using a dummy non-cpu execution device marker.
    wrapper._execution_device = target_dev
    assert wrapper.execution_device == target_dev
    monkeypatch.setattr(
        torch,
        "tensor",
        lambda data, device=None, dtype=None: torch.as_tensor(data, dtype=dtype),
    )
    encoded = wrapper.encode(["hello"])
    assert len(encoded) == 1
    hidden_tuple, mask = encoded[0]
    assert len(hidden_tuple) == 3
    assert all(h.shape == (1, 3, 16) for h in hidden_tuple)
    assert mask.shape == (1, 3)
    assert moves == ["meta", "cpu"]
    assert next(backbone.parameters()).device.type == "cpu"
