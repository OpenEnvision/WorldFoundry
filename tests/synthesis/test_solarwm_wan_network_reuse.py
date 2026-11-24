# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Solar's native Wan projections retain checkpoint and causal-cache contracts."""

from pathlib import Path
from typing import Any

import pytest
import torch

from worldfoundry.base_models.diffusion_model.models.networks.wan.mixins import WanTransformerMethodsMixin
from worldfoundry.base_models.diffusion_model.models.networks.wan.reference_22 import WanRMSNorm, WanSelfAttention
from worldfoundry.base_models.diffusion_model.models.networks.wan.variants.moverse.causal import CausalHead
from worldfoundry.base_models.diffusion_model.models.networks.wan.variants.solarwm import (
    SolarWanAttentionProjections,
    SolarWanRMSNorm,
)
from worldfoundry.synthesis.visual_generation.solarwm.solarwm_runtime.solarwm.backends.wan22.runtime import loader
from worldfoundry.synthesis.visual_generation.solarwm.solarwm_runtime.solarwm.backends.wan22.runtime.modeling import (
    causal_model,
)

pytestmark = pytest.mark.unit


def _config(**updates: Any) -> dict[str, Any]:
    return {
        "model_type": "ti2v",
        "patch_size": (1, 2, 2),
        "text_len": 8,
        "in_dim": 4,
        "dim": 32,
        "ffn_dim": 64,
        "freq_dim": 16,
        "text_dim": 12,
        "out_dim": 4,
        "num_heads": 2,
        "num_layers": 1,
        "local_attn_size": 3,
        "sink_size": 1,
        "frame_seq_length": 4,
        **updates,
    }


def _cpu_text_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, **kwargs: Any) -> torch.Tensor:
    assert kwargs == {"k_lens": None}
    return torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    ).transpose(1, 2)


@pytest.mark.parametrize("model_prefix", [False, True])
def test_solarwm_wan_shard_loader_preserves_native_projection_keys(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, model_prefix: bool
) -> None:
    from safetensors.torch import save_file

    monkeypatch.setattr(causal_model, "flash_attention", _cpu_text_attention)
    config = {
        key: value for key, value in _config().items() if key not in causal_model.CausalWanModel.ignore_for_config
    }
    reference = causal_model.CausalWanModel.from_config(config)
    assert isinstance(reference, causal_model.CausalWanModel)
    reference = reference.bfloat16().eval()
    assert isinstance(reference, WanTransformerMethodsMixin)
    assert type(reference.head) is CausalHead
    assert isinstance(reference.blocks[0].self_attn, WanSelfAttention)
    assert isinstance(reference.blocks[0].self_attn, SolarWanAttentionProjections)
    assert type(reference.blocks[0].self_attn.norm_q) is SolarWanRMSNorm
    with torch.no_grad():
        for value in reference.parameters():
            value.uniform_(-0.1, 0.1)
    prefix = "model." if model_prefix else ""
    weights = {prefix + key: value for key, value in reference.state_dict().items()}
    names = sorted(weights)
    midpoint = len(names) // 2
    for index, keys in enumerate((names[:midpoint], names[midpoint:])):
        save_file({key: weights[key] for key in keys}, tmp_path / f"part-{index}.safetensors")
    loaded = causal_model.CausalWanModel.from_config(config)
    assert isinstance(loaded, causal_model.CausalWanModel)
    loaded = loaded.bfloat16().eval()
    report = loader.load_base_weights(loaded, tmp_path)
    assert report.missing_keys == report.unexpected_keys == ()
    assert report.stripped_model_prefix is model_prefix
    assert report.source_keys == report.target_keys == len(weights)
    torch.testing.assert_close(loaded.state_dict(), reference.state_dict(), rtol=0, atol=0)
    inputs = {
        "x": torch.randn(1, 4, 1, 4, 4, dtype=torch.bfloat16),
        "t": torch.full((1, 4), 500.0),
        "context": torch.randn(1, 5, reference.text_dim, dtype=torch.bfloat16),
        "seq_len": 4,
    }
    with torch.no_grad():
        torch.testing.assert_close(loaded(**inputs), reference(**inputs), rtol=0, atol=0)


def test_solarwm_wan_base_loading_initializes_anyflow_delta(tmp_path: Path) -> None:
    from safetensors.torch import save_file

    base = causal_model.CausalWanModel(**_config()).bfloat16()
    save_file(base.state_dict(), tmp_path / "base.safetensors")
    extended = causal_model.CausalWanModel(
        **_config(add_control_adapter=True, flow_objective="anyflow_forward_map")
    ).bfloat16()
    report = loader.load_base_weights(extended, tmp_path)
    assert report.missing_keys
    assert all(".cam_self_attn." in key or key.startswith("delta_embedding.") for key in report.missing_keys)
    torch.testing.assert_close(
        extended.delta_embedding.state_dict(), extended.time_embedding.state_dict(), rtol=0, atol=0
    )
    assert "anyflow_gate_buffer" not in extended.state_dict()


def test_solarwm_wan_rmsnorm_keeps_bfloat16_reduction() -> None:
    torch.manual_seed(7)
    x = torch.randn(2, 7, 32, dtype=torch.bfloat16)
    solar = SolarWanRMSNorm(32).bfloat16()
    native = WanRMSNorm(32).bfloat16()
    expected = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + solar.eps)
    torch.testing.assert_close(solar(x), expected, rtol=0, atol=0)
    assert (solar(x) - native(x)).abs().max() > 0


@pytest.mark.parametrize("camera_mode", ["parallel", "fused_prope"])
def test_solarwm_wan_causal_cache_is_request_owned(monkeypatch: pytest.MonkeyPatch, camera_mode: str) -> None:
    monkeypatch.setattr(causal_model, "flash_attention", _cpu_text_attention)
    model = (
        causal_model.CausalWanModel(**_config(add_control_adapter=True, camera_attention_mode=camera_mode))
        .bfloat16()
        .eval()
    )
    with torch.no_grad():
        for value in model.parameters():
            value.uniform_(-0.1, 0.1)

    def rollout(seed: int) -> tuple[list[torch.Tensor], list[dict[str, Any]]]:
        torch.manual_seed(seed)
        cache = [
            {
                "k": torch.zeros(1, 12, 2, 16, dtype=torch.bfloat16),
                "v": torch.zeros(1, 12, 2, 16, dtype=torch.bfloat16),
                "global_end_index": torch.zeros(1, dtype=torch.long),
                "local_end_index": torch.zeros(1, dtype=torch.long),
            }
        ]
        text_cache = [{"is_init": False}]
        context = torch.randn(1, 5, 12, dtype=torch.bfloat16)
        outputs = []
        with torch.no_grad():
            for frame in range(5):
                viewmats = torch.eye(4)[None, None].repeat(1, 4, 1, 1)
                viewmats[:, :, 0, 3] = frame * 0.2
                intrinsics = torch.eye(3)[None, None].repeat(1, 4, 1, 1)
                args = {
                    "x": torch.randn(1, 4, 1, 4, 4, dtype=torch.bfloat16),
                    "t": torch.full((1, 4), 500.0),
                    "context": context,
                    "seq_len": 4,
                    "y_camera": {"viewmats": viewmats, "K": intrinsics},
                    "kv_cache": cache,
                    "crossattn_cache": text_cache,
                    "current_start": frame * 4,
                    "cache_start": frame * 4,
                }
                before = cache[0]["k"].clone()
                model(**args, cache_update_policy="none")
                torch.testing.assert_close(cache[0]["k"], before, rtol=0, atol=0)
                outputs.append(model(**args).clone())
                model(**(args | {"t": torch.zeros(1, 4)}))
        assert text_cache[0]["is_init"]
        assert cache[0]["global_end_index"].item() == 20
        assert cache[0]["local_end_index"].item() == 12
        return outputs, cache

    first = rollout(17)
    other = rollout(29)
    repeat = rollout(17)
    torch.testing.assert_close(first, repeat, rtol=0, atol=0)
    assert not torch.equal(first[0][0], other[0][0])
    assert first[1][0]["k"].data_ptr() != repeat[1][0]["k"].data_ptr()
