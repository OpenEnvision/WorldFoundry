# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from types import ModuleType

import pytest
from packaging.version import parse

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _package_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    metadata = ModuleType("pkg_resources")
    monkeypatch.setattr(metadata, "parse_version", parse, raising=False)
    monkeypatch.setitem(sys.modules, "pkg_resources", metadata)


def test_load_easyanimate_components_returns_the_owned_classes() -> None:
    from worldfoundry.base_models.diffusion_model.models.initializers.easyanimate_conditioning import (
        get_image_to_video_latent,
    )
    from worldfoundry.synthesis.visual_generation.easyanimate.easyanimate_runtime import load_easyanimate_components
    from worldfoundry.synthesis.visual_generation.easyanimate.easyanimate_runtime.easyanimate.models import (
        name_to_autoencoder_magvit,
        name_to_transformer3d,
    )
    from worldfoundry.synthesis.visual_generation.easyanimate.easyanimate_runtime.easyanimate.pipeline.pipeline_easyanimate_inpaint import (
        EasyAnimateInpaintPipeline,
    )
    from worldfoundry.synthesis.visual_generation.easyanimate.easyanimate_runtime.easyanimate.utils.fp8_optimization import (
        convert_weight_dtype_wrapper,
    )

    components = load_easyanimate_components()
    assert components.name_to_autoencoder_magvit is name_to_autoencoder_magvit
    assert components.name_to_transformer3d is name_to_transformer3d
    assert components.inpaint_pipeline_cls is EasyAnimateInpaintPipeline
    assert components.multi_text_encoder_inpaint_pipeline_cls is EasyAnimateInpaintPipeline
    assert components.convert_weight_dtype_wrapper is convert_weight_dtype_wrapper
    assert components.get_image_to_video_latent is get_image_to_video_latent


def test_load_easyanimate_components_preserves_import_paths_and_unrelated_namespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from worldfoundry.synthesis.visual_generation.easyanimate.easyanimate_runtime import load_easyanimate_components

    unrelated = ModuleType("easyanimate")
    monkeypatch.setitem(sys.modules, "easyanimate", unrelated)
    paths = list(sys.path)
    components = load_easyanimate_components()
    assert components.inpaint_pipeline_cls.__module__.startswith(
        "worldfoundry.synthesis.visual_generation.easyanimate."
    )
    assert sys.path == paths
    assert sys.modules["easyanimate"] is unrelated


def test_easyanimate_text_projection_reuses_qwen_norm() -> None:
    import torch

    from worldfoundry.base_models.llm_mllm_core.mllm.qwen.beingh.qwen2.modeling_qwen2 import Qwen2RMSNorm
    from worldfoundry.synthesis.visual_generation.easyanimate.easyanimate_runtime.easyanimate.models.transformer3d import (
        EasyAnimateTransformer3DModel,
    )

    model = EasyAnimateTransformer3DModel(
        num_attention_heads=2,
        attention_head_dim=8,
        in_channels=4,
        out_channels=4,
        patch_size=2,
        sample_height=8,
        sample_width=8,
        time_embed_dim=16,
        num_layers=1,
        mmdit_layers=1,
        text_embed_dim=16,
        text_embed_dim_t5=None,
        add_norm_text_encoder=True,
    ).eval()
    projection = model.text_proj
    assert isinstance(projection, torch.nn.Sequential)
    norm = projection[0]
    assert type(norm) is Qwen2RMSNorm
    assert set(norm.state_dict()) == {"weight"}
    assert "text_proj.0.weight" in model.state_dict()
    x = torch.randn(2, 3, 16)
    with torch.no_grad():
        normalized = norm.weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + norm.variance_epsilon))
        expected = projection[1](normalized)
        actual = projection(x)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_easyanimate_activation_registry_preserves_elu_extension() -> None:
    import torch

    from worldfoundry.base_models.diffusion_model.models.networks.step_video.normalization import get_activation
    from worldfoundry.synthesis.visual_generation.easyanimate.easyanimate_runtime.easyanimate.vae.ldm.modules.vaemodules import (
        activations,
    )

    for name in ("swish", "silu", "mish", "gelu", "relu"):
        assert activations.get_activation(name) is get_activation(name)
    elu = activations.get_activation("ELU")
    assert isinstance(elu, torch.nn.ELU)
    assert activations.get_activation("elu") is elu
    x = torch.randn(32)
    torch.testing.assert_close(elu(x), torch.nn.functional.elu(x), atol=0, rtol=0)
    with pytest.raises(ValueError, match="Unsupported activation function: missing"):
        activations.get_activation("MISSING")
