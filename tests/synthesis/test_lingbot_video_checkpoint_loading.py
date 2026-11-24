# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest
import torch
from diffusers.configuration_utils import ConfigMixin
from diffusers.models.modeling_utils import ModelMixin
from diffusers.pipelines import pipeline_utils

from worldfoundry.base_models.diffusion_model.schedulers.flow_unipc_compat import FlowUniPCMultistepScheduler
from worldfoundry.synthesis.visual_generation.lingbot_video import runner
from worldfoundry.synthesis.visual_generation.lingbot_video.pipeline_lingbot_video import LingBotVideoPipeline
from worldfoundry.synthesis.visual_generation.lingbot_video.pipeline_lingbot_video_i2v import (
    LingBotVideoImageToVideoPipeline,
)
from worldfoundry.synthesis.visual_generation.lingbot_video.transformer_lingbot_video import (
    LingBotVideoTransformer3DModel,
)

pytestmark = pytest.mark.unit

# Both released indexes/configs are identical: robbyant/lingbot-video-dense-1.3b
# at f9789a7d9b4772a47aba62d4eb5282ddefd1da21 and lingbot-video-moe-30b-a3b
# at f2e538f64afe00cc4ae674db2aeb52e2945edfd5.
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "lingbot_video"
PACKAGE = "worldfoundry.synthesis.visual_generation.lingbot_video"
SCHEDULER_MODULE = "worldfoundry.base_models.diffusion_model.schedulers.flow_unipc_compat"


@pytest.mark.parametrize("legacy_prefix", ["lingbot_video", PACKAGE])
@pytest.mark.parametrize("pipeline_class", [LingBotVideoPipeline, LingBotVideoImageToVideoPipeline])
def test_lingbot_video_maps_released_component_names_without_changing_the_index(
    pipeline_class: type[LingBotVideoPipeline],
    legacy_prefix: str,
) -> None:
    original = (FIXTURES / "model_index.json").read_text()
    config = json.loads(original)
    config["scheduler"][0] = f"{legacy_prefix}.scheduling_flow_unipc"
    original_config = json.dumps(config, sort_keys=True)

    components, unused, hidden = pipeline_class.extract_init_dict(config)

    assert components == {
        "transformer": [f"{PACKAGE}.transformer_lingbot_video", "LingBotVideoTransformer3DModel"],
        "scheduler": [SCHEDULER_MODULE, "FlowUniPCMultistepScheduler"],
        "vae": ["diffusers", "AutoencoderKLWan"],
        "text_encoder": ["transformers", "Qwen3VLForConditionalGeneration"],
        "processor": ["transformers", "Qwen3VLProcessor"],
    }
    assert unused == {}
    assert hidden["_class_name"] == "LingBotVideoPipeline"
    assert json.dumps(config, sort_keys=True) == original_config


class _AuxiliaryModel(ModelMixin, ConfigMixin):
    def __init__(self, dtype: torch.dtype) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1, dtype=dtype))


class _Processor:
    """Weight-free processor stand-in for the component-loading contract."""


@pytest.mark.parametrize("legacy_prefix", ["lingbot_video", PACKAGE])
@pytest.mark.parametrize("mode", ["t2v", "ti2v"])
@pytest.mark.parametrize("defer_transformer", [False, True])
def test_lingbot_video_loads_packaged_components_without_namespace_aliases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str, defer_transformer: bool, legacy_prefix: str
) -> None:
    config = json.loads((FIXTURES / "model_index.json").read_text())
    config["scheduler"][0] = f"{legacy_prefix}.scheduling_flow_unipc"
    index = json.dumps(config).encode()
    (tmp_path / "model_index.json").write_bytes(index)
    scheduler_dir = tmp_path / "scheduler"
    scheduler_dir.mkdir()
    (scheduler_dir / "scheduler_config.json").write_bytes((FIXTURES / "scheduler_config.json").read_bytes())
    transformer = LingBotVideoTransformer3DModel(
        patch_size=(1, 1, 1),
        in_channels=2,
        out_channels=2,
        hidden_size=24,
        num_attention_heads=2,
        depth=1,
        intermediate_size=32,
        text_dim=8,
        freq_dim=8,
        axes_dims=(4, 4, 4),
        axes_lens=(4, 4, 4),
    )
    transformer.save_pretrained(tmp_path / "refiner")
    loaded: dict[str, tuple[str, str, torch.dtype]] = {}
    auxiliary: dict[str, Any] = {}
    original_load = pipeline_utils.load_sub_model
    for name in ("lingbot_video", "lingbot_video.scheduling_flow_unipc", "lingbot_video.transformer_lingbot_video"):
        monkeypatch.delitem(sys.modules, name, raising=False)

    def load_sub_model(**kwargs: Any) -> Any:
        assert not any(name == "lingbot_video" or name.startswith("lingbot_video.") for name in sys.modules)
        name = kwargs["name"]
        loaded[name] = kwargs["library_name"], kwargs["class_name"], kwargs["torch_dtype"]
        if name == "scheduler":
            return original_load(**kwargs)
        auxiliary[name] = _Processor() if name == "processor" else _AuxiliaryModel(kwargs["torch_dtype"])
        return auxiliary[name]

    monkeypatch.setattr(pipeline_utils, "load_sub_model", load_sub_model)
    monkeypatch.setattr(runner, "_default_device", lambda: torch.device("cpu"))
    dtype_map = {
        "default": torch.float32,
        "transformer": torch.float32,
        "text_encoder": torch.bfloat16,
        "vae": torch.float32,
    }

    pipe = runner._load_diffusers_pipe(tmp_path, dtype_map, mode, "refiner", defer_transformer)

    assert type(pipe) is (LingBotVideoImageToVideoPipeline if mode == "ti2v" else LingBotVideoPipeline)
    assert isinstance(pipe.transformer, LingBotVideoTransformer3DModel)
    assert isinstance(pipe.scheduler, FlowUniPCMultistepScheduler)
    for key, tensor in transformer.state_dict().items():
        torch.testing.assert_close(pipe.transformer.state_dict()[key], tensor, rtol=0, atol=0)
    noise = torch.randn(1, 2, 1, 2, 2)
    text = torch.randn(1, 2, 8)
    timestep = torch.tensor([500.0])
    with torch.inference_mode():
        expected = transformer.eval()(noise, timestep, text).sample
        actual = pipe.transformer(noise, timestep, text).sample
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert pipe.vae is auxiliary["vae"]
    assert pipe.text_encoder is auxiliary["text_encoder"]
    assert pipe.processor is auxiliary["processor"]
    assert loaded == {
        "scheduler": (SCHEDULER_MODULE, "FlowUniPCMultistepScheduler", torch.float32),
        "vae": ("diffusers", "AutoencoderKLWan", torch.float32),
        "text_encoder": ("transformers", "Qwen3VLForConditionalGeneration", torch.bfloat16),
        "processor": ("transformers", "Qwen3VLProcessor", torch.float32),
    }
    assert (tmp_path / "model_index.json").read_bytes() == index
    assert not any(name == "lingbot_video" or name.startswith("lingbot_video.") for name in sys.modules)
