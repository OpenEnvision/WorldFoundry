# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch

from worldfoundry.synthesis.visual_generation.show_o.show_o_runtime.models import MAGVITv2
from worldfoundry.synthesis.visual_generation.show_o.show_o_runtime.models.modeling_utils import (
    ConfigMixin,
    ModelMixin,
    register_to_config,
)
from worldfoundry.synthesis.visual_generation.show_o.worldfoundry_runtime import ShowORuntime

pytestmark = pytest.mark.unit


class _TinyModel(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self, width: int = 2) -> None:
        super().__init__()
        self.projection = torch.nn.Linear(width, width)


def test_vq_factory_uses_the_package_class_with_foreign_models_namespace(monkeypatch: pytest.MonkeyPatch) -> None:
    foreign = ModuleType("models")
    monkeypatch.setitem(sys.modules, "models", foreign)

    assert ShowORuntime._get_vq_model_class("magvitv2") is MAGVITv2
    assert sys.modules["models"] is foreign
    with pytest.raises(ValueError, match="not supported"):
        ShowORuntime._get_vq_model_class("unknown")


def test_from_flax_uses_diffusers_converter(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _TinyModel().save_config(tmp_path)
    checkpoint = tmp_path / "diffusion_flax_model.msgpack"
    checkpoint.touch()
    calls = []

    def convert(model: _TinyModel, path: str) -> _TinyModel:
        calls.append(Path(path))
        with torch.no_grad():
            model.projection.weight.fill_(0.25)
        return model

    converter = ModuleType("diffusers.models.modeling_pytorch_flax_utils")
    monkeypatch.setattr(converter, "load_flax_checkpoint_in_pytorch_model", convert, raising=False)
    monkeypatch.setitem(sys.modules, converter.__name__, converter)

    loaded = _TinyModel.from_pretrained(tmp_path, from_flax=True, local_files_only=True, low_cpu_mem_usage=False)

    assert calls == [checkpoint]
    assert not loaded.training
    torch.testing.assert_close(loaded.projection.weight, torch.full((2, 2), 0.25), rtol=0, atol=0)


def test_runtime_package_does_not_register_global_aliases() -> None:
    import subprocess

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import worldfoundry.synthesis.visual_generation.show_o.show_o_runtime; "
            "assert 'models' not in sys.modules; assert 'inference_support' not in sys.modules; "
            "assert 'torch' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_ensure_runtime_loads_package_classes_without_path_or_namespace_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    from worldfoundry.synthesis.visual_generation.show_o.show_o_runtime import models

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0, "[BOS]": 1, "[EOS]": 2}, unk_token="[UNK]")),
        unk_token="[UNK]",
        bos_token="[BOS]",
        eos_token="[EOS]",
    )
    calls = []
    model, vq_model = torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)

    def load_tokenizer(path: str, *, padding_side: str) -> PreTrainedTokenizerFast:
        calls.append(("tokenizer", path, padding_side))
        return tokenizer

    def load_model(path: str, *, llm_model_path: str, local_files_only: bool) -> torch.nn.Module:
        calls.append(("model", path, llm_model_path, local_files_only))
        return model

    def load_vq(path: str) -> torch.nn.Module:
        calls.append(("vq", path))
        return vq_model

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", load_tokenizer)
    monkeypatch.setattr(models.Showo, "from_pretrained", load_model)
    monkeypatch.setattr(models.MAGVITv2, "from_pretrained", load_vq)
    foreign = ModuleType("models")
    monkeypatch.setitem(sys.modules, "models", foreign)
    search_path = sys.path.copy()
    runtime = ShowORuntime(
        device="cpu",
        pretrained_model_path="/weights/show-o",
        vq_model_path="/weights/magvit",
        llm_model_path="/weights/phi",
    )

    loaded = runtime._ensure_runtime()

    assert loaded["model"] is model
    assert loaded["vq_model"] is vq_model
    assert loaded["tokenizer"] is tokenizer
    assert not model.training and not vq_model.training
    assert all(not parameter.requires_grad for parameter in vq_model.parameters())
    assert calls == [
        ("tokenizer", "/weights/phi", "left"),
        ("vq", "/weights/magvit"),
        ("model", "/weights/show-o", "/weights/phi", True),
    ]
    assert runtime._ensure_runtime() is loaded
    assert len(calls) == 3
    assert sys.path == search_path
    assert sys.modules["models"] is foreign
