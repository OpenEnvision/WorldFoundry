# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""OpenFlamingo imports stay lightweight and retain RoboFlamingo decoder lookup."""

import subprocess
import sys

import pytest

from worldfoundry.base_models.llm_mllm_core.mllm.open_flamingo.src.factory import _infer_decoder_layers_attr_name

pytestmark = pytest.mark.unit


def test_open_flamingo_package_preserves_import_paths() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; paths = list(sys.path); "
            "from worldfoundry.base_models.llm_mllm_core.mllm.open_flamingo.src import factory; "
            "assert not hasattr(factory, 'create_model_and_transforms'); "
            "assert paths == sys.path; "
            "assert 'open_flamingo' not in sys.modules; "
            "assert not {'torch', 'open_clip', 'transformers'} & sys.modules.keys()",
        ],
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize(
    ("class_name", "attribute"),
    [
        ("OPTForCausalLM", "model.decoder.layers"),
        ("GPTJForCausalLM", "transformer.h"),
        ("GPT-J", "transformer.h"),
        ("PythiaForCausalLM", "gpt_neox.layers"),
        ("LlamaForCausalLM", "model.layers"),
        ("GPTNeoXForCausalLM", "gpt_neox.layers"),
        ("MPTForCausalLM", "transformer.blocks"),
        ("MosaicGPT", "transformer.blocks"),
    ],
)
def test_open_flamingo_decoder_layers_attr_name(class_name: str, attribute: str) -> None:
    model = type(class_name, (), {})()

    assert _infer_decoder_layers_attr_name(model) == attribute


def test_open_flamingo_decoder_layers_attr_name_requires_a_known_backbone() -> None:
    with pytest.raises(ValueError, match="Please supply this string manually"):
        _infer_decoder_layers_attr_name(object())
