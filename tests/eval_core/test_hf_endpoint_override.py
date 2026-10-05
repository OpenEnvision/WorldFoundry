"""Hugging Face mirrors must remain an explicit operator choice."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from worldfoundry.runtime.env import HF_ENDPOINT_OVERRIDE_ENV, apply_hf_endpoint_override


def _source(module_name: str) -> str:
    spec = importlib.util.find_spec(module_name)
    assert spec is not None and spec.origin
    return Path(spec.origin).read_text(encoding="utf-8")


def test_no_endpoint_by_default() -> None:
    env: dict[str, str] = {}
    assert apply_hf_endpoint_override(env) is None
    assert "HF_ENDPOINT" not in env


def test_inherited_endpoint_is_preserved() -> None:
    env = {"HF_ENDPOINT": "https://huggingface.example"}
    assert apply_hf_endpoint_override(env) == "https://huggingface.example"
    assert env["HF_ENDPOINT"] == "https://huggingface.example"


def test_opt_in_override_wins_over_inherited_endpoint() -> None:
    env = {
        HF_ENDPOINT_OVERRIDE_ENV: " https://mirror.example ",
        "HF_ENDPOINT": "https://huggingface.example",
    }
    assert apply_hf_endpoint_override(env) == "https://mirror.example"
    assert env["HF_ENDPOINT"] == "https://mirror.example"


def test_blank_override_is_ignored() -> None:
    env = {HF_ENDPOINT_OVERRIDE_ENV: "   "}
    assert apply_hf_endpoint_override(env) is None
    assert "HF_ENDPOINT" not in env


def test_studio_dispatch_has_no_default_third_party_mirror() -> None:
    source = _source("worldfoundry.studio.inference.dispatch")
    assert "hf-mirror.com" not in source
    assert "apply_hf_endpoint_override(env)" in source


def test_prepare_script_has_no_default_third_party_mirror() -> None:
    script = Path(__file__).resolve().parents[2] / "scripts" / "inference" / "prepare_model_infer.sh"
    source = script.read_text(encoding="utf-8")
    assert "HF_ENDPOINT:-https://hf-mirror.com" not in source
    assert 'WORLDFOUNDRY_HF_ENDPOINT:-${HF_ENDPOINT:-}' in source
