"""The public Self-Forcing and Causal-Forcing IDs resolve to the native pipelines."""

from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from worldfoundry.base_models.diffusion_model.recipes.registry import default_native_diffusion_registry
from worldfoundry.pipelines import native_diffusion
from worldfoundry.studio.inference.catalog import find_entry
from worldfoundry.studio.inference.catalog import resolve as catalog_resolve
from worldfoundry.studio.inference.catalog.models import world_models
from worldfoundry.zoo.catalog.registry import discover_model_registry

NATIVE_TARGETS = {
    "self-forcing": "worldfoundry.pipelines.forcing.pipeline_native_forcing:NativeSelfForcingPipeline",
    "causal-forcing": "worldfoundry.pipelines.forcing.pipeline_native_forcing:NativeCausalForcingPipeline",
}
OFFICIAL_RUNTIME_OPTIONS = {
    "use_ema",
    "runtime_root",
    "repo_root",
    "config_path",
    "python_executable",
    "wan_models_root",
    "wan_root",
    "num_output_frames",
    "save_with_index",
    "i2v",
}


def _binding_class(model_id: str) -> Any:
    module_name, class_name = discover_model_registry().get(model_id).pipeline_target.split(":")
    return getattr(importlib.import_module(module_name), class_name)


@pytest.mark.parametrize("model_id", sorted(NATIVE_TARGETS))
def test_forcing_ids_bind_the_native_pipelines(model_id: str) -> None:
    registry = discover_model_registry()
    for name in (model_id, model_id.replace("-", "_")):
        assert registry.get(name).pipeline_target == NATIVE_TARGETS[model_id]
    entry = find_entry(model_id)
    assert f"{entry.module_path}:{entry.class_name}" == NATIVE_TARGETS[model_id]


def test_forcing_studio_entries_declare_native_parameters() -> None:
    self_forcing, causal_forcing = find_entry("self-forcing"), find_entry("causal-forcing")

    for entry in (self_forcing, causal_forcing):
        assert entry.supports_from_pretrained and entry.supports_stream
        assert entry.stream_params == ("prompt",)
        assert {"native", "prompt-scheduled"} <= set(entry.tags)
        assert "official-runtime" not in entry.tags
        assert {"checkpoint_path", "checkpoint_overrides", "torch_dtype", "attention_backend"} <= set(entry.load_params)
        assert {"prompt", "num_frames", "height", "width", "seed", "fps"} <= set(entry.call_params)
        declared = {*entry.load_params, *entry.call_params, *entry.default_call_kwargs, *entry.default_load_kwargs}
        assert not declared & OFFICIAL_RUNTIME_OPTIONS
        assert entry.default_call_kwargs == {
            "height": 480,
            "width": 832,
            "num_frames": 81,
            "fps": 16,
            "seed": 0,
            "return_dict": True,
        }
    assert "kv_window" in self_forcing.load_params
    assert "variant" in causal_forcing.load_params
    assert "images" not in self_forcing.call_params
    assert "images" in causal_forcing.call_params


def test_forcing_default_refs_are_staged_release_directories(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for name in ("gdhe17--Self-Forcing", "zhuhz22--Causal-Forcing"):
        (tmp_path / name).mkdir()
    monkeypatch.setenv("WORLDFOUNDRY_CKPT_DIR", str(tmp_path))

    assert world_models._self_forcing_default_ref() == str(tmp_path / "gdhe17--Self-Forcing")
    assert world_models._causal_forcing_default_ref() == str(tmp_path / "zhuhz22--Causal-Forcing")

    monkeypatch.setattr(catalog_resolve, "_checkpoint_model_ref", lambda *names: str(tmp_path / "missing"))
    assert world_models._self_forcing_default_ref() == ""
    assert world_models._causal_forcing_default_ref() == ""


@pytest.mark.parametrize(
    ("model_id", "options", "requested", "recipe"),
    [
        ("self-forcing", {}, "self-forcing", "self-forcing"),
        ("causal-forcing", {}, "causal-forcing", "causal-forcing-chunkwise"),
        (
            "causal-forcing",
            {"variant": "causal-forcing-framewise"},
            "causal-forcing-framewise",
            "causal-forcing-framewise",
        ),
    ],
)
def test_forcing_bindings_load_the_native_recipe_from_the_release_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    model_id: str,
    options: dict[str, str],
    requested: str,
    recipe: str,
) -> None:
    captured: dict[str, Any] = {}

    def load(recipe_id: str, **kwargs: Any) -> Any:
        captured.update(model_id=recipe_id, **kwargs)
        return SimpleNamespace(model_id=default_native_diffusion_registry().resolve(recipe_id).model_id)

    monkeypatch.setattr(native_diffusion.NativeDiffusionPipeline, "from_pretrained", load)

    pipeline = _binding_class(model_id).from_pretrained(
        model_path=str(tmp_path), model_id=model_id, device="cpu", **options
    )

    assert captured["model_id"] == requested
    assert captured["checkpoint_overrides"] == {"dit": str(tmp_path)}
    assert pipeline.model_id == recipe
