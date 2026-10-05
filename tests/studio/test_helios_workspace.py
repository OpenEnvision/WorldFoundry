from __future__ import annotations

import pytest

# This test module imports worldfoundry code that requires the optional
# "fastapi" dependency at import time; skip when it is unavailable.
pytest.importorskip("fastapi")

import json

from worldfoundry.runtime.inference_catalog import get_model_inference_spec
from worldfoundry.studio.inference.catalog import find_entry
from worldfoundry.studio.serving.workspace import JobCreateRequest, _inference_run_kwargs, _model_payload
from worldfoundry.synthesis.visual_generation.official_video_runtime import OfficialVideoRuntime


def test_helios_workspace_exposes_each_checkpoint_once() -> None:
    entry = find_entry("helios")
    payload = _model_payload(entry)

    assert payload["template_id"] == "text-video"
    assert payload["workload_type"] == "t2v"
    assert payload["default_variant_id"] == "helios-distilled"
    assert [variant["variant_id"] for variant in payload["variants"]] == [
        "helios-distilled",
        "helios-base",
        "helios-mid",
    ]
    assert len({variant["model_ref"] for variant in payload["variants"]}) == 3
    assert all(variant["variant_id"] != "default" for variant in payload["variants"])


def test_helios_workspace_routes_variants_to_the_official_runtime() -> None:
    expected = {
        "helios-distilled": ("Helios-Distilled", "distilled"),
        "helios-base": ("Helios-Base", "base"),
        "helios-mid": ("Helios-Mid", "mid"),
    }

    for variant_id, (checkpoint_name, runtime_variant) in expected.items():
        _, run_kwargs = _inference_run_kwargs(
            JobCreateRequest(model_id="helios", variant_id=variant_id, prompt="Helios demo"),
            validate=True,
        )
        call_kwargs = json.loads(run_kwargs["call_kwargs_text"])
        assert run_kwargs["model_ref"].endswith(checkpoint_name)
        assert call_kwargs["variant"] == runtime_variant
        assert call_kwargs["sample_type"] == "t2v"
        assert call_kwargs["num_frames"] == 240
        assert call_kwargs["fps"] == 24


def test_helios_aliases_do_not_create_duplicate_variants() -> None:
    spec = get_model_inference_spec("helios")

    assert spec is not None
    assert spec.variant("distilled").variant_id == "helios-distilled"
    assert spec.variant("base").variant_id == "helios-base"
    assert spec.variant("mid").variant_id == "helios-mid"
    assert len(spec.variants) == 3


def test_helios_runtime_does_not_shadow_the_external_kernels_package() -> None:
    runtime = OfficialVideoRuntime.from_model_id("helios")

    assert runtime.runtime["cwd"].endswith("WorldFoundry")
    assert runtime.runtime["prepend_repo_root_to_pythonpath"] is False
    assert "--master_port={master_port}" in runtime.runtime["command"]
