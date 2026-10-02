from __future__ import annotations

import sys

import pytest

# This test module imports worldfoundry code that requires the optional
# "transformers" dependency at import time; skip when it is unavailable.
pytest.importorskip("transformers")

from pathlib import Path

import numpy as np
from PIL import Image

import worldfoundry.evaluation.models.pipelines as model_pipelines
from worldfoundry.evaluation.models.catalog.registry import discover_model_registry
from worldfoundry.synthesis.visual_generation.multiworld import (
    ittakestwo_runtime as runtime_module,
)
from worldfoundry.pipelines.multiworld.pipeline_multiworld_ittakestwo import (
    MultiWorldItTakesTwoPipeline,
)
from worldfoundry.synthesis.visual_generation.multiworld import (
    multiworld_ittakestwo_synthesis as synthesis_module,
)


def _example_action() -> dict[str, np.ndarray]:
    return {
        "discrete_action": np.zeros((1, 9, 2, 10), dtype=np.int64),
        "continuous_action": np.zeros((1, 9, 2, 2), dtype=np.float32),
    }


class _DummySynthesis:
    def __init__(self) -> None:
        self.calls = []

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "video": np.zeros((2, 8, 8, 3), dtype=np.uint8),
            "generated_video_path": "/tmp/fake.mp4",
        }


def test_multiworld_pipeline_process_requires_action() -> None:
    pipe = MultiWorldItTakesTwoPipeline(synthesis_model=_DummySynthesis())

    try:
        pipe.process(images=Image.new("RGB", (16, 16), "red"), action={})
    except ValueError as exc:
        assert "action" in str(exc)
    else:
        raise AssertionError("Expected missing action payload to fail.")


def test_multiworld_pipeline_call_forwards_image_action_and_env() -> None:
    synthesis = _DummySynthesis()
    pipe = MultiWorldItTakesTwoPipeline(synthesis_model=synthesis)
    image = Image.new("RGB", (16, 16), "red")
    action = _example_action()
    env_obv = np.zeros((1, 1, 2, 3, 8, 8), dtype=np.float32)

    result = pipe(
        images=image,
        action=action,
        env_obv=env_obv,
        return_dict=True,
        save_name="demo",
    )

    assert result["generated_video_path"] == "/tmp/fake.mp4"
    assert synthesis.calls
    assert synthesis.calls[0]["image"] is image
    assert synthesis.calls[0]["action"] is action
    assert synthesis.calls[0]["env_obv"] is env_obv
    assert synthesis.calls[0]["save_name"] == "demo"


def test_load_multiworld_loader_forwards_runtime_components(monkeypatch) -> None:
    captured = {}

    def fake_from_pretrained(cls, model_path=None, required_components=None, device="cuda", **kwargs):
        captured["model_path"] = model_path
        captured["required_components"] = required_components
        captured["device"] = device
        captured["kwargs"] = kwargs
        return "pipeline"

    monkeypatch.setattr(
        MultiWorldItTakesTwoPipeline,
        "from_pretrained",
        classmethod(fake_from_pretrained),
    )

    pipeline = MultiWorldItTakesTwoPipeline.from_pretrained(
        model_path="/tmp/multiworld/model.safetensors",
        required_components={
            "runtime_root": "/tmp/MultiWorld",
            "config_path": "/tmp/MultiWorld/ittakestwo/configs/inference.yaml",
            "python_executable": "/tmp/env/bin/python",
            "derive_env_obv_from_image": False,
        },
        device="cuda:3",
        num_inference_steps=12,
        inference_seed=7,
    )

    assert pipeline == "pipeline"
    assert captured["model_path"] == "/tmp/multiworld/model.safetensors"
    assert captured["required_components"] == {
        "runtime_root": "/tmp/MultiWorld",
        "config_path": "/tmp/MultiWorld/ittakestwo/configs/inference.yaml",
        "python_executable": "/tmp/env/bin/python",
        "derive_env_obv_from_image": False,
    }
    assert captured["device"] == "cuda:3"
    assert captured["kwargs"] == {
        "num_inference_steps": 12,
        "inference_seed": 7,
    }


def test_multiworld_pipeline_has_no_generic_infer_facade() -> None:
    assert not hasattr(model_pipelines, "infer_multiworld_ittakestwo_pipeline")


def test_multiworld_default_runtime_uses_in_tree_runtime(monkeypatch, tmp_path: Path) -> None:
    from worldfoundry.synthesis.visual_generation.multiworld import runtime_env

    monkeypatch.delenv("WORLDFOUNDRY_MULTIWORLD_RUNTIME_ROOT", raising=False)
    monkeypatch.delenv("MULTIWORLD_RUNTIME_ROOT", raising=False)
    monkeypatch.setenv("WORLDFOUNDRY_MODEL_SOURCE_DIR", str(tmp_path / "model_sources"))
    monkeypatch.setenv("WORLDFOUNDRY_GITHUB_REPOS_ROOT", str(tmp_path / "github_repos"))

    source = Path(runtime_env.__file__).read_text(encoding="utf-8")
    assert "official_runtime_repo_path" not in source
    assert "github_repos" not in source
    assert "WORLDFOUNDRY_MODEL_SOURCE_DIR" not in source

    assert runtime_env.default_runtime_root() == runtime_env.IN_TREE_RUNTIME_ROOT.resolve()


def test_multiworld_catalog_defaults_map_all_native_checkpoint_roles(monkeypatch, tmp_path: Path) -> None:
    from worldfoundry.studio.inference import catalog as catalog
    from worldfoundry.synthesis.visual_generation.multiworld import runtime_env

    checkpoint_root = tmp_path / "ckpts"
    checkpoint_path = checkpoint_root / "MultiWorld" / "multiworld_480p_toydata.safetensors"
    base_model_root = checkpoint_root / "Wan-AI--Wan2.2-TI2V-5B"
    vggt_root = checkpoint_root / "facebook--VGGT-1B"
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_path.write_bytes(b"multiworld")
    base_model_root.mkdir()
    vggt_root.mkdir()
    monkeypatch.setenv("WORLDFOUNDRY_CKPT_DIR", str(checkpoint_root))
    monkeypatch.setenv("WORLDFOUNDRY_HFD_ROOT", str(tmp_path / "ckpt"))

    components = catalog._multiworld_ittakestwo_default_load_kwargs()["required_components"]

    assert Path(components["runtime_root"]) == Path(runtime_env.__file__).resolve().parent
    assert Path(components["config_path"]).is_file()
    assert Path(components["checkpoint_path"]) == checkpoint_path
    assert Path(components["base_model_root"]) == base_model_root
    assert Path(components["vggt_root"]) == vggt_root


def test_model_registry_contains_multiworld_ittakestwo() -> None:
    registry = discover_model_registry()
    model = registry.get("multiworld-ittakestwo")

    assert model.family == "video_generation"
    assert model.has_loader is True
    assert model.has_infer is False


def test_multiworld_synthesis_uses_native_runtime(monkeypatch, tmp_path: Path) -> None:
    captured = {}

    class _FakeRuntime:
        def predict(self, **kwargs):
            captured["predict"] = kwargs
            return {
                "video": np.zeros((3, 8, 8, 3), dtype=np.uint8),
                "generated_video_path": str(tmp_path / "out" / "demo.mp4"),
            }

    def fake_from_pretrained(cls, **kwargs):
        captured["load"] = kwargs
        return _FakeRuntime()

    monkeypatch.setattr(
        runtime_module.MultiWorldItTakesTwoRuntime,
        "from_pretrained",
        classmethod(fake_from_pretrained),
    )
    synthesis = synthesis_module.MultiWorldItTakesTwoSynthesis(
        runtime_root=str(tmp_path),
        config_path=str(tmp_path / "inference.yaml"),
        checkpoint_path=str(tmp_path / "model.safetensors"),
        python_executable=sys.executable,
        device="cuda:1",
        derive_env_obv_from_image=True,
        num_inference_steps=21,
        inference_seed=5,
        fps=13,
    )
    image = Image.new("RGB", (16, 16), "red")
    action = _example_action()
    result = synthesis.predict(
        image=image,
        action=action,
        output_dir=str(tmp_path / "out"),
        save_name="demo",
        return_dict=True,
        show_progress=False,
    )

    assert captured["load"] == {
        "runtime_root": str(tmp_path),
        "config_path": str(tmp_path / "inference.yaml"),
        "checkpoint_path": str(tmp_path / "model.safetensors"),
        "python_executable": sys.executable,
        "device": "cuda:1",
        "derive_env_obv_from_image": True,
        "num_inference_steps": 21,
        "inference_seed": 5,
        "fps": 13,
    }
    assert captured["predict"]["image"] is image
    assert captured["predict"]["action"] is action
    assert captured["predict"]["show_progress"] is False
    assert result["video"].shape == (3, 8, 8, 3)
