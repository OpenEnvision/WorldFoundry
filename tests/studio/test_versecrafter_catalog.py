from __future__ import annotations

import numpy as np
import pytest

import worldfoundry.studio.inference.catalog as catalog
from worldfoundry.studio.inference.catalog import (
    _versecrafter_default_load_kwargs,
    _versecrafter_default_ref,
)
from worldfoundry.pipelines.versecrafter.pipeline_versecrafter import VerseCrafterPipeline
from worldfoundry.synthesis.visual_generation.versecrafter.worldfoundry_runtime import (
    VerseCrafterRuntime,
)


def _checkpoint_tree(tmp_path):
    checkpoint = tmp_path / "ckpts" / "TencentARC--VerseCrafter"
    base = tmp_path / "ckpts" / "Wan-AI--Wan2.1-T2V-14B"
    moge = tmp_path / "ckpts" / "moge-2-vitl-normal"
    for path in (checkpoint, base, moge):
        path.mkdir(parents=True)
    return checkpoint, base, moge


def test_versecrafter_catalog_resolves_all_local_checkpoints(tmp_path, monkeypatch) -> None:
    checkpoint, base, moge = _checkpoint_tree(tmp_path)
    monkeypatch.setenv("WORLDFOUNDRY_CKPT_DIR", str(tmp_path / "ckpt"))
    monkeypatch.setenv("WORLDFOUNDRY_HFD_ROOT", str(tmp_path / "hfd"))

    assert _versecrafter_default_ref() == str(checkpoint)
    assert _versecrafter_default_load_kwargs() == {
        "base_model_path": str(base),
        "moge_model_path": str(moge),
    }


def test_versecrafter_catalog_uses_public_hub_fallbacks(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("WORLDFOUNDRY_CKPT_DIR", str(tmp_path / "ckpt"))
    monkeypatch.setenv("WORLDFOUNDRY_HFD_ROOT", str(tmp_path / "hfd"))
    monkeypatch.setattr(catalog, "_cache_candidates", lambda *names: [])

    assert _versecrafter_default_ref() == "TencentARC/VerseCrafter"
    assert _versecrafter_default_load_kwargs() == {
        "base_model_path": "Wan-AI/Wan2.1-T2V-14B",
        "moge_model_path": "Ruicheng/moge-2-vitl-normal",
    }


def test_versecrafter_default_trajectory_is_deterministic(tmp_path) -> None:
    first = VerseCrafterRuntime._write_default_trajectory(tmp_path / "first.npz", 5)
    second = VerseCrafterRuntime._write_default_trajectory(tmp_path / "second.npz", 5)

    first_extrinsics = np.load(first)["extrinsics"]
    second_extrinsics = np.load(second)["extrinsics"]
    np.testing.assert_array_equal(first_extrinsics, second_extrinsics)
    assert first_extrinsics.shape == (5, 4, 4)
    assert first_extrinsics[-1, 0, 3] > first_extrinsics[0, 0, 3]


def test_versecrafter_torchrun_uses_isolated_standalone_rendezvous() -> None:
    command = VerseCrafterRuntime._torchrun_prefix("/env/bin/python")

    assert command == [
        "/env/bin/python",
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=1",
    ]


def test_versecrafter_workspace_uses_memory_safe_component_offload() -> None:
    runtime = VerseCrafterRuntime(python_executable="/env/bin/python")

    assert runtime.gpu_memory_mode == "model_cpu_offload"
    assert runtime.plan()["gpu_memory_mode"] == "model_cpu_offload"


def test_versecrafter_rejects_unknown_gpu_memory_mode() -> None:
    with pytest.raises(ValueError, match="gpu_memory_mode"):
        VerseCrafterRuntime(
            python_executable="/env/bin/python",
            gpu_memory_mode="not-a-mode",
        )


def test_versecrafter_model_ref_is_forwarded_as_checkpoint_path(monkeypatch) -> None:
    captured = {}

    def fake_from_pretrained(checkpoint, **kwargs):
        captured.update(checkpoint=checkpoint, kwargs=kwargs)
        return object()

    monkeypatch.setattr(
        "worldfoundry.pipelines.versecrafter.pipeline_versecrafter.VerseCrafterSynthesis.from_pretrained",
        fake_from_pretrained,
    )

    pipeline = VerseCrafterPipeline.from_pretrained(
        "TencentARC/VerseCrafter",
        required_components={"base_model_path": "Wan-AI/Wan2.1-T2V-14B"},
        device="cuda",
    )

    assert pipeline.synthesis_model is not None
    assert captured == {
        "checkpoint": "TencentARC/VerseCrafter",
        "kwargs": {"device": "cuda", "base_model_path": "Wan-AI/Wan2.1-T2V-14B"},
    }
