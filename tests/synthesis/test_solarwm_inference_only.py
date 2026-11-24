# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""SolarWM retains inference loaders and cache allocators without training closures."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from worldfoundry.synthesis.visual_generation.solarwm.solarwm_runtime.solarwm.backends.wan22.runtime import stage2

pytestmark = pytest.mark.unit
ROOT = Path(stage2.__file__).resolve().parents[3]


@pytest.mark.parametrize(
    ("relative_path", "removed_names"),
    [
        ("backends/minimax_h3/codec.py", {"H3PreencodedSample", "validate_preencoded_sample", "assemble_model_inputs"}),
        ("backends/ltx25/runtime.py", {"TrainingRuntimeBundle", "PreencodeRuntimeBundle"}),
        ("backends/ltx25/artifact.py", {"PreencodedSample", "TensorArtifact", "artifact_contract"}),
        ("backends/wan22/sgf.py", {"sgf_student_loss", "sgf_critic_flow_loss", "student_update_steps"}),
        (
            "backends/wan22/runtime/stage2.py",
            {"save_stage2_checkpoint", "_local_rng_state", "_validate_stage2_progress"},
        ),
    ],
)
def test_solarwm_removes_training_and_preencoding_definitions(relative_path: str, removed_names: set[str]) -> None:
    tree = ast.parse((ROOT / relative_path).read_text())
    definitions = {node.name for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))}
    assert definitions.isdisjoint(removed_names)


@pytest.mark.parametrize("relative_path", ["preencode/shards.py", "runtime/safe_state.py", "runtime/ema.py"])
def test_solarwm_removes_training_only_modules(relative_path: str) -> None:
    assert not (ROOT / relative_path).exists()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_solarwm_stage2_cache_allocators_preserve_inference_layout(dtype: torch.dtype) -> None:
    model = SimpleNamespace(blocks=[object(), object()], num_heads=2, dim=8)
    adapter: Any = SimpleNamespace(
        _root=stage2.Wan5BStage2Runtime._root,
        student=SimpleNamespace(module=model),
        _kv_cache_allocator=None,
        _crossattn_cache_allocator=None,
        config={"model": {"local_attn_size": 3, "frame_sequence_length": 4}},
    )
    caches = stage2.Wan5BStage2Runtime.allocate_kv_cache(adapter, 2, dtype=dtype, device="cpu")
    cross = stage2.Wan5BStage2Runtime.allocate_crossattn_cache(adapter, 2, dtype=dtype, device="cpu")
    assert len(caches) == len(cross) == 2
    for cache in caches:
        for key in ("k", "v"):
            assert cache[key].shape == (2, 12, 2, 4)
            assert cache[key].dtype == dtype
            assert torch.count_nonzero(cache[key]).item() == 0
    metadata = caches[0]["_fused_prope_camera_metadata"]
    assert metadata["viewmats"].shape == (2, 12, 4, 4)
    assert metadata["K"].shape == (2, 12, 3, 3)
    assert metadata["viewmats"].dtype == metadata["K"].dtype == torch.float32
    for cache in cross:
        assert cache["k"].shape == cache["v"].shape == (2, 512, 2, 4)
        assert cache["k"].dtype == cache["v"].dtype == dtype
        assert cache["is_init"] is False


def test_solarwm_stage2_keeps_only_shared_inference_cache_methods() -> None:
    tree = ast.parse(Path(stage2.__file__).read_text())
    runtime = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Wan5BStage2Runtime")
    assert {node.name for node in runtime.body if isinstance(node, ast.FunctionDef)} == {
        "_root",
        "allocate_kv_cache",
        "allocate_crossattn_cache",
    }


def test_solarwm_main_module_import_does_not_dispatch_cli_arguments() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import worldfoundry.synthesis.visual_generation.solarwm.solarwm_runtime.solarwm.__main__",
            "not-a-solarwm-command",
        ],
        check=True,
        timeout=60,
    )


def test_solarwm_h3_validation_schedule_requires_dp_wave_divisibility() -> None:
    import json

    from worldfoundry.synthesis.visual_generation.solarwm.solarwm_runtime.solarwm.backends.minimax_h3.runtime import (
        _validation_schedule,
    )
    from worldfoundry.synthesis.visual_generation.solarwm.solarwm_runtime.solarwm.errors import (
        BackendContractError,
    )
    from worldfoundry.synthesis.visual_generation.solarwm.solarwm_runtime.solarwm.runtime.topology import Topology

    cases = json.loads(
        Path(__file__).resolve().parents[1].joinpath("manual", "geometry_regression_cases.json").read_text()
    )
    sample_count = cases["solarwm-h3-33b-checkpoint-replay"]["call"]["overrides"]["validation.sample_count"]
    topology = Topology(raw_world_size=8, raw_rank=0, local_world_size=8, local_rank=0, sp_size=2)

    assert sample_count == 4
    assert _validation_schedule({"sample_count": sample_count}, topology) == (4, 1)
    with pytest.raises(BackendContractError, match="complete logical-DP waves"):
        _validation_schedule({"sample_count": 1}, topology)
