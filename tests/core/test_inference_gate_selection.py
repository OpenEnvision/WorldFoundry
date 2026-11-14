# SPDX-FileCopyrightText: 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0
"""Model-runtime contracts run in the tensor gate, not the Torch-free gate."""

from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path

import pytest
from packaging.requirements import Requirement

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

pytestmark = pytest.mark.unit
_ROOT = Path(__file__).resolve().parents[2]
_TENSOR_CONTRACTS = (
    "tests/evaluation/execution/test_videobench_text_alignment.py",
    "tests/evaluation/execution/test_wbench_sam2_tracking_summary.py",
    "tests/evaluation/execution/test_vbench_sampling.py",
    "tests/evaluation/execution/test_vbench_metadata_errors.py",
    "tests/evaluation/execution/test_t2vworldbench_evaluator_imports.py",
    "tests/evaluation/execution/test_t2v_compbench_mllm_metric_imports.py",
    "tests/evaluation/execution/test_fetv_blip_modes.py",
    "tests/evaluation/execution/test_fetv_video_frames.py",
    "tests/evaluation/execution/test_vmbench_qalign_helpers.py",
    "tests/evaluation/execution/test_vmbench_cas_writer.py",
    "tests/evaluation/metrics/test_fvd_spatial_squeeze.py",
    "tests/base_models/test_transnetv2_unsupported_options.py",
)


def _selected_tests(target: str) -> set[str]:
    result = subprocess.run(
        ["make", "--no-print-directory", "--dry-run", target, "PYTHON=python"],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    tokens = shlex.split(result.stdout)
    selected = {tokens[index + 1] for index, token in enumerate(tokens) if token == "--test"}
    if "--include-public-defaults" in tokens:
        project = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        selected.update(project["tool"]["pytest"]["ini_options"]["testpaths"])
    return selected


@pytest.fixture(scope="module")
def gate_selections() -> tuple[set[str], set[str]]:
    return _selected_tests("test-infer"), _selected_tests("test-infer-tensors")


@pytest.mark.parametrize("path", _TENSOR_CONTRACTS)
def test_tensor_contracts_keep_required_execution_outside_public_gate(
    gate_selections: tuple[set[str], set[str]], path: str
) -> None:
    public, tensors = gate_selections
    assert path not in public
    assert not any(str(parent) in public for parent in Path(path).parents)
    assert path in tensors


def test_public_extra_supplies_the_required_artifact_readers() -> None:
    project = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = project["project"]["dependencies"] + project["project"]["optional-dependencies"]["test"]
    names = {Requirement(dependency).name.lower() for dependency in dependencies}
    assert {"av", "protobuf", "soundfile"} <= names
