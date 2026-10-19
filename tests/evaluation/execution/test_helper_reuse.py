# SPDX-FileCopyrightText: 2026 OpenEnvision contributors
# SPDX-License-Identifier: Apache-2.0
"""Contracts for evaluation helpers shared across catalogs and benchmark runners."""

from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from worldfoundry.evaluation.api.json_contract import tuple_of_str
from worldfoundry.evaluation.tasks.execution.framework.io import coerce_unit_score, optional_float, try_float
from worldfoundry.evaluation.tasks.execution.framework.runner_common import build_video_coverage

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "module_name,name",
    [
        ("models.catalog.schema", "_tuple_of_str"),
        ("tasks.catalog.schema", "_tuple_of_str"),
        ("tasks.catalog.yaml", "_string_tuple"),
        ("tasks.catalog.dispatch", "_str_tuple"),
        ("tasks.execution.orchestration.plan", "_coerce_str_tuple"),
    ],
)
def test_catalog_tuple_coercers_share_the_same_contract(module_name: str, name: str) -> None:
    coerce = getattr(importlib.import_module("worldfoundry.evaluation." + module_name), name)
    assert coerce is tuple_of_str
    assert coerce(None) == ()
    assert coerce("score") == ("score",)
    assert coerce(["score", 2, None]) == ("score", "2", "None")


@pytest.mark.parametrize(
    "benchmark,module_name",
    [
        ("ewmbench", "ewmbench"),
        ("ipv_bench", "ipv_bench"),
        ("iworldbench", "iworldbench"),
        ("phyground", "phyground"),
        ("videophy", "videophy"),
        ("videophy2", "videophy2"),
        ("world_in_world", "world_in_world"),
    ],
)
def test_official_coverage_has_one_owner(tmp_path: Path, benchmark: str, module_name: str) -> None:
    module = importlib.import_module(
        f"worldfoundry.evaluation.tasks.execution.runners.{benchmark}.run_{module_name}_official_runner"
    )
    assert module._coverage is build_video_coverage
    for name in ["a.MP4", "unexpected.webm", "ignored.txt"]:
        (tmp_path / name).write_bytes(b"")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "b.mp4").write_bytes(b"")
    assert module._coverage({"a", "b"}, tmp_path) == {
        "expected_count": 2,
        "actual_count": 2,
        "matched_count": 1,
        "missing_count": 1,
        "unexpected_count": 1,
        "complete": False,
        "missing_ids": ["b"],
        "unexpected_ids": ["unexpected"],
    }
    assert module._coverage(set(), None)["complete"] is False
    assert module._coverage({"a"}, tmp_path)["complete"] is True


@pytest.mark.parametrize(
    "value,expected",
    [(None, None), ("", None), ("bad", None), ({}, None), ("2.5", 2.5), (-2, -2.0), (True, 1.0), (False, 0.0)],
)
def test_optional_numeric_cell_coercion(value: Any, expected: float | None) -> None:
    assert try_float(value) == expected


@pytest.mark.parametrize("value", ("safe", {"score": 4}, [0.5], float("nan"), float("inf"), float("-inf")))
@pytest.mark.parametrize("scale_max", (None, 5.0))
def test_unit_score_leaves_structured_labels_and_nonfinite_values_unscored(value: Any, scale_max: float | None) -> None:
    assert coerce_unit_score(value, scale_max=scale_max) is None


def test_strict_optional_float_still_rejects_malformed_configuration() -> None:
    with pytest.raises(ValueError):
        optional_float("bad")


def test_external_evaluator_coercion_and_failure_are_shared_without_catalog_semantics() -> None:
    from worldfoundry.evaluation.tasks.execution.runners._benchmark_metrics import evaluators, local_evaluators

    assert evaluators._tuple_of_str is local_evaluators._tuple_of_str
    assert evaluators._tuple_of_str(["score", 2, None]) == ("score", "2")
    assert evaluators._tuple_of_str(2) == ("2",)
    assert evaluators._metric_key is local_evaluators._metric_key
    assert evaluators._metric_key(" Frame_SCORE ") == "frame-score"
    assert evaluators._failure_result is local_evaluators._failure_result
    result = evaluators._failure_result(
        SimpleNamespace(sample_id="sample", metric_id="metric"), skip_reason="missing_artifact", message="video absent"
    )
    assert result.valid is False and result.coverage == 0.0
    assert result.skip_reason == "missing_artifact"
    assert result.diagnostics == {"message": "video absent"}


def test_dataset_checksum_export_uses_canonical_file_reader() -> None:
    from worldfoundry.evaluation.api.json_contract import sha256_file
    from worldfoundry.evaluation.tasks.datasets import file_sha256

    assert file_sha256 is sha256_file


def test_sadpad_registration_retains_both_public_compute_targets() -> None:
    from worldfoundry.evaluation.tasks.metrics import sadpad

    assert sadpad.compute is sadpad.compute_attribute_sad
    assert [(entry.id, entry.implementation) for entry in sadpad.METRIC_MODULES] == [
        ("attribute_sad", "worldfoundry.evaluation.tasks.metrics.sadpad:compute_attribute_sad"),
        ("attribute_pad", "worldfoundry.evaluation.tasks.metrics.sadpad:compute_attribute_pad"),
    ]
