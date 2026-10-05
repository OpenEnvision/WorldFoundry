from __future__ import annotations

from pathlib import Path

import pytest

from worldfoundry.evaluation.tasks.catalog.benchmark_catalog import load_benchmark_catalog_shard_entries
from worldfoundry.evaluation.tasks.execution.framework.normalizers import NormalizerSpecError, apply_normalizer, parse_normalizer


def test_benchmark_zoo_normalizers_apply_common_specs() -> None:
    assert apply_normalizer(None, 0.25) == 0.25
    assert apply_normalizer("identity", 2.5) == 2.5
    assert apply_normalizer("identity:0:1", 0.75) == 0.75
    assert apply_normalizer("scale_max:5", 2.5) == 0.5
    assert apply_normalizer("scale_max:5", 8.0) == 1.0
    assert apply_normalizer("percent_or_fraction_to_unit", 80) == 0.8
    assert apply_normalizer("percent_or_fraction_to_unit", 0.8) == 0.8
    assert apply_normalizer("official_vbench_minmax:0.2:0.6", 0.4) == pytest.approx(0.5)


@pytest.mark.parametrize(
    "spec",
    [
        "unknown",
        "scale_max",
        "scale_max:0",
        "percent_or_fraction_to_unit:100",
        "official_vbench_minmax:1:1",
        "official_vbench_minmax:0",
    ],
)
def test_benchmark_zoo_normalizers_reject_malformed_specs(spec: str) -> None:
    with pytest.raises(NormalizerSpecError):
        parse_normalizer(spec)


def test_benchmark_zoo_manifest_normalizers_are_parseable() -> None:
    for entry in load_benchmark_catalog_shard_entries("video"):
        for metric in entry.metrics:
            parse_normalizer(metric.normalizer)
