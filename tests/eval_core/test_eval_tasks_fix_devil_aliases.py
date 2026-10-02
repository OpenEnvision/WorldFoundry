"""CPU-only regression tests for the ET-13 devil-dynamics alias cleanup.

The runner config used to alias cross-benchmark generic metric names
(`subject_consistency`, `background_consistency`, `motion_smoothness`,
`naturalness`) onto DEVIL dynamics metrics. Those names never appear in the
official `devil_dynamics_results.json` (which emits canonical ids) and are
only legitimate as row keys of the quality xlsx, where the explicit
`DEVIL_KEY_TO_METRIC` table maps them. The alias table now only carries
official DEVIL result keys.
"""

from __future__ import annotations

from worldfoundry.evaluation.tasks.execution.framework import official_runner as ors
from worldfoundry.evaluation.tasks.execution.runners.devil_dynamics.run_devil_dynamics_official_runner import (
    CONFIG,
    DEVIL_KEY_TO_METRIC,
)


def test_generic_metric_names_no_longer_alias_to_dynamics_metrics() -> None:
    for generic_name in ("subject_consistency", "background_consistency", "motion_smoothness", "naturalness"):
        assert generic_name not in CONFIG.metric_aliases, generic_name
        assert ors.metric_id_from_key(generic_name, CONFIG) is None, generic_name


def test_official_json_canonical_keys_still_map() -> None:
    payload = {
        "dynamics_range": 0.41,
        "dynamics_controllability": 0.52,
        "dynamics_quality": 0.63,
        "devil_dynamics_average": 0.52,
    }
    extracted = ors.generic_extract_metrics(payload, CONFIG, "devil_dynamics_results.json")
    assert set(extracted) == {
        "dynamics_range",
        "dynamics_controllability",
        "dynamics_quality",
        "devil_dynamics_average",
    }
    assert extracted["dynamics_range"]["raw_score"] == 0.41


def test_quality_xlsx_generic_row_keys_map_through_explicit_table() -> None:
    assert DEVIL_KEY_TO_METRIC == {
        "motion_smoothness": "dynamics_controllability",
        "naturalness": "dynamics_quality",
        "subject_consistency": "dynamics_range",
        "background_consistency": "dynamics_range",
    }
