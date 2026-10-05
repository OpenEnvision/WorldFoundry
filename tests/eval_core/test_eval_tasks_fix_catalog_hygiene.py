"""CPU-only regression tests for catalog hygiene fixes ET-17/ET-18/ET-19.

- ET-17: unrecognized free-text catalog statuses used to degrade to
  "unknown"/"planned" with no signal; they now emit a warning (once per value).
- ET-18: `benchmark_catalog_ids` only honored the `id:` key while the path
  index honored both `benchmark_id:` and `id:`; both now share one extractor.
- ET-19: malformed catalog shards were skipped with a bare `except Exception`;
  the skip is now typed and logged with the shard path.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from worldfoundry.evaluation.tasks.catalog import schema as catalog_schema
from worldfoundry.evaluation.tasks.catalog.benchmark_catalog import (
    _catalog_benchmark_path_index,
    benchmark_catalog_ids,
)
from worldfoundry.evaluation.tasks.catalog.schema import (
    _normalize_integration_status,
    _normalize_source_status,
)


@pytest.fixture(autouse=True)
def _reset_status_warning_dedupe() -> None:
    catalog_schema._WARNED_UNRECOGNIZED_STATUSES.clear()
    yield
    catalog_schema._WARNED_UNRECOGNIZED_STATUSES.clear()


def test_unknown_source_status_warns_and_degrades(caplog: pytest.LogCaptureFixture) -> None:
    # Use a synthetic value: real catalog statuses (e.g.
    # confirmed_official_code_in_github) are registered aliases now and no
    # longer exercise the unknown-status path.
    with caplog.at_level(logging.WARNING, logger="worldfoundry.evaluation.tasks.catalog.schema"):
        assert _normalize_source_status("definitely_not_a_registered_source_status") == "unknown"
    assert any("definitely_not_a_registered_source_status" in record.message for record in caplog.records)

    # Deduped: the same value does not warn twice.
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="worldfoundry.evaluation.tasks.catalog.schema"):
        assert _normalize_source_status("definitely_not_a_registered_source_status") == "unknown"
    assert not caplog.records


def test_known_source_aliases_do_not_warn(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="worldfoundry.evaluation.tasks.catalog.schema"):
        assert _normalize_source_status("confirmed_official_code_and_data_in_github") == "open_source"
        assert _normalize_source_status("open_source") == "open_source"
        assert _normalize_source_status(None) == "unknown"
        assert _normalize_source_status("paper_only") == "unknown"
    assert not caplog.records


def test_unknown_integration_status_warns_and_degrades(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="worldfoundry.evaluation.tasks.catalog.schema"):
        assert _normalize_integration_status("integrated_v2_totally_new") == "planned"
    assert any("integrated_v2_totally_new" in record.message for record in caplog.records)

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="worldfoundry.evaluation.tasks.catalog.schema"):
        assert _normalize_integration_status("integrated") == "integrated"
        assert _normalize_integration_status("blocked_on_assets") == "blocked"
        assert _normalize_integration_status("pending_review") == "planned"
        assert _normalize_integration_status(None) == "planned"
    assert not caplog.records


def test_benchmark_id_only_shards_appear_in_id_universe(tmp_path: Path) -> None:
    shard_dir = tmp_path / "catalog"
    shard_dir.mkdir()
    (shard_dir / "alpha.yaml").write_text("id: alpha-bench\nname: Alpha\n", encoding="utf-8")
    (shard_dir / "beta.yaml").write_text("benchmark_id: beta-bench\nname: Beta\n", encoding="utf-8")

    ids = benchmark_catalog_ids(str(shard_dir))
    assert ids == ("alpha-bench", "beta-bench")

    index = _catalog_benchmark_path_index(str(shard_dir))
    assert set(index) == {"alpha-bench", "beta-bench"}


def test_malformed_shard_is_skipped_with_warning(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    shard_dir = tmp_path / "catalog"
    shard_dir.mkdir()
    (shard_dir / "good.yaml").write_text("id: good-bench\nname: Good\n", encoding="utf-8")
    bad_path = shard_dir / "bad.yaml"
    bad_path.write_text("id: [unclosed\n  nope: {", encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger="worldfoundry.evaluation.tasks.catalog.benchmark_catalog"):
        index = _catalog_benchmark_path_index(str(shard_dir))

    assert set(index) == {"good-bench"}
    assert any("bad.yaml" in record.message for record in caplog.records), "skip must name the malformed shard"
