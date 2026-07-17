"""Public facade for :mod:`worldfoundry.evaluation.reporting`.

Re-exports scorecard, manifest, index, comparison, and validation builders so
callers can ``from worldfoundry.evaluation.reporting import …``.

Sections:

* **Schema constants** — version strings for each artifact type.
* **Re-exports** — run manifest, scorecard, index, browser, comparison helpers.
* **Runtime evidence** — scorecard flag inspection utilities.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .comparison_identity import (
    COMPARISON_IDENTITY_SCHEMA_VERSION,
    build_comparison_identity,
    compare_identities,
    comparison_identity_from_summary,
)

from .run_comparison import (
    RUN_COMPARISON_SCHEMA_VERSION,
    build_markdown_comparison,
    build_run_comparison,
    write_run_comparison,
)
from .run_index import (
    RUN_INDEX_SCHEMA_VERSION,
    build_markdown_run_index,
    build_run_index,
    discover_run_summaries,
    load_run_index,
    run_paths_from_index,
    select_run_index_rows,
    write_run_index,
)
from .run_report import (
    RUN_SUMMARY_SCHEMA_VERSION,
    build_markdown_report,
    build_run_summary,
    load_run_summary,
    write_run_report_artifacts,
)
from .run_browser import RUN_BROWSER_SCHEMA_VERSION, build_run_browser_html, write_run_browser
from .run_manifest import (
    ENVIRONMENT_SCHEMA_VERSION,
    ENV_REQUIREMENTS_SCHEMA_VERSION,
    RUN_MANIFEST_SCHEMA_VERSION,
    build_env_requirements,
    build_environment,
    build_run_manifest,
    redact_secrets,
    write_run_manifest_artifacts,
)
from .scorecard import SCORECARD_SCHEMA_VERSION, build_scorecard, write_scorecard
from .validation import (
    CONTRACT_ARTIFACT_KIND_CHOICES,
    CONTRACT_VALIDATION_SCHEMA_VERSION,
    build_markdown_contract_validation,
    normalize_contract_artifact_kind,
    validate_contract_file,
    validate_contract_paths,
)


# ---------------------------------------------------------------------------
# Runtime evidence inspection
# ---------------------------------------------------------------------------

RUNTIME_EVIDENCE_KEYS = (
    "official_benchmark_verified",
    "integration_evidence",
    "normalization_ok",
    "official_results_imported",
)


def inspect_scorecard_runtime_flags(path: str | Path) -> dict[str, Any]:
    """Extract runtime-evidence booleans from a ``scorecard.json`` file."""
    scorecard_path = Path(path)
    if not scorecard_path.is_file():
        return {"path": str(scorecard_path), "found": False}
    try:
        scorecard = json.loads(scorecard_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"path": str(scorecard_path), "found": False, "error": f"could not inspect scorecard: {exc}"}

    result: dict[str, Any] = {"path": str(scorecard_path), "found": True}
    if isinstance(scorecard, Mapping):
        for key in RUNTIME_EVIDENCE_KEYS:
            value = scorecard.get(key)
            if isinstance(value, bool):
                result[key] = value
        cv = scorecard.get("worldfoundry_contract_validation_evidence")
        if isinstance(cv, bool):
            result["worldfoundry_contract_validation_evidence"] = cv
        run = scorecard.get("run")
        if isinstance(run, Mapping):
            result["run_status"] = run.get("status")
            result["run_command_present"] = run.get("command") is not None
        evaluation = scorecard.get("evaluation")
        if isinstance(evaluation, Mapping):
            result["evaluation_kind"] = evaluation.get("kind")
        validation = scorecard.get("validation")
        if isinstance(validation, Mapping):
            for key in ("normalizer_only", "official_runtime_executed", "official_results_imported"):
                value = validation.get(key)
                if isinstance(value, bool):
                    result[f"validation_{key}"] = value
    return result


def has_official_runtime_evidence(flags: Mapping[str, Any]) -> bool:
    """Return True when flags indicate genuine official runtime execution."""
    if flags.get("worldfoundry_contract_validation_evidence") is True:
        return False
    return flags.get("official_benchmark_verified") is True and flags.get("integration_evidence") is True

__all__ = [
    "CONTRACT_VALIDATION_SCHEMA_VERSION",
    "CONTRACT_ARTIFACT_KIND_CHOICES",
    "COMPARISON_IDENTITY_SCHEMA_VERSION",
    "ENVIRONMENT_SCHEMA_VERSION",
    "ENV_REQUIREMENTS_SCHEMA_VERSION",
    "RUNTIME_EVIDENCE_KEYS",
    "RUN_COMPARISON_SCHEMA_VERSION",
    "RUN_BROWSER_SCHEMA_VERSION",
    "RUN_INDEX_SCHEMA_VERSION",
    "RUN_MANIFEST_SCHEMA_VERSION",
    "RUN_SUMMARY_SCHEMA_VERSION",
    "SCORECARD_SCHEMA_VERSION",
    "build_env_requirements",
    "build_comparison_identity",
    "build_environment",
    "build_markdown_contract_validation",
    "build_markdown_comparison",
    "build_markdown_run_index",
    "build_markdown_report",
    "build_run_browser_html",
    "build_run_comparison",
    "build_run_index",
    "build_run_manifest",
    "build_run_summary",
    "discover_run_summaries",
    "has_official_runtime_evidence",
    "inspect_scorecard_runtime_flags",
    "load_run_index",
    "load_run_summary",
    "normalize_contract_artifact_kind",
    "redact_secrets",
    "run_paths_from_index",
    "select_run_index_rows",
    "validate_contract_file",
    "validate_contract_paths",
    "write_run_comparison",
    "write_run_browser",
    "write_run_index",
    "build_scorecard",
    "compare_identities",
    "comparison_identity_from_summary",
    "write_run_manifest_artifacts",
    "write_run_report_artifacts",
    "write_scorecard",
]
