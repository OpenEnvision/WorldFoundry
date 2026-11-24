"""Model × benchmark matrix suite orchestrator.

Expands model-zoo and benchmark-zoo selections into a grid of cells, runs
:func:`run_model_benchmark` per cell, and aggregates index and comparison dashboards.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from worldfoundry.evals.utils import (
    jsonable,
    write_json,
    write_text,
)

from .suite_cells import _coerce_request, _fingerprint_request, _load_previous_suite_cells, _selected_model_ids
from .suite_reports import _suite_summary, _write_suite_artifacts, build_markdown_suite_report
from .suite_requests import (
    MODEL_BENCHMARK_SUITE_RESULT_SCHEMA_VERSION,
    MODEL_BENCHMARK_SUITE_SCHEMA_VERSION,
    ModelBenchmarkSuiteRequest,
    ModelBenchmarkSuiteResult,
)
from .suite_workers import (
    _resolve_model_worker_plan,
    _run_model_cells,
    _run_models_in_spawn_workers,
    _worker_annotated_cells,
)


def run_model_benchmark_suite(
    request: ModelBenchmarkSuiteRequest | Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> ModelBenchmarkSuiteResult:
    """Execute or plan a model × benchmark matrix sweep.

    Execution flow:

    * Expand selected models × benchmarks (or suite presets).
    * Skip or block incompatible artifact pairings.
    * Run or resume each cell via :func:`run_model_benchmark`.
    * Aggregate index, comparison, and ``suite_manifest.json``.
    """
    suite_request = _coerce_request(request, kwargs)
    if suite_request.metric_batch_size < 1:
        raise ValueError("metric_batch_size must be positive")
    if suite_request.contract_fixture and suite_request.mode != "contract":
        suite_request = replace(suite_request, mode="contract")
    root = Path(suite_request.output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    run_fingerprint = _fingerprint_request(suite_request)
    previous_cells = _load_previous_suite_cells(root) if suite_request.resume else {}
    selected_model_ids = _selected_model_ids(suite_request)
    if not selected_model_ids:
        raise ValueError(
            "model-benchmark suites require at least one model id. Pass --model, choose a suite preset "
            "that declares model_ids, or set contract_fixture=True to run benchmark contract validation cells."
        )

    worker_plan = _resolve_model_worker_plan(suite_request, model_count=len(selected_model_ids))
    if worker_plan.use_spawn_workers:
        cells = _run_models_in_spawn_workers(
            suite_request,
            root=root,
            run_fingerprint=run_fingerprint,
            previous_cells=previous_cells,
            model_ids=selected_model_ids,
            worker_plan=worker_plan,
        )
    else:
        cells = []
        parent_pid = os.getpid()
        inherited_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        for model_id in selected_model_ids:
            model_cells = _run_model_cells(
                suite_request,
                root=root,
                run_fingerprint=run_fingerprint,
                previous_cells=previous_cells,
                model_id=model_id,
            )
            cells.extend(
                _worker_annotated_cells(
                    model_cells,
                    pid=parent_pid,
                    cuda_visible_devices=inherited_devices,
                )
            )

    summary = _suite_summary(cells, execute=suite_request.execute)
    worker_pids = sorted({int(cell["model_worker_pid"]) for cell in cells if cell.get("model_worker_pid") is not None})
    scheduler = worker_plan.to_dict()
    scheduler["worker_pids"] = worker_pids
    summary.update(
        {
            "model_workers_requested": worker_plan.requested_workers,
            "model_workers_used": worker_plan.worker_count if suite_request.execute else 0,
            "parallel_model_execution": worker_plan.use_spawn_workers and worker_plan.worker_count > 1,
            "worker_cuda_device_groups": scheduler["cuda_device_groups"],
        }
    )
    artifacts = _write_suite_artifacts(root, cells)
    failed = int(summary["failed"])
    skipped = int(summary["skipped"])
    exit_code = 1 if failed or (suite_request.fail_on_skipped and skipped) else 0
    if not suite_request.execute and not failed:
        status = "planned"
    elif failed:
        status = "failed"
    elif skipped and not any(cell["status"] == "succeeded" for cell in cells):
        status = "skipped"
    else:
        status = "succeeded"
    payload = {
        "schema_version": MODEL_BENCHMARK_SUITE_SCHEMA_VERSION,
        "status": status,
        "exit_code": exit_code,
        "run_fingerprint": run_fingerprint,
        "request": jsonable(asdict(suite_request)),
        "scheduler": scheduler,
        "summary": summary,
        "cells": cells,
        "artifacts": artifacts,
    }
    manifest_path = root / "suite_manifest.json"
    report_path = root / "suite_report.md"
    write_json(manifest_path, payload)
    write_text(report_path, build_markdown_suite_report(payload))
    return ModelBenchmarkSuiteResult(
        schema_version=MODEL_BENCHMARK_SUITE_RESULT_SCHEMA_VERSION,
        status=status,
        exit_code=exit_code,
        run_fingerprint=run_fingerprint,
        output_dir=root,
        suite_manifest_path=manifest_path,
        suite_report_path=report_path,
        summary=summary,
        cells=tuple(cells),
        artifacts=artifacts,
    )
