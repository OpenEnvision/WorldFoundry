"""Per-model suite execution in the current process or in spawned workers."""

from __future__ import annotations

import os
import pickle
import sys
from collections import deque
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from .suite_cells import (
    _cell_dir,
    _cell_request,
    _cleanup_suite_model_runner,
    _model_outputs,
    _plan_cell,
    _reset_suite_model_runner,
    _resolve_suite_model_runner,
    _resume_cell,
    _run_cell,
    _safe_name,
    _selected_benchmarks,
    _suite_generation_memo_stats,
)
from .suite_requests import ModelBenchmarkSuiteRequest, _ModelWorkerPlan


def _resolve_model_worker_plan(
    request: ModelBenchmarkSuiteRequest,
    *,
    model_count: int,
) -> _ModelWorkerPlan:
    """Resolve process count and non-overlapping CUDA affinity without importing Torch."""
    from worldfoundry.runtime.device_pool import (
        cuda_device_discovery_source,
        default_cuda_device_groups,
        normalize_cuda_device_groups,
    )

    configured_workers = int(request.model_workers)
    if configured_workers < 1:
        raise ValueError("model_workers must be at least 1")
    explicit_groups = normalize_cuda_device_groups(request.worker_cuda_devices)
    explicit_group_widths = {len(group.split(",")) for group in explicit_groups}
    if len(explicit_group_widths) > 1:
        raise ValueError(
            "parallel model workers require equal-sized CUDA device groups; "
            "run models with different GPU counts as separate suites"
        )
    requested_workers = max(configured_workers, len(explicit_groups))
    # Even a serial multi-model suite gets one fresh process per model.  This
    # makes process exit the hard CUDA-release boundary when a third-party
    # runner retains module-level tensors or allocator state after cleanup().
    use_spawn_workers = bool(request.execute and (model_count > 1 or requested_workers > 1 or explicit_groups))

    if explicit_groups:
        available_groups: tuple[str | None, ...] = explicit_groups
        device_source = "explicit"
    elif use_spawn_workers and requested_workers > 1:
        discovered_groups = default_cuda_device_groups()
        if discovered_groups:
            available_groups = discovered_groups
            device_source = cuda_device_discovery_source()
        else:
            # CPU-only suites can still use process parallelism. GPU inference
            # should pass explicit groups when discovery is unavailable.
            available_groups = (None,) * requested_workers
            device_source = "unassigned"
    else:
        available_groups = (None,)
        device_source = "inherited"

    worker_count = min(max(model_count, 1), requested_workers, len(available_groups))
    try:
        available_cpus = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        available_cpus = os.cpu_count() or worker_count
    default_cpu_threads = max(available_cpus // worker_count, 1)
    configured_cpu_threads = os.getenv("WORLDFOUNDRY_SUITE_CPU_THREADS", "").strip()
    try:
        cpu_threads_per_worker = max(int(configured_cpu_threads), 1) if configured_cpu_threads else default_cpu_threads
    except ValueError as exc:
        raise ValueError("WORLDFOUNDRY_SUITE_CPU_THREADS must be a positive integer") from exc
    return _ModelWorkerPlan(
        requested_workers=requested_workers,
        worker_count=worker_count,
        cuda_device_groups=available_groups[:worker_count],
        device_source=device_source,
        use_spawn_workers=use_spawn_workers,
        cpu_threads_per_worker=cpu_threads_per_worker,
    )


def _run_model_cells(
    request: ModelBenchmarkSuiteRequest,
    *,
    root: Path,
    run_fingerprint: str,
    previous_cells: Mapping[str, Mapping[str, Any]],
    model_id: str,
) -> list[dict[str, Any]]:
    """Run all benchmark cells for one model while keeping its runner resident."""
    cells: list[dict[str, Any]] = []
    model_outputs, known_model, canonical_model_id = _model_outputs(model_id, request.model_manifest_dir)
    resolved_runner: Any | None = None
    runner_use_count = 0
    try:
        for benchmark in _selected_benchmarks(request):
            plan = _plan_cell(
                request,
                run_fingerprint=run_fingerprint,
                model_id=model_id,
                canonical_model_id=canonical_model_id,
                known_model=known_model,
                model_outputs=model_outputs,
                benchmark=benchmark,
            )
            base_cell = plan.to_base_cell()
            if plan.compatibility == "benchmark_unavailable" and request.skip_incompatible:
                cells.append({**base_cell, "status": "skipped", "exit_code": 0, "reason": plan.reason})
                continue
            if plan.reason and request.skip_incompatible:
                cells.append({**base_cell, "status": "skipped", "exit_code": 0, "reason": plan.reason})
                continue
            if not request.execute:
                status = "planned" if not plan.reason else "blocked"
                cells.append(
                    {
                        **base_cell,
                        "status": status,
                        "exit_code": 0 if not plan.reason else 1,
                        "reason": plan.reason,
                    }
                )
                continue
            if plan.output_artifact is None:
                cells.append({**base_cell, "status": "failed", "exit_code": 1, "reason": plan.reason})
                continue
            try:
                if request.resume:
                    resumed = _resume_cell(
                        _cell_dir(root, plan.model_id, plan.benchmark.benchmark_id),
                        previous_cells.get(plan.cell_fingerprint),
                        request=_cell_request(request, root=root, plan=plan),
                    )
                    if resumed is not None:
                        cells.append({**base_cell, **dict(resumed)})
                        continue

                runner_state: dict[str, Any] = {"reused": None, "reset": None, "memo_before": None}

                def acquire_runner(plan=plan, runner_state=runner_state) -> Any:
                    nonlocal resolved_runner, runner_use_count
                    if resolved_runner is None:
                        resolved_runner = _resolve_suite_model_runner(request, plan)
                    runner_state["memo_before"] = _suite_generation_memo_stats(resolved_runner)
                    runner_state["reused"] = runner_use_count > 0
                    if runner_state["reused"]:
                        runner_state["reset"] = _reset_suite_model_runner(resolved_runner)
                    runner_use_count += 1
                    return resolved_runner

                run_cell = _run_cell(request, root=root, plan=plan, runner_factory=acquire_runner)
                memo_before = runner_state["memo_before"]
                memo_after = _suite_generation_memo_stats(resolved_runner)
                memo_deltas = {
                    "generation_batches_reused": 0,
                    "generation_requests_reused": 0,
                    "generation_requests_executed": 0,
                }
                if memo_before is not None and memo_after is not None:
                    memo_deltas = {
                        "generation_batches_reused": max(
                            memo_after["batch_hits"] - memo_before["batch_hits"],
                            0,
                        ),
                        "generation_requests_reused": max(
                            memo_after["requests_reused"] - memo_before["requests_reused"],
                            0,
                        ),
                        "generation_requests_executed": max(
                            memo_after["requests_executed"] - memo_before["requests_executed"],
                            0,
                        ),
                    }
                run_cell = {
                    **dict(run_cell),
                    "runner_reused": runner_state["reused"],
                    "runner_state_reset": runner_state["reset"],
                    **memo_deltas,
                }
                cells.append({**base_cell, **run_cell})
            except Exception as exc:  # keep suite execution moving across cells.
                cells.append({**base_cell, "status": "failed", "exit_code": 1, "reason": str(exc)})
    finally:
        _cleanup_suite_model_runner(resolved_runner)
    return cells


def _worker_annotated_cells(
    cells: Sequence[Mapping[str, Any]],
    *,
    pid: int | None,
    cuda_visible_devices: str | None,
) -> list[dict[str, Any]]:
    """Attach process and CUDA-affinity evidence to model cell records."""
    return [
        {
            **dict(cell),
            "model_worker_pid": pid,
            "model_worker_cuda_devices": cuda_visible_devices,
        }
        for cell in cells
    ]


def _configure_model_worker(
    cuda_devices: str | None,
    *,
    cpu_threads: int,
) -> None:
    """Pin a spawned worker before any model or CUDA runtime is loaded."""
    if cuda_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_devices)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        # Spawned workers must not inherit host-wide values such as 32/64 from
        # managed notebook images. With eight workers that can create hundreds
        # of runnable threads and make CPU preprocessing slower than inference.
        os.environ[name] = str(max(int(cpu_threads), 1))
    os.environ.setdefault("OMP_DYNAMIC", "FALSE")
    os.environ.setdefault("MKL_DYNAMIC", "FALSE")
    os.environ["WORLDFOUNDRY_SUITE_MODEL_WORKER"] = "1"
    torch = sys.modules.get("torch")
    set_num_threads = getattr(torch, "set_num_threads", None)
    if callable(set_num_threads):
        set_num_threads(max(int(cpu_threads), 1))


def _run_model_worker(
    send_connection: Any,
    request: ModelBenchmarkSuiteRequest,
    root: Path,
    run_fingerprint: str,
    previous_cells: Mapping[str, Mapping[str, Any]],
    model_id: str,
    cuda_devices: str | None,
    cpu_threads: int,
) -> None:
    """Run exactly one model and return its cell records through a pipe."""
    _configure_model_worker(cuda_devices, cpu_threads=cpu_threads)
    pid = os.getpid()
    try:
        cells = _run_model_cells(
            request,
            root=root,
            run_fingerprint=run_fingerprint,
            previous_cells=previous_cells,
            model_id=model_id,
        )
        payload = {
            "ok": True,
            "model_id": model_id,
            "pid": pid,
            "cuda_visible_devices": cuda_devices,
            "cells": _worker_annotated_cells(
                cells,
                pid=pid,
                cuda_visible_devices=cuda_devices,
            ),
        }
    except BaseException as exc:  # parent needs structured worker failure evidence.
        payload = {
            "ok": False,
            "model_id": model_id,
            "pid": pid,
            "cuda_visible_devices": cuda_devices,
            "error": f"{type(exc).__name__}: {exc}",
        }
    try:
        send_connection.send(payload)
    finally:
        send_connection.close()


def _failed_model_worker_cells(
    request: ModelBenchmarkSuiteRequest,
    *,
    run_fingerprint: str,
    model_id: str,
    reason: str,
) -> list[dict[str, Any]]:
    """Materialize deterministic failed cells when a spawned model worker dies."""
    model_outputs, known_model, canonical_model_id = _model_outputs(model_id, request.model_manifest_dir)
    cells: list[dict[str, Any]] = []
    for benchmark in _selected_benchmarks(request):
        plan = _plan_cell(
            request,
            run_fingerprint=run_fingerprint,
            model_id=model_id,
            canonical_model_id=canonical_model_id,
            known_model=known_model,
            model_outputs=model_outputs,
            benchmark=benchmark,
        )
        base_cell = plan.to_base_cell()
        if plan.reason and request.skip_incompatible:
            cells.append({**base_cell, "status": "skipped", "exit_code": 0, "reason": plan.reason})
        else:
            cells.append({**base_cell, "status": "failed", "exit_code": 1, "reason": reason})
    return cells


def _join_model_worker(process: Any, *, timeout_seconds: float = 5.0) -> None:
    """Reap a completed worker, escalating only after its result was collected."""
    process.join(timeout=max(float(timeout_seconds), 0.0))
    if process.is_alive():
        process.terminate()
        process.join(timeout=max(float(timeout_seconds), 0.0))
    if process.is_alive() and hasattr(process, "kill"):
        process.kill()
        process.join()


def _run_models_in_spawn_workers(
    request: ModelBenchmarkSuiteRequest,
    *,
    root: Path,
    run_fingerprint: str,
    previous_cells: Mapping[str, Mapping[str, Any]],
    model_ids: Sequence[str],
    worker_plan: _ModelWorkerPlan,
) -> list[dict[str, Any]]:
    """Dynamically run one model per spawned process on immutable CUDA groups."""
    import multiprocessing
    from multiprocessing.connection import Connection
    from multiprocessing.connection import wait as wait_for_connections

    try:
        pickle.dumps(
            (request, root, run_fingerprint, previous_cells, tuple(model_ids)),
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    except (AttributeError, pickle.PickleError, TypeError) as exc:
        raise ValueError(
            "parallel model suites require a serializable request; use mapping-based model_config values "
            "or set model_workers=1"
        ) from exc

    context = multiprocessing.get_context("spawn")
    pending = deque(str(model_id) for model_id in model_ids)
    available_devices = deque(worker_plan.cuda_device_groups)
    cpu_threads = worker_plan.cpu_threads_per_worker
    cells_by_model: dict[str, list[dict[str, Any]]] = {}
    active: dict[Any, tuple[Any, str, str | None]] = {}

    try:
        while pending or active:
            while pending and available_devices:
                model_id = pending.popleft()
                cuda_devices = available_devices.popleft()
                receive_connection, send_connection = context.Pipe(duplex=False)
                process = context.Process(
                    target=_run_model_worker,
                    args=(
                        send_connection,
                        request,
                        root,
                        run_fingerprint,
                        previous_cells,
                        model_id,
                        cuda_devices,
                        cpu_threads,
                    ),
                    name=f"worldfoundry-model-{_safe_name(model_id)}",
                )
                try:
                    process.start()
                except Exception as exc:  # record startup failure and continue the suite.
                    receive_connection.close()
                    send_connection.close()
                    available_devices.append(cuda_devices)
                    cells_by_model[model_id] = _failed_model_worker_cells(
                        request,
                        run_fingerprint=run_fingerprint,
                        model_id=model_id,
                        reason=f"model worker failed to start: {type(exc).__name__}: {exc}",
                    )
                else:
                    send_connection.close()
                    active[receive_connection] = (process, model_id, cuda_devices)

            if not active:
                continue

            ready = set(cast(list[Connection], wait_for_connections(tuple(active), timeout=0.25)))
            ready.update(connection for connection, (process, _, _) in active.items() if not process.is_alive())
            for connection in ready:
                process, model_id, cuda_devices = active.pop(connection)
                payload: Any = None
                receive_error: str | None = None
                try:
                    payload = connection.recv()
                except Exception as exc:  # turn corrupt/dead worker payloads into failed cells.
                    receive_error = f"{type(exc).__name__}: {exc}"
                finally:
                    connection.close()

                _join_model_worker(process)
                available_devices.append(cuda_devices)
                worker_pid = process.pid
                if (
                    isinstance(payload, Mapping)
                    and payload.get("ok") is True
                    and isinstance(payload.get("cells"), list)
                ):
                    cells_by_model[model_id] = [dict(cell) for cell in payload["cells"]]
                    continue

                payload_error = payload.get("error") if isinstance(payload, Mapping) else None
                invalid_payload = None if payload is None or isinstance(payload, Mapping) else "invalid worker payload"
                reason_detail = (
                    payload_error or receive_error or invalid_payload or f"worker exited with code {process.exitcode}"
                )
                failed_cells = _failed_model_worker_cells(
                    request,
                    run_fingerprint=run_fingerprint,
                    model_id=model_id,
                    reason=f"model worker failed: {reason_detail}",
                )
                cells_by_model[model_id] = _worker_annotated_cells(
                    failed_cells,
                    pid=worker_pid,
                    cuda_visible_devices=cuda_devices,
                )
    except BaseException:
        for connection, (process, _, _) in active.items():
            connection.close()
            if process.is_alive():
                process.terminate()
            _join_model_worker(process)
        raise

    return [cell for model_id in model_ids for cell in cells_by_model[model_id]]
