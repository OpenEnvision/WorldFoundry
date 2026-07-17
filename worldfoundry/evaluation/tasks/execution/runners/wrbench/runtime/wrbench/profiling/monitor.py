"""Command-level resource monitor for wrbench generation runs."""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import socket
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

from .stage import utc_now_iso


RESOURCE_PROFILE_SCHEMA_VERSION = "wrbench_generation_resource_profile_v1"
SampleFn = Callable[[Optional[list[str]]], list[dict[str, Any]]]


def run_profiled_command(
    cmd: list[str],
    *,
    cwd: str | Path,
    env: dict[str, str] | None = None,
    summary_path: str | Path,
    trace_path: str | Path | None = None,
    stage_events_path: str | Path | None = None,
    log_path: str | Path | None = None,
    external_log_path: str | Path | None = None,
    run_identity: dict[str, Any] | None = None,
    sampling_interval_seconds: float = 0.5,
    generation_status: str | None = None,
    sampler: SampleFn | None = None,
) -> dict[str, Any]:
    """Run ``cmd`` and write a resource profile summary JSON."""

    cwd_path = Path(cwd)
    summary = Path(summary_path)
    trace = Path(trace_path) if trace_path else None
    stage_events = Path(stage_events_path) if stage_events_path else None
    log = Path(log_path) if log_path else None
    summary.parent.mkdir(parents=True, exist_ok=True)
    if trace is not None:
        trace.parent.mkdir(parents=True, exist_ok=True)
    if stage_events is not None:
        stage_events.parent.mkdir(parents=True, exist_ok=True)
        stage_events.write_text("", encoding="utf-8")
    if log is not None:
        log.parent.mkdir(parents=True, exist_ok=True)

    child_env = dict(env or os.environ.copy())
    if stage_events is not None:
        child_env["WRBENCH_STAGE_EVENTS_PATH"] = str(stage_events)
    child_env["WRBENCH_PROFILE_RESOURCES"] = "1"

    identity = dict(run_identity or {})
    identity.setdefault("command", cmd)
    identity.setdefault("cwd", str(cwd_path))
    identity.setdefault("cuda_visible_devices", child_env.get("CUDA_VISIBLE_DEVICES"))

    started_at = utc_now_iso()
    command_id = _command_id(identity, started_at)
    identity["command_id"] = command_id
    child_env["WRBENCH_RESOURCE_COMMAND_ID"] = command_id

    visible_gpu_ids = _parse_visible_gpu_ids(child_env.get("CUDA_VISIBLE_DEVICES"))
    sample_fn = sampler or sample_gpu_memory
    samples: list[dict[str, Any]] = []
    sampler_state = {"available": True, "error": None}
    stop_event = threading.Event()

    def poll() -> None:
        while not stop_event.is_set():
            try:
                rows = sample_fn(visible_gpu_ids)
                now = utc_now_iso()
                monotonic_now = time.monotonic()
                for row in rows:
                    sample = dict(row)
                    sample.setdefault("sampled_at", now)
                    sample.setdefault("sample_monotonic", monotonic_now)
                    samples.append(sample)
            except Exception as exc:
                sampler_state["available"] = False
                sampler_state["error"] = f"{type(exc).__name__}: {exc}"
                return
            stop_event.wait(max(0.05, float(sampling_interval_seconds)))

    thread = threading.Thread(target=poll, name="wrbench_resource_sampler", daemon=True)
    thread.start()

    log_handle = None
    start_mono = time.monotonic()
    rc = 1
    try:
        stdout_target = None
        stderr_target = None
        if log is not None:
            log_handle = log.open("w", encoding="utf-8")
            log_handle.write("$ " + " ".join(shlex.quote(part) for part in cmd) + "\n")
            log_handle.flush()
            stdout_target = log_handle
            stderr_target = subprocess.STDOUT
        proc = subprocess.run(
            cmd,
            cwd=str(cwd_path),
            env=child_env,
            stdout=stdout_target,
            stderr=stderr_target,
            text=True,
        )
        rc = int(proc.returncode)
    finally:
        end_mono = time.monotonic()
        stop_event.set()
        thread.join(timeout=max(1.0, float(sampling_interval_seconds) * 2.0))
        if log_handle is not None:
            log_handle.write(f"\n[exit] {rc}\n")
            log_handle.close()

    if trace is not None:
        with trace.open("w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample, sort_keys=True) + "\n")

    profile = build_profile_summary(
        run_identity=identity,
        exit_code=rc,
        started_at=started_at,
        ended_at=utc_now_iso(),
        wall_time_seconds=max(0.0, end_mono - start_mono),
        sampling_interval_seconds=float(sampling_interval_seconds),
        samples=samples,
        gpu_monitor_available=bool(sampler_state["available"]),
        gpu_monitor_error=sampler_state["error"],
        stage_events_path=stage_events,
        trace_path=trace,
        summary_path=summary,
        log_path=log,
        external_log_path=Path(external_log_path) if external_log_path else None,
        generation_status=generation_status or ("generated" if rc == 0 else "failed"),
        cwd=cwd_path,
    )
    summary.write_text(json.dumps(profile, indent=2, sort_keys=True), encoding="utf-8")
    return profile


def build_profile_summary(
    *,
    run_identity: dict[str, Any],
    exit_code: int,
    started_at: str,
    ended_at: str,
    wall_time_seconds: float,
    sampling_interval_seconds: float,
    samples: list[dict[str, Any]],
    gpu_monitor_available: bool,
    gpu_monitor_error: str | None,
    stage_events_path: Path | None,
    trace_path: Path | None,
    summary_path: Path,
    log_path: Path | None,
    external_log_path: Path | None,
    generation_status: str,
    cwd: Path,
) -> dict[str, Any]:
    stage_summary = summarize_stage_events(stage_events_path)
    gpu_observation = summarize_gpu_samples(
        samples,
        sampling_interval_seconds=sampling_interval_seconds,
        available=gpu_monitor_available,
        error=gpu_monitor_error,
    )
    output_video_seconds = _float_or_none(run_identity.get("output_video_seconds"))
    gpu_width = _int_or_default(run_identity.get("gpu_width"), _gpu_width_from_identity(run_identity))
    derived = derive_metrics(stage_summary, gpu_width=gpu_width, output_video_seconds=output_video_seconds)
    return {
        "schema_version": RESOURCE_PROFILE_SCHEMA_VERSION,
        "producer": "wrbench",
        "measurement_scope": "command_envelope_with_stage_spans",
        "run_identity": run_identity,
        "status": {
            "exit_code": exit_code,
            "started_at": started_at,
            "ended_at": ended_at,
            "wall_time_seconds": wall_time_seconds,
            "timed_out": False,
            "ok": exit_code == 0,
            "generation_status": generation_status,
        },
        "stage_summary": stage_summary,
        "gpu_observation": gpu_observation,
        "artifacts": {
            "generation_log": str(log_path or external_log_path) if (log_path or external_log_path) else None,
            "resource_summary_json": str(summary_path),
            "resource_trace_jsonl": str(trace_path) if trace_path else None,
            "stage_events_jsonl": str(stage_events_path) if stage_events_path else None,
        },
        "environment": {
            "hostname": socket.gethostname(),
            "cuda_visible_devices": run_identity.get("cuda_visible_devices"),
            "wrbench_git_commit": _git_commit(cwd),
        },
        "derived_metrics": derived,
        "notes": _summary_notes(stage_summary, gpu_observation),
    }


def update_summary_derived_metrics(
    summary_path: str | Path,
    *,
    output_video_seconds: float | None = None,
    generation_status: str | None = None,
) -> dict[str, Any]:
    """Update post-QC derived fields after output durations are known."""

    path = Path(summary_path)
    profile = json.loads(path.read_text(encoding="utf-8"))
    if output_video_seconds is not None:
        profile["run_identity"]["output_video_seconds"] = float(output_video_seconds)
    if generation_status is not None:
        profile["status"]["generation_status"] = generation_status
    gpu_width = _int_or_default(
        profile.get("derived_metrics", {}).get("gpu_width"),
        _gpu_width_from_identity(profile.get("run_identity", {})),
    )
    profile["derived_metrics"] = derive_metrics(
        profile.get("stage_summary", {}),
        gpu_width=gpu_width,
        output_video_seconds=_float_or_none(profile.get("run_identity", {}).get("output_video_seconds")),
    )
    path.write_text(json.dumps(profile, indent=2, sort_keys=True), encoding="utf-8")
    return profile


def summarize_stage_events(stage_events_path: Path | None) -> dict[str, Any]:
    stages = {
        "model_load": 0.0,
        "input_decode": 0.0,
        "preprocess": 0.0,
        "inference": 0.0,
        "postprocess": 0.0,
        "encode_write": 0.0,
        "sidecar_write": 0.0,
    }
    events: list[dict[str, Any]] = []
    observed_stages: set[str] = set()
    if stage_events_path is not None and stage_events_path.is_file():
        for line in stage_events_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            events.append(event)
            stage = str(event.get("stage") or "")
            if stage in stages:
                observed_stages.add(stage)
                stages[stage] += float(event.get("duration_seconds") or 0.0)

    stage_status = "available" if events else "stage_unavailable"
    benchmark = None
    if stage_status == "available" and {"preprocess", "inference"}.issubset(observed_stages):
        benchmark = stages["preprocess"] + stages["inference"]
    return {
        "stage_status": stage_status,
        "stage_source": "wrbench_stage_recorder" if events else None,
        "stages": stages,
        "event_count": len(events),
        "model_load_seconds": stages["model_load"],
        "input_decode_seconds": stages["input_decode"],
        "preprocess_seconds": stages["preprocess"],
        "inference_seconds": stages["inference"],
        "postprocess_seconds": stages["postprocess"],
        "encode_write_seconds": stages["encode_write"],
        "sidecar_write_seconds": stages["sidecar_write"],
        "benchmark_generation_seconds": benchmark,
    }


def summarize_gpu_samples(
    samples: list[dict[str, Any]],
    *,
    sampling_interval_seconds: float,
    available: bool,
    error: str | None,
) -> dict[str, Any]:
    peaks: dict[str, float] = {}
    names: dict[str, str] = {}
    for sample in samples:
        gpu_id = str(sample.get("gpu_id"))
        used = _float_or_none(sample.get("memory_used_mib"))
        if used is None:
            continue
        peaks[gpu_id] = max(peaks.get(gpu_id, 0.0), used)
        if sample.get("gpu_name"):
            names[gpu_id] = str(sample["gpu_name"])
    peak_values = list(peaks.values())
    return {
        "gpu_monitor_available": bool(available and samples),
        "monitor_backend": "nvidia-smi",
        "monitor_error": error,
        "sampling_interval_seconds": sampling_interval_seconds,
        "gpu_ids": sorted(peaks.keys(), key=_natural_gpu_key),
        "gpu_name_by_id": names,
        "sample_count": len(samples),
        "peak_memory_mib_by_gpu": peaks,
        "peak_memory_mib_max": max(peak_values) if peak_values else None,
        "peak_memory_mib_sum": sum(peak_values) if peak_values else None,
    }


def derive_metrics(
    stage_summary: dict[str, Any],
    *,
    gpu_width: int,
    output_video_seconds: float | None,
) -> dict[str, Any]:
    benchmark = _float_or_none(stage_summary.get("benchmark_generation_seconds"))
    preprocess = _float_or_none(stage_summary.get("preprocess_seconds"))
    inference = _float_or_none(stage_summary.get("inference_seconds"))
    benchmark_gpu_seconds = benchmark * gpu_width if benchmark is not None else None
    return {
        "gpu_width": int(gpu_width),
        "output_video_seconds": output_video_seconds,
        "benchmark_generation_seconds": benchmark,
        "benchmark_generation_seconds_per_output_second": _ratio(benchmark, output_video_seconds),
        "benchmark_gpu_seconds": benchmark_gpu_seconds,
        "benchmark_gpu_seconds_per_output_second": _ratio(benchmark_gpu_seconds, output_video_seconds),
        "preprocess_seconds_per_output_second": _ratio(preprocess, output_video_seconds),
        "inference_seconds_per_output_second": _ratio(inference, output_video_seconds),
    }


def sample_gpu_memory(gpu_ids: list[str] | None = None) -> list[dict[str, Any]]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.used",
        "--format=csv,noheader,nounits",
    ]
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "nvidia-smi failed")
    wanted = set(gpu_ids or [])
    rows = []
    for line in proc.stdout.splitlines():
        parts = [part.strip() for part in line.split(",", 3)]
        if len(parts) != 4:
            continue
        gpu_id, uuid, name, used = parts
        if wanted and gpu_id not in wanted:
            continue
        rows.append(
            {
                "gpu_id": gpu_id,
                "gpu_uuid": uuid,
                "gpu_name": name,
                "memory_used_mib": float(used),
            }
        )
    return rows


def _command_id(identity: dict[str, Any], started_at: str) -> str:
    payload = dict(identity)
    payload["started_at"] = started_at
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _parse_visible_gpu_ids(value: str | None) -> list[str] | None:
    if not value:
        return None
    ids = []
    for part in value.split(","):
        part = part.strip()
        if part.isdigit():
            ids.append(part)
    return ids or None


def _gpu_width_from_identity(identity: dict[str, Any]) -> int:
    explicit = identity.get("gpu_width")
    if explicit is not None:
        try:
            return max(1, int(explicit))
        except (TypeError, ValueError):
            pass
    visible = _parse_visible_gpu_ids(str(identity.get("cuda_visible_devices") or ""))
    return max(1, len(visible or []))


def _git_commit(cwd: Path) -> str | None:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if proc.returncode == 0:
        return proc.stdout.strip()
    return None


def _summary_notes(stage_summary: dict[str, Any], gpu_observation: dict[str, Any]) -> list[str]:
    notes = []
    if stage_summary.get("stage_status") == "stage_unavailable":
        notes.append("stage spans unavailable; benchmark_generation_seconds is null")
    if not gpu_observation.get("gpu_monitor_available"):
        notes.append("GPU monitor unavailable or no GPU samples captured")
    return notes


def _ratio(value: float | None, denom: float | None) -> float | None:
    if value is None or denom is None or denom <= 0:
        return None
    return value / denom


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_default(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _natural_gpu_key(value: str) -> tuple[int, str]:
    try:
        return (int(value), value)
    except ValueError:
        return (10**9, value)
