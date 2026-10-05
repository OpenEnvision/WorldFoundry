from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from worldfoundry.studio.inference.catalog import find_entry
from worldfoundry.studio.inference.execution import (
    TORCHRUN_DISTRIBUTED_ENV,
    TORCHRUN_LINGBOT_FAST_ENV,
    StudioManager,
    _persist_studio_performance_metadata,
    _prepared_inputs_from_payload,
    _prepared_inputs_payload,
    _torchrun_rank,
    _torchrun_world_size,
    ensure_torchrun_lingbot_fast_runtime,
    shutdown_torchrun_lingbot_fast_runtime,
)

SECRET_ENV_REF_KEY = "__worldfoundry_secret_env__"


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"payload must be a JSON object: {path}")
    return payload


def _resolve_secret_refs(value: Any) -> Any:
    if isinstance(value, dict):
        if set(value) == {SECRET_ENV_REF_KEY}:
            return os.getenv(str(value[SECRET_ENV_REF_KEY]), "")
        return {str(key): _resolve_secret_refs(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_secret_refs(item) for item in value]
    return value


_RUN_PREPARE_KEYS = (
    "model_id",
    "prompt",
    "input_path",
    "image",
    "video",
    "last_frame",
    "reference_files",
    "interactions_text",
    "camera_view_text",
    "task_type",
    "intrinsics_text",
    "meta_path",
    "panorama_path",
    "scene_name",
    "fps",
    "num_frames",
    "call_kwargs_text",
    "load_kwargs_text",
    "model_ref",
    "backend",
    "endpoint",
    "api_key",
    "device",
    "infer_metadata",
)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp_path, path)


def _wait_for_prepared_payload(prepared_path: Path, error_path: Path) -> dict[str, Any]:
    timeout = float(os.getenv("WORLDFOUNDRY_STUDIO_DISTRIBUTED_PREPARE_TIMEOUT", "1800") or "1800")
    deadline = time.monotonic() + timeout
    while True:
        if prepared_path.is_file():
            return _load_json(prepared_path)
        if error_path.is_file():
            payload = _load_json(error_path)
            detail = payload.get("traceback") or payload.get("error") or f"see {error_path}"
            raise RuntimeError(f"rank0 failed before sharing prepared Studio request: {detail}")
        if time.monotonic() > deadline:
            raise TimeoutError(f"Timed out waiting for distributed Studio request: {prepared_path}")
        time.sleep(0.2)


def _shutdown_torch_distributed_runtime() -> None:
    try:
        import torch.distributed as dist
    except Exception:
        return
    try:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
    except Exception as exc:
        print(
            f"[studio][torchrun][rank {_torchrun_rank()}] failed to destroy process group: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )


def _run_distributed_manager_payload(
    args: argparse.Namespace,
    manager: StudioManager,
    run_kwargs: dict[str, Any],
) -> int:
    rank = _torchrun_rank()
    prepared_path = Path(args.result_path).with_name("prepared_request.json")
    error_path = Path(args.result_path).with_name("prepared_request.error.json")
    action = str(run_kwargs.get("action") or "run")
    perf_segments: dict[str, float] = {}
    wall_t0 = time.perf_counter()

    if rank == 0:
        for stale in (prepared_path, error_path):
            try:
                stale.unlink()
            except FileNotFoundError:
                pass
        try:
            prepare_kwargs = {key: run_kwargs.get(key) for key in _RUN_PREPARE_KEYS}
            entry, request, perf_segments, wall_t0 = manager._prepare_run_request(**prepare_kwargs)
            _write_json_atomic(
                prepared_path,
                {
                    "model_id": entry.model_id,
                    "action": action,
                    "request": _prepared_inputs_payload(request),
                },
            )
        except Exception as exc:
            _write_json_atomic(
                error_path,
                {
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                },
            )
            raise
    else:
        payload = _wait_for_prepared_payload(prepared_path, error_path)
        entry = find_entry(str(payload["model_id"]))
        action = str(payload.get("action") or action)
        request = _prepared_inputs_from_payload(dict(payload["request"]))

    try:
        record = manager._run_local_prepared_request(
            entry=entry,
            action=action,
            request=request,
            progress_callback=None,
            perf_segments=perf_segments if rank == 0 else None,
            materialize=rank == 0,
        )
    finally:
        _shutdown_torch_distributed_runtime()
    if rank != 0:
        return 0
    if record is None:
        raise RuntimeError("distributed torchrun did not produce a rank-0 Studio run record")

    perf_segments["total_client_ms"] = (time.perf_counter() - wall_t0) * 1000.0
    _persist_studio_performance_metadata(record, perf_segments)
    result_path = Path(args.result_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(record.to_manifest(), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": record.status, "run_id": record.run_id, "output_dir": record.output_dir}, sort_keys=True))
    return 0


def _run_manager_payload(args: argparse.Namespace) -> int:
    payload = _load_json(args.payload_path)
    run_kwargs = payload.get("run_kwargs")
    if not isinstance(run_kwargs, dict):
        raise ValueError("payload.run_kwargs must be a JSON object")
    run_kwargs = _resolve_secret_refs(run_kwargs)
    if not isinstance(run_kwargs, dict):
        raise ValueError("resolved payload.run_kwargs must be a JSON object")

    manager = StudioManager(workspace_root=str(payload.get("workspace_root") or ""))
    torchrun_lingbot = (
        os.getenv(TORCHRUN_LINGBOT_FAST_ENV, "").strip().lower() in {"1", "true", "yes", "on"}
        and _torchrun_world_size() > 1
    )
    torchrun_distributed = (
        os.getenv(TORCHRUN_DISTRIBUTED_ENV, "").strip().lower() in {"1", "true", "yes", "on"}
        and _torchrun_world_size() > 1
    )
    if torchrun_distributed:
        try:
            return _run_distributed_manager_payload(args, manager, run_kwargs)
        finally:
            manager.close()

    if torchrun_lingbot and _torchrun_rank() != 0:
        try:
            ensure_torchrun_lingbot_fast_runtime()
            manager.run_torchrun_worker_loop()
        finally:
            try:
                shutdown_torchrun_lingbot_fast_runtime()
            finally:
                manager.close()
        return 0

    try:
        record = manager.run(
            **run_kwargs,
            progress_callback=None,
            materialize=not torchrun_distributed or _torchrun_rank() == 0,
        )
    finally:
        if torchrun_lingbot:
            try:
                manager.shutdown_torchrun_workers()
            finally:
                shutdown_torchrun_lingbot_fast_runtime()
        manager.close()

    if torchrun_distributed and _torchrun_rank() != 0:
        return 0
    if record is None:
        raise RuntimeError("distributed torchrun did not produce a rank-0 Studio run record")

    result_path = Path(args.result_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(record.to_manifest(), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": record.status, "run_id": record.run_id, "output_dir": record.output_dir}, sort_keys=True))
    return 0


def _run_manager_worker(args: argparse.Namespace) -> int:
    manager = StudioManager(workspace_root=str(args.workspace_root or ""))
    print(
        json.dumps(
            {
                "status": "ready",
                "worker": "worldfoundry.studio.runtime_job",
                "pid": os.getpid(),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    for raw_line in sys.stdin.buffer:
        if not raw_line.strip():
            continue
        try:
            payload = json.loads(raw_line.decode("utf-8"))
        except Exception:
            print(f"[studio-worker] ignoring invalid request line: {raw_line!r}", file=sys.stderr, flush=True)
            continue
        if not isinstance(payload, dict):
            print("[studio-worker] ignoring non-object request", file=sys.stderr, flush=True)
            continue
        if str(payload.get("command") or "").strip().lower() == "shutdown":
            print(json.dumps({"status": "shutdown"}, sort_keys=True), flush=True)
            manager.close()
            return 0

        result_path = Path(str(payload.get("result_path") or ""))
        error_path = Path(str(payload.get("error_path") or result_path.with_name("error.json")))
        request_id = str(payload.get("request_id") or "")
        for stale_path in (result_path, error_path):
            try:
                stale_path.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                pass

        try:
            run_kwargs = payload.get("run_kwargs")
            if not isinstance(run_kwargs, dict):
                raise ValueError("payload.run_kwargs must be a JSON object")
            run_kwargs = _resolve_secret_refs(run_kwargs)
            if not isinstance(run_kwargs, dict):
                raise ValueError("resolved payload.run_kwargs must be a JSON object")
            record = manager.run(
                **run_kwargs,
                progress_callback=None,
                materialize=True,
            )
            if record is None:
                raise RuntimeError("Studio manager did not produce a run record")
            _write_json_atomic(result_path, record.to_manifest())
            print(
                json.dumps(
                    {
                        "status": record.status,
                        "request_id": request_id,
                        "run_id": record.run_id,
                        "output_dir": record.output_dir,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001 - surfaced to the parent Studio job.
            _write_json_atomic(
                error_path,
                {
                    "request_id": request_id,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                },
            )
            traceback.print_exc()
    manager.close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a normalized Studio runtime payload.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_payload = subparsers.add_parser(
        "run-manager-payload",
        help="Run a pre-normalized StudioManager.run payload and write the resulting RunRecord manifest.",
    )
    run_payload.add_argument("--payload-path", required=True)
    run_payload.add_argument("--result-path", required=True)
    run_payload.set_defaults(func=_run_manager_payload)
    worker = subparsers.add_parser(
        "run-manager-worker",
        help="Run a resident StudioManager worker that accepts JSON requests on stdin.",
    )
    worker.add_argument("--workspace-root", required=True)
    worker.set_defaults(func=_run_manager_worker)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
