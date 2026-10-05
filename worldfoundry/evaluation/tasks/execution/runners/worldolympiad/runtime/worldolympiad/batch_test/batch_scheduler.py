#!/usr/bin/env python3
"""Controlled batch scheduler for score_video_physical_3d.py.

This scheduler intentionally lives under `batch_test/` and does not change the
main single-video pipeline. It uses a fixed worker pool, per-worker GPU pinning,
QwenVL service URLs, resumable outputs, and per-item logs.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REQUIRED_MANIFEST_FIELDS = ("video", "gt_video", "prompt_json", "chunk_json", "output")


@dataclass(frozen=True)
class WorkerSpec:
    worker_id: int
    gpu: str
    qwen_url: str
    sam3_url: str | None = None
    reward_3d_url: str | None = None


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def load_manifest(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            item = json.loads(stripped)
            item["_line_number"] = line_number
            items.append(item)
    return items


def validate_manifest_items(items: list[dict[str, Any]], manifest_path: str | os.PathLike[str]) -> None:
    errors: list[str] = []
    for item in items:
        missing = [field for field in REQUIRED_MANIFEST_FIELDS if not item.get(field)]
        if missing:
            item_id = item.get("id") or item.get("_line_number") or "unknown"
            errors.append(f"line {item.get('_line_number')}: id={item_id!r} missing {', '.join(missing)}")
    if errors:
        preview = "\n".join(errors[:10])
        suffix = f"\n... and {len(errors) - 10} more" if len(errors) > 10 else ""
        raise ValueError(f"Invalid manifest {manifest_path}:\n{preview}{suffix}")


def resolve_path(value: str | None) -> str | None:
    if value is None:
        return None
    return str(Path(value))


def build_command(
    *,
    python_executable: str,
    worldeval_root: Path,
    project_root: Path,
    item: dict[str, Any],
    args: argparse.Namespace,
) -> list[str]:
    script_path = worldeval_root / "scripts" / "score_video_physical_3d.py"
    model_paths = resolve_model_paths(worldeval_root, project_root, args)
    command = [
        python_executable,
        str(script_path),
        "--video",
        item["video"],
        "--gt-video",
        item["gt_video"],
        "--prompt-json",
        item["prompt_json"],
        "--chunk-json",
        item["chunk_json"],
        "--physical-max-frames",
        str(args.physical_max_frames),
        "--physical-batch-mode",
        args.physical_batch_mode,
        "--three-d-max-frames",
        str(args.three_d_max_frames),
        "--sam-device",
        "0",
        "--da-device",
        "0",
        "--vlm-model",
        model_paths["vlm_model"],
        "--interaction-vlm-model",
        model_paths["interaction_vlm_model"],
        "--three-d-scoring-model",
        model_paths["three_d_scoring_model"],
        "--three-d-model-name",
        model_paths["three_d_model_name"],
        "--sam3-model",
        model_paths["sam3_model"],
        "--clip-download-root",
        model_paths["clip_download_root"],
        "--output",
        item["output"],
    ]

    if args.run_clip_interaction:
        command.append("--run-clip-interaction")
    if args.skip_physical:
        command.append("--skip-physical")
    if args.skip_interaction:
        command.append("--skip-interaction")
    if args.skip_3d:
        command.append("--skip-3d")
    if args.three_d_no_mask_dynamic_objects:
        command.append("--three-d-no-mask-dynamic-objects")
    if args.extra_arg:
        command.extend(args.extra_arg)
    return command


def default_weight_prefix(worldeval_root: Path, project_root: Path) -> str:
    if worldeval_root == project_root:
        return "."
    try:
        relative = os.path.relpath(worldeval_root, project_root)
    except ValueError:
        return str(worldeval_root)
    return relative


def join_default_path(prefix: str, relative_path: str) -> str:
    if prefix == ".":
        return relative_path
    return str(Path(prefix) / relative_path)


def resolve_model_paths(worldeval_root: Path, project_root: Path, args: argparse.Namespace) -> dict[str, str]:
    prefix = default_weight_prefix(worldeval_root, project_root)
    return {
        "vlm_model": args.vlm_model or join_default_path(prefix, "weights/QwenVL"),
        "interaction_vlm_model": args.interaction_vlm_model or join_default_path(prefix, "weights/QwenVL"),
        "three_d_scoring_model": args.three_d_scoring_model or join_default_path(prefix, "weights/QwenVL"),
        "three_d_model_name": args.three_d_model_name or join_default_path(prefix, "weights/da3"),
        "sam3_model": args.sam3_model or join_default_path(prefix, "weights/sam3/sam3.pt"),
        "clip_download_root": args.clip_download_root or join_default_path(prefix, "weights/clip"),
    }


def should_skip(item: dict[str, Any], force: bool) -> bool:
    output = item.get("output")
    return bool(output) and Path(output).exists() and not force


def run_item(
    *,
    item: dict[str, Any],
    worker: WorkerSpec,
    command: list[str],
    cwd: Path,
    log_dir: Path,
    dry_run: bool,
) -> int:
    item_id = str(item.get("id") or item.get("_line_number"))
    log_path = log_dir / f"{item_id}.worker{worker.worker_id}.log"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = worker.gpu
    env["QWENVL_SERVER_URL"] = worker.qwen_url.rstrip("/")
    env["VLM_BACKEND"] = "qwenvl_server"
    env["REWARD_3D_SCORER"] = "qwenvl_server"
    if worker.sam3_url:
        env["SAM3_SERVER_URL"] = worker.sam3_url.rstrip("/")
    if worker.reward_3d_url:
        env["REWARD_3D_SERVER_URL"] = worker.reward_3d_url.rstrip("/")

    command_text = " ".join(command)
    if dry_run:
        print(f"[dry-run][worker {worker.worker_id} gpu {worker.gpu}] {command_text}")
        return 0

    log_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"worker_id={worker.worker_id}\n")
        log_file.write(f"gpu={worker.gpu}\n")
        log_file.write(f"qwen_url={worker.qwen_url}\n")
        if worker.sam3_url:
            log_file.write(f"sam3_url={worker.sam3_url}\n")
        if worker.reward_3d_url:
            log_file.write(f"reward_3d_url={worker.reward_3d_url}\n")
        log_file.write(f"command={command_text}\n\n")
        log_file.flush()
        process = subprocess.run(
            command,
            cwd=str(cwd),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        elapsed = time.time() - start
        log_file.write(f"\nexit_code={process.returncode}\n")
        log_file.write(f"elapsed_seconds={elapsed:.2f}\n")
    return int(process.returncode)


def worker_loop(
    *,
    worker: WorkerSpec,
    task_queue: "queue.Queue[dict[str, Any]]",
    result_lock: threading.Lock,
    results: list[dict[str, Any]],
    worldeval_root: Path,
    project_root: Path,
    args: argparse.Namespace,
) -> None:
    while True:
        try:
            item = task_queue.get_nowait()
        except queue.Empty:
            return

        item_id = str(item.get("id") or item.get("_line_number"))
        if should_skip(item, args.force):
            print(f"[worker {worker.worker_id}] skip existing: {item_id}")
            with result_lock:
                results.append({"id": item_id, "status": "skipped", "output": item.get("output")})
            task_queue.task_done()
            continue

        command = build_command(
            python_executable=sys.executable,
            worldeval_root=worldeval_root,
            project_root=project_root,
            item=item,
            args=args,
        )
        print(f"[worker {worker.worker_id} gpu {worker.gpu}] start: {item_id}")
        exit_code = run_item(
            item=item,
            worker=worker,
            command=command,
            cwd=project_root,
            log_dir=Path(args.log_dir),
            dry_run=args.dry_run,
        )
        status = "ok" if exit_code == 0 else "failed"
        print(f"[worker {worker.worker_id} gpu {worker.gpu}] {status}: {item_id}")
        with result_lock:
            results.append(
                {
                    "id": item_id,
                    "status": status,
                    "exit_code": exit_code,
                    "output": item.get("output"),
                }
            )
        task_queue.task_done()


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch scheduler for worldeval scoring")
    parser.add_argument("--worldeval-root", default="worldeval", help="Path to worldeval submodule directory")
    parser.add_argument(
        "--project-root",
        default=None,
        help="Working directory for scoring commands and relative manifest paths. Defaults to parent of --worldeval-root.",
    )
    parser.add_argument("--manifest", required=True, help="Input JSONL manifest")
    parser.add_argument("--gpu-slots", required=True, help="Comma-separated GPUs used by scoring workers")
    parser.add_argument(
        "--qwen-server-urls",
        default="http://127.0.0.1:8008",
        help="Comma-separated QwenVL service URLs. Workers use them round-robin.",
    )
    parser.add_argument(
        "--sam3-server-urls",
        default="",
        help="Optional comma-separated persistent SAM3 service URLs. Workers use them round-robin.",
    )
    parser.add_argument(
        "--reward-3d-server-urls",
        default="",
        help="Optional comma-separated persistent 3D reward service URLs. Workers use them round-robin.",
    )
    parser.add_argument("--workers", type=int, default=0, help="Worker count. Defaults to len(gpu-slots).")
    parser.add_argument("--log-dir", default="batch_logs", help="Directory for per-item logs")
    parser.add_argument("--summary", default="batch_logs/summary.jsonl", help="Batch summary JSONL path")
    parser.add_argument("--force", action="store_true", help="Run even if item output already exists")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running")

    parser.add_argument("--physical-max-frames", type=int, default=64)
    parser.add_argument("--physical-batch-mode", choices=["none", "dimension", "all"], default="dimension")
    parser.add_argument("--three-d-max-frames", type=int, default=32)
    parser.add_argument("--vlm-model", default=None)
    parser.add_argument("--interaction-vlm-model", default=None)
    parser.add_argument("--three-d-scoring-model", default=None)
    parser.add_argument("--three-d-model-name", default=None)
    parser.add_argument("--sam3-model", default=None)
    parser.add_argument("--clip-download-root", default=None)
    parser.add_argument("--run-clip-interaction", action="store_true")
    parser.add_argument("--skip-physical", action="store_true")
    parser.add_argument("--skip-interaction", action="store_true")
    parser.add_argument("--skip-3d", action="store_true")
    parser.add_argument("--three-d-no-mask-dynamic-objects", action="store_true")
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Append one extra argument to every scoring command. Repeat for multiple args.",
    )
    args = parser.parse_args()

    worldeval_root = Path(args.worldeval_root).resolve()
    project_root = Path(args.project_root).resolve() if args.project_root else worldeval_root.parent
    if not (worldeval_root / "scripts" / "score_video_physical_3d.py").exists():
        raise FileNotFoundError(f"Invalid worldeval root: {worldeval_root}")

    items = load_manifest(args.manifest)
    validate_manifest_items(items, args.manifest)
    gpu_slots = parse_csv(args.gpu_slots)
    qwen_urls = parse_csv(args.qwen_server_urls)
    sam3_urls = parse_csv(args.sam3_server_urls)
    reward_3d_urls = parse_csv(args.reward_3d_server_urls)
    if not gpu_slots:
        raise ValueError("--gpu-slots must not be empty.")
    if not qwen_urls:
        raise ValueError("--qwen-server-urls must not be empty.")

    worker_count = args.workers or len(gpu_slots)
    workers = [
        WorkerSpec(
            worker_id=index,
            gpu=gpu_slots[index % len(gpu_slots)],
            qwen_url=qwen_urls[index % len(qwen_urls)],
            sam3_url=sam3_urls[index % len(sam3_urls)] if sam3_urls else None,
            reward_3d_url=reward_3d_urls[index % len(reward_3d_urls)] if reward_3d_urls else None,
        )
        for index in range(worker_count)
    ]

    task_queue: "queue.Queue[dict[str, Any]]" = queue.Queue()
    for item in items:
        task_queue.put(item)

    print(f"Loaded {len(items)} item(s)")
    print(f"Workers: {worker_count}")
    print(f"GPU slots: {gpu_slots}")
    print(f"QwenVL services: {qwen_urls}")
    if sam3_urls:
        print(f"SAM3 services: {sam3_urls}")
    if reward_3d_urls:
        print(f"3D reward services: {reward_3d_urls}")

    results: list[dict[str, Any]] = []
    result_lock = threading.Lock()
    threads = [
        threading.Thread(
            target=worker_loop,
            kwargs={
                "worker": worker,
                "task_queue": task_queue,
                "result_lock": result_lock,
                "results": results,
                "worldeval_root": worldeval_root,
                "project_root": project_root,
                "args": args,
            },
            daemon=True,
        )
        for worker in workers
    ]

    start = time.time()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    elapsed = time.time() - start

    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        for result in sorted(results, key=lambda item: str(item.get("id"))):
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")

    ok_count = sum(1 for result in results if result.get("status") == "ok")
    skipped_count = sum(1 for result in results if result.get("status") == "skipped")
    failed_count = sum(1 for result in results if result.get("status") == "failed")
    print(
        f"Done in {elapsed:.2f}s: ok={ok_count}, skipped={skipped_count}, "
        f"failed={failed_count}. Summary: {summary_path}"
    )


if __name__ == "__main__":
    main()
