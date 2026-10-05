#!/usr/bin/env python3
"""Generate world-model videos for all benchmark tasks.

Usage:
    python -m generation.run_world_models \\
        --models veo hunyuan \\
        --tasks_root benchmark/tasks/ \\
        --output_root outputs/ \\
        --run_name test_2 \\
        --workers 4

    # Run all configured models:
    python -m generation.run_world_models --models all --run_name test_2

    # Re-generate even if output already exists:
    python -m generation.run_world_models --models veo --run_name test_2 --overwrite

Output layout (mirrors existing convention):
    outputs/{model}_{run_name}/
        {task_id}.mp4
        {model}_output_map_{run_name}.json   ← written incrementally after each success
"""

import argparse
import collections
import fnmatch
import json
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Tuple

from filelock import FileLock

import yaml

from generation.world_models import WorldModelRunner, build_runner

# ---------------------------------------------------------------------------
# Occlusion-variant deduplication (camera-controlled models only)
# ---------------------------------------------------------------------------

_OCCLUSION_SUFFIXES = ("_cardboard", "_curtain", "_lightoff")


def _occlusion_base(task_id: str) -> str:
    for s in _OCCLUSION_SUFFIXES:
        if task_id.endswith(s):
            return task_id[: -len(s)]
    return task_id


def _dedup_for_camera_model(
    tasks: List[Tuple[str, dict, Optional[Path]]],
) -> List[Tuple[str, dict, Optional[Path]]]:
    """For camera-controlled models, collapse occlusion variants to one per base name.

    Keeps the first variant encountered (tasks are sorted alphabetically by
    discover_tasks, so _cardboard wins over _curtain/_lightoff).
    The returned task_id is the base name (suffix stripped), so the output
    video and map entry use e.g. "ice_on_burner" rather than "ice_on_burner_cardboard".
    """
    seen: set = set()
    result: List[Tuple[str, dict, Optional[Path]]] = []
    for task_id, task, init_frame in tasks:
        base = _occlusion_base(task_id)
        if base not in seen:
            seen.add(base)
            result.append((base, task, init_frame))
    return result


# ---------------------------------------------------------------------------
# RPM limiter
# ---------------------------------------------------------------------------

class RpmLimiter:
    """Sliding-window rate limiter. Thread-safe.

    Tracks the start time of every request in the last 60 seconds.
    acquire() blocks until the number of requests in that window is
    below rpm, then stamps the current time and returns.
    """

    def __init__(self, rpm: int) -> None:
        self.rpm = rpm
        self._timestamps: collections.deque = collections.deque()
        self._lock = Lock()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                # Evict timestamps older than 60 s
                while self._timestamps and now - self._timestamps[0] >= 60.0:
                    self._timestamps.popleft()
                if len(self._timestamps) < self.rpm:
                    self._timestamps.append(now)
                    return
                # Sleep until the oldest slot expires
                wait = 60.0 - (now - self._timestamps[0])
            time.sleep(max(wait, 0.1))


# ---------------------------------------------------------------------------
# Task discovery
# ---------------------------------------------------------------------------

def discover_tasks(tasks_root: Path) -> List[Tuple[str, dict, Optional[Path]]]:
    """Walk tasks_root recursively and return one entry per valid task directory.

    A directory is treated as a task dir if it contains a YAML file named
    after the directory itself (e.g. tasks/ice_on_burner_01/ice_on_burner_01.yaml).

    Returns:
        List of (task_id, task_yaml_data, init_frame_path_or_None).
        task_id is taken from the YAML 'id' field, falling back to the folder name.
    """
    tasks_root = Path(tasks_root).expanduser().resolve()
    found: List[Tuple[str, dict, Optional[Path]]] = []

    for d in sorted(tasks_root.rglob("*")):
        if not d.is_dir():
            continue
        yaml_path = d / f"{d.name}.yaml"
        if not yaml_path.exists():
            continue

        with yaml_path.open(encoding="utf-8") as f:
            task = yaml.safe_load(f) or {}

        task_id = str(task.get("id") or d.name)
        init_frame = None
        for ext in (".png", ".jpg", ".jpeg"):
            candidate = d / f"{d.name}_init_frame{ext}"
            if candidate.exists():
                init_frame = candidate
                break
        found.append((task_id, task, init_frame))

    return found


# ---------------------------------------------------------------------------
# Output map helpers — written incrementally, thread-safe
# ---------------------------------------------------------------------------

def _update_map(map_path: Path, task_id: str, filename: str, lock: Lock) -> None:
    """Atomically add/update one entry in the JSON output map.

    Uses both a threading.Lock (intra-process) and a FileLock on disk
    (cross-process), so concurrent runs against the same output folder
    from separate terminals are safe.
    """
    lock_path = map_path.with_suffix(".lock")
    with lock:                          # serialise threads within this process
        with FileLock(str(lock_path)):  # serialise across processes
            data: dict = {}
            if map_path.exists():
                data = json.loads(map_path.read_text(encoding="utf-8"))
            data[task_id] = filename
            map_path.write_text(json.dumps(data, indent=4), encoding="utf-8")


# ---------------------------------------------------------------------------
# Per-task work unit
# ---------------------------------------------------------------------------

def _run_one(
    runner: WorldModelRunner,
    task_id: str,
    task: dict,
    init_frame: Optional[Path],
    output_path: Path,
) -> bool:
    """Extract the right prompt + camera_control from the task YAML and call generate()."""
    prompts = task.get("prompts") or {}
    prompt = (prompts.get(runner.prompt_field) or "").strip()

    camera_control: Optional[str] = None
    if runner.camera_control_field:
        cc = (task.get("camera_control") or {}).get(runner.camera_control_field)
        if cc and str(cc).strip().lower() not in ("null", "none", ""):
            camera_control = str(cc).strip()

    return runner.generate(
        task_id=task_id,
        prompt=prompt,
        init_frame=init_frame,
        output_path=output_path,
        camera_control=camera_control,
    )


# ---------------------------------------------------------------------------
# Daemon batch execution (local models only — loads weights once)
# ---------------------------------------------------------------------------

def _num_frames_from_pose(pose_string: str) -> int:
    """Compute the number of video frames from a camera pose string.

    Each token has the form 'action-duration' where duration is the number of
    camera latents.  Total latents are summed, then converted to frames via:
        num_frames = total_latents * 4 + 1

    Example: "s-3, right-10, left-20, right-10" → 43 latents → 173 frames.
    """
    total = 0
    for token in pose_string.split(","):
        token = token.strip()
        if not token:
            continue
        parts = token.split("-", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid pose token: '{token}'")
        total += int(float(parts[1].strip()))
    return total * 4 + 1


def _build_daemon_task_list(
    runner,
    work_queue: List[Tuple[str, dict, Optional[Path], Path]],
) -> list:
    """Translate a work-queue slice into the JSON task list expected by daemon scripts."""
    tasks_for_daemon = []
    for task_id, task, init_frame, out_path in work_queue:
        out_path.parent.mkdir(parents=True, exist_ok=True)

        prompts = task.get("prompts") or {}
        prompt = (prompts.get(runner.prompt_field) or "").strip()

        camera_control: Optional[str] = None
        if runner.camera_control_field:
            cc = (task.get("camera_control") or {}).get(runner.camera_control_field)
            if cc and str(cc).strip().lower() not in ("null", "none", ""):
                camera_control = str(cc).strip()
        effective_cc = camera_control or getattr(runner, "camera_control_default", None) or ""

        num_frames: Optional[int] = None
        if effective_cc:
            try:
                num_frames = _num_frames_from_pose(effective_cc)
            except Exception as e:
                print(f"[{runner.name}] WARN: could not compute num_frames for '{effective_cc}': {e}")

        tasks_for_daemon.append({
            "task_id": task_id,
            "prompt": prompt,
            "image": str(init_frame) if (init_frame and init_frame.exists()) else "",
            "output": str(out_path),
            "camera_control": effective_cc,
            "num_frames": num_frames,
        })
    return tasks_for_daemon


def _collect_daemon_results(
    runner,
    work_queue: List[Tuple[str, dict, Optional[Path], Path]],
    map_path: Path,
    map_lock: Lock,
) -> Dict[str, bool]:
    """Check which outputs were produced and update the output map."""
    results: Dict[str, bool] = {}
    for task_id, _task, _init_frame, out_path in work_queue:
        ok = out_path.exists()
        if ok:
            _update_map(map_path, task_id, out_path.name, map_lock)
            print(f"[{runner.name}] DONE  {task_id}")
        else:
            print(f"[{runner.name}] FAIL  {task_id}")
        results[task_id] = ok
    return results


def _run_batch_daemon(
    runner,
    work_queue: List[Tuple[str, dict, Optional[Path], Path]],
    map_path: Path,
    map_lock: Lock,
) -> Dict[str, bool]:
    """Launch daemon subprocess(es) to process all tasks.

    When the runner has gpu_per_task < n_gpu, tasks are distributed across
    multiple daemon processes running in parallel (one per GPU slice).
    Otherwise a single daemon handles the full queue sequentially.
    """
    gpu_per_task: Optional[int] = getattr(runner, "gpu_per_task", None)
    total_gpus: int = runner.n_gpu or 1

    if gpu_per_task is not None and total_gpus > gpu_per_task:
        return _run_batch_daemon_task_parallel(
            runner, work_queue, map_path, map_lock,
            gpu_per_task=gpu_per_task,
            n_workers=total_gpus // gpu_per_task,
        )

    # ── Single-daemon path (original behaviour) ──────────────────────────────
    tasks_for_daemon = _build_daemon_task_list(runner, work_queue)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as tmp:
        json.dump(tasks_for_daemon, tmp, indent=2)
        tasks_json_path = tmp.name

    try:
        cmd = runner.build_daemon_cmd(tasks_json_path)
        print(f"[{runner.name}] Launching daemon ({len(work_queue)} tasks): {' '.join(cmd[:3])} ...")
        proc = subprocess.run(
            cmd,
            cwd=str(runner.repo_path),
            capture_output=not runner.verbose,
            text=not runner.verbose,
        )
        if not runner.verbose and proc.returncode != 0:
            if proc.stdout:
                print(proc.stdout, end="")
            if proc.stderr:
                print(proc.stderr, end="")
        if proc.returncode != 0:
            print(f"[{runner.name}] Daemon exited with code {proc.returncode}")
    finally:
        Path(tasks_json_path).unlink(missing_ok=True)

    return _collect_daemon_results(runner, work_queue, map_path, map_lock)


def _run_batch_daemon_task_parallel(
    runner,
    work_queue: List[Tuple[str, dict, Optional[Path], Path]],
    map_path: Path,
    map_lock: Lock,
    gpu_per_task: int,
    n_workers: int,
) -> Dict[str, bool]:
    """Run N daemon processes in parallel, each pinned to a disjoint GPU slice.

    Tasks are distributed round-robin across workers so each worker gets a
    roughly equal share.  Each worker is launched with CUDA_VISIBLE_DEVICES
    set to its assigned GPUs and --nproc_per_node=gpu_per_task.
    """
    # Round-robin partition
    chunks = [work_queue[i::n_workers] for i in range(n_workers)]
    chunks = [c for c in chunks if c]   # drop empty tail slots
    actual_workers = len(chunks)

    print(
        f"[{runner.name}] Task-parallel daemon: {actual_workers} worker(s) × "
        f"{gpu_per_task} GPU(s) each — {len(work_queue)} tasks total"
    )

    tmp_files: List[str] = []
    procs: List[Tuple[subprocess.Popen, List, str, int]] = []

    for w_idx, chunk in enumerate(chunks):
        gpu_start = w_idx * gpu_per_task
        gpu_ids = ",".join(str(g) for g in range(gpu_start, gpu_start + gpu_per_task))

        task_list = _build_daemon_task_list(runner, chunk)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as tmp:
            json.dump(task_list, tmp, indent=2)
            tmp_path = tmp.name
        tmp_files.append(tmp_path)

        port = 29500 + w_idx
        cmd = runner.build_daemon_cmd(tmp_path, nproc_per_node=gpu_per_task, master_port=port)

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_ids

        print(f"[{runner.name}] Worker {w_idx}: GPU={gpu_ids}, {len(chunk)} task(s)")
        proc = subprocess.Popen(
            cmd,
            cwd=str(runner.repo_path),
            env=env,
            stdout=None if runner.verbose else subprocess.PIPE,
            stderr=None if runner.verbose else subprocess.PIPE,
            text=True,
        )
        procs.append((proc, chunk, gpu_ids, w_idx))

    # Collect output concurrently to avoid pipe-buffer deadlocks.
    # If any worker fails, kill the rest immediately to avoid GPU memory leaks.
    def _wait(item):
        proc, chunk, gpu_ids, w_idx = item
        stdout, stderr = proc.communicate()
        return proc.returncode, stdout, stderr, gpu_ids, w_idx

    with ThreadPoolExecutor(max_workers=actual_workers) as pool:
        futures = {pool.submit(_wait, item): item for item in procs}
        any_failed = False
        for future in as_completed(futures):
            returncode, stdout, stderr, gpu_ids, w_idx = future.result()
            if returncode != 0:
                any_failed = True
                print(f"[{runner.name}] Worker {w_idx} (GPU={gpu_ids}) exited {returncode}")
                if stdout:
                    print(stdout, end="")
                if stderr:
                    print(stderr, end="")
                # Kill any still-running sibling workers to free GPU memory.
                for other_proc, _, other_gpus, other_idx in procs:
                    if other_proc.poll() is None:
                        print(
                            f"[{runner.name}] Killing worker {other_idx} "
                            f"(GPU={other_gpus}) due to sibling failure."
                        )
                        other_proc.kill()

    for f in tmp_files:
        Path(f).unlink(missing_ok=True)

    return _collect_daemon_results(runner, work_queue, map_path, map_lock)


# ---------------------------------------------------------------------------
# Per-model run loop
# ---------------------------------------------------------------------------

def run_model(
    *,
    runner: WorldModelRunner,
    model_type: str,
    tasks: List[Tuple[str, dict, Optional[Path]]],
    out_dir: Path,
    run_name: str,
    workers: int,
    overwrite: bool,
    rpm_limit: Optional[int],
) -> Dict[str, bool]:
    """Run all tasks through one model. Returns {task_id: success}."""
    out_dir.mkdir(parents=True, exist_ok=True)
    map_path = out_dir / f"{runner.name}_output_map_{run_name}.json"
    map_lock = Lock()
    limiter = RpmLimiter(rpm_limit) if rpm_limit else None

    # Build work queue — skip tasks that are already done unless --overwrite
    work_queue: List[Tuple[str, dict, Optional[Path], Path]] = []
    for task_id, task, init_frame in tasks:
        out_path = out_dir / f"{task_id}.mp4"
        if out_path.exists() and not overwrite:
            print(f"[{runner.name}] SKIP {task_id} (already exists)")
            continue
        work_queue.append((task_id, task, init_frame, out_path))

    print(f"\n[{runner.name}] {len(work_queue)} task(s) to generate → {out_dir}")

    # Batch mode: single subprocess invocation, model loaded once
    if getattr(runner, "batch_script", None) and work_queue:
        return _run_batch_local(runner, work_queue, map_path, map_lock)

    if limiter:
        print(f"[{runner.name}] RPM limit: {rpm_limit} requests/min")

    # Daemon path: load weights once, iterate the whole queue in one subprocess.
    if work_queue and getattr(runner, "daemon_script", None):
        return _run_batch_daemon(runner, work_queue, map_path, map_lock)

    results: Dict[str, bool] = {}

    def do_one(item: Tuple) -> Tuple[str, bool]:
        task_id, task, init_frame, out_path = item
        if limiter:
            limiter.acquire()
        ok = _run_one(runner, task_id, task, init_frame, out_path)
        if ok:
            _update_map(map_path, task_id, out_path.name, map_lock)
            print(f"[{runner.name}] DONE  {task_id}")
        else:
            print(f"[{runner.name}] FAIL  {task_id}")
        return task_id, ok

    # Local models run sequentially to avoid GPU resource contention.
    # API models can be parallelized via --workers.
    effective_workers = 1 if model_type == "local" else max(1, workers)

    if effective_workers == 1:
        for item in work_queue:
            task_id, ok = do_one(item)
            results[task_id] = ok
    else:
        with ThreadPoolExecutor(max_workers=effective_workers) as pool:
            futures = {pool.submit(do_one, item): item[0] for item in work_queue}
            for future in as_completed(futures):
                task_id, ok = future.result()
                results[task_id] = ok

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _load_models_config(config_path: Path) -> dict:
    with config_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f).get("models", {})


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run benchmark tasks through world models and collect video outputs."
    )
    parser.add_argument(
        "--models", nargs="+", required=True, metavar="MODEL",
        help="Model name(s) defined in configs/models.yaml, or 'all' to run every model.",
    )
    parser.add_argument(
        "--tasks_root", default="benchmark/tasks/",
        help="Root directory of task YAML folders (default: benchmark/tasks/).",
    )
    parser.add_argument(
        "--pattern", default=None, metavar="GLOB",
        help="Glob pattern to filter task IDs, e.g. 'pouring_water_into_cup*'. "
             "If omitted, all tasks under tasks_root are processed.",
    )
    parser.add_argument(
        "--output_root", default="outputs/",
        help="Root directory for model output folders (default: outputs/).",
    )
    parser.add_argument(
        "--run_name", required=True,
        help="Label appended to each output folder, e.g. 'test_2' → outputs/veo_test_2/.",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help=(
            "Number of parallel workers for API models (default: 1). "
            "Local models always run sequentially regardless of this setting."
        ),
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-generate videos that already exist on disk.",
    )
    parser.add_argument(
        "--n_gpu", type=int, default=None,
        help="Override the n_gpu value from the model config for all local models.",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Stream script stdout/stderr live to the terminal (local models only).",
    )
    parser.add_argument(
        "--config", default="generation/configs/models.yaml",
        help="Path to the model registry YAML (default: generation/configs/models.yaml).",
    )
    args = parser.parse_args()

    # ---- Load config & instantiate runners ----
    config_path = Path(args.config).expanduser().resolve()
    if not config_path.exists():
        print(f"[ERROR] Config not found: {config_path}", file=sys.stderr)
        return 1

    models_config = _load_models_config(config_path)
    requested = list(models_config) if args.models == ["all"] else args.models

    runners: Dict[str, WorldModelRunner] = {}
    for name in requested:
        if name not in models_config:
            print(
                f"[ERROR] Unknown model '{name}'. "
                f"Available: {list(models_config)}",
                file=sys.stderr,
            )
            return 1
        try:
            cfg = models_config[name]
            if cfg.get("type") == "local":
                overrides = {}
                if args.n_gpu is not None:
                    overrides["n_gpu"] = args.n_gpu
                if args.verbose:
                    overrides["verbose"] = True
                if overrides:
                    cfg = {**cfg, **overrides}
            runners[name] = build_runner(name, cfg)
        except Exception as e:
            print(f"[ERROR] Could not initialize runner for '{name}': {e}", file=sys.stderr)
            return 1

    # ---- Discover tasks ----
    tasks_root = Path(args.tasks_root).expanduser().resolve()
    if not tasks_root.exists():
        print(f"[ERROR] tasks_root not found: {tasks_root}", file=sys.stderr)
        return 1

    tasks = discover_tasks(tasks_root)
    if not tasks:
        print(f"[ERROR] No tasks found under: {tasks_root}", file=sys.stderr)
        return 1

    if args.pattern:
        tasks = [(tid, t, f) for tid, t, f in tasks if fnmatch.fnmatch(tid, args.pattern)]
        if not tasks:
            print(f"[ERROR] No tasks match pattern '{args.pattern}'", file=sys.stderr)
            return 1

    print(f"Discovered {len(tasks)} task(s) under {tasks_root}")

    output_root = Path(args.output_root).expanduser().resolve()

    # ---- Run each model in sequence (tasks within a model may be parallel) ----
    overall_failed: Dict[str, List[str]] = {}

    for model_name, runner in runners.items():
        model_tasks = tasks
        if runner.camera_control_field:
            model_tasks = _dedup_for_camera_model(tasks)
            print(
                f"[{model_name}] Camera-controlled: deduplicated"
                f" {len(tasks)} → {len(model_tasks)} tasks (one per occlusion group)"
            )

        out_dir = output_root / f"{model_name}_{args.run_name}"
        results = run_model(
            runner=runner,
            model_type=models_config[model_name].get("type", "local"),
            tasks=model_tasks,
            out_dir=out_dir,
            run_name=args.run_name,
            workers=args.workers,
            overwrite=args.overwrite,
            rpm_limit=models_config[model_name].get("rpm_limit") or None,
        )
        failed = [tid for tid, ok in results.items() if not ok]
        if failed:
            overall_failed[model_name] = failed

    # ---- Summary ----
    print("\n" + "=" * 50)
    if not overall_failed:
        print(f"All tasks completed successfully across {len(runners)} model(s).")
        return 0
    else:
        for model_name, failed_tasks in overall_failed.items():
            print(f"[{model_name}] {len(failed_tasks)} failure(s):")
            for t in failed_tasks:
                print(f"    - {t}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
