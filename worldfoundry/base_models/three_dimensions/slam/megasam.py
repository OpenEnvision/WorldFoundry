"""MegaSAM runtime and pose precompute entry points used by WBench."""

from __future__ import annotations

import argparse
import multiprocessing
import os
import shutil
import subprocess
import sys
from pathlib import Path

from worldfoundry.base_models.capabilities import BASE_MODEL_CAPABILITIES

RUNTIME_ROOT = Path(__file__).resolve().parent / "mega_sam_runtime"


def _asset_path(asset_id: str) -> Path:
    for asset in BASE_MODEL_CAPABILITIES["wbench_megasam"].assets:
        if asset.id == asset_id:
            status = asset.check()
            return Path(status["matched_path"] or status["local_path"])
    raise RuntimeError(f"wbench_megasam asset is not registered: {asset_id}")


def runtime_root() -> Path:
    """Return the canonical MegaSAM runtime."""
    return RUNTIME_ROOT


def checkpoint_path() -> Path:
    return _asset_path("wbench_megasam_checkpoint")


def depth_anything_checkpoint_path() -> Path:
    return _asset_path("wbench_megasam_depth_anything_checkpoint")


def weights_dir() -> Path:
    return checkpoint_path().parent


def compute_stride(video_path: str | os.PathLike[str], target_fps: float = 15.0) -> tuple[int, float, float]:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    cap.release()
    stride = max(1, int(fps / target_fps))
    return stride, fps, fps / stride


def extract_frames(video_path: str | os.PathLike[str], frames_dir: Path, stride: int = 1) -> int:
    import cv2

    frames_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    idx, saved = 0, 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if idx % stride == 0:
            cv2.imwrite(str(frames_dir / f"{saved:05d}.jpg"), frame)
            saved += 1
        idx += 1
    cap.release()
    return saved


def setup_env(device: str | int | None = None) -> dict[str, str]:
    """Prepare a worker process without writing caches into the source tree."""
    env = os.environ.copy()
    if device is not None and "CUDA_VISIBLE_DEVICES" not in env:
        env["CUDA_VISIBLE_DEVICES"] = str(device)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(Path(__file__).resolve().parents[4]), env.get("PYTHONPATH")]))
    return env


def run_single(
    video_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    *,
    device: str | int = "0",
    target_fps: float = 15.0,
    cpu_list: str | None = None,
    n_threads: int = 4,
) -> None:
    """Run the shared resident pipeline in a GPU-isolated Python worker."""
    env = setup_env(device)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = str(n_threads)
    command = [
        sys.executable, "-m", "worldfoundry.base_models.three_dimensions.slam.megasam",
        "--video", str(Path(video_path).resolve()), "--output", str(Path(output_path).resolve()),
        "--target_fps", str(target_fps), "--worker",
    ]
    if cpu_list and shutil.which("taskset"):
        command = ["taskset", "-c", cpu_list, *command]
    subprocess.run(command, env=env, check=True)


def _gpu_worker_process(gpu_id: int, worker_idx: int, n_workers: int, task_list: list[tuple[str, str]], target_fps: float) -> None:
    try:
        available = sorted(os.sched_getaffinity(0))
    except AttributeError:
        available = list(range(os.cpu_count() or 64))
    total = len(available)
    n_cores = max(1, total // n_workers)
    n_threads = max(1, total // (2 * n_workers))
    start = worker_idx * n_cores
    cpu_ids = available[start : start + n_cores]
    cpu_list = ",".join(str(c) for c in cpu_ids)

    try:
        os.sched_setaffinity(0, cpu_ids)
    except (AttributeError, OSError):
        pass

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["OMP_NUM_THREADS"] = str(n_threads)
    os.environ["MKL_NUM_THREADS"] = str(n_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n_threads)

    tag = f"[GPU{gpu_id}]"
    cpu_span = f"{cpu_ids[0]}-{cpu_ids[-1]}" if cpu_ids else "n/a"
    print(f"  {tag} Worker started: {len(task_list)} videos, cpus={cpu_span}, threads={n_threads}", flush=True)

    ok, fail = 0, 0
    for idx, (video_path, output_path) in enumerate(task_list):
        print(f"  {tag} [{idx + 1}/{len(task_list)}] {os.path.basename(video_path)}", flush=True)
        try:
            run_single(video_path, output_path, device="0", target_fps=target_fps, cpu_list=cpu_list, n_threads=n_threads)
            ok += 1
        except subprocess.CalledProcessError as exc:
            stderr_msg = exc.stderr.decode(errors="replace")[-500:] if exc.stderr else "no stderr"
            print(f"  {tag} FAIL {os.path.basename(video_path)}:\n    {stderr_msg}", flush=True)
            fail += 1
        except Exception as exc:
            print(f"  {tag} FAIL {os.path.basename(video_path)}: {exc}", flush=True)
            fail += 1
    print(f"  {tag} Done: {ok}/{len(task_list)} ok, {fail} fail", flush=True)


def run_batch(
    video_dir: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    *,
    gpus: str = "0",
    target_fps: float = 15.0,
    force: bool = False,
) -> None:
    gpu_ids = [int(gpu) for gpu in str(gpus).split(",") if str(gpu).strip()]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted(Path(video_dir).glob("case_*_combined.mp4"))
    tasks: list[tuple[str, str]] = []
    for video in videos:
        output = output_dir / f"{video.stem}.npz"
        if output.exists() and not force:
            continue
        tasks.append((str(video), str(output)))

    print(f"Found {len(videos)} videos, {len(tasks)} to process, {len(gpu_ids)} GPUs")
    if not tasks:
        return

    n_workers = min(len(gpu_ids), len(tasks))
    worker_tasks = [[] for _ in range(n_workers)]
    for idx, task in enumerate(tasks):
        worker_tasks[idx % n_workers].append(task)

    ctx = multiprocessing.get_context("spawn")
    processes = []
    for worker_idx in range(n_workers):
        process = ctx.Process(
            target=_gpu_worker_process,
            args=(gpu_ids[worker_idx], worker_idx, n_workers, worker_tasks[worker_idx], target_fps),
        )
        process.start()
        processes.append(process)
    for process in processes:
        process.join()
        if process.exitcode:
            raise RuntimeError(f"MegaSAM worker exited with code {process.exitcode}")
    print("Done: all workers finished")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MegaSAM pose inference")
    parser.add_argument("--video", type=str, help="Single video path")
    parser.add_argument("--video_dir", type=str, help="Batch: video directory")
    parser.add_argument("--output", type=str, help="Output .npz path in single mode")
    parser.add_argument("--output_dir", type=str, help="Output directory in batch mode")
    parser.add_argument("--gpus", type=str, default="0", help="GPU IDs (comma-separated)")
    parser.add_argument("--target_fps", type=float, default=15.0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.video:
        output = args.output or str(Path(args.video).with_suffix(".npz"))
        if args.worker:
            from worldfoundry.base_models.three_dimensions.slam.megasam_resident import ResidentMegaSamPipeline

            ResidentMegaSamPipeline(target_fps=args.target_fps).evaluate(Path(args.video), Path(output))
        else:
            run_single(args.video, output, device=args.gpus.split(",")[0], target_fps=args.target_fps)
        return 0

    if args.video_dir:
        output_dir = args.output_dir or str(Path(args.video_dir).parent / "megasam")
        run_batch(args.video_dir, output_dir, gpus=args.gpus, target_fps=args.target_fps, force=args.force)
        return 0

    parser.error("--video or --video_dir is required")
    return 2


__all__ = [
    "RUNTIME_ROOT",
    "checkpoint_path",
    "compute_stride",
    "depth_anything_checkpoint_path",
    "extract_frames",
    "main",
    "run_batch",
    "run_single",
    "runtime_root",
    "setup_env",
    "weights_dir",
]


if __name__ == "__main__":
    raise SystemExit(main())
