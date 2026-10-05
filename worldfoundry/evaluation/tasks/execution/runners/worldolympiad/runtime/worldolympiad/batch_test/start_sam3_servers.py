#!/usr/bin/env python3
"""Start one persistent SAM3 service per selected GPU."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def resolve_project_root(worldeval_root: Path) -> Path:
    return worldeval_root.parent


def resolve_model_arg(project_root: Path, worldeval_root: Path, value: str) -> str:
    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path)
    project_candidate = project_root / path
    if project_candidate.exists():
        return str(project_candidate.resolve())
    worldeval_candidate = worldeval_root / path
    if worldeval_candidate.exists():
        return str(worldeval_candidate.resolve())
    return value


def query_gpu_uuid_by_index() -> dict[str, str]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    mapping: dict[str, str] = {}
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",", maxsplit=1)]
        if len(parts) == 2 and parts[0] and parts[1]:
            mapping[parts[0]] = parts[1]
    return mapping


def resolve_visible_gpu(gpu: str, *, use_uuid: bool, uuid_by_index: dict[str, str] | None) -> str:
    if not use_uuid or gpu.startswith("GPU-"):
        return gpu
    if not gpu.isdigit():
        return gpu
    if uuid_by_index is None:
        raise RuntimeError("GPU UUID mapping was not loaded.")
    try:
        return uuid_by_index[gpu]
    except KeyError as exc:
        valid = ", ".join(sorted(uuid_by_index))
        raise ValueError(f"GPU index {gpu!r} was not found by nvidia-smi. Valid indexes: {valid}") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Start SAM3 service processes on multiple GPUs")
    parser.add_argument("--worldeval-root", default="worldeval", help="Path to worldeval submodule directory")
    parser.add_argument("--gpus", required=True, help="Comma-separated physical GPU ids, e.g. 2,3")
    parser.add_argument("--ports", required=True, help="Comma-separated service ports, e.g. 8090,8091")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    parser.add_argument("--model", default="worldeval/weights/sam3/sam3.pt", help="SAM3 weights path passed to the service")
    parser.add_argument("--device", default="cuda:0", help="Visible CUDA device inside each service process")
    parser.add_argument("--vlm-backend", default="qwenvl_server", help="VLM backend used for object extraction")
    parser.add_argument("--vlm-model", default=None, help="VLM model used for object extraction")
    parser.add_argument(
        "--qwen-server-urls",
        default="",
        help="Optional comma-separated QwenVL service URLs assigned round-robin to SAM3 services.",
    )
    parser.add_argument(
        "--no-gpu-uuid",
        action="store_true",
        help=(
            "Use the raw --gpus values in CUDA_VISIBLE_DEVICES. By default, numeric GPU indexes "
            "are resolved to GPU UUIDs with nvidia-smi to avoid CUDA/NVML index mismatches."
        ),
    )
    parser.add_argument("--max-frames", type=int, default=128)
    parser.add_argument("--quiet", action="store_true", help="Suppress HTTP access logs")
    args = parser.parse_args()

    worldeval_root = Path(args.worldeval_root).resolve()
    project_root = resolve_project_root(worldeval_root)
    server_script = worldeval_root / "physical" / "serve_sam3.py"
    if not server_script.exists():
        raise FileNotFoundError(f"SAM3 server script not found: {server_script}")
    sam3_model = resolve_model_arg(project_root, worldeval_root, args.model)

    gpus = parse_csv(args.gpus)
    ports = parse_csv(args.ports)
    qwen_urls = parse_csv(args.qwen_server_urls)
    if len(gpus) != len(ports):
        raise ValueError("--gpus and --ports must have the same number of entries.")

    use_gpu_uuid = not args.no_gpu_uuid
    uuid_by_index = None
    if use_gpu_uuid and any(gpu.isdigit() for gpu in gpus):
        try:
            uuid_by_index = query_gpu_uuid_by_index()
        except (OSError, subprocess.CalledProcessError) as exc:
            raise RuntimeError(
                "Failed to resolve numeric --gpus to GPU UUIDs with nvidia-smi. "
                "Either fix nvidia-smi access, pass GPU UUIDs directly, or use --no-gpu-uuid."
            ) from exc

    processes: list[subprocess.Popen] = []
    for index, (gpu, port) in enumerate(zip(gpus, ports)):
        visible_gpu = resolve_visible_gpu(gpu, use_uuid=use_gpu_uuid, uuid_by_index=uuid_by_index)
        env = os.environ.copy()
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["CUDA_VISIBLE_DEVICES"] = visible_gpu
        if qwen_urls:
            env["QWENVL_SERVER_URL"] = qwen_urls[index % len(qwen_urls)].rstrip("/")
        command = [
            sys.executable,
            str(server_script),
            "--model",
            sam3_model,
            "--device",
            args.device,
            "--host",
            args.host,
            "--port",
            port,
            "--vlm-backend",
            args.vlm_backend,
            "--max-frames",
            str(args.max_frames),
        ]
        if args.vlm_model:
            command.extend(["--vlm-model", args.vlm_model])
        if args.quiet:
            command.append("--quiet")

        print(
            f"Starting SAM3 server on requested GPU {gpu}, "
            f"CUDA_VISIBLE_DEVICES={visible_gpu}, port {port}"
        )
        process = subprocess.Popen(command, cwd=str(project_root), env=env)
        processes.append(process)

    def shutdown(signum=None, frame=None):
        print("\nStopping SAM3 service processes...")
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
        raise SystemExit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    print("SAM3 services are running:")
    for port in ports:
        print(f"  http://{args.host}:{port}")
    print("Press Ctrl-C to stop.")

    while True:
        for process in processes:
            if process.poll() is not None:
                shutdown()
        time.sleep(2)


if __name__ == "__main__":
    main()
