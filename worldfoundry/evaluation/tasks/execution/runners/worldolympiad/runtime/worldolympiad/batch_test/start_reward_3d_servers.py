#!/usr/bin/env python3
"""Start one persistent 3D reward service per selected GPU."""

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Start DA3/3D reward service processes on multiple GPUs")
    parser.add_argument("--worldeval-root", default="worldeval", help="Path to worldeval submodule directory")
    parser.add_argument("--gpus", required=True, help="Comma-separated physical GPU ids, e.g. 3,4")
    parser.add_argument("--ports", required=True, help="Comma-separated service ports, e.g. 8092,8093")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    parser.add_argument("--model", default="worldeval/weights/da3", help="DA3 model path or name")
    parser.add_argument("--vlm-backend", default="qwenvl_server", help="VLM backend for GS/meta scoring")
    parser.add_argument("--vlm-model", default=None, help="VLM model for GS/meta scoring")
    parser.add_argument(
        "--qwen-server-urls",
        default="",
        help="Optional comma-separated QwenVL service URLs assigned round-robin to DA3 services.",
    )
    parser.add_argument("--lpips", dest="lpips", action="store_true", help="Default to LPIPS GS scoring")
    parser.add_argument("--no-lpips", dest="lpips", action="store_false", help="Default to VLM GS scoring")
    parser.set_defaults(lpips=False)
    args = parser.parse_args()

    worldeval_root = Path(args.worldeval_root).resolve()
    project_root = resolve_project_root(worldeval_root)
    server_script = worldeval_root / "3d_metrics" / "serve_reward_3d.py"
    if not server_script.exists():
        raise FileNotFoundError(f"3D reward server script not found: {server_script}")
    model_name = resolve_model_arg(project_root, worldeval_root, args.model)

    gpus = parse_csv(args.gpus)
    ports = parse_csv(args.ports)
    qwen_urls = parse_csv(args.qwen_server_urls)
    if len(gpus) != len(ports):
        raise ValueError("--gpus and --ports must have the same number of entries.")

    processes: list[subprocess.Popen] = []
    for index, (gpu, port) in enumerate(zip(gpus, ports)):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        if qwen_urls:
            env["QWENVL_SERVER_URL"] = qwen_urls[index % len(qwen_urls)].rstrip("/")

        command = [
            sys.executable,
            str(server_script),
            "--host",
            args.host,
            "--port",
            port,
            "--model-name",
            model_name,
            "--gpus",
            gpu,
            "--num-workers",
            "1",
            "--vlm-backend",
            args.vlm_backend,
        ]
        if args.vlm_model:
            command.extend(["--vlm-model", args.vlm_model])
        command.append("--lpips" if args.lpips else "--no-lpips")

        print(f"Starting 3D reward server on GPU {gpu}, port {port}")
        process = subprocess.Popen(command, cwd=str(project_root), env=env)
        processes.append(process)

    def shutdown(signum=None, frame=None):
        print("\nStopping 3D reward service processes...")
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
        raise SystemExit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    print("3D reward services are running:")
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
