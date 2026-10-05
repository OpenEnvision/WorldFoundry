#!/usr/bin/env python3
"""Start one persistent QwenVL service per selected GPU."""

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
    parser = argparse.ArgumentParser(description="Start QwenVL service processes on multiple GPUs")
    parser.add_argument("--worldeval-root", default="worldeval", help="Path to worldeval submodule directory")
    parser.add_argument("--gpus", required=True, help="Comma-separated physical GPU ids, e.g. 0,1")
    parser.add_argument("--ports", required=True, help="Comma-separated service ports, e.g. 8008,8009")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    parser.add_argument("--model", default="worldeval/weights/QwenVL", help="Model path or name passed to qwen service")
    parser.add_argument("--warmup", action="store_true", help="Warm up each model at startup")
    parser.add_argument("--quiet", action="store_true", help="Suppress HTTP access logs")
    args = parser.parse_args()

    worldeval_root = Path(args.worldeval_root).resolve()
    project_root = resolve_project_root(worldeval_root)
    server_script = worldeval_root / "model" / "qwenvl_server.py"
    if not server_script.exists():
        raise FileNotFoundError(f"QwenVL server script not found: {server_script}")
    model_name = resolve_model_arg(project_root, worldeval_root, args.model)

    gpus = parse_csv(args.gpus)
    ports = parse_csv(args.ports)
    if len(gpus) != len(ports):
        raise ValueError("--gpus and --ports must have the same number of entries.")

    processes: list[subprocess.Popen] = []
    for gpu, port in zip(gpus, ports):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        env["QWENVL_DEVICE"] = "cuda:0"
        command = [
            sys.executable,
            str(server_script),
            "--model",
            model_name,
            "--host",
            args.host,
            "--port",
            port,
        ]
        if args.warmup:
            command.append("--warmup")
        if args.quiet:
            command.append("--quiet")

        print(f"Starting QwenVL server on GPU {gpu}, port {port}")
        process = subprocess.Popen(command, cwd=str(project_root), env=env)
        processes.append(process)

    def shutdown(signum=None, frame=None):
        print("\nStopping QwenVL service processes...")
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

    print("QwenVL services are running:")
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
