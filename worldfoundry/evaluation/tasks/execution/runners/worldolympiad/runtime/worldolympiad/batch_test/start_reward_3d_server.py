#!/usr/bin/env python3
"""Start a persistent 3D reward service backed by DA3 worker processes."""

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
    parser = argparse.ArgumentParser(description="Start the persistent DA3/3D reward service")
    parser.add_argument("--worldeval-root", default="worldeval", help="Path to worldeval submodule directory")
    parser.add_argument("--gpus", required=True, help="Comma-separated physical GPU ids used by DA3 workers")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", default="8092", help="Service port")
    parser.add_argument("--model", default="worldeval/weights/da3", help="DA3 model path or name")
    parser.add_argument("--num-workers", type=int, default=0, help="Defaults to number of --gpus entries")
    parser.add_argument("--vlm-backend", default="qwenvl_server", help="VLM backend for GS/meta scoring")
    parser.add_argument("--vlm-model", default=None, help="VLM model for GS/meta scoring")
    parser.add_argument("--qwen-server-url", default="", help="QwenVL service URL used by 3D VLM scoring")
    parser.add_argument("--lpips", dest="lpips", action="store_true", help="Default to LPIPS GS scoring when requests omit use_lpips")
    parser.add_argument("--no-lpips", dest="lpips", action="store_false", help="Default to VLM GS scoring when requests omit use_lpips")
    parser.set_defaults(lpips=False)
    args = parser.parse_args()

    worldeval_root = Path(args.worldeval_root).resolve()
    project_root = resolve_project_root(worldeval_root)
    server_script = worldeval_root / "3d_metrics" / "serve_reward_3d.py"
    if not server_script.exists():
        raise FileNotFoundError(f"3D reward server script not found: {server_script}")
    model_name = resolve_model_arg(project_root, worldeval_root, args.model)

    gpus = parse_csv(args.gpus)
    if not gpus:
        raise ValueError("--gpus must not be empty.")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)
    if args.qwen_server_url:
        env["QWENVL_SERVER_URL"] = args.qwen_server_url.rstrip("/")

    command = [
        sys.executable,
        str(server_script),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--model-name",
        model_name,
        "--gpus",
        ",".join(gpus),
        "--num-workers",
        str(args.num_workers or len(gpus)),
        "--vlm-backend",
        args.vlm_backend,
    ]
    if args.vlm_model:
        command.extend(["--vlm-model", args.vlm_model])
    if args.lpips:
        command.append("--lpips")
    else:
        command.append("--no-lpips")

    print(f"Starting 3D reward server on GPUs {','.join(gpus)}, port {args.port}")
    process = subprocess.Popen(command, cwd=str(project_root), env=env)

    def shutdown(signum=None, frame=None):
        print("\nStopping 3D reward service...")
        if process.poll() is None:
            process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
        raise SystemExit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    print(f"3D reward service is running: http://{args.host}:{args.port}")
    print("Press Ctrl-C to stop.")

    while True:
        if process.poll() is not None:
            shutdown()
        time.sleep(2)


if __name__ == "__main__":
    main()
