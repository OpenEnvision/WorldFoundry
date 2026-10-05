#!/usr/bin/env python3
"""Persistent SAM3 HTTP service for physical preprocessing.

The service keeps one SAM3 predictor alive and exposes a small JSON endpoint
used by scripts/score_video_physical_3d.py. Start one service per GPU.

Example from the OpenWorldLib root:
CUDA_VISIBLE_DEVICES=2 python worldeval/physical/serve_sam3.py \
  --model worldeval/weights/sam3/sam3.pt --device cuda:0 --port 8090
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPO_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.vlm import resolve_model_name, resolve_vlm_backend

DEFAULT_MAX_FRAMES = 128
DEFAULT_SAM_CONF = 0.7
DEFAULT_MIN_MASK_AREA_RATIO = 0.005
DEFAULT_MAX_MASKS_PER_FRAME = 3


def resolve_model_arg(value: str) -> str:
    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path)
    project_candidate = PROJECT_ROOT / path
    if project_candidate.exists():
        return str(project_candidate.resolve())
    repo_candidate = REPO_ROOT / path
    if repo_candidate.exists():
        return str(repo_candidate.resolve())
    return value


class SAM3RequestHandler(BaseHTTPRequestHandler):
    server_version = "SAM3HTTP/0.1"

    def log_message(self, format: str, *args: Any) -> None:
        if getattr(self.server, "quiet", False):
            return
        super().log_message(format, *args)

    def _write_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/health":
            cuda_info: dict[str, Any] = {}
            try:
                import torch

                cuda_info = {
                    "torch_cuda_available": torch.cuda.is_available(),
                    "torch_current_device": torch.cuda.current_device() if torch.cuda.is_available() else None,
                    "torch_device_name": (
                        torch.cuda.get_device_name(torch.cuda.current_device())
                        if torch.cuda.is_available()
                        else None
                    ),
                    "torch_device_count": torch.cuda.device_count(),
                }
            except Exception as exc:
                cuda_info = {"torch_cuda_error": str(exc)}
            self._write_json(
                200,
                {
                    "status": "ok",
                    "model": getattr(self.server, "sam3_model_path", None),
                    "device": getattr(self.server, "device", None),
                    "conf": getattr(self.server, "conf", None),
                    "iou": getattr(self.server, "iou", None),
                    "max_frames": getattr(self.server, "max_frames", None),
                    "min_mask_area_ratio": getattr(self.server, "min_mask_area_ratio", None),
                    "max_masks_per_frame": getattr(self.server, "max_masks_per_frame", None),
                    "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES", ""),
                    "cuda_device_order": os.getenv("CUDA_DEVICE_ORDER", ""),
                    **cuda_info,
                },
            )
            return
        self._write_json(404, {"error": f"Unsupported path: {self.path}"})

    def do_POST(self) -> None:
        if self.path != "/segment":
            self._write_json(404, {"error": f"Unsupported path: {self.path}"})
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
            video_path = payload.get("video_path") or payload.get("video")
            if not video_path:
                raise ValueError("Request JSON must contain 'video_path'.")

            backend = resolve_vlm_backend(payload.get("backend") or getattr(self.server, "vlm_backend", None))
            model_name = resolve_model_name(payload.get("model_name") or getattr(self.server, "vlm_model", None), backend)
            max_frames = int(payload.get("max_frames") or getattr(self.server, "max_frames", DEFAULT_MAX_FRAMES))
            min_mask_area_ratio = float(
                payload.get("min_mask_area_ratio")
                if payload.get("min_mask_area_ratio") is not None
                else getattr(self.server, "min_mask_area_ratio", DEFAULT_MIN_MASK_AREA_RATIO)
            )
            max_masks_per_frame = int(
                payload.get("max_masks_per_frame")
                if payload.get("max_masks_per_frame") is not None
                else getattr(self.server, "max_masks_per_frame", DEFAULT_MAX_MASKS_PER_FRAME)
            )

            with self.server.inference_lock:
                result = self.server.run_sam_process(
                    video_path=str(Path(video_path).resolve()),
                    video_prompt=payload.get("video_prompt") or payload.get("prompt"),
                    model_name=model_name,
                    backend=backend,
                    sam3_model_path=getattr(self.server, "sam3_model_path"),
                    device=getattr(self.server, "device"),
                    max_frames=max_frames,
                    sam3_predictor=getattr(self.server, "predictor"),
                    sam_conf=getattr(self.server, "conf"),
                    sam_iou=getattr(self.server, "iou"),
                    min_mask_area_ratio=min_mask_area_ratio,
                    max_masks_per_frame=max_masks_per_frame,
                )

            result["sam3_server"] = {
                "url": f"http://{self.server.server_address[0]}:{self.server.server_address[1]}",
                "model": getattr(self.server, "sam3_model_path"),
                "device": getattr(self.server, "device"),
            }
            self._write_json(200, {"result": result})
        except Exception as exc:
            self._write_json(
                500,
                {
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a persistent SAM3 segmentation service")
    parser.add_argument("--host", default=os.getenv("SAM3_HOST", "127.0.0.1"), help="Bind host")
    parser.add_argument("--port", type=int, default=int(os.getenv("SAM3_PORT", "8090")), help="Bind port")
    parser.add_argument("--model", default=os.getenv("SAM3_MODEL", "worldeval/weights/sam3/sam3.pt"), help="SAM3 weights path")
    parser.add_argument("--device", default=os.getenv("SAM3_DEVICE", "cuda:0"), help="SAM3 device inside this process")
    parser.add_argument("--vlm-backend", default=os.getenv("VLM_BACKEND"), help="Default VLM backend for object extraction")
    parser.add_argument("--vlm-model", default=os.getenv("VLM_MODEL"), help="Default VLM model for object extraction")
    parser.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES)
    parser.add_argument("--conf", type=float, default=DEFAULT_SAM_CONF)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument(
        "--min-mask-area-ratio",
        type=float,
        default=DEFAULT_MIN_MASK_AREA_RATIO,
        help="Drop SAM masks smaller than this fraction of frame area.",
    )
    parser.add_argument(
        "--max-masks-per-frame",
        type=int,
        default=DEFAULT_MAX_MASKS_PER_FRAME,
        help="Keep only the largest N masks per frame after confidence and area filtering; <=0 keeps all.",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress HTTP access logs")
    args = parser.parse_args()

    from physical.sam_process import build_sam3_predictor, run_sam_process

    sam3_model_path = resolve_model_arg(args.model)
    print(f"Loading persistent SAM3 predictor: model={sam3_model_path}, device={args.device}", flush=True)
    predictor = build_sam3_predictor(
        sam3_model_path=sam3_model_path,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
    )
    print("SAM3 predictor is ready.", flush=True)

    server = ThreadingHTTPServer((args.host, args.port), SAM3RequestHandler)
    server.predictor = predictor
    server.sam3_model_path = sam3_model_path
    server.device = args.device
    server.vlm_backend = args.vlm_backend
    server.vlm_model = args.vlm_model
    server.max_frames = args.max_frames
    server.conf = args.conf
    server.iou = args.iou
    server.min_mask_area_ratio = args.min_mask_area_ratio
    server.max_masks_per_frame = args.max_masks_per_frame
    server.quiet = args.quiet
    server.inference_lock = threading.Lock()
    server.run_sam_process = run_sam_process

    print(f"SAM3 service listening on http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down SAM3 service.", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
