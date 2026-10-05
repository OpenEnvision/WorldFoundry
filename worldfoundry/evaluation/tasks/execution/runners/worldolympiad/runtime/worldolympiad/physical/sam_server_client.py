"""Lightweight client helpers for the persistent SAM3 service."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def sam3_server_url() -> str | None:
    raw = (
        os.getenv("SAM3_SERVER_URL")
        or os.getenv("SAM_SERVER_URL")
        or os.getenv("PHYSICAL_SAM_SERVER_URL")
        or ""
    ).strip()
    return raw.rstrip("/") if raw else None


def run_sam_process_via_server(
    *,
    server_url: str,
    video_path: str,
    video_prompt: str | None,
    model_name: str | None = None,
    backend: str | None = None,
    sam3_model_path: str | None = None,
    device: str | None = None,
    max_frames: int = 128,
    min_mask_area_ratio: float | None = None,
    max_masks_per_frame: int | None = None,
    timeout: int | None = None,
) -> dict[str, Any]:
    """Run SAM processing through a persistent SAM3 HTTP service."""
    import requests

    request_timeout = timeout or int(os.getenv("SAM3_SERVER_TIMEOUT", "3600"))
    payload = {
        "video_path": str(Path(video_path).resolve()),
        "video_prompt": video_prompt,
        "model_name": model_name,
        "backend": backend,
        "sam3_model_path": sam3_model_path,
        "device": device,
        "max_frames": max_frames,
    }
    if min_mask_area_ratio is not None:
        payload["min_mask_area_ratio"] = min_mask_area_ratio
    if max_masks_per_frame is not None:
        payload["max_masks_per_frame"] = max_masks_per_frame
    response = requests.post(
        f"{server_url.rstrip('/')}/segment",
        json=payload,
        timeout=request_timeout,
    )
    try:
        response_json = response.json()
    except Exception as exc:
        raise RuntimeError(
            f"SAM3 server response is not JSON. status={response.status_code}, text={response.text[:500]}"
        ) from exc
    if response.status_code != 200:
        raise RuntimeError(f"SAM3 server HTTP {response.status_code}: {response_json}")
    return response_json["result"]
