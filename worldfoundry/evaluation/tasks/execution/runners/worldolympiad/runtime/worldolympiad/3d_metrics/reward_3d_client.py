"""Lightweight client helpers for the persistent 3D reward service."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def reward_3d_server_url() -> str | None:
    raw = (
        os.getenv("REWARD_3D_SERVER_URL")
        or os.getenv("DA3_SERVER_URL")
        or os.getenv("REWARD_3D_URL")
        or ""
    ).strip()
    return raw.rstrip("/") if raw else None


def score_video_file_via_server(
    *,
    server_url: str,
    video_path: str | os.PathLike[str],
    prompt: str = "",
    max_frames: int | None = 32,
    camera_trajectory: Any = None,
    dynamic_mask_video_path: str | os.PathLike[str] | None = None,
    use_lpips: bool = False,
    timeout: int | None = None,
) -> dict[str, Any]:
    """Score a video through a persistent 3D reward HTTP service."""
    import requests

    request_timeout = timeout or int(os.getenv("REWARD_3D_SERVER_TIMEOUT", "7200"))
    payload = {
        "video_path": str(Path(video_path).resolve()),
        "prompt": prompt,
        "max_frames": max_frames,
        "camera_trajectory": camera_trajectory,
        "dynamic_mask_video_path": (
            None if dynamic_mask_video_path is None else str(Path(dynamic_mask_video_path).resolve())
        ),
        "use_lpips": use_lpips,
    }
    response = requests.post(
        f"{server_url.rstrip('/')}/score_file",
        json=payload,
        timeout=request_timeout,
    )
    try:
        response_json = response.json()
    except Exception as exc:
        raise RuntimeError(
            f"3D reward server response is not JSON. status={response.status_code}, text={response.text[:500]}"
        ) from exc
    if response.status_code != 200:
        raise RuntimeError(f"3D reward server HTTP {response.status_code}: {response_json}")
    return response_json["result"]


def extract_camera_trajectory_via_server(
    *,
    server_url: str,
    video_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str] | None = None,
    max_frames: int | None = 32,
    process_res: int = 504,
    timeout: int | None = None,
) -> tuple[Any, str]:
    """Extract a DA3 camera trajectory through a persistent 3D reward service."""
    import requests

    request_timeout = timeout or int(os.getenv("REWARD_3D_SERVER_TIMEOUT", "7200"))
    payload = {
        "video_path": str(Path(video_path).resolve()),
        "output_path": None if output_path is None else str(Path(output_path).resolve()),
        "max_frames": max_frames,
        "process_res": process_res,
    }
    response = requests.post(
        f"{server_url.rstrip('/')}/extract_trajectory",
        json=payload,
        timeout=request_timeout,
    )
    try:
        response_json = response.json()
    except Exception as exc:
        raise RuntimeError(
            f"3D reward server response is not JSON. status={response.status_code}, text={response.text[:500]}"
        ) from exc
    if response.status_code != 200:
        raise RuntimeError(f"3D reward server HTTP {response.status_code}: {response_json}")
    return response_json["trajectory"], response_json["path"]
