# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Data-only request contract shared by the VideoX-Fun camera entrypoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TypedDict, cast


class CameraInferenceRequest(TypedDict):
    GPU_memory_mode: str
    config_path: str
    model_name: str
    sample_size: list[int]
    video_length: int
    fps: int
    control_video: None
    control_camera_txt: str
    start_image: str
    ref_image: None
    prompt: str
    seed: int
    num_inference_steps: int
    enable_teacache: bool
    save_path: str


def parse_camera_request() -> CameraInferenceRequest:
    """Read the JSON request written by the model-owned runtime bridge."""
    parser = argparse.ArgumentParser(description="Run a VideoX-Fun camera inference request.")
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    return cast(CameraInferenceRequest, json.loads(args.request.read_text(encoding="utf-8")))
