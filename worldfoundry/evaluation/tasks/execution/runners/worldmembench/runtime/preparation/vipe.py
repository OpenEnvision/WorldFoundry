# SPDX-FileCopyrightText: Copyright (c) 2026 ZJU-REAL/WorldMemBench contributors
# SPDX-License-Identifier: MIT

"""Prepare full-frame ViPE poses for video.mp4 cases using the default pose-only pipeline."""

import json
import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory, TemporaryFile


def vipe_command(python, source, video, output):
    return [
        *python,
        str(source / "run.py"),
        "pipeline=default",
        "streams=raw_mp4_stream",
        "streams.base_path=" + json.dumps(str(video), ensure_ascii=False),
        "streams.frame_start=0",
        "streams.frame_end=-1",
        "streams.frame_skip=1",
        "streams.cached=true",
        "pipeline.post.depth_align_model=null",
        "pipeline.output.path=" + json.dumps(str(output), ensure_ascii=False),
        "pipeline.output.save_artifacts=true",
        "pipeline.output.save_slam_map=false",
        "pipeline.output.save_viz=false",
    ]


def convert_output(native: Path, output: Path, total_frames: int):
    import numpy as np

    from worldfoundry.evaluation.tasks.execution.runners.worldmembench.runtime.errors import EvaluationError
    from worldfoundry.evaluation.tasks.execution.runners.worldmembench.runtime.metrics.camera.schemas import (
        PoseTrajectory,
    )

    with np.load(native, allow_pickle=False) as data:
        poses = np.asarray(data["data"], dtype=np.float64)
        indices = np.asarray(data["inds"])
    if poses.shape != (total_frames, 4, 4) or not np.array_equal(indices, np.arange(total_frames)):
        raise EvaluationError("ViPE must register every frame in exact 0..F-1 order; sparse output is invalid")
    PoseTrajectory(poses, source="vipe")
    np.savez(output, c2w=poses, frame_indices=indices.astype(np.int64))


def ensure_artifacts(sample, context):
    from worldfoundry.evaluation.tasks.execution.runners.worldmembench.runtime.errors import EvaluationError
    from worldfoundry.evaluation.tasks.execution.runners.worldmembench.runtime.video import VideoReader

    output = Path(sample["artifacts"]["vipe"])
    if output.is_file():
        return
    video = Path(sample["video"]).absolute()
    if output != video.parent / "vipe/results.npz":
        raise EvaluationError("missing manifest ViPE artifact; automatic generation uses case/vipe/results.npz")
    if output.parent.exists() and any(output.parent.iterdir()):
        raise EvaluationError(f"incomplete ViPE artifacts in {output.parent}; move them aside before regenerating")
    if not context.device.startswith("cuda"):
        raise EvaluationError("ViPE generation requires CUDA; CPU scoring can reuse existing poses")
    config = context.config.get("vipe") or {}
    if not config.get("source"):
        raise EvaluationError("Configure vipe.source for the official ViPE default pipeline, or supply ViPE poses")
    source = Path(config["source"]).expanduser().resolve()
    if not (source / "run.py").is_file():
        raise EvaluationError("Configured ViPE model runtime must contain run.py")
    weights = (
        Path(config.get("weights") or context.cache_dir / "metrics/camera_translation/vipe").expanduser().resolve()
    )
    python = [str(Path(config.get("python") or sys.executable).expanduser())]
    count = VideoReader().probe(str(video)).total_frames
    env = dict(os.environ)
    index = int(context.device.split(":")[1]) if ":" in context.device else 0
    visible = env.get("CUDA_VISIBLE_DEVICES")
    env["CUDA_VISIBLE_DEVICES"] = visible.split(",")[index].strip() if visible else str(index)
    env.update(
        TORCH_HOME=str(weights / "torch"),
        HF_HOME=str(weights / "huggingface"),
        HF_HUB_CACHE=str(weights / "huggingface/hub"),
        TRANSFORMERS_CACHE=str(weights / "huggingface/hub"),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        PYTHONDONTWRITEBYTECODE="1",
    )
    with TemporaryDirectory(prefix=".vipe-", dir=video.parent) as directory:
        work = Path(directory)
        native = work / "native"
        with TemporaryFile(dir=work) as log:
            try:
                subprocess.run(
                    vipe_command(python, source, video, native),
                    cwd=source,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            except subprocess.CalledProcessError as error:
                end = log.seek(0, os.SEEK_END)
                log.seek(max(0, end - 8192))
                tail = "\n".join(log.read().decode("utf-8", errors="replace").splitlines()[-20:])
                raise EvaluationError(f"ViPE failed (exit {error.returncode}):\n{tail}") from error
        converted = work / "results.npz"
        convert_output(native / "pose" / f"{video.stem}.npz", converted, count)
        output.parent.mkdir(exist_ok=True)
        output.hardlink_to(converted)
