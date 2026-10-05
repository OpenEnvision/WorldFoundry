# Copyright (C) 2026 Tencent. Adapted for WorldFoundry inference.
# Source revision: 7e57e4d0448e8661eb6ae078f53e19638bd07102
# Academic-use terms: THIRD-PARTY-NOTICES (WorldCrafter)
"""Video output and resumable chunk state."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import torch

from worldfoundry.core.io.filesystem.file_utils import file_sha256 as sha256
from worldfoundry.core.media.codecs.video import _resolve_ffmpeg_executable


def save_chunk_state(
    chunk_index: int,
    state: dict[str, object],
    *,
    state_output_dir: Path,
    run_contract: dict,
    history_selection: list,
) -> None:
    state = dict(state)
    state["run_contract"] = run_contract
    state["history_selection"] = history_selection
    checkpoint_path = state_output_dir / f"chunk_{chunk_index:03d}_complete.pt"
    temporary_path = checkpoint_path.with_suffix(".pt.tmp")
    torch.save(state, temporary_path)
    os.replace(temporary_path, checkpoint_path)
    metadata = {
        "format": state["format"],
        "completed_chunk_index": state["completed_chunk_index"],
        "next_chunk_index": state["next_chunk_index"],
        "checkpoint": checkpoint_path.name,
        "checkpoint_sha256": sha256(checkpoint_path),
        "run_contract": run_contract,
    }
    metadata_path = state_output_dir / f"chunk_{chunk_index:03d}_complete.json"
    temporary_metadata_path = metadata_path.with_suffix(".json.tmp")
    temporary_metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    os.replace(temporary_metadata_path, metadata_path)
    latest_path = state_output_dir / "latest.json"
    temporary_latest_path = latest_path.with_suffix(".json.tmp")
    temporary_latest_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    os.replace(temporary_latest_path, latest_path)
    print(f"[worldcrafter] saved resumable state {checkpoint_path}", flush=True)


def assemble_resumed_video(
    output_path: Path,
    chunk_output_dir: Path,
    final_chunk_index: int,
    fps: int,
    overlap_frames: int = 1,
) -> None:
    chunk_paths = [
        chunk_output_dir / f"chunk_{index:03d}_33f.mp4"
        for index in range(final_chunk_index + 1)
    ]
    missing_chunks = [str(path) for path in chunk_paths if not path.is_file()]
    if missing_chunks:
        raise FileNotFoundError(
            "cannot assemble resumed output; missing chunks: "
            + ", ".join(missing_chunks)
        )
    ffmpeg = _resolve_ffmpeg_executable()
    if ffmpeg is None:
        raise RuntimeError("FFmpeg is required to assemble resumed WorldCrafter video")
    if overlap_frames not in (0, 1):
        raise ValueError("WorldCrafter chunk overlap must be zero (Fast) or one (Base)")
    if len(chunk_paths) == 1:
        shutil.copyfile(chunk_paths[0], output_path)
        return
    if overlap_frames == 0:
        concat_path = output_path.with_suffix(".concat.txt")
        concat_path.write_text(
            "".join(
                "file '" + str(path.resolve()).replace("'", "'\\''") + "'\n"
                for path in chunk_paths
            ),
            encoding="utf-8",
        )
        try:
            subprocess.run(
                [ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(concat_path),
                 "-c", "copy", str(output_path)],
                check=True,
            )
        finally:
            concat_path.unlink(missing_ok=True)
        return

    # Each 33-frame chunk repeats the previous chunk's last frame. A stream
    # copy would keep both copies at every boundary (66 frames for two chunks,
    # instead of the 65 frames produced by uninterrupted Base inference).
    filters = []
    for index in range(len(chunk_paths)):
        trim = "trim=start_frame=1," if index else ""
        filters.append(f"[{index}:v]{trim}setpts=PTS-STARTPTS[v{index}]")
    inputs = "".join(f"[v{index}]" for index in range(len(chunk_paths)))
    filters.append(
        f"{inputs}concat=n={len(chunk_paths)}:v=1:a=0,format=yuv420p[v]"
    )
    subprocess.run(
        [
            ffmpeg,
            "-y",
            *[option for path in chunk_paths for option in ("-i", str(path))],
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[v]",
            "-c",
            "libx264",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-r",
            str(fps),
            str(output_path),
        ],
        check=True,
    )
