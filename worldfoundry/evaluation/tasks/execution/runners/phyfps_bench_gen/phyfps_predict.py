"""PhyFPS prediction helpers for in-tree and upstream Visual Chronometer execution."""

from __future__ import annotations

import csv
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from worldfoundry.evaluation.tasks.execution.framework.runner_common import VIDEO_SUFFIXES
from worldfoundry.evaluation.tasks.execution.runners.phyfps_bench_gen.phyfps_metrics import VideoPhyFPSRecord
from worldfoundry.evaluation.tasks.execution.runners.phyfps_bench_gen.visual_chronometer_runtime import (
    predict_video_directory,
)



@dataclass(frozen=True)
class PhyFPSPredictConfig:
    backend: str
    stride: int = 4
    clip_length: int = 30
    resolution: int = 216
    device: str = "cuda:0"
    python_executable: str = sys.executable
    chronometer_root: Path | None = None
    config_path: Path | None = None
    ckpt_path: Path | None = None
    timeout_seconds: float | None = None


def _stable_mock_phyfps(video_name: str, *, segment_index: int | None = None) -> float:
    seed = f"{video_name}:{segment_index}"
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    base = 10.0 + (int(digest[:8], 16) % 5000) / 100.0
    if segment_index is not None:
        jitter = ((int(digest[8:16], 16) % 200) - 100) / 100.0
        return round(max(10.0, min(60.0, base + jitter)), 1)
    return round(base, 1)


def list_generated_videos(video_dir: Path) -> list[Path]:
    if not video_dir.is_dir():
        return []
    return sorted(
        path
        for path in video_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
    )


def mock_predict_directory(video_dir: Path, *, stride: int = 4, clip_length: int = 30) -> list[VideoPhyFPSRecord]:
    records: list[VideoPhyFPSRecord] = []
    for video_path in list_generated_videos(video_dir):
        segment_count = max(1, 4 + (hash(video_path.name) % 5))
        segment_values = [
            _stable_mock_phyfps(video_path.name, segment_index=index) for index in range(segment_count)
        ]
        avg_phyfps = round(sum(segment_values) / len(segment_values), 1)
        records.append(
            VideoPhyFPSRecord(
                video=video_path.name,
                avg_phyfps=avg_phyfps,
                segment_phyfps=tuple(segment_values),
            )
        )
    return records


def build_official_predict_command(
    *,
    config: PhyFPSPredictConfig,
    video_dir: Path,
    output_csv: Path,
) -> list[str]:
    if config.chronometer_root is None:
        raise ValueError("Visual Chronometer root is required for official PhyFPS prediction")
    inference_root = config.chronometer_root / "inference"
    predict_script = inference_root / "predict.py"
    if not predict_script.is_file():
        raise FileNotFoundError(f"Visual Chronometer predict.py not found: {predict_script}")
    command = [
        config.python_executable,
        str(predict_script),
        "--video_dir",
        str(video_dir),
        "--stride",
        str(config.stride),
        "--clip_length",
        str(config.clip_length),
        "--resolution",
        str(config.resolution),
        "--device",
        config.device,
        "--output_csv",
        str(output_csv),
    ]
    if config.config_path is not None:
        command.extend(["--config", str(config.config_path)])
    if config.ckpt_path is not None:
        command.extend(["--ckpt", str(config.ckpt_path)])
    return command


def run_official_predict(
    *,
    config: PhyFPSPredictConfig,
    video_dir: Path,
    output_csv: Path,
    stdout_path: Path | None = None,
    stderr_path: Path | None = None,
) -> dict[str, Any]:
    del stdout_path, stderr_path  # in-process runtime does not capture subprocess streams
    if config.chronometer_root is None:
        raise ValueError("Visual Chronometer root is required for official PhyFPS prediction")
    records = predict_video_directory(
        video_dir=video_dir,
        chronometer_root=config.chronometer_root,
        config_path=config.config_path,
        ckpt_path=config.ckpt_path,
        device=config.device,
        clip_length=config.clip_length,
        stride=config.stride,
        resolution=config.resolution,
    )
    write_results_csv(output_csv, records)
    return {
        "command": None,
        "returncode": 0,
        "output_csv": str(output_csv.resolve()),
        "backend": "official",
        "video_count": len(records),
        "runtime": "visual_chronometer_in_process",
    }


def run_phyfps_predict(
    *,
    config: PhyFPSPredictConfig,
    video_dir: Path,
    output_csv: Path,
    stdout_path: Path | None = None,
    stderr_path: Path | None = None,
) -> dict[str, Any]:
    if config.backend == "mock":
        records = mock_predict_directory(video_dir, stride=config.stride, clip_length=config.clip_length)
        write_results_csv(output_csv, records)
        return {
            "command": None,
            "returncode": 0,
            "output_csv": str(output_csv.resolve()),
            "backend": "mock",
            "video_count": len(records),
        }
    runtime = run_official_predict(
        config=config,
        video_dir=video_dir,
        output_csv=output_csv,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )
    runtime["backend"] = "official"
    return runtime


def write_results_csv(output_csv: Path, records: list[VideoPhyFPSRecord]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["video", "segment", "start_frame", "mid_frame", "end_frame", "predicted_phyfps"])
        for record in records:
            for index, value in enumerate(record.segment_phyfps):
                start = index * 4
                writer.writerow([record.video, index, start, start + 15, start + 29, value])
            writer.writerow([record.video, "AVG", "", "", "", record.avg_phyfps])


def parse_results_csv(results_path: Path) -> list[VideoPhyFPSRecord]:
    if not results_path.is_file():
        raise FileNotFoundError(f"PhyFPS results CSV not found: {results_path}")
    by_video: dict[str, dict[str, Any]] = {}
    with results_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            video = str(row.get("video") or "").strip()
            if not video:
                continue
            segment = str(row.get("segment") or "").strip()
            value = row.get("predicted_phyfps")
            try:
                phyfps = float(value)
            except (TypeError, ValueError):
                continue
            bucket = by_video.setdefault(video, {"segments": [], "avg": None})
            if segment.upper() == "AVG":
                bucket["avg"] = phyfps
            else:
                bucket["segments"].append(phyfps)
    records: list[VideoPhyFPSRecord] = []
    for video, payload in sorted(by_video.items()):
        segments = tuple(float(value) for value in payload["segments"])
        avg_phyfps = payload["avg"]
        if avg_phyfps is None and segments:
            avg_phyfps = round(sum(segments) / len(segments), 1)
        if avg_phyfps is None:
            continue
        records.append(
            VideoPhyFPSRecord(
                video=video,
                avg_phyfps=float(avg_phyfps),
                segment_phyfps=segments,
            )
        )
    return records


def load_meta_fps_map(
    *,
    explicit_manifest: Path | None = None,
    default_meta_fps: float | None = None,
    video_names: list[str] | None = None,
) -> dict[str, float]:
    import json

    mapping: dict[str, float] = {}
    if explicit_manifest is not None and explicit_manifest.is_file():
        payload = json.loads(explicit_manifest.read_text(encoding="utf-8"))
        if isinstance(payload, Mapping):
            for key, value in payload.items():
                try:
                    mapping[str(key)] = float(value)
                except (TypeError, ValueError):
                    continue
    if default_meta_fps is not None and video_names:
        for video_name in video_names:
            mapping.setdefault(video_name, float(default_meta_fps))
            mapping.setdefault(Path(video_name).stem, float(default_meta_fps))
    return mapping
