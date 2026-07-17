"""FETV prompt requests and the generated-video to official-frame bridge."""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from worldfoundry.core.io.serialization import read_jsonl_objects, write_jsonl
from worldfoundry.evaluation.api import (
    GenerationRequest,
    GenerationResult,
    is_generation_result_successful,
    local_path_for_uri,
)
from worldfoundry.evaluation.tasks.execution.framework.benchmark_assets import bundled_benchmark_asset

BENCHMARK_ID = "fetv"
CANONICAL_PROMPT_COUNT = 619
FETV_FRAME_COUNT = 16
FETV_GENERATION_MANIFEST_NAME = "fetv_generation_manifest.jsonl"
FETV_BOUNDED_PROMPT_NAME = "fetv_prompt_subset.jsonl"
FETV_PROMPT_ASSET = bundled_benchmark_asset(BENCHMARK_ID, "fetv_data.json")


def resolve_fetv_prompt_file(explicit: Path | None = None) -> Path:
    """Resolve the exact JSONL prompt file consumed by official FETV-EVAL."""

    env_value = os.environ.get("WORLDFOUNDRY_FETV_PROMPT_FILE")
    path = explicit or (Path(env_value) if env_value else FETV_PROMPT_ASSET)
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"FETV prompt JSONL file not found: {path}")
    return path


def load_fetv_prompt_records(*, prompt_file: Path | None = None) -> list[dict[str, Any]]:
    """Load official prompts while preserving their physical zero-based line index."""

    source = resolve_fetv_prompt_file(prompt_file)
    lines = source.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"FETV prompt JSONL file is empty: {source}")

    records: list[dict[str, Any]] = []
    for sent_index, line in enumerate(lines):
        # The official VideoDataset calls json.loads on every physical line, so
        # silently discarding blank lines would change every following sent id.
        if not line.strip():
            raise ValueError(f"blank FETV prompt record at zero-based line {sent_index}: {source}")
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid FETV prompt JSON at zero-based line {sent_index}: {source}") from exc
        if not isinstance(payload, dict):
            raise TypeError(f"FETV prompt line {sent_index} must contain a JSON object: {source}")
        prompt = payload.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"FETV prompt line {sent_index} has no non-empty prompt: {source}")
        if "video_id" not in payload:
            raise ValueError(f"FETV prompt line {sent_index} has no video_id: {source}")
        records.append(
            {
                "sent_index": sent_index,
                "prompt_record": payload,
                "official_video_name": f"{sent_index}.mp4",
                "official_frame_dir_name": f"sent{sent_index}_frames",
            }
        )
    if source == FETV_PROMPT_ASSET.resolve() and len(records) != CANONICAL_PROMPT_COUNT:
        raise ValueError(
            f"bundled FETV prompt suite has {len(records)} rows; expected {CANONICAL_PROMPT_COUNT}: {source}"
        )
    return records


def materialize_fetv_generation_requests(
    *,
    limit: int | None = None,
    prompt_file: Path | None = None,
) -> tuple[GenerationRequest, ...]:
    """Build one model-independent generation request per official FETV line."""

    records = load_fetv_prompt_records(prompt_file=prompt_file)
    if limit is not None:
        if int(limit) <= 0:
            raise ValueError("FETV generation limit must be a positive integer")
        records = records[: int(limit)]

    requests: list[GenerationRequest] = []
    for record in records:
        sent_index = int(record["sent_index"])
        prompt_record = dict(record["prompt_record"])
        requests.append(
            GenerationRequest(
                sample_id=f"fetv-sent-{sent_index:04d}",
                task_name=BENCHMARK_ID,
                split="standard",
                inputs={
                    "prompt": prompt_record["prompt"],
                    "prompt_id": sent_index,
                    "video_id": prompt_record.get("video_id"),
                    "sent_index": sent_index,
                    "official_video_name": record["official_video_name"],
                    "official_frame_dir_name": record["official_frame_dir_name"],
                    # Persist the exact source row so a bounded official run can
                    # use the original metadata without reconstructing text.
                    "official_prompt_record": prompt_record,
                },
                output_schema={"generated_video": {"kind": "video"}},
            )
        )
    return tuple(requests)


def _request_metadata(request: GenerationRequest) -> tuple[int, str, str, dict[str, Any]]:
    raw_index = request.inputs.get("sent_index")
    if isinstance(raw_index, bool) or not isinstance(raw_index, int) or raw_index < 0:
        raise ValueError(f"FETV request {request.sample_id!r} has invalid sent_index: {raw_index!r}")
    sent_index = int(raw_index)
    video_name = request.inputs.get("official_video_name")
    frame_dir_name = request.inputs.get("official_frame_dir_name")
    expected_video_name = f"{sent_index}.mp4"
    expected_frame_dir_name = f"sent{sent_index}_frames"
    if video_name != expected_video_name:
        raise ValueError(
            f"FETV request {request.sample_id!r} must persist official_video_name "
            f"{expected_video_name!r}, got {video_name!r}"
        )
    if frame_dir_name != expected_frame_dir_name:
        raise ValueError(
            f"FETV request {request.sample_id!r} must persist official_frame_dir_name "
            f"{expected_frame_dir_name!r}, got {frame_dir_name!r}"
        )
    prompt_record = request.inputs.get("official_prompt_record")
    if not isinstance(prompt_record, Mapping):
        raise ValueError(f"FETV request {request.sample_id!r} is missing official_prompt_record")
    prompt_record = dict(prompt_record)
    if prompt_record.get("prompt") != request.inputs.get("prompt"):
        raise ValueError(f"FETV request {request.sample_id!r} prompt differs from official_prompt_record")
    if prompt_record.get("video_id") != request.inputs.get("video_id"):
        raise ValueError(f"FETV request {request.sample_id!r} video_id differs from official_prompt_record")
    return sent_index, expected_video_name, expected_frame_dir_name, prompt_record


def decode_fetv_uniform_frames(
    *,
    video_path: Path,
    target_dir: Path,
    frame_count: int = FETV_FRAME_COUNT,
) -> tuple[Path, ...]:
    """Decode a video with FETV-EVAL's official uniform frame sampling rule."""

    if frame_count <= 0:
        raise ValueError("FETV frame_count must be positive")
    source = video_path.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"FETV generated video does not exist: {source}")
    if target_dir.exists():
        raise FileExistsError(f"refusing to overwrite an existing FETV frame directory: {target_dir}")

    try:
        import numpy as np
        from moviepy.editor import VideoFileClip
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("FETV video staging requires numpy, Pillow, moviepy==1.0.3, and its ffmpeg backend") from exc

    clip = None
    try:
        clip = VideoFileClip(str(source), audio=False)
        duration = float(clip.duration)
        fps = float(clip.fps)
        if not math.isfinite(duration) or duration <= 0 or not math.isfinite(fps) or fps <= 0:
            raise ValueError(f"invalid duration/fps reported for FETV video {source}: {duration=}, {fps=}")
        # This mirrors official utils/video2frames.py: increase the decode fps
        # for short videos until at least the requested number can be sampled.
        while math.floor(duration * fps) < frame_count:
            fps += 1.0
            if fps > 10_000:
                raise ValueError(f"unable to derive {frame_count} frames from FETV video: {source}")
        decoded_frames = [Image.fromarray(frame) for frame in clip.iter_frames(fps=fps, dtype="uint8")]
    except Exception as exc:
        raise ValueError(f"unable to decode FETV generated video {source}: {exc}") from exc
    finally:
        if clip is not None:
            clip.close()

    if len(decoded_frames) < frame_count:
        raise ValueError(
            f"FETV generated video decoded to {len(decoded_frames)} frames; expected at least {frame_count}: {source}"
        )
    interval = max(len(decoded_frames) / frame_count, 1)
    source_indices = np.arange(0, len(decoded_frames), interval, dtype=int).tolist()
    if len(source_indices) != frame_count or len(set(source_indices)) != frame_count:
        raise ValueError(
            f"official FETV uniform sampling did not yield exactly {frame_count} unique frames for {source}"
        )

    target_dir.mkdir(parents=True)
    output_paths: list[Path] = []
    try:
        for frame_index, source_index in enumerate(source_indices):
            output_path = target_dir / f"frame{frame_index}.jpg"
            # Match official FETV-EVAL utils/video2frames.py, including its
            # Pillow keyword, so reproduced JPEGs follow the same encoder path.
            decoded_frames[source_index].save(output_path, q=95)
            with Image.open(output_path) as image:
                image.verify()
            output_paths.append(output_path)
    except Exception:
        shutil.rmtree(target_dir, ignore_errors=True)
        raise
    return tuple(output_paths)


def copy_fetv_generated_videos(
    *,
    generation_output_dir: Path,
    generated_artifact_dir: Path,
    artifact_manifest_path: Path,
    output_artifact: str = "generated_video",
) -> tuple[int, int]:
    """Strictly join model results to FETV sent ids and decode official frames."""

    requests_path = generation_output_dir / "requests.jsonl"
    results_path = generation_output_dir / "results.jsonl"
    if not requests_path.is_file():
        raise FileNotFoundError(f"FETV generation request manifest is missing: {requests_path}")
    if not results_path.is_file():
        raise FileNotFoundError(f"FETV generation result manifest is missing: {results_path}")

    requests_by_id: dict[str, GenerationRequest] = {}
    metadata_by_id: dict[str, tuple[int, str, str, dict[str, Any]]] = {}
    sent_indices: dict[int, str] = {}
    for row in read_jsonl_objects(requests_path):
        request = GenerationRequest.from_dict(row)
        if request.sample_id in requests_by_id:
            raise ValueError(f"duplicate FETV request sample_id: {request.sample_id!r}")
        metadata = _request_metadata(request)
        sent_index = metadata[0]
        if sent_index in sent_indices:
            raise ValueError(
                f"duplicate FETV sent_index {sent_index} for {sent_indices[sent_index]!r} and {request.sample_id!r}"
            )
        requests_by_id[request.sample_id] = request
        metadata_by_id[request.sample_id] = metadata
        sent_indices[sent_index] = request.sample_id
    if not requests_by_id:
        raise ValueError(f"FETV generation request manifest is empty: {requests_path}")
    if sorted(sent_indices) != list(range(len(sent_indices))):
        raise ValueError("bounded FETV generation requests must cover a contiguous zero-based sent_index prefix")

    results_by_id: dict[str, GenerationResult] = {}
    for row in read_jsonl_objects(results_path):
        result = GenerationResult.from_dict(row)
        if result.sample_id in results_by_id:
            raise ValueError(f"duplicate FETV result sample_id: {result.sample_id!r}")
        results_by_id[result.sample_id] = result
    missing_results = sorted(requests_by_id.keys() - results_by_id.keys())
    unexpected_results = sorted(results_by_id.keys() - requests_by_id.keys())
    if missing_results or unexpected_results:
        raise ValueError(
            "FETV generation manifest coverage mismatch: "
            f"missing results={missing_results[:8]}, unexpected results={unexpected_results[:8]}"
        )

    destination = generated_artifact_dir.expanduser().resolve()
    if destination.exists() and not destination.is_dir():
        raise FileExistsError(f"FETV artifact destination is not a directory: {destination}")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"refusing to mix FETV frames with an existing artifact directory: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".fetv-stage-", dir=destination.parent))
    artifact_rows: list[dict[str, Any]] = []
    prompt_rows: list[dict[str, Any]] = []
    committed = False
    try:
        for sent_index in sorted(sent_indices):
            sample_id = sent_indices[sent_index]
            request = requests_by_id[sample_id]
            result = results_by_id[sample_id]
            _, official_video_name, frame_dir_name, prompt_record = metadata_by_id[sample_id]
            if not is_generation_result_successful(result):
                raise ValueError(
                    f"FETV generation failed for {sample_id!r}; complete prompt coverage is required: {result.error}"
                )
            artifact = result.artifacts.get(output_artifact)
            if artifact is None:
                raise ValueError(f"FETV result {sample_id!r} has no requested {output_artifact!r} artifact")
            source_path = local_path_for_uri(artifact.uri, base_dir=generation_output_dir)
            if source_path is None or not source_path.is_file():
                raise FileNotFoundError(
                    f"FETV result {sample_id!r} does not reference a readable local video: {artifact.uri!r}"
                )
            frame_paths = decode_fetv_uniform_frames(
                video_path=source_path,
                target_dir=stage / frame_dir_name,
            )
            prompt_rows.append(prompt_record)
            artifact_rows.append(
                {
                    "sample_id": sample_id,
                    "request_id": result.request_id or request.request_id,
                    "sent_index": sent_index,
                    "artifact_name": output_artifact,
                    "source_uri": artifact.uri,
                    "official_video_name": official_video_name,
                    "frame_dir": str((destination / frame_dir_name).resolve()),
                    "frame_dir_name": frame_dir_name,
                    "frame_count": len(frame_paths),
                    "prompt_record": prompt_record,
                    "status": "decoded",
                    "placeholder": False,
                }
            )

        write_jsonl(stage / FETV_GENERATION_MANIFEST_NAME, artifact_rows)
        write_jsonl(stage / FETV_BOUNDED_PROMPT_NAME, prompt_rows)
        if destination.exists():
            destination.rmdir()
        stage.replace(destination)
        committed = True
        write_jsonl(artifact_manifest_path, artifact_rows)
    except Exception:
        shutil.rmtree(destination if committed else stage, ignore_errors=True)
        raise
    return len(artifact_rows), 0


__all__ = [
    "BENCHMARK_ID",
    "CANONICAL_PROMPT_COUNT",
    "FETV_BOUNDED_PROMPT_NAME",
    "FETV_FRAME_COUNT",
    "FETV_GENERATION_MANIFEST_NAME",
    "copy_fetv_generated_videos",
    "decode_fetv_uniform_frames",
    "load_fetv_prompt_records",
    "materialize_fetv_generation_requests",
    "resolve_fetv_prompt_file",
]
