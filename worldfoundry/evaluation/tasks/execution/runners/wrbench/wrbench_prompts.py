"""Natural-25 generation request materialization for WRBench."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from worldfoundry.core.io.serialization import read_jsonl_objects, write_json, write_jsonl
from worldfoundry.evaluation.api import (
    GenerationRequest,
    GenerationResult,
    is_generation_result_successful,
    local_path_for_uri,
)

from .wrbench_paths import natural25_root


CANONICAL_FAMILY_COUNT = 25
CANONICAL_SEMANTIC_VARIANT_COUNT = 100
CANONICAL_REQUEST_COUNT = 500
CAMERA_INTENTS = ("static", "yaw_LR", "pan_LR", "yaw_RL", "pan_RL")
CAMERA_SCRIPTS = {
    "static": "static@81",
    "yaw_LR": "yaw:left:60@40,yaw:right:60@41",
    "pan_LR": "pan:left:0.5@40,pan:right:0.5@41",
    "yaw_RL": "yaw:right:60@40,yaw:left:60@41",
    "pan_RL": "pan:right:0.5@40,pan:left:0.5@41",
}

# Request-side controls describe intent, not certified D1 target artifacts.  A
# real sidecar may only enter WRBench alongside the corresponding generation
# artifact; this video-only bridge must not advertise one from request metadata.
_UNVERIFIED_CAMERA_ARTIFACT_FIELDS = frozenset(
    {
        "camera_sidecar",
        "camera_sidecar_path",
        "camera_trajectory_path",
        "d1_target_sidecar",
        "poses_c2w",
        "target_camera_poses",
        "target_c2w",
        "target_pose_path",
        "trajectory_c2w_path",
    }
)


def load_natural25_variants(*, repo_root: Path | None = None) -> list[dict[str, Any]]:
    path = natural25_root(repo_root) / "variants.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"WRBench Natural-25 variants are empty: {path}")
    return [dict(row) for row in rows if isinstance(row, dict)]


def materialize_wrbench_generation_requests(
    *,
    limit: int | None = None,
    camera_intents: tuple[str, ...] | None = None,
    repo_root: Path | None = None,
) -> tuple[GenerationRequest, ...]:
    """Build the official 100-semantic-variant × 5-camera-control request set."""
    selected_intents = set(camera_intents or CAMERA_INTENTS)
    unknown = selected_intents.difference(CAMERA_INTENTS)
    if unknown:
        raise ValueError(f"unsupported WRBench camera intents: {sorted(unknown)}")

    data_root = natural25_root(repo_root)
    semantic_rows = [row for row in load_natural25_variants(repo_root=repo_root) if row.get("oov_gap") == "none"]
    rows = [(row, camera) for row in semantic_rows for camera in CAMERA_INTENTS if camera in selected_intents]
    if limit is not None:
        rows = rows[: int(limit)]

    requests: list[GenerationRequest] = []
    for row, camera in rows:
        variant_id = str(row["variant_id"])
        family_id = str(row["family_id"])
        output_id = f"{variant_id}__{camera}"
        first_frame = data_root / "first_frames" / f"{family_id}.png"
        if not first_frame.is_file():
            raise FileNotFoundError(f"WRBench first frame is missing: {first_frame}")
        requests.append(
            GenerationRequest(
                sample_id=output_id,
                task_name="wrbench",
                split="natural25",
                inputs={
                    "prompt": str(row["ti2v_prompt"]),
                    "world_state_prompt": str(row["world_state_prompt"]),
                    "expected_state": str(row["expected_state"]),
                    "first_frame": str(first_frame.resolve()),
                    "family_id": family_id,
                    "variant_id": variant_id,
                    "output_id": output_id,
                    "reasoning_tier": row.get("reasoning_tier"),
                    "event_delta": row.get("event_delta"),
                    "divergence_id": row.get("divergence_id"),
                    "official_video_name": f"{output_id}.mp4",
                },
                controls={
                    "camera_intent": camera,
                    "camera_preset": f"preset:{camera}",
                    "camera_script": CAMERA_SCRIPTS[camera],
                    "requires_go_return": camera != "static",
                },
                output_schema={
                    "generated_video": {"kind": "video"},
                    "camera_sidecar": {"kind": "json"},
                    "target_camera_poses": {"kind": "array"},
                },
            )
        )
    return tuple(requests)


def _official_video_name(request: GenerationRequest) -> str:
    raw_name = str(request.inputs.get("official_video_name") or f"{request.sample_id}.mp4").strip()
    if (
        not raw_name
        or raw_name in {".", ".."}
        or "/" in raw_name
        or "\\" in raw_name
        or Path(raw_name).name != raw_name
    ):
        raise ValueError(f"unsafe WRBench official_video_name for {request.sample_id!r}: {raw_name!r}")
    return raw_name


def _runtime_manifest_metadata(request: GenerationRequest) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for values in (request.inputs, request.controls):
        metadata.update(
            {
                str(key): value
                for key, value in values.items()
                if str(key) not in _UNVERIFIED_CAMERA_ARTIFACT_FIELDS
            }
        )

    prompt = request.inputs.get("prompt")
    if prompt not in (None, ""):
        metadata.setdefault("generation_prompt", prompt)
        metadata.setdefault("ti2v_prompt", prompt)

    camera_intent = (
        request.controls.get("camera_intent")
        or request.controls.get("camera")
        or request.inputs.get("camera_type")
        or request.inputs.get("camera")
    )
    if camera_intent not in (None, ""):
        metadata["camera"] = camera_intent
        metadata["camera_type"] = camera_intent
    return metadata


def copy_wrbench_generated_videos(
    *,
    generation_output_dir: Path,
    generated_artifact_dir: Path,
    artifact_manifest_path: Path,
    output_artifact: str = "generated_video",
) -> tuple[int, int]:
    """Join generation contracts into WRBench's official video manifest."""

    requests_path = generation_output_dir / "requests.jsonl"
    results_path = generation_output_dir / "results.jsonl"
    if not requests_path.is_file():
        raise FileNotFoundError(f"WRBench generation requests are missing: {requests_path}")
    if not results_path.is_file():
        raise FileNotFoundError(f"WRBench generation results are missing: {results_path}")

    requests: list[GenerationRequest] = []
    request_ids: set[str] = set()
    for row in read_jsonl_objects(requests_path):
        request = GenerationRequest.from_dict(row)
        if request.sample_id in request_ids:
            raise ValueError(f"duplicate WRBench generation request sample_id: {request.sample_id!r}")
        request_ids.add(request.sample_id)
        requests.append(request)

    results_by_sample: dict[str, GenerationResult] = {}
    for row in read_jsonl_objects(results_path):
        result = GenerationResult.from_dict(row)
        if result.sample_id in results_by_sample:
            raise ValueError(f"duplicate WRBench generation result sample_id: {result.sample_id!r}")
        results_by_sample[result.sample_id] = result

    generated_artifact_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, Any]] = []
    artifact_rows: list[dict[str, Any]] = []
    target_names: dict[str, str] = {}

    for request in requests:
        target_name = _official_video_name(request)
        prior_sample = target_names.get(target_name)
        if prior_sample is not None:
            raise ValueError(
                f"duplicate WRBench official video name {target_name!r} for "
                f"{prior_sample!r} and {request.sample_id!r}"
            )
        target_names[target_name] = request.sample_id
        target_path = generated_artifact_dir / target_name
        result = results_by_sample.pop(request.sample_id, None)
        artifact_row: dict[str, Any] = {
            "sample_id": request.sample_id,
            "request_id": request.request_id,
            "artifact_name": output_artifact,
            "destination": str(target_path.resolve()),
            "status": "missing_result",
            "placeholder": False,
        }
        if result is None:
            artifact_rows.append(artifact_row)
            continue

        artifact_row["request_id"] = result.request_id or request.request_id
        artifact_row["generation_status"] = result.status
        if not is_generation_result_successful(result):
            artifact_row["status"] = "generation_failed"
            artifact_row["error"] = result.error
            artifact_rows.append(artifact_row)
            continue

        artifact_name = output_artifact if output_artifact in result.artifacts else "generated_video"
        artifact = result.artifacts.get(artifact_name)
        artifact_row["artifact_name"] = artifact_name
        if artifact is None:
            artifact_row["status"] = "missing_artifact"
            artifact_rows.append(artifact_row)
            continue

        artifact_row["source_uri"] = artifact.uri
        source_path = local_path_for_uri(artifact.uri, base_dir=generation_output_dir)
        if source_path is None or not source_path.is_file():
            artifact_row["status"] = "missing_source"
            artifact_rows.append(artifact_row)
            continue

        if source_path.resolve() != target_path.resolve():
            shutil.copy2(source_path, target_path)
        artifact_row["status"] = "copied"
        artifact_rows.append(artifact_row)

        video_path = str(target_path.resolve())
        manifest_row = _runtime_manifest_metadata(request)
        manifest_row.update(
            {
                "video_id": request.sample_id,
                "sample_id": request.sample_id,
                "path": video_path,
                "video_path": video_path,
                "model": result.model_id,
                "model_id": result.model_id,
                "task_name": request.task_name,
                "split": request.split,
            }
        )
        request_id = result.request_id or request.request_id
        if request_id:
            manifest_row["request_id"] = request_id
        manifest_rows.append(manifest_row)

    for result in results_by_sample.values():
        artifact_rows.append(
            {
                "sample_id": result.sample_id,
                "request_id": result.request_id,
                "artifact_name": output_artifact,
                "status": "missing_request",
                "placeholder": False,
            }
        )

    write_json(generated_artifact_dir / "videos_manifest.json", manifest_rows)
    write_jsonl(artifact_manifest_path, artifact_rows)
    return len(manifest_rows), 0


__all__ = [
    "CAMERA_INTENTS",
    "CAMERA_SCRIPTS",
    "CANONICAL_FAMILY_COUNT",
    "CANONICAL_REQUEST_COUNT",
    "CANONICAL_SEMANTIC_VARIANT_COUNT",
    "copy_wrbench_generated_videos",
    "load_natural25_variants",
    "materialize_wrbench_generation_requests",
]
