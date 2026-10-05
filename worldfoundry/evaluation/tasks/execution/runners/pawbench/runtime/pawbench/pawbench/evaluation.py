"""PAWBench evaluation implementation used by the repository script."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from pawbench.metrics import EXPECTED_ROLLOUTS, compute_metrics, validate_scene_policy
from pawbench.paweval.evidence.extraction import extract_video_frames
from pawbench.paweval.evidence.package import EvidencePackage
from pawbench.paweval.evidence.sampling import FrameSamplingSpec
from pawbench.paweval.evidence.source_image import SourceImageIdentity
from pawbench.paweval.judge.client import (
    OpenAICompatibleJudgeClient,
    ProviderConfig,
    validate_provider_readiness,
)
from pawbench.paweval.judge.requests import RowIdentity
from pawbench.paweval.judgment import JudgmentConfig, JudgmentPreflightError, JudgmentSample, judge
from pawbench.paweval.rubrics.loader import load_rubric


def evaluate(
    benchmark_path: str | Path,
    videos: Sequence[Mapping[str, Any]],
    *,
    model_or_lane: str,
    vlm: Mapping[str, Any],
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Judge a complete local PAWBench video grid and compute official metrics.

    ``benchmark_path`` is an already-downloaded package containing
    ``manifest.json`` and its ``scenes.jsonl`` table. Each video row provides
    ``sample_id``, ``scene_id``, ``repeat_index``, and ``video_path``. PAWEval
    turns each supplied video plus its source image into an evidence
    package, judges it with the scene's two rubrics, and returns metric-ready
    rows without provider payloads or local media paths. When ``output_dir`` is
    provided, each completed row is checkpointed and a matching interrupted
    run resumes without judging completed rollout slots again.
    """

    if not isinstance(model_or_lane, str) or not model_or_lane:
        raise ValueError("model_or_lane must be a nonempty string")
    root = Path(benchmark_path)
    policy, source_paths = _read_benchmark(root, model_or_lane)
    provider = _provider_config(vlm)
    expected = {
        (scene["scene_id"], repeat): scene
        for track in policy["tracks"].values()
        for scene in track["scenes"]
        for repeat in scene["expected_repeat_indices"]
    }
    supplied, blockers = _index_videos(videos, expected)
    output = _prepare_output(output_dir, root, model_or_lane, provider)
    completed = _load_checkpoints(output, expected, model_or_lane)
    rows_by_slot: dict[tuple[str, int], dict[str, Any]] = {}
    provider_failure = _provider_failure_code(provider)

    for slot, scene in expected.items():
        scene_id, repeat = slot
        video = supplied.get(slot)
        fallback_id = f"{model_or_lane}::{scene_id}::r{repeat:03d}"
        fingerprint = _video_fingerprint(video)
        checkpoint = completed.get(slot)
        if checkpoint and checkpoint["video_fingerprint"] == fingerprint:
            row = checkpoint["row"]
            if row.get("observation") in {"outcome", "null_observation"}:
                rows_by_slot[slot] = row
                continue

        if video is None:
            blockers.append(f"missing_video:{scene_id}:{repeat}")
            row = _failure(
                fallback_id,
                scene_id,
                scene["track"],
                model_or_lane,
                repeat,
                "missing_video",
            )
        elif "error" in video:
            row = _failure(
                str(video.get("sample_id") or fallback_id),
                scene_id,
                scene["track"],
                model_or_lane,
                repeat,
                str(video["error"]),
            )
        elif provider_failure:
            blockers.append(f"provider:{provider_failure}")
            row = _failure(
                str(video["sample_id"]),
                scene_id,
                scene["track"],
                model_or_lane,
                repeat,
                provider_failure,
            )
        else:
            row = _evaluate_video(
                video=video,
                scene=scene,
                source_image_path=source_paths[scene_id],
                model_or_lane=model_or_lane,
                provider=provider,
            )
        rows_by_slot[slot] = row
        _append_checkpoint(output, slot, fingerprint, row)

    rows = [rows_by_slot[slot] for slot in expected]

    metrics = compute_metrics(rows, policy)
    blockers = sorted(set(blockers + list(metrics["blockers"])))
    result = {
        "status": "blocked" if blockers else "ok",
        "blockers": blockers,
        "rows": rows,
        "metrics": metrics,
    }
    artifacts = _write_outputs(output, rows, metrics)
    if artifacts:
        result["artifacts"] = artifacts
    return result


def _evaluate_video(
    *,
    video: Mapping[str, Any],
    scene: Mapping[str, Any],
    source_image_path: Path,
    model_or_lane: str,
    provider: ProviderConfig,
) -> dict[str, Any]:
    scene_id = str(scene["scene_id"])
    repeat = int(video["repeat_index"])
    try:
        with TemporaryDirectory(prefix="pawbench-paweval-") as temporary_dir:
            sample = _build_sample(
                video=video,
                scene=scene,
                source_image_path=source_image_path,
                model_or_lane=model_or_lane,
                frame_root=Path(temporary_dir),
            )
            return _judge_samples([sample], provider)[0]
    except JudgmentPreflightError as exc:
        code = str(exc)
    except (OSError, RuntimeError, ValueError) as exc:
        code = f"evidence_build_failed:{type(exc).__name__}"
    return _failure(
        str(video["sample_id"]),
        scene_id,
        str(scene["track"]),
        model_or_lane,
        repeat,
        code,
    )


def _build_sample(
    *,
    video: Mapping[str, Any],
    scene: Mapping[str, Any],
    source_image_path: Path,
    model_or_lane: str,
    frame_root: Path,
) -> JudgmentSample:
    sample_id = str(video["sample_id"])
    scene_id = str(scene["scene_id"])
    video_path = Path(str(video["video_path"]))
    frames = extract_video_frames(
        sample_id=sample_id,
        video_path=video_path,
        output_dir=frame_root,
        sampling=FrameSamplingSpec(),
    )
    source = _source_identity(source_image_path)
    package = EvidencePackage(
        sample_id=sample_id,
        scene_id=scene_id,
        frames=frames,
        source_image=source,
    )
    return JudgmentSample(
        package=package,
        track=scene["track"],
        row_identity=RowIdentity.from_row(video, model_or_lane=model_or_lane),
    )


def _source_identity(path: Path) -> SourceImageIdentity:
    if path.is_file():
        return SourceImageIdentity(attached=True, uri=path.resolve().as_uri())
    return SourceImageIdentity(
        attached=False, absent_reason="source_image_missing_in_local_package"
    )


def _judge_samples(samples: list[JudgmentSample], provider: ProviderConfig) -> list[dict[str, Any]]:
    return judge(
        samples=samples,
        adapter=OpenAICompatibleJudgeClient(provider),
        config=JudgmentConfig(),
    ).normalized_rows()


def _provider_failure_code(provider: ProviderConfig) -> str | None:
    readiness = validate_provider_readiness(provider)
    if readiness["status"] == "pass":
        return None
    blocked = readiness.get("blocked") or []
    if blocked and isinstance(blocked[0], Mapping):
        return str(blocked[0].get("reason") or "provider_preflight_failed")
    return "provider_preflight_failed"


def _prepare_output(
    output_dir: str | Path | None,
    benchmark_root: Path,
    model_or_lane: str,
    provider: ProviderConfig,
) -> dict[str, Path] | None:
    if output_dir is None:
        return None
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "root": root,
        "run": root / "run.json",
        "checkpoint": root / "checkpoint.jsonl",
        "rows": root / "rows.jsonl",
        "metrics": root / "metrics.json",
    }
    run = _run_record(benchmark_root, model_or_lane, provider)
    if paths["run"].is_file():
        try:
            existing = json.loads(paths["run"].read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid PAWBench run metadata: {paths['run']}") from exc
        if existing.get("fingerprint") != run["fingerprint"]:
            raise ValueError(
                "output_dir belongs to a different benchmark, model, or VLM configuration"
            )
    else:
        _atomic_write_json(paths["run"], run)
    paths["checkpoint"].touch(exist_ok=True)
    return paths


def _run_record(
    benchmark_root: Path, model_or_lane: str, provider: ProviderConfig
) -> dict[str, Any]:
    benchmark_digest = _benchmark_digest(benchmark_root)
    evaluator_digest = _evaluator_digest()
    provider_record = {
        "provider": provider.provider,
        "model": provider.model,
        "base_url": provider.credential_status()["base_url"],
        "api_key_env": provider.api_key_env,
        "timeout": provider.timeout,
        "max_tokens": provider.max_tokens,
    }
    identity = {
        "benchmark_digest": benchmark_digest,
        "evaluator_digest": evaluator_digest,
        "model_or_lane": model_or_lane,
        "provider": {
            **provider_record,
            "base_url_digest": hashlib.sha256(provider.base_url.encode("utf-8")).hexdigest(),
        },
    }
    fingerprint = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": "pawbench.run/v1",
        "fingerprint": fingerprint,
        "benchmark": {
            "path": str(benchmark_root.resolve()),
            "digest": benchmark_digest,
        },
        "evaluator_digest": evaluator_digest,
        "model_or_lane": model_or_lane,
        "provider": provider_record,
    }


def _benchmark_digest(root: Path) -> str:
    manifest_path = root / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    scene_table_path = root / str(manifest["scene_table"])
    scene_table_bytes = scene_table_path.read_bytes()
    scenes = [json.loads(line) for line in scene_table_bytes.splitlines() if line]
    digest = hashlib.sha256()
    for name, contents in (
        ("manifest.json", manifest_bytes),
        (str(manifest["scene_table"]), scene_table_bytes),
    ):
        digest.update(name.encode("utf-8") + b"\0" + contents + b"\0")
    for scene in scenes:
        relative = str(scene.get("source_image_path") or "")
        path = root / relative
        contents = path.read_bytes() if path.is_file() else b"<missing>"
        digest.update(relative.encode("utf-8") + b"\0" + contents + b"\0")
    return digest.hexdigest()


def _evaluator_digest() -> str:
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    files = sorted(
        path for path in root.rglob("*") if path.is_file() and path.suffix in {".py", ".yaml"}
    )
    for path in files:
        relative = str(path.relative_to(root))
        digest.update(relative.encode("utf-8") + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def _video_fingerprint(video: Mapping[str, Any] | None) -> str:
    if video is None:
        value: dict[str, Any] = {"status": "missing"}
    elif "error" in video:
        value = {
            "status": "invalid",
            "sample_id": str(video.get("sample_id") or ""),
            "error": str(video["error"]),
        }
    else:
        path = Path(str(video["video_path"])).expanduser().resolve()
        try:
            stat = path.stat()
            file_identity: dict[str, Any] = {
                "status": "present",
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        except OSError:
            file_identity = {"status": "unreadable"}
        value = {
            **file_identity,
            "sample_id": str(video["sample_id"]),
            "path": str(path),
        }
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _load_checkpoints(
    output: dict[str, Path] | None,
    expected: Mapping[tuple[str, int], Mapping[str, Any]],
    model_or_lane: str,
) -> dict[tuple[str, int], dict[str, Any]]:
    if output is None or not output["checkpoint"].is_file():
        return {}
    lines = output["checkpoint"].read_text(encoding="utf-8").splitlines()
    checkpoints: dict[tuple[str, int], dict[str, Any]] = {}
    for index, line in enumerate(lines):
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            if index == len(lines) - 1:
                break
            raise ValueError(f"invalid PAWBench checkpoint line {index + 1}") from exc
        if not isinstance(item, Mapping) or item.get("schema_version") != "pawbench.checkpoint/v1":
            raise ValueError(f"invalid PAWBench checkpoint line {index + 1}")
        scene_id = item.get("scene_id")
        repeat = item.get("repeat_index")
        row = item.get("row")
        slot = (scene_id, repeat)
        if (
            not isinstance(scene_id, str)
            or not isinstance(repeat, int)
            or slot not in expected
            or not isinstance(row, dict)
            or row.get("scene_id") != scene_id
            or row.get("repeat_index") != repeat
            or row.get("model_or_lane") != model_or_lane
            or not isinstance(item.get("video_fingerprint"), str)
        ):
            raise ValueError(f"invalid PAWBench checkpoint line {index + 1}")
        checkpoints[slot] = {
            "video_fingerprint": item["video_fingerprint"],
            "row": row,
        }
    return checkpoints


def _append_checkpoint(
    output: dict[str, Path] | None,
    slot: tuple[str, int],
    video_fingerprint: str,
    row: Mapping[str, Any],
) -> None:
    if output is None or row.get("observation") not in {"outcome", "null_observation"}:
        return
    item = {
        "schema_version": "pawbench.checkpoint/v1",
        "scene_id": slot[0],
        "repeat_index": slot[1],
        "video_fingerprint": video_fingerprint,
        "row": row,
    }
    with output["checkpoint"].open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(item, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_outputs(
    output: dict[str, Path] | None,
    rows: Sequence[Mapping[str, Any]],
    metrics: Mapping[str, Any],
) -> dict[str, str] | None:
    if output is None:
        return None
    rows_text = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows
    )
    _atomic_write_text(output["rows"], rows_text)
    _atomic_write_json(output["metrics"], metrics)
    return {
        "run": str(output["run"]),
        "checkpoint": str(output["checkpoint"]),
        "rows": str(output["rows"]),
        "metrics": str(output["metrics"]),
    }


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_write_text(
        path,
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )


def _atomic_write_text(path: Path, value: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _read_benchmark(root: Path, model_or_lane: str) -> tuple[dict[str, Any], dict[str, Path]]:
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        scene_table = root / str(manifest["scene_table"])
        scenes = [
            json.loads(line)
            for line in scene_table.read_text(encoding="utf-8").splitlines()
            if line
        ]
    except (KeyError, OSError, json.JSONDecodeError) as exc:
        raise ValueError("invalid local PAWBench package") from exc
    if manifest.get("schema_version") != "pawbench.benchmark_inputs/v1" or len(scenes) != 50:
        raise ValueError("benchmark package must contain the released 50-scene contract")

    tracks: dict[str, dict[str, list[dict[str, Any]]]] = {
        "calibration": {"scenes": []},
        "coverage": {"scenes": []},
    }
    source_paths: dict[str, Path] = {}
    scene_ids: set[str] = set()
    for source in scenes:
        if not isinstance(source, Mapping):
            raise ValueError("benchmark scene rows must be JSON objects")
        track = source.get("split")
        scene_id = source.get("scene_id")
        labels = source.get("outcome_labels")
        if track not in tracks or not isinstance(scene_id, str) or not isinstance(labels, list):
            raise ValueError("benchmark scene has an invalid policy")
        if scene_id in scene_ids:
            raise ValueError(f"benchmark package contains duplicate scene_id: {scene_id}")
        scene_ids.add(scene_id)
        scene = {
            "scene_id": scene_id,
            "track": track,
            "group": source.get("group") or track,
            "expected_repeat_indices": list(range(EXPECTED_ROLLOUTS)),
        }
        if track == "calibration":
            scene["reference_distribution"] = source.get("reference_distribution")
        else:
            scene["support_labels"] = labels
        tracks[track]["scenes"].append(scene)
        source_paths[scene_id] = root / str(source.get("source_image_path") or "")
    policy = {"model_or_lanes": [model_or_lane], "tracks": tracks}
    validate_scene_policy(policy)
    for scene_id in scene_ids:
        for axis in ("outcome", "trustworthiness"):
            try:
                load_rubric(axis, scene_id)
            except (FileNotFoundError, ValueError) as exc:
                raise ValueError(f"benchmark scene has no valid {axis} rubric: {scene_id}") from exc
    return policy, source_paths


def _provider_config(values: Mapping[str, Any]) -> ProviderConfig:
    if not isinstance(values, Mapping):
        raise ValueError("vlm must be an object")
    try:
        return ProviderConfig(
            provider=str(values.get("provider") or "openai_compatible"),
            model=str(values["model"]),
            base_url=str(values["base_url"]),
            api_key_env=str(values["api_key_env"]),
            timeout=int(values.get("timeout", 120)),
            max_tokens=int(values.get("max_tokens", 2048)),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("vlm must contain base_url, model, and api_key_env") from exc


def _index_videos(
    videos: Sequence[Mapping[str, Any]], expected: Mapping[tuple[str, int], Mapping[str, Any]]
) -> tuple[dict[tuple[str, int], dict[str, Any]], list[str]]:
    supplied: dict[tuple[str, int], dict[str, Any]] = {}
    blockers: list[str] = []
    sample_ids: dict[str, set[str]] = {}
    for video in videos:
        if not isinstance(video, Mapping):
            blockers.append("malformed_video_item")
            continue
        scene_id, repeat, sample_id, video_path = (
            video.get("scene_id"),
            video.get("repeat_index"),
            video.get("sample_id"),
            video.get("video_path"),
        )
        if (
            not isinstance(scene_id, str)
            or not isinstance(repeat, int)
            or (scene_id, repeat) not in expected
        ):
            blockers.append("unexpected_video_item")
            continue
        slot = (scene_id, repeat)
        if slot in supplied:
            supplied[slot] = {"sample_id": sample_id, "error": "duplicate_video_item"}
            blockers.append(f"duplicate_video_item:{scene_id}:{repeat}")
            continue
        if (
            not isinstance(sample_id, str)
            or not sample_id
            or not isinstance(video_path, (str, Path))
        ):
            supplied[slot] = {"sample_id": sample_id, "error": "malformed_video_item"}
            blockers.append(f"malformed_video_item:{scene_id}:{repeat}")
            continue
        seen = sample_ids.setdefault(scene_id, set())
        if sample_id in seen:
            supplied[slot] = {"sample_id": sample_id, "error": "duplicate_sample_id"}
            blockers.append(f"duplicate_sample_id:{scene_id}:{sample_id}")
            continue
        seen.add(sample_id)
        supplied[slot] = {
            "sample_id": sample_id,
            "scene_id": scene_id,
            "repeat_index": repeat,
            "video_path": str(video_path),
        }
    return supplied, blockers


def _failure(
    sample_id: str, scene_id: str, track: str, model_or_lane: str, repeat_index: int, code: str
) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "scene_id": scene_id,
        "track": track,
        "model_or_lane": model_or_lane,
        "repeat_index": repeat_index,
        "observation": "infrastructure_failure",
        "failure_code": code,
    }
