"""In-tree PAI-Bench-G evaluation using shared VBench runtimes and Qwen judging."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .io import VIDEO_SUFFIXES, load_json, load_records, prediction_index, write_json
from .judges import build_judge
from .metrics import aggregate_generation_vqa, normalize_binary_answer
from .prompts import prompt_config, render_prompt

CLASSIC_DIMENSIONS = (
    "aesthetic_quality",
    "background_consistency",
    "imaging_quality",
    "motion_smoothness",
    "overall_consistency",
    "subject_consistency",
)
I2V_DIMENSIONS = ("i2v_background", "i2v_subject")
GENERATION_METRICS = (*CLASSIC_DIMENSIONS, *I2V_DIMENSIONS, "vqa_accuracy")
VQA_CATEGORIES = ("common_sense", "industry", "physics", "human", "robot", "misc", "av")


@dataclass(frozen=True)
class GenerationRequest:
    generated_video_dir: Path
    output_dir: Path
    dataset_root: Path | None = None
    prompt_file: Path | None = None
    reference_image_dir: Path | None = None
    vqa_questions_dir: Path | None = None
    prediction_manifest: Path | None = None
    metrics: tuple[str, ...] = GENERATION_METRICS
    judge_backend: str | None = None
    judge_model: str | Path | None = None
    judge_base_url: str | None = None
    max_frames: int = 16
    limit: int | None = None


def _prompt_rows(request: GenerationRequest) -> list[dict[str, Any]]:
    prompt_file = request.prompt_file
    if prompt_file is None and request.dataset_root is not None:
        candidate = request.dataset_root / "cosmos_predict2_bench_full_info.json"
        prompt_file = candidate if candidate.is_file() else None
    if prompt_file is None:
        return []
    payload = load_json(prompt_file)
    if isinstance(payload, dict):
        payload = payload.get("records", payload.get("data", payload))
    if isinstance(payload, dict):
        return [{"video_id": key, "prompt": value} for key, value in payload.items()]
    if not isinstance(payload, list):
        raise ValueError(f"unsupported PAI-Bench-G prompt metadata: {prompt_file}")
    return [dict(row) for row in payload if isinstance(row, dict)]


def _prompt_text(row: dict[str, Any]) -> str:
    for key in ("prompt", "prompt_en", "caption", "caption_text", "text"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return str(row.get("video_id") or "")


def _video_files(request: GenerationRequest) -> list[Path]:
    files = sorted(
        path
        for path in request.generated_video_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
    )
    return files[: request.limit] if request.limit is not None else files


def _delegate_video_dir(request: GenerationRequest) -> Path:
    direct = [
        path
        for path in request.generated_video_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
    ]
    files = _video_files(request)
    if len(direct) == len(files):
        return request.generated_video_dir
    target = request.output_dir / "runtime" / "delegate_videos"
    target.mkdir(parents=True, exist_ok=True)
    for video in files:
        alias = target / video.name
        if alias.exists():
            if alias.resolve() != video.resolve():
                raise ValueError(f"duplicate generated video basename: {video.name}")
            continue
        try:
            alias.symlink_to(video.resolve())
        except OSError:
            import shutil

            shutil.copy2(video, alias)
    return target


def _vbench_prompt_map(request: GenerationRequest, rows: list[dict[str, Any]]) -> Path:
    by_id = {str(row.get("video_id")): _prompt_text(row) for row in rows if row.get("video_id") is not None}
    mapping = {}
    for video in sorted(_delegate_video_dir(request).iterdir()):
        video_id = video.stem.split("__", 1)[0]
        mapping[video.name] = by_id.get(video_id, video_id)
    return write_json(request.output_dir / "runtime" / "vbench_prompts.json", mapping)


def _extract_summary(scorecard: dict[str, Any]) -> dict[str, float]:
    metrics = scorecard.get("metrics", {})
    result: dict[str, float] = {}
    leaderboard = metrics.get("leaderboard", {})
    if isinstance(leaderboard, dict):
        for key, value in leaderboard.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                result[str(key)] = float(value)
    summary = metrics.get("summary", {})
    if isinstance(summary, dict):
        for key, value in summary.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                result[str(key)] = float(value)
            elif isinstance(value, dict):
                scalar = value.get("score", value.get("raw_score", value.get("value")))
                if isinstance(scalar, (int, float)):
                    result[str(key)] = float(scalar)
    per_metric = metrics.get("per_metric", {})
    metric_rows = (
        per_metric.values() if isinstance(per_metric, dict) else per_metric if isinstance(per_metric, list) else ()
    )
    for row in metric_rows:
        if isinstance(row, dict):
            value = row.get("score", row.get("raw_score"))
            if isinstance(value, (int, float)):
                result[str(row.get("metric_id"))] = float(value)
    return result


def _run_classic_vbench(request: GenerationRequest, dimensions: list[str], prompt_map: Path) -> dict[str, Any]:
    from worldfoundry.evaluation.tasks.execution.runners.vbench.vbench_official_impl import (
        VBenchRunRequest,
        run_vbench,
    )

    return run_vbench(
        VBenchRunRequest(
            output_dir=request.output_dir / "delegates" / "vbench",
            videos_path=_delegate_video_dir(request),
            dimensions=tuple(dimensions),
            benchmark_id="physical-ai-bench",
            mode="custom_input",
            prompt_file=prompt_map,
        )
    )


def _stage_reference_aliases(request: GenerationRequest) -> Path:
    source = request.reference_image_dir
    if source is None and request.dataset_root is not None:
        source = request.dataset_root / "condition_image"
    if source is None or not source.is_dir():
        raise FileNotFoundError("PAI-Bench-G I2V metrics require --reference-image-dir or dataset/condition_image")
    target = request.output_dir / "runtime" / "condition_image_aliases"
    target.mkdir(parents=True, exist_ok=True)
    images = {
        path.stem: path for path in source.rglob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    }
    for video in sorted(_delegate_video_dir(request).iterdir()):
        video_id = video.stem.split("__", 1)[0]
        image = images.get(video_id)
        if image is None:
            continue
        alias = target / f"{video.stem}{image.suffix.lower()}"
        if not alias.exists():
            try:
                alias.symlink_to(image.resolve())
            except OSError:
                import shutil

                shutil.copy2(image, alias)
    return target


def _run_i2v_vbench(request: GenerationRequest, dimensions: list[str]) -> dict[str, Any]:
    from worldfoundry.evaluation.tasks.execution.runners.vbench_2_0.vbench_shared_official_impl import (
        build_parser,
        run_series,
    )

    aliases = _stage_reference_aliases(request)
    argv = [
        "--benchmark-id",
        "physical-ai-bench",
        "--variant",
        "i2v",
        "--videos-path",
        str(_delegate_video_dir(request)),
        "--output-dir",
        str(request.output_dir / "delegates" / "vbench_i2v"),
        "--mode",
        "custom_input",
        "--custom-image-folder",
        str(aliases),
    ]
    for dimension in dimensions:
        argv.extend(["--dimension", dimension])
    args = build_parser(variant_choices=("i2v",)).parse_args(argv)
    return run_series(args)


def _question_rows(request: GenerationRequest) -> list[dict[str, Any]]:
    root = request.vqa_questions_dir
    if root is None and request.dataset_root is not None:
        root = request.dataset_root / "vqa"
    if root is None or not root.exists():
        raise FileNotFoundError("PAI-Bench-G VQA requires --vqa-questions-dir or dataset/vqa")
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*.json")):
        payload = load_json(path)
        values = payload if isinstance(payload, list) else payload.get("questions", [payload])
        for position, value in enumerate(values):
            if not isinstance(value, dict):
                continue
            row = dict(value)
            row.setdefault("video_id", path.stem.split("__", 1)[0])
            row.setdefault("question_id", row.get("uid", f"{path.stem}:{position}"))
            video_id = str(row["video_id"])
            category = next(
                (candidate for candidate in VQA_CATEGORIES if video_id.startswith(f"{candidate}_")),
                row.get("task", path.parent.name),
            )
            row.setdefault("category", category)
            rows.append(row)
    return rows


def _vqa_prediction(question: dict[str, Any], video: Path, predictions: dict[str, dict[str, Any]]) -> str | None:
    keys = (question.get("uid"), question.get("question_id"), f"{video.stem}:{question.get('uid')}", video.stem)
    record = next((predictions.get(str(key)) for key in keys if key is not None and predictions.get(str(key))), None)
    if record is None:
        return None
    for key in ("prediction", "pred", "response", "output", "answer"):
        if record.get(key) is not None:
            return str(record[key])
    return None


def _evaluate_vqa(request: GenerationRequest) -> dict[str, Any]:
    questions = _question_rows(request)
    videos_by_id: dict[str, list[Path]] = {}
    for video in _video_files(request):
        videos_by_id.setdefault(video.stem.split("__", 1)[0], []).append(video)
    predictions = prediction_index(load_records(request.prediction_manifest)) if request.prediction_manifest else {}
    judge = None
    if request.judge_backend:
        if request.judge_model is None:
            raise ValueError("--judge-model is required when --judge-backend is used")
        judge = build_judge(
            request.judge_backend,
            model=request.judge_model,
            base_url=request.judge_base_url,
            max_frames=request.max_frames,
        )
    if not predictions and judge is None:
        raise ValueError("PAI-Bench-G VQA needs --prediction-manifest or a configured judge backend")
    detailed: list[dict[str, Any]] = []
    for question in questions:
        video_id = str(question.get("video_id"))
        for video in videos_by_id.get(video_id, []):
            prediction = _vqa_prediction(question, video, predictions)
            if prediction is None and judge is not None:
                config = prompt_config("generation_binary_vqa")
                prediction = judge.generate(
                    prompt=render_prompt(
                        "generation_binary_vqa",
                        question=str(question.get("question") or "").strip(),
                    ),
                    video_path=video,
                    system_prompt=str(config.get("system_prompt") or "") or None,
                )
            mapping = question.get("index2ans") if isinstance(question.get("index2ans"), dict) else None
            gold = normalize_binary_answer(question.get("answer"), mapping)
            parsed = normalize_binary_answer(prediction, mapping)
            detailed.append(
                {
                    "video_id": video_id,
                    "seed_video": video.stem,
                    "question_id": question.get("question_id"),
                    "category": question.get("category"),
                    "prediction": prediction,
                    "parsed_prediction": parsed,
                    "answer": gold,
                    "correct": parsed == gold if parsed is not None and gold is not None else False,
                }
            )
    return {"summary": aggregate_generation_vqa(detailed), "samples": detailed}


def evaluate_generation(request: GenerationRequest) -> dict[str, Any]:
    request.output_dir.mkdir(parents=True, exist_ok=True)
    requested = [metric for metric in request.metrics if metric in GENERATION_METRICS]
    unknown = sorted(set(request.metrics) - set(requested))
    if unknown:
        raise ValueError(f"unknown PAI-Bench-G metrics: {', '.join(unknown)}")
    rows = _prompt_rows(request)
    prompt_map = _vbench_prompt_map(request, rows)
    summary: dict[str, Any] = {}
    artifacts: dict[str, Any] = {}
    classic = [metric for metric in requested if metric in CLASSIC_DIMENSIONS]
    if classic:
        scorecard = _run_classic_vbench(request, classic, prompt_map)
        summary.update({key: value for key, value in _extract_summary(scorecard).items() if key in classic})
        artifacts["vbench"] = scorecard.get("artifacts", {})
    i2v = [metric for metric in requested if metric in I2V_DIMENSIONS]
    if i2v:
        scorecard = _run_i2v_vbench(request, i2v)
        summary.update({key: value for key, value in _extract_summary(scorecard).items() if key in i2v})
        artifacts["vbench_i2v"] = scorecard.get("artifacts", {})
    samples: list[dict[str, Any]] = []
    if "vqa_accuracy" in requested:
        vqa = _evaluate_vqa(request)
        summary["vqa_accuracy"] = vqa["summary"]["vqa_accuracy"]
        summary.update(vqa["summary"]["category_scores"])
        samples.extend(vqa["samples"])
    return {
        "track": "generation",
        "summary": summary,
        "samples": samples,
        "artifacts": artifacts,
        "delegates": ["worldfoundry.vbench", "worldfoundry.vbench_plus_plus.i2v"],
    }
