"""In-tree VideoVerse VLM judge execution.

Ports the official VideoVerse evaluation loop (event ordering + verification checks)
into WorldFoundry so ``official-run`` can produce ``eval_res.json`` without calling
upstream scripts directly.
"""

from __future__ import annotations

import copy
import json
import os
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from worldfoundry.evals.tasks.execution.framework.io import write_json, write_jsonl

EVAL_NUMBER_MAPPING = ("A", "B", "C", "D", "E", "F", "G", "H")
EVAL_PROMPT_FOR_OVERALL_EVENTS = (
    "These are several events that may happened in the video."
    "Please sort them in the order in which they occurred. "
    "If an event does not occur, the letter corresponding to the event is ignored."
    "Only output the corresponding letters in order, use commas to separate, and wrap your response with <output></output>. Such as <output>B,D</output> or <output>A,B,C</output>"
    "{}"
)
EVAL_PROMPT_FOR_STATISTIC_QUESTION = "In this video.\n{}\nAnswer with Yes or No."

JudgeFn = Callable[[str, Path | str], str]


class VideoVerseJudgeError(RuntimeError):
    """Raised when the configured VideoVerse judge backend cannot run."""


@dataclass(frozen=True)
class VideoVerseJudgeConfig:
    backend: str
    model_name: str
    api_key: str | None = None
    video_base_url: str | None = None
    local_vlm_path: str | None = None
    max_attempts_event: int = 4
    max_attempts_check: int = 3
    enable_sub_questions: bool = False
    checkpoint_every: int = 3


def filter_and_join_letters(input_string: str) -> str:
    return "".join(char for char in input_string if "A" <= char <= "Z")


def extract_first_yes_or_no(text: str) -> str:
    match = re.search(r"yes|no", text, re.IGNORECASE)
    return match.group(0).lower() if match else "wrong"


def _first_env(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value.strip()
    return None


def judge_config_from_env() -> VideoVerseJudgeConfig:
    backend = (_first_env("WORLDFOUNDRY_VIDEOVERSE_JUDGE_BACKEND") or "gemini").lower()
    enable_sub = (_first_env("WORLDFOUNDRY_VIDEOVERSE_ENABLE_SUB_QUESTIONS") or "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    return VideoVerseJudgeConfig(
        backend=backend,
        model_name=_first_env("WORLDFOUNDRY_VIDEOVERSE_JUDGE_MODEL") or "gemini-2.5-pro",
        api_key=_first_env(
            "WORLDFOUNDRY_VIDEOVERSE_GEMINI_API_KEY",
            "GEMINI_API_KEY",
            "GOOGLE_API_KEY",
        ),
        video_base_url=_first_env("WORLDFOUNDRY_VIDEOVERSE_VIDEO_BASE_URL"),
        local_vlm_path=_first_env("WORLDFOUNDRY_VIDEOVERSE_LOCAL_VLM_PATH"),
        enable_sub_questions=enable_sub,
    )


def missing_judge_requirements(config: VideoVerseJudgeConfig | None = None) -> list[str]:
    cfg = config or judge_config_from_env()
    missing: list[str] = []
    if cfg.backend in {"gemini", "google", "url"}:
        if not cfg.api_key:
            missing.append("GEMINI_API_KEY or WORLDFOUNDRY_VIDEOVERSE_GEMINI_API_KEY")
        if cfg.backend == "url" and not cfg.video_base_url:
            missing.append("WORLDFOUNDRY_VIDEOVERSE_VIDEO_BASE_URL")
    elif cfg.backend in {"local_vlm", "qwen", "open_vlm"}:
        if not cfg.local_vlm_path:
            missing.append("WORLDFOUNDRY_VIDEOVERSE_LOCAL_VLM_PATH")
    elif cfg.backend == "mock":
        return []
    else:
        missing.append(f"unsupported WORLDFOUNDRY_VIDEOVERSE_JUDGE_BACKEND={cfg.backend!r}")
    return missing


def resolve_video_path(prompt_id: str, generated_video_dir: Path, *, video_base_url: str | None) -> Path | str:
    if video_base_url:
        return video_base_url.rstrip("/") + f"/{prompt_id}.mp4"
    for suffix in (".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"):
        candidate = generated_video_dir / f"{prompt_id}{suffix}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"missing generated video for prompt {prompt_id!r} under {generated_video_dir}")


def _gemini_generate(*, prompt: str, video: Path | str, model_name: str, api_key: str) -> str:
    try:
        genai = __import__("google", fromlist=["genai"]).genai
    except ImportError as exc:
        raise VideoVerseJudgeError(
            "Gemini judge requires the google-genai package. Install google-genai or set "
            "WORLDFOUNDRY_VIDEOVERSE_JUDGE_BACKEND=mock for tests."
        ) from exc

    client = genai.Client(api_key=api_key)
    if isinstance(video, Path):
        uploaded = client.files.upload(file=str(video))
        while getattr(uploaded, "state", None) is not None and getattr(uploaded.state, "name", "") == "PROCESSING":
            time.sleep(1.0)
            uploaded = client.files.get(name=uploaded.name)
        contents: list[Any] = [uploaded, prompt]
    else:
        contents = [{"file_data": {"file_uri": str(video)}}, prompt]
    response = client.models.generate_content(model=model_name, contents=contents)
    text = getattr(response, "text", None)
    if text:
        return str(text)
    candidates = getattr(response, "candidates", None) or []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) if content is not None else None
        if not parts:
            continue
        for part in parts:
            part_text = getattr(part, "text", None)
            if part_text:
                return str(part_text)
    return ""


def build_judge_fn(config: VideoVerseJudgeConfig, *, override: JudgeFn | None = None) -> JudgeFn:
    if override is not None:
        return override

    if config.backend == "mock":

        def _mock_judge(prompt: str, video: Path | str) -> str:
            _ = video
            if "sort them in the order" in prompt.lower():
                return "<output>A,B</output>"
            if "yes or no" in prompt.lower():
                return "yes"
            return "yes"

        return _mock_judge

    if config.backend in {"gemini", "google", "url"}:
        if not config.api_key:
            raise VideoVerseJudgeError("Gemini judge requires GEMINI_API_KEY")

        def _gemini_judge(prompt: str, video: Path | str) -> str:
            target = video
            if config.backend == "url" and isinstance(video, Path):
                if not config.video_base_url:
                    raise VideoVerseJudgeError("URL judge requires WORLDFOUNDRY_VIDEOVERSE_VIDEO_BASE_URL")
                target = config.video_base_url.rstrip("/") + f"/{video.stem}.mp4"
            return _gemini_generate(
                prompt=prompt, video=target, model_name=config.model_name, api_key=cast(str, config.api_key)
            )

        return _gemini_judge

    if config.backend in {"local_vlm", "qwen", "open_vlm"}:
        if not config.local_vlm_path:
            raise VideoVerseJudgeError("Local VLM judge requires WORLDFOUNDRY_VIDEOVERSE_LOCAL_VLM_PATH")

        def _local_vlm_judge(prompt: str, video: Path | str) -> str:
            if not isinstance(video, Path):
                raise VideoVerseJudgeError("Local VLM judge requires on-disk generated videos")
            from worldfoundry.evals.tasks.execution.runners.videoverse.videoverse_local_vlm import (
                local_vlm_generate,
            )

            return local_vlm_generate(
                prompt=prompt,
                video_path=video,
                model_path=Path(cast(str, config.local_vlm_path)),
            )

        return _local_vlm_judge

    raise VideoVerseJudgeError(f"unsupported judge backend: {config.backend}")


def _evaluate_event_order(
    *,
    record: Mapping[str, Any],
    judge_fn: JudgeFn,
    video: Path | str,
    max_attempts: int,
    responses: list[dict[str, Any]],
    prompt_id: str,
) -> None:
    event_info = record.get("t2v_eval_event_info")
    if not isinstance(event_info, Mapping):
        return
    plan = event_info.get("verification_plan")
    if not isinstance(plan, list) or not plan:
        return
    overall_check = ""
    for index, item in enumerate(plan):
        if not isinstance(item, Mapping):
            continue
        overall_check += "\n" + EVAL_NUMBER_MAPPING[index] + ": " + str(item.get("event_description") or "")
    prompt = EVAL_PROMPT_FOR_OVERALL_EVENTS.format(overall_check)
    for attempt in range(1, max_attempts + 1):
        output_response = judge_fn(prompt, video)
        responses.append(
            {
                "prompt_id": prompt_id,
                "kind": "event_order",
                "attempt": attempt,
                "prompt": prompt,
                "response": output_response,
            }
        )
        if "<output>" not in output_response or "</output>" not in output_response:
            continue
        match = re.search(r"<output>(.*?)</output>", output_response, re.DOTALL)
        if match is None:
            continue
        raw = match.group(1)
        event_info["overall_event_res"] = raw
        event_info["overall_event_processed_res"] = filter_and_join_letters(raw)
        return


def _evaluate_verification_checks(
    *,
    record: Mapping[str, Any],
    judge_fn: JudgeFn,
    video: Path | str,
    max_attempts: int,
    responses: list[dict[str, Any]],
    prompt_id: str,
    enable_sub_questions: bool,
) -> None:
    checks = record.get("verification_checks")
    if not isinstance(checks, list):
        return
    for index, check in enumerate(checks):
        if not isinstance(check, Mapping):
            continue
        mutable = dict(check)
        if mutable.get("check") is False:
            continue
        if enable_sub_questions and isinstance(mutable.get("decomposed_sub_questions"), list):
            mutable["sub_question_results"] = []
            all_yes = True
            for sub_q in mutable["decomposed_sub_questions"]:
                sub_prompt = EVAL_PROMPT_FOR_STATISTIC_QUESTION.format(sub_q)
                parsed = "wrong"
                for attempt in range(1, max_attempts + 1):
                    raw = judge_fn(sub_prompt, video)
                    responses.append(
                        {
                            "prompt_id": prompt_id,
                            "kind": "sub_question",
                            "attempt": attempt,
                            "prompt": sub_prompt,
                            "response": raw,
                        }
                    )
                    parsed = extract_first_yes_or_no(raw)
                    if parsed != "wrong" or attempt >= max_attempts:
                        break
                mutable["sub_question_results"].append({"question": sub_q, "res": parsed})
                if parsed != "yes":
                    all_yes = False
            mutable["all_sub_questions_yes"] = all_yes
            mutable["res"] = "yes" if all_yes else "no"
            checks[index] = mutable
            continue
        prompt = EVAL_PROMPT_FOR_STATISTIC_QUESTION.format(mutable.get("question") or "")
        parsed = "wrong"
        for attempt in range(1, max_attempts + 1):
            raw = judge_fn(prompt, video)
            responses.append(
                {
                    "prompt_id": prompt_id,
                    "kind": "verification_check",
                    "attempt": attempt,
                    "prompt": prompt,
                    "response": raw,
                }
            )
            parsed = extract_first_yes_or_no(raw)
            if parsed != "wrong" or attempt >= max_attempts:
                break
        mutable["res"] = parsed
        checks[index] = mutable


def run_videoverse_judge(
    *,
    prompt_manifest: Mapping[str, Mapping[str, Any]],
    generated_video_dir: Path,
    output_path: Path,
    decomposed_manifest: Mapping[str, Mapping[str, Any]] | None = None,
    limit: int | None = None,
    config: VideoVerseJudgeConfig | None = None,
    judge_fn: JudgeFn | None = None,
    judge_responses_path: Path | None = None,
    resume_path: Path | None = None,
) -> dict[str, Any]:
    """Execute the VideoVerse judge loop and write ``eval_res.json``."""
    cfg = config or judge_config_from_env()
    missing = missing_judge_requirements(cfg)
    if missing and judge_fn is None and cfg.backend != "mock":
        raise VideoVerseJudgeError("missing judge requirements: " + ", ".join(missing))
    complete = build_judge_fn(cfg, override=judge_fn)

    source_manifest = dict(prompt_manifest)
    if decomposed_manifest:
        for prompt_id, record in decomposed_manifest.items():
            if prompt_id in source_manifest and isinstance(record, Mapping):
                merged = copy.deepcopy(source_manifest[prompt_id])
                merged_checks = merged.get("verification_checks")
                decomposed_checks = record.get("verification_checks")
                if isinstance(merged_checks, list) and isinstance(decomposed_checks, list):
                    by_question = {
                        str(item.get("question")): item
                        for item in decomposed_checks
                        if isinstance(item, Mapping) and item.get("question") is not None
                    }
                    for check in merged_checks:
                        if not isinstance(check, Mapping):
                            continue
                        decomposed = by_question.get(str(check.get("question")))
                        if isinstance(decomposed, Mapping) and "decomposed_sub_questions" in decomposed:
                            check["decomposed_sub_questions"] = list(decomposed["decomposed_sub_questions"])
                source_manifest[prompt_id] = merged

    results: dict[str, Any] = {}
    if resume_path is not None and resume_path.is_file():
        payload = json.loads(resume_path.read_text(encoding="utf-8"))
        if isinstance(payload, Mapping):
            results = {str(key): dict(value) for key, value in payload.items() if isinstance(value, Mapping)}

    responses: list[dict[str, Any]] = []
    prompt_ids = sorted(source_manifest)
    if limit is not None:
        prompt_ids = prompt_ids[: int(limit)]

    processed = 0
    for prompt_id in prompt_ids:
        if prompt_id in results:
            continue
        record = copy.deepcopy(source_manifest[prompt_id])
        video = resolve_video_path(
            prompt_id,
            generated_video_dir,
            video_base_url=cfg.video_base_url if cfg.backend == "url" else None,
        )
        _evaluate_event_order(
            record=record,
            judge_fn=complete,
            video=video,
            max_attempts=cfg.max_attempts_event,
            responses=responses,
            prompt_id=prompt_id,
        )
        _evaluate_verification_checks(
            record=record,
            judge_fn=complete,
            video=video,
            max_attempts=cfg.max_attempts_check,
            responses=responses,
            prompt_id=prompt_id,
            enable_sub_questions=cfg.enable_sub_questions,
        )
        results[prompt_id] = record
        processed += 1
        if processed % max(cfg.checkpoint_every, 1) == 0:
            write_json(output_path, results, atomic=False)
            if judge_responses_path is not None:
                write_jsonl(judge_responses_path, responses, atomic=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, results, atomic=False)
    if judge_responses_path is not None:
        write_jsonl(judge_responses_path, responses, atomic=False)
    return {
        "processed_count": len(results),
        "new_count": processed,
        "output_path": str(output_path.resolve()),
        "judge_backend": cfg.backend,
        "judge_model": cfg.model_name,
    }
