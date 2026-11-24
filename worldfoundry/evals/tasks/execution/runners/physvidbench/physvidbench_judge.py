"""In-tree PhysVidBench Gemini QA judge ported from upstream multi_ask.py."""

from __future__ import annotations

import csv
import hashlib
import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from worldfoundry.evals.tasks.execution.runners.physvidbench.physvidbench_captions import (
    CAPTION_SUFFIXES,
    captions_for_prompt_id,
    load_caption_matrix,
)

YES_NO_REGEX = re.compile(r"\bQ\s*(\d+)\s*[:\-]\s*(yes|no)\b", re.I)
JudgeFn = Callable[[str], str]


@dataclass(frozen=True)
class PhysVidBenchJudgeConfig:
    backend: str
    model_name: str = "models/gemini-2.0-flash"
    api_key: str | None = None
    temperature: float = 0.0
    max_retries: int = 5


def judge_config_from_env() -> PhysVidBenchJudgeConfig:
    backend = (
        os.environ.get("WORLDFOUNDRY_PHYSVIDBENCH_JUDGE_BACKEND")
        or os.environ.get("WORLDFOUNDRY_PHYSVIDBENCH_QA_BACKEND")
        or "official"
    ).lower()
    if backend == "official":
        backend = "gemini"
    return PhysVidBenchJudgeConfig(
        backend=backend,
        model_name=os.environ.get("WORLDFOUNDRY_PHYSVIDBENCH_JUDGE_MODEL") or "models/gemini-2.0-flash",
        api_key=(
            os.environ.get("WORLDFOUNDRY_PHYSVIDBENCH_GEMINI_API_KEY")
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
        ),
    )


def _stable_mock_answer(prompt_id: int, question_index: int) -> str:
    seed = f"{prompt_id}:{question_index}"
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    return "Yes" if int(digest[:8], 16) % 5 != 0 else "No"


def build_judge_fn(config: PhysVidBenchJudgeConfig) -> JudgeFn:
    if config.backend == "mock":

        def _mock_judge(prompt: str) -> str:
            prompt_id_match = re.search(r"PromptID\s*(\d+)", prompt)
            prompt_id = int(prompt_id_match.group(1)) if prompt_id_match else 0
            lines = []
            question_index = 0
            for line in prompt.splitlines():
                if line.startswith("Q") and ":" in line:
                    answer = _stable_mock_answer(prompt_id, question_index)
                    lines.append(f"Q{question_index + 1}: {answer}")
                    question_index += 1
            if lines:
                return "\n".join(lines)
            return "Q1: Yes"

        return _mock_judge

    if config.backend in {"gemini", "google"}:
        if not config.api_key:
            raise RuntimeError("PhysVidBench Gemini judge requires GEMINI_API_KEY")

        def _gemini_judge(prompt: str) -> str:
            try:
                genai = __import__("google", fromlist=["genai"]).genai
                types = __import__("google.genai", fromlist=["types"]).types
                ServerError = __import__("google.genai.errors", fromlist=["ServerError"]).ServerError
            except ImportError as exc:
                raise RuntimeError(
                    "PhysVidBench Gemini judge requires google-genai. "
                    "Install google-genai or set WORLDFOUNDRY_PHYSVIDBENCH_JUDGE_BACKEND=mock."
                ) from exc
            client = genai.Client(api_key=config.api_key)
            for attempt in range(config.max_retries):
                try:
                    response = client.models.generate_content(
                        model=config.model_name,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            temperature=config.temperature,
                            max_output_tokens=1_000_000,
                        ),
                    )
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
                except ServerError as err:
                    if getattr(err, "code", None) == 503 and attempt + 1 < config.max_retries:
                        time.sleep(2 ** (attempt + 2))
                        continue
                    raise
            raise RuntimeError("Gemini returned 503 too many times.")

        return _gemini_judge

    raise RuntimeError(f"Unsupported PhysVidBench judge backend: {config.backend}")


def build_gemini_prompt(*, captions: list[str], questions: list[str]) -> str:
    prompt = (
        "You are given 8 captions describing different aspects of the same video.\n"
        "Answer “Yes” ONLY if at least one caption supports it, otherwise answer “No”.\n\n"
    )
    for index, caption in enumerate(captions):
        prompt += f"Caption {index + 1}: {caption}\n\n"
    for index, question in enumerate(questions):
        prompt += f"Q{index + 1}: {question}\n"
    prompt += "\nRespond only as:\nQ1: Yes\nQ2: No\n..."
    return prompt


def parse_model_answers(text: str, question_count: int) -> dict[int, str]:
    answers: dict[int, str] = {}
    for line in text.splitlines():
        match = YES_NO_REGEX.search(line)
        if match:
            answers[int(match.group(1)) - 1] = match.group(2).capitalize()
    for index in range(question_count):
        answers.setdefault(index, "N/A")
    return answers


def run_physvidbench_qa(
    *,
    question_csv: Path,
    caption_base: Path,
    output_csv: Path,
    config: PhysVidBenchJudgeConfig | None = None,
    suffixes: tuple[str, ...] = CAPTION_SUFFIXES,
    limit: int | None = None,
) -> dict[str, Any]:
    config = config or judge_config_from_env()
    judge_fn = build_judge_fn(config)
    caption_matrix = load_caption_matrix(caption_base, suffixes=suffixes)
    frame = pd.read_csv(question_csv)
    frame = frame[frame["Upsampled"] == True]  # noqa: E712 - Elementwise pandas comparison.
    if frame.empty:
        raise ValueError(f"No upsampled PhysVidBench entries found in {question_csv}")

    groups = frame.groupby("PromptID")
    pending_ids = sorted(int(prompt_id) for prompt_id in groups.groups)
    if limit is not None:
        pending_ids = pending_ids[: int(limit)]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    prompts_processed = 0
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["PromptID", "Question", "Types", "Model_Answer", "Match"])
        for prompt_id in pending_ids:
            qa_rows = groups.get_group(prompt_id).to_dict("records")
            captions = captions_for_prompt_id(caption_matrix, prompt_id)
            questions = [str(row["Question"]) for row in qa_rows]
            prompt = build_gemini_prompt(captions=captions, questions=questions)
            prompt = f"PromptID {prompt_id}\n{prompt}"
            text = judge_fn(prompt)
            model_answers = parse_model_answers(text, len(questions))
            for index, row in enumerate(qa_rows):
                pred = model_answers.get(index, "N/A")
                is_match = pred.lower() == "yes"
                writer.writerow([prompt_id, row["Question"], row["Types"], pred, is_match])
                rows_written += 1
            prompts_processed += 1

    return {
        "backend": config.backend,
        "output_csv": str(output_csv.resolve()),
        "prompt_count": prompts_processed,
        "row_count": rows_written,
        "caption_base": str(caption_base),
    }
