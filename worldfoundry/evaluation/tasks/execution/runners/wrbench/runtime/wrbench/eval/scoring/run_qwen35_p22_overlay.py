#!/usr/bin/env python3
"""Run Runtime V2 Qwen3.5 scoring with the current P22 D5/D6 overlay.

P22 keeps the P9 probe catalog and aggregation, but renders D5/D6 score probes
as native one-question hidden-continuity judgments. D3/D4 score probes and all
Qwen35 diagnostic gate probes delegate to the P9 text. This file is kept inside
``metric/`` so the public scoring package does not depend on Teamwork experiment
paths.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


P22_PROMPT_MODE = "runtime_v2_probe_logprob_p22_native_d5d6_score_only"
BASE_PROMPT_MODE_NAME = "runtime_v2_probe_logprob_p9_d4_p8_d5_p6_combined"


def _metric_root(path: Path) -> Path | None:
    """Return a legacy metric package root for sys.path, or None when wrbench is importable."""
    path = path.resolve()
    candidates = (
        path,
        path / "src",
        path.parent if path.name == "wrbench" else None,
    )
    for root in candidates:
        if root is None:
            continue
        if (root / "scoring" / "prompts_v2_probe.py").exists():
            return root
        if (root / "metric" / "scoring" / "prompts_v2_probe.py").exists():
            return root / "metric"
        if (root / "wrbench" / "eval" / "scoring" / "prompts_v2_probe.py").exists():
            return None
        if (root / "src" / "wrbench" / "eval" / "scoring" / "prompts_v2_probe.py").exists():
            return None
    try:
        import wrbench.eval.scoring.prompts_v2_probe  # noqa: F401
    except ImportError:
        return None
    return None


def _ensure_p22_prompt_mode(runner_args: list[str]) -> list[str]:
    normalized = list(runner_args)
    for idx, arg in enumerate(normalized):
        if arg == "--prompt-mode":
            if idx + 1 >= len(normalized):
                raise SystemExit("--prompt-mode requires a value")
            if normalized[idx + 1] != P22_PROMPT_MODE:
                raise SystemExit(f"this P22 wrapper only supports --prompt-mode {P22_PROMPT_MODE}")
            return normalized
        if arg.startswith("--prompt-mode="):
            if arg.split("=", 1)[1] != P22_PROMPT_MODE:
                raise SystemExit(f"this P22 wrapper only supports --prompt-mode {P22_PROMPT_MODE}")
            return normalized
    return ["--prompt-mode", P22_PROMPT_MODE, *normalized]


def _sampling_text(*, fps: str | None, frames_used: int | None) -> str:
    bits: list[str] = []
    if fps:
        bits.append(f"frames sampled at {fps} fps")
    if frames_used is not None:
        bits.append(f"approximately {frames_used} video frames")
    return ", ".join(bits) if bits else "the frames provided by the runner"


def _native_score_rule(probe: Any) -> str:
    if probe.dimension == "spatial_reasoning":
        return """How to judge this question:
- Compare the last clear view before the relevant person, object, or region became hard to judge with the later clear view after it becomes judgeable again.
- Use stable scene anchors to judge whether the later evidence fits the same world-region continuation.
- A later similar-looking object is not enough by itself to prove spatial continuity for the original prompt-relevant object.
- Brief duplicate-looking blur, partial silhouettes, transparent remnants, motion trails, or render echoes should be ignored unless they are clear, separate, persistent across later sampled evidence, and directly relevant to the prompt.
- Missing middle frames or a hidden interval alone is not failure.
- For questions asking about counterevidence, answer Yes only when clear visible counterevidence is actually shown."""

    if probe.dimension == "state_reasoning":
        return """How to judge this question:
- Compare the last clear action, pose, result, or state before it became hard to judge with the later clear evidence after it becomes judgeable again.
- Judge whether the later evidence looks like a plausible continuation of the same prompt-relevant action or result.
- A later similar-looking person, object, or result is not enough by itself to prove that the original action or result continued.
- Brief duplicate-looking blur, partial silhouettes, transparent remnants, motion trails, or render echoes should be ignored unless they are clear, separate, persistent across later sampled evidence, and directly relevant to the prompt.
- Missing middle frames or a hidden interval alone is not failure.
- For questions asking about counterevidence, answer Yes only when clear visible counterevidence is actually shown."""

    raise RuntimeError(f"no P22 native score rule for {probe.probe_id}")


def install_p22_overlay(repo_root: Path) -> None:
    metric_root = _metric_root(repo_root)
    if metric_root is not None:
        sys.path.insert(0, str(metric_root))

    from wrbench.eval.scoring import prompts_v2_probe as prompts

    base_prompt_mode = prompts.PROMPT_MODE_P9_D4_P8_D5_P6_COMBINED
    if base_prompt_mode != BASE_PROMPT_MODE_NAME:
        raise RuntimeError(f"unexpected P9 prompt mode constant: {base_prompt_mode}")

    base_validate_prompt_mode = prompts.validate_prompt_mode
    base_question_for_probe = prompts.question_for_probe
    base_active_probe_catalog = prompts.active_probe_catalog
    base_build_runtime_v2_probe_prompt = prompts.build_runtime_v2_probe_prompt

    prompts.PROMPT_MODE_P22_NATIVE_ONE_QUESTION = P22_PROMPT_MODE
    prompts.SUPPORTED_PROMPT_MODES = set(prompts.SUPPORTED_PROMPT_MODES) | {P22_PROMPT_MODE}

    def validate_prompt_mode(prompt_mode: str) -> str:
        if prompt_mode == P22_PROMPT_MODE:
            return prompt_mode
        return base_validate_prompt_mode(prompt_mode)

    def question_for_probe(probe: Any, prompt_mode: str = prompts.DEFAULT_PROMPT_MODE) -> str:
        if prompt_mode == P22_PROMPT_MODE:
            return base_question_for_probe(probe, base_prompt_mode)
        return base_question_for_probe(probe, prompt_mode)

    def active_probe_catalog(prompt_mode: str = prompts.DEFAULT_PROMPT_MODE) -> tuple[Any, ...]:
        if prompt_mode == P22_PROMPT_MODE:
            return base_active_probe_catalog(base_prompt_mode)
        return base_active_probe_catalog(prompt_mode)

    def build_runtime_v2_probe_prompt(
        *,
        world_state_prompt: str,
        video_id: str,
        probe: Any,
        task_context: dict[str, Any] | None = None,
        fps: str | None = None,
        frames_used: int | None = None,
        prompt_mode: str = prompts.DEFAULT_PROMPT_MODE,
        evidence_context: dict[str, Any] | None = None,
        evidence_context_mode: str | None = None,
    ) -> str:
        if prompt_mode != P22_PROMPT_MODE:
            return base_build_runtime_v2_probe_prompt(
                world_state_prompt=world_state_prompt,
                video_id=video_id,
                probe=probe,
                task_context=task_context,
                fps=fps,
                frames_used=frames_used,
                prompt_mode=prompt_mode,
                evidence_context=evidence_context,
                evidence_context_mode=evidence_context_mode,
            )
        if probe.role != "score" or probe.dimension not in {"spatial_reasoning", "state_reasoning"}:
            return base_build_runtime_v2_probe_prompt(
                world_state_prompt=world_state_prompt,
                video_id=video_id,
                probe=probe,
                task_context=task_context,
                fps=fps,
                frames_used=frames_used,
                prompt_mode=base_prompt_mode,
                evidence_context=evidence_context,
                evidence_context_mode=evidence_context_mode,
            )

        context_text = prompts.format_task_context(task_context)
        evidence_text = prompts.format_evidence_context(
            evidence_context, evidence_context_mode=evidence_context_mode
        )
        extra_context_block = ""
        if context_text.strip() not in {"", "{}", "- none"} or evidence_text.strip():
            extra_context_block = f"""
Additional prompt/task context, when available:
{context_text}
{evidence_text}
"""

        return f"""You are judging one visual question about an AI-generated video.

Answer exactly one token: Yes or No.
Do not output JSON, explanations, markdown, punctuation, or extra words.

Watch the whole sampled video before answering. You are seeing {_sampling_text(fps=fps, frames_used=frames_used)}, not necessarily every source frame.

Use the text prompt only to identify the main person, object, target, action, and result that matter for this question. Answer from the video evidence, not from the text prompt alone.

{_native_score_rule(probe)}

Text prompt used to generate the video:
{world_state_prompt}
{extra_context_block}
Question:
{question_for_probe(probe, P22_PROMPT_MODE)}

Answer exactly one token: Yes or No."""

    prompts.validate_prompt_mode = validate_prompt_mode
    prompts.question_for_probe = question_for_probe
    prompts.active_probe_catalog = active_probe_catalog
    prompts.build_runtime_v2_probe_prompt = build_runtime_v2_probe_prompt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run Qwen3.5 probe scorer with P22 native D5/D6 score prompts"
    )
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("runner_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    runner_args = args.runner_args
    if runner_args and runner_args[0] == "--":
        runner_args = runner_args[1:]
    if not runner_args:
        parser.error("pass runner arguments after --")

    runner_args = _ensure_p22_prompt_mode(runner_args)
    install_p22_overlay(args.repo_root)
    from wrbench.eval.scoring import run_local_qwen35_probe_logprob_scorer as runner

    return int(runner.main(runner_args))


if __name__ == "__main__":
    raise SystemExit(main())
