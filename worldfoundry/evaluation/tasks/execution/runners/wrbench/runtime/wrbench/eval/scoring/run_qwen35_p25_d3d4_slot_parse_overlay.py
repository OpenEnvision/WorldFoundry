#!/usr/bin/env python3
"""Run current P25/P22 Runtime V2 scoring.

P25 renders D3/D4 score probes as clean task-slot parsing questions, while
delegating D5/D6 score probes to the P22 native hidden-continuity prompt. The
wrapper rejects metadata context by default; current benchmark scoring uses
``--task-context-mode none`` and gets D5/D6 applicability from the separate E14
Qwen3VL evidence gate.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from wrbench.eval.scoring import run_qwen35_p22_overlay as p22


P25_PROMPT_MODE = "runtime_v2_probe_logprob_p25_d3d4_slot_parse"

P25_D3D4_QUESTIONS: dict[str, str] = {
    "D3_POSITION_RELATION": (
        "During visible, judgeable moments, is the task-critical subject or target "
        "in a plausible world-space position, relation, contact, placement, and "
        "path relative to the parsed task slots and stable scene anchors?"
    ),
    "D3_STATIC_ANCHORS": (
        "Do stable scene anchors support the same spatial interpretation for the "
        "task-critical subject, target, support surface, destination, or region?"
    ),
    "D3_NEG_WRONG_SPATIAL": (
        "Is there clear visible spatial counterevidence, such as the subject going "
        "to the wrong target, wrong side, impossible placement, missing required "
        "contact/support/containment, frame-locked sliding, or a path that "
        "contradicts the parsed task slots?"
    ),
    "D4_ACTION_STATE": (
        "Does visible evidence show the task-critical subject or target performing "
        "or reaching the required action, pose, contact result, state change, or "
        "final state from the parsed task slots?"
    ),
    "D4_TEMPORAL_PROGRESS": (
        "Does the required action or state visibly progress, complete, or remain "
        "maintained in a way consistent with the parsed task slots?"
    ),
    "D4_NEG_ABSENT_FREEZE": (
        "Is there clear visible state counterevidence, such as absent action, wrong "
        "behavior, freeze, loop, reset, regression, wrong final result, or an "
        "impossible state transition relative to the parsed task slots?"
    ),
}


def _sampling_text(*, fps: str | None, frames_used: int | None) -> str:
    bits: list[str] = []
    if fps:
        bits.append(f"frames sampled at {fps} fps")
    if frames_used is not None:
        bits.append(f"approximately {frames_used} video frames")
    return ", ".join(bits) if bits else "the frames provided by the runner"


def _is_native_d3d4_score(probe: Any) -> bool:
    return (
        str(getattr(probe, "role", "")) == "score"
        and str(getattr(probe, "dimension", "")) in {"spatial_fidelity", "state_fidelity"}
    )


def _is_spatial_fidelity(probe: Any) -> bool:
    return str(getattr(probe, "dimension", "")) == "spatial_fidelity"


def _is_negative(probe: Any) -> bool:
    return str(getattr(probe, "polarity", "")) == "negative"


def _dimension_guidance(probe: Any) -> str:
    if _is_spatial_fidelity(probe):
        return """Spatial judgment guidance:
- Use the parsed subject, target object, support surface, destination, and region to decide what spatial relation matters.
- Judge coarse 3D world relation, contact, support, containment, placement, and path relative to stable anchors.
- Do not require exact pixel position, identical camera framing, or a fully visible trajectory.
- Camera motion, cropping, and brief hidden intervals are not spatial failures by themselves."""
    return """Action/state judgment guidance:
- Use the parsed subject, target object, required action, contact/result, and final state to decide what state evidence matters.
- A clear final result or stable maintained state can support Yes even if the middle of the action is partly hidden.
- Do not require perfect animation of every intermediate step when the visible evidence supports the requested result.
- Camera motion, cropping, and brief hidden intervals are not state failures by themselves."""


def _answer_rule(probe: Any) -> str:
    if _is_negative(probe):
        return """How to answer this failure question:
- Answer Yes only when clear visible video evidence shows the failure described in the question.
- Do not answer Yes from weak motion, brief artifacts, ambiguous glimpses, or the task description alone.
- If the failure is missing, unclear, or only possible but not visible, answer No."""
    return """How to answer this support question:
- Answer Yes when visible evidence reasonably supports the requested spatial relation, action, result, or maintained state.
- Answer No when the required subject/target/result is absent, too ambiguous to identify, only implied by the task description, or clearly contradicted by the video.
- Prefer moderate support from clear before/after or final-state evidence over demanding a perfect continuous trajectory."""


def _native_d3d4_prompt(
    *,
    world_state_prompt: str,
    probe: Any,
    fps: str | None,
    frames_used: int | None,
    question: str,
) -> str:
    return f"""You are judging one visual question about an AI-generated video.

Answer exactly one token: Yes or No.
Do not output JSON, explanations, markdown, punctuation, or extra words.

Watch the whole sampled video before answering. You are seeing {_sampling_text(fps=fps, frames_used=frames_used)}, not necessarily every source frame.

Task description:
{world_state_prompt}

Before answering, silently parse the task description into these task slots:
- primary subject or actor;
- manipulated or target object;
- support surface, container, destination, region, or other spatial anchor;
- required action, pose, contact, movement, or state change;
- expected contact result, final state, or maintained state.

Then inspect the video directly:
1. Focus on the parsed task-critical subject, target object, region, action, and result.
2. Use stable scene anchors such as the floor, walls, furniture, counters, doors, and fixed background structure.
3. Ignore incidental objects unless they directly affect the parsed task slots.
4. Judge visible evidence, not the task description alone.

{_dimension_guidance(probe)}

{_answer_rule(probe)}

Question:
{question}

Answer exactly one token: Yes or No."""


def _ensure_p25_prompt_mode(runner_args: list[str]) -> list[str]:
    normalized = list(runner_args)
    for idx, arg in enumerate(normalized):
        if arg == "--prompt-mode":
            if idx + 1 >= len(normalized):
                raise SystemExit("--prompt-mode requires a value")
            if normalized[idx + 1] != P25_PROMPT_MODE:
                raise SystemExit(f"this P25 wrapper only supports --prompt-mode {P25_PROMPT_MODE}")
            return normalized
        if arg.startswith("--prompt-mode="):
            if arg.split("=", 1)[1] != P25_PROMPT_MODE:
                raise SystemExit(f"this P25 wrapper only supports --prompt-mode {P25_PROMPT_MODE}")
            return normalized
    return ["--prompt-mode", P25_PROMPT_MODE, *normalized]


def _ensure_no_task_context(runner_args: list[str]) -> list[str]:
    normalized = list(runner_args)
    for idx, arg in enumerate(normalized):
        if arg == "--task-context-mode":
            if idx + 1 >= len(normalized):
                raise SystemExit("--task-context-mode requires a value")
            if normalized[idx + 1] != "none":
                raise SystemExit("current P25/P22 scoring only supports --task-context-mode none")
            return normalized
        if arg.startswith("--task-context-mode="):
            if arg.split("=", 1)[1] != "none":
                raise SystemExit("current P25/P22 scoring only supports --task-context-mode none")
            return normalized
    return ["--task-context-mode", "none", *normalized]


def _reject_evidence_context_args(runner_args: list[str]) -> None:
    for idx, arg in enumerate(runner_args):
        if arg == "--evidence-jsonl":
            if idx + 1 >= len(runner_args):
                raise SystemExit("--evidence-jsonl requires a value")
            raise SystemExit("P25 clean task-slot parsing does not support --evidence-jsonl")
        if arg.startswith("--evidence-jsonl="):
            raise SystemExit("P25 clean task-slot parsing does not support --evidence-jsonl")


def install_p25_overlay(repo_root: Path) -> None:
    p22.install_p22_overlay(repo_root)
    metric_root = p22._metric_root(repo_root)
    if metric_root is not None:
        sys.path.insert(0, str(metric_root))

    from wrbench.eval.scoring import prompts_v2_probe as prompts

    p22_build_runtime_v2_probe_prompt = prompts.build_runtime_v2_probe_prompt
    p22_question_for_probe = prompts.question_for_probe
    p22_active_probe_catalog = prompts.active_probe_catalog
    p22_validate_prompt_mode = prompts.validate_prompt_mode

    prompts.PROMPT_MODE_P25_D3D4_SLOT_PARSE = P25_PROMPT_MODE
    prompts.SUPPORTED_PROMPT_MODES = set(prompts.SUPPORTED_PROMPT_MODES) | {P25_PROMPT_MODE}

    def validate_prompt_mode(prompt_mode: str) -> str:
        if prompt_mode == P25_PROMPT_MODE:
            return prompt_mode
        return p22_validate_prompt_mode(prompt_mode)

    def active_probe_catalog(prompt_mode: str = prompts.DEFAULT_PROMPT_MODE) -> tuple[Any, ...]:
        if prompt_mode == P25_PROMPT_MODE:
            return p22_active_probe_catalog(p22.P22_PROMPT_MODE)
        return p22_active_probe_catalog(prompt_mode)

    def question_for_probe(probe: Any, prompt_mode: str = prompts.DEFAULT_PROMPT_MODE) -> str:
        if prompt_mode == P25_PROMPT_MODE and _is_native_d3d4_score(probe):
            return P25_D3D4_QUESTIONS[probe.probe_id]
        if prompt_mode == P25_PROMPT_MODE:
            return p22_question_for_probe(probe, p22.P22_PROMPT_MODE)
        return p22_question_for_probe(probe, prompt_mode)

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
        if prompt_mode != P25_PROMPT_MODE:
            return p22_build_runtime_v2_probe_prompt(
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
        if evidence_context:
            raise ValueError("P25 clean task-slot parsing does not accept evidence_context")
        if _is_native_d3d4_score(probe):
            return _native_d3d4_prompt(
                world_state_prompt=world_state_prompt,
                probe=probe,
                fps=fps,
                frames_used=frames_used,
                question=question_for_probe(probe, P25_PROMPT_MODE),
            )
        return p22_build_runtime_v2_probe_prompt(
            world_state_prompt=world_state_prompt,
            video_id=video_id,
            probe=probe,
            task_context=task_context,
            fps=fps,
            frames_used=frames_used,
            prompt_mode=p22.P22_PROMPT_MODE,
            evidence_context=evidence_context,
            evidence_context_mode=evidence_context_mode,
        )

    prompts.validate_prompt_mode = validate_prompt_mode
    prompts.active_probe_catalog = active_probe_catalog
    prompts.question_for_probe = question_for_probe
    prompts.build_runtime_v2_probe_prompt = build_runtime_v2_probe_prompt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Qwen3.5 current P25/P22 probe scorer")
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("runner_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    runner_args = args.runner_args
    if runner_args and runner_args[0] == "--":
        runner_args = runner_args[1:]
    if not runner_args:
        parser.error("pass runner arguments after --")

    runner_args = _ensure_p25_prompt_mode(runner_args)
    runner_args = _ensure_no_task_context(runner_args)
    _reject_evidence_context_args(runner_args)
    install_p25_overlay(args.repo_root)
    from wrbench.eval.scoring import run_local_qwen35_probe_logprob_scorer as runner

    return int(runner.main(runner_args))


if __name__ == "__main__":
    raise SystemExit(main())
