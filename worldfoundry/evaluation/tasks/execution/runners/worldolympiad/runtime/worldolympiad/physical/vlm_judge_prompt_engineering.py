"""Cross-physics VLM judge using prompt engineering registries only.

This is behaviorally aligned with `physical/vlm_judge.py`, but it sources all
dimension-specific prompts from:
- mechanics/prompt_engineering.py
- thermotics/prompt_engineering.py
- material/prompt_engineering.py

It does not import other modules from those three pipeline directories.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from material.prompt import (
    build_compliance_prompt as build_material_compliance_prompt,
    build_relevance_prompt as build_material_relevance_prompt,
)
from mechanics.prompt import (
    build_compliance_prompt as build_mechanics_compliance_prompt,
    build_relevance_prompt as build_mechanics_relevance_prompt,
)
from model.vlm import (
    chat_vlm_call,
    parse_json_content,
    resolve_model_name,
    resolve_vlm_backend,
    video_vlm_call,
)
from problem.problem_set import ALL_QUESTIONS, get_question_by_id
from thermotics.prompt import (
    build_compliance_prompt as build_thermotics_compliance_prompt,
    build_relevance_prompt as build_thermotics_relevance_prompt,
)

load_dotenv()


BatchMode = str


def encode_video_to_data_url(video_path: str) -> Dict:
    with open(video_path, "rb") as video_file:
        base64_video = base64.b64encode(video_file.read()).decode("utf-8")
    return {
        "type": "video_url",
        "video_url": {"url": f"data:video/mp4;base64,{base64_video}"},
    }


def select_questions(question_ids: Optional[List[str]] = None):
    if question_ids:
        selected_questions = []
        for question_id in question_ids:
            question = get_question_by_id(question_id)
            if question is None:
                raise ValueError(f"Unknown question id: {question_id}")
            selected_questions.append(question)
        return selected_questions
    return list(ALL_QUESTIONS)


def question_payload(question) -> Dict:
    return {
        "question_id": question.qid,
        "dimension": question.dimension,
        "question": question.question,
        "success_condition": question.success_condition,
    }


def question_batches(questions: List, batch_mode: BatchMode) -> List[List]:
    normalized = (batch_mode or "none").strip().lower()
    if normalized == "all":
        return [questions] if questions else []
    if normalized == "dimension":
        batches = []
        for dimension in ("mechanics", "thermotics", "material"):
            dim_questions = [question for question in questions if question.dimension == dimension]
            if dim_questions:
                batches.append(dim_questions)
        return batches
    if normalized == "none":
        return [[question] for question in questions]
    raise ValueError("Unsupported batch_mode. Use one of: none, dimension, all.")


def build_batch_relevance_prompts(questions: List, video_prompt: str) -> Tuple[str, str]:
    question_list = [question_payload(question) for question in questions]
    system_prompt = """You are PhysicsFilterBatch, an expert video physics relevance evaluator.
You will receive one reference/ground-truth video, its prompt, and a list of physics questions.
For each question, decide whether the reference video contains enough visual evidence to judge that physical rule.

Use related=true when the rule can be judged from visible objects, materials, contacts, state changes, motion, support, heat/cold cues, liquid/gas/solid behavior, deformation, color/material behavior, burning, dissolving, or other relevant physical evidence.
Use related=false only when the rule truly cannot be evaluated from the shown scene.

Return strict JSON only:
{
  "results": [
    {"question_id": "id", "related": true, "confidence": 0.0, "reason": "short evidence"}
  ]
}
Every input question_id must appear exactly once."""
    user_prompt = f"""Video description:
{video_prompt}

Physics questions:
{json.dumps(question_list, ensure_ascii=False, indent=2)}

Evaluate relevance for every question_id. Return only the JSON object."""
    return system_prompt, user_prompt


def build_batch_compliance_prompts(questions: List, video_prompt: str) -> Tuple[str, str]:
    question_list = [question_payload(question) for question in questions]
    system_prompt = """You are PhysicsJudgeBatch, an expert video physics evaluator.
You will receive two videos:
1. Generated candidate video: the video to judge.
2. Ground-truth reference video: context for the intended event, scene, timing, and physical evidence.

For each physics question, judge whether the generated candidate follows the physical rule and remains consistent with the reference. Do not require frame-exact matching, but require plausible physics, object/material identity consistency, temporal order, and visible evidence.

Return strict JSON only:
{
  "results": [
    {
      "question_id": "id",
      "compliant": true,
      "confidence": 0.0,
      "explanation": "3-5 sentences with specific visual evidence",
      "observations": "short concrete evidence"
    }
  ]
}
Every input question_id must appear exactly once."""
    user_prompt = f"""Video description:
{video_prompt}

Related physics questions to judge:
{json.dumps(question_list, ensure_ascii=False, indent=2)}

Compare the generated candidate against the reference for every question_id. Return only the JSON object."""
    return system_prompt, user_prompt


def normalize_batch_results(parsed: Dict, expected_ids: List[str], stage: str) -> Dict[str, Dict]:
    if not isinstance(parsed, dict) or not isinstance(parsed.get("results"), list):
        raise ValueError(f"Batch {stage} response must be a JSON object with a 'results' list.")

    results_by_id: Dict[str, Dict] = {}
    for item in parsed["results"]:
        if not isinstance(item, dict):
            continue
        question_id = str(item.get("question_id", "")).strip()
        if not question_id or question_id not in expected_ids:
            continue
        results_by_id[question_id] = item

    missing = [question_id for question_id in expected_ids if question_id not in results_by_id]
    if missing:
        raise ValueError(f"Batch {stage} response missing question_id(s): {missing}")
    return results_by_id


def relevance_prompts(question_id: str, video_prompt: str) -> Tuple[str, str]:
    question = get_question_by_id(question_id)
    if question is None:
        raise ValueError(f"Unknown question id: {question_id}")

    if question.dimension == "thermotics":
        return build_thermotics_relevance_prompt(question_id, video_prompt)
    if question.dimension == "material":
        return build_material_relevance_prompt(question_id, video_prompt)
    return build_mechanics_relevance_prompt(question_id, video_prompt)


def compliance_prompts(question_id: str, video_prompt: str) -> Tuple[str, str]:
    question = get_question_by_id(question_id)
    if question is None:
        raise ValueError(f"Unknown question id: {question_id}")

    if question.dimension == "thermotics":
        return build_thermotics_compliance_prompt(question_id, video_prompt)
    if question.dimension == "material":
        return build_material_compliance_prompt(question_id, video_prompt)
    return build_mechanics_compliance_prompt(question_id, video_prompt)


def call_relevance_vlm(
    video_path: str,
    system_prompt: str,
    user_prompt: str,
    model_name: str | None,
    backend: str | None = None,
) -> Dict:
    data_url = encode_video_to_data_url(video_path)
    response = video_vlm_call(
        data_url=data_url,
        system_prompt=system_prompt,
        user_content=user_prompt,
        model_name=model_name,
        backend=backend,
    )

    print(parse_json_content(response))
    return parse_json_content(response)


def call_compliance_vlm(
    gen_video: str,
    gt_video: str,
    system_prompt: str,
    user_prompt: str,
    model_name: str | None,
    backend: str | None = None,
) -> Dict:
    generated_data = encode_video_to_data_url(gen_video)
    reference_data = encode_video_to_data_url(gt_video)
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {"type": "text", "text": "Generated candidate video:"},
                generated_data,
                {"type": "text", "text": "Ground-truth reference video:"},
                reference_data,
            ],
        },
    ]
    response = chat_vlm_call(
        messages,
        model_name=model_name,
        backend=backend,
        timeout=240,
        max_retries=3,
    )
    return parse_json_content(response)


def call_batch_relevance_vlm(
    gt_video: str,
    questions: List,
    video_prompt: str,
    model_name: str | None,
    backend: str | None = None,
) -> Dict[str, Dict]:
    system_prompt, user_prompt = build_batch_relevance_prompts(questions, video_prompt)
    reference_data = encode_video_to_data_url(gt_video)
    response = chat_vlm_call(
        [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "text", "text": "Ground-truth reference video:"},
                    reference_data,
                ],
            },
        ],
        model_name=model_name,
        backend=backend,
        timeout=240,
        max_retries=3,
    )
    parsed = parse_json_content(response)
    return normalize_batch_results(parsed, [question.qid for question in questions], "relevance")


def call_batch_compliance_vlm(
    gen_video: str,
    gt_video: str,
    questions: List,
    video_prompt: str,
    model_name: str | None,
    backend: str | None = None,
) -> Dict[str, Dict]:
    system_prompt, user_prompt = build_batch_compliance_prompts(questions, video_prompt)
    generated_data = encode_video_to_data_url(gen_video)
    reference_data = encode_video_to_data_url(gt_video)
    response = chat_vlm_call(
        [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "text", "text": "Generated candidate video:"},
                    generated_data,
                    {"type": "text", "text": "Ground-truth reference video:"},
                    reference_data,
                ],
            },
        ],
        model_name=model_name,
        backend=backend,
        timeout=300,
        max_retries=3,
    )
    parsed = parse_json_content(response)
    return normalize_batch_results(parsed, [question.qid for question in questions], "compliance")


def evaluate_video_batch(
    gen_video: str,
    gt_video: str,
    video_prompt: str,
    model_name: str | None = None,
    backend: str | None = None,
    question_ids: Optional[List[str]] = None,
    batch_mode: BatchMode = "dimension",
) -> List[Dict]:
    selected_questions = select_questions(question_ids)
    results_by_qid: Dict[str, Dict] = {}

    for batch in question_batches(selected_questions, batch_mode):
        label = "all" if batch_mode == "all" else batch[0].dimension
        print(f"\n=== Batch relevance for {label}: {len(batch)} question(s) ===")
        try:
            relevance_by_qid = call_batch_relevance_vlm(
                gt_video,
                batch,
                video_prompt,
                model_name,
                backend=backend,
            )
        except Exception as exc:
            print(f"[WARN] Batch relevance failed for {label}; falling back to per-question. Error: {exc}")
            fallback_results = evaluate_video(
                gen_video=gen_video,
                gt_video=gt_video,
                video_prompt=video_prompt,
                model_name=model_name,
                backend=backend,
                question_ids=[question.qid for question in batch],
                batch_mode="none",
            )
            for item in fallback_results:
                results_by_qid[item["question_id"]] = item
            continue

        related_questions = []
        for question in batch:
            rel = relevance_by_qid[question.qid]
            related = bool(rel.get("related", False))
            relevance_confidence = float(rel.get("confidence", 0.0))
            relevance_reason = str(rel.get("reason", ""))
            if related:
                related_questions.append(question)
                continue
            print(f"{question.qid}: not related (conf={relevance_confidence:.2f})")
            results_by_qid[question.qid] = {
                "question_id": question.qid,
                "dimension": question.dimension,
                "related": False,
                "relevance_confidence": relevance_confidence,
                "relevance_reason": relevance_reason,
                "compliant": None,
                "confidence": 0.0,
            }

        if not related_questions:
            continue

        print(f"=== Batch compliance for {label}: {len(related_questions)} related question(s) ===")
        try:
            compliance_by_qid = call_batch_compliance_vlm(
                gen_video,
                gt_video,
                related_questions,
                video_prompt,
                model_name,
                backend=backend,
            )
        except Exception as exc:
            print(f"[WARN] Batch compliance failed for {label}; falling back to per-question. Error: {exc}")
            fallback_results = evaluate_video(
                gen_video=gen_video,
                gt_video=gt_video,
                video_prompt=video_prompt,
                model_name=model_name,
                backend=backend,
                question_ids=[question.qid for question in related_questions],
                batch_mode="none",
            )
            for item in fallback_results:
                results_by_qid[item["question_id"]] = item
            continue

        for question in related_questions:
            rel = relevance_by_qid[question.qid]
            comp = compliance_by_qid[question.qid]
            compliant = bool(comp.get("compliant", False))
            confidence = float(comp.get("confidence", 0.0))
            explanation = str(comp.get("explanation", comp.get("observations", "")))
            print(f"{question.qid}: related -> compliant={compliant} (conf={confidence:.2f})")
            results_by_qid[question.qid] = {
                "question_id": question.qid,
                "dimension": question.dimension,
                "related": True,
                "relevance_confidence": float(rel.get("confidence", 0.0)),
                "relevance_reason": str(rel.get("reason", "")),
                "compliant": compliant,
                "confidence": confidence,
                "explanation": explanation,
                "observations": str(comp.get("observations", "")),
            }

    return [results_by_qid[question.qid] for question in selected_questions if question.qid in results_by_qid]


def evaluate_video(
    gen_video: str,
    gt_video: str,
    video_prompt: str,
    model_name: str | None = None,
    backend: str | None = None,
    question_ids: Optional[List[str]] = None,
    batch_mode: BatchMode = "none",
) -> List[Dict]:
    if (batch_mode or "none").strip().lower() != "none":
        return evaluate_video_batch(
            gen_video=gen_video,
            gt_video=gt_video,
            video_prompt=video_prompt,
            model_name=model_name,
            backend=backend,
            question_ids=question_ids,
            batch_mode=batch_mode,
        )

    results: List[Dict] = []
    selected_questions = select_questions(question_ids)

    for question in selected_questions:
        print(f"\n=== Evaluating question {question.qid} ({question.dimension}) ===")
        try:
            rel_system, rel_user = relevance_prompts(question.qid, video_prompt)
            # Relevance should reflect whether the benchmark task contains this physics concept.
            # Using the reference video is more stable than using the generated candidate,
            # which may omit or corrupt the very interaction we want to judge.
            relevance = call_relevance_vlm(gt_video, rel_system, rel_user, model_name, backend=backend)
            related = bool(relevance.get("related", False))
            relevance_confidence = float(relevance.get("confidence", 0.0))
            relevance_reason = str(relevance.get("reason", ""))
        except Exception as exc:
            print(f"[WARN] Relevance check failed for {question.qid}: {exc}")
            results.append(
                {
                    "question_id": question.qid,
                    "dimension": question.dimension,
                    "error": f"relevance_failed: {exc}",
                    "related": None,
                    "compliant": None,
                    "confidence": 0.0,
                }
            )
            continue

        if not related:
            print(f"Not related (conf={relevance_confidence:.2f}). Skipping compliance.")
            results.append(
                {
                    "question_id": question.qid,
                    "dimension": question.dimension,
                    "related": False,
                    "relevance_confidence": relevance_confidence,
                    "relevance_reason": relevance_reason,
                    "compliant": None,
                    "confidence": 0.0,
                }
            )
            continue

        try:
            comp_system, comp_user = compliance_prompts(question.qid, video_prompt)
            compliance = call_compliance_vlm(
                gen_video,
                gt_video,
                comp_system,
                comp_user,
                model_name,
                backend=backend,
            )
            compliant = bool(compliance.get("compliant", False))
            confidence = float(compliance.get("confidence", 0.0))
            explanation = str(compliance.get("explanation", compliance.get("observations", "")))
        except Exception as exc:
            print(f"[WARN] Compliance check failed for {question.qid}: {exc}")
            results.append(
                {
                    "question_id": question.qid,
                    "dimension": question.dimension,
                    "related": True,
                    "relevance_confidence": relevance_confidence,
                    "relevance_reason": relevance_reason,
                    "error": f"compliance_failed: {exc}",
                    "compliant": None,
                    "confidence": 0.0,
                }
            )
            continue

        print(f"Related -> Compliant: {compliant} (conf={confidence:.2f})")
        results.append(
            {
                "question_id": question.qid,
                "dimension": question.dimension,
                "related": True,
                "relevance_confidence": relevance_confidence,
                "relevance_reason": relevance_reason,
                "compliant": compliant,
                "confidence": confidence,
                "explanation": explanation,
            }
        )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="VLM judge comparing generated video vs ground-truth using prompt engineering registries"
    )
    parser.add_argument("--gen-video", required=True, help="Path to generated/candidate video")
    parser.add_argument("--gt-video", required=True, help="Path to ground-truth reference video")
    parser.add_argument(
        "--prompt",
        help="Text description of the video content; if omitted, read prompt.txt next to the ground-truth video",
    )
    parser.add_argument(
        "--backend",
        default=None,
        help="VLM backend: qwenvl_server, local/qwenvl, or api/openrouter. Defaults to qwenvl_server.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="VLM model name or local model path. Defaults to the selected backend's default model.",
    )
    parser.add_argument("--output", help="Optional path to save JSON results")
    parser.add_argument(
        "--question-ids",
        nargs="+",
        help="Specific question IDs to evaluate (e.g., hardness color_mixing); defaults to all questions",
    )
    parser.add_argument(
        "--batch-mode",
        choices=["none", "dimension", "all"],
        default="none",
        help="Batch physical questions into fewer VLM calls. 'dimension' is recommended for speed/stability.",
    )
    args = parser.parse_args()

    if args.prompt:
        video_prompt = args.prompt
    else:
        prompt_path = Path(args.gt_video).parent / "prompt.txt"
        if not prompt_path.exists():
            raise FileNotFoundError(f"No prompt provided and {prompt_path} not found")
        video_prompt = prompt_path.read_text(encoding="utf-8").strip()
        print(f"Loaded prompt from {prompt_path}")
    vlm_backend = resolve_vlm_backend(args.backend)
    model_name = resolve_model_name(args.model, vlm_backend)

    results = evaluate_video(
        gen_video=args.gen_video,
        gt_video=args.gt_video,
        video_prompt=video_prompt,
        model_name=model_name,
        backend=vlm_backend,
        question_ids=args.question_ids,
        batch_mode=args.batch_mode,
    )

    summary = {
        "generated_video": args.gen_video,
        "ground_truth_video": args.gt_video,
        "video_prompt": video_prompt,
        "backend": vlm_backend,
        "model": model_name,
        "batch_mode": args.batch_mode,
        "results": results,
    }

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()
