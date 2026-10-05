"""
Cross-physics VLM judge for generated videos vs ground-truth references.

For each physics rule (mechanics, thermotics, material), the pipeline:
1) Uses the ground-truth/reference video + prompt to judge if the rule is relevant.
2) If relevant, compares the generated video against a ground-truth video to
   decide compliance with the rule.

Prompts borrow the domain phrasing from:
- thermotics/vlm_analysis.py
- material/vlm_judge.py
- mechanics/vlm_judge.py

 python physical/vlm_judge.py --gen-video data/robotics_test/zip_bag/zip_bag_longlive_bbox_h264.mp4 --gt-video data/robotics_test/zip_bag/output_fixed.mp4 

python physical/vlm_judge.py --gen-video data/robotics_test/zip_bag/output.mp4 --gt-video data/robotics_test/zip_bag/output_fixed.mp4 
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.vlm import (
    chat_vlm_call,
    parse_json_content,
    resolve_model_name,
    resolve_vlm_backend,
    video_vlm_call,
)
from material.prompt import (
    build_compliance_prompt as build_material_compliance_prompt,
    build_relevance_prompt as build_material_relevance_prompt,
)
from mechanics.prompt import (
    build_compliance_prompt as build_mechanics_compliance_prompt,
    build_relevance_prompt as build_mechanics_relevance_prompt,
)
from problem.problem_set import ALL_QUESTIONS, get_question_by_id
from thermotics.prompt import (
    build_compliance_prompt as build_thermotics_compliance_prompt,
    build_relevance_prompt as build_thermotics_relevance_prompt,
)

load_dotenv()


def encode_video_to_data_url(video_path: str) -> Dict:
    with open(video_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{b64}"}}


def relevance_prompts(question_id: str, video_prompt: str) -> Tuple[str, str]:
    q = get_question_by_id(question_id)
    if q is None:
        raise ValueError(f"Unknown question id: {question_id}")

    if q.dimension == "thermotics":
        return build_thermotics_relevance_prompt(question_id, video_prompt)
    if q.dimension == "material":
        return build_material_relevance_prompt(question_id, video_prompt)
    return build_mechanics_relevance_prompt(question_id, video_prompt)


def compliance_prompts(question_id: str, video_prompt: str) -> Tuple[str, str]:
    q = get_question_by_id(question_id)
    if q is None:
        raise ValueError(f"Unknown question id: {question_id}")

    if q.dimension == "thermotics":
        return build_thermotics_compliance_prompt(question_id, video_prompt)
    if q.dimension == "material":
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
    resp = video_vlm_call(
        data_url=data_url,
        system_prompt=system_prompt,
        user_content=user_prompt,
        model_name=model_name,
        backend=backend,
    )

    return parse_json_content(resp)


def call_compliance_vlm(
    gen_video: str,
    gt_video: str,
    system_prompt: str,
    user_prompt: str,
    model_name: str | None,
    backend: str | None = None,
) -> Dict:
    gen_data = encode_video_to_data_url(gen_video)
    gt_data = encode_video_to_data_url(gt_video)
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {"type": "text", "text": "Generated candidate video:"},
                gen_data,
                {"type": "text", "text": "Ground-truth reference video:"},
                gt_data,
            ],
        },
    ]
    resp_json = chat_vlm_call(
        messages,
        model_name=model_name,
        backend=backend,
        timeout=240,
        max_retries=3,
    )
    return parse_json_content(resp_json)


def evaluate_video(
    gen_video: str,
    gt_video: str,
    video_prompt: str,
    model_name: str | None = None,
    backend: str | None = None,
    question_ids: Optional[List[str]] = None,
) -> List[Dict]:
    results: List[Dict] = []
    if question_ids:
        selected_questions = []
        for qid in question_ids:
            q = get_question_by_id(qid)
            if q is None:
                raise ValueError(f"Unknown question id: {qid}")
            selected_questions.append(q)
    else:
        selected_questions = ALL_QUESTIONS

    for q in selected_questions:
        print(f"\n=== Evaluating question {q.qid} ({q.dimension}) ===")
        try:
            rel_system, rel_user = relevance_prompts(q.qid, video_prompt)
            # Relevance should reflect the benchmark task itself. The reference video
            # is more stable than the generated candidate, which may omit the target event.
            rel = call_relevance_vlm(gt_video, rel_system, rel_user, model_name, backend=backend)
            related = bool(rel.get("related", False))
            rel_conf = float(rel.get("confidence", 0.0))
            rel_reason = str(rel.get("reason", ""))
        except Exception as exc:
            print(f"[WARN] Relevance check failed for {q.qid}: {exc}")
            results.append(
                {
                    "question_id": q.qid,
                    "dimension": q.dimension,
                    "error": f"relevance_failed: {exc}",
                    "related": None,
                    "compliant": None,
                    "confidence": 0.0,
                }
            )
            continue

        if not related:
            print(f"Not related (conf={rel_conf:.2f}). Skipping compliance.")
            results.append(
                {
                    "question_id": q.qid,
                    "dimension": q.dimension,
                    "related": False,
                    "relevance_confidence": rel_conf,
                    "relevance_reason": rel_reason,
                    "compliant": None,
                    "confidence": 0.0,
                }
            )
            continue

        try:
            comp_system, comp_user = compliance_prompts(q.qid, video_prompt)
            comp = call_compliance_vlm(
                gen_video,
                gt_video,
                comp_system,
                comp_user,
                model_name,
                backend=backend,
            )
            compliant = bool(comp.get("compliant", False))
            comp_conf = float(comp.get("confidence", 0.0))
            explanation = str(comp.get("explanation", comp.get("observations", "")))
        except Exception as exc:
            print(f"[WARN] Compliance check failed for {q.qid}: {exc}")
            results.append(
                {
                    "question_id": q.qid,
                    "dimension": q.dimension,
                    "related": True,
                    "relevance_confidence": rel_conf,
                    "relevance_reason": rel_reason,
                    "error": f"compliance_failed: {exc}",
                    "compliant": None,
                    "confidence": 0.0,
                }
            )
            continue

        print(f"Related ✓ -> Compliant: {compliant} (conf={comp_conf:.2f})")
        results.append(
            {
                "question_id": q.qid,
                "dimension": q.dimension,
                "related": True,
                "relevance_confidence": rel_conf,
                "relevance_reason": rel_reason,
                "compliant": compliant,
                "confidence": comp_conf,
                "explanation": explanation,
            }
        )

    return results


def main():
    parser = argparse.ArgumentParser(description="VLM judge comparing generated video vs ground-truth across all physics questions")
    parser.add_argument("--gen-video", required=True, help="Path to generated/candidate video")
    parser.add_argument("--gt-video", required=True, help="Path to ground-truth reference video")
    parser.add_argument(
        "--prompt",
        help="Text description of the video content; if omitted, read prompt.txt next to the ground-truth video",
    )
    parser.add_argument(
        "--backend",
        default=None,
        help="VLM backend: api/openrouter or local/qwenvl. Defaults to VLM_BACKEND/openrouter.",
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
    args = parser.parse_args()

    gen_video = args.gen_video
    gt_video = args.gt_video
    if args.prompt:
        video_prompt = args.prompt
    else:
        prompt_path = Path(gt_video).parent / "prompt.txt"
        if not prompt_path.exists():
            raise FileNotFoundError(f"No prompt provided and {prompt_path} not found")
        video_prompt = prompt_path.read_text(encoding="utf-8").strip()
        print(f"Loaded prompt from {prompt_path}")
    vlm_backend = resolve_vlm_backend(args.backend)
    model_name = resolve_model_name(args.model, vlm_backend)

    results = evaluate_video(
        gen_video,
        gt_video,
        video_prompt,
        model_name=model_name,
        backend=vlm_backend,
        question_ids=args.question_ids,
    )

    summary = {
        "generated_video": gen_video,
        "ground_truth_video": gt_video,
        "video_prompt": video_prompt,
        "backend": vlm_backend,
        "model": model_name,
        "results": results,
    }

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
