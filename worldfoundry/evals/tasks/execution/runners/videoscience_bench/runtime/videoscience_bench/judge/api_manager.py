from __future__ import annotations

from typing import Any

try:
    from .api_providers import (
        AnthropicVLMAPI,
        GeminiVLMAPI,
        OpenAIVLMAPI,
    )
except ImportError:
    from judge.api_providers import (
        AnthropicVLMAPI,
        GeminiVLMAPI,
        OpenAIVLMAPI,
    )

RUBRIC_TEXT = (
    "You are VLM-Judge evaluating a generated science video.\n"
    "Score each rubric from 1–4 (1=absent/contradictory, 2=weak/partly wrong, 3=mostly correct, 4=clearly correct):\n"
    "a) prompt_consistency — follows instructions: correct setup and correct experiment execution.\n"
    "b) expected_phenomenon — expected physical/chemical outcome is present and correct.\n"
    "c) immutability — objects remain intact/unchanged unless changes are explicitly expected.\n"
    "d) dynamism — other physical laws are obeyed.\n"
    "e) coherence — natural transitions across frames; no flicker/teleport/identity swap.\n"
)

SCHEMA_HINT = (
    "Return JSON with fields:\n"
    '{ "scores": {\n'
    '    "prompt_consistency":1-4,\n'
    '    "expected_phenomenon":1-4,\n'
    '    "immutability":1-4,\n'
    '    "dynamism":1-4,\n'
    '    "coherence":1-4\n'
    "  },\n"
    '  "explanations": {"summary": string, "issues": [string]},\n'
    '  "evidence": {"candidate": [{"t":"0.0s","observation":""}],'
    '               "reference": [{"t":"0.0s","observation":""}]}\n'
    "}\n"
)


def _build_prompt(
    phenomenon: str,
    gt_description: str,
    checklist: dict[str, Any] | None = None,
    cv_metrics: dict[str, Any] | None = None,
) -> str:
    base = (
        RUBRIC_TEXT
        + "\n"
        + SCHEMA_HINT
        + "\n"
        + f"Ground-truth phenomenon: {phenomenon}\n\n"
        + "Ground-truth description (authoritative):\n"
        + gt_description.strip()
    )

    if checklist:
        base += "\n\n" + _format_checklist_for_prompt(checklist)

    if cv_metrics:
        base += "\n\n" + _format_cv_metrics_for_prompt(cv_metrics)

    return base


def _format_checklist_for_prompt(checklist: dict[str, Any]) -> str:
    """
    Turn checklist JSON into human-readable bullets for the judge prompt.
    """
    if not checklist:
        return ""

    cat_names = {
        "phenomenon_congruency": "PHENOMENON CONGRUENCY",
        "correct_dynamism": "CORRECT DYNAMISM",
        "spatio_temporal_continuity": "SPATIO-TEMPORAL CONTINUITY",
        "immutability": "IMMUTABILITY",
        "interaction_realism": "INTERACTION REALISM",
    }

    lines: list[str] = [
        "",
        "Expected phenomenon checklist (each bullet describes what SHOULD be observable in the video):",
    ]

    for key, heading in cat_names.items():
        items = checklist.get(key) or []
        if not isinstance(items, list) or not items:
            continue
        lines.append(f"\n{heading}:")
        for it in items:
            lines.append(f"- {str(it).strip()}")

    lines.append(
        "\nUse this checklist as ground truth for what a correct video should show. "
        "If many items for a category are violated, that category's score should be low. "
        "If almost all items are satisfied, that category's score should be high."
    )

    return "\n".join(lines)


def _format_cv_metrics_for_prompt(cv_metrics: dict[str, Any]) -> str:
    """
    Take CV metrics JSON (from cv_tools/cv_metrics.py) and format a concise,
    human-readable summary for the judge prompt.
    """
    if not cv_metrics:
        return ""

    results = cv_metrics.get("results") or {}
    if not isinstance(results, dict):
        results = {}

    lines: list[str] = [
        "",
        "Computer-vision analysis (noisy; use as secondary evidence, not ground truth):",
    ]

    def _safe_float(d: dict[str, Any] | None, key: str) -> float | None:
        if not isinstance(d, dict):
            return None
        v = d.get(key)
        try:
            return float(v)
        except Exception:
            return None

    # GroundingDINO – presence of requested entities
    gd = results.get("grounding_dino") or results.get("GroundingDINO")
    if isinstance(gd, dict):
        ratio = _safe_float(gd, "frames_with_any_detection_ratio")
        if ratio is not None:
            lines.append(f"- GroundingDINO: target entities detected in ~{ratio * 100:.1f}% of analyzed frames.")

    # ByteTrack – coherence of tracked identities
    bt = results.get("bytetrack")
    if isinstance(bt, dict):
        coh = _safe_float(bt, "coherence_score")
        num_tracks = bt.get("num_tracks")
        avg_len = _safe_float(bt, "avg_track_length_frames")
        pieces = []
        if coh is not None:
            pieces.append(f"coherence_score≈{coh:.2f} (1.0 means long, stable tracks)")
        if isinstance(num_tracks, (int, float)):
            pieces.append(f"{int(num_tracks)} active tracks")
        if avg_len is not None:
            pieces.append(f"avg track length≈{avg_len:.1f} frames")
        if pieces:
            lines.append("- ByteTrack: " + ", ".join(pieces) + ".")

    # RAFT – motion magnitude/direction
    rf = results.get("raft")
    if isinstance(rf, dict):
        mag = _safe_float(rf, "mean_flow_magnitude")
        ang = _safe_float(rf, "mean_flow_direction_degrees")
        if mag is not None:
            desc = f"mean flow magnitude≈{mag:.3f}"
            if ang is not None:
                desc += f", mean direction≈{ang:.1f}° (image coords)."
            else:
                desc += "."
            lines.append("- RAFT optical flow: " + desc)

    # CLIP4Clip – text–video alignment
    clip = results.get("clip4clip") or results.get("CLIP4Clip")
    if isinstance(clip, dict):
        sim = _safe_float(clip, "similarity_cosine")
        if sim is not None:
            lines.append(
                f"- CLIP4Clip text–video alignment: cosine similarity≈{sim:.3f} "
                "(higher generally means better alignment of video with the prompt)."
            )

    # LPIPS – perceptual closeness to reference
    lp = results.get("lpips")
    if isinstance(lp, dict):
        lp_mean = _safe_float(lp, "lpips_mean")
        if lp_mean is not None:
            lines.append(
                f"- LPIPS (VGG, candidate vs ref): mean LPIPS≈{lp_mean:.3f} "
                "(lower means more perceptually similar to the reference video)."
            )

    if len(lines) == 2:
        # nothing useful was added
        return ""

    lines.append(
        "\nUse these signals only as hints: if they strongly contradict what you see in the frames, "
        "trust the actual video content over these metrics."
    )

    return "\n".join(lines)


def judge_experiment(
    provider: str,
    model: str,
    video_path: str,
    phenomenon: str,
    gt_description: str,
    ref_video_path: str | None = None,
    *,
    max_frames: int = 24,
    fps: float | None = None,
    timeout_s: int = 600,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Dispatch to a VLM provider. Providers return unparsed `output_text` and minimal evidence.
    The frontend is responsible for parsing/normalization/scoring.
    """
    extra = dict(extra or {})

    # Pull out optional checklist so we don't accidentally pass it to providers
    checklist = extra.pop("checklist", None)
    cv_metrics = extra.pop("cv_metrics", None)

    extra["judge_prompt"] = _build_prompt(phenomenon, gt_description, checklist, cv_metrics)

    p = (provider or "").lower()
    if p in ("openai", "gpt", "gpt-4o", "gpt-4.1", "o3", "omni", "gpt-5", "gpt-5-pro"):
        api = OpenAIVLMAPI()
    elif p in ("gemini", "google", "google-gemini", "gemini-2.5"):
        api = GeminiVLMAPI()
    elif p in ("anthropic", "claude"):
        api = AnthropicVLMAPI()
    else:
        raise ValueError(f"Unknown provider: {provider}")

    result = api.analyze(
        model=model,
        video_path=video_path,
        phenomenon=phenomenon,
        ref_video_path=ref_video_path,
        max_frames=max_frames,
        fps=fps,
        timeout_s=timeout_s,
        extra=extra,
    )

    # DEBUGGING PRINT
    print("result:")
    print(result.get("output_text", ""))

    return {
        "provider": result.get("provider", provider),
        "model": result.get("model", model),
        "output_text": result.get("output_text", ""),
        "output_mime": result.get("output_mime", "text/plain"),
        "evidence": result.get("evidence", {}),
        "raw": result.get("raw", {}),
    }
