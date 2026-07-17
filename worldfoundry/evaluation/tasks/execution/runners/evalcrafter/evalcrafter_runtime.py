"""EvalCrafter scorer runtime for WorldFoundry-generated artifacts."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from worldfoundry.evaluation.tasks.execution.runners.evalcrafter.evalcrafter_prompts import (
    CANONICAL_PROMPT_COUNT,
    EXPECTED_VIDEO_COUNT,
    load_prompt_records,
    resolve_evalcrafter_root,
    unique_prompt_records,
)
from worldfoundry.evaluation.tasks.execution.runners.evalcrafter.evalcrafter_raw_metrics import (
    run_raw_video_metrics,
)


def latest_final_result(results_dir: Path) -> Path:
    if results_dir.is_file():
        return results_dir
    direct = results_dir / "final_result.txt"
    if direct.is_file():
        return direct
    candidates = sorted(results_dir.glob("*final_result*.txt"), key=lambda path: path.stat().st_mtime, reverse=True)
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"EvalCrafter results must contain final_result.txt: {results_dir}")

def run_evalcrafter_scorer(
    *,
    generated_artifact_dir: Path,
    output_dir: Path,
    evalcrafter_root: Path | None = None,
    prompt700_path: Path | None = None,
    limit: int | None = None,
    metrics: str | list[str] | tuple[str, ...] | None = None,
    clip_model_path: Path | None = None,
    device: str = "auto",
    batch_size: int = 8,
    use_official_metric_prompts: bool = True,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_records = unique_prompt_records(load_prompt_records(prompt700_path=prompt700_path))
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be positive")
        prompt_records = prompt_records[: int(limit)]
    backend = (
        os.environ.get("WORLDFOUNDRY_EVALCRAFTER_SCORER_BACKEND")
        or os.environ.get("WORLDFOUNDRY_EVALCRAFTER_RUNTIME_BACKEND")
        or "raw-video"
    ).strip().lower()
    if backend not in {"raw-video", "raw_video", "clip"}:
        raise ValueError(
            f"EvalCrafter official-run only supports the raw-video backend, not {backend!r}; "
            "use --mode official-validation to import final_result.txt"
        )
    selected_metrics = metrics or os.environ.get("WORLDFOUNDRY_EVALCRAFTER_METRICS")
    env_clip_model = os.environ.get("WORLDFOUNDRY_EVALCRAFTER_CLIP_MODEL")
    resolved_clip_model = clip_model_path or (Path(env_clip_model) if env_clip_model else None)
    resolved_device = os.environ.get("WORLDFOUNDRY_EVALCRAFTER_DEVICE") or device
    return run_raw_video_metrics(
        videos_dir=generated_artifact_dir,
        prompt_records=prompt_records,
        output_dir=output_dir,
        metrics=selected_metrics,
        clip_model_path=resolved_clip_model,
        evalcrafter_root=evalcrafter_root or resolve_evalcrafter_root(),
        device=resolved_device,
        batch_size=batch_size,
        use_official_metric_prompts=use_official_metric_prompts,
    )


def validate_official_inputs(evalcrafter_root: Path, videos_dir: Path) -> dict[str, Any]:
    from worldfoundry.evaluation.tasks.execution.runners.evalcrafter.evalcrafter_prompts import resolve_prompt700_path

    prompt_path = resolve_prompt700_path(repo_root=evalcrafter_root)
    prompt_lines = [line.strip() for line in prompt_path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
    prompt = {
        "path": str(prompt_path),
        "exists": prompt_path.is_file(),
        "line_count": len(prompt_lines),
        "expected_line_count": CANONICAL_PROMPT_COUNT,
    }
    prompt["ok"] = prompt["exists"] and prompt["line_count"] == CANONICAL_PROMPT_COUNT

    expected_names = {f"{index:04d}.mp4" for index in range(EXPECTED_VIDEO_COUNT)}
    mp4_names: set[str] = set()
    non_video_entries: list[str] = []
    subdirectories: list[str] = []
    if videos_dir.is_dir():
        for path in sorted(videos_dir.iterdir()):
            if path.is_dir():
                subdirectories.append(path.name)
            elif path.is_file() and path.suffix.lower() == ".mp4":
                mp4_names.add(path.name)
            else:
                non_video_entries.append(path.name)

    missing = sorted(expected_names - mp4_names)
    unexpected = sorted(mp4_names - expected_names)
    candidate_video_dirs: list[dict[str, Any]] = []
    if videos_dir.is_dir():
        for candidate in sorted(path for path in videos_dir.rglob("*") if path.is_dir()):
            direct_mp4_count = sum(1 for item in candidate.iterdir() if item.is_file() and item.suffix.lower() == ".mp4")
            if direct_mp4_count:
                candidate_video_dirs.append({"path": str(candidate), "mp4_count": direct_mp4_count})
    videos = {
        "path": str(videos_dir),
        "exists": videos_dir.is_dir(),
        "mp4_count": len(mp4_names) if videos_dir.is_dir() else None,
        "expected_mp4_count": EXPECTED_VIDEO_COUNT,
        "missing_count": len(missing),
        "missing_examples": missing[:20],
        "unexpected_count": len(unexpected),
        "unexpected_examples": unexpected[:20],
        "non_video_entry_count": len(non_video_entries),
        "subdirectory_count": len(subdirectories),
        "candidate_video_dirs": candidate_video_dirs[:20],
    }
    videos["ok"] = (
        videos["exists"]
        and len(mp4_names) == EXPECTED_VIDEO_COUNT
        and not missing
        and not unexpected
        and not non_video_entries
        and not subdirectories
    )

    result = {
        "ok": bool(prompt["ok"] and videos["ok"]),
        "prompt": prompt,
        "videos": videos,
    }
    reasons: list[str] = []
    if not prompt["ok"]:
        reasons.append(f"prompt700.txt must contain exactly {CANONICAL_PROMPT_COUNT} prompts")
    if not videos["ok"]:
        reasons.append(
            f"videos-dir must contain exactly {EXPECTED_VIDEO_COUNT} direct files named 0000.mp4 through 0699.mp4"
        )
    result["reasons"] = reasons
    return result
