"""PhyGenBench official judge runtime (mock results or optional upstream dispatch)."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from worldfoundry.core.io.filesystem.file_utils import materialize_file
from worldfoundry.evaluation.tasks.execution.framework.official_runner import default_benchmark_timeout
from worldfoundry.evaluation.tasks.execution.framework.runner_common import VIDEO_SUFFIXES
from worldfoundry.evaluation.tasks.execution.runners.phygenbench.phygenbench_prompts import (
    load_prompt_records,
    official_video_filename_for_record,
    resolve_phygenbench_root,
    resolve_prompts_json_path,
    unique_generation_records,
)

PHYGENEVAL_DIR_REL = Path("PhyGenEval")
OVERALL_SCRIPT_REL = PHYGENEVAL_DIR_REL / "overall.py"


def resolve_phygeneval_dir(*, explicit: Path | None = None, repo_root: Path | None = None) -> Path | None:
    if explicit is not None:
        path = explicit.expanduser().resolve()
        return path if path.is_dir() else None
    env_dir = os.environ.get("WORLDFOUNDRY_PHYGENEVAL_DIR")
    if env_dir:
        geneval_path = Path(env_dir).expanduser().resolve()
        return geneval_path if geneval_path.is_dir() else None
    root = repo_root or resolve_phygenbench_root()
    if root is None:
        return None
    candidate = root / PHYGENEVAL_DIR_REL
    return candidate if candidate.is_dir() else None


def resolve_overall_script(*, repo_root: Path | None = None) -> Path | None:
    root = repo_root or resolve_phygenbench_root()
    if root is None:
        return None
    candidate = root / OVERALL_SCRIPT_REL
    return candidate if candidate.is_file() else None


def resolve_phyvideos_dir(
    *,
    model_name: str | None = None,
    repo_root: Path | None = None,
) -> Path:
    root = repo_root or resolve_phygenbench_root()
    if root is None:
        raise FileNotFoundError(
            "PhyGenBench in-tree runtime is missing; cannot resolve the PhyVideos working directory."
        )
    resolved_model = model_name or os.environ.get("WORLDFOUNDRY_PHYGENBENCH_MODEL_NAME") or "worldfoundry"
    return (root / "PhyVideos" / resolved_model).resolve()



@dataclass(frozen=True)
class PhyGenBenchJudgeConfig:
    backend: str
    repo_root: Path | None
    prompts_json_path: Path | None
    model_name: str
    strict: bool = False


def judge_config_from_env(*, repo_root: Path | None = None) -> PhyGenBenchJudgeConfig:
    backend = (
        os.environ.get("WORLDFOUNDRY_PHYGENBENCH_JUDGE_BACKEND")
        or os.environ.get("WORLDFOUNDRY_PHYGENBENCH_SCORER_BACKEND")
        or "mock"
    ).strip().lower()
    repo_root = resolve_phygenbench_root(repo_root)
    prompts_json_path = None
    try:
        prompts_json_path = resolve_prompts_json_path(repo_root=repo_root)
    except FileNotFoundError:
        prompts_json_path = None
    strict = os.environ.get("WORLDFOUNDRY_PHYGENBENCH_STRICT", "").strip().lower() in {"1", "true", "yes", "on"}
    return PhyGenBenchJudgeConfig(
        backend=backend,
        repo_root=repo_root,
        prompts_json_path=prompts_json_path,
        model_name=os.environ.get("WORLDFOUNDRY_PHYGENBENCH_MODEL_NAME", "worldfoundry"),
        strict=strict,
    )


def _deterministic_stage_score(*, seed: str, offset: int) -> int:
    digest = hashlib.sha256(f"{seed}:{offset}".encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 4
    return bucket


def _video_stems(generated_artifact_dir: Path) -> set[str]:
    if not generated_artifact_dir.is_dir():
        return set()
    return {
        path.stem
        for path in generated_artifact_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
    }


def _mock_results_payload(
    *,
    generated_artifact_dir: Path,
    prompt_records: list[Mapping[str, Any]],
    model_name: str,
    limit: int | None,
) -> list[dict[str, Any]]:
    available_stems = _video_stems(generated_artifact_dir)
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(prompt_records):
        if limit is not None and index >= int(limit):
            break
        prompt_id = str(record.get("prompt_id") or index + 1)
        official_name = official_video_filename_for_record(dict(record))
        official_stem = Path(official_name).stem
        if available_stems and official_stem not in available_stems and prompt_id not in available_stems:
            continue
        seed = f"phygenbench-mock:{prompt_id}"
        single = _deterministic_stage_score(seed=seed, offset=0)
        multi_gpt = _deterministic_stage_score(seed=seed, offset=1)
        video_gpt = _deterministic_stage_score(seed=seed, offset=2)
        semantic = _deterministic_stage_score(seed=seed, offset=3)
        average = round((single + multi_gpt + video_gpt) / 3)
        rows.append(
            {
                "prompt_id": prompt_id,
                "caption": record.get("caption") or record.get("prompt"),
                "physical_laws": record.get("physical_laws"),
                "sub_category": record.get("sub_category"),
                "main_category": record.get("main_category"),
                "single": single,
                "multi_gpt": multi_gpt,
                "video_gpt": video_gpt,
                "semantic_score": semantic,
                f"{model_name}_average": average,
                "video": str((generated_artifact_dir / official_name).resolve()),
            }
        )
    if not rows and prompt_records:
        record = prompt_records[0]
        prompt_id = str(record.get("prompt_id") or "1")
        seed = f"phygenbench-mock:{prompt_id}"
        rows.append(
            {
                "prompt_id": prompt_id,
                "caption": record.get("caption") or record.get("prompt"),
                "single": 2,
                "multi_gpt": 2,
                "video_gpt": 2,
                "semantic_score": 3,
                f"{model_name}_average": 2,
            }
        )
    return rows


def _materialize_phyvideos(
    *,
    generated_artifact_dir: Path,
    prompt_records: list[Mapping[str, Any]],
    repo_root: Path,
    model_name: str,
) -> Path:
    target_dir = resolve_phyvideos_dir(model_name=model_name, repo_root=repo_root)
    target_dir.mkdir(parents=True, exist_ok=True)
    for record in prompt_records:
        official_name = official_video_filename_for_record(dict(record))
        source = generated_artifact_dir / official_name
        if not source.is_file():
            alt = generated_artifact_dir / f"{record['prompt_id']}.mp4"
            source = alt if alt.is_file() else source
        if source.is_file():
            materialize_file(source, target_dir / official_name, writable=False)
    return target_dir


def _run_upstream_overall(*, repo_root: Path, model_name: str) -> Path:
    overall_script = resolve_overall_script(repo_root=repo_root)
    if overall_script is None:
        raise FileNotFoundError(f"PhyGenEval overall.py not found under {repo_root}")
    result_path = repo_root / "result" / f"{model_name}.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(repo_root.resolve()))
    env["PHYGENBENCH_ROOT"] = str(repo_root.resolve())
    completed = subprocess.run(
        [
            os.environ.get("WORLDFOUNDRY_UNIFIED_PYTHON", "python3"),
            str(overall_script.resolve()),
            "--root",
            str(repo_root.resolve()),
            "--model-name",
            model_name,
        ],
        cwd=str(repo_root.resolve()),
        text=True,
        capture_output=True,
        check=False,
        env=env,
        timeout=default_benchmark_timeout(),
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "PhyGenBench upstream overall.py failed "
            f"(exit={completed.returncode}): {completed.stderr.strip() or completed.stdout.strip()}"
        )
    if not result_path.is_file():
        raise FileNotFoundError(
            "PhyGenBench overall.py did not write expected result JSON. "
            f"Expected {result_path}. Provide the official single/multi/video stage result JSONs under PhyGenEval before running the aggregator."
        )
    return result_path


def run_phygenbench_judge(
    *,
    generated_artifact_dir: Path,
    output_dir: Path,
    config: PhyGenBenchJudgeConfig,
    prompts_json_path: Path | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    results_json = output_dir / "phygenbench_results.json"
    prompts_path = prompts_json_path or config.prompts_json_path
    if prompts_path is None:
        prompts_path = resolve_prompts_json_path(repo_root=config.repo_root)
    prompt_records = unique_generation_records(load_prompt_records(prompts_json_path=prompts_path))
    if limit is not None:
        prompt_records = prompt_records[: int(limit)]

    if config.backend == "mock":
        payload = _mock_results_payload(
            generated_artifact_dir=generated_artifact_dir,
            prompt_records=prompt_records,
            model_name=config.model_name,
            limit=limit,
        )
        results_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return {
            "backend": "mock",
            "results_json": str(results_json.resolve()),
            "video_count": len(_video_stems(generated_artifact_dir)),
            "prompt_count": len(prompt_records),
            "prompts_json_path": str(prompts_path.resolve()),
        }

    repo_root = config.repo_root
    if repo_root is None:
        raise FileNotFoundError(
            "PhyGenBench in-tree runtime is missing. Expected runtime/phygenbench next to this runner, "
            "or set WORLDFOUNDRY_PHYGENBENCH_ROOT to an equivalent local copy."
        )
    _materialize_phyvideos(
        generated_artifact_dir=generated_artifact_dir,
        prompt_records=prompt_records,
        repo_root=repo_root,
        model_name=config.model_name,
    )
    upstream_result = _run_upstream_overall(repo_root=repo_root, model_name=config.model_name)
    materialize_file(upstream_result, results_json)
    return {
        "backend": "official",
        "results_json": str(results_json.resolve()),
        "upstream_result_path": str(upstream_result.resolve()),
        "repo_root": str(repo_root.resolve()),
        "model_name": config.model_name,
        "prompts_json_path": str(prompts_path.resolve()),
    }
