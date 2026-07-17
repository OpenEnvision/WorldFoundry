"""In-tree FETV-EVAL official runtime adapter."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

from worldfoundry.base_models.capabilities import vbench_asset_path
from worldfoundry.core.io.serialization import read_jsonl_objects, write_jsonl
from worldfoundry.evaluation.tasks.execution.framework.benchmark_assets import bundled_benchmark_asset
from worldfoundry.evaluation.tasks.execution.runners.fetv.fetv_prompts import (
    FETV_FRAME_COUNT,
    FETV_GENERATION_MANIFEST_NAME,
    load_fetv_prompt_records,
)

OFFICIAL_RUNTIME_ROOT = Path(__file__).resolve().parent / "runtime" / "fetv_eval"
REPO_ROOT = Path(__file__).resolve().parents[6]
FETV_BLIP_ROOT = REPO_ROOT / "worldfoundry" / "base_models" / "perception_core" / "video_text" / "fetv_blip"
BLIP_CONFIG_PATH = FETV_BLIP_ROOT / "blip_config.yaml"
FETV_STYLEGAN_V_ROOT = (
    REPO_ROOT / "worldfoundry" / "base_models" / "perception_core" / "video_quality" / "fetv_stylegan_v"
)

DEFAULT_MODEL_NAME = "modelscope-t2v"
DEFAULT_METRICS = ("clip_score",)
OFFICIAL_SEEDS = (12, 22, 32, 42)
FVD_METRIC_LIST = "fvd32_16f,fvd64_16f,fvd128_16f,fvd256_16f,fvd300_16f,fvd512_16f,fvd1024_16f"
FETV_PROMPT_ASSET = bundled_benchmark_asset("fetv", "fetv_data.json")
FETV_FID_GEN_META_ASSET = bundled_benchmark_asset("fetv", "sampled_prompts_for_fid_fvd", "prompts_gen.json")
FETV_FID_REAL_META_ASSET = bundled_benchmark_asset("fetv", "sampled_prompts_for_fid_fvd", "prompts_real.json")

REQUIRED_RUNTIME_FILES = (
    "auto_eval.py",
    "compute_fid.py",
    "video_dataset.py",
    "metrics/__init__.py",
    "metrics/clips.py",
)

BASE_MODEL_RUNTIME_FILES = (
    "__init__.py",
    "blip_config.yaml",
    "configs/med_config.json",
    "models/__init__.py",
    "models/blip.py",
    "models/blip_retrieval.py",
    "models/med.py",
    "models/vit.py",
    "utils.py",
)

BASE_MODEL_STYLEGAN_V_FILES = (
    "__init__.py",
    "src/scripts/calc_metrics_for_dataset.py",
    "src/metrics/metric_main.py",
    "src/metrics/metric_utils.py",
    "src/metrics/frechet_video_distance.py",
    "src/dnnlib/util.py",
    "src/training/dataset.py",
    "src/training/layers.py",
    "src/torch_utils/misc.py",
    "src/torch_utils/training_stats.py",
    "src/torch_utils/custom_ops.py",
)

PACKAGE_IMPORTS = {
    "torch": "torch",
    "torchvision": "torchvision",
    "PIL": "PIL",
    "numpy": "numpy",
    "tqdm": "tqdm",
    "clip": "clip",
    "torchmetrics": "torchmetrics",
    "cleanfid": "cleanfid",
    "click": "click",
    "omegaconf": "omegaconf",
    "scipy": "scipy",
    "transformers": "transformers",
    "timm": "timm",
    "fairscale": "fairscale",
    "ruamel.yaml": "ruamel.yaml",
}

METRIC_ALIASES = {
    "clip": "clip_score",
    "clipscore": "clip_score",
    "clip_score": "clip_score",
    "blip": "blip_score",
    "blipscore": "blip_score",
    "blip_score": "blip_score",
    "fid": "fid",
    "fvd": "fvd",
}

_FETV_FRAME_DIR_PATTERN = re.compile(r"^sent(?P<index>\d+)_frames$")


def _has_import(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except Exception:
        return False


def resolve_fetv_runtime_root(explicit: Path | None = None) -> Path:
    if explicit is not None:
        return explicit.expanduser().resolve()
    env_root = os.environ.get("WORLDFOUNDRY_FETV_RUNTIME_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    return OFFICIAL_RUNTIME_ROOT


def resolve_clip_checkpoint(value: str) -> str:
    """Resolve official CLIP aliases to a staged local checkpoint."""

    configured = value.strip()
    alias_assets = {
        "ViT-B/32": "vbench_clip_vit_b32_checkpoint",
        "ViT-L/14": "vbench_clip_vit_l14_checkpoint",
    }
    if configured in alias_assets:
        return str(vbench_asset_path(alias_assets[configured]))
    candidate = Path(configured).expanduser()
    if candidate.is_file():
        return str(candidate.resolve())
    if "/" in configured or "\\" in configured or candidate.suffix:
        raise FileNotFoundError(f"FETV CLIP checkpoint does not exist: {candidate}")
    return configured


def official_runtime_preflight(runtime_root: Path | None = None) -> dict[str, Any]:
    root = resolve_fetv_runtime_root(runtime_root)
    runtime_files = {relative: (root / relative).exists() for relative in REQUIRED_RUNTIME_FILES}
    base_model_files = {relative: (FETV_BLIP_ROOT / relative).exists() for relative in BASE_MODEL_RUNTIME_FILES}
    stylegan_v_files = {relative: (FETV_STYLEGAN_V_ROOT / relative).exists() for relative in BASE_MODEL_STYLEGAN_V_FILES}
    package_imports = {name: _has_import(module) for name, module in PACKAGE_IMPORTS.items()}
    missing = [f"runtime:{name}" for name, ok in runtime_files.items() if not ok]
    missing.extend(f"base_model:fetv_blip/{name}" for name, ok in base_model_files.items() if not ok)
    missing.extend(f"base_model:fetv_stylegan_v/{name}" for name, ok in stylegan_v_files.items() if not ok)
    missing.extend(f"package:{name}" for name, ok in package_imports.items() if not ok)
    return {
        "runtime_root": str(root),
        "base_model_runtime_root": str(FETV_BLIP_ROOT),
        "base_model_stylegan_v_root": str(FETV_STYLEGAN_V_ROOT),
        "runtime_files": runtime_files,
        "base_model_files": base_model_files,
        "base_model_stylegan_v_files": stylegan_v_files,
        "package_imports": package_imports,
        "missing": missing,
        "ok_for_import": not any(not ok for ok in runtime_files.values()) and not any(
            not ok for ok in base_model_files.values()
        ) and not any(
            not ok for ok in stylegan_v_files.values()
        ),
        "ok_for_full_run": not missing,
    }


def _split_tokens(raw: str | Iterable[str] | None) -> tuple[str, ...]:
    if raw is None:
        return ()
    if isinstance(raw, str):
        return tuple(token for token in raw.replace(",", " ").split() if token)
    return tuple(str(token) for token in raw if str(token).strip())


def normalize_metric_tokens(raw: str | Iterable[str] | None) -> tuple[str, ...]:
    tokens = _split_tokens(raw)
    if not tokens:
        tokens = DEFAULT_METRICS
    normalized: list[str] = []
    for token in tokens:
        key = token.strip().lower().replace("-", "_")
        if key == "all":
            for metric in ("clip_score", "blip_score", "fid", "fvd"):
                if metric not in normalized:
                    normalized.append(metric)
            continue
        metric = METRIC_ALIASES.get(key)
        if metric is None:
            raise ValueError(f"unsupported FETV metric: {token}")
        if metric not in normalized:
            normalized.append(metric)
    return tuple(normalized)


def validate_fetv_clip_frame_layout(
    *,
    generated_video_dir: Path,
    prompt_file: Path,
    metrics: Iterable[str],
    limit: int,
) -> dict[str, Any]:
    """Validate the index-sensitive frame layout consumed by FETV-EVAL.

    The official dataset joins prompts to generated media by JSONL line number,
    not by prompt text or ``video_id``.  Guessing that index from an arbitrary
    MP4 filename can silently score a video against the wrong prompt, so this
    adapter intentionally accepts only a manifest-proven ``sentN_frames``
    layout whose prompt rows still match the official source JSONL.
    """

    root = generated_video_dir.expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"FETV generated frame directory does not exist: {root}")
    if limit <= 0:
        raise ValueError("FETV --limit must be a positive integer")

    metric_ids = normalize_metric_tokens(metrics)
    prompt_records = load_fetv_prompt_records(prompt_file=prompt_file)
    prompt_total = len(prompt_records)
    required_count = min(prompt_total, limit)

    manifest_path = root / FETV_GENERATION_MANIFEST_NAME
    if not manifest_path.is_file():
        raise ValueError(
            f"FETV generated artifacts are missing {FETV_GENERATION_MANIFEST_NAME}; "
            "the evaluator will not guess sample_id-to-sent_index mappings from filenames or prompt text"
        )
    manifest_rows = read_jsonl_objects(manifest_path)
    if not manifest_rows:
        raise ValueError(f"FETV generation manifest is empty: {manifest_path}")

    manifest_by_index: dict[int, dict[str, Any]] = {}
    sample_ids: set[str] = set()
    for row_number, row in enumerate(manifest_rows, start=1):
        sample_id = row.get("sample_id")
        sent_index = row.get("sent_index")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"FETV generation manifest row {row_number} has no sample_id: {manifest_path}")
        if sample_id in sample_ids:
            raise ValueError(f"duplicate FETV generation manifest sample_id: {sample_id!r}")
        sample_ids.add(sample_id)
        if isinstance(sent_index, bool) or not isinstance(sent_index, int) or sent_index < 0:
            raise ValueError(f"FETV generation manifest row {row_number} has invalid sent_index: {sent_index!r}")
        if sent_index in manifest_by_index:
            raise ValueError(f"duplicate FETV generation manifest sent_index: {sent_index}")
        expected_name = f"sent{sent_index}_frames"
        if row.get("frame_dir_name") != expected_name:
            raise ValueError(
                f"FETV generation manifest sent_index {sent_index} must name frame_dir_name {expected_name!r}"
            )
        if row.get("status") != "decoded":
            raise ValueError(f"FETV generation manifest sent_index {sent_index} is not a successfully decoded video")
        if sent_index >= prompt_total or row.get("prompt_record") != prompt_records[sent_index]["prompt_record"]:
            raise ValueError(
                f"FETV generation manifest sent_index {sent_index} does not match the official prompt JSONL"
            )
        manifest_by_index[sent_index] = row

    missing_manifest_indices = [index for index in range(required_count) if index not in manifest_by_index]
    if missing_manifest_indices:
        raise ValueError(
            "FETV generation manifest does not cover the bounded prompt prefix; missing sent_index values: "
            + ", ".join(map(str, missing_manifest_indices[:8]))
        )

    indexed_dirs: dict[int, Path] = {}
    for path in root.iterdir():
        match = _FETV_FRAME_DIR_PATTERN.fullmatch(path.name)
        if match is not None and path.is_dir():
            indexed_dirs[int(match.group("index"))] = path

    missing = [index for index in range(required_count) if index not in indexed_dirs]
    if missing:
        flat_videos = sorted(path.name for path in root.iterdir() if path.is_file() and path.suffix.lower() == ".mp4")
        video_note = (
            f" Found {len(flat_videos)} flat MP4 file(s), but they were not converted because their prompt indices "
            "cannot be proven from arbitrary filenames."
            if flat_videos
            else ""
        )
        preview = ", ".join(f"sent{index}_frames" for index in missing[:8])
        raise ValueError(
            "FETV official CLIP/BLIP evaluation requires frame directories named "
            f"sent{{prompt-line-index}}_frames; missing {len(missing)} required directories, starting with {preview}."
            f"{video_note} Stage frames using the exact zero-based order of {prompt_file}."
        )

    empty_or_invalid: list[str] = []
    for index in range(required_count):
        frame_dir = indexed_dirs[index]
        expected_frames = [frame_dir / f"frame{frame_index}.jpg" for frame_index in range(FETV_FRAME_COUNT)]
        entries = [path for path in frame_dir.iterdir() if not path.name.startswith(".")]
        if len(entries) != FETV_FRAME_COUNT or any(not path.is_file() for path in expected_frames):
            empty_or_invalid.append(frame_dir.name)
            continue
        try:
            from PIL import Image

            for path in expected_frames:
                with Image.open(path) as image:
                    image.verify()
        except (ImportError, OSError, ValueError):
            empty_or_invalid.append(frame_dir.name)
    if empty_or_invalid:
        preview = ", ".join(empty_or_invalid[:8])
        raise ValueError(
            f"FETV frame directories must contain exactly frame0.jpg..frame{FETV_FRAME_COUNT - 1}.jpg "
            "readable by the official dataset; "
            f"invalid directories: {preview}"
        )

    return {
        "layout": "fetv_sent_indexed_frames",
        "prompt_file": str(prompt_file),
        "prompt_count": prompt_total,
        "required_frame_directories": required_count,
        "available_frame_directories": len(indexed_dirs),
        "manifest_path": str(manifest_path),
        "manifest_record_count": len(manifest_rows),
        "bounded": required_count < prompt_total,
        "metrics": list(metric_ids),
        "mapping": "prompt_jsonl_zero_based_line_index",
    }


def materialize_bounded_fetv_prompt_file(
    *,
    prompt_file: Path,
    output_dir: Path,
    prompt_count: int,
) -> Path:
    """Write an exact prefix of official rows for a bounded official run."""

    records = load_fetv_prompt_records(prompt_file=prompt_file)
    if prompt_count <= 0 or prompt_count > len(records):
        raise ValueError(f"FETV bounded prompt count must be in [1, {len(records)}], got {prompt_count}")
    destination = output_dir / "runtime_inputs" / f"fetv_data_first_{prompt_count}.jsonl"
    write_jsonl(destination, [record["prompt_record"] for record in records[:prompt_count]])
    return destination.resolve()


def _parse_ints(raw: str | Iterable[int] | None, default: tuple[int, ...]) -> tuple[int, ...]:
    if raw is None:
        return default
    if isinstance(raw, str):
        tokens = raw.replace(",", " ").split()
    else:
        tokens = [str(item) for item in raw]
    values = tuple(int(token) for token in tokens if token.strip())
    return values or default


def _runtime_env(runtime_root: Path, *, cuda_visible_devices: str | None = None) -> dict[str, str]:
    env = os.environ.copy()
    pythonpath_parts = [
        str(REPO_ROOT),
        str(runtime_root),
        str(FETV_STYLEGAN_V_ROOT),
        str(FETV_STYLEGAN_V_ROOT / "src"),
        env.get("PYTHONPATH", ""),
    ]
    env["PYTHONPATH"] = os.pathsep.join(part for part in pythonpath_parts if part)
    if cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    env.setdefault("WORLDFOUNDRY_FETV_ASSETS_ROOT", str(FETV_PROMPT_ASSET.parent))
    return env


def _run_command(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    output_dir: Path,
    name: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = output_dir / f"{name}.stdout.log"
    stderr_path = output_dir / f"{name}.stderr.log"
    started = time.monotonic()
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            stdout=stdout,
            stderr=stderr,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    record = {
        "name": name,
        "command": command,
        "cwd": str(cwd),
        "returncode": completed.returncode,
        "duration_seconds": time.monotonic() - started,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
    }
    if completed.returncode != 0:
        raise RuntimeError(f"FETV official metric {name} failed with exit code {completed.returncode}; see {stderr_path}")
    return record


def _scalar_number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    if isinstance(value, dict):
        for key in ("score", "value", "mean", "average"):
            if key in value:
                score = _scalar_number(value[key])
                if score is not None:
                    return score
    return None


def _mean(values: Iterable[float]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return None if not clean else sum(clean) / len(clean)


def _load_score_json(path: Path) -> list[float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return [score for value in payload.values() if (score := _scalar_number(value)) is not None]
    if isinstance(payload, list):
        return [score for value in payload if (score := _scalar_number(value)) is not None]
    score = _scalar_number(payload)
    return [] if score is None else [score]


def _load_distance_jsonl(path: Path, result_key: str) -> list[float]:
    scores: list[float] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            continue
        results = payload.get("results")
        if not isinstance(results, dict):
            continue
        score = _scalar_number(results.get(result_key))
        if score is not None:
            scores.append(score)
    return scores


def collect_fetv_metric_rows(*, result_root: Path, model_name: str) -> list[dict[str, Any]]:
    metric_values: dict[str, list[float]] = {}

    auto_files = {
        "clip_score": result_root / "CLIPScore" / f"auto_eval_results_{model_name}.json",
        "blip_score": result_root / "BLIPScore" / f"auto_eval_results_{model_name}.json",
    }
    for metric_id, path in auto_files.items():
        if path.is_file():
            metric_values.setdefault(metric_id, []).extend(_load_score_json(path))

    distance_specs = {
        "fid": (result_root / "fid_results" / model_name, "metric-fid1024_16f.jsonl", "fid1024_16f"),
        "fvd": (result_root / "fvd_results" / model_name, "metric-fvd1024_16f.jsonl", "fvd1024_16f"),
    }
    for metric_id, (root, filename, result_key) in distance_specs.items():
        if not root.is_dir():
            continue
        for path in sorted(root.glob(f"*/{filename}")):
            metric_values.setdefault(metric_id, []).extend(_load_distance_jsonl(path, result_key))

    rows = []
    for metric_id, values in sorted(metric_values.items()):
        score = _mean(values)
        if score is None:
            continue
        rows.append(
            {
                "metric_id": metric_id,
                "score": score,
                "source": "fetv_eval_in_tree_runtime",
                "model_name": model_name,
                "record_count": len(values),
            }
        )
    return rows


def write_metric_csv(rows: list[dict[str, Any]], path: Path) -> Path:
    if not rows:
        raise FileNotFoundError("no FETV-EVAL metric outputs were produced by the in-tree runtime")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("metric_id", "score", "source", "model_name", "record_count"))
        writer.writeheader()
        writer.writerows(rows)
    return path


def run_official_fetv_runtime(
    *,
    generated_video_dir: Path,
    output_dir: Path,
    runtime_root: Path | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
    metrics: Iterable[str] | str | None = None,
    python: str = sys.executable,
    prompt_file: Path | None = None,
    clip_model: str = "ViT-B/32",
    is_clip_ft: bool = False,
    blip_config: Path | None = None,
    fid_generated_video_dir: Path | None = None,
    fid_real_video_dir: Path | None = None,
    fvd_generated_video_dir: Path | None = None,
    fvd_real_video_dir: Path | None = None,
    seeds: Iterable[int] = OFFICIAL_SEEDS,
    cuda_visible_devices: str | None = None,
    fvd_gpus: int = 1,
    fvd_resolution: int = 256,
    max_frame_num: int = 64,
    limit: int = 1000,
    timeout_seconds: int = 7200,
) -> dict[str, Any]:
    root = resolve_fetv_runtime_root(runtime_root)
    metric_ids = normalize_metric_tokens(metrics)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_root = output_dir / "auto_eval_results"
    prompt_path = (prompt_file or FETV_PROMPT_ASSET).expanduser().resolve()
    if not prompt_path.is_file():
        prompt_path = (root / "datas" / "fetv_data.json").expanduser().resolve()
    commands: list[dict[str, Any]] = []
    env = _runtime_env(root, cuda_visible_devices=cuda_visible_devices)
    input_layout: dict[str, Any] | None = None

    if "clip_score" in metric_ids or "blip_score" in metric_ids:
        clip_model = resolve_clip_checkpoint(clip_model)
        input_layout = validate_fetv_clip_frame_layout(
            generated_video_dir=generated_video_dir,
            prompt_file=prompt_path,
            metrics=metric_ids,
            limit=limit,
        )
        evaluator_prompt_path = prompt_path
        evaluator_limit = int(input_layout["required_frame_directories"])
        if input_layout["bounded"]:
            evaluator_prompt_path = materialize_bounded_fetv_prompt_file(
                prompt_file=prompt_path,
                output_dir=output_dir,
                prompt_count=evaluator_limit,
            )
        input_layout["evaluator_prompt_file"] = str(evaluator_prompt_path)
        command = [
            python,
            "auto_eval.py",
            "--eval_model",
            clip_model,
            "--prompt_file",
            str(evaluator_prompt_path),
            "--gen_path",
            str(generated_video_dir),
            "--t2v_model",
            model_name,
            "--is_clip_ft",
            "true" if is_clip_ft else "false",
            "--save_results",
            "true",
            "--result_path",
            str(result_root),
            "--max_frm_num",
            str(max_frame_num),
            "--limit",
            str(evaluator_limit),
        ]
        if "blip_score" in metric_ids:
            command.extend(["--blip_config", str((blip_config or BLIP_CONFIG_PATH).resolve())])
        commands.append(
            _run_command(
                command,
                cwd=root,
                env=env,
                output_dir=output_dir,
                name="fetv_clip_blip",
                timeout_seconds=timeout_seconds,
            )
        )

    if "fid" in metric_ids:
        fid_gen = (fid_generated_video_dir or generated_video_dir).expanduser().resolve()
        if fid_real_video_dir is None:
            raise ValueError("--fid-real-video-dir is required when running the FETV FID metric")
        command = [
            python,
            "compute_fid.py",
            "--model",
            model_name,
            "--gen_path",
            str(fid_gen),
            "--real_path",
            str(fid_real_video_dir.expanduser().resolve()),
            "--result_path",
            str(result_root),
            "--gen_meta",
            str(FETV_FID_GEN_META_ASSET),
            "--real_meta",
            str(FETV_FID_REAL_META_ASSET),
        ]
        commands.append(
            _run_command(
                command,
                cwd=root,
                env=env,
                output_dir=output_dir,
                name="fetv_fid",
                timeout_seconds=timeout_seconds,
            )
        )

    if "fvd" in metric_ids:
        fvd_gen = (fvd_generated_video_dir or generated_video_dir).expanduser().resolve()
        if fvd_real_video_dir is None:
            raise ValueError("--fvd-real-video-dir is required when running the FETV FVD metric")
        stylegan_root = FETV_STYLEGAN_V_ROOT
        for seed in seeds:
            command = [
                python,
                "src/scripts/calc_metrics_for_dataset.py",
                "--real_data_path",
                str(fvd_real_video_dir.expanduser().resolve()),
                "--fake_data_path",
                str(fvd_gen),
                "--save_path",
                str(result_root / "fvd_results" / model_name),
                "--metrics",
                FVD_METRIC_LIST,
                "--mirror",
                "1",
                "--gpus",
                str(fvd_gpus),
                "--resolution",
                str(fvd_resolution),
                "--verbose",
                "0",
                "--use_cache",
                "0",
                "--seed",
                str(seed),
            ]
            commands.append(
                _run_command(
                    command,
                    cwd=stylegan_root,
                    env=env,
                    output_dir=output_dir,
                    name=f"fetv_fvd_seed_{seed}",
                    timeout_seconds=timeout_seconds,
                )
            )

    rows = collect_fetv_metric_rows(result_root=result_root, model_name=model_name)
    results_path = write_metric_csv(rows, output_dir / f"fetv_eval_results_{model_name}.csv")
    summary = {
        "backend": "fetv_eval_in_tree",
        "runtime_root": str(root),
        "preflight": official_runtime_preflight(root),
        "model_name": model_name,
        "metrics": list(metric_ids),
        "input_layout": input_layout,
        "commands": commands,
        "results_path": str(results_path),
        "result_root": str(result_root),
    }
    (output_dir / "fetv_official_runtime_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the in-tree FETV-EVAL runtime.")
    parser.add_argument("--generated-video-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--runtime-root", type=Path, default=None)
    parser.add_argument("--model-name", default=os.environ.get("WORLDFOUNDRY_FETV_MODEL_NAME", DEFAULT_MODEL_NAME))
    parser.add_argument("--metrics", nargs="+", default=_split_tokens(os.environ.get("WORLDFOUNDRY_FETV_METRICS")))
    parser.add_argument("--prompt-file", type=Path, default=None)
    parser.add_argument("--clip-model", default=os.environ.get("WORLDFOUNDRY_FETV_CLIP_MODEL", "ViT-B/32"))
    parser.add_argument("--is-clip-ft", action="store_true", default=os.environ.get("WORLDFOUNDRY_FETV_IS_CLIP_FT") == "1")
    parser.add_argument("--blip-config", type=Path, default=None)
    parser.add_argument("--fid-generated-video-dir", type=Path, default=None)
    parser.add_argument("--fid-real-video-dir", type=Path, default=None)
    parser.add_argument("--fvd-generated-video-dir", type=Path, default=None)
    parser.add_argument("--fvd-real-video-dir", type=Path, default=None)
    parser.add_argument("--seeds", default=os.environ.get("WORLDFOUNDRY_FETV_SEEDS", ",".join(map(str, OFFICIAL_SEEDS))))
    parser.add_argument("--cuda-visible-devices", default=os.environ.get("CUDA_VISIBLE_DEVICES"))
    parser.add_argument("--fvd-gpus", type=int, default=int(os.environ.get("WORLDFOUNDRY_FETV_FVD_GPUS", "1")))
    parser.add_argument("--fvd-resolution", type=int, default=int(os.environ.get("WORLDFOUNDRY_FETV_FVD_RESOLUTION", "256")))
    parser.add_argument("--max-frame-num", type=int, default=int(os.environ.get("WORLDFOUNDRY_FETV_MAX_FRAME_NUM", "64")))
    parser.add_argument("--limit", type=int, default=int(os.environ.get("WORLDFOUNDRY_FETV_LIMIT", "1000")))
    parser.add_argument("--python", default=os.environ.get("PYTHON", sys.executable))
    parser.add_argument("--timeout", type=int, default=int(os.environ.get("WORLDFOUNDRY_FETV_TIMEOUT_SECONDS", "7200")))
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.preflight:
        report = official_runtime_preflight(args.runtime_root)
        print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
        return 0 if report["ok_for_import"] else 1
    if args.output_dir is None:
        print("error: --output-dir is required", file=sys.stderr)
        return 2
    if args.generated_video_dir is None:
        print("error: --generated-video-dir is required", file=sys.stderr)
        return 2
    try:
        summary = run_official_fetv_runtime(
            generated_video_dir=args.generated_video_dir.expanduser().resolve(),
            output_dir=args.output_dir.expanduser().resolve(),
            runtime_root=args.runtime_root,
            model_name=args.model_name,
            metrics=args.metrics,
            python=args.python,
            prompt_file=args.prompt_file,
            clip_model=args.clip_model,
            is_clip_ft=args.is_clip_ft,
            blip_config=args.blip_config,
            fid_generated_video_dir=args.fid_generated_video_dir,
            fid_real_video_dir=args.fid_real_video_dir,
            fvd_generated_video_dir=args.fvd_generated_video_dir,
            fvd_real_video_dir=args.fvd_real_video_dir,
            seeds=_parse_ints(args.seeds, OFFICIAL_SEEDS),
            cuda_visible_devices=args.cuda_visible_devices,
            fvd_gpus=args.fvd_gpus,
            fvd_resolution=args.fvd_resolution,
            max_frame_num=args.max_frame_num,
            limit=args.limit,
            timeout_seconds=args.timeout,
        )
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
    else:
        print(f"fetv: in-tree official runtime wrote {summary['results_path']}")
    return 0


__all__ = [
    "OFFICIAL_RUNTIME_ROOT",
    "REQUIRED_RUNTIME_FILES",
    "collect_fetv_metric_rows",
    "normalize_metric_tokens",
    "official_runtime_preflight",
    "resolve_fetv_runtime_root",
    "run_official_fetv_runtime",
    "validate_fetv_clip_frame_layout",
]


if __name__ == "__main__":
    raise SystemExit(main())
