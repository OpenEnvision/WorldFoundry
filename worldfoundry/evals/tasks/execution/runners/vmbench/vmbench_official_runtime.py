"""In-tree VMBench official evaluation runtime.

This module vendors only the VMBench metric orchestration and small benchmark
glue. Heavy perception/model dependencies are resolved from
``worldfoundry.base_models`` or normal Python packages instead of an external
VMBench checkout.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from worldfoundry.evals.tasks.execution.runners.vmbench.vmbench_prompts import (
    materialize_vmbench_meta_info,
)
from worldfoundry.evals.utils import REPO_ROOT

OFFICIAL_RUNTIME_ROOT = Path(__file__).resolve().parent / "runtime" / "official"

DEFAULT_METRIC_ORDER = (
    "pas",
    "ois",
    "tcs",
    "cas",
    "mss",
)

METRIC_SCRIPT = {
    "pas": "perceptible_amplitude_score.py",
    "ois": "object_integrity_score.py",
    "tcs": "temporal_coherence_score.py",
    "cas": "commonsense_adherence_score.py",
    "mss": "motion_smoothness_score.py",
}

REQUIRED_RUNTIME_FILES = (
    "perceptible_amplitude_score.py",
    "object_integrity_score.py",
    "temporal_coherence_score.py",
    "commonsense_adherence_score.py",
    "motion_smoothness_score.py",
    "bench_utils/calculate_score.py",
    "bench_utils/create_meta_info.py",
    "bench_utils/cas_utils.py",
    "bench_utils/tcs_utils.py",
    "bench_utils/pose_utils.py",
    "q_align/model/builder.py",
    "VideoMAEv2/engine_for_finetuning.py",
    "mmpose/configs/body_2d_keypoint/rtmpose/body8/rtmpose-m_8xb256-420e_body8-256x192.py",
)

BASE_MODEL_PATHS = {
    "grounding_dino": "worldfoundry/base_models/perception_core/detection/grounding_dino/models/__init__.py",
    "sam_v1": "worldfoundry/base_models/perception_core/segment/sam_v1/__init__.py",
    "sam2": "worldfoundry/base_models/perception_core/segment/sam2/build_sam.py",
    "cotracker": "worldfoundry/base_models/perception_core/tracking/cotracker/__init__.py",
}

PACKAGE_IMPORTS = {
    "torch": "torch",
    "cv2": "cv2",
    "numpy": "numpy",
    "PIL": "PIL",
    "scipy": "scipy",
    "decord": "decord",
    "timm": "timm",
    "deepspeed": "deepspeed",
    "hydra-core": "hydra",
    "omegaconf": "omegaconf",
    "mmcv": "mmcv",
    "mmengine": "mmengine",
    "mmpose": "mmpose",
    "mmdet": "mmdet",
    "json_tricks": "json_tricks",
    "supervision": "supervision",
}


@dataclass(frozen=True)
class VMBenchOfficialRuntimeConfig:
    python: str = sys.executable
    device: str = "cuda"
    timeout_seconds: int = 7200
    metrics: tuple[str, ...] = DEFAULT_METRIC_ORDER
    use_torchrun_for_cas: bool = True


def _has_import(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except Exception:
        return False


def official_runtime_preflight() -> dict[str, Any]:
    runtime_files = {relative: (OFFICIAL_RUNTIME_ROOT / relative).exists() for relative in REQUIRED_RUNTIME_FILES}
    base_model_paths = {name: (REPO_ROOT / relative).exists() for name, relative in BASE_MODEL_PATHS.items()}
    package_imports = {name: _has_import(module) for name, module in PACKAGE_IMPORTS.items()}
    missing = [f"runtime:{name}" for name, ok in runtime_files.items() if not ok]
    missing.extend(f"base_model:{name}" for name, ok in base_model_paths.items() if not ok)
    missing.extend(f"package:{name}" for name, ok in package_imports.items() if not ok)
    return {
        "runtime_root": str(OFFICIAL_RUNTIME_ROOT),
        "runtime_files": runtime_files,
        "base_model_paths": base_model_paths,
        "package_imports": package_imports,
        "missing": missing,
        "ok_for_import": not any(not ok for ok in runtime_files.values()) and all(base_model_paths.values()),
        "ok_for_full_run": not missing,
    }


def config_from_env(*, python: str | None = None, metrics: Iterable[str] | None = None) -> VMBenchOfficialRuntimeConfig:
    metric_tokens = metrics
    if metric_tokens is None:
        raw_metrics = os.environ.get("WORLDFOUNDRY_VMBENCH_METRICS", "")
        metric_tokens = raw_metrics.replace(",", " ").split() if raw_metrics else DEFAULT_METRIC_ORDER
    use_torchrun = os.environ.get("WORLDFOUNDRY_VMBENCH_USE_TORCHRUN", "1").strip().lower() not in {
        "0",
        "false",
        "no",
    }
    timeout = int(os.environ.get("WORLDFOUNDRY_VMBENCH_TIMEOUT_SECONDS", "7200"))
    return VMBenchOfficialRuntimeConfig(
        python=python or os.environ.get("WORLDFOUNDRY_VMBENCH_PYTHON") or sys.executable,
        device=os.environ.get("WORLDFOUNDRY_VMBENCH_DEVICE", "cuda"),
        timeout_seconds=timeout,
        metrics=tuple(str(metric).strip().lower() for metric in metric_tokens if str(metric).strip()),
        use_torchrun_for_cas=use_torchrun,
    )


def _runtime_env() -> dict[str, str]:
    env = os.environ.copy()
    pythonpath_parts = [
        str(REPO_ROOT),
        str(OFFICIAL_RUNTIME_ROOT),
        env.get("PYTHONPATH", ""),
    ]
    env["PYTHONPATH"] = os.pathsep.join(part for part in pythonpath_parts if part)
    return env


def _metric_command(
    metric: str, *, config: VMBenchOfficialRuntimeConfig, meta_info_path: Path, output_dir: Path
) -> list[str]:
    script = METRIC_SCRIPT[metric]
    if metric == "pas":
        return [
            config.python,
            script,
            "--meta_info_path",
            str(meta_info_path),
            "--box_threshold",
            "0.25",
            "--text_threshold",
            "0.20",
            "--grid_size",
            "30",
            "--device",
            config.device,
        ]
    if metric == "ois":
        return [
            config.python,
            script,
            "--meta-info-path",
            str(meta_info_path),
            "--save-predictions",
            "--device",
            config.device,
        ]
    if metric == "tcs":
        return [
            config.python,
            script,
            "--meta_info_path",
            str(meta_info_path),
            "--box_threshold",
            "0.25",
            "--text_threshold",
            "0.20",
            "--grid_size",
            "50",
            "--device",
            config.device,
        ]
    if metric == "cas":
        base = [
            "commonsense_adherence_score.py",
            "--model",
            "vit_giant_patch14_224",
            "--data_set",
            "Commonsense-Adherence",
            "--nb_classes",
            "5",
            "--meta_info_path",
            str(meta_info_path),
            "--data_path",
            str(output_dir),
            "--finetune",
            os.environ.get("WORLDFOUNDRY_VMBENCH_CAS_CKPT", ".cache/vit_g_vmbench.pt"),
            "--log_dir",
            str(output_dir),
            "--output_dir",
            str(output_dir),
            "--batch_size",
            os.environ.get("WORLDFOUNDRY_VMBENCH_CAS_BATCH_SIZE", "10"),
            "--input_size",
            "224",
            "--num_workers",
            os.environ.get("WORLDFOUNDRY_VMBENCH_NUM_WORKERS", "10"),
            "--drop_path",
            "0.3",
            "--dist_eval",
            "--enable_deepspeed",
            "--eval",
        ]
        if config.use_torchrun_for_cas:
            return [
                "torchrun",
                "--nproc_per_node",
                os.environ.get("WORLDFOUNDRY_VMBENCH_GPUS_PER_NODE", "1"),
                "--master_port",
                os.environ.get("WORLDFOUNDRY_VMBENCH_MASTER_PORT", "16888"),
                *base,
            ]
        return [config.python, *base]
    if metric == "mss":
        return [
            config.python,
            script,
            "--meta_info_path",
            str(meta_info_path),
            "--device",
            config.device if config.device.startswith("cuda") else "cpu",
        ]
    raise ValueError(f"Unsupported VMBench metric {metric!r}")


def _run_command(
    command: list[str], *, output_dir: Path, name: str, config: VMBenchOfficialRuntimeConfig
) -> dict[str, Any]:
    stdout_path = output_dir / f"{name}.stdout.log"
    stderr_path = output_dir / f"{name}.stderr.log"
    started = time.monotonic()
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        completed = subprocess.run(
            command,
            cwd=OFFICIAL_RUNTIME_ROOT,
            env=_runtime_env(),
            stdout=stdout,
            stderr=stderr,
            text=True,
            timeout=config.timeout_seconds,
            check=False,
        )
    return {
        "name": name,
        "command": command,
        "returncode": completed.returncode,
        "duration_seconds": time.monotonic() - started,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
    }


def run_official_vmbench_runtime(
    *,
    video_dir: Path,
    output_dir: Path,
    prompt_manifest: Path | None = None,
    repo_root: Path | None = None,
    config: VMBenchOfficialRuntimeConfig | None = None,
) -> dict[str, Any]:
    config = config or config_from_env()
    output_dir.mkdir(parents=True, exist_ok=True)
    upstream_dir = output_dir / "upstream"
    upstream_dir.mkdir(parents=True, exist_ok=True)
    meta_info_path = upstream_dir / "results.json"
    rows = materialize_vmbench_meta_info(
        video_dir=video_dir,
        output_path=meta_info_path,
        prompt_suite_path=prompt_manifest,
        repo_root=repo_root,
    )
    command_results: list[dict[str, Any]] = []
    for metric in config.metrics:
        command_results.append(
            _run_command(
                _metric_command(metric, config=config, meta_info_path=meta_info_path, output_dir=upstream_dir),
                output_dir=upstream_dir,
                name=metric,
                config=config,
            )
        )
        if command_results[-1]["returncode"] != 0:
            break
    scores_path = upstream_dir / "scores.csv"
    if all(item["returncode"] == 0 for item in command_results):
        command_results.append(
            _run_command(
                [config.python, "bench_utils/calculate_score.py", "-i", str(meta_info_path), "-o", str(scores_path)],
                output_dir=upstream_dir,
                name="calculate_score",
                config=config,
            )
        )
    return {
        "backend": "official_runtime",
        "runtime_root": str(OFFICIAL_RUNTIME_ROOT),
        "preflight": official_runtime_preflight(),
        "meta_info_path": str(meta_info_path),
        "meta_info_count": len(rows),
        "scores_path": str(scores_path),
        "results_path": str(scores_path if scores_path.is_file() else meta_info_path),
        "commands": command_results,
        "returncode": 0 if command_results and all(item["returncode"] == 0 for item in command_results) else 1,
    }


__all__ = [
    "OFFICIAL_RUNTIME_ROOT",
    "VMBenchOfficialRuntimeConfig",
    "config_from_env",
    "official_runtime_preflight",
    "run_official_vmbench_runtime",
]
