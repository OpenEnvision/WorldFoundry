"""In-tree evaluator for NVIDIA's public SANA-WM 80-scene benchmark.

The runner owns the benchmark-specific layout, prompt binding, revisit-pair
protocol, temporal-window aggregation, and scorecard contract.  It deliberately
reuses WorldFoundry's resident VBench and Pi3 implementations rather than
executing an NVlabs/Sana checkout at evaluation time.
"""

from __future__ import annotations



from worldfoundry.evaluation.tasks.execution.framework.runner_common import SCORECARD_SCHEMA_VERSION, VIDEO_SUFFIXES

import argparse
import contextlib
import csv
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping

from .sana_wm_bench_prompts import (
    CANONICAL_FPS,
    CANONICAL_FRAME_COUNT,
    CANONICAL_SCENE_COUNT,
    SPLITS,
    load_manifest_rows,
    normalize_split,
)

BENCHMARK_ID = "sana-wm-bench"
VBENCH_DIMS = (
    "subject_consistency", "background_consistency", "temporal_flickering", "motion_smoothness",
    "dynamic_degree", "aesthetic_quality", "imaging_quality", "overall_consistency", "temporal_style",
)
TEMPORAL_DIMS = ("subject_consistency", "background_consistency", "temporal_flickering", "imaging_quality")
METRIC_SPECS = {
    "vbench_overall": ("VBench Overall", True, "vbench"),
    "vbench_total_score": ("VBench total_score (diagnostic)", True, "vbench"),
    "revisit_psnr": ("Revisit PSNR", True, "revisit"),
    "revisit_ssim": ("Revisit SSIM", True, "revisit"),
    "revisit_lpips": ("Revisit LPIPS", False, "revisit"),
    "rot_err_deg": ("Camera rotation error (degrees)", False, "camera"),
    "trans_err_rel": ("Camera translation error (relative)", False, "camera"),
    "cam_mc_rel": ("Camera motion consistency error (relative)", False, "camera"),
    "delta_iq": ("Temporal imaging-quality degradation", False, "temporal"),
}
_LPIPS_MODEL: Any | None = None


def env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else None


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _video_files(video_dir: Path, limit: int | None = None) -> list[Path]:
    files = sorted(video_dir.glob("*_generated.mp4"))
    return files if limit is None else files[: max(0, int(limit))]


def _split_meta_path(dataset_root: Path, split: str) -> Path:
    return dataset_root / SPLITS[split] / "scene_trajectories_v2.json"


def _validate_dataset(dataset_root: Path, split: str, *, strict: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = load_manifest_rows(dataset_root, split)
    meta_path = _split_meta_path(dataset_root, split)
    if not meta_path.is_file():
        raise FileNotFoundError(f"SANA-WM Bench metadata is missing: {meta_path}")
    metadata = _json(meta_path)
    if not isinstance(metadata, dict):
        raise ValueError(f"SANA-WM Bench metadata must be an object: {meta_path}")
    if strict and len(rows) != CANONICAL_SCENE_COUNT:
        raise ValueError(f"{split} must contain {CANONICAL_SCENE_COUNT} scenes, found {len(rows)}")
    for row in rows:
        scene_id = str(row["id"])
        image = dataset_root / str(row.get("image_path") or f"images/{scene_id}.png")
        camera = dataset_root / str(row.get("camera_path") or "")
        if not image.is_file() or not camera.is_file():
            missing = image if not image.is_file() else camera
            raise FileNotFoundError(f"SANA-WM Bench scene asset is missing: {missing}")
    return rows, metadata


@contextlib.contextmanager
def _torch_load_compat():
    """Allow trusted official VBench weight files under PyTorch 2.6+."""
    import torch

    original = torch.load
    def compatible(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("weights_only", False)
        return original(*args, **kwargs)
    torch.load = compatible
    try:
        yield
    finally:
        torch.load = original


def _load_vbench() -> tuple[type[Any], Path]:
    from worldfoundry.evaluation.tasks.execution.runners.vbench.runtime import vbench as in_tree_vbench

    # VBench dynamically imports ``vbench.<dimension>``.  Pin that name to our
    # resident module even when a pip package with the same name is installed.
    sys.modules["vbench"] = in_tree_vbench
    full_info = Path(in_tree_vbench.__file__).with_name("VBench_full_info.json")
    if not full_info.is_file():
        from worldfoundry.evaluation.tasks.execution.runners.vbench.vbench_official_impl import VBENCH_FULL_INFO_ASSET
        full_info = VBENCH_FULL_INFO_ASSET
    if not full_info.is_file():
        raise FileNotFoundError(f"in-tree VBench full-info asset is missing: {full_info}")
    return in_tree_vbench.VBench, full_info


def _raw_vbench_value(path: Path, dimension: str) -> float | None:
    if not path.is_file():
        return None
    value = _json(path)
    payload = value.get(dimension) if isinstance(value, dict) else None
    if isinstance(payload, list) and payload:
        return _number(payload[0])
    return _number(payload)


def _vbench_scores(video_dir: Path, output_dir: Path, prompts: Mapping[str, str], dimensions: Iterable[str]) -> dict[str, float]:
    import torch

    VBench, full_info = _load_vbench()
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_videos = video_dir.resolve()
    prompt_map = {path.name: prompts.get(path.stem.removesuffix("_generated"), path.stem) for path in _video_files(resolved_videos)}
    scores: dict[str, float] = {}
    for dimension in dimensions:
        result_path = output_dir / f"eval_{dimension}_eval_results.json"
        score = _raw_vbench_value(result_path, dimension)
        if score is not None:
            scores[dimension] = score
            continue
        runner = VBench(torch.device("cuda"), str(full_info), str(output_dir))
        with _torch_load_compat():
            runner.evaluate(
                videos_path=str(resolved_videos), name=f"eval_{dimension}", dimension_list=[dimension],
                mode="custom_input", prompt_list=prompt_map, local=dimension == "subject_consistency",
            )
        score = _raw_vbench_value(result_path, dimension)
        if score is None:
            raise RuntimeError(f"VBench did not write a valid {dimension} result under {output_dir}")
        scores[dimension] = score
    return scores


def _vbench_aggregate(raw: Mapping[str, float]) -> dict[str, float]:
    from worldfoundry.evaluation.tasks.execution.runners.vbench.vbench_official_impl import (
        VBENCH_DIMENSION_WEIGHTS,
        VBENCH_NORMALIZATION,
    )

    def normalized(name: str) -> float:
        bounds = VBENCH_NORMALIZATION[name]
        return max(0.0, min(1.0, (raw[name] - bounds["min"]) / (bounds["max"] - bounds["min"])))

    quality_dims = ("subject_consistency", "background_consistency", "temporal_flickering", "motion_smoothness", "dynamic_degree", "aesthetic_quality", "imaging_quality")
    semantic_dims = ("overall_consistency", "temporal_style")
    quality_weight = sum(VBENCH_DIMENSION_WEIGHTS[name] for name in quality_dims)
    quality = sum(normalized(name) * VBENCH_DIMENSION_WEIGHTS[name] for name in quality_dims) / quality_weight
    semantic = mean(normalized(name) for name in semantic_dims)
    # The official SANA-WM script averages whichever semantic dimensions are
    # present in its nine-dimension run, then keeps total_score as diagnostic.
    return {"vbench_overall": round(quality * 100.0, 4), "vbench_total_score": round((4.0 * quality + semantic) / 5.0, 4), **dict(raw)}


def _frame_reader(path: Path) -> tuple[float, Any, int]:
    import numpy as np

    try:
        from decord import VideoReader
        reader = VideoReader(str(path))
        return float(reader.get_avg_fps()), lambda index: np.asarray(reader[index].asnumpy()), len(reader)
    except ImportError:
        import imageio.v3 as iio
        frames = iio.imread(path)
        return float(CANONICAL_FPS), lambda index: np.asarray(frames[index]), len(frames)


def _pair_indices(pair: Any) -> tuple[int, int] | None:
    if isinstance(pair, (list, tuple)) and len(pair) >= 2:
        return int(pair[0]), int(pair[1])
    if isinstance(pair, Mapping):
        left = next((pair.get(key) for key in ("frame_a", "frame1", "source_frame", "i", "start")), None)
        right = next((pair.get(key) for key in ("frame_b", "frame2", "target_frame", "j", "end")), None)
        if left is not None and right is not None:
            return int(left), int(right)
    return None


def _psnr(left: np.ndarray, right: np.ndarray) -> float:
    import numpy as np

    mse = float(np.mean((left.astype(np.float64) - right.astype(np.float64)) ** 2))
    return 100.0 if mse < 1e-10 else float(10.0 * np.log10(255.0**2 / mse))


def _ssim(left: np.ndarray, right: np.ndarray) -> float:
    import numpy as np

    try:
        from scipy.ndimage import uniform_filter
    except ImportError as exc:
        raise RuntimeError("SANA-WM Bench revisit scoring requires scipy") from exc
    values = []
    for channel in range(min(left.shape[-1], 3)):
        a, b = left[..., channel].astype(np.float64), right[..., channel].astype(np.float64)
        ma, mb = uniform_filter(a, 11), uniform_filter(b, 11)
        va, vb = uniform_filter(a * a, 11) - ma * ma, uniform_filter(b * b, 11) - mb * mb
        cov = uniform_filter(a * b, 11) - ma * mb
        values.append(float(np.mean(((2 * ma * mb + 6.5025) * (2 * cov + 58.5225)) / ((ma * ma + mb * mb + 6.5025) * (va + vb + 58.5225)))))
    return float(mean(values))


def _lpips(left: np.ndarray, right: np.ndarray, device: str) -> float:
    global _LPIPS_MODEL
    import torch

    try:
        import lpips
    except ImportError as exc:
        raise RuntimeError("--revisit-lpips requires the lpips package") from exc
    if _LPIPS_MODEL is None:
        _LPIPS_MODEL = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
        for parameter in _LPIPS_MODEL.parameters():
            parameter.requires_grad_(False)
    first = torch.from_numpy(left).permute(2, 0, 1).float().unsqueeze(0).to(device) / 127.5 - 1.0
    second = torch.from_numpy(right).permute(2, 0, 1).float().unsqueeze(0).to(device) / 127.5 - 1.0
    with torch.no_grad():
        return float(_LPIPS_MODEL(first, second).item())


def _revisit(video_dir: Path, metadata: Mapping[str, Any], *, max_pairs: int, use_lpips: bool, device: str) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    all_psnr: list[float] = []
    all_ssim: list[float] = []
    all_lpips: list[float] = []
    for scene in metadata.get("scenes", []):
        if not isinstance(scene, Mapping):
            continue
        scene_id = str(scene.get("scene_id") or scene.get("id") or "")
        video = video_dir / f"{scene_id}_generated.mp4"
        pairs = [pair for pair in scene.get("evaluation_pairs", []) if _pair_indices(pair) is not None][:max_pairs]
        if not scene_id or not video.is_file() or not pairs:
            continue
        fps, frame, count = _frame_reader(video)
        values = []
        for pair in pairs:
            first, second = _pair_indices(pair) or (0, 0)
            first = min(max(round(first * fps / CANONICAL_FPS), 0), count - 1)
            second = min(max(round(second * fps / CANONICAL_FPS), 0), count - 1)
            left, right = frame(first), frame(second)
            value = {"frame_a": first, "frame_b": second, "psnr": _psnr(left, right), "ssim": _ssim(left, right)}
            if use_lpips:
                value["lpips"] = _lpips(left, right, device)
                all_lpips.append(value["lpips"])
            values.append(value)
        all_psnr.extend(value["psnr"] for value in values)
        all_ssim.extend(value["ssim"] for value in values)
        row = {"scene_type": scene.get("scene_type", "unknown"), "pairs": values, "mean_psnr": mean(value["psnr"] for value in values), "mean_ssim": mean(value["ssim"] for value in values)}
        if use_lpips:
            row["mean_lpips"] = mean(value["lpips"] for value in values)
        rows[scene_id] = row
    summary = {"overall_mean_psnr": mean(all_psnr) if all_psnr else None, "overall_mean_ssim": mean(all_ssim) if all_ssim else None, "n_total_pairs": len(all_psnr)}
    if use_lpips:
        summary["overall_mean_lpips"] = mean(all_lpips) if all_lpips else None
    return {"per_scene": rows, "summary": summary}


def _temporal(video_dir: Path, output_dir: Path, prompts: Mapping[str, str], *, window_seconds: float) -> dict[str, Any]:
    import imageio.v3 as iio
    import numpy as np

    windows: dict[str, Path] = {}
    for video in _video_files(video_dir):
        fps, frame, count = _frame_reader(video)
        size = max(16, int(round(fps * window_seconds)))
        for index, start in enumerate(range(0, count, size)):
            end = min(start + size, count)
            if end - start < 16:
                continue
            name = f"w{index}_{int(start / fps)}s-{int(end / fps)}s"
            destination = output_dir / "windows" / name / video.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                iio.imwrite(destination, np.stack([frame(i) for i in range(start, end)]), fps=max(1, int(round(fps))))
            windows[name] = destination.parent
    scores = {name: _vbench_scores(path, output_dir / "vbench" / name, prompts, TEMPORAL_DIMS) for name, path in sorted(windows.items())}
    trend = {}
    for dimension in TEMPORAL_DIMS:
        values = [score[dimension] for _, score in sorted(scores.items()) if dimension in score]
        if len(values) > 1:
            trend[dimension] = {"first_window": values[0], "last_window": values[-1], "degradation": values[0] - values[-1], "per_window": values}
    return {"windows": scores, "trend": trend}


def _camera_summary(path: Path | None) -> dict[str, float]:
    if path is None or not path.is_file():
        return {}
    payload = _json(path)
    rows = payload.values() if isinstance(payload, Mapping) else []
    values: dict[str, list[float]] = {"rot_err_deg": [], "trans_err_rel": [], "cam_mc_rel": []}
    aliases = {"rot_err_deg": "RotErr", "trans_err_rel": "TransErr_rel", "cam_mc_rel": "CamMC_rel"}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        for name, key in aliases.items():
            value = _number(row.get(key))
            if value is not None:
                if name == "rot_err_deg" and str(row.get("RotErr_unit", "")).lower() == "rad":
                    value = math.degrees(value)
                values[name].append(value)
    return {key: mean(value) for key, value in values.items() if value}


def _normalise_records(payload: Any) -> dict[str, dict[str, float]]:
    records = payload.get("records", []) if isinstance(payload, Mapping) else []
    result: dict[str, dict[str, float]] = {}
    for record in records:
        if not isinstance(record, Mapping) or str(record.get("split")) not in SPLITS:
            continue
        split = str(record["split"])
        values: dict[str, float] = {}
        vbench = record.get("vbench", {})
        revisit = record.get("revisit", {})
        camera = record.get("camera", {})
        temporal = record.get("temporal", {})
        for destination, source, key in (("vbench_overall", vbench, "quality"), ("vbench_total_score", vbench, "total"), ("revisit_psnr", revisit, "psnr"), ("revisit_ssim", revisit, "ssim"), ("revisit_lpips", revisit, "lpips"), ("rot_err_deg", camera, "rot_err_deg"), ("trans_err_rel", camera, "trans_err_rel"), ("cam_mc_rel", camera, "cam_mc_rel"), ("delta_iq", temporal, "imaging_quality_drop")):
            value = _number(source.get(key)) if isinstance(source, Mapping) else None
            if value is not None:
                values[destination] = round(value * 100.0, 4) if destination == "vbench_overall" else value
        result[split] = values
    return result


def _scorecard(*, output_dir: Path, per_split: Mapping[str, Mapping[str, float]], dataset_root: Path | None, video_root: Path | None, source: str, runtime_executed: bool, coverage: Mapping[str, Any] | None = None) -> dict[str, Any]:
    metric_rows = []
    leaderboard: dict[str, float] = {}
    for split, metrics in sorted(per_split.items()):
        for metric_id, (name, higher, group) in METRIC_SPECS.items():
            value = _number(metrics.get(metric_id))
            row = {"metric_id": metric_id, "split": split, "name": name, "group": group, "higher_is_better": higher, "available": value is not None, "raw_score": value, "normalized_score": value, "source": source}
            metric_rows.append(row)
            if value is not None:
                leaderboard[f"{split}.{metric_id}"] = value
    output_dir.mkdir(parents=True, exist_ok=True)
    scorecard_path, metric_path = output_dir / "scorecard.json", output_dir / "raw_metric_table.jsonl"
    write_jsonl(metric_path, metric_rows)
    full_splits = set(per_split) == set(SPLITS)
    result = {
        "schema_version": SCORECARD_SCHEMA_VERSION,
        "run": {"status": "official_bounded" if runtime_executed else "normalized", "started_at": utc_now_iso(), "runner": "sana_wm_bench_in_tree_runner", "returncode": 0},
        "benchmark": {"benchmark_id": BENCHMARK_ID, "name": "SANA-WM 80-Scene Benchmark", "contract_only": False, "requires_upstream_runtime": False},
        "dataset": {"dataset_root": None if dataset_root is None else str(dataset_root), "generated_artifact_dir": None if video_root is None else str(video_root), "coverage": dict(coverage or {})},
        "metrics": {"leaderboard": leaderboard, "per_split": {key: dict(value) for key, value in per_split.items()}, "per_metric": metric_rows},
        "validation": {"normalizer_only": not runtime_executed, "official_runtime_executed": runtime_executed, "canonical_splits_present": full_splits, "pose_note": "Pi3 pose results are imported from <split>/eval_poses.json or --pose-results-path; distributed Pi3 inference remains a separately schedulable in-tree stage."},
        "eligibility": {"leaderboard_valid": False, "reasons": ["local scorecards are not an NVIDIA leaderboard submission", "full 80-scene videos and evaluator weights are required for a comparable run"]},
        "artifacts": {"scorecard": str(scorecard_path.resolve()), "raw_metric_table": str(metric_path.resolve())},
        "normalization_ok": bool(leaderboard), "official_benchmark_verified": False, "integration_evidence": runtime_executed and bool(leaderboard), "leaderboard_valid": False,
    }
    write_json(scorecard_path, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-id", default=BENCHMARK_ID)
    parser.add_argument("--run-official", action="store_true")
    parser.add_argument("--run-fixture", action="store_true")
    parser.add_argument("--official-results-path", type=Path)
    parser.add_argument("--dataset-root", type=Path, default=env_path("WORLDFOUNDRY_SANA_WM_BENCH_DATASET_ROOT"))
    parser.add_argument("--generated-video-dir", "--generated-artifact-dir", dest="generated_video_dir", type=Path, default=env_path("WORLDFOUNDRY_GENERATED_ARTIFACT_DIR"))
    parser.add_argument("--output-dir", type=Path, default=env_path("WORLDFOUNDRY_BENCHMARK_OUTPUT_DIR"))
    parser.add_argument("--split", default="all", choices=("all", *SPLITS))
    parser.add_argument("--metrics", default="vbench,revisit,temporal,camera")
    parser.add_argument("--pose-results-path", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-pairs", type=int, default=5)
    parser.add_argument("--revisit-lpips", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--window-seconds", type=float, default=10.0)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.benchmark_id != BENCHMARK_ID:
        raise ValueError(f"expected --benchmark-id {BENCHMARK_ID!r}")
    if args.output_dir is None:
        raise ValueError("--output-dir or WORLDFOUNDRY_BENCHMARK_OUTPUT_DIR is required")
    output_dir = args.output_dir.expanduser().resolve()
    if args.run_fixture:
        payload = {"records": [{"split": "simple_60s", "vbench": {"quality": 0.811, "total": 0.72}, "revisit": {"psnr": 24.0, "ssim": 0.83, "lpips": 0.14}, "camera": {"rot_err_deg": 3.0, "trans_err_rel": 0.12, "cam_mc_rel": 0.20}, "temporal": {"imaging_quality_drop": 0.03}}]}
        fixture = output_dir / "fixture_aggregate_results.json"
        fixture.parent.mkdir(parents=True, exist_ok=True)
        write_json(fixture, payload)
        scorecard = _scorecard(output_dir=output_dir, per_split=_normalise_records(payload), dataset_root=None, video_root=None, source="sana_wm_bench_fixture", runtime_executed=False)
    elif args.official_results_path is not None:
        scorecard = _scorecard(output_dir=output_dir, per_split=_normalise_records(_json(args.official_results_path)), dataset_root=args.dataset_root, video_root=args.generated_video_dir, source="official_result_import", runtime_executed=False)
    elif args.run_official:
        if args.dataset_root is None or args.generated_video_dir is None:
            raise ValueError("--run-official requires --dataset-root and --generated-video-dir")
        root, video_root = args.dataset_root.expanduser().resolve(), args.generated_video_dir.expanduser().resolve()
        selected = tuple(SPLITS) if args.split == "all" else (normalize_split(args.split),)
        wanted = {item.strip() for item in args.metrics.split(",") if item.strip()}
        per_split: dict[str, dict[str, float]] = {}
        coverage: dict[str, Any] = {}
        for split in selected:
            manifest, metadata = _validate_dataset(root, split, strict=args.limit is None)
            split_videos = video_root / split if (video_root / split).is_dir() else video_root
            videos = _video_files(split_videos, args.limit)
            expected = min(len(manifest), args.limit) if args.limit is not None else len(manifest)
            coverage[split] = {"expected_videos": expected, "found_videos": len(videos), "complete": len(videos) == expected}
            if not videos:
                raise FileNotFoundError(f"no *_generated.mp4 files found for {split}: {split_videos}")
            prompts = {str(row["id"]): str(row.get("prompt") or row["id"]) for row in manifest}
            values: dict[str, float] = {}
            eval_root = output_dir / "eval" / split
            if "vbench" in wanted:
                values.update(_vbench_aggregate(_vbench_scores(split_videos, eval_root / "vbench", prompts, VBENCH_DIMS)))
            if "revisit" in wanted:
                revisit = _revisit(split_videos, metadata, max_pairs=args.max_pairs, use_lpips=args.revisit_lpips, device=args.device)
                write_json(eval_root / "revisit_consistency.json", revisit)
                values.update({"revisit_psnr": revisit["summary"].get("overall_mean_psnr"), "revisit_ssim": revisit["summary"].get("overall_mean_ssim"), "revisit_lpips": revisit["summary"].get("overall_mean_lpips")})
            if "temporal" in wanted:
                temporal = _temporal(split_videos, eval_root / "temporal", prompts, window_seconds=args.window_seconds)
                write_json(eval_root / "temporal_degradation.json", temporal)
                values["delta_iq"] = _number(temporal.get("trend", {}).get("imaging_quality", {}).get("degradation"))
            if "camera" in wanted:
                pose = args.pose_results_path or split_videos / "eval_poses.json"
                values.update(_camera_summary(pose))
            per_split[split] = {key: value for key, value in values.items() if value is not None}
        scorecard = _scorecard(output_dir=output_dir, per_split=per_split, dataset_root=root, video_root=video_root, source="sana_wm_bench_in_tree_runtime", runtime_executed=True, coverage=coverage)
    else:
        raise ValueError("pass --run-official, --official-results-path, or --run-fixture")
    result = {"ok": scorecard["normalization_ok"], "benchmark_id": BENCHMARK_ID, "scorecard": scorecard["artifacts"]["scorecard"], "raw_metric_table": scorecard["artifacts"]["raw_metric_table"]}
    print(json.dumps(result, indent=2, sort_keys=True) if args.json else f"{BENCHMARK_ID}: {result['scorecard']}")
    return 0 if result["ok"] else 1
