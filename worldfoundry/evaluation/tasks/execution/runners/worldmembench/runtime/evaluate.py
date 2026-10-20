# SPDX-FileCopyrightText: Copyright (c) 2026 ZJU-REAL/WorldMemBench contributors
# SPDX-License-Identifier: MIT

"""Evaluation orchestration with per-case errors and complete result reporting."""

from datetime import datetime
from pathlib import Path

from worldfoundry.evaluation.tasks.execution.runners.worldmembench.runtime.context import EvaluationContext
from worldfoundry.evaluation.tasks.execution.runners.worldmembench.runtime.data import load_samples, write_manifest
from worldfoundry.evaluation.tasks.execution.runners.worldmembench.runtime.execution import (
    evaluate_parallel,
    evaluate_shard,
    merge_shards,
    parse_gpus,
)
from worldfoundry.evaluation.tasks.execution.runners.worldmembench.runtime.metrics import MetricRegistry, registry
from worldfoundry.evaluation.tasks.execution.runners.worldmembench.runtime.paths import DEFAULT_WEIGHTS_DIR
from worldfoundry.evaluation.tasks.execution.runners.worldmembench.runtime.results import ResultStatus, ResultWriter


def evaluate(
    source: str | Path,
    *,
    metrics: list[str] | None = None,
    models: list[str] | None = None,
    categories: list[str] | None = None,
    trajectories: list[str] | None = None,
    cases: list[str] | None = None,
    expected_revisits: int | None = None,
    output: str | Path | None = None,
    device: str = "cuda",
    gpus: str | None = None,
    weights_dir: str | Path = DEFAULT_WEIGHTS_DIR,
    config: dict | None = None,
    metric_registry: MetricRegistry = registry,
    command: list[str] | None = None,
) -> tuple[Path, bool]:
    names = metric_registry.names() if metrics is None else metrics
    if not names:
        raise ValueError("No metrics selected. Choose an available WorldMemBench metric.")
    if len(names) != len(set(names)):
        raise ValueError("each metric may be selected only once")
    selected = [metric_registry.get(name) for name in names]
    gpu_ids = parse_gpus(gpus) if gpus is not None else None
    if gpu_ids is not None and device != "cuda":
        raise ValueError("use --device for one device or --gpus for GPU selection, not both")
    if gpu_ids is not None:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("--gpus requires available CUDA devices")
        if max(gpu_ids) >= torch.cuda.device_count():
            raise ValueError(f"GPU index out of range; {torch.cuda.device_count()} device(s) visible")
    samples = load_samples(source, models=models, categories=categories, trajectories=trajectories, cases=cases)
    output = Path(output) if output is not None else Path("eval_results") / datetime.now().strftime("%Y%m%d-%H%M%S")
    if output.exists():
        raise FileExistsError(f"output already exists; choose another --output: {output}")
    writer = ResultWriter(output)
    writer.write_command(command)
    if {"human_identity", "human_appearance"}.intersection(names):
        from worldfoundry.evaluation.tasks.execution.runners.worldmembench.runtime.metrics.human.preparation import (
            prepare_for_evaluation,
        )

        preparation_context = EvaluationContext(
            device=f"cuda:{gpu_ids[0]}" if gpu_ids is not None else device,
            cache_dir=weights_dir,
            config=config,
        )
        try:
            prepare_for_evaluation(
                samples,
                preparation_context,
                expected_revisits=expected_revisits,
                weight_metric="human_identity" if "human_identity" in names else "human_appearance",
            )
        finally:
            preparation_context.close()
    if {"object_appearance", "object_geometry"}.intersection(names):
        from worldfoundry.evaluation.tasks.execution.runners.worldmembench.runtime.metrics.object.preparation import (
            prepare_for_evaluation,
        )

        prepare_for_evaluation(samples, expected_revisits=expected_revisits)
    if {"environment_visible_memory", "environment_depth_fidelity", "environment_scene_reconstruction"}.intersection(
        names
    ):
        from worldfoundry.evaluation.tasks.execution.runners.worldmembench.runtime.metrics.environment.preparation import (
            prepare_for_evaluation,
        )

        prepare_for_evaluation(samples)
    if "dynamic_memory" in names:
        from worldfoundry.evaluation.tasks.execution.runners.worldmembench.runtime.metrics.dynamic.preparation import (
            prepare_for_evaluation,
        )

        prepare_for_evaluation(samples)
    if {"camera_rotation", "camera_translation"}.intersection(names):
        from worldfoundry.evaluation.tasks.execution.runners.worldmembench.runtime.metrics.camera.preparation import (
            prepare_for_evaluation,
        )

        prepare_for_evaluation(samples, required=set(names) <= {"camera_rotation", "camera_translation"})
    write_manifest(samples, output / "manifest.jsonl")
    if gpu_ids is None:
        shards = [evaluate_shard(samples, selected, device, weights_dir, config)]
    else:
        devices = [f"cuda:{index}" for index in gpu_ids]
        print(f"GPU devices: {', '.join(devices)}; one video shard per active GPU.", flush=True)
        shards = evaluate_parallel(samples, selected, devices, weights_dir, config)
    item_results = merge_shards(shards, samples, selected)
    success = True
    for metric in selected:
        items = item_results[metric.name]
        if any(item.status in {ResultStatus.DATA_INVALID, ResultStatus.EVAL_FAILED} for item in items):
            success = False
        summary = metric.aggregate_summary(items)
        writer.write_metric(metric.name, items, summary)
    return output, success
