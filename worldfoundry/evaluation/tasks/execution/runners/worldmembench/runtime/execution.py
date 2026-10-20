# SPDX-FileCopyrightText: Copyright (c) 2026 ZJU-REAL/WorldMemBench contributors
# SPDX-License-Identifier: MIT

"""Evaluate video shards and merge item-level results in manifest order."""

import json
import multiprocessing
import sys
from dataclasses import asdict
from queue import Empty

from worldfoundry.evaluation.tasks.execution.runners.worldmembench.runtime.context import EvaluationContext
from worldfoundry.evaluation.tasks.execution.runners.worldmembench.runtime.errors import DataInvalidError
from worldfoundry.evaluation.tasks.execution.runners.worldmembench.runtime.results import ItemResult, ResultStatus


def parse_gpus(value: str) -> list[int]:
    parts = value.split(",")
    if any(not part.strip().isdigit() for part in parts):
        raise ValueError("--gpus requires comma-separated GPU indices, e.g. 0,1")
    ids = [int(part) for part in parts]
    if len(ids) != len(set(ids)):
        raise ValueError("--gpus must not contain duplicate GPU indices")
    return ids


def evaluate_shard(samples, metrics, device, weights_dir, config):
    context = EvaluationContext(device=device, cache_dir=weights_dir, config=config)
    items_by_metric = {metric.name: [] for metric in metrics}
    try:
        names = [metric.name for metric in metrics]
        if {
            "environment_visible_memory",
            "environment_depth_fidelity",
            "environment_scene_reconstruction",
        }.intersection(names):
            from worldfoundry.evaluation.tasks.execution.runners.worldmembench.runtime.metrics.environment.preparation import (
                prepare_artifacts,
            )

            prepare_artifacts(samples, context, names)
        if {"camera_rotation", "camera_translation"}.intersection(names):
            from worldfoundry.evaluation.tasks.execution.runners.worldmembench.runtime.metrics.camera.preparation import (
                prepare_artifacts,
            )

            prepare_artifacts(samples, context, names)
        for index, sample in enumerate(samples, 1):
            identity = {
                key: sample.get(key) for key in ("item_id", "model_id", "category", "sample_id", "condition_id")
            }
            for metric in metrics:
                try:
                    units = metric.evaluate(sample, context)
                    if not units:
                        raise ValueError("metric returned no unit results")
                    item = metric.aggregate_item(sample, units)
                    item.status = ResultStatus(item.status)
                    for key, value in identity.items():
                        setattr(item, key, value)
                    item.num_units = len(units)
                    json.dumps(asdict(item), allow_nan=False)
                except Exception as error:
                    item = ItemResult(
                        **identity,
                        status=(
                            ResultStatus.DATA_INVALID
                            if isinstance(error, DataInvalidError)
                            else ResultStatus.EVAL_FAILED
                        ),
                        error=f"{type(error).__name__}: {error}",
                    )
                if item.status in {ResultStatus.DATA_INVALID, ResultStatus.EVAL_FAILED}:
                    item.scores = {}
                    print(
                        f"[{device}] {metric.name} / {sample['item_id']}: {item.status.value}: {item.error}",
                        file=sys.stderr,
                        flush=True,
                    )
                items_by_metric[metric.name].append(item)
            context.clear_frame_cache()
            print(f"[{device}] {index}/{len(samples)} {sample['item_id']}", flush=True)
        return items_by_metric
    finally:
        context.close()


def _worker(samples, metrics, device, weights_dir, config):
    if device.startswith("cuda"):
        import torch

        torch.cuda.set_device(device)
        torch.set_num_threads(4)
    return evaluate_shard(samples, metrics, device, weights_dir, config)


def evaluate_parallel(samples, metrics, devices, weights_dir, config):
    """Spawn one independent worker per active device; never fork CUDA state."""
    active = devices[: len(samples)]
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    processes = []
    received = {}
    try:
        for index, device in enumerate(active):
            process = context.Process(
                target=_report_worker,
                args=(queue, index, samples[index :: len(active)], metrics, device, weights_dir, config),
            )
            process.start()
            processes.append(process)
        while len(received) < len(processes):
            try:
                index, result, error = queue.get(timeout=0.2)
            except Empty:
                for index, process in enumerate(processes):
                    if index not in received and process.exitcode is not None and process.exitcode != 0:
                        raise RuntimeError(f"worker {active[index]} exited with code {process.exitcode}")
                continue
            if error:
                raise RuntimeError(error)
            received[index] = result
        return [received[index] for index in range(len(processes))]
    finally:
        # Terminate only workers created by this run when any shard fails.
        if len(received) < len(processes):
            for process in processes:
                if process.is_alive():
                    process.terminate()
        for process in processes:
            process.join()
        queue.close()
        queue.join_thread()


def _report_worker(queue, index, samples, metrics, device, weights_dir, config):
    try:
        result = _worker(samples, metrics, device, weights_dir, config)
    except BaseException as error:
        queue.put((index, None, f"{type(error).__name__}: {error}"))
    else:
        queue.put((index, result, None))


def merge_shards(shards, samples, metrics):
    order = {sample["item_id"]: index for index, sample in enumerate(samples)}
    items = {metric.name: [] for metric in metrics}
    for shard_items in shards:
        for metric in metrics:
            items[metric.name].extend(shard_items[metric.name])
    for metric in metrics:
        rows = items[metric.name]
        if len(rows) != len(samples) or {row.item_id for row in rows} != set(order):
            raise RuntimeError(f"incomplete or duplicate shard results for {metric.name}")
        rows.sort(key=lambda row: order[row.item_id])
    return items
