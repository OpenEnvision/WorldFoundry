"""Request materializers for simulator closed-loop rollouts."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from worldfoundry.evals.api import GenerationRequest


def _slug(value: Any, fallback: str) -> str:
    text = str(value or fallback).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return text or fallback


def materialize_libero_rollout_requests(
    suite: str,
    *,
    max_tasks: int | None,
    episodes_per_task: int,
    seed: int,
    tasks: Sequence[str | int | Mapping[str, Any]] | None = None,
) -> tuple[GenerationRequest, ...]:
    """Materialize LIBERO task x episode requests without importing LIBERO."""
    if episodes_per_task < 1:
        raise ValueError("episodes_per_task must be >= 1")
    task_entries: list[Mapping[str, Any]] = []
    if tasks:
        for index, task in enumerate(tasks):
            if isinstance(task, Mapping):
                task_id = int(task.get("task_id", task.get("id", index)))
                task_name = str(task.get("name") or task.get("task_name") or f"{suite}/task_{task_id:03d}")
                task_entries.append({"task_id": task_id, "task_name": task_name})
            else:
                task_id = int(task) if isinstance(task, int) or str(task).isdigit() else index
                task_name = str(task) if not isinstance(task, int) else f"{suite}/task_{task_id:03d}"
                task_entries.append({"task_id": task_id, "task_name": task_name})
    else:
        count = int(max_tasks or 1)
        task_entries = [{"task_id": i, "task_name": f"{suite}/task_{i:03d}"} for i in range(count)]

    if max_tasks is not None:
        task_entries = task_entries[: int(max_tasks)]

    requests: list[GenerationRequest] = []
    for task in task_entries:
        task_id = int(task["task_id"])
        task_name = str(task["task_name"])
        for episode_idx in range(int(episodes_per_task)):
            requests.append(
                GenerationRequest(
                    sample_id=f"{suite}-task{task_id:03d}-ep{episode_idx:03d}",
                    task_name=task_name,
                    inputs={
                        "suite": suite,
                        "task_id": task_id,
                        "seed": int(seed),
                        "episode_idx": episode_idx,
                    },
                    controls={"sample_controls": {"seed": int(seed), "episode_idx": episode_idx}},
                    output_schema={"rollout_metrics": {"kind": "embodied_rollout_metrics"}},
                )
            )
    return tuple(requests)


def materialize_embodied_rollout_requests(config: Mapping[str, Any]) -> tuple[GenerationRequest, ...]:
    """Materialize rollout requests from a canonical embodied benchmark config."""
    benchmark_id = str(config.get("benchmark_id") or config.get("id") or "libero")
    params = dict(config.get("benchmark_kwargs") or config.get("params") or {})
    episodes_per_task = int(config.get("episodes_per_task", params.get("episodes_per_task", 1)))
    max_tasks = config.get("max_tasks", params.get("max_tasks"))
    if max_tasks is None and params.get("num_sequences") is not None:
        max_tasks = params["num_sequences"]
    if max_tasks is None and params.get("episode_count") is not None:
        max_tasks = params["episode_count"]
    seed_value = params.get("seed", config.get("seed", 7))
    seed = 0 if seed_value is None else int(seed_value)
    tasks = config.get("tasks") or params.get("tasks")
    if benchmark_id == "libero":
        return materialize_libero_rollout_requests(
            str(params.get("suite", config.get("suite", "libero_spatial"))),
            max_tasks=None if max_tasks is None else int(max_tasks),
            episodes_per_task=episodes_per_task,
            seed=seed,
            tasks=tasks,
        )

    task_entries: list[dict[str, Any]] = []
    explicit_task_name = params.get("task_name") or config.get("task_name")
    if explicit_task_name:
        count = int(max_tasks or params.get("test_num") or 1)
        first_task_id = int(params.get("task_id", 0))
        task_entries.extend(
            {"task_id": first_task_id + offset, "task_name": str(explicit_task_name)} for offset in range(count)
        )
    elif tasks:
        for index, task in enumerate(tasks):
            if isinstance(task, Mapping):
                task_id = int(task.get("task_id", task.get("id", index)))
                task_name = str(
                    task.get("name") or task.get("task_name") or task.get("id") or f"{benchmark_id}/task_{task_id:03d}"
                )
            else:
                task_id = int(task) if isinstance(task, int) or str(task).isdigit() else index
                task_name = str(task) if not isinstance(task, int) else f"{benchmark_id}/task_{task_id:03d}"
            task_entries.append({"task_id": task_id, "task_name": task_name})
    else:
        count = int(max_tasks or 1)
        task_entries = [
            {"task_id": task_index, "task_name": f"{benchmark_id}/task_{task_index:03d}"} for task_index in range(count)
        ]
    if max_tasks is not None:
        task_entries = task_entries[: int(max_tasks)]

    requests: list[GenerationRequest] = []
    for task in task_entries:
        task_index = int(task["task_id"])
        task_name = str(task["task_name"])
        sample_prefix = _slug(config.get("id") or task_name, benchmark_id)
        for episode_idx in range(episodes_per_task):
            requests.append(
                GenerationRequest(
                    sample_id=f"{sample_prefix}-task{task_index:03d}-ep{episode_idx:03d}",
                    task_name=task_name,
                    inputs={
                        "task_id": task_index,
                        "task_name": task_name,
                        "seed": seed,
                        "episode_idx": episode_idx,
                    },
                    output_schema={"rollout_metrics": {"kind": "embodied_rollout_metrics"}},
                )
            )
    return tuple(requests)


__all__ = ["materialize_embodied_rollout_requests", "materialize_libero_rollout_requests"]
