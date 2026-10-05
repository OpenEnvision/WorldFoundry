"""Catalog-backed scoring hook lookup.

``in_tree`` and ``physics_contract`` membership and module paths come from
catalog YAML. Engine implementations stay in the per-bench hook modules.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Mapping

from worldfoundry.evaluation.tasks.catalog.benchmark_id import normalize_benchmark_id
from worldfoundry.evaluation.tasks.catalog.dispatch import parse_scoring_hook
from worldfoundry.evaluation.tasks.execution.framework.lazy_registry import (
    load_module_attr,
    lookup_module_attr,
    supported_ids,
)


@lru_cache(maxsize=1)
def _in_tree_modules() -> dict[str, str]:
    from worldfoundry.evaluation.tasks.catalog.dispatch import scoring_hook_table

    table: dict[str, str] = {}
    for benchmark_id, hook in scoring_hook_table("in_tree").items():
        module, _, _attr = hook.rpartition(":")
        table[benchmark_id] = module or hook
    return table


def get_in_tree_benchmark_config(benchmark_id: str) -> Mapping[str, object]:
    return lookup_module_attr(
        _in_tree_modules(),
        benchmark_id,
        attr="IN_TREE_CONFIG",
        kind="in-tree benchmark",
    )


def supported_in_tree_benchmark_ids() -> tuple[str, ...]:
    return supported_ids(_in_tree_modules())


@lru_cache(maxsize=1)
def target_benchmark_metrics() -> dict[str, tuple[str, ...]]:
    return {
        benchmark_id: tuple(get_in_tree_benchmark_config(benchmark_id)["metric_ids"])  # type: ignore[arg-type]
        for benchmark_id in supported_in_tree_benchmark_ids()
    }


def _contract_evaluators() -> dict[str, tuple[str, str]]:
    from worldfoundry.evaluation.tasks.catalog.dispatch import scoring_hook_table

    table: dict[str, tuple[str, str]] = {}
    for benchmark_id, hook in scoring_hook_table("physics_contract").items():
        table[benchmark_id] = parse_scoring_hook(hook)
    return table


def has_benchmark_contract_evaluator(benchmark_id: str) -> bool:
    return normalize_benchmark_id(benchmark_id) in _contract_evaluators()


def contract_evaluator_kind(benchmark_id: str) -> str:
    module_name, _ = _contract_evaluators()[normalize_benchmark_id(benchmark_id)]
    return load_module_attr(module_name, "EVALUATOR_KIND")


def write_benchmark_contract_evaluation(*, benchmark_id: str, **kwargs: Any) -> dict[str, Any]:
    key = normalize_benchmark_id(benchmark_id)
    if key not in _contract_evaluators():
        raise KeyError(f"no in-tree contract evaluator registered for {benchmark_id!r}")
    writer = lookup_module_attr(_contract_evaluators(), key, kind="contract evaluator")
    return writer(**kwargs)
