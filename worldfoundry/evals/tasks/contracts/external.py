"""Benchmark contracts built from the ``contract`` block of each benchmark catalog entry.

A contract states what an external benchmark consumes (``input_keys``), what it produces
(``output_keys``), which metrics it reports, and whether it needs its upstream runtime. The
catalog entry supplies the display name and, unless the block overrides ``metric_ids``, the
metric list. Catalog runner targets name a contract by benchmark id, for example
``worldfoundry.evals.tasks.contracts.external:vbench``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .registry import (
    ExternalBenchmarkContract,
    ExternalBenchmarkContractRegistry,
    UnknownExternalBenchmarkContractError,
    register_external_benchmark_contract,
)


def _str_tuple(value: Any) -> tuple[str, ...]:
    return tuple(str(item) for item in value or ())


def contract_from_catalog_entry(entry: Mapping[str, Any]) -> ExternalBenchmarkContract | None:
    """Build the contract declared by one raw catalog entry, or ``None`` when it declares none."""
    block = entry.get("contract")
    if not isinstance(block, Mapping):
        return None
    benchmark_id = str(entry.get("id") or entry.get("benchmark_id"))
    metric_ids = block.get("metric_ids")
    if metric_ids is None:
        metric_ids = [
            metric.get("id") or metric.get("metric_id")
            for metric in entry.get("metrics") or ()
            if isinstance(metric, Mapping)
        ]
    return ExternalBenchmarkContract(
        benchmark_id=benchmark_id,
        display_name=str(entry.get("name") or benchmark_id),
        input_keys=_str_tuple(block.get("input_keys")),
        output_keys=_str_tuple(block.get("output_keys")),
        metric_ids=_str_tuple(metric_ids),
        requires_upstream_runtime=bool(block.get("requires_upstream_runtime", True)),
        notes=_str_tuple(block.get("notes")),
    )


def _catalog_contracts() -> tuple[ExternalBenchmarkContract, ...]:
    from worldfoundry.evals.tasks.catalog.benchmark_catalog import readable_catalog_shards

    contracts = (contract_from_catalog_entry(entry) for shard in readable_catalog_shards() for entry in shard.entries)
    return tuple(contract for contract in contracts if contract is not None)


for _contract in _catalog_contracts():
    register_external_benchmark_contract(_contract, source="builtin")

_CONTRACT_REGISTRY = ExternalBenchmarkContractRegistry(include_registered=True)
del _contract


def get_external_benchmark_contract(benchmark_id: str) -> ExternalBenchmarkContract:
    """Return the contract registered for *benchmark_id* or one of its aliases."""
    return _CONTRACT_REGISTRY.get(benchmark_id)


def list_external_benchmark_contracts() -> tuple[ExternalBenchmarkContract, ...]:
    """Return every registered benchmark contract, sorted by benchmark id."""
    return _CONTRACT_REGISTRY.list()


def __getattr__(name: str) -> ExternalBenchmarkContract:
    """Resolve ``contracts.external:<benchmark-id>`` catalog runner targets."""
    try:
        return _CONTRACT_REGISTRY.get(name)
    except UnknownExternalBenchmarkContractError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None


__all__ = [
    "ExternalBenchmarkContract",
    "ExternalBenchmarkContractRegistry",
    "contract_from_catalog_entry",
    "get_external_benchmark_contract",
    "list_external_benchmark_contracts",
    "register_external_benchmark_contract",
]
