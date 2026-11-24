"""External benchmark contracts and their registry."""

from __future__ import annotations

from .external import (
    ExternalBenchmarkContract,
    ExternalBenchmarkContractRegistry,
    contract_from_catalog_entry,
    get_external_benchmark_contract,
    list_external_benchmark_contracts,
    register_external_benchmark_contract,
)

__all__ = [
    "ExternalBenchmarkContract",
    "ExternalBenchmarkContractRegistry",
    "contract_from_catalog_entry",
    "get_external_benchmark_contract",
    "list_external_benchmark_contracts",
    "register_external_benchmark_contract",
]
