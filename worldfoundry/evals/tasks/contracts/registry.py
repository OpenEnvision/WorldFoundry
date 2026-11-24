"""Global Registry for External Benchmark Contracts.

This module provides the central registration and resolution service for integrating
third-party evaluation benchmarks. It acts as the backbone for the WorldFoundry plugin architecture,
ensuring that custom metrics, artifacts, and runtime dependencies for external suites
(e.g., VBench, RoboTwin) can be securely registered and safely retrieved without
name collisions.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Literal


@dataclass(frozen=True)
class ExternalBenchmarkContract:
    """Rigid structural contract for a registered external evaluation benchmark.

    This class statically defines the exact signature of inputs expected and outputs
    guaranteed by third-party benchmark runners (like RoboTwin or VBench). It prevents
    data leaks and missing dependencies by acting as a strict schema validation layer.

    Attributes:
        benchmark_id: Canonical unique identifier used by the central registry.
        display_name: Human-readable name used in scorecards and leaderboard UI.
        input_keys: Explicit list of required input data/assets (e.g., 'dataset_root', 'generated_video_dir').
        output_keys: Explicit list of guaranteed output artifacts (e.g., 'scorecard', 'raw_metric_table').
        metric_ids: Sequence of official metric scalar keys this benchmark produces.
        requires_upstream_runtime: If True, indicates that the benchmark must be executed physically
            within its native Python package (e.g., requiring IsaacGym/SAPIEN simulators),
            rather than purely via text/json-level parsing.
        notes: Explanatory annotations and developer cautions.
    """

    benchmark_id: str
    display_name: str
    input_keys: tuple[str, ...]
    output_keys: tuple[str, ...]
    metric_ids: tuple[str, ...]
    requires_upstream_runtime: bool
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        """Serializes the contract definition into a standard Python mapping."""
        return asdict(self)


ContractSource = Literal["builtin", "extension"]


class ExternalBenchmarkContractRegistryError(ValueError):
    """Base error for external benchmark contract registry failures."""


class DuplicateExternalBenchmarkContractError(ExternalBenchmarkContractRegistryError):
    """Raised when a built-in benchmark contract registers a duplicate id."""


class UnknownExternalBenchmarkContractError(KeyError):
    """Raised when a benchmark contract lookup cannot be resolved."""


# Process-global singleton storing all active base benchmark contracts.
_SUPPORTED_CONTRACTS: dict[str, ExternalBenchmarkContract] = {}


def _normalise_key(value: str, field_name: str = "benchmark contract id") -> str:
    """Normalizes registry lookup keys, enforcing non-empty string types and case-folding.

    Args:
        value: The raw contract registry key to normalize.
        field_name: The name of the field being normalized, used for error formatting.

    Returns:
        The normalized, case-folded key.

    Raises:
        TypeError: If value is not a string.
        ValueError: If value is empty or only whitespace.
    """
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    value = value.strip()
    if not value:
        raise ValueError(f"{field_name} must be non-empty")
    # Normalize to lowercase to ensure case-insensitive unique registration
    return value.casefold()


def register_external_benchmark_contract(
    contract: ExternalBenchmarkContract,
    *,
    source: ContractSource,
    target: dict[str, ExternalBenchmarkContract] | None = None,
) -> ExternalBenchmarkContract:
    """Registers one benchmark contract with source-aware collision handling.

    If the source is `"builtin"`, ensures no identical system-level benchmark
    is accidentally overwritten by another internal definition.

    Args:
        contract: The contract object to register.
        source: The source category ("builtin" or "extension").
        target: The storage dictionary to register into. Defaults to process-global.

    Returns:
        The registered contract.

    Raises:
        DuplicateExternalBenchmarkContractError: If a system-level duplicate registration is detected.
    """
    if target is None:
        target = _SUPPORTED_CONTRACTS
    key = _normalise_key(contract.benchmark_id)
    # Check for name collision in registration
    if key in target:
        if source == "builtin":
            existing = target[key]
            raise DuplicateExternalBenchmarkContractError(
                f"duplicate built-in external benchmark contract {contract.benchmark_id!r}: "
                f"already registered as {existing.display_name!r}, new entry is {contract.display_name!r}"
            )
        # If it's an extension, gracefully skip or return the existing definition
        return target[key]
    target[key] = contract
    return contract


class ExternalBenchmarkContractRegistry:
    """Manages dynamic discovery, registration, and collision-handling for external benchmark definitions.

    Ensures that benchmark suites (e.g. VBench, RoboTwin) can be instantiated seamlessly
    by providing a global canonical registry that validates inputs against collision risks.
    """

    def __init__(
        self,
        contracts: Iterable[ExternalBenchmarkContract] = (),
        *,
        include_registered: bool = False,
    ) -> None:
        """Initializes a local isolated registry, optionally preloading global system-level contracts.

        Args:
            contracts: An optional iterable of contracts to register upon initialization.
            include_registered: If True, pre-loads already globally registered built-in contracts.
        """
        self._contracts: dict[str, ExternalBenchmarkContract] = {}
        if include_registered:
            # Copy all process-global contracts to this local registry
            self._contracts.update(_SUPPORTED_CONTRACTS)
        for contract in contracts:
            self.register(contract, source="builtin")

    def register(
        self,
        contract: ExternalBenchmarkContract,
        *,
        source: ContractSource,
    ) -> ExternalBenchmarkContract:
        """Registers a new external evaluation contract.

        Args:
            contract: The ExternalBenchmarkContract object to register.
            source: Source of the contract (e.g. "builtin" or "extension").

        Returns:
            The registered contract.
        """
        return register_external_benchmark_contract(contract, source=source, target=self._contracts)

    def get(self, benchmark_id: str) -> ExternalBenchmarkContract:
        """Retrieves a specific evaluation contract, returning explicit KeyErrors mapping out what is known if failed.

        Args:
            benchmark_id: Canonical identifier of the contract to look up.

        Returns:
            The corresponding ExternalBenchmarkContract.

        Raises:
            UnknownExternalBenchmarkContractError: If the benchmark ID is not in this registry.
        """
        key = _normalise_key(benchmark_id)
        if key not in self._contracts:
            known = ", ".join(sorted(self._contracts))
            raise UnknownExternalBenchmarkContractError(
                f"unknown external benchmark contract {benchmark_id!r}; known: {known}"
            )
        return self._contracts[key]

    def list(self) -> tuple[ExternalBenchmarkContract, ...]:
        """Returns all dynamically and statically registered benchmark contracts.

        Returns:
            A tuple of sorted ExternalBenchmarkContract objects.
        """
        return tuple(self._contracts[key] for key in sorted(self._contracts))


__all__ = [
    "ContractSource",
    "DuplicateExternalBenchmarkContractError",
    "ExternalBenchmarkContract",
    "ExternalBenchmarkContractRegistry",
    "ExternalBenchmarkContractRegistryError",
    "UnknownExternalBenchmarkContractError",
    "register_external_benchmark_contract",
]
