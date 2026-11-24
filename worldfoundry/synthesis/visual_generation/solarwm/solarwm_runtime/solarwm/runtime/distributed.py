"""Small fail-closed checks around model-family distributed process groups."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from typing import Any, TypeVar

from ..config.loader import canonical_json
from ..errors import BackendContractError

T = TypeVar("T")


def _collective_coordinates(
    dist: Any | None,
    *,
    rank: int | None,
    world_size: int | None,
) -> tuple[int, int, bool]:
    """Resolve one raw-world collective without importing Torch.

    Callers pass ``torch.distributed`` in heavy runtimes and a small fake in
    CPU tests.  An explicitly multi-rank call may never silently degrade to a
    local check when the process group is unavailable.
    """

    initialized = bool(dist is not None and dist.is_initialized())
    resolved_rank = int(dist.get_rank()) if rank is None and initialized else int(0 if rank is None else rank)
    resolved_world = (
        int(dist.get_world_size())
        if world_size is None and initialized
        else int(1 if world_size is None else world_size)
    )
    if resolved_world < 1 or not 0 <= resolved_rank < resolved_world:
        raise BackendContractError("collective failure propagation received an invalid rank/world_size")
    if resolved_world > 1 and not initialized:
        raise BackendContractError("multi-rank failure propagation requires initialized torch.distributed")
    if initialized:
        actual_rank = int(dist.get_rank())
        actual_world = int(dist.get_world_size())
        if (resolved_rank, resolved_world) != (actual_rank, actual_world):
            raise BackendContractError("collective failure propagation rank/world_size differs from the process group")
    return resolved_rank, resolved_world, initialized


def propagate_collective_error(
    local_error: str | None,
    *,
    dist: Any | None,
    label: str,
    rank: int | None = None,
    world_size: int | None = None,
    error_type: type[Exception] = BackendContractError,
) -> None:
    """Raise the same rank-ordered failure on every raw-world peer.

    Only strings cross the process-group boundary.  Exception instances are
    deliberately never pickled or broadcast.  Every peer must call this at
    the same phase boundary.
    """

    resolved_rank, resolved_world, initialized = _collective_coordinates(
        dist,
        rank=rank,
        world_size=world_size,
    )
    message = str(local_error or "")
    local = (resolved_rank, message)
    gathered: list[Any] = [local]
    if initialized and resolved_world > 1:
        gathered = [None] * resolved_world
        dist.all_gather_object(gathered, local)

    normalized: list[tuple[int, str]] = []
    for item in gathered:
        if not isinstance(item, (tuple, list)) or len(item) != 2 or isinstance(item[0], bool):
            raise error_type(f"{label} failure exchange returned a malformed rank record")
        try:
            peer_rank = int(item[0])
        except (TypeError, ValueError) as exc:
            raise error_type(f"{label} failure exchange returned a malformed rank value") from exc
        peer_error = str(item[1] or "")
        if not 0 <= peer_rank < resolved_world:
            raise error_type(f"{label} failure exchange returned an out-of-range rank")
        normalized.append((peer_rank, peer_error))
    if len(normalized) != resolved_world or {item[0] for item in normalized} != set(range(resolved_world)):
        raise error_type(f"{label} failure exchange did not cover every raw rank")
    failures = [(peer_rank, error) for peer_rank, error in sorted(normalized) if error]
    if failures:
        detail = " | ".join(f"rank {peer_rank}: {error}" for peer_rank, error in failures)
        raise error_type(f"{label} failed collectively: {detail}")


def collective_call(
    call: Callable[[], T],
    *,
    dist: Any | None,
    label: str,
    rank: int | None = None,
    world_size: int | None = None,
    error_type: type[Exception] = BackendContractError,
) -> T:
    """Run rank-local work, then exchange errors before any later collective."""

    result: T | None = None
    local_exception: Exception | None = None
    local_error = ""
    try:
        result = call()
    except Exception as exc:
        local_exception = exc
        local_error = f"{type(exc).__name__}: {exc}"
    try:
        propagate_collective_error(
            local_error,
            dist=dist,
            label=label,
            rank=rank,
            world_size=world_size,
            error_type=error_type,
        )
    except Exception as exc:
        if local_exception is not None:
            raise exc from local_exception
        raise
    return result  # type: ignore[return-value]


def identity_fingerprint(value: Mapping[str, Any]) -> str:
    return hashlib.blake2s(canonical_json(value)).hexdigest()


def assert_peer_fingerprints(values: Sequence[str]) -> str:
    if not values or any(not value for value in values):
        raise BackendContractError("SP peer fingerprints are missing")
    if len(set(values)) != 1:
        raise BackendContractError(f"SP peers loaded different sample/RNG identities: {values}")
    return values[0]


def gather_and_assert_sp_identity(value: Mapping[str, Any], *, sp_size: int, group: Any = None) -> str:
    """Collect identity hashes in the already-created SP group and compare."""

    local = identity_fingerprint(value)
    if sp_size == 1:
        return local
    try:
        import torch.distributed as dist
    except ImportError as exc:
        raise BackendContractError("torch is required for sequence parallelism") from exc
    if not dist.is_available() or not dist.is_initialized():
        raise BackendContractError("SP identity check requires initialized torch.distributed")
    peers: list[str | None] = [None] * sp_size
    dist.all_gather_object(peers, local, group=group)
    return assert_peer_fingerprints([str(value or "") for value in peers])
