"""Platform-neutral accelerator discovery."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from .base import PlatformProvider
from .providers import CpuPlatformProvider, builtin_accelerator_providers
from .types import AcceleratorDescriptor, PlatformKind


def _preference_order(
    preferred: PlatformKind | str | Sequence[PlatformKind | str] | None,
) -> tuple[PlatformKind, ...]:
    if preferred is None:
        return ()
    if isinstance(preferred, (PlatformKind, str)):
        values: Sequence[PlatformKind | str] = (preferred,)
    else:
        values = preferred
    result: list[PlatformKind] = []
    for value in values:
        kind = PlatformKind.parse(value)
        if kind not in result:
            result.append(kind)
    return tuple(result)


def detect_accelerators(
    preferred: PlatformKind | str | Sequence[PlatformKind | str] | None = None,
    providers: Iterable[PlatformProvider] | None = None,
) -> list[AcceleratorDescriptor]:
    """Detect visible accelerators and return CPU only when none are found.

    ``preferred`` controls result and probe ordering without hiding other
    visible accelerators.  Supplying ``providers`` is useful for tests and for
    callers that need a constrained in-tree probe set; it never loads code from
    another repository.
    """

    provider_list = list(builtin_accelerator_providers() if providers is None else providers)
    preference = _preference_order(preferred)
    preference_index = {kind: index for index, kind in enumerate(preference)}
    original_index = {id(provider): index for index, provider in enumerate(provider_list)}
    provider_list.sort(
        key=lambda provider: (
            preference_index.get(provider.kind, len(preference)),
            original_index[id(provider)],
        )
    )

    cpu_providers: list[PlatformProvider] = []
    devices: list[AcceleratorDescriptor] = []
    for provider in provider_list:
        if provider.kind is PlatformKind.CPU:
            cpu_providers.append(provider)
            continue
        devices.extend(provider.detect())

    if devices:
        return devices

    cpu_provider = cpu_providers[0] if cpu_providers else CpuPlatformProvider()
    return cpu_provider.detect()
