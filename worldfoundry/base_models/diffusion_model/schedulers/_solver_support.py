"""Small native compatibility primitives for stateful diffusion solvers.

These types intentionally cover only the inference-time surface used by the
canonical WorldFoundry schedulers.  They keep numerical code independent of
Diffusers without recreating its training, serialization, or Hub APIs.

:class:`SchedulerConfig` is deepcopy-safe so the multi-modal runner can copy
per-modality schedulers.  :func:`randn_tensor` supports one generator per
batch item.  A generator-list length mismatch raises :exc:`ValueError`.
"""

from __future__ import annotations

import inspect
import warnings
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, TypeVar

import torch
from torch import Tensor


class SchedulerConfig(Mapping[str, Any]):
    """Attribute-accessible inference configuration owned by a scheduler."""

    def __init__(self, **values: Any) -> None:
        self._values = dict(values)

    def __getattr__(self, name: str) -> Any:
        # ``copy.deepcopy`` allocates an uninitialized instance before
        # restoring ``__dict__`` and probes it for ``__setstate__``.  Looking
        # up ``self._values`` through this same method at that point recurses
        # indefinitely, which breaks the shared multi-modal runner's
        # per-modality scheduler copies.
        values = self.__dict__.get("_values")
        if values is None:
            raise AttributeError(name)
        try:
            return values[name]
        except KeyError as error:
            raise AttributeError(name) from error

    def __getitem__(self, key: str) -> Any:
        return self._values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def with_updates(self, **values: Any) -> SchedulerConfig:
        updated = dict(self._values)
        updated.update(values)
        return type(self)(**updated)

    def to_dict(self) -> dict[str, Any]:
        """Return a shallow copy suitable for logging or recipe inspection."""

        return dict(self._values)


@dataclass(frozen=True, slots=True)
class SchedulerOutput:
    """The native return object shared by stateful numerical schedulers."""

    prev_sample: Tensor

    def __getitem__(self, key: int | str) -> Tensor:
        if key in (0, "prev_sample"):
            return self.prev_sample
        raise IndexError(key) if isinstance(key, int) else KeyError(key)

    def to_tuple(self) -> tuple[Tensor]:
        return (self.prev_sample,)


class NativeSchedulerMixin:
    """Minimal lifecycle shared by the legacy-compatible native schedulers."""

    config: SchedulerConfig

    def register_to_config(self, **values: Any) -> None:
        current = getattr(self, "config", SchedulerConfig())
        self.config = current.with_updates(**values)


_Init = TypeVar("_Init", bound=Callable[..., None])


def register_to_config(init: _Init) -> _Init:
    """Capture constructor values before running a scheduler initializer."""

    signature = inspect.signature(init)

    @wraps(init)
    def wrapped(self: NativeSchedulerMixin, *args: Any, **kwargs: Any) -> None:
        bound = signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        values = {
            name: list(value) if isinstance(value, list) else value
            for name, value in bound.arguments.items()
            if name != "self"
        }
        self.config = SchedulerConfig(**values)
        init(self, *args, **kwargs)

    return wrapped  # type: ignore[return-value]


def deprecate(name: str, version: str, message: str) -> None:
    """Emit a lightweight compatibility warning for deprecated solver inputs."""

    warnings.warn(
        f"{name} is deprecated and will be removed in {version}: {message}",
        FutureWarning,
        stacklevel=3,
    )


def randn_tensor(
    shape: tuple[int, ...] | torch.Size,
    *,
    generator: torch.Generator | list[torch.Generator] | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Create random noise, including one-generator-per-batch semantics."""

    shape = tuple(shape)
    if isinstance(generator, list):
        if not shape:
            raise ValueError("a generator list requires a batched tensor shape")
        if len(generator) != shape[0]:
            raise ValueError(f"generator list length must match the batch dimension: {len(generator)} != {shape[0]}")
        sample_shape = (1, *shape[1:])
        return torch.cat(
            [
                torch.randn(
                    sample_shape,
                    generator=batch_generator,
                    device=device,
                    dtype=dtype,
                )
                for batch_generator in generator
            ],
            dim=0,
        )
    return torch.randn(shape, generator=generator, device=device, dtype=dtype)


__all__ = [
    "NativeSchedulerMixin",
    "SchedulerConfig",
    "SchedulerOutput",
    "deprecate",
    "randn_tensor",
    "register_to_config",
]
