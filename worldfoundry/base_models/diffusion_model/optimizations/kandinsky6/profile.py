"""Kandinsky-6 profiling configuration and inactive sampling context."""
from contextlib import contextmanager
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Literal
from .base import ConfigModel
ProfileMode = Literal["none", "time_mem_module"]

class ProfileConfig(ConfigModel):
    """Which profiler to attach. ``none`` leaves the pipeline uninstrumented."""

    mode: ProfileMode = "none"


@dataclass(frozen=True, slots=True)
class ModuleProfile:
    """One component over a single ``pipeline.__call__``.

    ``time`` is the sum of that component's calls, in seconds.
    Peaks are the maximum CUDA high-water mark across those calls, in bytes.
    A component that did not run is absent from the report.
    """

    time: float
    peak_allocated_mem: int
    peak_reserved_mem: int


@dataclass(frozen=True, slots=True)
class ProfileReport:
    """Totals cover the whole call, including work outside any one component."""

    modules: dict[str, ModuleProfile]
    total_time: float
    total_peak_allocated_mem: int
    total_peak_reserved_mem: int
    sr_time: float | None = None


class NoOpProfile:
    mode: ProfileMode = "none"

    def apply(self, pipe: object) -> None:
        del pipe

    def clear(self, pipe: object) -> None:
        del pipe

    @contextmanager
    def pipeline(self) -> Iterator[None]:
        yield

    def note_call(self, *, seed: int | None = None, **raw: Any) -> None:
        del seed, raw

    def note_prompt(self, prompt: str | list[str]) -> None:
        del prompt

    def note_sr(self, elapsed: float) -> None:
        del elapsed

    @property
    def report(self) -> ProfileReport:
        return ProfileReport(
            modules={},
            total_time=0.0,
            total_peak_allocated_mem=0,
            total_peak_reserved_mem=0,
        )
