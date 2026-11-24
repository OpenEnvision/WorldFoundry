"""Benchmark execution package.

Directory layout (read this before adding files here)
---------------------------------------------------

``framework/``
    Shared official-runner infrastructure: scorecard, CLI, I/O helpers, registry.

``orchestration/``
    Model×benchmark pipeline: ``plan``, ``evaluate``, ``benchmark_runner``, suites.

``runners/<bench-id>/``
    **Video-generation benchmarks only.** One folder per bench; metric parsing and
    official-runner commands. Embodied-action closed-loop eval uses
    ``worldfoundry.evals.tasks.embodied`` (vendored harness pending).

Prefer importing from subpackages; avoid adding new root-level modules here.
"""

from __future__ import annotations

__all__: list[str] = []
