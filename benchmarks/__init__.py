"""WorldFoundry performance benchmark harness (F0 foundation).

Reproducible, fingerprinted paired A/B measurement for operators and, later,
full inference. Results reuse ``worldfoundry.runtime.performance`` primitives
(``RuntimeFingerprint`` + ``PerformanceManifest``) so every number carries the
hardware/software identity needed to decide whether two runs are comparable.
"""

from benchmarks.harness import (
    PairedABResult,
    SampleStats,
    bench_paired,
    summarize_markdown,
    write_result,
)

__all__ = [
    "PairedABResult",
    "SampleStats",
    "bench_paired",
    "summarize_markdown",
    "write_result",
]
