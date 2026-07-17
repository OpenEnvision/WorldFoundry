#!/usr/bin/env python3
"""Aggregate one to four official Physics-IQ raw-metric runs."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

from worldfoundry.evaluation.tasks.execution.runners.physics_iq.physics_iq_metrics import (
    load_official_results,
    metric_key_map,
)
from worldfoundry.evaluation.tasks.execution.runners.physics_iq.protocols import resolve_protocol


def aggregate_runs(paths: list[Path], *, benchmark_id: str, protocol: str | None = None) -> dict:
    """Return official metric mean/std values in the upstream 0–100 presentation unit."""

    if not 1 <= len(paths) <= 4:
        raise ValueError("Physics-IQ aggregation accepts between one and four runs.")
    spec = resolve_protocol(benchmark_id=benchmark_id, protocol=protocol)
    run_scores = [load_official_results(path, spec)[0] for path in paths]
    metrics = {}
    for metric_id, score_key in metric_key_map(spec).items():
        values = [float(scores[score_key]) * 100.0 for scores in run_scores]
        metrics[metric_id] = {
            "mean": statistics.fmean(values),
            "std": statistics.stdev(values) if len(values) > 1 else None,
            "values": values,
        }
    return {
        "benchmark_id": spec.benchmark_id,
        "protocol": spec.protocol,
        "run_count": len(paths),
        "source_paths": [str(path.resolve()) for path in paths],
        "unit": "percent",
        "metrics": metrics,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--benchmark-id", default="physics-iq-verified")
    parser.add_argument("--protocol", choices=("original", "verified"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = aggregate_runs(args.results, benchmark_id=args.benchmark_id, protocol=args.protocol)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix.lower() == ".csv":
        with args.output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=("metric_id", "mean", "std"))
            writer.writeheader()
            for metric_id, values in payload["metrics"].items():
                writer.writerow({"metric_id": metric_id, "mean": values["mean"], "std": values["std"]})
    else:
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
