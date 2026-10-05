"""Shared official-runner framework used by every ``runners/<bench>/`` entrypoint.

Layering under ``evaluation/tasks/execution/``
==============================================

- ``framework/`` — scorecard, CLI, env resolution, tabular metric extraction, result normalizers (this package)
- ``orchestration/`` — model×benchmark plans, zoo manifest CLI, suite runs
- ``runners/<bench>/`` — **one real benchmark per folder**; only bench-specific hooks

Runner tiers (under ``runners/<bench>/``)
=========================================

**Hook runner** (~50–250 lines): ``CONFIG`` + ``extract_metrics`` (+ optional discover/prepare) → ``official_runner.run_main()``

**Custom impl**: large ``*_official_impl.py`` + thin ``run_*_official_runner.py`` entry (vbench, videoscore, …)

Import convention for bench runners::

    from worldfoundry.evaluation.tasks.execution.framework import official_runner as ors
    from worldfoundry.evaluation.tasks.execution.framework.io import env_path
    from worldfoundry.evaluation.tasks.execution.framework.runner_common import (
        SCORECARD_SCHEMA_VERSION,
        VIDEO_SUFFIXES,
        resolve_env_path,
    )

**Legacy custom runners** still carry bench-specific scorecard assembly; new work
should prefer ``official_runner.run_main()`` plus ``runner_common`` helpers.
"""

from worldfoundry.evaluation.tasks.execution.framework.official_runner import (
    BenchRunnerConfig,
    RunnerHooks,
    apply_component_aggregates,
    build_common_parser,
    build_runner_config_from_contract,
    build_scorecard,
    extract_tabular_official_metrics,
    generic_extract_metrics,
    load_upstream_payload,
    metric_row,
    normalizer_only_hooks,
    run_main,
    run_official_pipeline,
)
from worldfoundry.evaluation.tasks.execution.framework.runner_common import (
    SCORECARD_SCHEMA_VERSION,
    VIDEO_SUFFIXES,
    build_import_metric_rows,
    build_video_coverage,
    iter_video_files,
    resolve_env_path,
    video_stems_in_directory,
)

__all__ = [
    "BenchRunnerConfig",
    "RunnerHooks",
    "apply_component_aggregates",
    "build_common_parser",
    "build_runner_config_from_contract",
    "build_scorecard",
    "extract_tabular_official_metrics",
    "generic_extract_metrics",
    "load_upstream_payload",
    "metric_row",
    "normalizer_only_hooks",
    "run_main",
    "run_official_pipeline",
    "SCORECARD_SCHEMA_VERSION",
    "VIDEO_SUFFIXES",
    "build_import_metric_rows",
    "build_video_coverage",
    "iter_video_files",
    "resolve_env_path",
    "video_stems_in_directory",
]
