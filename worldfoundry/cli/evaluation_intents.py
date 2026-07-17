"""Small CLI surface for prepared evaluation intents."""

from __future__ import annotations

import argparse
from pathlib import Path

from worldfoundry.evaluation.tasks.execution.orchestration.service import (
    GenerateAndScoreIntent,
    ReproduceIntent,
    ScoreArtifactsIntent,
    ScoreResultsIntent,
    execute_prepared_evaluation,
    prepare_evaluation,
)
from worldfoundry.evaluation.utils import BENCHMARK_ZOO_DIR, MODEL_ZOO_DIR, TMP_ROOT

from .utils import json_dump, parse_key_value_mapping


def _print_prepared(prepared, *, as_json: bool) -> None:
    payload = prepared.to_dict()
    if as_json:
        json_dump(payload)
        return
    print(f"intent: {payload['intent_kind']}")
    print(f"classification: {payload['classification']}")
    print(f"ready: {payload['ready']}")
    for issue in payload["issues"]:
        print(f"{issue['severity']}: {issue['code']}: {issue['message']}")


def _finish(prepared, args: argparse.Namespace) -> int:
    if args.plan_only or not prepared.ready:
        _print_prepared(prepared, as_json=args.json)
        return 0 if prepared.ready else 1
    result = execute_prepared_evaluation(prepared)
    payload = result.to_dict()
    if args.json:
        json_dump(payload)
    else:
        print(f"status: {payload['status']}")
        print(f"output_dir: {payload['output_dir']}")
    return int(payload.get("exit_code", 0))


def _handle_score(args: argparse.Namespace) -> int:
    if args.benchmark:
        if args.artifacts is None:
            raise ValueError("score --benchmark requires --artifacts")
        intent = ScoreArtifactsIntent(
            output_dir=args.output_dir,
            benchmark_id=args.benchmark,
            artifact_dir=args.artifacts,
            dataset_id=args.dataset_id,
            benchmark_env=parse_key_value_mapping(args.env),
            benchmark_parameters=parse_key_value_mapping(args.parameter),
            benchmark_mode=args.mode,
            benchmark_manifest_dir=args.benchmark_manifest_dir,
            leaderboard_candidate=args.leaderboard_candidate,
            run_id=args.run_id,
        )
    else:
        if args.results is None or not args.metric:
            raise ValueError("score without --benchmark requires --results and at least one --metric")
        intent = ScoreResultsIntent(
            output_dir=args.output_dir,
            results_path=args.results,
            requests_path=args.requests,
            metrics=tuple(args.metric),
            benchmark_id=args.metric_context,
            dataset_id=args.dataset_id,
            required_artifacts=tuple(args.required_artifact or ()),
            run_id=args.run_id,
        )
    return _finish(prepare_evaluation(intent), args)


def _handle_reproduce(args: argparse.Namespace) -> int:
    prepared = prepare_evaluation(
        ReproduceIntent(
            output_dir=args.output_dir,
            recipe_path=args.recipe,
            profile_id=args.profile,
            benchmark_id=args.benchmark,
        )
    )
    return _finish(prepared, args)


def _handle_generate_score(args: argparse.Namespace) -> int:
    intent = GenerateAndScoreIntent(
        output_dir=args.output_dir,
        model_id=args.model,
        model_variant_id=args.variant,
        dataset_manifest=args.dataset_manifest,
        benchmark_id=args.benchmark,
        metrics=tuple(args.metric),
        task_name=args.task_name,
        input_keys=tuple(args.input_key or ()),
        output_keys=tuple(args.output_key or ("generated_video",)),
        required_artifacts=tuple(args.required_artifact or ()),
        generation_defaults=parse_key_value_mapping(args.generation_default),
        model_parameters=parse_key_value_mapping(args.model_parameter),
        model_runtime=parse_key_value_mapping(args.model_runtime),
        model_manifest_dir=args.model_manifest_dir,
        num_samples=args.num_samples,
        generation_cache_dir=args.generation_cache_dir,
        generation_cache_mode=args.generation_cache_mode,
        run_id=args.run_id,
    )
    return _finish(prepare_evaluation(intent), args)


def register_evaluation_intent_subparsers(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register user-oriented scoring and reproduction commands."""

    score = subparsers.add_parser(
        "score",
        help="Score user artifacts with a benchmark suite or executable metric ids",
    )
    score.add_argument("--benchmark", help="Benchmark id whose complete metric protocol should run.")
    score.add_argument("--artifacts", type=Path, help="Directory containing user-provided generated artifacts.")
    score.add_argument("--results", type=Path, help="WorldFoundry JSON/JSONL generation results.")
    score.add_argument("--requests", type=Path, help="Optional matching WorldFoundry requests JSON/JSONL.")
    score.add_argument("--metric", action="append", help="Executable existing-results metric id; repeatable.")
    score.add_argument("--metric-context", help="Optional benchmark id for benchmark-specific in-tree metrics.")
    score.add_argument("--required-artifact", action="append")
    score.add_argument("--dataset-id")
    score.add_argument("--benchmark-manifest-dir", type=Path, default=BENCHMARK_ZOO_DIR)
    score.add_argument("--mode", choices=("official-run", "official-validation", "normalizer"), default="official-run")
    score.add_argument("--env", action="append", default=None, metavar="KEY=VALUE")
    score.add_argument("--parameter", action="append", default=None, metavar="KEY=VALUE")
    score.add_argument("--leaderboard-candidate", action="store_true")
    score.add_argument("--run-id")
    score.add_argument("--output-dir", type=Path, default=TMP_ROOT / "score")
    score.add_argument("--plan-only", action="store_true")
    score.add_argument("--json", action="store_true")
    score.set_defaults(func=_handle_score)

    generate_score = subparsers.add_parser(
        "generate-score",
        help="Run an in-tree model on a dataset manifest, then score its results",
    )
    generate_score.add_argument("--model", required=True)
    generate_score.add_argument("--variant")
    generate_score.add_argument("--dataset-manifest", type=Path, required=True)
    generate_score.add_argument("--metric", action="append", required=True)
    generate_score.add_argument("--benchmark", help="Optional benchmark context for benchmark-specific metrics.")
    generate_score.add_argument("--task-name", default="custom-dataset")
    generate_score.add_argument("--input-key", action="append")
    generate_score.add_argument("--output-key", action="append")
    generate_score.add_argument("--required-artifact", action="append")
    generate_score.add_argument("--generation-default", action="append", metavar="KEY=VALUE")
    generate_score.add_argument("--model-parameter", action="append", metavar="KEY=VALUE")
    generate_score.add_argument("--model-runtime", action="append", metavar="KEY=VALUE")
    generate_score.add_argument("--model-manifest-dir", type=Path, default=MODEL_ZOO_DIR)
    generate_score.add_argument("--num-samples", type=int)
    generate_score.add_argument("--generation-cache-dir", type=Path)
    generate_score.add_argument(
        "--generation-cache-mode",
        choices=("off", "read", "write", "read-write", "refresh"),
        default="read-write",
    )
    generate_score.add_argument("--run-id")
    generate_score.add_argument("--output-dir", type=Path, default=TMP_ROOT / "generate-score")
    generate_score.add_argument("--plan-only", action="store_true")
    generate_score.add_argument("--json", action="store_true")
    generate_score.set_defaults(func=_handle_generate_score)

    reproduce = subparsers.add_parser(
        "reproduce",
        help="Run a checked-in profile or a custom model x benchmark recipe",
    )
    reproduction_source = reproduce.add_mutually_exclusive_group(required=True)
    reproduction_source.add_argument("--profile", help="Checked-in reproduction profile id.")
    reproduction_source.add_argument("--benchmark", help="Benchmark id whose default reproduction profile should run.")
    reproduction_source.add_argument("--recipe", type=Path, help="Custom reproduction recipe YAML.")
    reproduce.add_argument("--output-dir", type=Path, default=TMP_ROOT / "reproduce")
    reproduce.add_argument("--plan-only", action="store_true")
    reproduce.add_argument("--json", action="store_true")
    reproduce.set_defaults(func=_handle_reproduce)


__all__ = ["register_evaluation_intent_subparsers"]
