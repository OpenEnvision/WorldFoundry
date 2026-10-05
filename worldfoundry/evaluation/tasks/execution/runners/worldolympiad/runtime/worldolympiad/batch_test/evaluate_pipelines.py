#!/usr/bin/env python3
"""Unified batch evaluation entry point for generated pipeline outputs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    from .make_manifest import PIPELINE_BY_ALIAS, sanitize_filename_part
except ImportError:
    from make_manifest import PIPELINE_BY_ALIAS, sanitize_filename_part


DEFAULT_DOMAINS = ("general", "gaming", "embodied")
DEFAULT_PIPELINES = (
    "cosmos-predict",
    "hunyuan-gamecraft",
    "hunyuan-worldplay",
    "lingbot-world",
    "longlive",
    "matrix-game2",
    "rolling-forcing",
    "wow",
    "yume1p5",
)


def split_values(values: list[str] | None, default: tuple[str, ...]) -> list[str]:
    if not values:
        return list(default)
    result: list[str] = []
    for value in values:
        for part in value.split(","):
            stripped = part.strip()
            if stripped:
                result.append(stripped)
    return result


def parse_skip_pairs(values: list[str] | None) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for value in split_values(values, ()):
        if ":" not in value:
            raise ValueError(f"Invalid --skip-pair {value!r}; expected DOMAIN:PIPELINE")
        domain, pipeline = value.split(":", maxsplit=1)
        pairs.add((domain.strip(), resolve_pipeline_name(pipeline.strip())))
    return pairs


def resolve_pipeline_name(value: str) -> str:
    try:
        return PIPELINE_BY_ALIAS[value].name
    except KeyError as exc:
        valid = ", ".join(sorted(PIPELINE_BY_ALIAS))
        raise ValueError(f"Unknown pipeline {value!r}. Valid aliases: {valid}") from exc


def pipeline_short_name(pipeline_name: str) -> str:
    return sanitize_filename_part(PIPELINE_BY_ALIAS[pipeline_name].output_prefix)


def is_done(item: dict[str, Any]) -> bool:
    output = item.get("output")
    if not output:
        return False
    path = Path(str(output))
    return path.exists() and path.stat().st_size > 0


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            row["_line_number"] = line_number
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            clean_row = {key: value for key, value in row.items() if not key.startswith("_")}
            handle.write(json.dumps(clean_row, ensure_ascii=False) + "\n")


def write_progress_summary(manifest_path: Path, summary_path: Path) -> tuple[int, int]:
    items = load_jsonl(manifest_path)
    rows: list[dict[str, Any]] = []
    done = 0
    for item in items:
        item_done = is_done(item)
        done += int(item_done)
        rows.append(
            {
                "id": item.get("id") or item.get("_line_number"),
                "status": "ok" if item_done else "pending",
                "exit_code": 0 if item_done else None,
                "output": item.get("output"),
            }
        )
    write_jsonl(summary_path, rows)
    return done, len(items)


def build_selected_manifest(
    *,
    raw_manifest: Path,
    selected_manifest: Path,
    pending_only: bool,
    force: bool,
    limit: int | None,
) -> tuple[int, int]:
    items = load_jsonl(raw_manifest)
    selected: list[dict[str, Any]] = []
    for item in items:
        if pending_only and not force and is_done(item):
            continue
        selected.append(item)
        if limit is not None and len(selected) >= limit:
            break
    write_jsonl(selected_manifest, selected)
    return len(selected), len(items)


def run_command(command: list[str], *, cwd: Path, dry_run: bool = False) -> int:
    print(" ".join(command))
    if dry_run:
        return 0
    return subprocess.run(command, cwd=str(cwd), check=False).returncode


def project_path(project_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return project_root / path


def append_if_value(command: list[str], flag: str, value: str | None) -> None:
    if value:
        command.extend([flag, value])


def build_make_manifest_command(
    *,
    python_bin: str,
    worldeval_root: Path,
    root: Path,
    pipeline: str,
    output: Path,
    print_skipped: bool,
    ref_pattern: str,
    gen_pattern: str | None,
    chunk_pattern: str | None,
    output_name_template: str | None,
) -> list[str]:
    command = [
        python_bin,
        str(worldeval_root / "batch_test" / "make_manifest.py"),
        "--root",
        str(root),
        "--pipeline",
        pipeline,
        "--output",
        str(output),
        "--ref-pattern",
        ref_pattern,
    ]
    append_if_value(command, "--gen-pattern", gen_pattern)
    append_if_value(command, "--chunk-pattern", chunk_pattern)
    append_if_value(command, "--output-name-template", output_name_template)
    if print_skipped:
        command.append("--print-skipped")
    return command


def build_scheduler_command(
    *,
    args: argparse.Namespace,
    python_bin: str,
    worldeval_root: Path,
    project_root: Path,
    manifest: Path,
    log_dir: Path,
    summary: Path,
) -> list[str]:
    command = [
        python_bin,
        str(worldeval_root / "batch_test" / "batch_scheduler.py"),
        "--worldeval-root",
        str(worldeval_root),
        "--project-root",
        str(project_root),
        "--manifest",
        str(manifest),
        "--gpu-slots",
        args.gpu_slots,
        "--qwen-server-urls",
        args.qwen_server_urls,
        "--workers",
        str(args.workers),
        "--physical-batch-mode",
        args.physical_batch_mode,
        "--physical-max-frames",
        str(args.physical_max_frames),
        "--three-d-max-frames",
        str(args.three_d_max_frames),
        "--log-dir",
        str(log_dir),
        "--summary",
        str(summary),
    ]
    append_if_value(command, "--sam3-server-urls", args.sam3_server_urls)
    append_if_value(command, "--reward-3d-server-urls", args.reward_3d_server_urls)
    append_if_value(command, "--vlm-model", args.vlm_model)
    append_if_value(command, "--interaction-vlm-model", args.interaction_vlm_model)
    append_if_value(command, "--three-d-scoring-model", args.three_d_scoring_model)
    append_if_value(command, "--three-d-model-name", args.three_d_model_name)
    append_if_value(command, "--sam3-model", args.sam3_model)
    append_if_value(command, "--clip-download-root", args.clip_download_root)
    if args.run_clip_interaction:
        command.append("--run-clip-interaction")
    if args.skip_physical:
        command.append("--skip-physical")
    if args.skip_interaction:
        command.append("--skip-interaction")
    if args.skip_3d:
        command.append("--skip-3d")
    if args.three_d_no_mask_dynamic_objects:
        command.append("--three-d-no-mask-dynamic-objects")
    if args.force:
        command.append("--force")
    if args.dry_run:
        command.append("--dry-run")
    for extra_arg in args.extra_arg:
        command.extend(["--extra-arg", extra_arg])
    return command


def summarize_scores(
    *,
    python_bin: str,
    worldeval_root: Path,
    project_root: Path,
    progress_summary: Path,
    log_dir: Path,
    dry_run: bool,
) -> int:
    command = [
        python_bin,
        str(worldeval_root / "batch_test" / "summarize_scores.py"),
        str(progress_summary),
        "--include-non-ok",
        "--output-json",
        str(log_dir / "score_summary.json"),
        "--output-csv",
        str(log_dir / "score_cases.csv"),
    ]
    return run_command(command, cwd=project_root, dry_run=dry_run)


def print_pipeline_table() -> None:
    print("Supported pipeline aliases:")
    for pipeline in DEFAULT_PIPELINES:
        spec = PIPELINE_BY_ALIAS[pipeline]
        aliases = ", ".join(spec.aliases)
        print(f"  {spec.name}: output_prefix={spec.output_prefix}; aliases={aliases}")


def parse_args() -> argparse.Namespace:
    script_path = Path(__file__).resolve()
    default_worldeval_root = script_path.parents[1]
    default_project_root = default_worldeval_root.parent

    parser = argparse.ArgumentParser(
        description=(
            "Build manifests and run WorldEval scoring for one or more generated "
            "pipeline outputs under outputs_batch/<domain>/."
        )
    )
    parser.add_argument("--list-pipelines", action="store_true", help="Print pipeline aliases and exit")
    parser.add_argument("--worldeval-root", default=str(default_worldeval_root), help="Path to worldeval")
    parser.add_argument("--project-root", default=str(default_project_root), help="OpenWorldLib project root")
    parser.add_argument("--python-bin", default=sys.executable, help="Python executable used for child commands")
    parser.add_argument("--output-root", default="outputs_batch", help="Root containing domain directories")
    parser.add_argument(
        "--root",
        help=(
            "Evaluate a single case-root directory instead of outputs_batch/<domain>. "
            "When set, --domains defaults to custom."
        ),
    )
    parser.add_argument("--domain-name", default="custom", help="Name used in logs when --root is set")
    parser.add_argument("--domains", nargs="*", help="Domains to evaluate, comma or space separated")
    parser.add_argument("--pipelines", nargs="*", help="Pipeline aliases to evaluate, comma or space separated")
    parser.add_argument(
        "--skip-pair",
        action="append",
        help="Skip a domain/pipeline pair, formatted as DOMAIN:PIPELINE. Can be repeated or comma separated.",
    )
    parser.add_argument("--manifest-dir", default="batch_manifests", help="Directory for generated manifests")
    parser.add_argument("--log-root", default="batch_logs", help="Directory for scheduler logs and score summaries")
    parser.add_argument("--limit", type=int, help="Maximum selected items per domain/pipeline")
    parser.add_argument(
        "--include-existing",
        action="store_true",
        help="Include cases with existing score JSONs in the scheduler manifest.",
    )
    parser.add_argument("--force", action="store_true", help="Recompute cases whose score JSON already exists")
    parser.add_argument("--dry-run", action="store_true", help="Print child commands without running scoring")
    parser.add_argument("--no-summarize", action="store_true", help="Skip score aggregation after each run")
    parser.add_argument("--print-skipped", action="store_true", help="Print incomplete cases while building manifests")

    parser.add_argument("--gpu-slots", required=False, default="0", help="Comma-separated GPUs for scoring workers")
    parser.add_argument("--workers", type=int, default=1, help="Number of scoring workers")
    parser.add_argument("--qwen-server-urls", default="http://127.0.0.1:8008")
    parser.add_argument("--sam3-server-urls", default="")
    parser.add_argument("--reward-3d-server-urls", default="")
    parser.add_argument("--physical-max-frames", type=int, default=64)
    parser.add_argument("--physical-batch-mode", choices=["none", "dimension", "all"], default="dimension")
    parser.add_argument("--three-d-max-frames", type=int, default=32)
    parser.add_argument("--vlm-model")
    parser.add_argument("--interaction-vlm-model")
    parser.add_argument("--three-d-scoring-model")
    parser.add_argument("--three-d-model-name")
    parser.add_argument("--sam3-model")
    parser.add_argument("--clip-download-root")
    parser.add_argument("--run-clip-interaction", dest="run_clip_interaction", action="store_true", default=True)
    parser.add_argument("--no-run-clip-interaction", dest="run_clip_interaction", action="store_false")
    parser.add_argument("--skip-physical", action="store_true")
    parser.add_argument("--skip-interaction", action="store_true")
    parser.add_argument("--skip-3d", action="store_true")
    parser.add_argument("--three-d-no-mask-dynamic-objects", action="store_true")
    parser.add_argument("--extra-arg", action="append", default=[], help="Extra argument passed to the scoring script")

    parser.add_argument("--ref-pattern", default="ref_*.mp4", help="Reference video glob inside each case directory")
    parser.add_argument("--gen-pattern", help="Generated video glob override")
    parser.add_argument("--chunk-pattern", help="Chunk timestamp glob override")
    parser.add_argument("--output-name-template", help="Per-case score JSON filename template")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.list_pipelines:
        print_pipeline_table()
        return 0

    worldeval_root = Path(args.worldeval_root).resolve()
    project_root = Path(args.project_root).resolve()
    manifest_dir = project_path(project_root, args.manifest_dir)
    log_root = project_path(project_root, args.log_root)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    pipeline_names = [resolve_pipeline_name(value) for value in split_values(args.pipelines, DEFAULT_PIPELINES)]
    skip_pairs = parse_skip_pairs(args.skip_pair)
    if args.root:
        domain_roots = [(args.domain_name, project_path(project_root, args.root))]
    else:
        output_root = project_path(project_root, args.output_root)
        domain_roots = [
            (domain, output_root / domain)
            for domain in split_values(args.domains, DEFAULT_DOMAINS)
        ]

    print(f"Project root: {project_root}")
    print(f"WorldEval root: {worldeval_root}")
    print(f"Domains: {', '.join(domain for domain, _ in domain_roots)}")
    print(f"Pipelines: {', '.join(pipeline_names)}")
    print(f"GPU slots: {args.gpu_slots}; workers: {args.workers}")
    print()

    overall_failed = False
    for domain, root in domain_roots:
        if not root.exists():
            print(f"Skip missing domain root: {root}")
            continue
        for pipeline in pipeline_names:
            if (domain, pipeline) in skip_pairs:
                print(f"Skip configured pair: {domain}/{pipeline}")
                continue

            short_name = pipeline_short_name(pipeline)
            raw_manifest = manifest_dir / f"{domain}_{short_name}_all.jsonl"
            selected_suffix = "selected" if args.include_existing or args.force else "pending"
            if args.limit is not None:
                selected_suffix += f"_limit{args.limit}"
            selected_manifest = manifest_dir / f"{domain}_{short_name}_{selected_suffix}.jsonl"
            log_dir = log_root / f"{domain}_{short_name}"
            latest_summary = log_dir / ("summary_dry_run.jsonl" if args.dry_run else "summary_latest.jsonl")
            progress_summary = log_dir / "summary_from_outputs.jsonl"

            print(f"=== {domain}/{pipeline} ===")
            make_manifest_cmd = build_make_manifest_command(
                python_bin=args.python_bin,
                worldeval_root=worldeval_root,
                root=root,
                pipeline=pipeline,
                output=raw_manifest,
                print_skipped=args.print_skipped,
                ref_pattern=args.ref_pattern,
                gen_pattern=args.gen_pattern,
                chunk_pattern=args.chunk_pattern,
                output_name_template=args.output_name_template,
            )
            exit_code = run_command(make_manifest_cmd, cwd=project_root)
            if exit_code != 0:
                overall_failed = True
                print(f"Manifest build failed for {domain}/{pipeline}")
                continue

            selected_count, total_count = build_selected_manifest(
                raw_manifest=raw_manifest,
                selected_manifest=selected_manifest,
                pending_only=not args.include_existing,
                force=args.force,
                limit=args.limit,
            )
            done_count, _ = write_progress_summary(raw_manifest, progress_summary)
            print(
                f"Manifest items: total={total_count}, done={done_count}, "
                f"selected={selected_count} -> {selected_manifest}"
            )

            if selected_count > 0:
                scheduler_cmd = build_scheduler_command(
                    args=args,
                    python_bin=args.python_bin,
                    worldeval_root=worldeval_root,
                    project_root=project_root,
                    manifest=selected_manifest,
                    log_dir=log_dir,
                    summary=latest_summary,
                )
                exit_code = run_command(scheduler_cmd, cwd=project_root)
                if exit_code != 0:
                    overall_failed = True
                done_count, _ = write_progress_summary(raw_manifest, progress_summary)
                print(f"Progress after scheduler: {done_count}/{total_count} done")
            else:
                print("No selected items; refreshed progress summary only.")

            if not args.no_summarize:
                exit_code = summarize_scores(
                    python_bin=args.python_bin,
                    worldeval_root=worldeval_root,
                    project_root=project_root,
                    progress_summary=progress_summary,
                    log_dir=log_dir,
                    dry_run=args.dry_run,
                )
                if exit_code != 0:
                    overall_failed = True
            print()

    return 1 if overall_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
