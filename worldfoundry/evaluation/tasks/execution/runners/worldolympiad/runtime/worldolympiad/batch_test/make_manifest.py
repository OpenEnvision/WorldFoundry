#!/usr/bin/env python3
"""Create a JSONL manifest for batch scoring directories."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from batch_test.test_bench import PIPELINE_BY_ALIAS, PIPELINE_SPECS, sanitize_filename_part
except Exception:
    @dataclass(frozen=True)
    class PipelineSpec:
        name: str
        output_prefix: str
        aliases: tuple[str, ...]

    PIPELINE_SPECS = (
        PipelineSpec("cosmos-predict", "cosmos", ("cosmos", "cosmos-predict", "cosmos_predict", "cosmos-predict2p5", "cosmos_predict2p5")),
        PipelineSpec("hunyuan-gamecraft", "hunyuan_gamecraft", ("hunyuan-gamecraft", "hunyuan_gamecraft", "gamecraft")),
        PipelineSpec("hunyuan-worldplay", "hunyuan_worldplay", ("hunyuan-worldplay", "hunyuan_worldplay", "worldplay")),
        PipelineSpec("lingbot-world", "lingbot_world", ("lingbot-world", "lingbot_world", "lingbot")),
        PipelineSpec("longlive", "longlive", ("longlive",)),
        PipelineSpec("matrix-game2", "matrix_game2", ("matrix-game2", "matrix_game2", "matrix", "matrix-game-2", "matrix_game_2")),
        PipelineSpec("rolling-forcing", "rolling_forcing", ("rolling-forcing", "rolling_forcing")),
        PipelineSpec("wow", "wow", ("wow",)),
        PipelineSpec("yume1p5", "yume1p5", ("yume1p5", "yume-1p5", "yume_1p5", "yume")),
    )
    PIPELINE_BY_ALIAS = {
        alias: spec
        for spec in PIPELINE_SPECS
        for alias in (spec.name, *spec.aliases)
    }

    def sanitize_filename_part(value: str) -> str:
        import os
        import re

        value = value.strip().replace(os.sep, "_")
        if os.altsep:
            value = value.replace(os.altsep, "_")
        value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
        value = value.strip("._-")
        return value or "sample"


def find_one(directory: Path, pattern: str) -> Path | None:
    matches = sorted(directory.glob(pattern))
    return matches[0] if matches else None


def build_item(
    case_dir: Path,
    *,
    output_name: str | None,
    output_name_template: str,
    output_prefix: str | None,
    gen_pattern: str,
    ref_pattern: str,
    chunk_pattern: str | None,
    pipeline: str | None,
) -> tuple[dict[str, Any] | None, str | None]:
    gen_video = find_one(case_dir, gen_pattern)
    ref_video = find_one(case_dir, ref_pattern)
    prompt_json = case_dir / "prompt.json"
    chunk_json = None
    if gen_video is not None:
        exact_chunk_json = case_dir / f"{gen_video.stem}_chunk_timestamps.json"
        if exact_chunk_json.exists():
            chunk_json = exact_chunk_json
    if chunk_json is None and chunk_pattern:
        chunk_json = find_one(case_dir, chunk_pattern)

    missing = []
    if not gen_video:
        missing.append(f"generated video matching {gen_pattern!r}")
    if not ref_video:
        missing.append(f"reference video matching {ref_pattern!r}")
    if not prompt_json.exists():
        missing.append("prompt.json")
    if not chunk_json:
        missing.append(
            f"chunk timestamps matching {chunk_pattern!r}"
            if chunk_pattern
            else "chunk timestamps matching generated video stem"
        )
    if missing:
        return None, "; ".join(missing)

    score_output_name = output_name or output_name_template.format(
        id=case_dir.name,
        pipeline=pipeline or "",
        prefix=output_prefix or "",
    )
    item = {
        "id": case_dir.name,
        "video": str(gen_video),
        "gt_video": str(ref_video),
        "prompt_json": str(prompt_json),
        "chunk_json": str(chunk_json),
        "output": str(case_dir / score_output_name),
    }
    if pipeline:
        item["pipeline"] = pipeline
    if output_prefix:
        item["pipeline_output_prefix"] = output_prefix
    return item, None


def resolve_pipeline(pipeline: str | None, pipeline_output_prefix: str | None) -> tuple[str | None, str | None]:
    if not pipeline and not pipeline_output_prefix:
        return None, None
    if pipeline:
        try:
            spec = PIPELINE_BY_ALIAS[pipeline]
        except KeyError as exc:
            valid = ", ".join(sorted(PIPELINE_BY_ALIAS))
            raise ValueError(f"Unknown pipeline alias {pipeline!r}. Valid aliases: {valid}") from exc
        pipeline_name = spec.name
        output_prefix = pipeline_output_prefix or spec.output_prefix
    else:
        pipeline_name = None
        output_prefix = pipeline_output_prefix
    return pipeline_name, sanitize_filename_part(output_prefix)


def default_patterns(output_prefix: str | None, gen_pattern: str | None, chunk_pattern: str | None) -> tuple[str, str | None]:
    if gen_pattern:
        resolved_gen_pattern = gen_pattern
    elif output_prefix:
        resolved_gen_pattern = f"{output_prefix}_gen_*.mp4"
    else:
        resolved_gen_pattern = "matrix_game2_gen_*.mp4"

    if chunk_pattern:
        resolved_chunk_pattern = chunk_pattern
    elif output_prefix:
        resolved_chunk_pattern = f"{output_prefix}_gen_*_chunk_timestamps.json"
    elif gen_pattern:
        resolved_chunk_pattern = None
    else:
        resolved_chunk_pattern = "matrix_game2_gen_*_chunk_timestamps.json"
    return resolved_gen_pattern, resolved_chunk_pattern


def default_output_name_template(output_prefix: str | None, output_name_template: str | None) -> str:
    if output_name_template:
        return output_name_template
    if output_prefix:
        return "{prefix}_judge_{id}.json"
    return "score_physical_interaction_3d.json"


def print_pipeline_table() -> None:
    print("Supported pipelines from batch_test/test_bench.py:")
    for spec in PIPELINE_SPECS:
        aliases = ", ".join(spec.aliases)
        print(f"  {spec.name}: output_prefix={spec.output_prefix}; aliases={aliases}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create batch JSONL manifest from case directories")
    parser.add_argument("--root", help="Directory containing per-video case directories")
    parser.add_argument("--output", help="Output JSONL manifest path")
    parser.add_argument("--list-pipelines", action="store_true", help="Print pipeline aliases from batch_test/test_bench.py and exit")
    parser.add_argument(
        "--pipeline",
        choices=sorted(PIPELINE_BY_ALIAS),
        help=(
            "Pipeline alias from batch_test/test_bench.py / test_bench_pool_runner.py, "
            "e.g. cosmos-predict, longlive, matrix-game2. The alias resolves to the pipeline output_prefix."
        ),
    )
    parser.add_argument(
        "--pipeline-output-prefix",
        help=(
            "Override the resolved output prefix, matching test_bench.py --pipeline-output-prefix. "
            "Use this if generation used a custom prefix."
        ),
    )
    parser.add_argument(
        "--gen-pattern",
        help="Generated video glob inside each case directory. Overrides --pipeline. Example: 'cosmos_predict_*.mp4'.",
    )
    parser.add_argument("--ref-pattern", default="ref_*.mp4", help="Reference video glob inside each case directory")
    parser.add_argument(
        "--chunk-pattern",
        help=(
            "Chunk timestamp glob inside each case directory. If omitted, the script first tries "
            "<generated-video-stem>_chunk_timestamps.json."
        ),
    )
    parser.add_argument(
        "--output-name",
        help=(
            "Static per-case score JSON filename. If set, every case uses this exact filename. "
            "Usually prefer --output-name-template for pipeline-specific output names."
        ),
    )
    parser.add_argument(
        "--output-name-template",
        help=(
            "Per-case score JSON filename template. Available fields: {id}, {pipeline}, {prefix}. "
            "Defaults to {prefix}_judge_{id}.json when a pipeline/output prefix is set."
        ),
    )
    parser.add_argument(
        "--print-skipped",
        action="store_true",
        help="Print skipped case directories and the missing inputs.",
    )
    args = parser.parse_args()

    if args.list_pipelines:
        print_pipeline_table()
        return
    if not args.root:
        parser.error("--root is required unless --list-pipelines is used")
    if not args.output:
        parser.error("--output is required unless --list-pipelines is used")

    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(f"Root directory does not exist: {root}")

    pipeline_name, output_prefix = resolve_pipeline(args.pipeline, args.pipeline_output_prefix)
    gen_pattern, chunk_pattern = default_patterns(output_prefix, args.gen_pattern, args.chunk_pattern)
    output_name_template = default_output_name_template(output_prefix, args.output_name_template)

    items = []
    skipped = []
    for case_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        item, reason = build_item(
            case_dir,
            output_name=args.output_name,
            output_name_template=output_name_template,
            output_prefix=output_prefix,
            gen_pattern=gen_pattern,
            ref_pattern=args.ref_pattern,
            chunk_pattern=chunk_pattern,
            pipeline=pipeline_name,
        )
        if item is None:
            skipped.append((str(case_dir), reason))
            continue
        items.append(item)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Wrote {len(items)} item(s) to {output_path}")
    if args.pipeline:
        print(f"Pipeline: {pipeline_name}")
    if output_prefix:
        print(f"Pipeline output prefix: {output_prefix}")
    print(f"Generated video pattern: {gen_pattern}")
    print(f"Chunk timestamp pattern: {chunk_pattern or '<generated-video-stem>_chunk_timestamps.json'}")
    print(f"Per-case score output name: {args.output_name or output_name_template}")
    if skipped:
        print(f"Skipped {len(skipped)} incomplete case directorie(s).")
        if args.print_skipped:
            for case_dir, reason in skipped:
                print(f"  {case_dir}: {reason}")


if __name__ == "__main__":
    main()
