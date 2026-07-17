"""Command-line interface for wrbench.

Sub-commands
------------
models           List supported models.
presets          List preset camera motions.
actions          Parse and validate a frame-action camera script.
generate         Compile a camera script and write model-native payload sidecars.
doctor           Validate registry and adapter wiring for all (or one) model(s).
profile          Profile a generation command (wall time, GPU memory, stage spans).
profile-summary  Aggregate resource profile JSON files.
prompt           Generate scene/task/camera prompts.
firstframe       Generate first-frame images from T2I prompts.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _col(value: Any, width: int) -> str:
    """Left-justify *value* in a field of *width* characters."""
    return str(value).ljust(width)


def _print_table(rows: list[list[str]], headers: list[str]) -> None:
    """Print a simple fixed-width table to stdout."""
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    sep = "  ".join("-" * w for w in widths)
    header_line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    print(header_line)
    print(sep)
    for row in rows:
        print("  ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)))


# ---------------------------------------------------------------------------
# Sub-command: models
# ---------------------------------------------------------------------------

def _cmd_models(args: argparse.Namespace) -> int:
    import wrbench

    records = wrbench.all_records()
    if not args.deferred:
        records = [r for r in records if not r.is_deferred]

    if args.json:
        data = []
        for r in records:
            data.append(
                {
                    "key": r.key,
                    "aliases": list(r.aliases),
                    "status": r.status,
                    "input_kind": r.input_kind,
                    "adapter": r.adapter,
                    "payload_type": r.payload_type,
                    "calibration_status": r.amplitude.calibration_status,
                    "rotation_gain": r.amplitude.rotation_gain,
                    "translation_gain": r.amplitude.translation_gain,
                    "max_amount": r.amplitude.max_amount,
                    "translation_unit": r.amplitude.translation_unit,
                    "capabilities": r.capabilities,
                    "notes": r.notes,
                    "is_deferred": r.is_deferred,
                }
            )
        print(json.dumps(data, indent=2))
        return 0

    rows = [
        [r.key, r.input_kind, r.adapter, r.payload_type, r.amplitude.calibration_status]
        for r in records
    ]
    _print_table(rows, ["key", "input_kind", "adapter", "payload_type", "calibration_status"])
    return 0


# ---------------------------------------------------------------------------
# Sub-command: presets
# ---------------------------------------------------------------------------

def _cmd_presets(args: argparse.Namespace) -> int:
    import wrbench

    names = wrbench.presets.preset_names()
    descriptions = {
        "static": "Hold the camera still for the entire clip.",
        "yaw_LR": "Yaw left to peak_deg, then return right (go-return).",
        "yaw_RL": "Yaw right to peak_deg, then return left (go-return).",
        "pan_LR": "Pan left then return right (go-return).",
        "pan_RL": "Pan right then return left (go-return).",
    }

    if args.json:
        data = []
        for name in names:
            script = wrbench.presets.build_preset(name)
            data.append(
                {
                    "name": name,
                    "description": descriptions.get(name, ""),
                    "default_script": script.to_string(),
                    "default_frame_count": script.frame_count,
                }
            )
        print(json.dumps(data, indent=2))
        return 0

    rows = []
    for name in names:
        script = wrbench.presets.build_preset(name)
        desc = descriptions.get(name, "")
        rows.append([name, desc, script.to_string()])
    _print_table(rows, ["name", "description", "default_expansion"])
    return 0


# ---------------------------------------------------------------------------
# Sub-command: actions
# ---------------------------------------------------------------------------

def _cmd_actions(args: argparse.Namespace) -> int:
    import wrbench

    camera_text: str = args.camera
    fps: int = args.fps

    try:
        script = wrbench.parse_camera_script(camera_text, fps=fps)
    except (ValueError, KeyError) as exc:
        print(f"error: invalid camera script: {exc}", file=sys.stderr)
        return 1

    normalized = script.to_string()
    frame_count = script.frame_count

    segments = []
    for action in script.actions:
        seg: dict[str, Any] = {
            "kind": action.kind,
            "direction": action.direction,
            "frames": action.frames,
        }
        if action.degrees is not None:
            seg["value"] = action.degrees
            seg["value_unit"] = "degrees"
        elif action.amount is not None:
            seg["value"] = action.amount
            seg["value_unit"] = "amount"
        else:
            seg["value"] = None
            seg["value_unit"] = None
        segments.append(seg)

    if args.json:
        print(
            json.dumps(
                {
                    "normalized_script": normalized,
                    "frame_count": frame_count,
                    "fps": fps,
                    "segments": segments,
                },
                indent=2,
            )
        )
        return 0

    print(f"normalized : {normalized}")
    print(f"frame_count: {frame_count if frame_count is not None else '(unset)'}")
    print(f"fps        : {fps}")
    print()
    rows = []
    for i, seg in enumerate(segments):
        value_str = str(seg["value"]) if seg["value"] is not None else "-"
        unit_str = seg["value_unit"] or "-"
        rows.append([str(i), seg["kind"], seg["direction"], value_str, unit_str, str(seg["frames"])])
    _print_table(rows, ["#", "kind", "direction", "value", "unit", "frames"])
    return 0


# ---------------------------------------------------------------------------
# Sub-command: generate
# ---------------------------------------------------------------------------

def _resolve_preset_camera(camera_str: str, args: argparse.Namespace):
    """Return a CameraScript or camera string for ``--camera preset:<name>``."""
    import wrbench

    if not camera_str.startswith("preset:"):
        return camera_str  # raw script string

    preset_name = camera_str[len("preset:"):]
    kwargs: dict[str, Any] = {}
    if args.peak_deg is not None:
        kwargs["peak_deg"] = args.peak_deg
    if args.amount is not None:
        kwargs["amount"] = args.amount
    if args.frames is not None:
        kwargs["frames"] = args.frames

    try:
        script = wrbench.presets.build_preset(preset_name, **kwargs)
    except KeyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        names = wrbench.presets.preset_names()
        print(f"  valid presets: {', '.join(names)}", file=sys.stderr)
        raise SystemExit(1)
    return script


def _cmd_generate(args: argparse.Namespace) -> int:
    import wrbench

    model_name: str = args.model
    camera_arg: str = args.camera
    out_path: str = args.out
    dry_run: bool = not args.no_dry_run

    # Resolve model key early so errors are clear
    try:
        key = wrbench.canonical_model_key(model_name)
    except KeyError:
        known = wrbench.list_models(include_deferred=True)
        print(f"error: unknown model {model_name!r}", file=sys.stderr)
        print(f"  known models: {', '.join(known)}", file=sys.stderr)
        return 1

    # Check deferred status
    record = wrbench.model_record(key)
    if record.is_deferred:
        print(
            f"error: model {key!r} is deferred (not yet dispatchable). "
            "Generation is not supported for deferred models.",
            file=sys.stderr,
        )
        return 1

    # Resolve camera
    camera = _resolve_preset_camera(camera_arg, args)

    # Validate inputs based on input_kind
    input_kind = record.input_kind
    image_path: str | None = args.image
    source_video_path: str | None = args.source_video

    if input_kind == "image":
        if source_video_path:
            print(
                f"error: model {key!r} is an image/TI2V model; use --image, not --source-video.",
                file=sys.stderr,
            )
            return 1
        if not image_path:
            print(
                f"error: model {key!r} is an image/TI2V model and requires --image <path>.",
                file=sys.stderr,
            )
            return 1
    elif input_kind == "source_video":
        if image_path:
            print(
                f"error: model {key!r} is a source-video/V2V model; use --source-video, not --image.",
                file=sys.stderr,
            )
            return 1
        if not source_video_path:
            print(
                f"error: model {key!r} is a source-video/V2V model and requires --source-video <path>.",
                file=sys.stderr,
            )
            return 1
    elif input_kind == "none":
        if image_path or source_video_path:
            print(
                f"error: model {key!r} does not use --image or --source-video; use --prompt for text conditioning.",
                file=sys.stderr,
            )
            return 1
        if args.no_dry_run and not str(args.prompt).strip():
            print(
                f"error: model {key!r} is prompt-only and requires --prompt for --no-dry-run.",
                file=sys.stderr,
            )
            return 1

    if args.no_dry_run:
        from wrbench.backends.registry import resolve_backend
        from wrbench.runtime import load_runtime_config

        runtime = load_runtime_config(Path(args.runtime_config)) if args.runtime_config else None
        backend = resolve_backend(key, runtime=runtime)
        if hasattr(backend, "available_for"):
            ok, reason = backend.available_for(key)  # type: ignore[attr-defined]
        else:
            ok, reason = backend.available()
        if not ok:
            print(
                f"error: --no-dry-run requires a runnable in-tree WorldFoundry model for {key!r}: {reason}",
                file=sys.stderr,
            )
            print("  Omit --no-dry-run to compile camera payload sidecars only.", file=sys.stderr)
            return 1
        dry_run = False

    try:
        result = wrbench.compile_camera(
            model=key,
            camera=camera,
            out=out_path,
            image=image_path,
            source_video=source_video_path,
            prompt=args.prompt,
            width=args.width,
            height=args.height,
            fps=args.fps,
            num_frames=args.num_frames,
            work_dir=args.work_dir,
            runtime_config=args.runtime_config,
            dry_run=dry_run,
        )
    except (ValueError, KeyError, OSError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    artifacts: dict[str, str] = result.get("artifacts", {})
    payload = result.get("payload")
    print(f"model    : {result['model']}")
    print(f"dry_run  : {result['dry_run']}")
    print("artifacts:")
    for name, path in artifacts.items():
        exists_mark = " (written)" if Path(path).exists() else " (missing)"
        print(f"  {name}: {path}{exists_mark}")
    generation = result.get("generation")
    if generation:
        print(f"generation: success={generation.get('success')} message={generation.get('message')}")
        if generation.get("output_path"):
            print(f"  output: {generation['output_path']}")
    return 0


# ---------------------------------------------------------------------------
# Sub-command: doctor
# ---------------------------------------------------------------------------

def _doctor_check_model(key: str) -> tuple[bool, list[str]]:
    """Run all doctor checks for one model key. Returns (pass, [lines])."""
    lines: list[str] = []

    # Import here to ensure adapter registry is populated
    import wrbench
    from wrbench.adapters.base import adapter_for_model, registered_model_keys

    # Check 1: registry record loads
    try:
        record = wrbench.model_record(key)
    except Exception as exc:
        lines.append(f"  FAIL  registry: {exc}")
        return False, lines

    lines.append(f"  pass  registry: record loaded (status={record.status})")

    # Deferred models: that is a pass for doctor
    if record.is_deferred:
        lines.append(f"  pass  deferred: model is deferred/not-dispatchable (expected)")
        lines.append(f"  PASS  {key}  [deferred]")
        return True, lines

    # Check 2: input_kind is valid
    from wrbench.registry import VALID_INPUT_KINDS
    if record.input_kind not in VALID_INPUT_KINDS:
        lines.append(f"  FAIL  input_kind: {record.input_kind!r} is not valid")
        return False, lines
    lines.append(f"  pass  input_kind: {record.input_kind}")

    # Check 3: adapter is registered
    if key not in registered_model_keys():
        lines.append(f"  FAIL  adapter: no adapter registered for {key!r}")
        return False, lines
    try:
        adapter = adapter_for_model(key)
        lines.append(f"  pass  adapter: {type(adapter).__name__}")
    except Exception as exc:
        lines.append(f"  FAIL  adapter: {exc}")
        return False, lines

    # Check 4: dry-run compile of static@16
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "doctor_test.mp4")
        dummy_image = os.path.join(tmp, "dummy_image.png")
        dummy_video = os.path.join(tmp, "dummy_video.mp4")
        # Create dummy files so path presence is satisfied
        Path(dummy_image).write_bytes(b"")
        Path(dummy_video).write_bytes(b"")

        caps = record.capabilities or {}
        if caps.get("supports_static", True):
            camera = f"static@{int(record.default_frames)}"
        else:
            camera = wrbench.presets.yaw_LR(peak_deg=30, frames=int(record.default_frames)).to_string()

        compile_kwargs: dict[str, Any] = {
            "model": key,
            "camera": camera,
            "out": out,
            "dry_run": True,
        }
        if record.input_kind == "image":
            compile_kwargs["image"] = dummy_image
        elif record.input_kind == "source_video":
            compile_kwargs["source_video"] = dummy_video

        try:
            result = wrbench.compile_camera(**compile_kwargs)
            artifacts = result.get("artifacts", {})
            written = [p for p in artifacts.values() if Path(p).exists()]
            lines.append(f"  pass  dry-run: compile succeeded, {len(written)} artifacts written")
        except Exception as exc:
            lines.append(f"  FAIL  dry-run: {exc}")
            return False, lines

    lines.append(f"  PASS  {key}")
    return True, lines


def _cmd_doctor(args: argparse.Namespace) -> int:
    import wrbench

    if args.all:
        all_rec = wrbench.all_records()
        targets = [r.key for r in all_rec]
    elif args.model:
        try:
            targets = [wrbench.canonical_model_key(args.model)]
        except KeyError:
            known = wrbench.list_models(include_deferred=True)
            print(f"error: unknown model {args.model!r}", file=sys.stderr)
            print(f"  known models: {', '.join(known)}", file=sys.stderr)
            return 1
    else:
        # Default: active models only
        targets = wrbench.list_models(include_deferred=False)

    if not targets:
        print("No models to check.")
        return 0

    total = len(targets)
    passed = 0
    failed_keys: list[str] = []

    for key in targets:
        print(f"\n[{key}]")
        ok, lines = _doctor_check_model(key)
        for line in lines:
            print(line)
        if ok:
            passed += 1
        else:
            failed_keys.append(key)

    print(f"\n{'='*60}")
    print(f"Summary: {passed}/{total} passed")
    if failed_keys:
        print(f"Failed:  {', '.join(failed_keys)}")
        return 1
    print("All checks passed.")
    return 0


# ---------------------------------------------------------------------------
# Sub-command: profile
# ---------------------------------------------------------------------------

def _cmd_profile(args: argparse.Namespace) -> int:
    from wrbench.profiling import run_profiled_command

    cmd = list(args.cmd)
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        print("profile: missing command after --", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.name or "run"
    summary = out_dir / f"{stem}.resource_profile.json"
    trace = out_dir / f"{stem}.resource_trace.jsonl" if not args.no_trace else None
    events = out_dir / f"{stem}.stage_events.jsonl"
    log = out_dir / f"{stem}.generation.log" if args.log else None

    profile = run_profiled_command(
        cmd,
        cwd=args.cwd,
        summary_path=summary,
        trace_path=trace,
        stage_events_path=events,
        log_path=log,
        run_identity={
            "model": args.model or "",
            "profile": args.profile or "",
            "camera": args.camera or "",
            "scene_id": args.scene_id or "",
            "gpu_width": args.gpu_width,
            "output_video_seconds": args.output_video_seconds,
        },
        sampling_interval_seconds=args.sampling_interval,
        generation_status=args.generation_status,
    )
    if args.json:
        print(json.dumps(profile, indent=2))
    else:
        dm = profile.get("derived_metrics", {})
        print(f"Summary: {summary}")
        print(f"exit_code={profile['status']['exit_code']} wall={profile['status']['wall_time_seconds']:.3f}s")
        bg = dm.get("benchmark_generation_seconds")
        if bg is not None:
            print(f"benchmark_generation_seconds={bg:.3f}")
        gpus = dm.get("benchmark_gpu_seconds_per_output_second")
        if gpus is not None:
            print(f"benchmark_gpu_seconds_per_output_second={gpus:.3f}")
    return int(profile["status"]["exit_code"])


# ---------------------------------------------------------------------------
# Sub-command: profile-summary
# ---------------------------------------------------------------------------

def _cmd_profile_summary(args: argparse.Namespace) -> int:
    from wrbench.profiling import format_rows, load_profiles, summarize

    rows = summarize(load_profiles([Path(p) for p in args.paths]))
    text = format_rows(rows, args.format)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
        print(f"Wrote {args.output}")
    else:
        print(text)
    return 0


# ---------------------------------------------------------------------------
# Sub-command: prompt
# ---------------------------------------------------------------------------

def _cmd_prompt(args: argparse.Namespace) -> int:
    if args.prompt_command == "camera":
        from wrbench.prompts.camera_text import (
            assemble_ti2v_prompt,
            build_prompt_to_send,
            preset_camera_text,
        )

        if args.preset:
            if not args.pronoun or not args.offscreen_area:
                print("prompt camera --preset requires --pronoun and --offscreen-area", file=sys.stderr)
                return 2
            text = preset_camera_text(args.preset, pronoun=args.pronoun, offscreen_area=args.offscreen_area)
            result = {"preset": args.preset, "camera_text": text}
        elif args.assemble:
            if not all((args.scene_start, args.event, args.pronoun, args.offscreen_area, args.gap)):
                print(
                    "prompt camera --assemble requires --scene-start, --event, --pronoun, --offscreen-area, and --gap",
                    file=sys.stderr,
                )
                return 2
            result = {
                "ti2v_prompt": assemble_ti2v_prompt(
                    args.scene_start,
                    args.event,
                    args.pronoun,
                    args.offscreen_area,
                    args.gap,
                )
            }
        elif args.model and args.source_prompt:
            if not args.camera_motion:
                print("prompt camera --model + --source-prompt requires --camera-motion", file=sys.stderr)
                return 2
            result = {
                "prompt_to_send": build_prompt_to_send(
                    args.source_prompt,
                    args.camera_motion,
                    model=args.model,
                )
            }
        else:
            print("prompt camera: use --preset, --assemble, or --model + --source-prompt", file=sys.stderr)
            return 2
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0

    if args.prompt_command == "scene":
        from wrbench.prompts.scene import enrich_family_with_t2i_scene

        family = json.loads(Path(args.family_json).read_text(encoding="utf-8"))
        enriched = enrich_family_with_t2i_scene(
            family,
            model=args.model,
            temperature=args.temperature,
            provider=args.provider,
            api_key=args.api_key,
            base_url=args.base_url,
        )
        print(json.dumps(enriched, indent=2, ensure_ascii=False))
        return 0

    if args.prompt_command == "task":
        from wrbench.prompts.task import (
            generate_variants_deterministic,
            load_jsonl,
            write_jsonl,
        )

        if args.deterministic:
            from wrbench.datasets import build_natural25_candidates, natural25_families_path

            candidates_path = args.candidates_json
            families_path = args.families_jsonl
            if bool(candidates_path) ^ bool(families_path):
                print(
                    "prompt task --deterministic: provide both --candidates-json and "
                    "--families-jsonl, or omit both to use bundled Natural-25 data.",
                    file=sys.stderr,
                )
                return 2
            if candidates_path and families_path:
                candidates = json.loads(Path(candidates_path).read_text(encoding="utf-8"))
                families = {r["family_id"]: r for r in load_jsonl(families_path)}
            else:
                candidates = build_natural25_candidates()
                families = {r["family_id"]: r for r in load_jsonl(natural25_families_path())}
            missing = [c["candidate_id"] for c in candidates if c["candidate_id"] not in families]
            if missing:
                print(
                    f"prompt task --deterministic: warning: {len(missing)} candidate(s) "
                    f"have no matching family and will be skipped.",
                    file=sys.stderr,
                )
            variants = generate_variants_deterministic(candidates, families)
            if args.output:
                write_jsonl(args.output, variants)
                print(f"Wrote {len(variants)} variants to {args.output}")
            else:
                print(json.dumps(variants[:3], indent=2, ensure_ascii=False))
                if len(variants) > 3:
                    print(f"... and {len(variants) - 3} more")
            return 0

        print("prompt task: only --deterministic path is implemented without LLM; use scene+LLM for tier groups", file=sys.stderr)
        return 2

    print(f"Unknown prompt subcommand: {args.prompt_command}", file=sys.stderr)
    return 2


# ---------------------------------------------------------------------------
# Sub-command: firstframe
# ---------------------------------------------------------------------------

def _cmd_firstframe(args: argparse.Namespace) -> int:
    from wrbench.firstframe import generate_first_frame, generate_first_frames_from_families, write_manifest
    from wrbench.prompts.task import load_jsonl

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.families_jsonl:
        families = list(load_jsonl(args.families_jsonl))
        manifests = generate_first_frames_from_families(
            families,
            out_dir=out_dir,
            provider=args.provider,
            model=args.model,
            api_key=args.api_key,
            endpoint=args.endpoint,
            size=args.size,
            n=args.n,
            skip_existing=args.skip_existing,
        )
    elif args.prompt and args.family_id:
        manifests = [
            generate_first_frame(
                family_id=args.family_id,
                prompt=args.prompt,
                out_dir=out_dir,
                provider=args.provider,
                model=args.model,
                api_key=args.api_key,
                endpoint=args.endpoint,
                size=args.size,
                n=args.n,
            )
        ]
    else:
        print("firstframe: provide --families-jsonl or --family-id + --prompt", file=sys.stderr)
        return 2

    manifest_path = out_dir / "first_frames_manifest.json"
    write_manifest(manifest_path, manifests)
    print(f"Generated {len(manifests)} frame(s); manifest: {manifest_path}")
    for m in manifests:
        print(f"  {m.family_id} -> {m.image_path}")
    return 0


# ---------------------------------------------------------------------------
# Sub-command: eval (D1-D6)
# ---------------------------------------------------------------------------

def _cmd_eval(args: argparse.Namespace) -> int:
    from pathlib import Path

    from wrbench.eval.runtime import (
        build_table,
        contract_path,
        d1_score,
        d1_camalign_score,
        d1_vggt_batch,
        d2_extract,
        d3d6_score,
        eval_run,
        require_eval_runtime,
    )

    runtime_path = Path(args.runtime_config) if args.runtime_config else None

    if args.eval_command == "contract":
        path = contract_path()
        if args.json:
            import json

            print(json.dumps(json.loads(path.read_text(encoding="utf-8")), indent=2))
        else:
            md = path.with_suffix(".md")
            print(md.read_text(encoding="utf-8") if md.is_file() else path.read_text(encoding="utf-8"))
        return 0

    if args.eval_command == "table":
        if not args.runtime_scores or not args.out_csv or not args.out_md or not args.out_summary:
            print(
                "eval table requires --runtime-scores --out-csv --out-md --out-summary",
                file=sys.stderr,
            )
            return 2
        argv = [
            "--runtime-scores",
            str(args.runtime_scores),
            "--out-csv",
            str(args.out_csv),
            "--out-md",
            str(args.out_md),
            "--out-summary",
            str(args.out_summary),
        ]
        if args.d1_scores:
            argv.extend(["--d1-scores", str(args.d1_scores)])
        if args.d1_camalign_scores:
            argv.extend(["--d1-camalign-scores", str(args.d1_camalign_scores)])
        if args.d2_scores:
            argv.extend(["--d2-scores", str(args.d2_scores)])
        return build_table(argv)

    if args.eval_command == "d1":
        # D1 scoring is pure-numpy over an existing pose cache; it does not need
        # the external model scorers, so do not require an eval.scorers block.
        return d1_score(
            input_jsonl=Path(args.input_jsonl),
            output_jsonl=Path(args.output_jsonl),
            summary_csv=Path(args.summary_csv),
            pose_cache_root=Path(args.pose_cache_root),
            pose_backend=args.pose_backend,
            poses_file=args.poses_file,
            default_frames=args.default_frames,
            sidecar_profile_gate=args.sidecar_profile_gate,
            predicted_pose_type=args.predicted_pose_type,
            predicted_camera_convention=args.predicted_camera_convention,
            target_camera_convention=args.target_camera_convention,
            rot_scale_deg=args.rot_scale_deg,
            trans_scale=args.trans_scale,
            yaw_weak_threshold_deg=args.yaw_weak_threshold_deg,
            pan_weak_threshold=args.pan_weak_threshold,
            static_rot_threshold_deg=args.static_rot_threshold_deg,
            static_trans_threshold=args.static_trans_threshold,
        )

    if args.eval_command == "d1-camalign":
        return d1_camalign_score(
            input_jsonl=Path(args.input_jsonl),
            output_jsonl=Path(args.output_jsonl),
            pose_cache_root=Path(args.pose_cache_root),
            poses_file=args.poses_file,
        )

    eval_runtime = require_eval_runtime(runtime_path)

    if args.eval_command == "d1-vggt":
        return d1_vggt_batch(
            eval_runtime=eval_runtime,
            input_jsonl=Path(args.input_jsonl),
            output_root=Path(args.output_root),
            cache_root=Path(args.cache_root),
            execution_mode=args.execution_mode,
        )

    if args.eval_command == "d2":
        return d2_extract(
            eval_runtime=eval_runtime,
            videos_manifest=Path(args.videos_manifest),
            out_jsonl=Path(args.out_jsonl),
            model_dir=Path(args.model_dir) if args.model_dir else None,
        )

    if args.eval_command == "d3d6":
        return d3d6_score(
            eval_runtime=eval_runtime,
            manifest=Path(args.manifest),
            out_dir=Path(args.out_dir),
            stage=args.stage,
            scorer_profile=args.scorer_profile,
        )

    if args.eval_command == "run":
        return eval_run(
            eval_runtime=eval_runtime,
            manifest=Path(args.manifest),
            out_dir=Path(args.out_dir),
            scorer_profile=args.scorer_profile,
            sidecar_profile_gate=args.sidecar_profile_gate,
        )

    print(f"eval: unknown sub-command {args.eval_command!r}", file=sys.stderr)
    return 2


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wrbench",
        description="Unified camera control for video-generation models.",
    )
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")
    sub = parser.add_subparsers(dest="command", metavar="<command>")
    sub.required = True

    # models
    p_models = sub.add_parser("models", help="List supported models.")
    p_models.add_argument(
        "--json", action="store_true", help="Output full records as a JSON array."
    )
    p_models.add_argument(
        "--deferred", action="store_true", help="Include deferred (not-yet-dispatchable) models."
    )

    # presets
    p_presets = sub.add_parser("presets", help="List preset camera motions.")
    p_presets.add_argument("--json", action="store_true", help="Output as JSON.")

    # actions
    p_actions = sub.add_parser(
        "actions", help="Parse and validate a frame-action camera script."
    )
    p_actions.add_argument(
        "--camera",
        required=True,
        metavar="SCRIPT",
        help='Camera script, e.g. "yaw:left:60@40,yaw:right:60@41".',
    )
    p_actions.add_argument(
        "--fps", type=int, required=True, metavar="N", help="Frames per second."
    )
    p_actions.add_argument("--json", action="store_true", help="Output as JSON.")

    # generate
    p_gen = sub.add_parser(
        "generate", help="Compile a camera script and write model-native payload sidecars."
    )
    p_gen.add_argument("--model", required=True, metavar="MODEL", help="Target model key.")
    p_gen.add_argument(
        "--camera",
        required=True,
        metavar="SCRIPT",
        help=(
            'Camera script or "preset:<name>", e.g. "yaw:left:60@40" or "preset:yaw_LR". '
            "Combine with --peak-deg, --amount, --frames to parametrize presets."
        ),
    )
    p_gen.add_argument("--out", required=True, metavar="PATH", help="Output file path (e.g. out.mp4).")
    p_gen.add_argument("--image", metavar="PATH", help="Input image (required for TI2V models).")
    p_gen.add_argument(
        "--source-video", metavar="PATH", dest="source_video",
        help="Input source video (required for V2V models).",
    )
    p_gen.add_argument("--prompt", required=True, metavar="TEXT", help="Text prompt.")
    p_gen.add_argument("--width", type=int, metavar="N", help="Override model registry frame width.")
    p_gen.add_argument("--height", type=int, metavar="N", help="Override model registry frame height.")
    p_gen.add_argument("--fps", type=int, metavar="N", help="Override model registry frames per second.")
    p_gen.add_argument(
        "--num-frames", type=int, dest="num_frames",
        metavar="N", help="Override total frame count.",
    )
    p_gen.add_argument("--work-dir", metavar="DIR", dest="work_dir", help="Working directory for scratch files.")
    p_gen.add_argument(
        "--runtime-config",
        metavar="PATH",
        dest="runtime_config",
        help="Optional legacy checkpoint overrides forwarded to the WorldFoundry runner.",
    )
    p_gen.add_argument(
        "--no-dry-run",
        action="store_true",
        dest="no_dry_run",
        help=(
            "Disable dry-run mode and invoke the canonical in-tree WorldFoundry model runner."
        ),
    )
    # Preset parametrisation flags
    p_gen.add_argument(
        "--peak-deg", type=float, dest="peak_deg",
        metavar="DEG", help="Override preset peak_deg (yaw/pitch presets).",
    )
    p_gen.add_argument(
        "--amount", type=float,
        metavar="A", help="Override preset amount (translation presets).",
    )
    p_gen.add_argument(
        "--frames", type=int,
        metavar="N", help="Override preset total frame count.",
    )

    # profile
    p_prof = sub.add_parser("profile", help="Profile a generation command.")
    p_prof.add_argument("--out-dir", required=True, metavar="DIR", help="Output directory for profile artifacts.")
    p_prof.add_argument("--name", required=True, metavar="NAME", help="Artifact filename stem.")
    p_prof.add_argument("--model", metavar="MODEL")
    p_prof.add_argument("--profile", metavar="PROFILE")
    p_prof.add_argument("--camera", metavar="CAMERA")
    p_prof.add_argument("--scene-id", dest="scene_id", metavar="ID")
    p_prof.add_argument("--gpu-width", type=int, required=True, dest="gpu_width")
    p_prof.add_argument("--output-video-seconds", type=float, dest="output_video_seconds")
    p_prof.add_argument("--generation-status", dest="generation_status")
    p_prof.add_argument("--sampling-interval", type=float, required=True, dest="sampling_interval")
    p_prof.add_argument("--cwd", required=True, metavar="DIR")
    p_prof.add_argument("--log", action="store_true", help="Capture command stdout/stderr to a log file.")
    p_prof.add_argument("--no-trace", action="store_true", help="Skip GPU trace JSONL.")
    p_prof.add_argument("--json", action="store_true", help="Print full profile JSON to stdout.")
    p_prof.add_argument("cmd", nargs=argparse.REMAINDER, help="Command after --")

    # profile-summary
    p_psum = sub.add_parser("profile-summary", help="Summarize resource profile JSON files.")
    p_psum.add_argument("paths", nargs="+", metavar="PATH", help="Profile JSON, JSONL, or directory.")
    p_psum.add_argument("--format", choices=("json", "csv", "markdown"), required=True)
    p_psum.add_argument("--output", metavar="PATH")

    # prompt
    p_prompt = sub.add_parser("prompt", help="Generate scene/task/camera prompts.")
    p_prompt_sub = p_prompt.add_subparsers(dest="prompt_command", metavar="<kind>")
    p_prompt_sub.required = True

    p_cam = p_prompt_sub.add_parser("camera", help="Camera prompt text or API assembly.")
    p_cam.add_argument("--preset", metavar="NAME", help="wrbench preset name (yaw_LR, static, …).")
    p_cam.add_argument("--pronoun")
    p_cam.add_argument("--offscreen-area", dest="offscreen_area")
    p_cam.add_argument("--assemble", action="store_true", help="Assemble ti2v_prompt from parts.")
    p_cam.add_argument("--scene-start", dest="scene_start")
    p_cam.add_argument("--event")
    p_cam.add_argument("--gap")
    p_cam.add_argument("--model", metavar="API_MODEL", help="API model for prompt_to_send assembly.")
    p_cam.add_argument("--source-prompt", dest="source_prompt", metavar="TEXT")
    p_cam.add_argument("--camera-motion", dest="camera_motion")

    p_scene = p_prompt_sub.add_parser("scene", help="Generate t2i_scene via LLM (requires wrbench[prompts]).")
    p_scene.add_argument("--family-json", required=True, dest="family_json", metavar="PATH")
    p_scene.add_argument("--model", required=True)
    p_scene.add_argument("--provider", required=True)
    p_scene.add_argument("--base-url", required=True, dest="base_url")
    p_scene.add_argument("--api-key", required=True, dest="api_key")
    p_scene.add_argument("--temperature", type=float, required=True)

    p_task = p_prompt_sub.add_parser("task", help="Generate ti2v variant prompts.")
    p_task.add_argument("--deterministic", action="store_true", help="Deterministic Natural-25 style path.")
    p_task.add_argument(
        "--candidates-json",
        dest="candidates_json",
        metavar="PATH",
        help="Optional candidates JSON; omit to build from bundled data/natural25/scene_events_25x4.csv.",
    )
    p_task.add_argument(
        "--families-jsonl",
        dest="families_jsonl",
        metavar="PATH",
        help="Optional families JSONL; omit to use bundled data/natural25/families.jsonl.",
    )
    p_task.add_argument("--output", metavar="PATH")

    # firstframe
    p_ff = sub.add_parser("firstframe", help="Generate first-frame images.")
    p_ff.add_argument("--out", required=True, metavar="DIR", help="Output directory for PNGs.")
    p_ff.add_argument("--families-jsonl", dest="families_jsonl", metavar="PATH")
    p_ff.add_argument("--family-id", dest="family_id", metavar="ID")
    p_ff.add_argument("--prompt", metavar="TEXT", help="T2I prompt (with --family-id).")
    p_ff.add_argument("--provider", required=True, help="T2I provider: dashscope, mock.")
    p_ff.add_argument("--model", required=True, help="T2I model name.")
    p_ff.add_argument("--api-key", dest="api_key", help="T2I API key.")
    p_ff.add_argument("--endpoint", required=True, help="T2I API endpoint.")
    p_ff.add_argument("--size", required=True, help="T2I output size, provider-specific.")
    p_ff.add_argument("--n", required=True, help="T2I output count, provider-specific.")
    ff_existing = p_ff.add_mutually_exclusive_group(required=True)
    ff_existing.add_argument("--skip-existing", action="store_true", help="Skip existing PNGs and record them in the manifest.")
    ff_existing.add_argument("--overwrite-existing", action="store_true", help="Regenerate existing PNGs.")

    # doctor
    p_doc = sub.add_parser(
        "doctor", help="Validate registry and adapter wiring."
    )
    model_group = p_doc.add_mutually_exclusive_group()
    model_group.add_argument(
        "--model", metavar="MODEL", help="Check a single model."
    )
    model_group.add_argument(
        "--all", action="store_true", help="Check all models including deferred."
    )

    # eval
    p_eval = sub.add_parser(
        "eval",
        help="WRBench diagnostic evaluation (core): D1-D6 metrics + main table.",
    )
    eval_sub = p_eval.add_subparsers(dest="eval_command", metavar="<eval-command>")
    eval_sub.required = True
    p_eval.add_argument(
        "--runtime-config",
        dest="runtime_config",
        help="Explicit path to wrbench.runtime.json for eval scorer runtime paths.",
    )

    p_eval_contract = eval_sub.add_parser("contract", help="Print the D1-D6 metric contract.")
    p_eval_contract.add_argument("--json", action="store_true", help="Print JSON contract.")

    p_eval_d1 = eval_sub.add_parser("d1", help="Score D1 camera accuracy from pose cache + sidecars.")
    p_eval_d1.add_argument("--input-jsonl", required=True)
    p_eval_d1.add_argument("--output-jsonl", required=True)
    p_eval_d1.add_argument("--summary-csv", required=True)
    p_eval_d1.add_argument("--pose-cache-root", required=True)
    p_eval_d1.add_argument("--pose-backend", required=True)
    p_eval_d1.add_argument("--poses-file", required=True)
    p_eval_d1.add_argument("--default-frames", type=int, required=True)
    p_eval_d1.add_argument("--predicted-pose-type", required=True)
    p_eval_d1.add_argument("--predicted-camera-convention", required=True)
    p_eval_d1.add_argument("--target-camera-convention", required=True)
    p_eval_d1.add_argument("--rot-scale-deg", type=float, required=True)
    p_eval_d1.add_argument("--trans-scale", type=float, required=True)
    p_eval_d1.add_argument("--yaw-weak-threshold-deg", type=float, required=True)
    p_eval_d1.add_argument("--pan-weak-threshold", type=float, required=True)
    p_eval_d1.add_argument("--static-rot-threshold-deg", type=float, required=True)
    p_eval_d1.add_argument("--static-trans-threshold", type=float, required=True)
    p_eval_d1.add_argument(
        "--sidecar-profile-gate",
        choices=("main", "certified_opencv"),
        required=True,
        help=(
            "Target sidecar validation gate. 'main' is the canonical main-table profile; "
            "'certified_opencv' accepts certified OpenCV C2W sidecars after external QC."
        ),
    )

    p_eval_camalign = eval_sub.add_parser(
        "d1-camalign",
        help="Score D1 prompt-camera alignment (CamAlign) from pose cache.",
    )
    p_eval_camalign.add_argument("--input-jsonl", required=True)
    p_eval_camalign.add_argument("--output-jsonl", required=True)
    p_eval_camalign.add_argument("--pose-cache-root", required=True)
    p_eval_camalign.add_argument("--poses-file", required=True)

    p_eval_vggt = eval_sub.add_parser("d1-vggt", help="Export VGGT-Omega poses for D1 scoring.")
    p_eval_vggt.add_argument("--input-jsonl", required=True)
    p_eval_vggt.add_argument("--output-root", required=True)
    p_eval_vggt.add_argument("--cache-root", required=True)
    p_eval_vggt.add_argument(
        "--execution-mode",
        choices=("subprocess", "inprocess"),
        required=True,
    )

    p_eval_d2 = eval_sub.add_parser("d2", help="Extract DINOv2 D2 visual-integrity features.")
    p_eval_d2.add_argument("--videos-manifest", required=True)
    p_eval_d2.add_argument("--out-jsonl", required=True)
    p_eval_d2.add_argument("--model-dir")

    p_eval_run = eval_sub.add_parser(
        "run",
        help="Run full eval pipeline: D1 pose -> D1 score -> D2 -> D3-D6 -> table.",
    )
    p_eval_run.add_argument("--manifest", required=True, help="JSON manifest of videos to score.")
    p_eval_run.add_argument("--out-dir", required=True, help="Output directory for all eval artifacts.")
    p_eval_run.add_argument(
        "--scorer-profile",
        required=True,
        choices=("wrbench_default", "ablation_manifest_metadata", "custom"),
    )
    p_eval_run.add_argument(
        "--sidecar-profile-gate",
        choices=("main", "certified_opencv"),
        required=True,
        help=(
            "Target sidecar validation gate. 'main' is the canonical main-table profile; "
            "'certified_opencv' accepts certified OpenCV C2W sidecars after external QC."
        ),
    )

    p_eval_d3d6 = eval_sub.add_parser("d3d6", help="Run visible/returned VLM scoring stages (power users).")
    p_eval_d3d6.add_argument("--manifest", required=True)
    p_eval_d3d6.add_argument("--out-dir", required=True)
    p_eval_d3d6.add_argument(
        "--stage",
        required=True,
        choices=(
            "preflight",
            "qwen35",
            "merge_qwen35",
            "gate_binary",
            "merge_binary",
            "build_rescue",
            "gate_rescue",
            "merge_rescue",
            "overlay_gate",
            "export",
            "all",
        ),
    )
    p_eval_d3d6.add_argument(
        "--scorer-profile",
        required=True,
        choices=("wrbench_default", "ablation_manifest_metadata", "custom"),
    )

    p_eval_table = eval_sub.add_parser("table", help="Build D1-D6 main benchmark table.")
    p_eval_table.add_argument("--runtime-scores", required=True)
    p_eval_table.add_argument("--d1-scores")
    p_eval_table.add_argument("--d1-camalign-scores")
    p_eval_table.add_argument("--d2-scores")
    p_eval_table.add_argument("--out-csv", required=True)
    p_eval_table.add_argument("--out-md", required=True)
    p_eval_table.add_argument("--out-summary", required=True)

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

_DISPATCH = {
    "models": _cmd_models,
    "presets": _cmd_presets,
    "actions": _cmd_actions,
    "generate": _cmd_generate,
    "doctor": _cmd_doctor,
    "profile": _cmd_profile,
    "profile-summary": _cmd_profile_summary,
    "prompt": _cmd_prompt,
    "firstframe": _cmd_firstframe,
    "eval": _cmd_eval,
}


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    handler = _DISPATCH.get(args.command)
    if handler is None:
        parser.print_help()
        return 1
    return handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
