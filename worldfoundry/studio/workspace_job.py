from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from worldfoundry.studio.inference.catalog import find_entry
from worldfoundry.studio.inference.dispatch import dispatch_spec_for_inference, run_manager_payload_in_conda
from worldfoundry.studio.inference.execution import StudioManager
from worldfoundry.studio.serving.workspace import (
    SETTINGS,
    JobCreateRequest,
    _call_param_names,
    _inference_run_kwargs,
)


def _blank_to_none(value: str | None) -> str | None:
    text = str(value or "").strip()
    return text or None


def _int_value(value: str | int | None) -> int | None:
    if value in {"", None}:
        return None
    return int(value)


def _float_value(value: str | float | None) -> float | None:
    if value in {"", None}:
        return None
    return float(value)


def _parse_json_mapping(value: str | None, *, label: str) -> dict[str, Any]:
    if not value or not value.strip():
        return {}
    payload = json.loads(value)
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return dict(payload)




def _apply_size_param(params: dict[str, Any], value: str | None) -> None:
    text = str(value or "").strip().lower()
    if not text:
        return
    parts = [item for item in re.split(r"[x*,: ]+", text) if item]
    if len(parts) != 2:
        raise ValueError("size must be formatted as WIDTHxHEIGHT or WIDTH*HEIGHT")
    width, height = (int(parts[0]), int(parts[1]))
    params["width"] = width
    params["height"] = height


def _truthy_string(value: str | None) -> bool | None:
    if value in {"", None}:
        return None
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _build_infer_payload(args: argparse.Namespace) -> JobCreateRequest:
    entry = find_entry(args.model_id)
    params: dict[str, Any] = {}
    if args.frames not in {"", None}:
        params["num_frames"] = _int_value(args.frames)
    if args.steps not in {"", None}:
        params["num_inference_steps"] = _int_value(args.steps)
    if args.seed not in {"", None}:
        params["seed"] = _int_value(args.seed)
    if args.fps not in {"", None}:
        params["fps"] = _int_value(args.fps)
    if args.interactions:
        params["interactions"] = args.interactions

    call_param_names = _call_param_names(entry)
    call_kwargs = _parse_json_mapping(args.call_json, label="call-json")
    load_kwargs = _parse_json_mapping(args.load_json, label="load-json")
    if args.guidance_scale not in {"", None}:
        guidance_value = _float_value(args.guidance_scale)
        guidance_param = next(
            (name for name in ("guidance_scale", "cfg_scale", "guidance", "scale") if name in call_param_names),
            None,
        )
        if guidance_param is None:
            params["guidance_scale"] = guidance_value
        else:
            call_kwargs[guidance_param] = guidance_value
    if args.size:
        if "size" in call_param_names:
            call_kwargs["size"] = args.size
        else:
            _apply_size_param(params, args.size)
    if args.output_path:
        call_kwargs["output_path"] = args.output_path
    optional_call_values: dict[str, Any] = {
        "input_dir": args.input_dir,
        "trajectory_file": args.trajectory_file,
        "task": args.task,
        "mode": args.mode,
        "resize_mode": args.resize_mode,
        "frames_per_generation": _int_value(args.frames_per_generation),
        "dtype": args.dtype,
        "max_sequence_length": _int_value(args.max_sequence_length),
        "cam_type": _int_value(args.cam_type),
        "output_formats": args.output_formats,
        "trajectory": args.trajectory,
        "angle": _float_value(args.angle),
        "distance": _float_value(args.distance),
        "orbit_radius": _float_value(args.orbit_radius),
        "zoom_ratio": _float_value(args.zoom_ratio),
        "alpha_threshold": _float_value(args.alpha_threshold),
        "static_scene": _truthy_string(args.static_scene),
        "low_vram": _truthy_string(args.low_vram),
        "disable_lora": _truthy_string(args.disable_lora),
        "vis_rendering": _truthy_string(args.vis_rendering),
        "offload_t5": _truthy_string(args.offload_t5),
        "offload_transformer_during_vae": _truthy_string(args.offload_transformer_during_vae),
        "offload_vae": _truthy_string(args.offload_vae),
    }
    for key, value in optional_call_values.items():
        if value in {"", None}:
            continue
        if key in call_param_names:
            call_kwargs[key] = value

    input_path = _blank_to_none(args.input_path) or _blank_to_none(args.video_path) or ""
    return JobCreateRequest(
        job_type="inference",
        model_id=args.model_id,
        variant_id=args.variant_id or "",
        task_profile_id=args.task_profile_id or "",
        prompt=args.prompt or "",
        negative_prompt=args.negative_prompt or "",
        input_path=input_path,
        model_ref=args.model_ref or "",
        backend=args.backend or "auto",
        endpoint=args.endpoint or "",
        api_key=args.api_key or "",
        device=args.device or str(SETTINGS.get("device") or "cuda"),
        params=params,
        call_kwargs=call_kwargs,
        load_kwargs=load_kwargs,
    )


def _run_infer(args: argparse.Namespace) -> int:
    payload = _build_infer_payload(args)
    entry, run_kwargs = _inference_run_kwargs(payload)
    workspace_root = str(Path(args.output_dir).expanduser())
    backend = str(run_kwargs.get("backend") or "auto")
    if backend == "auto" and not entry.supports_from_pretrained and entry.supports_api_init:
        backend = "api_init"
    spec = dispatch_spec_for_inference(entry.model_id, backend=backend)
    if spec is not None:
        record = run_manager_payload_in_conda(
            model_id=entry.model_id,
            spec=spec,
            workspace_root=workspace_root,
            run_kwargs=run_kwargs,
            dispatch_root=Path(workspace_root) / "runtime_jobs" / entry.model_id,
        )
    else:
        manager = StudioManager(workspace_root=workspace_root)
        try:
            record = manager.run(**run_kwargs, progress_callback=None)
        finally:
            manager.close()
    print(json.dumps(record.to_manifest(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a WorldFoundry Workspace job from the command line.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    infer = subparsers.add_parser("infer", help="Run an inference job through the shared Studio runtime.")
    infer.add_argument("--model-id", required=True)
    infer.add_argument("--variant-id", default="")
    infer.add_argument("--task-profile-id", default="")
    infer.add_argument("--output-dir", default="runs/tui/infer")
    infer.add_argument("--input-path", default="")
    infer.add_argument("--input-dir", default="")
    infer.add_argument("--video-path", default="")
    infer.add_argument("--trajectory-file", default="")
    infer.add_argument("--prompt", default="")
    infer.add_argument("--negative-prompt", default="")
    infer.add_argument("--interactions", default="")
    infer.add_argument("--task", default="")
    infer.add_argument("--mode", default="")
    infer.add_argument("--resize-mode", default="")
    infer.add_argument("--size", default="")
    infer.add_argument("--frames", default="")
    infer.add_argument("--steps", default="")
    infer.add_argument("--frames-per-generation", default="")
    infer.add_argument("--guidance-scale", default="")
    infer.add_argument("--seed", default="")
    infer.add_argument("--fps", default="")
    infer.add_argument("--dtype", default="")
    infer.add_argument("--max-sequence-length", default="")
    infer.add_argument("--cam-type", default="")
    infer.add_argument("--output-formats", default="")
    infer.add_argument("--trajectory", default="")
    infer.add_argument("--angle", default="")
    infer.add_argument("--distance", default="")
    infer.add_argument("--orbit-radius", default="")
    infer.add_argument("--zoom-ratio", default="")
    infer.add_argument("--alpha-threshold", default="")
    infer.add_argument("--static-scene", default="")
    infer.add_argument("--low-vram", default="")
    infer.add_argument("--disable-lora", default="")
    infer.add_argument("--vis-rendering", default="")
    infer.add_argument("--offload-t5", default="")
    infer.add_argument("--offload-transformer-during-vae", default="")
    infer.add_argument("--offload-vae", default="")
    infer.add_argument("--output-path", default="")
    infer.add_argument("--model-ref", default="")
    infer.add_argument("--backend", default="auto")
    infer.add_argument("--endpoint", default="")
    infer.add_argument("--api-key", default="")
    infer.add_argument("--device", default="")
    infer.add_argument("--call-json", default="")
    infer.add_argument("--load-json", default="")
    infer.set_defaults(func=_run_infer)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except HTTPException as exc:
        print(str(exc.detail), file=sys.stderr)
        return int(exc.status_code or 1)
    except (RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
