"""Unified camera-control entrypoint.

``compile_camera`` is the single function that turns a frame-action camera script
into a model-native payload plus auditable sidecars, for any supported model.
Input kind (image for TI2V, source video for V2V, or no extra media for T2V)
is resolved from the registry, so callers do not branch on the model family.
With ``dry_run=True`` it compiles the payload and writes sidecars without
invoking any heavy model pipeline. Real generation is delegated to
WorldFoundry's canonical in-tree model runner.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from wrbench.actions import CameraScript, parse_camera_script
from wrbench.adapters import compile_camera_payload  # noqa: F401 ensures adapters register
from wrbench.adapters._utils import unified_sidecar_extra
from wrbench.builder import build_camera_trajectory
from wrbench.registry import canonical_model_key, input_kind, model_record
from wrbench.runtime import load_runtime_config


def _require_inputs(
    key: str,
    image: str | None,
    source_video: str | None,
    *,
    prompt: str = "",
    dry_run: bool = True,
) -> None:
    kind = input_kind(key)
    if kind == "image":
        if not image:
            raise ValueError(f"{key} is an image/TI2V model and requires image=...")
    elif kind == "source_video":
        if not source_video:
            raise ValueError(f"{key} is a source-video/V2V model and requires source_video=...")
    elif kind == "none":
        if image or source_video:
            raise ValueError(f"{key} does not use image= or source_video=; pass prompt=... for text conditioning")
        if not dry_run and not str(prompt).strip():
            raise ValueError(f"{key} is a prompt-only model and requires prompt=... for real generation")
    else:
        raise ValueError(f"{key} has unknown input_kind {kind!r}")
    contract = model_record(key).execution_contract or {}
    policy = contract.get("wrbench_policy") if isinstance(contract.get("wrbench_policy"), dict) else {}
    if not dry_run and bool(policy.get("requires_prompt")) and not str(prompt).strip():
        raise ValueError(f"{key} requires prompt=... for real generation")


def _sidecar_metadata(key: str, payload: Any, camera: str, output_path: Path) -> dict[str, Any]:
    record = model_record(key)
    peak = payload.metadata.get("megasam_precision_inputs", {}).get("rotation_peak_signed_deg")
    meta = unified_sidecar_extra(
        payload_metadata=dict(payload.metadata),
        payload_type=payload.payload_type,
        camera_script=camera,
        target_yaw_peak_deg=abs(float(peak)) if peak not in (None, 0) else None,
    )
    meta.update(
        {
            "canonical_model_key": key,
            "adapter": record.adapter,
            "payload_type": payload.payload_type,
            "official_camera_entrypoint": payload.official_camera_entrypoint,
            "coordinate_convention": payload.target_trajectory.coordinate_convention,
            "camera_script": camera,
            "requested_frame_count": payload.metadata.get("frame_mapping", {}).get(
                "requested_frames", payload.target_trajectory.frame_count
            ),
            "model_frame_count": payload.target_trajectory.frame_count,
            "generated_frame_count": payload.target_trajectory.frame_count,
            "target_c2w_path": str(output_path.with_suffix(output_path.suffix + ".target_c2w.npy")),
            "camera_trajectory_json_path": str(output_path.with_suffix(output_path.suffix + ".camera_trajectory.json")),
            "model_control_samples_path": str(output_path.with_suffix(output_path.suffix + ".model_control_samples.json")),
            "calibration_status": payload.calibration_status,
        }
    )
    return meta


def _write_model_control_samples(out: Path, payload: Any) -> str:
    samples_path = out.with_suffix(out.suffix + ".model_control_samples.json")
    samples_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "payload_type": payload.payload_type,
                "official_camera_entrypoint": payload.official_camera_entrypoint,
                "model_control_timeline": payload.metadata.get("model_control_timeline", {}),
                "target_frame_count": payload.target_trajectory.frame_count,
                "target_coordinate_convention": payload.target_trajectory.coordinate_convention,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return str(samples_path)


def _jsonable(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def compile_camera(
    *,
    model: str,
    camera: str | CameraScript,
    out: str | Path,
    image: str | None = None,
    source_video: str | None = None,
    prompt: str = "",
    width: int | None = None,
    height: int | None = None,
    fps: int | None = None,
    num_frames: int | None = None,
    camera_type: str | None = None,
    work_dir: str | Path | None = None,
    runtime_config: str | Path | None = None,
    dry_run: bool = True,
) -> dict[str, Any]:
    """Compile a camera script into a model payload and write sidecars.

    Returns a dict with the canonical model key, the :class:`CameraPayload`, and
    the written artifact paths.
    """

    key = canonical_model_key(model)
    record = model_record(key)
    _require_inputs(key, image, source_video, prompt=prompt, dry_run=dry_run)
    width = int(width if width is not None else record.default_width)
    height = int(height if height is not None else record.default_height)
    fps = int(fps if fps is not None else record.default_fps)
    if isinstance(camera, CameraScript):
        script = camera
        if script.fps != fps:
            script = parse_camera_script(script.to_string(), fps=fps)
        camera_str = script.to_string()
    else:
        camera_str = str(camera)
        script = parse_camera_script(camera_str, fps=fps)
    frames = int(num_frames if num_frames is not None else record.default_frames)
    trajectory = build_camera_trajectory(script, width=width, height=height, fps=fps, camera_type=camera_type)
    payload = compile_camera_payload(
        trajectory,
        model_name=key,
        width=width,
        height=height,
        num_frames=frames,
        work_dir=work_dir or Path(out).parent,
        prompt=prompt,
    )
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_model_control_samples(out_path, payload)
    artifacts = payload.target_trajectory.write_target_artifacts(
        out_path,
        conversion_mode=payload.payload_type,
        target_certification_status="compiled",
        target_certification_basis="wrbench compiled frame action script into model payload",
        extra_sidecar=_sidecar_metadata(key, payload, camera_str, out_path),
    )
    payload_doc = {
        "model": key,
        "camera_script": camera_str,
        "payload_type": payload.payload_type,
        "payload": _jsonable(payload.payload),
        "official_camera_entrypoint": payload.official_camera_entrypoint,
        "target_coordinate_convention": payload.target_trajectory.coordinate_convention,
        "metadata": payload.metadata,
    }
    payload_path = out_path.with_suffix(out_path.suffix + ".payload.json")
    payload_path.write_text(json.dumps(payload_doc, indent=2, sort_keys=True), encoding="utf-8")
    artifacts["payload_json"] = str(payload_path)

    generation: dict[str, Any] | None = None
    if not dry_run:
        from wrbench.backends.base import GenerationRequest
        from wrbench.backends.registry import resolve_backend

        runtime = load_runtime_config(Path(runtime_config)) if runtime_config is not None else None
        backend = resolve_backend(key, runtime=runtime)
        if getattr(backend, "name", "") == "dry_run":
            raise RuntimeError(
                f"Real generation requested for {key!r} but only the dry-run backend is available. "
                "Use the WorldFoundry in-tree backend or set dry_run=True."
            )
        if hasattr(backend, "available_for"):
            ok, reason = backend.available_for(key)  # type: ignore[attr-defined]
        else:
            ok, reason = backend.available()
        if not ok:
            raise RuntimeError(
                f"Real generation requested for {key!r} but backend unavailable: {reason}. "
                "Ensure the model has a runnable WorldFoundry catalog entry or use dry_run=True."
            )
        gen_result = backend.generate(
            GenerationRequest(
                model=key,
                prompt=prompt,
                payload=payload,
                output_path=out_path,
                image_path=Path(image) if image else None,
                source_video_path=Path(source_video) if source_video else None,
                work_dir=Path(work_dir) if work_dir else out_path.parent,
                extra={"camera_script": camera_str},
            )
        )
        generation = {
            "success": gen_result.success,
            "message": gen_result.message,
            "output_path": str(gen_result.output_path) if gen_result.output_path else None,
            "artifacts": dict(gen_result.artifacts),
        }
        if not gen_result.success:
            raise RuntimeError(gen_result.message)

    return {
        "model": key,
        "prompt": prompt,
        "payload": payload,
        "artifacts": artifacts,
        "dry_run": dry_run,
        "generation": generation,
    }
