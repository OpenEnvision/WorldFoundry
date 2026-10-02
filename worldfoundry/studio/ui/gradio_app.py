from __future__ import annotations

import json
import os
import re
import shutil
import time
from html import escape
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image

from worldfoundry.studio.inference.catalog import (
    CatalogEntry,
    catalog_as_table,
    find_entry,
    lingbot_world_fast_load_kwargs,
)
from worldfoundry.studio.inference.config import env_first
from worldfoundry.studio.inference.execution import (
    LINGBOT_VARIANT_FAST,
    TORCHRUN_LINGBOT_FAST_ENV,
    RunRecord,
    StudioManager,
    _is_gaussian_splat_ply,
    _torchrun_rank,
    _torchrun_world_size,
    ensure_torchrun_lingbot_fast_runtime,
    parse_interactions,
    parse_jsonish,
    recent_runs_table,
    shutdown_torchrun_lingbot_fast_runtime,
)
from worldfoundry.studio.inference.variants import (
    LINGBOT_VARIANT_BASE_ACT,
    LINGBOT_VARIANT_BASE_CAM,
)
from worldfoundry.studio.inference.variants import (
    normalize_cli_token as _normalize_cli_token,
)
from worldfoundry.studio.inference.variants import (
    resolve_cli_variant_id as _resolve_shared_cli_variant_id,
)
from worldfoundry.studio.serving import require_auth_token_for_host, token_matches
from worldfoundry.studio.ui import CUSTOM_CSS, HEAD_HTML, SPARK_ROOT, hero_html, profile_html, summary_html
from worldfoundry.studio.ui.catalog import (
    _filter_studio_catalog,
    _studio_catalog,
    _studio_stats,
    _supports_live_controls,
)
from worldfoundry.studio.ui.gradio_runtime import (
    gr,
    gradio_progress,
    install_gradio_patches,
    mask_socks_proxy_env_for_gradio,
)
from worldfoundry.studio.ui.interfaces import interface_spec_for_entry
from worldfoundry.studio.ui.launch_config import StudioLaunchConfig, launch_uses_lingbot_torchrun_rollout
from worldfoundry.studio.ui.launch_config import parse_launch_config as _parse_launch_config_core
from worldfoundry.studio.ui.status import status_block as _status_block
from worldfoundry.studio.ui.tray import (
    DEMO_IMAGE_LIBRARY_FILES,
    DEMO_IMAGE_LIBRARY_ROOT,
    STUDIO_ASSET_DIR,
    TRAY_DEMO_IMAGE_FILES,
    _default_tray_head_html,
    _demo_gallery_items,
    _extract_uploaded_path,
    _load_tray_image_source,
    _logo_nav_markup,
    _on_demo_image_select,
    _on_tray_image_select,
    _tray_gallery_items,
    _use_tray_image_as_input,
    _world_tray_html,
)
from worldfoundry.studio.ui.viewports import (
    _embodied_viewport_for_record,
    _embodied_viewport_idle_html,
    _points_viewport_for_record,
    _points_viewport_idle_html,
    _spatial_stage_for_record,
    _spatial_stage_html,
)
from worldfoundry.studio.visualization.backends.frontends import (
    NATIVE_FRONTENDS,
    host_for_frontend,
    port_for_frontend,
    print_remote_access,
    resolve_frontend_mode,
    serve_native_frontend,
)
from worldfoundry.studio.visualization.backends.viser import STUDIO_VISER
from worldfoundry.studio.visualization.core.capabilities import summarize_routing_hints
from worldfoundry.studio.visualization.studio import prepare_run_record

MANAGER = StudioManager(record_processor=prepare_run_record)
VIDEO_INPUT_PREVIEW_EXTS = {".mp4", ".mov", ".avi", ".webm", ".mkv"}
SPATIAL_ASSET_EXTS = {".ply", ".spz", ".splat", ".ksplat", ".sog"}
LINGBOT_ACTION_SEGMENT_FRAMES = 9
LINGBOT_TOKEN_TO_ACTION_KEYS = {
    "forward": "w",
    "backward": "s",
    "left": "a",
    "right": "d",
    "forward_left": "wa",
    "forward_right": "wd",
    "backward_left": "sa",
    "backward_right": "sd",
    "camera_up": "i",
    "camera_down": "k",
    "camera_l": "j",
    "camera_r": "l",
    "camera_ul": "ij",
    "camera_ur": "il",
    "camera_dl": "kj",
    "camera_dr": "kl",
}


def _pad_frame_count_4n_plus_1(frame_count: int) -> int:
    if frame_count <= 1:
        return 1
    remainder = (frame_count - 1) % 4
    if remainder == 0:
        return frame_count
    return frame_count + (4 - remainder)


def _json_object_or_default(text: str) -> dict[str, Any]:
    parsed = parse_jsonish(text, default={}) or {}
    if not isinstance(parsed, dict):
        raise ValueError("JSON overrides must decode to an object.")
    return dict(parsed)


def _scene_preview_from_inputs(
    image: Any,
    input_path: str,
    last_frame: Any,
) -> tuple[Image.Image | None, str, str]:
    if isinstance(image, Image.Image):
        return image.convert("RGB"), "Main Image", ""

    staged_path = (input_path or "").strip()
    if staged_path:
        preview_image = _load_tray_image_source(staged_path)
        if preview_image is not None:
            return preview_image, "Input Path", staged_path

    if isinstance(last_frame, Image.Image):
        return last_frame.convert("RGB"), "Last Frame", ""

    return None, "", staged_path


def _sync_input_scene_preview(
    model_id: str,
    image: Any,
    video: Any,
    input_path: str,
    last_frame: Any,
):
    preview_image, source_label, source_path = _scene_preview_from_inputs(image, input_path, last_frame)
    video_path = _extract_uploaded_path(video)
    if not video_path and source_path and Path(source_path).suffix.lower() in VIDEO_INPUT_PREVIEW_EXTS:
        video_path = source_path
    entry = None
    try:
        entry = find_entry(model_id)
        display_name = entry.display_name
    except Exception:
        display_name = model_id or "WorldFoundry"

    def ready_copy(media_kind: str) -> tuple[str, str, str]:
        if entry is None:
            return ("input ready", "run when ready", "Run the selected model when the rest of the inputs are ready.")
        template_id = interface_spec_for_entry(entry).template_id
        if template_id == "scene-3d":
            return ("scene source ready", "press BUILD to reconstruct or RENDER to use cached geometry", "Build or render the scene when camera settings are ready.")
        if template_id == "depth-geometry":
            return ("geometry source ready", "press RUN to extract depth or geometry", "Run extraction when the source and JSON settings are ready.")
        if template_id == "embodied-policy":
            return ("observation ready", "press ACT to produce the next policy output", "Run the policy when observation and action-token context are ready.")
        if template_id == "visual-action":
            return ("visual context ready", "press INFER to produce latent action tokens", "Infer action tokens when the context is ready.")
        if template_id == "hosted-api":
            return ("provider input ready", "press CALL after endpoint credentials are set", "Call the provider when credentials and request settings are ready.")
        if template_id in {"conditioned-video", "text-video"}:
            return (f"{media_kind} ready", "press RUN to generate video", "Run generation when the prompt and media conditions are ready.")
        return ("input scene ready", "press INIT or tap WASD / IJKL to start interactive generation", "Use INIT to seed the interactive state, or use live controls to stream one chunk at a time.")

    if preview_image is not None:
        source_value = source_path or source_label
        status_title, status_guidance, summary_guidance = ready_copy("image")
        manifest = {
            "stage": status_title.replace(" ", "_"),
            "model_id": model_id,
            "display_name": display_name,
            "source": source_label,
            "source_path": source_path,
            "guidance": summary_guidance,
        }
        return (
            _status_block(
                f"{status_title} · {display_name}\n"
                f"source={source_value}\n"
                f"{status_guidance}"
            ),
            summary_html(
                status_title.title(),
                display_name,
                pills=("input", "ready"),
                lines=(
                    f"Source: {source_value}.",
                    "The stage now mirrors the currently selected source before generation starts.",
                    summary_guidance,
                ),
            ),
            None,
            preview_image,
            None,
            [],
            manifest,
            None,
        )

    if video_path:
        source_value = video_path
        status_title, status_guidance, summary_guidance = ready_copy("video")
        manifest = {
            "stage": status_title.replace(" ", "_"),
            "model_id": model_id,
            "display_name": display_name,
            "source": "Video",
            "source_path": video_path,
            "guidance": summary_guidance,
        }
        return (
            _status_block(
                f"{status_title} · {display_name}\n"
                f"source={source_value}\n"
                f"{status_guidance}"
            ),
            summary_html(
                status_title.title(),
                display_name,
                pills=("video", "ready"),
                lines=(
                    f"Source: {source_value}.",
                    "The stage now mirrors the currently selected video before generation starts.",
                    summary_guidance,
                ),
            ),
            video_path,
            None,
            None,
            [],
            manifest,
            None,
        )

    if source_path:
        manifest = {
            "stage": "input_source_staged",
            "model_id": model_id,
            "display_name": display_name,
            "source": "Input Path",
            "source_path": source_path,
        }
        return (
            _status_block(
                f"input source staged · {display_name}\n"
                f"path={source_path}\n"
                "this model expects files or folders; run when ready"
            ),
            summary_html(
                "Input Source Staged",
                display_name,
                pills=("path", "ready"),
                lines=(
                    f"Path: {source_path}.",
                    "This source could not be previewed as a single image on the main stage.",
                    "Run the selected model when the rest of the inputs are ready.",
                ),
            ),
            None,
            None,
            None,
            [],
            manifest,
            None,
        )

    return (
        _status_block("idle"),
        _run_overview_html(entry=entry),
        None,
        None,
        None,
        [],
        None,
        None,
    )


def _lingbot_variant_profiles(entry: CatalogEntry) -> dict[str, dict[str, Any]]:
    fast_load_defaults = lingbot_world_fast_load_kwargs()
    fast_model_path = str(fast_load_defaults.get("fast_model_path", "") or "")
    base_model_ref = entry.default_model_ref or ""
    fast_model_ref = fast_model_path or base_model_ref
    fast_load_kwargs = dict(fast_load_defaults)
    fast_load_kwargs.setdefault("runtime_variant", "fast")
    if fast_model_ref == base_model_ref and fast_model_path:
        fast_load_kwargs["fast_model_path"] = fast_model_path
    return {
        LINGBOT_VARIANT_BASE_CAM: {
            "label": "Base Cam",
            "summary": "High-quality camera-pose control with the base checkpoint only.",
            "lines": (
                "Matches the README camera-pose examples more closely than the fast path.",
                "Better for longer guided camera moves than low-latency button tapping.",
                "Studio defaults this variant to shorter chunks so the base model stays usable interactively.",
                "Keeps the standard camera-control token flow in the interaction box.",
            ),
            "prompt": entry.default_prompt,
            "task_type": "",
            "interactions": "forward, left, camera_r",
            "load_kwargs": {"runtime_variant": None, "fast_model_path": None},
            "call_kwargs": {"num_frames": 21, "sampling_steps": 20, "seed": 42},
            "model_ref": base_model_ref,
        },
        LINGBOT_VARIANT_BASE_ACT: {
            "label": "Base Act Preview",
            "summary": "Action-string control for the act2cam path using the base checkpoint.",
            "lines": (
                "Studio converts the current token list into LingBot action-string segments at run time.",
                "Use this when you want WASD / IJKL style control semantics instead of pure camera poses.",
                "You can still override `action_string` manually inside Call Kwargs JSON when needed.",
            ),
            "prompt": entry.default_prompt,
            "task_type": "",
            "interactions": "forward, camera_l",
            "load_kwargs": {"runtime_variant": None, "fast_model_path": None},
            "call_kwargs": {"seed": 42, "allow_act2cam": True, "sampling_steps": 20},
            "model_ref": base_model_ref,
        },
        LINGBOT_VARIANT_FAST: {
            "label": "Fast",
            "summary": "Closest to the LingBot-World-Fast interactive demo path.",
            "lines": (
                "Uses the fast checkpoint as the main launch ref and auto-resolves the base camera sibling.",
                "Defaults to 480P short chunks so step latency stays closer to the realtime demo path.",
                "Hold a key or joystick direction to keep rolling the world forward autoregressively.",
            ),
            "prompt": entry.default_prompt,
            "task_type": "",
            "interactions": "forward",
            "load_kwargs": fast_load_kwargs,
            "call_kwargs": {
                "num_frames": 9,
                "seed": 42,
                "max_area": 480 * 832,
                "offload_model": False,
                "wmfactory_action_controls": True,
            },
            "model_ref": fast_model_ref,
        },
    }


def _model_variant_profiles(entry: CatalogEntry) -> dict[str, dict[str, Any]]:
    if entry.model_id == "lingbot-world":
        return _lingbot_variant_profiles(entry)
    return {}


def _variant_payload(entry: CatalogEntry | None) -> tuple[list[tuple[str, str]], str | None, bool, str]:
    if entry is None:
        return [], None, False, ""
    profiles = _model_variant_profiles(entry)
    if not profiles:
        return [], None, False, ""
    choices = [(profile["label"], variant_id) for variant_id, profile in profiles.items()]
    default_variant = next(iter(profiles))
    return choices, default_variant, True, _variant_summary_html(entry, default_variant)


def _variant_summary_html(entry: CatalogEntry, variant_id: str) -> str:
    profiles = _model_variant_profiles(entry)
    profile = profiles.get(variant_id) or next(iter(profiles.values()), None)
    if profile is None:
        return ""
    return summary_html(
        "Model Variant",
        profile["label"],
        pills=(entry.display_name, "variant"),
        lines=(
            profile["summary"],
            *profile["lines"],
        ),
    )


def _variant_ui_updates(entry: CatalogEntry | None):
    choices, value, visible, notes = _variant_payload(entry)
    del visible
    return gr.update(choices=choices, value=value, visible=False), notes


def _on_variant_change(model_id: str, variant_id: str):
    try:
        entry = find_entry(model_id)
    except Exception:
        return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), ""

    profiles = _model_variant_profiles(entry)
    profile = profiles.get(variant_id) or next(iter(profiles.values()), None)
    if profile is None:
        return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), ""

    return (
        profile["prompt"],
        profile["task_type"],
        profile["interactions"],
        json.dumps(profile["load_kwargs"], indent=2, ensure_ascii=False),
        json.dumps(profile["call_kwargs"], indent=2, ensure_ascii=False),
        profile["model_ref"] or "",
        _variant_summary_html(entry, variant_id),
    )


def _lingbot_action_string_from_interactions(interactions_text: str) -> tuple[str | None, int]:
    parsed = parse_interactions(interactions_text)
    if isinstance(parsed, dict):
        parsed = parsed.get("action_list") or parsed.get("interactions") or parsed.get("actions")
    if parsed is None:
        return None, 0
    if isinstance(parsed, str):
        parsed = [parsed]
    if not isinstance(parsed, (list, tuple)):
        raise ValueError("LingBot Base Act expects the interaction plan to be a token list or JSON array.")

    segments: list[str] = []
    total_frames = 0
    unsupported_tokens: list[str] = []
    for raw_token in parsed:
        token = str(raw_token).strip().lower()
        if not token:
            continue
        keys = LINGBOT_TOKEN_TO_ACTION_KEYS.get(token)
        if keys is None:
            unsupported_tokens.append(token)
            continue
        segments.append(f"{keys}-{LINGBOT_ACTION_SEGMENT_FRAMES}")
        total_frames += LINGBOT_ACTION_SEGMENT_FRAMES

    if unsupported_tokens:
        raise ValueError(
            "LingBot Base Act cannot auto-convert these interaction tokens: "
            + ", ".join(sorted(dict.fromkeys(unsupported_tokens)))
        )

    if not segments:
        return None, 0
    return ",".join(segments), total_frames


def _apply_variant_runtime_overrides(
    model_id: str,
    variant_id: str,
    interactions_text: str,
    load_kwargs_text: str,
    call_kwargs_text: str,
    model_ref: str,
) -> tuple[str, str, str]:
    try:
        entry = find_entry(model_id)
    except Exception:
        return load_kwargs_text, call_kwargs_text, model_ref

    profiles = _model_variant_profiles(entry)
    profile = profiles.get(variant_id)
    if profile is None:
        return load_kwargs_text, call_kwargs_text, model_ref

    load_payload = dict(profile["load_kwargs"])
    load_payload.update(_json_object_or_default(load_kwargs_text))

    call_payload = dict(profile["call_kwargs"])
    call_payload.update(_json_object_or_default(call_kwargs_text))

    effective_model_ref = (model_ref or profile["model_ref"] or entry.default_model_ref or "").strip()

    if variant_id == LINGBOT_VARIANT_BASE_ACT:
        call_payload["allow_act2cam"] = True
        if not call_payload.get("action_string"):
            action_string, total_frames = _lingbot_action_string_from_interactions(interactions_text)
            if action_string:
                call_payload["action_string"] = action_string
                call_payload.setdefault("num_frames", _pad_frame_count_4n_plus_1(total_frames))

    return (
        json.dumps(load_payload, indent=2, ensure_ascii=False),
        json.dumps(call_payload, indent=2, ensure_ascii=False),
        effective_model_ref,
    )


def _resolve_cli_variant_id(entry: CatalogEntry, raw_variant: str | None) -> str | None:
    profiles = _model_variant_profiles(entry)
    if not profiles:
        return _resolve_shared_cli_variant_id(entry, raw_variant)

    if raw_variant is None or not raw_variant.strip():
        return None

    try:
        resolved = _resolve_shared_cli_variant_id(entry, raw_variant)
        if resolved in profiles:
            return resolved
    except ValueError:
        pass

    normalized = _normalize_cli_token(raw_variant)
    if normalized in {"default", "auto"}:
        return next(iter(profiles))

    alias_map: dict[str, str] = {}
    for variant_id, profile in profiles.items():
        candidates = {
            variant_id,
            profile["label"],
            variant_id.replace("lingbot_", ""),
        }
        if entry.model_id == "lingbot-world":
            if variant_id == LINGBOT_VARIANT_FAST:
                candidates.update({"fast", "realtime", "fast-realtime", "fast_realtime", "lingbot-fast"})
            elif variant_id == LINGBOT_VARIANT_BASE_CAM:
                candidates.update({"base-camera", "base_cam", "basecam", "camera", "cam"})
            elif variant_id == LINGBOT_VARIANT_BASE_ACT:
                candidates.update({"base-act", "base_action", "baseact", "act", "action", "act2cam"})
        for candidate in candidates:
            alias_map[_normalize_cli_token(candidate)] = variant_id

    resolved = alias_map.get(normalized)
    if resolved is None:
        raise ValueError(f"Unknown variant `{raw_variant}` for {entry.display_name}.")
    return resolved


def parse_launch_config(
    argv: Sequence[str] | None = None,
    *,
    entries: Sequence[CatalogEntry] | None = None,
) -> StudioLaunchConfig:
    """CLI-facing wrapper around ``launch_config.parse_launch_config`` with Studio defaults."""
    return _parse_launch_config_core(
        argv,
        entries=entries,
        studio_catalog=_studio_catalog,
        resolve_cli_variant_id=_resolve_cli_variant_id,
    )


def _launch_defaults(
    launch_config: StudioLaunchConfig,
    catalog: Sequence[CatalogEntry],
) -> tuple[CatalogEntry, tuple[CatalogEntry, ...], list[Any], str, str]:
    del catalog
    entry = find_entry(launch_config.model_id)
    active_entries = (entry,)
    default_state = list(_selection_state(entry.model_id, "", "All", active_entries))
    _, default_variant_value, _, variant_notes = _variant_payload(entry)
    resolved_variant = launch_config.variant_id or default_variant_value or ""

    if resolved_variant:
        (
            default_state[5],
            default_state[6],
            default_state[7],
            default_state[8],
            default_state[9],
            default_state[10],
            variant_notes,
        ) = _on_variant_change(entry.model_id, resolved_variant)

    if launch_config.model_ref:
        default_state[10] = launch_config.model_ref
    if launch_config.backend:
        default_state[11] = launch_config.backend
    if launch_config.endpoint:
        default_state[12] = launch_config.endpoint

    return entry, active_entries, default_state, resolved_variant, variant_notes


def _load_spatial_asset(uploaded: Any) -> tuple[str, str]:
    raw_path = _extract_uploaded_path(uploaded)
    if not raw_path:
        return _spatial_stage_for_record()

    src_path = Path(raw_path).expanduser().resolve()
    suffix = src_path.suffix.lower()
    if suffix not in SPATIAL_ASSET_EXTS:
        return (
            _spatial_stage_html(
                note="Unsupported spatial asset. Use a Gaussian Splat export such as .ply, .spz, .splat, .ksplat, or .sog."
            ),
            f"Unsupported spatial asset format: {suffix or 'unknown'}",
        )
    if suffix == ".ply" and not _is_gaussian_splat_ply(src_path):
        return (
            _spatial_stage_html(
                note="Unsupported .ply asset. Spark only accepts Gaussian Splat PLY exports with opacity, scale, SH, and rotation channels."
            ),
            f"Unsupported .ply spatial asset: {src_path.name} is not a Gaussian Splat export.",
        )

    uploads_dir = Path(MANAGER.workspace_root) / "spatial_uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    staged_path = uploads_dir / f"{int(time.time() * 1000)}-{src_path.name}"
    shutil.copy2(src_path, staged_path)
    title = src_path.stem.replace("-", " ").strip() or "Imported World"
    return (
        _spatial_stage_html(
            title=title,
            splat_path=str(staged_path),
            note="Imported Gaussian Splat ready for exploration with drag, wheel, and WASD controls.",
        ),
        f"Imported spatial asset · {staged_path.name}",
    )


def _clear_spatial_asset() -> tuple[str, str]:
    return _spatial_stage_for_record()


def _ensure_localhost_no_proxy() -> None:
    hosts = ["127.0.0.1", "localhost", "0.0.0.0"]
    merged = []
    for key in ("NO_PROXY", "no_proxy"):
        current = os.getenv(key, "")
        parts = [part.strip() for part in current.split(",") if part.strip()]
        for host in hosts:
            if host not in parts:
                parts.append(host)
        merged_value = ",".join(parts)
        os.environ[key] = merged_value
        merged = parts
    if merged:
        os.environ["NO_PROXY"] = ",".join(merged)
        os.environ["no_proxy"] = os.environ["NO_PROXY"]




def _entry_profile(entry: CatalogEntry) -> tuple[str, str, str, str, str, str, str]:
    interface_spec = interface_spec_for_entry(entry)
    pills = [
        entry.category,
        entry.family,
        interface_spec.template_id,
        "stream" if entry.supports_stream else "one-shot",
        entry.default_backend,
    ]
    if interface_spec.local_repo.status == "present":
        pills.append("local source")
    if interface_spec.profile_available:
        pills.append("runtime profile")
    if entry.default_model_ref:
        ref_path = Path(entry.default_model_ref).expanduser()
        pills.append("local ckpt" if ref_path.exists() else "repo ref")
    load_json = (
        json.dumps(entry.default_load_kwargs, indent=2, ensure_ascii=False)
        if entry.default_load_kwargs
        else "{}"
    )
    call_json = (
        json.dumps(entry.default_call_kwargs, indent=2, ensure_ascii=False)
        if entry.default_call_kwargs
        else "{}"
    )
    interactions = ", ".join(entry.default_interactions)
    notes = entry.notes
    if interface_spec.local_repo.path:
        notes = "\n".join(part for part in (notes, f"Local source: {interface_spec.local_repo.path}") if part)
    if interface_spec.gui_refs:
        notes = "\n".join(part for part in (notes, f"GUI reference: {interface_spec.gui_refs[0]}") if part)
    return (
        profile_html(entry.display_name, entry.summary, pills, notes=notes),
        entry.default_prompt,
        entry.default_task_type,
        interactions,
        load_json,
        call_json,
        entry.default_model_ref or "",
    )




def _category_slug(entry: CatalogEntry | None) -> str:
    if entry is None:
        return "none"
    return re.sub(r"[^a-z0-9]+", "-", entry.category.lower()).strip("-") or "model"


def _uses_state_init(entry: CatalogEntry | None) -> bool:
    template_id = interface_spec_for_entry(entry).template_id if entry is not None else ""
    return bool(
        entry
        and entry.supports_stream
        and entry.runtime_kind == "default"
        and template_id == "interactive-world"
        and "state-init" in entry.tags
    )


def _category_frontend_profile(entry: CatalogEntry) -> dict[str, Any]:
    interface_spec = interface_spec_for_entry(entry)
    template_id = interface_spec.template_id
    params = set(entry.call_params) | set(entry.stream_params)
    is_gen3c = entry.model_id == "gen3c"
    profile: dict[str, Any] = {
        "slug": _category_slug(entry),
        "template_id": interface_spec.template_id,
        "mode_title": "Video World",
        "mode_copy": "Prompt and condition media feed a visual world rollout.",
        "prompt_label": "Prompt",
        "prompt_placeholder": "describe the world, scene, or instruction",
        "prompt_lines": 4,
        "prompt_visible": "prompt" in params or template_id in {"interactive-world", "conditioned-video", "text-video", "hosted-api", "embodied-policy", "visual-action"},
        "actions_label": "Actions",
        "actions_placeholder": "forward, left, camera_r",
        "actions_visible": template_id not in {"conditioned-video", "text-video", "hosted-api", "depth-geometry"} and bool(entry.default_interactions or "interactions" in params or "interaction_signal" in params),
        "image_label": "Image",
        "image_visible": bool(
            {"images", "image", "image_path", "data_path"} & params
            or entry.runtime_kind in {"two_stage_3dgs", "pointcloud_nav", "worldfm"}
            or template_id in {"depth-geometry", "embodied-policy", "visual-action"}
        ),
        "video_label": "Video",
        "video_visible": bool({"video", "videos", "video_path"} & params or template_id in {"visual-action", "embodied-policy"}),
        "path_label": "Path",
        "path_placeholder": "local path when the runtime expects files or folders",
        "path_visible": bool({"input_path", "data_path", "image_path", "video_path"} & params or template_id in {"scene-3d", "depth-geometry"}),
        "advanced_run_visible": bool(entry.default_task_type or entry.suggested_task_types or entry.category == "3D Scene" or is_gen3c),
        "advanced_run_open": False,
        "task_visible": bool(entry.default_task_type or entry.suggested_task_types or entry.category == "3D Scene"),
        "more_inputs_visible": template_id in {"embodied-policy", "visual-action"},
        "more_inputs_open": False,
        "camera_panel_visible": is_gen3c or template_id == "scene-3d" or entry.runtime_kind in {"two_stage_3dgs", "pointcloud_nav", "worldfm"},
        "camera_panel_open": False,
        "spatial_panel_visible": template_id == "scene-3d",
        "spatial_panel_open": False,
        "runtime_advanced_visible": template_id == "hosted-api",
        "runtime_advanced_open": False,
        "endpoint_visible": template_id == "hosted-api",
        "api_key_visible": template_id == "hosted-api",
        "fps_frames_visible": template_id in {"interactive-world", "conditioned-video", "text-video", "scene-3d", "hosted-api", "visual-action", "embodied-policy"},
        "json_visible": template_id == "hosted-api",
    }

    if template_id == "scene-3d":
        profile.update(
            {
                "mode_title": "3D Scene",
                "mode_copy": "Reconstruct or attach a scene, then render camera views or trajectories.",
                "advanced_run_open": True,
                "prompt_placeholder": "optional scene caption",
                "actions_label": "Camera Tokens",
                "actions_placeholder": "forward, left, camera_zoom_in",
                "image_label": "Source Image",
                "video_label": "Source Video",
                "path_label": "Scene Path",
                "path_placeholder": "image, folder, or video path for reconstruction",
                "camera_panel_open": True,
                "spatial_panel_open": True,
            }
        )
    elif template_id == "depth-geometry":
        profile.update(
            {
                "mode_title": "Depth / Geometry",
                "mode_copy": "Image or video input is routed as a file path for geometry extraction.",
                "prompt_visible": False,
                "actions_visible": False,
                "image_label": "Depth Image",
                "video_label": "Depth Video",
                "video_visible": True,
                "path_label": "Data Path",
                "path_placeholder": "image, video, directory, or txt list",
            }
        )
    elif template_id == "embodied-policy":
        profile.update(
            {
                "mode_title": "Embodied Policy",
                "mode_copy": "Instruction, robot observation, and action tokens produce inspectable policy artifacts.",
                "prompt_label": "Instruction",
                "prompt_placeholder": "put the object on the target area",
                "actions_label": "Action Tokens",
                "actions_placeholder": '{"robot_action": [0, 0, 0, 0, 0, 0, 1]}',
                "image_label": "Observation Image",
                "video_label": "Observation Video",
                "video_visible": True,
                "path_label": "Episode Path",
                "path_visible": True,
                "more_inputs_open": True,
                "runtime_advanced_visible": True,
                "runtime_advanced_open": True,
                "json_visible": True,
            }
        )
    elif template_id == "visual-action":
        profile.update(
            {
                "mode_title": "Visual Action",
                "mode_copy": "Frame or video context is routed to latent-action inference, returning token artifacts.",
                "prompt_label": "Instruction",
                "prompt_placeholder": "optional caption or visual-action instruction",
                "actions_label": "Latent Tokens",
                "actions_placeholder": '{"latent_action_tokens": [12, 48, 7]}',
                "image_label": "Context Frame",
                "video_label": "Context Video",
                "video_visible": True,
                "path_label": "Episode Path",
                "path_visible": True,
                "more_inputs_open": True,
                "runtime_advanced_visible": True,
                "runtime_advanced_open": True,
                "json_visible": True,
            }
        )
    elif template_id == "hosted-api":
        profile.update(
            {
                "mode_title": "Hosted API",
                "mode_copy": "Prompt and optional media are sent through the selected provider endpoint.",
                "actions_visible": False,
                "image_label": "Condition Image",
                "video_label": "Condition Video",
                "video_visible": bool({"video", "videos", "video_path", "audio"} & params),
                "path_visible": False,
                "runtime_advanced_open": True,
            }
        )
    elif entry.model_id == "longvie-2":
        profile.update(
            {
                "mode_title": "Queued Long-Video Segments",
                "mode_copy": (
                    "Generate one full-quality 81-frame segment per request. This is queued "
                    "diffusion generation, not WASD realtime; EXTEND preserves the previous "
                    "final frame, eight-frame history, and noise. Upload a dense depth segment "
                    "and its matching sparse pointmap/track segment in the two video fields."
                ),
                "advanced_run_open": True,
                "image_label": "Initial Image (first segment)",
                "video_label": "Dense Depth Control Segment",
                "video_visible": True,
                "runtime_advanced_visible": True,
                "runtime_advanced_open": True,
                "json_visible": True,
            }
        )
    elif is_gen3c:
        profile.update(
            {
                "mode_title": "Camera Workbench",
                "mode_copy": "Seed an image or preprocessed video, author a camera path, then generate the next world video.",
                "advanced_run_open": True,
                "actions_label": "Camera Tokens",
                "actions_placeholder": "forward, left, camera_r",
                "image_label": "Seed Image",
                "video_label": "Seed Video",
                "video_visible": True,
                "path_label": "Preprocessed Seed Path",
                "path_visible": True,
                "path_placeholder": "GEN3C image, video, or preprocessed folder",
                "camera_panel_open": True,
            }
        )
    elif "navigation" in entry.tags or "camera-control" in entry.tags:
        profile.update(
            {
                "mode_title": "Interactive Video",
                "mode_copy": "A seed image starts the world state; short camera tokens continue it.",
                "image_label": "Seed Image",
            }
        )

    return profile


def _frontend_mode_html(entry: CatalogEntry) -> str:
    profile = _category_frontend_profile(entry)
    interface_spec = interface_spec_for_entry(entry)
    repo_label = "local source present" if interface_spec.local_repo.status == "present" else "local source missing"
    output_label = ", ".join(interface_spec.output_groups[:2]) or "artifact"
    gui_label = ", ".join(interface_spec.gui_refs[:2]) if interface_spec.gui_refs else "none"
    return summary_html(
        "Run Mode",
        profile["mode_title"],
        pills=(entry.category, interface_spec.template_id, "stream" if entry.supports_stream else "one-shot"),
        lines=(
            profile["mode_copy"],
            _recommended_input_text(entry),
            f"Outputs: {output_label}.",
            f"Local source: {repo_label}.",
            f"GUI integration: {gui_label}.",
        ),
    )


def _interface_contract_html(entry: CatalogEntry) -> str:
    interface_spec = interface_spec_for_entry(entry)
    input_text = ", ".join(interface_spec.input_groups) or "n/a"
    output_text = ", ".join(interface_spec.output_groups) or "n/a"
    repo = interface_spec.local_repo
    repo_text = repo.path if repo.path else "No matching local source under the configured model root."
    entrypoints = ", ".join(repo.entrypoints) if repo.entrypoints else "none detected"
    env_files = ", ".join(repo.env_files) if repo.env_files else "none detected"
    gui_entrypoints = ", ".join(repo.gui_entrypoints) if repo.gui_entrypoints else "none detected"
    gui_refs = ", ".join(interface_spec.gui_refs) if interface_spec.gui_refs else "none"
    launch_hints = " | ".join(interface_spec.launch_hints) if interface_spec.launch_hints else "none"
    runtime_status = interface_spec.runtime_status or "inferred from Studio catalog"
    return summary_html(
        "Interface Contract",
        interface_spec.template_title,
        pills=(
            interface_spec.interaction_model,
            interface_spec.artifact_kind or "catalog-output",
            repo.status,
        ),
        lines=(
            f"Inputs: {input_text}.",
            f"Outputs: {output_text}.",
            f"Runtime status: {runtime_status}.",
            f"Local source: {repo_text}.",
            f"Repo entrypoints: {entrypoints}.",
            f"Env files: {env_files}.",
            f"GUI refs: {gui_refs}.",
            f"GUI entrypoints: {gui_entrypoints}.",
            f"Launch hints: {launch_hints}.",
        ),
    )


def _template_workbench_spec(entry: CatalogEntry) -> tuple[str, tuple[tuple[str, str], ...], str]:
    spec = interface_spec_for_entry(entry)
    template_id = spec.template_id
    if template_id == "scene-3d":
        return (
            "3D Scene Workbench",
            (
                ("Source", "image/video/path"),
                ("Camera", "pose + path JSON"),
                ("Viewer", "Spark 3DGS"),
                ("Output", "splat + trajectory"),
            ),
            "BUILD / RENDER",
        )
    if template_id == "depth-geometry":
        return (
            "Geometry Extraction Bench",
            (
                ("Source", "image/video/folder"),
                ("Batch", "data path + JSON"),
                ("Preview", "depth / geometry"),
                ("Output", "maps + files"),
            ),
            "RUN",
        )
    if template_id == "embodied-policy":
        return (
            "Embodied Policy Console",
            (
                ("Instruction", "language goal"),
                ("Observation", "image/video/state"),
                ("Policy", "action tokens"),
                ("Replay", "sim video + trace"),
            ),
            "ACT / NEXT",
        )
    if template_id == "visual-action":
        return (
            "Visual Action Console",
            (
                ("Context", "frame/video"),
                ("History", "latent tokens"),
                ("Inference", "token stream"),
                ("Output", "action JSON"),
            ),
            "INFER / NEXT",
        )
    if template_id == "hosted-api":
        return (
            "Hosted API Console",
            (
                ("Prompt", "provider request"),
                ("Media", "condition assets"),
                ("Endpoint", "provider config"),
                ("Output", "artifact manifest"),
            ),
            "CALL",
        )
    if template_id in {"conditioned-video", "text-video", "video-to-video"}:
        bench_title = "Video-to-Video Bench" if template_id == "video-to-video" else "Video Generation Bench"
        return (
            bench_title,
            (
                ("Prompt", "text guidance"),
                ("Condition", "source video / trajectory"),
                ("Runtime", "frames + fps"),
                ("Output", "video + gallery"),
            ),
            "RUN / EXTEND",
        )
    return (
        "Interactive World Bench",
        (
            ("Seed", "image/video"),
            ("State", "cached rollout"),
            ("Control", "camera tokens"),
            ("Output", "world video"),
        ),
        "INIT / STEP",
    )


def _template_workbench_html(entry: CatalogEntry) -> str:
    title, lanes, action = _template_workbench_spec(entry)
    spec = interface_spec_for_entry(entry)
    lane_html = "\n".join(
        f'<div class="wa-workbench-lane"><span>{escape(label)}</span><strong>{escape(value)}</strong></div>'
        for label, value in lanes
    )
    return f"""
<section class="wa-template-workbench wa-template-workbench-{escape(spec.template_id, quote=True)}">
  <div class="wa-workbench-head">
    <span>{escape(spec.template_id)}</span>
    <strong>{escape(title)}</strong>
  </div>
  <div class="wa-workbench-lanes">
    {lane_html}
  </div>
  <div class="wa-workbench-action">{escape(action)}</div>
</section>
"""


def _recommended_input_text(entry: CatalogEntry) -> str:
    template_id = interface_spec_for_entry(entry).template_id
    if entry.model_id == "gen3c":
        return "seed image or preprocessed video plus editable camera path JSON"
    if template_id == "embodied-policy":
        return "instruction plus observation image/video, optional episode path, and action-token context"
    if template_id == "visual-action":
        return "context frame or video plus optional latent-action token history"
    if template_id == "depth-geometry":
        return "image, video, folder, or data path"
    if entry.runtime_kind == "two_stage_3dgs":
        return "single image or image path"
    if entry.runtime_kind == "pointcloud_nav":
        return "single image, image folder, or video"
    if entry.runtime_kind == "worldfm":
        return "single image plus K intrinsics, or a WorldFM meta file"
    if template_id == "hosted-api":
        return "prompt plus provider-specific media inputs when supported"
    if "videos" in entry.call_params or "video_path" in entry.call_params:
        return "prompt with optional image or video condition"
    if "images" in entry.call_params or entry.default_interactions:
        return "single image with optional prompt"
    return "prompt or files accepted by the pipeline signature"


def _follow_up_text(entry: CatalogEntry) -> str:
    template_id = interface_spec_for_entry(entry).template_id
    if entry.model_id == "gen3c":
        return "reuse the seeded 3D cache, edit camera keyframes, and generate another trajectory"
    if template_id == "embodied-policy":
        return "continue policy inference with the next observation and action-token chunk"
    if template_id == "visual-action":
        return "continue latent-action inference with the next frame or video context"
    if entry.runtime_kind == "two_stage_3dgs":
        return "run once to reconstruct, then keep exploring with orbit or camera tokens"
    if entry.runtime_kind == "pointcloud_nav":
        return "switch to render_view or render_trajectory without rerunning reconstruction"
    if entry.runtime_kind == "worldfm":
        return "reuse cached scene context and stream new pose targets"
    if entry.supports_stream and entry.default_interactions:
        return "continue from the current state with short interaction tokens"
    if entry.supports_stream:
        return "continue from cached state with new text or media conditions"
    return "launch fresh variants with different prompt, task, or input media"


def _scope_label(search: str, category: str) -> str:
    scope_bits = []
    if category and category != "All":
        scope_bits.append(category)
    if search:
        scope_bits.append(f"search: {search}")
    return " · ".join(scope_bits) if scope_bits else "All runnable pipelines"


def _atlas_overview_html(
    filtered_entries: Sequence[CatalogEntry],
    model_id: str,
    search: str,
    category: str,
) -> str:
    stats = _studio_stats()
    try:
        selected_title = find_entry(model_id).display_name if model_id else "Awaiting selection"
    except KeyError:
        selected_title = "Awaiting selection"
    return f"""
<section class="wa-atlas-card">
  <div class="wa-atlas-kicker">World Model Catalog</div>
  <div class="wa-atlas-title-row">
    <h4>WorldFoundry Atlas</h4>
    <span>{len(filtered_entries)} in scope</span>
  </div>
  <p class="wa-atlas-copy">
    Unified demo surface for video worlds, 3D reconstruction, geometry extraction, action policies, and hosted API-backed runs.
  </p>
  <div class="wa-atlas-metrics">
    <div class="wa-atlas-metric"><strong>{stats.get("total", 0)}</strong><span>Pipelines</span></div>
    <div class="wa-atlas-metric"><strong>{stats.get("stream", 0)}</strong><span>Stream</span></div>
    <div class="wa-atlas-metric"><strong>{stats.get("video", 0)}</strong><span>Video</span></div>
    <div class="wa-atlas-metric"><strong>{stats.get("remote", 0)}</strong><span>API</span></div>
  </div>
  <div class="wa-atlas-focus">
    <div class="wa-atlas-focus-label">Current Scope</div>
    <strong>{escape(_scope_label(search, category))}</strong>
  </div>
  <div class="wa-atlas-focus">
    <div class="wa-atlas-focus-label">Selected Model</div>
    <strong>{escape(selected_title)}</strong>
  </div>
</section>
"""


def _workflow_summary_html(entry: CatalogEntry) -> str:
    interface_spec = interface_spec_for_entry(entry)
    defaults = ", ".join(entry.default_interactions) if entry.default_interactions else "none"
    tasks = entry.task_suggestions_text() or "default"
    return summary_html(
        "Launch Pattern",
        interface_spec.template_title,
        pills=(entry.category, interface_spec.template_id, "stream" if entry.supports_stream else "one-shot"),
        lines=(
            f"Primary inputs: {', '.join(interface_spec.input_groups)}.",
            f"Primary outputs: {', '.join(interface_spec.output_groups)}.",
            f"Best follow-up: {_follow_up_text(entry)}.",
            f"Suggested task types: {tasks}.",
            f"Default interaction tokens: {defaults}.",
        ),
    )


def _capability_summary_html(entry: CatalogEntry) -> str:
    interface_spec = interface_spec_for_entry(entry)
    call_args = ", ".join(entry.call_params) or "n/a"
    stream_args = ", ".join(entry.stream_params) or "n/a"
    load_args = ", ".join(entry.load_params) or "n/a"
    tags = ", ".join(entry.tags) or "n/a"
    aliases = ", ".join(entry.aliases) or "n/a"
    repo_path = interface_spec.local_repo.path or "not found"
    runtime_status = interface_spec.runtime_status or "catalog inferred"
    gui_refs = ", ".join(interface_spec.gui_refs) or "none"
    return summary_html(
        "Runtime Surface",
        "Driver contract and exposed runtime controls.",
        pills=(entry.default_backend, entry.runtime_kind, interface_spec.repo_status),
        lines=(
            f"Load args: {load_args}.",
            f"Call args: {call_args}.",
            f"Stream args: {stream_args}.",
            f"Runtime status: {runtime_status}.",
            f"Local source path: {repo_path}.",
            f"GUI references: {gui_refs}.",
            f"Tags: {tags}.",
            f"Aliases: {aliases}.",
        ),
    )


def _run_button_label(entry: CatalogEntry | None) -> str:
    template_id = interface_spec_for_entry(entry).template_id if entry is not None else ""
    if entry is not None and _uses_state_init(entry):
        return "INIT"
    if entry is not None and template_id == "scene-3d":
        return "BUILD"
    if entry is not None and template_id == "hosted-api":
        return "CALL"
    if entry is not None and template_id == "embodied-policy":
        return "ACT"
    if entry is not None and template_id == "visual-action":
        return "INFER"
    return "RUN"


def _stream_button_update(entry: CatalogEntry | None):
    template_id = interface_spec_for_entry(entry).template_id if entry is not None else ""
    if entry is not None and template_id == "scene-3d":
        return gr.update(value="RENDER", visible=bool(entry.supports_stream))
    if entry is not None and template_id in {"embodied-policy", "visual-action"}:
        return gr.update(value="NEXT", visible=True)
    if entry is not None and template_id in {"conditioned-video", "text-video"}:
        return gr.update(value="EXTEND", visible=bool(entry.supports_stream))
    return gr.update(value="STEP", visible=bool(entry and entry.supports_stream))


def _live_controls_bridge_update(entry: CatalogEntry | None):
    return gr.update(visible=bool(entry and _supports_live_controls(entry)))


def _resolve_primary_action(model_id: str, interactions_text: str) -> str:
    del interactions_text
    try:
        entry = find_entry(model_id)
    except Exception:
        return "run"

    return "init" if _uses_state_init(entry) else "run"


def _studio_header(
    filtered_entries: Sequence[CatalogEntry],
    model_id: str,
    search: str,
    category: str,
) -> str:
    try:
        selected_title = find_entry(model_id).display_name if model_id else "No model selected"
    except KeyError:
        selected_title = "No model selected"
    return hero_html(
        stats=_studio_stats(),
        filtered_total=len(filtered_entries),
        selected_title=selected_title,
        category=category,
        search=search,
    )


def _selection_state(
    model_id: str,
    search: str = "",
    category: str = "All",
    filtered_entries: Sequence[CatalogEntry] | None = None,
):
    active_entries = (
        tuple(filtered_entries)
        if filtered_entries is not None
        else _filter_studio_catalog(search, category)
    )
    header = _studio_header(active_entries, model_id, search, category)
    atlas_overview = _atlas_overview_html(active_entries, model_id, search, category)

    if not model_id:
        return (
            header,
            atlas_overview,
            profile_html("No model", "Adjust the current filters to select a pipeline.", ["empty state"]),
            summary_html(
                "Selection Guide",
                "Filter by name, family, capability, or category.",
                pills=("catalog",),
                lines=(
                    "Use the search box to jump to a specific pipeline.",
                    "Switch category lanes to separate open-source video generation from hosted API-backed models.",
                ),
            ),
            summary_html(
                "Runtime Surface",
                "No pipeline is currently selected.",
                pills=("idle",),
                lines=("Pick a model to inspect its runtime contract and runtime knobs.",),
            ),
            "",
            "",
            "",
            "{}",
            "{}",
            "",
            "auto",
            "",
            _live_controls_bridge_update(None),
            _stream_button_update(None),
            gr.update(value=_run_button_label(None)),
            *_variant_ui_updates(None),
        )

    try:
        entry = find_entry(model_id)
    except KeyError:
        return (
            header,
            atlas_overview,
            profile_html("Missing model", "The selected model id is no longer available.", ["stale state"]),
            summary_html(
                "Selection Guide",
                "Refresh the catalog or choose another model.",
                pills=("stale",),
                lines=("The previous selection no longer exists in the active catalog snapshot.",),
            ),
            summary_html(
                "Runtime Surface",
                "Model lookup failed.",
                pills=("error",),
                lines=("A valid model selection is required before runtime details can be shown.",),
                variant="danger",
            ),
            "",
            "",
            "",
            "{}",
            "{}",
            "",
            "auto",
            "",
            _live_controls_bridge_update(None),
            _stream_button_update(None),
            gr.update(value=_run_button_label(None)),
            *_variant_ui_updates(None),
        )

    profile, prompt, task_type, interactions, load_json, call_json, model_ref = _entry_profile(entry)
    return (
        header,
        atlas_overview,
        profile,
        _workflow_summary_html(entry),
        _capability_summary_html(entry),
        prompt,
        task_type,
        interactions,
        load_json,
        call_json,
        model_ref,
        entry.default_backend,
        entry.default_endpoint,
        _live_controls_bridge_update(entry),
        _stream_button_update(entry),
        gr.update(value=_run_button_label(entry)),
        *_variant_ui_updates(entry),
    )


def _idle_run_overview_lines(entry: CatalogEntry | None) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    if entry is None:
        return (
            "No Run Yet",
            (
                "Use `RUN` for one-shot inference, or `INIT` to seed the first interactive world state.",
                "Use the secondary stream control, or hold WASD / IJKL when available, to continue one interaction chunk at a time.",
            ),
            ("idle",),
        )

    spec = interface_spec_for_entry(entry)
    if spec.template_id == "scene-3d":
        return (
            "No Scene Build Yet",
            (
                "Provide source media or a scene path, then build the reconstruction.",
                "Use the Camera Path JSON and 3DGS import panel for trajectory-oriented inspection.",
            ),
            ("idle", "3d"),
        )
    if spec.template_id == "depth-geometry":
        return (
            "No Geometry Run Yet",
            (
                "Provide an image, video, folder, or data path for depth and geometry extraction.",
                "Outputs land in the image, gallery, and artifact tabs.",
            ),
            ("idle", "geometry"),
        )
    if spec.template_id == "embodied-policy":
        return (
            "No Policy Action Yet",
            (
                "Provide instruction and observation context, then run the policy.",
                "Structured action traces, simulator videos, and episode metadata populate Embodied Sim.",
            ),
            ("idle", "policy"),
        )
    if spec.template_id == "visual-action":
        return (
            "No Action Tokens Yet",
            (
                "Provide frame or video context, then infer latent action tokens.",
                "The artifact panel carries the structured token trace.",
            ),
            ("idle", "visual-action"),
        )
    if spec.template_id == "hosted-api":
        return (
            "No Provider Call Yet",
            (
                "Provide provider endpoint credentials and request payload.",
                "Outputs are captured as provider artifacts plus a manifest.",
            ),
            ("idle", "api"),
        )
    if spec.template_id in {"conditioned-video", "text-video"}:
        return (
            "No Video Run Yet",
            (
                "Provide prompt and condition media when supported, then run generation.",
                "Video, image, gallery, and manifest tabs are populated from returned artifacts.",
            ),
            ("idle", "video"),
        )
    return (
        "No World State Yet",
        (
            "Seed the current world state before streaming continuation steps.",
            "Camera tokens and live controls apply only to stream-capable interactive world models.",
        ),
        ("idle", "interactive"),
    )


def _run_overview_html(
    record: RunRecord | None = None,
    *,
    entry: CatalogEntry | None = None,
    stage: str = "idle",
    message: str = "",
) -> str:
    if record is None:
        if stage == "preparing":
            title = message or "Loading the pipeline and normalizing current inputs."
            return f"""
<section class="wa-summary-card wa-progress-card">
  <h4>Preparing Run</h4>
  <div class="wa-summary-subtitle">{escape(title)}</div>
  <div class="wa-summary-pills">
    <span class="wa-summary-pill">working</span>
    <span class="wa-summary-pill">loading</span>
  </div>
  <div class="wa-progress-shell" aria-label="Run loading progress">
    <div class="wa-progress-bar is-indeterminate"></div>
  </div>
  <div class="wa-summary-lines">
    <div>Stages: normalize inputs · restore cached weights · bind the Studio action · materialize previews.</div>
    <div>Weights and caches are reused when possible; streamed worlds stay hot between STREAM steps.</div>
    <div>Outputs populate preview, gallery, artifact, and spatial tabs.</div>
  </div>
</section>
"""
        if stage == "error":
            return summary_html(
                "Run Failed",
                message or "The request could not be completed.",
                pills=("error",),
                lines=("Check the status block for the exception type and message.",),
                variant="danger",
            )
        if stage == "cache":
            return summary_html(
                "Studio State Updated",
                message,
                pills=("cache",),
                lines=("Cached pipelines and recent-run records stay available in the side panels.",),
            )
        title, lines, pills = _idle_run_overview_lines(entry)
        return summary_html(
            title,
            "Awaiting inputs for the selected interface.",
            pills=pills,
            lines=lines,
        )

    request = record.metadata.get("request", {})
    pills = [record.mode, record.status, f"{len(record.artifacts)} artifacts"]
    if record.preview_video:
        pills.append("video")
    if record.preview_image:
        pills.append("image")
    if record.preview_splat:
        pills.append("3DGS")
    if record.preview_model:
        pills.append("3D")
    if record.rrd_path:
        pills.append("rerun")

    task_text = request.get("task_type") or "default"
    backend_text = request.get("backend") or "auto"
    lines = [
        f"Task: {task_text}",
        f"Backend: {backend_text}",
        f"Run folder: {Path(record.output_dir).name}",
    ]
    if record.gallery:
        lines.append(f"Gallery items: {len(record.gallery)}")

    perf = record.metadata.get("studio_performance") if isinstance(record.metadata, dict) else None
    if isinstance(perf, dict) and perf:
        perf_bits: list[str] = []
        mapping = (
            ("prepare_inputs_ms", "normalize"),
            ("load_pipeline_ms", "load"),
            ("execute_ms", "infer"),
            ("torchrun_execute_ms", "torchrun"),
            ("total_client_ms", "wall"),
        )
        for key, label in mapping:
            if key in perf:
                perf_bits.append(f"{label} {perf[key]:,.1f} ms")
        if perf_bits:
            lines.append("Timings: " + " · ".join(perf_bits))

    interactive = record.metadata.get("studio_interactive") if isinstance(record.metadata, dict) else None
    if isinstance(interactive, dict):
        preview = interactive.get("interaction_tokens_preview")
        if preview and preview != "—":
            lines.append(f"Interaction plan: {preview}")
        step_kind = interactive.get("step_kind")
        if step_kind:
            lines.append(f"Studio step: {step_kind}")
        if interactive.get("streaming_model") is True:
            ready = interactive.get("memory_ready_after_step")
            if ready is True:
                if record.model_id == "longvie-2":
                    lines.append(
                        "Segment continuation is resident; upload the next dense/sparse controls and use EXTEND."
                    )
                else:
                    lines.append("Streaming memory is live; use STREAM for the next navigation chunk.")
            elif ready is False and record.mode == "stream":
                if record.model_id == "longvie-2":
                    lines.append("No resident LongVie segment state is available; run the first segment before EXTEND.")
                else:
                    lines.append(
                        "Streaming memory is not seeded; run INIT or provide a seed frame before STREAM continues."
                    )

    routing_hint = summarize_routing_hints(record)
    if routing_hint:
        lines.append(routing_hint)

    return summary_html(
        "Latest Run",
        f"{record.display_name} · {record.run_id}",
        pills=pills,
        lines=lines,
        variant="danger" if record.status not in {"ok", "succeeded"} else "default",
    )



def _actionable_error_message(entry: CatalogEntry, exc: Exception) -> str:
    """Format a failed run with the artifact root and concrete runtime hints."""

    error_text = f"{type(exc).__name__}: {exc}"
    lines = [
        error_text,
        "",
        f"Artifacts root: {Path(MANAGER.runs_root).resolve()}",
        "Check the manifest for the latest run under that folder, then verify model-ref/backend/device inputs.",
    ]
    if entry.default_model_ref and not Path(entry.default_model_ref).expanduser().exists():
        lines.append(f"Default model ref is not local: {entry.default_model_ref}")
    if entry.default_backend == "api_init":
        lines.append("Hosted API models also need endpoint and provider API key configuration.")
    return "\n".join(lines)


def _recent_choices() -> list[tuple[str, str]]:
    records = MANAGER.list_recent_runs()
    return [(f"{record.display_name} · {record.run_id}", record.run_id) for record in records]


def _render_run(record: RunRecord):
    prepare_run_record(record)
    spatial_stage, spatial_caption = _spatial_stage_for_record(record)
    points_viewport = _points_viewport_for_record(record)
    embodied_viewport = _embodied_viewport_for_record(record)
    status = _status_block(
        f"model={record.display_name}\nmode={record.mode}\nstatus={record.status}\nrun_id={record.run_id}\noutput_dir={record.output_dir}"
    )
    files = record.artifacts or []
    return (
        status,
        _run_overview_html(record),
        record.preview_video,
        record.preview_image,
        spatial_stage,
        points_viewport,
        embodied_viewport,
        record.preview_model,
        record.gallery,
        record.metadata,
        files,
        spatial_caption,
        gr.update(choices=_recent_choices(), value=record.run_id),
        recent_runs_table(MANAGER.list_recent_runs()),
    )


def _default_camera_path_payload(entry: CatalogEntry) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "mode": "keyframes",
        "loop": False,
        "fps": int(entry.default_call_kwargs.get("fps", 24) or 24),
        "duration_sec": 5,
        "keyframes": [
            {"t": 0.0, "position": [0.0, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0], "fov": 60.0},
            {"t": 1.0, "position": [0.0, 0.0, -1.0], "rotation": [0.0, 8.0, 0.0], "fov": 60.0},
        ],
    }
    if entry.model_id == "gen3c":
        payload.update(
            {
                "source": "gen3c_gui_reference",
                "seed_mode": "image_or_preprocessed_video",
                "export_cameras": True,
                "visualize_rendered_3d_cache": True,
            }
        )
    elif entry.category == "3D Scene":
        payload.update({"source": "spark_3dgs_stage", "render": "trajectory"})
    return payload


def _default_camera_path_json(model_id: str) -> str:
    try:
        entry = find_entry(model_id)
    except Exception:
        return "{}"
    return json.dumps(_default_camera_path_payload(entry), indent=2, ensure_ascii=False)


def _apply_camera_path_json(model_id: str, camera_path_text: str, call_kwargs_text: str):
    try:
        entry = find_entry(model_id)
    except Exception:
        return call_kwargs_text, _status_block("select a model before applying a camera path")

    try:
        camera_path = parse_jsonish(camera_path_text, default={}) or {}
        if not isinstance(camera_path, dict):
            raise ValueError("Camera Path JSON must decode to an object.")
        call_kwargs = _json_object_or_default(call_kwargs_text)
    except Exception as exc:
        return call_kwargs_text, _status_block(f"camera path rejected\n\n{type(exc).__name__}: {exc}")

    if entry.runtime_kind == "pointcloud_nav":
        call_kwargs["trajectory"] = camera_path
    else:
        call_kwargs["camera_path"] = camera_path
    if entry.model_id == "gen3c":
        call_kwargs.setdefault("export_cameras", True)
        call_kwargs.setdefault("visualize_rendered_3d_cache", True)
    if entry.category == "3D Scene":
        call_kwargs.setdefault("render_mode", "trajectory")

    return (
        json.dumps(call_kwargs, indent=2, ensure_ascii=False),
        _status_block(f"camera path applied · {entry.display_name}"),
    )


def _apply_longvie_sparse_control(sparse_video: Any, call_kwargs_text: str) -> str:
    """Bind LongVie 2's second control-video upload into call kwargs."""

    call_kwargs = _json_object_or_default(call_kwargs_text)
    path = ""
    if isinstance(sparse_video, str):
        path = sparse_video
    elif isinstance(sparse_video, Mapping):
        path = str(sparse_video.get("path") or sparse_video.get("video") or "")
    elif sparse_video is not None:
        path = str(getattr(sparse_video, "path", "") or getattr(sparse_video, "name", ""))
    if path:
        call_kwargs["sparse_video"] = path
    else:
        call_kwargs.pop("sparse_video", None)
    return json.dumps(call_kwargs, indent=2, ensure_ascii=False)


def _apply_preset(model_id: str, preset: str):
    if not model_id:
        return "", "", "", "{}", ""
    entry = find_entry(model_id)
    prompt = entry.default_prompt
    task_type = entry.default_task_type
    interactions = ", ".join(entry.default_interactions)
    call_kwargs = dict(entry.default_call_kwargs)
    camera_view_text = ""

    if preset == "orbit":
        interactions = ""
        if entry.runtime_kind == "two_stage_3dgs":
            call_kwargs.setdefault("num_orbit_frames", 36)
            call_kwargs.setdefault("yaw_step", 4.0)
        elif entry.runtime_kind == "pointcloud_nav":
            task_type = "render_view"
            camera_view_text = "0"
        elif entry.category in {"Video Generation", "Video-to-Video"}:
            call_kwargs.setdefault("num_frames", 64)
    elif preset == "trajectory":
        if entry.runtime_kind == "pointcloud_nav":
            task_type = "render_trajectory"
            interactions = ""
        elif entry.runtime_kind == "two_stage_3dgs":
            interactions = "forward,left,camera_r"
            call_kwargs.setdefault("frames_per_interaction", 10)
        elif entry.runtime_kind == "worldfm":
            interactions = "forward,left,camera_r"
        else:
            call_kwargs.setdefault("num_frames", 84)

    return (
        prompt,
        task_type,
        interactions,
        json.dumps(call_kwargs, indent=2, ensure_ascii=False),
        camera_view_text,
    )


def _run_action(
    action: str,
    model_id: str,
    prompt: str,
    input_path: str,
    image: Any,
    video: Any,
    last_frame: Any,
    reference_files: Any,
    interactions_text: str,
    camera_view_text: str,
    task_type: str,
    intrinsics_text: str,
    meta_path: str,
    panorama_path: str,
    scene_name: str,
    fps: int,
    num_frames: int,
    call_kwargs_text: str,
    load_kwargs_text: str,
    model_ref: str,
    backend: str,
    endpoint: str,
    api_key: str,
    device: str,
    variant_id: str,
    progress=gradio_progress(track_tqdm=False),
):
    del progress
    entry = find_entry(model_id)
    effective_load_kwargs_text, effective_call_kwargs_text, effective_model_ref = _apply_variant_runtime_overrides(
        model_id=model_id,
        variant_id=variant_id,
        interactions_text=interactions_text,
        load_kwargs_text=load_kwargs_text,
        call_kwargs_text=call_kwargs_text,
        model_ref=model_ref,
    )
    preparing_spatial, preparing_caption = (
        _spatial_stage_html(
            title=entry.display_name,
            note="Preparing the run. Spark will attach automatically if this job exports a Gaussian Splat.",
        ),
        "Preparing the run; any exported 3DGS will attach to 3D World automatically.",
    )
    yield (
        _status_block(f"preparing {entry.display_name} · action={action}"),
        _run_overview_html(stage="preparing", message=f"{entry.display_name} · {action}"),
        gr.update(),
        gr.update(),
        gr.update(value=preparing_spatial),
        gr.update(value=_points_viewport_idle_html()),
        gr.update(value=_embodied_viewport_idle_html()),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        preparing_caption,
        gr.update(choices=_recent_choices()),
        recent_runs_table(MANAGER.list_recent_runs()),
    )
    try:
        record = MANAGER.run(
            model_id=model_id,
            action=action,
            prompt=prompt,
            input_path=input_path,
            image=image,
            video=video,
            last_frame=last_frame,
            reference_files=reference_files,
            interactions_text=interactions_text,
            camera_view_text=camera_view_text,
            task_type=task_type,
            intrinsics_text=intrinsics_text,
            meta_path=meta_path,
            panorama_path=panorama_path,
            scene_name=scene_name,
            fps=fps,
            num_frames=num_frames,
            call_kwargs_text=effective_call_kwargs_text,
            load_kwargs_text=effective_load_kwargs_text,
            model_ref=effective_model_ref,
            backend=backend,
            endpoint=endpoint,
            api_key=api_key,
            device=device,
            progress_callback=None,
        )
    except Exception as exc:
        error_text = _actionable_error_message(entry, exc)
        yield (
            _status_block(f"{entry.display_name} failed\n\n{error_text}"),
            _run_overview_html(stage="error", message=error_text),
            gr.update(),
            gr.update(),
            gr.update(
                value=_spatial_stage_html(
                    title=entry.display_name,
                    note="This run failed before a spatial asset could be attached.",
                )
            ),
            gr.update(value=_points_viewport_idle_html()),
            gr.update(value=_embodied_viewport_idle_html()),
            gr.update(),
            gr.update(),
            {"error": error_text},
            gr.update(),
            "Run failed before any attachable spatial asset was produced.",
            gr.update(choices=_recent_choices()),
            recent_runs_table(MANAGER.list_recent_runs()),
        )
        return
    yield _render_run(record)




def _run_live_direction(
    direction: str,
    model_id: str,
    prompt: str,
    input_path: str,
    image: Any,
    video: Any,
    last_frame: Any,
    reference_files: Any,
    interactions_text: str,
    camera_view_text: str,
    task_type: str,
    intrinsics_text: str,
    meta_path: str,
    panorama_path: str,
    scene_name: str,
    fps: int,
    num_frames: int,
    call_kwargs_text: str,
    load_kwargs_text: str,
    model_ref: str,
    backend: str,
    endpoint: str,
    api_key: str,
    device: str,
    variant_id: str,
):
    del interactions_text
    yield from _run_action(
        "stream",
        model_id,
        prompt,
        input_path,
        image,
        video,
        last_frame,
        reference_files,
        direction,
        camera_view_text,
        task_type,
        intrinsics_text,
        meta_path,
        panorama_path,
        scene_name,
        fps,
        num_frames,
        call_kwargs_text,
        load_kwargs_text,
        model_ref,
        backend,
        endpoint,
        api_key,
        device,
        variant_id,
    )


def _run_start_action(*args: Any):
    # shared_inputs keeps `model_id` at index 0 and `interactions_text` at index 7.
    model_id = str(args[0]) if args else ""
    interactions_text = str(args[7]) if len(args) > 7 else ""
    yield from _run_action(_resolve_primary_action(model_id, interactions_text), *args)


def _run_stream_action(*args: Any):
    yield from _run_action("stream", *args)


def _make_live_direction_handler(direction: str):
    def handler(*args: Any):
        yield from _run_live_direction(direction, *args)

    return handler


def _reset_model_ui(model_id: str):
    try:
        entry = find_entry(model_id)
    except Exception:
        return (
            _status_block("select a model first"),
            _run_overview_html(stage="cache", message="Select a model before resetting its cached state."),
            gr.update(choices=_recent_choices()),
            recent_runs_table(MANAGER.list_recent_runs()),
        )

    try:
        message = MANAGER.reset_cached_model(model_id)
        record = MANAGER.make_message_record(
            entry,
            message,
            mode="reset",
            extra_metadata={"model_id": model_id},
        )
        return (
            _status_block(message),
            summary_html(
                "Interactive State Reset",
                entry.display_name,
                pills=("reset",),
                lines=(message, "Weights stay loaded unless you also unload the pipeline."),
            ),
            gr.update(choices=_recent_choices(), value=record.run_id),
            recent_runs_table(MANAGER.list_recent_runs()),
        )
    except Exception as exc:
        error_text = f"reset failed: {type(exc).__name__}: {exc}"
        return (
            _status_block(error_text),
            _run_overview_html(stage="error", message=error_text),
            gr.update(choices=_recent_choices()),
            recent_runs_table(MANAGER.list_recent_runs()),
        )


def _unload_models_ui(model_id: str):
    message = MANAGER.unload(model_id or None)
    STUDIO_VISER.shutdown()
    return (
        _status_block(message),
        summary_html(
            "Cache Update",
            "Pipeline cache changed.",
            pills=("cache",),
            lines=(message,),
        ),
        gr.update(choices=_recent_choices()),
        recent_runs_table(MANAGER.list_recent_runs()),
    )


def _load_recent_run(run_id: str):
    if not run_id:
        return (
            _status_block("select a recent run"),
            _run_overview_html(),
            None,
            None,
            _spatial_stage_for_record()[0],
            _points_viewport_idle_html(),
            _embodied_viewport_idle_html(),
            None,
            None,
            None,
            None,
            _spatial_stage_for_record()[1],
            gr.update(choices=_recent_choices()),
            recent_runs_table(MANAGER.list_recent_runs()),
        )
    try:
        record = MANAGER.load_run(run_id)
    except Exception as exc:
        error_text = f"could not load run: {exc}"
        return (
            _status_block(error_text),
            _run_overview_html(stage="error", message=error_text),
            None,
            None,
            _spatial_stage_html(note="Could not restore the spatial stage for this run."),
            _points_viewport_idle_html(),
            _embodied_viewport_idle_html(),
            None,
            None,
            {"error": str(exc)},
            None,
            "Could not restore the spatial stage linked to this run.",
            gr.update(choices=_recent_choices()),
            recent_runs_table(MANAGER.list_recent_runs()),
        )
    return _render_run(record)


def build_demo(launch_config: StudioLaunchConfig | None = None) -> gr.Blocks:
    # ``build_demo`` is the public programmatic launch boundary as well as the
    # CLI path.  Install compatibility patches here so importing Studio stays
    # side-effect free while ``build_demo().launch()`` remains fully patched.
    install_gradio_patches()
    catalog = _studio_catalog()
    fallback_model_id = catalog[0].model_id if catalog else ""
    active_launch = launch_config or StudioLaunchConfig(model_id=fallback_model_id)
    simple_launch_ui = not active_launch.show_aux_panels
    default_entry, filtered, default_state, default_variant_value, default_variant_notes = _launch_defaults(
        active_launch,
        catalog,
    )
    default_model_id = default_entry.model_id
    frontend_profile = _category_frontend_profile(default_entry)
    default_stream_button = _stream_button_update(default_entry)
    show_advanced_run = active_launch.show_aux_panels or bool(frontend_profile["advanced_run_visible"])
    show_more_inputs = active_launch.show_aux_panels or bool(frontend_profile["more_inputs_visible"])
    show_camera_panel = active_launch.show_aux_panels or bool(frontend_profile["camera_panel_visible"])
    show_spatial_panel = active_launch.show_aux_panels or bool(frontend_profile["spatial_panel_visible"])
    show_advanced_runtime = active_launch.show_aux_panels or bool(frontend_profile["runtime_advanced_visible"])
    default_spatial_stage, default_spatial_caption = _spatial_stage_for_record()
    default_points_viewport = _points_viewport_idle_html()
    default_embodied_viewport = _embodied_viewport_idle_html()
    logo_markup = _logo_nav_markup()

    with gr.Blocks(title="WorldFoundry Studio") as demo:
        model_state = gr.State(value=default_model_id)
        variant_state = gr.State(value=default_variant_value)
        gr.HTML(value=default_state[0], elem_classes=["wa-studio-header"])
        gr.HTML(
            f"""
<div class="wa-site-nav">
  <div class="wa-site-brand">
    <span class="wa-site-brand-mark" aria-hidden="true">
      {logo_markup}
    </span>
    <span class="wa-site-nav-left">WorldFoundry</span>
  </div>
  <div class="wa-site-nav-right">
    <button class="wa-site-nav-icon" type="button" data-wa-nav="joystick" aria-label="Joystick">
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        <path d="M6 12h12a4 4 0 0 1 4 4v1a2 2 0 0 1-2 2h-1l-2-3H7l-2 3H4a2 2 0 0 1-2-2v-1a4 4 0 0 1 4-4Z"></path>
        <path d="M6 12V9a6 6 0 0 1 12 0v3"></path>
        <path d="M8.5 14.5h3M10 13v3"></path>
        <path d="M16 14.5h.01M18 16.5h.01"></path>
      </svg>
    </button>
    <button class="wa-site-nav-icon" type="button" data-wa-nav="focus" aria-label="Focus stage">
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        <path d="M8 3H5a2 2 0 0 0-2 2v3"></path>
        <path d="M16 3h3a2 2 0 0 1 2 2v3"></path>
        <path d="M8 21H5a2 2 0 0 1-2-2v-3"></path>
        <path d="M16 21h3a2 2 0 0 0 2-2v-3"></path>
        <path d="M9 9h6v6H9z"></path>
      </svg>
    </button>
    <button class="wa-site-nav-icon" type="button" data-wa-nav="performance" aria-label="Use lighter rendering">
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        <path d="M13 2 4 14h7l-1 8 10-13h-7l1-7Z"></path>
      </svg>
    </button>
    <button class="wa-site-nav-icon" type="button" data-wa-nav="theme" aria-label="Theme">
      <span class="wa-theme-icon wa-theme-icon-moon" aria-hidden="true">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
          <path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z"></path>
        </svg>
      </span>
      <span class="wa-theme-icon wa-theme-icon-sun" aria-hidden="true">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
          <circle cx="12" cy="12" r="4"></circle>
          <path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M4.93 19.07l1.41-1.41M17.66 6.34l1.41-1.41"></path>
        </svg>
      </span>
    </button>
  </div>
</div>
""",
            elem_classes=["wa-site-nav-shell"],
        )

        main_grid_classes = ["wa-main-grid"]
        if simple_launch_ui:
            main_grid_classes.append("wa-main-grid-simple")
        main_grid_classes.append(f"wa-category-{frontend_profile['slug']}")
        main_grid_classes.append(f"wa-template-{frontend_profile['template_id']}")

        with gr.Row(equal_height=False, elem_classes=main_grid_classes):
            with gr.Column(
                scale=4,
                min_width=340,
                elem_classes=["wa-left-rail"],
                visible=active_launch.show_aux_panels,
            ):
                with gr.Column(elem_classes=["wa-panel-block"]):
                    gr.Markdown("### Model", elem_classes=["wa-panel-title"])
                    gr.HTML(value=default_state[2])
                    with gr.Accordion("Notes", open=False, visible=active_launch.show_aux_panels):
                        gr.HTML(value=default_state[1], elem_classes=["wa-atlas-overview"])
                        gr.HTML(value=default_state[3])
                        gr.HTML(value=default_state[4])

                with gr.Accordion(
                    "History",
                    open=False,
                    visible=active_launch.show_aux_panels,
                    elem_classes=["wa-panel-block"],
                ):
                    run_selector = gr.Dropdown(label="Snapshot", choices=_recent_choices())
                    load_run_button = gr.Button("Restore Snapshot", elem_classes=["wa-run-muted"])
                    with gr.Accordion("Recent Runs", open=False):
                        recent_table = gr.Dataframe(
                            headers=["Run ID", "Model", "Mode", "Status", "Folder"],
                            value=recent_runs_table(MANAGER.list_recent_runs()),
                            interactive=False,
                            wrap=True,
                            elem_classes=["wa-dataframe"],
                        )

                with gr.Accordion(
                    "Catalog",
                    open=False,
                    visible=active_launch.show_aux_panels,
                    elem_classes=["wa-panel-block"],
                ):
                    gr.Dataframe(
                        headers=["Display", "ID", "Category", "Family", "Stream", "Backend", "Tags"],
                        value=catalog_as_table(filtered),
                        interactive=False,
                        wrap=True,
                        elem_classes=["wa-dataframe"],
                    )

            with gr.Column(scale=7, min_width=520, elem_classes=["wa-stage-col"]):
                with gr.Column(
                    elem_classes=[
                        "wa-panel-block",
                        "wa-preview-panel",
                        f"wa-category-{frontend_profile['slug']}",
                        f"wa-template-{frontend_profile['template_id']}",
                    ]
                ):
                    gr.HTML(
                        """
<div class="wa-stage-empty" id="wa-stage-empty">
  <div class="wa-stage-empty-inner">
    <div class="wa-stage-empty-title" id="wa-stage-empty-title"></div>
    <div class="wa-stage-empty-copy" id="wa-stage-empty-copy"></div>
  </div>
</div>
""",
                        elem_classes=["wa-stage-empty-shell"],
                    )
                    status = gr.HTML(value=_status_block("idle"), elem_classes=["wa-status-host"])
                    run_overview = gr.HTML(value=_run_overview_html(entry=default_entry), elem_classes=["wa-run-overview"])
                    with gr.Tab("Preview Video"):
                        primary_video = gr.Video(
                            label="Primary Video",
                            show_label=False,
                            elem_id="wa-main-preview-video",
                            elem_classes=["wa-stage-media", "wa-stage-video"],
                        )
                    with gr.Tab("Preview Image"):
                        primary_image = gr.Image(
                            label="Primary Image",
                            show_label=False,
                            elem_id="wa-main-preview-image",
                            elem_classes=["wa-stage-media", "wa-stage-image"],
                        )
                    with gr.Tab("3D World"):
                        spatial_stage = gr.HTML(
                            value=default_spatial_stage,
                            elem_classes=["wa-stage-media", "wa-stage-splat-host"],
                        )
                        model_preview = gr.Model3D(
                            label="Interactive Model Preview",
                            show_label=False,
                            elem_classes=["wa-stage-media", "wa-stage-model"],
                        )
                        gr.Markdown(
                            "Spark drives Gaussian Splat exports in-browser. When a run only exposes a mesh or point cloud, the fallback `Model3D` preview stays available below.",
                            elem_classes=["wa-download-hint"],
                        )
                    with gr.Tab("Point Cloud (Viser)"):
                        points_viewport = gr.HTML(
                            value=default_points_viewport,
                            elem_classes=["wa-stage-media", "wa-stage-points-host"],
                        )
                        gr.Markdown(
                            "Loopback Viser session for geometry-friendly point clouds discovered in the run folder. "
                            "Install `worldfoundry[studio_pointcloud]` when the optional viewer is not available.",
                            elem_classes=["wa-download-hint"],
                        )
                    with gr.Tab("Embodied Sim"):
                        embodied_viewport = gr.HTML(
                            value=default_embodied_viewport,
                            elem_classes=["wa-stage-media", "wa-stage-embodied-host"],
                        )
                    with gr.Tab("Gallery"):
                        gallery = gr.Gallery(
                            label="Run Gallery",
                            show_label=False,
                            columns=4,
                            height=420,
                            elem_classes=["wa-stage-media", "wa-stage-gallery"],
                        )
                    with gr.Tab("Artifacts"):
                        manifest_json = gr.JSON(
                            label="Run Manifest",
                            show_label=False,
                            elem_classes=["wa-stage-media", "wa-stage-artifacts"],
                        )
                        artifact_files = gr.File(
                            label="Artifacts",
                            show_label=False,
                            file_count="multiple",
                            elem_classes=["wa-stage-media", "wa-stage-artifacts"],
                        )
                    gr.HTML(
                        """
<div class="wa-player-footer">
  <div class="wa-player-footer-left">
    <span class="wa-player-dot"></span>
    <span id="wa-player-footer-status"></span>
  </div>
  <div class="wa-player-footer-center" id="wa-player-footer-world"></div>
  <div class="wa-player-footer-right" id="wa-player-footer-time"></div>
</div>
""",
                        elem_classes=["wa-player-chrome"],
                    )
                    gr.HTML(
                        """
<div class="wa-joystick-dock" id="wa-joystick-dock" aria-hidden="true">
  <div class="wa-joystick-dock-copy">
    <div class="wa-joystick-dock-title">Navigation Pad</div>
    <div class="wa-joystick-dock-note" id="wa-joystick-note">Enable a stream-capable model to unlock move and camera sticks.</div>
  </div>
</div>
""",
                        elem_classes=["wa-joystick-dock-shell"],
                    )
                    with gr.Row(elem_classes=["wa-control-dock"]):
                        stream_button = gr.Button(
                            default_stream_button["value"],
                            variant="secondary",
                            visible=bool(default_stream_button["visible"]),
                            elem_classes=["wa-action-pill", "wa-action-step"],
                        )
                        run_button = gr.Button(
                            _run_button_label(default_entry),
                            variant="primary",
                            elem_classes=["wa-action-pill", "wa-action-run", "wa-action-start"],
                        )
                        reset_button = gr.Button(
                            "RESET",
                            elem_classes=["wa-action-pill", "wa-action-reset"],
                        )
                        unload_button = gr.Button("Unload", elem_classes=["wa-action-pill", "wa-action-hidden"])
                    gr.HTML(_world_tray_html(TRAY_DEMO_IMAGE_FILES), elem_classes=["wa-world-tray-shell"])
                    tray_input_gallery = gr.Gallery(
                        value=_tray_gallery_items(TRAY_DEMO_IMAGE_FILES),
                        label="Input Tray",
                        show_label=False,
                        columns=8,
                        rows=1,
                        height=96,
                        preview=False,
                        allow_preview=False,
                        object_fit="cover",
                        elem_classes=["wa-input-tray-gallery"],
                    )

            with gr.Column(scale=5, min_width=380, elem_classes=["wa-right-rail"]):
                with gr.Column(elem_classes=["wa-panel-block"]):
                    gr.Markdown("### Run", elem_classes=["wa-panel-title"])
                    gr.HTML(
                        value=_frontend_mode_html(default_entry),
                        elem_classes=["wa-mode-summary"],
                    )
                    gr.HTML(
                        value=_template_workbench_html(default_entry),
                        elem_classes=["wa-template-workbench-host"],
                    )
                    gr.HTML(
                        value=_interface_contract_html(default_entry),
                        elem_classes=["wa-mode-summary", "wa-interface-contract"],
                    )
                    prompt = gr.Textbox(
                        label=frontend_profile["prompt_label"],
                        lines=int(frontend_profile["prompt_lines"]),
                        value=default_state[5],
                        placeholder=frontend_profile["prompt_placeholder"],
                        visible=bool(frontend_profile["prompt_visible"]),
                    )
                    interactions_text = gr.Textbox(
                        label=frontend_profile["actions_label"],
                        lines=4,
                        value=default_state[7],
                        placeholder=frontend_profile["actions_placeholder"],
                        visible=bool(frontend_profile["actions_visible"]),
                    )
                    with gr.Accordion(
                        "Advanced Run",
                        open=bool(frontend_profile["advanced_run_open"]),
                        visible=show_advanced_run,
                    ):
                        gr.HTML(
                            value=default_variant_notes,
                            elem_classes=["wa-variant-notes"],
                            visible=show_advanced_run,
                        )
                        task_type = gr.Textbox(
                            label="Task",
                            value=default_state[6],
                            placeholder="optional model-specific task type",
                            visible=show_advanced_run and bool(frontend_profile["task_visible"]),
                        )
                        with gr.Row(elem_classes=["wa-preset-row"]):
                            preset_default = gr.Button(
                                "Default",
                                elem_classes=["wa-run-muted"],
                                visible=show_advanced_run,
                            )
                            preset_orbit = gr.Button(
                                "Orbit",
                                elem_classes=["wa-run-muted"],
                                visible=show_advanced_run,
                            )
                            preset_trajectory = gr.Button(
                                "Trajectory",
                                elem_classes=["wa-run-muted"],
                                visible=show_advanced_run,
                            )

                with gr.Column(elem_classes=["wa-panel-block"]):
                    gr.Markdown("### Inputs", elem_classes=["wa-panel-title"])
                    with gr.Tab("Main"):
                        image = gr.Image(
                            label=frontend_profile["image_label"],
                            type="pil",
                            visible=bool(frontend_profile["image_visible"]),
                        )
                        video = gr.Video(
                            label=frontend_profile["video_label"],
                            visible=bool(frontend_profile["video_visible"]),
                        )
                        sparse_control_video = gr.Video(
                            label="Sparse Pointmap / Track Control Segment",
                            visible=default_entry.model_id == "longvie-2",
                        )
                        input_path = gr.Textbox(
                            label=frontend_profile["path_label"],
                            placeholder=frontend_profile["path_placeholder"],
                            visible=bool(frontend_profile["path_visible"]),
                        )
                        if DEMO_IMAGE_LIBRARY_FILES:
                            demo_image_gallery = gr.Gallery(
                                value=_demo_gallery_items(),
                                label="Example Images",
                                show_label=False,
                                columns=5,
                                rows=2,
                                height=188,
                            )
                        else:
                            demo_image_gallery = gr.Gallery(
                                value=[],
                                label="Example Images",
                                show_label=False,
                                height=160,
                                visible=False,
                            )
                    with gr.Accordion(
                        "More Inputs",
                        open=bool(frontend_profile["more_inputs_open"]),
                        visible=show_more_inputs,
                    ):
                        last_frame = gr.Image(
                            label="Last Frame",
                            type="pil",
                            visible=show_more_inputs,
                        )
                        reference_files = gr.File(
                            label="References",
                            file_count="multiple",
                            file_types=["image"],
                            visible=show_more_inputs,
                        )
                    with gr.Accordion(
                        "Camera / Scene",
                        open=bool(frontend_profile["camera_panel_open"]),
                        visible=show_camera_panel,
                    ):
                        camera_view_text = gr.Textbox(
                            label="Camera Pose",
                            placeholder='[dx, dy, dz, theta_x, theta_z]',
                            visible=show_camera_panel,
                        )
                        camera_path_text = gr.Code(
                            label="Camera Path JSON",
                            language="json",
                            value=_default_camera_path_json(default_model_id),
                            visible=show_camera_panel,
                        )
                        with gr.Row(elem_classes=["wa-preset-row"]):
                            camera_path_apply = gr.Button(
                                "Apply Path",
                                elem_classes=["wa-run-muted"],
                                visible=show_camera_panel,
                            )
                            camera_path_reset = gr.Button(
                                "Reset Path",
                                elem_classes=["wa-run-muted"],
                                visible=show_camera_panel,
                            )
                        intrinsics_text = gr.Textbox(
                            label="Intrinsics",
                            placeholder="[[fx,0,cx],[0,fy,cy],[0,0,1]]",
                            visible=show_camera_panel,
                        )
                        meta_path = gr.Textbox(label="Meta Path", visible=show_camera_panel)
                        panorama_path = gr.Textbox(
                            label="Panorama Path",
                            visible=show_camera_panel,
                        )
                        scene_name = gr.Textbox(label="Scene Name", visible=show_camera_panel)

                with gr.Accordion(
                    "3DGS Import",
                    open=bool(frontend_profile["spatial_panel_open"]),
                    visible=show_spatial_panel,
                    elem_classes=["wa-panel-block", "wa-spatial-panel"],
                ):
                    spatial_asset = gr.File(
                        label="3DGS",
                        file_count="single",
                        file_types=sorted(SPATIAL_ASSET_EXTS),
                        visible=show_spatial_panel,
                    )
                    spatial_clear = gr.Button(
                        "Clear 3D World",
                        elem_classes=["wa-run-muted"],
                        visible=show_spatial_panel,
                    )
                    spatial_caption = gr.Markdown(
                        value=default_spatial_caption,
                        elem_classes=["wa-spatial-caption"],
                        visible=show_spatial_panel,
                    )

                tray_image_source = gr.Textbox(
                    label="Tray Image Source",
                    value="",
                    elem_id="wa-tray-image-source",
                    elem_classes=["wa-dom-bridge"],
                )
                tray_image_apply = gr.Button(
                    "Apply Tray Image",
                    elem_id="wa-tray-image-apply",
                    elem_classes=["wa-dom-bridge"],
                )
                joystick_bridge = gr.Row(
                    visible=bool(default_entry and _supports_live_controls(default_entry)),
                    elem_classes=["wa-joystick-bridge"],
                )
                gr.Markdown(
                    "`WASD` move · `IJKL` look · hold to rollout",
                    elem_classes=["wa-panel-copy", "wa-live-controls-copy"],
                )
                with joystick_bridge:
                    forward_button = gr.Button("W", elem_id="wa-live-forward")
                    left_button = gr.Button("A", elem_id="wa-live-left")
                    backward_button = gr.Button("S", elem_id="wa-live-backward")
                    right_button = gr.Button("D", elem_id="wa-live-right")
                    camera_left_button = gr.Button("J", elem_id="wa-live-camera-left")
                    camera_right_button = gr.Button("L", elem_id="wa-live-camera-right")
                    camera_up_button = gr.Button("I", elem_id="wa-live-camera-up")
                    camera_down_button = gr.Button("K", elem_id="wa-live-camera-down")

                with gr.Column(elem_classes=["wa-panel-block"]):
                    gr.Markdown("### Runtime", elem_classes=["wa-panel-title"])
                    device = gr.Textbox(label="Device", value=active_launch.device or "cuda")
                    with gr.Accordion(
                        "Advanced Runtime",
                        open=bool(frontend_profile["runtime_advanced_open"]),
                        visible=show_advanced_runtime,
                    ):
                        model_ref = gr.Textbox(
                            label="Checkpoint",
                            value=default_state[10],
                            visible=show_advanced_runtime and active_launch.show_aux_panels,
                        )
                        backend = gr.Dropdown(
                            label="Loader",
                            choices=["auto", "from_pretrained", "api_init"],
                            value=default_state[11],
                            visible=show_advanced_runtime and active_launch.show_aux_panels,
                        )
                        endpoint = gr.Textbox(
                            label="Endpoint",
                            value=default_state[12],
                            visible=show_advanced_runtime and bool(frontend_profile["endpoint_visible"]),
                        )
                        api_key = gr.Textbox(
                            label="API Key",
                            type="password",
                            visible=show_advanced_runtime and bool(frontend_profile["api_key_visible"]),
                        )
                        with gr.Row():
                            fps = gr.Slider(
                                label="FPS",
                                minimum=1,
                                maximum=60,
                                step=1,
                                value=16,
                                visible=show_advanced_runtime and bool(frontend_profile["fps_frames_visible"]),
                            )
                            num_frames = gr.Slider(
                                label="Num Frames",
                                minimum=0,
                                maximum=240,
                                step=1,
                                value=0,
                                visible=show_advanced_runtime and bool(frontend_profile["fps_frames_visible"]),
                            )
                        load_kwargs_text = gr.Code(
                            label="Load JSON",
                            language="json",
                            value=default_state[8],
                            visible=show_advanced_runtime and bool(frontend_profile["json_visible"]),
                        )
                        call_kwargs_text = gr.Code(
                            label="Call JSON",
                            language="json",
                            value=default_state[9],
                            visible=show_advanced_runtime and bool(frontend_profile["json_visible"]),
                        )

        input_preview_outputs = [
            status,
            run_overview,
            primary_video,
            primary_image,
            model_preview,
            gallery,
            manifest_json,
            artifact_files,
        ]

        preset_default.click(
            lambda model_id: _apply_preset(model_id, "default"),
            inputs=[model_state],
            outputs=[prompt, task_type, interactions_text, call_kwargs_text, camera_view_text],
            queue=False,
        )
        preset_orbit.click(
            lambda model_id: _apply_preset(model_id, "orbit"),
            inputs=[model_state],
            outputs=[prompt, task_type, interactions_text, call_kwargs_text, camera_view_text],
            queue=False,
        )
        preset_trajectory.click(
            lambda model_id: _apply_preset(model_id, "trajectory"),
            inputs=[model_state],
            outputs=[prompt, task_type, interactions_text, call_kwargs_text, camera_view_text],
            queue=False,
        )
        camera_path_apply.click(
            _apply_camera_path_json,
            inputs=[model_state, camera_path_text, call_kwargs_text],
            outputs=[call_kwargs_text, status],
            queue=False,
        )
        camera_path_reset.click(
            _default_camera_path_json,
            inputs=[model_state],
            outputs=[camera_path_text],
            queue=False,
        )
        sparse_control_video.change(
            _apply_longvie_sparse_control,
            inputs=[sparse_control_video, call_kwargs_text],
            outputs=[call_kwargs_text],
            queue=False,
        )

        shared_inputs = [
            model_state,
            prompt,
            input_path,
            image,
            video,
            last_frame,
            reference_files,
            interactions_text,
            camera_view_text,
            task_type,
            intrinsics_text,
            meta_path,
            panorama_path,
            scene_name,
            fps,
            num_frames,
            call_kwargs_text,
            load_kwargs_text,
            model_ref,
            backend,
            endpoint,
            api_key,
            device,
            variant_state,
        ]
        shared_outputs = [
            status,
            run_overview,
            primary_video,
            primary_image,
            spatial_stage,
            points_viewport,
            embodied_viewport,
            model_preview,
            gallery,
            manifest_json,
            artifact_files,
            spatial_caption,
            run_selector,
            recent_table,
        ]

        run_button.click(
            _run_start_action,
            inputs=shared_inputs,
            outputs=shared_outputs,
            concurrency_limit=1,
            show_progress="hidden",
        )
        stream_button.click(
            _run_stream_action,
            inputs=shared_inputs,
            outputs=shared_outputs,
            concurrency_limit=1,
            concurrency_id="wa-live-rollout",
            show_progress="hidden",
            trigger_mode="always_last",
        )
        spatial_asset.change(
            _load_spatial_asset,
            inputs=[spatial_asset],
            outputs=[spatial_stage, spatial_caption],
            queue=False,
        )
        spatial_clear.click(
            _clear_spatial_asset,
            outputs=[spatial_stage, spatial_caption],
            queue=False,
        )
        image.change(
            _sync_input_scene_preview,
            inputs=[model_state, image, video, input_path, last_frame],
            outputs=input_preview_outputs,
            queue=False,
        )
        video.change(
            _sync_input_scene_preview,
            inputs=[model_state, image, video, input_path, last_frame],
            outputs=input_preview_outputs,
            queue=False,
        )
        last_frame.change(
            _sync_input_scene_preview,
            inputs=[model_state, image, video, input_path, last_frame],
            outputs=input_preview_outputs,
            queue=False,
        )
        input_path.change(
            _sync_input_scene_preview,
            inputs=[model_state, image, video, input_path, last_frame],
            outputs=input_preview_outputs,
            queue=False,
        )
        tray_image_apply.click(
            _use_tray_image_as_input,
            inputs=[tray_image_source],
            outputs=[image, input_path, video, status],
            queue=False,
        ).then(
            _sync_input_scene_preview,
            inputs=[model_state, image, video, input_path, last_frame],
            outputs=input_preview_outputs,
            queue=False,
        )
        tray_input_gallery.select(
            _on_tray_image_select,
            outputs=[image, input_path, video, status],
            queue=False,
        ).then(
            _sync_input_scene_preview,
            inputs=[model_state, image, video, input_path, last_frame],
            outputs=input_preview_outputs,
            queue=False,
        )
        demo_image_gallery.select(
            _on_demo_image_select,
            outputs=[image, input_path, video, status],
            queue=False,
        ).then(
            _sync_input_scene_preview,
            inputs=[model_state, image, video, input_path, last_frame],
            outputs=input_preview_outputs,
            queue=False,
        )
        for button, token in [
            (forward_button, "forward"),
            (left_button, "left"),
            (backward_button, "backward"),
            (right_button, "right"),
            (camera_left_button, "camera_l"),
            (camera_right_button, "camera_r"),
            (camera_up_button, "camera_up"),
            (camera_down_button, "camera_down"),
        ]:
            button.click(
                _make_live_direction_handler(token),
                inputs=shared_inputs,
                outputs=shared_outputs,
                concurrency_limit=1,
                concurrency_id="wa-live-rollout",
                show_progress="hidden",
                trigger_mode="always_last",
            )

        reset_button.click(
            _reset_model_ui,
            inputs=[model_state],
            outputs=[status, run_overview, run_selector, recent_table],
            queue=False,
        )
        unload_button.click(
            _unload_models_ui,
            inputs=[model_state],
            outputs=[status, run_overview, run_selector, recent_table],
            queue=False,
        )
        load_run_button.click(
            _load_recent_run,
            inputs=[run_selector],
            outputs=shared_outputs,
            queue=False,
        )

        demo.queue(default_concurrency_limit=1, max_size=2)
    return demo


def main(argv: Sequence[str] | None = None) -> None:
    _ensure_localhost_no_proxy()
    launch_config = parse_launch_config(argv)
    entry = find_entry(launch_config.model_id)
    frontend_mode = resolve_frontend_mode(entry, launch_config.frontend, launch_config.asset_path or None)
    if frontend_mode in NATIVE_FRONTENDS:
        serve_native_frontend(entry, launch_config, frontend_mode)
        return

    use_torchrun_bridge = launch_uses_lingbot_torchrun_rollout(launch_config)
    if use_torchrun_bridge:
        os.environ[TORCHRUN_LINGBOT_FAST_ENV] = "1"
        ensure_torchrun_lingbot_fast_runtime()
        if _torchrun_rank() != 0:
            try:
                MANAGER.run_torchrun_worker_loop()
            finally:
                shutdown_torchrun_lingbot_fast_runtime()
            return
    elif _torchrun_world_size() > 1 and _torchrun_rank() != 0:
        try:
            while True:
                time.sleep(3600)
        finally:
            shutdown_torchrun_lingbot_fast_runtime()
    demo = build_demo(launch_config)
    host = host_for_frontend(launch_config)
    port = port_for_frontend(launch_config, frontend_mode)
    auth_token = require_auth_token_for_host(host, server_name="WorldFoundry Studio (Gradio)")
    share_text = env_first("WORLDFOUNDRY_STUDIO_SHARE").strip().lower()
    launch_kwargs: dict[str, Any] = {
        "server_name": host,
        "share": share_text in {"1", "true", "yes", "on"},
        "css": CUSTOM_CSS,
        "head": HEAD_HTML + _default_tray_head_html(TRAY_DEMO_IMAGE_FILES),
        "allowed_paths": [
            str(STUDIO_ASSET_DIR.resolve()),
            str(Path(MANAGER.workspace_root).resolve()),
        ],
    }
    if auth_token:
        # Gradio's auth machinery gates every page and API route; any username
        # is accepted as long as the password equals the shared Studio token.
        launch_kwargs["auth"] = lambda _username, password: token_matches(password, auth_token)
        launch_kwargs["auth_message"] = (
            "This Studio instance is bound to a non-loopback address. Enter any "
            "username and the WORLDFOUNDRY_STUDIO_AUTH_TOKEN value as the password."
        )
    if DEMO_IMAGE_LIBRARY_ROOT.exists():
        launch_kwargs["allowed_paths"].append(str(DEMO_IMAGE_LIBRARY_ROOT.resolve()))
    if SPARK_ROOT.exists():
        launch_kwargs["allowed_paths"].append(str(SPARK_ROOT.resolve()))
    launch_kwargs["server_port"] = port
    print_remote_access(frontend_mode, host, port)
    launch_proxy_env = mask_socks_proxy_env_for_gradio()
    try:
        demo.launch(**launch_kwargs)
    finally:
        os.environ.update(launch_proxy_env)
        if use_torchrun_bridge and _torchrun_rank() == 0:
            try:
                MANAGER.shutdown_torchrun_workers()
            except Exception:
                pass
        shutdown_torchrun_lingbot_fast_runtime()


if __name__ == "__main__":
    main()
