"""Standalone interactive world frontend for WorldFoundry Studio.

This module intentionally avoids Gradio and external demo frontends.  It owns
the page HTML/CSS/JS and launch helpers; the realtime HTTP/WebRTC transport
lives in ``world_realtime``.
"""

from __future__ import annotations

import base64
import contextlib
import json
import os
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from typing import Any

from worldfoundry.runtime.inference_specs.lingbot_world import (
    LINGBOT_VARIANT_BASE_ACT_PREVIEW,
    LINGBOT_VARIANT_BASE_CAM,
)
from worldfoundry.studio.inference.catalog import CatalogEntry, lingbot_world_fast_load_kwargs
from worldfoundry.studio.inference.config import env_first
from worldfoundry.studio.inference.execution import (
    IMAGE_EXTS,
    LINGBOT_VARIANT_FAST,
    LINGBOT_WORLD_MODEL_ID,
    TORCHRUN_LINGBOT_FAST_ENV,
    StudioManager,
    _torchrun_rank,
    ensure_torchrun_lingbot_fast_control_group,
    shutdown_torchrun_lingbot_fast_runtime,
)
from worldfoundry.studio.visualization.launch_config import StudioLaunchConfig, launch_uses_lingbot_torchrun_rollout

WORLD_REFERENCE_IMAGE_ROOT = Path(__file__).resolve().parents[2] / "assets" / "reference_images" / "world"
WORLD_WEB_ASSET_ROOT = Path(__file__).resolve().parents[2] / "assets" / "web"
DEFAULT_FPS = 16


def serve_world_frontend(
    entry: CatalogEntry,
    launch_config: StudioLaunchConfig,
    *,
    host: str,
    port: int,
    access_printer: Callable[[str, str, int], None],
) -> None:
    """Serve the standalone interactive world frontend and block."""

    manager = StudioManager()

    use_torchrun_lingbot = launch_uses_lingbot_torchrun_rollout(launch_config)
    if use_torchrun_lingbot:
        os.environ[TORCHRUN_LINGBOT_FAST_ENV] = "1"
        ensure_torchrun_lingbot_fast_control_group()
        if _torchrun_rank() != 0:
            try:
                manager.run_torchrun_worker_loop()
            finally:
                shutdown_torchrun_lingbot_fast_runtime()
            return

    demo_images = _demo_image_files()
    access_printer("world", host, port)
    from worldfoundry.studio.visualization.backends.world_realtime import (
        serve_realtime_world_frontend,
    )

    try:
        serve_realtime_world_frontend(
            entry=entry,
            launch_config=launch_config,
            manager=manager,
            host=host,
            port=port,
            demo_images=demo_images,
            allowed_roots=_world_allowed_roots(manager, launch_config),
        )
    finally:
        if use_torchrun_lingbot:
            with contextlib.suppress(Exception):
                manager.shutdown_torchrun_workers()
            shutdown_torchrun_lingbot_fast_runtime()


def world_frontend_html(entry: CatalogEntry, launch_config: StudioLaunchConfig) -> str:
    """Return the standalone world frontend HTML shell."""

    title = _html_escape(entry.display_name or entry.model_id)
    model_label = _html_escape(entry.model_id)
    variant = _html_escape(launch_config.variant_id or "default")
    prompt = _html_escape(entry.default_prompt or "")
    prompt_scheduled = "prompt-scheduled" in entry.tags
    queued_segments = "queued-segment-generation" in entry.tags
    non_keyboard_mode = prompt_scheduled or queued_segments
    if queued_segments:
        interaction_hint = "UPLOAD NEW DENSE + SPARSE CONTROLS FOR EACH EXTEND"
    elif prompt_scheduled:
        interaction_hint = "OPTIONAL IMAGE · EDIT PROMPT + PRESS ENTER BETWEEN SEGMENTS"
    else:
        interaction_hint = "HOLD WASD / DRAG STICKS"
    stick_class = "stick-well is-hidden" if non_keyboard_mode else "stick-well"
    look_stick_class = "stick-well stick-right is-hidden" if non_keyboard_mode else "stick-well stick-right"
    video_control_class = "video-only is-hidden" if non_keyboard_mode else "video-only"
    standard_control_class = "is-hidden" if queued_segments else ""
    queued_control_class = "" if queued_segments else "is-hidden"
    body_class = "queued-segment-mode" if queued_segments else ""
    controls_class = "controls-deck controls-centered" if non_keyboard_mode else "controls-deck"
    thumb_tray_class = "thumb-tray is-hidden" if queued_segments else "thumb-tray"
    start_label = "RUN" if queued_segments else "START"
    console_title = "QUEUED SEGMENTS" if queued_segments else "REALTIME STREAM"
    stats_label = "WS READY" if non_keyboard_mode else "RTC OFF"
    image_label = "INITIAL IMAGE" if queued_segments else "OPTIONAL IMAGE" if prompt_scheduled else "IMAGE"
    rollout_title = (
        "Full-quality segments run sequentially while weights and continuation state remain resident."
        if queued_segments
        else "Controls are resampled continuously while a bounded frame queue feeds the persistent video track."
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>WorldFoundry Studio - {title}</title>
  <link rel="icon" href="/favicon.svg" type="image/svg+xml">
  <link rel="stylesheet" href="/world.css">
</head>
<body class="{body_class}">
  <div class="site-chrome">
    <div class="brand-mark">
      <span class="brand-dot"></span>
      <span>WORLDFOUNDRY</span>
    </div>
  </div>

  <main class="world-shell">
    <section class="spatial-device" aria-label="Interactive world console">
      <div class="device-topline">
        <span>{title}</span>
        <span>{model_label} / {variant}</span>
      </div>
      <div class="screen-bezel">
        <div class="world-viewport" id="viewport">
          <img id="frameImage" alt="" draggable="false">
          <video id="frameVideo" muted playsinline></video>
          <canvas id="frameCanvas" aria-label="Decoded generated world frames"></canvas>
          <div id="sensorGrid" aria-label="Synchronized sensor views"></div>
          <div id="sensorControls" aria-label="Sensor presentation controls"></div>
          <div class="screen-noise"></div>
          <div class="floating-state" id="livePill">FROZEN</div>
          <div class="pointer-hint is-hidden" id="pointerHint" aria-live="polite"></div>
          <div class="start-overlay is-hidden" id="startOverlay" aria-hidden="true">
            <div class="overlay-status" id="overlayStatus">LOAD INPUT</div>
          </div>
          <div class="viewport-tools" aria-label="Viewport tools">
            <button class="viewport-tool" id="fullscreenButton" type="button" aria-pressed="false" title="Toggle immersive fullscreen (F)">
              <span class="viewport-tool-icon" aria-hidden="true"></span>
              <span id="fullscreenLabel">FULLSCREEN</span>
            </button>
            <button class="viewport-tool" id="logToggleButton" type="button" aria-pressed="false" aria-controls="runtimeLog" title="Toggle live runtime log (Shift+L)">
              <span class="live-log-dot" aria-hidden="true"></span>
              <span>LIVE LOG</span>
              <span class="log-count" id="logCount">0</span>
            </button>
          </div>
          <aside class="runtime-log studio-panel" id="runtimeLog" data-panel-id="logs" aria-label="Live runtime log" aria-hidden="true">
            <div class="runtime-log-header panel-handle">
              <div>
                <span class="live-log-dot" aria-hidden="true"></span>
                <strong>RUNTIME</strong>
                <span>LIVE</span>
              </div>
              <div class="panel-header-actions">
                <button id="clearLogButton" type="button">CLEAR</button>
                <button class="panel-collapse" data-collapse-panel type="button" aria-label="Collapse logs" aria-expanded="true">−</button>
              </div>
            </div>
            <div class="runtime-log-body studio-panel-body" data-panel-body>
              <ol class="runtime-log-list" id="runtimeLogList" role="log" aria-live="off"></ol>
            </div>
          </aside>
          <div class="immersive-hint" id="immersiveHint">F FULLSCREEN&nbsp;&nbsp; SHIFT+L LOG&nbsp;&nbsp; ESC EXIT</div>
        </div>
      </div>

      <section class="studio-panel status-panel" id="statusPanel" data-panel-id="status" aria-label="Session status">
        <div class="studio-panel-header panel-handle">
          <span>STATUS</span>
          <button class="panel-collapse" data-collapse-panel type="button" aria-label="Collapse status" aria-expanded="true">−</button>
        </div>
        <div class="studio-panel-body" data-panel-body>
          <div class="console-strip">
            <div class="power-light"><span></span> POWER</div>
            <div class="console-title" title="Input edges and generated frames travel independently over one persistent resident session.">{console_title}</div>
            <div class="stream-stats" id="streamStats">{stats_label}</div>
            <div class="timer" id="sessionTimer">00:00</div>
          </div>
          <div class="rollout-note" title="{rollout_title}">
            <span>RESIDENT RUNTIME</span>
            <span id="interactionHint">{interaction_hint}</span>
          </div>
        </div>
      </section>

      <section class="studio-panel controls-panel" id="controlsPanel" data-panel-id="controls" aria-label="World controls">
        <div class="studio-panel-header panel-handle">
          <span>CONTROLS</span>
          <button class="panel-collapse" data-collapse-panel type="button" aria-label="Collapse controls" aria-expanded="true">−</button>
        </div>
        <div class="studio-panel-body" data-panel-body>
          <div class="{controls_class}">
            <button class="{stick_class}" id="moveStick" data-stick="move" type="button" aria-label="Movement joystick"><span></span></button>
            <div class="session-actions">
              <button class="start-button" id="startButton" type="button">{start_label}</button>
              <button class="reset-button is-hidden" id="resetButton" type="button">RESET</button>
            </div>
            <button class="{look_stick_class}" id="lookStick" data-stick="look" type="button" aria-label="Camera joystick"><span></span></button>
          </div>
          <div class="realtime-control-bar">
            <button class="step-button" id="stepButton" type="button" disabled>STEP</button>
            <button class="step-button is-hidden" id="pauseButton" type="button" aria-pressed="false" disabled>PAUSE</button>
            <label class="resolution-control" for="resolutionSelect">
              <span>OUTPUT</span>
              <select id="resolutionSelect" aria-label="WebRTC output resolution">
                <option value="native">NATIVE</option>
              </select>
            </label>
            <span class="control-ack" id="controlAck" role="status" aria-live="polite">LOCAL</span>
          </div>
          <div class="event-trigger-bar" id="eventTriggerBar" aria-label="Text event controls">
            <span class="event-empty">NO TEXT EVENTS</span>
          </div>
        </div>
      </section>

      <section class="source-dock studio-panel scene-panel" id="scenePanel" data-panel-id="scene" aria-label="Initial scene and text events">
        <div class="studio-panel-header panel-handle">
          <span>INITIAL SCENE</span>
          <button class="panel-collapse" data-collapse-panel type="button" aria-label="Collapse initial scene" aria-expanded="true">−</button>
        </div>
        <div class="studio-panel-body" data-panel-body>
          <div class="source-toolbar">
            <div class="segmented {standard_control_class}" role="tablist" aria-label="World input mode">
              <button class="mode-tab is-active" data-mode="image" type="button">IMAGE WORLD</button>
              <button class="mode-tab {video_control_class}" data-mode="video" type="button">VIDEO WORLD</button>
            </div>
            <button class="load-button" id="loadModelButton" type="button">LOAD</button>
            <label class="file-chip image-only"><span id="imageInputLabel">{image_label}</span><input id="imageInput" type="file" accept="image/*"></label>
            <label class="file-chip {video_control_class} {standard_control_class}"><span>VIDEO</span><input id="videoInput" type="file" accept="video/*"></label>
            <label class="file-chip {queued_control_class}"><span id="denseInputLabel">DENSE DEPTH</span><input id="denseVideoInput" type="file" accept="video/*"></label>
            <label class="file-chip {queued_control_class}"><span id="sparseInputLabel">SPARSE TRACK</span><input id="sparseVideoInput" type="file" accept="video/*"></label>
            <input class="prompt-input" id="promptInput" type="text" value="{prompt}" aria-label="Prompt" placeholder="Describe the next segment">
          </div>
          <div class="segment-input-status {queued_control_class}" id="segmentInputStatus" aria-live="polite">
            <span id="imageFileState">INITIAL IMAGE REQUIRED</span>
            <span id="denseFileState">DENSE CONTROL REQUIRED</span>
            <span id="sparseFileState">SPARSE CONTROL REQUIRED</span>
          </div>
          <div class="{thumb_tray_class}" id="thumbTray" aria-label="Example worlds"></div>
          <div class="text-event-editor" id="textEventEditor">
            <div class="text-event-toolbar">
              <div><strong>TEXT EVENTS</strong><span>Editable during a live session</span></div>
              <div>
                <button id="addEventButton" type="button">ADD</button>
                <button id="applyEventsButton" type="button">APPLY</button>
              </div>
            </div>
            <div class="event-editor-rows" id="eventEditorRows"></div>
            <div class="event-catalog-status" id="eventCatalogStatus" role="status" aria-live="polite">EMPTY CATALOG</div>
          </div>
        </div>
      </section>
    </section>
  </main>

  <script src="/world.js"></script>
</body>
</html>"""


@lru_cache(maxsize=1)
def world_favicon_svg() -> str:
    """Return a tiny inline favicon for the standalone world frontend."""

    return """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64"><defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1"><stop stop-color="#f9f871"/><stop offset=".5" stop-color="#40d890"/><stop offset="1" stop-color="#48a7ff"/></linearGradient></defs><rect width="64" height="64" rx="16" fill="#10141c"/><circle cx="32" cy="32" r="19" fill="url(#g)"/></svg>"""


@lru_cache(maxsize=1)
def world_frontend_css() -> str:
    """Return CSS for the standalone world frontend."""

    return (WORLD_WEB_ASSET_ROOT / "world.css").read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def world_frontend_js() -> str:
    """Return the persistent WebRTC realtime frontend client."""

    return (WORLD_WEB_ASSET_ROOT / "world.js").read_text(encoding="utf-8")


def _launch_runtime_overrides(
    entry: CatalogEntry,
    launch_config: StudioLaunchConfig,
    *,
    interactions_text: str,
) -> tuple[str, str, str]:
    has_interactions = bool(str(interactions_text or "").strip())
    load_kwargs: dict[str, Any] = {}
    call_kwargs: dict[str, Any] = {}
    model_ref = (launch_config.model_ref or entry.default_model_ref or "").strip()

    if entry.model_id == LINGBOT_WORLD_MODEL_ID:
        variant_id = launch_config.variant_id or LINGBOT_VARIANT_BASE_CAM
        if variant_id == LINGBOT_VARIANT_FAST:
            load_kwargs = lingbot_world_fast_load_kwargs()
            load_kwargs.setdefault("runtime_variant", "fast")
            if not launch_uses_lingbot_torchrun_rollout(launch_config):
                # Single-process launches still use the stable one-GPU path.
                # Multi-GPU execution is enabled by launching this frontend with
                # torchrun; those ranks share work through the StudioManager
                # torchrun command bridge above.
                load_kwargs["dit_fsdp"] = False
                load_kwargs["t5_fsdp"] = False
                load_kwargs["ulysses_size"] = 1
                load_kwargs["t5_cpu"] = True
            # For torchrun, StudioManager chooses replication versus FSDP from
            # per-rank VRAM and applies the Ulysses degree. Leaving these unset
            # is essential for 24/40/48GB open-source deployments.
            fast_model_path = str(load_kwargs.get("fast_model_path", "") or "")
            # The fast variant still needs the base bundle (T5 encoder, VAE, tokenizer,
            # and the cam/act control type inferred from the base checkpoint name), so the
            # base checkpoint must stay as the pretrained path while the distilled DiT is
            # loaded from fast_model_path. Only fall back to the fast path as the pretrained
            # path when no base checkpoint is resolvable locally.
            if fast_model_path and not launch_config.model_ref and not model_ref:
                model_ref = fast_model_path
            call_kwargs = {
                "num_frames": 9,
                "seed": 42,
                "max_area": 480 * 832,
                "offload_model": False,
                "wmfactory_action_controls": True,
            }
            if has_interactions:
                call_kwargs["action_path"] = None
        elif variant_id == LINGBOT_VARIANT_BASE_CAM:
            load_kwargs = {
                "runtime_variant": None,
                "fast_model_path": None,
                "t5_fsdp": False,
                "dit_fsdp": False,
                "ulysses_size": 1,
            }
            call_kwargs = {"num_frames": 21, "sampling_steps": 20, "seed": 42}
            if has_interactions:
                call_kwargs["action_path"] = None
        elif variant_id == LINGBOT_VARIANT_BASE_ACT_PREVIEW:
            load_kwargs = {
                "runtime_variant": None,
                "fast_model_path": None,
                "t5_fsdp": False,
                "dit_fsdp": False,
                "ulysses_size": 1,
            }
            call_kwargs = {"seed": 42, "allow_act2cam": True, "sampling_steps": 20}
            if has_interactions:
                call_kwargs["action_path"] = None

    return (
        json.dumps(load_kwargs, ensure_ascii=False),
        json.dumps(call_kwargs, ensure_ascii=False),
        model_ref,
    )


@lru_cache(maxsize=8)
def _demo_image_files(root: Path | None = None) -> tuple[Path, ...]:
    source_root = (root or WORLD_REFERENCE_IMAGE_ROOT).expanduser()
    if not source_root.exists():
        return ()
    return tuple(
        sorted(
            path.resolve() for path in source_root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTS
        )
    )


def _reference_image_label(path: Path, index: int) -> str:
    label = path.stem.replace("_", " ").replace("-", " ").strip()
    return label.title() if label else f"Scene {index}"


def _world_allowed_roots(manager: StudioManager, launch_config: StudioLaunchConfig) -> tuple[Path, ...]:
    roots = [
        Path(manager.workspace_root).expanduser().resolve(),
        WORLD_REFERENCE_IMAGE_ROOT.expanduser().resolve(),
    ]
    if launch_config.asset_path:
        asset = Path(launch_config.asset_path).expanduser().resolve()
        roots.append(asset if asset.is_dir() else asset.parent)
    return tuple(dict.fromkeys(path for path in roots if path.exists()))


def _decode_data_url(value: str) -> tuple[bytes, str]:
    header = ""
    data = value
    if "," in value and value.startswith("data:"):
        header, data = value.split(",", 1)
    mime = "application/octet-stream"
    if header.startswith("data:"):
        mime = header[5:].split(";", 1)[0] or mime
    return base64.b64decode(data), mime


def _api_key() -> str:
    return env_first("WORLDFOUNDRY_STUDIO_API_KEY").strip()


def _html_escape(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
