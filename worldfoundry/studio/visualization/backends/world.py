"""Standalone interactive world frontend for WorldFoundry Studio.

This module intentionally avoids Gradio and external demo frontends.  It serves
a compact Studio-native HTTP API for the original browser UI.
"""

from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
import os
import random
import struct
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from functools import lru_cache
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from io import BytesIO
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, quote, unquote, urlparse

from PIL import Image

from worldfoundry.runtime.inference_catalog import (
    LINGBOT_VARIANT_BASE_ACT_PREVIEW,
    LINGBOT_VARIANT_BASE_CAM,
)
from worldfoundry.studio.inference.catalog import CatalogEntry, lingbot_world_fast_load_kwargs
from worldfoundry.studio.inference.execution import (
    IMAGE_EXTS,
    LINGBOT_VARIANT_FAST,
    LINGBOT_WORLD_MODEL_ID,
    TORCHRUN_LINGBOT_FAST_ENV,
    VIDEO_EXTS,
    RunRecord,
    StudioManager,
    _missing_runtime_validation_imports,
    _runtime_dependency_error,
    _torchrun_rank,
    ensure_torchrun_lingbot_fast_control_group,
    shutdown_torchrun_lingbot_fast_runtime,
)
from worldfoundry.studio.ui.interfaces import interface_spec_for_entry
from worldfoundry.studio.ui.launch_config import StudioLaunchConfig, launch_uses_lingbot_torchrun_rollout
from worldfoundry.studio.inference.config import env_first
from worldfoundry.studio.serving import (
    StudioServiceTelemetry,
    parse_byte_range,
    path_allowed,
    send_file_response,
    send_json_response,
    send_text_response,
)

WORLD_REFERENCE_IMAGE_ROOT = (
    Path(__file__).resolve().parents[2] / "assets" / "reference_images" / "world"
)
# Kept as a compatibility alias for callers that imported the previous name.
WORLD_DEMO_IMAGE_ROOT = WORLD_REFERENCE_IMAGE_ROOT
WORLD_UPLOAD_DIR_NAME = "world_frontend_inputs"
DEFAULT_FPS = 16
WORLD_ACTION_DEADZONE = 0.08
WORLD_STREAM_BASE_SEED = 42
WEBSOCKET_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


def _world_max_sessions() -> int:
    try:
        return max(int(os.getenv("WORLDFOUNDRY_STUDIO_WORLD_MAX_SESSIONS", "4") or "4"), 1)
    except Exception:
        return 4


@dataclass
class WorldSession:
    session_id: str
    mode: str
    prompt: str
    seed_image: Image.Image | None = None
    seed_video_path: str = ""
    last_record: RunRecord | None = None
    created_at: float = field(default_factory=time.time)
    step_count: int = 0


@dataclass
class WorldFrontendState:
    entry: CatalogEntry
    launch_config: StudioLaunchConfig
    manager: StudioManager
    demo_images: tuple[Path, ...]
    allowed_roots: tuple[Path, ...]
    telemetry: StudioServiceTelemetry
    max_sessions: int = 4
    lock: threading.RLock = field(default_factory=threading.RLock)
    model_loaded: bool = False
    sessions: dict[str, WorldSession] = field(default_factory=dict)


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
            try:
                manager.shutdown_torchrun_workers()
            except Exception:
                pass
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
    look_stick_class = (
        "stick-well stick-right is-hidden" if non_keyboard_mode else "stick-well stick-right"
    )
    video_control_class = "video-only is-hidden" if non_keyboard_mode else "video-only"
    standard_control_class = "is-hidden" if queued_segments else ""
    queued_control_class = "" if queued_segments else "is-hidden"
    body_class = "queued-segment-mode" if queued_segments else ""
    controls_class = "controls-deck controls-centered" if non_keyboard_mode else "controls-deck"
    thumb_tray_class = "thumb-tray is-hidden" if queued_segments else "thumb-tray"
    start_label = "RUN" if queued_segments else "START"
    console_title = "QUEUED SEGMENTS" if queued_segments else "REALTIME STREAM"
    stats_label = "WS READY" if non_keyboard_mode else "RTC OFF"
    image_label = (
        "INITIAL IMAGE" if queued_segments else "OPTIONAL IMAGE" if prompt_scheduled else "IMAGE"
    )
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
          <div class="screen-noise"></div>
          <div class="floating-state" id="livePill">FROZEN</div>
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
            <span>{interaction_hint}</span>
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

    return r"""
:root {
  color-scheme: dark;
  --page: #08090c;
  --page-2: #11141a;
  --device: #d9d9d5;
  --device-2: #bebfb9;
  --bezel: #0d0f13;
  --ink: #101217;
  --muted: #777f8c;
  --panel: #181b21;
  --accent: #43a5ff;
  --accent-2: #33d17a;
  --danger: #ff6767;
  --shadow: rgba(0, 0, 0, 0.42);
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

* {
  box-sizing: border-box;
}

html,
body {
  margin: 0;
  min-height: 100%;
  background: radial-gradient(circle at 50% 0%, #171a21 0, var(--page) 44rem);
  color: #f3f5f8;
}

body {
  min-height: 100vh;
  overflow-x: hidden;
}

button,
input {
  font: inherit;
}

.is-hidden {
  display: none !important;
}

button:disabled,
.file-chip.is-disabled {
  cursor: wait;
  opacity: 0.52;
}

.site-chrome {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  z-index: 20;
  display: flex;
  align-items: center;
  justify-content: space-between;
  height: 4rem;
  padding: 0 2rem;
  pointer-events: none;
}

.brand-mark {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  pointer-events: auto;
}

.brand-mark {
  font-size: 0.78rem;
  font-weight: 800;
  letter-spacing: 0;
  color: #f7f9fc;
}

.brand-dot {
  width: 1.1rem;
  height: 1.1rem;
  border-radius: 50%;
  background: linear-gradient(135deg, #f9f871, #40d890 52%, #48a7ff);
  box-shadow: 0 0 1.1rem rgba(64, 216, 144, 0.5);
}

.world-shell {
  min-height: 100vh;
  display: grid;
  place-items: center;
  padding: 4.75rem 1rem 1rem;
}

.spatial-device {
  width: min(74rem, calc(100vw - 2rem));
  border-radius: 1.75rem;
  padding: 0.78rem 0.78rem 0.9rem;
  background:
    linear-gradient(180deg, rgba(255,255,255,0.92), rgba(218,218,212,0.96) 38%, rgba(185,186,180,0.98)),
    var(--device);
  box-shadow:
    0 2.2rem 5.4rem var(--shadow),
    inset 0 0.12rem 0 rgba(255, 255, 255, 0.82),
    inset 0 -0.35rem 1rem rgba(0, 0, 0, 0.17);
}

.device-topline {
  display: flex;
  justify-content: space-between;
  gap: 1rem;
  color: #3a3d42;
  font-size: 0.72rem;
  font-weight: 800;
  padding: 0.05rem 0.55rem 0.52rem;
  text-transform: uppercase;
}

.screen-bezel {
  border-radius: 1.25rem;
  background: linear-gradient(180deg, #17191f, #08090d);
  padding: 0.72rem;
  box-shadow:
    inset 0 0 0 0.1rem rgba(255, 255, 255, 0.08),
    inset 0 1rem 3rem rgba(0, 0, 0, 0.72);
}

.world-viewport {
  position: relative;
  aspect-ratio: 16 / 9;
  overflow: hidden;
  border-radius: 0.95rem;
  background: #050608;
  display: grid;
  place-items: center;
  isolation: isolate;
  contain: layout paint style;
}

#frameImage,
#frameVideo,
#frameCanvas {
  width: 100%;
  height: 100%;
  object-fit: contain;
  background: #050608;
  display: none;
  backface-visibility: hidden;
  will-change: opacity;
}

#frameImage.is-visible,
#frameVideo.is-visible,
#frameCanvas.is-visible {
  display: block;
}

.screen-noise {
  position: absolute;
  inset: 0;
  pointer-events: none;
  opacity: 0.18;
  mix-blend-mode: screen;
  background-image:
    linear-gradient(rgba(255,255,255,0.05) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255,255,255,0.035) 1px, transparent 1px);
  background-size: 3px 3px;
  z-index: 2;
}

.floating-state {
  position: absolute;
  top: 0.85rem;
  right: 0.85rem;
  z-index: 5;
  border: 1px solid rgba(255, 255, 255, 0.14);
  border-radius: 999px;
  padding: 0.42rem 0.72rem;
  background: rgba(4, 7, 11, 0.62);
  color: #e8ecf2;
  font-size: 0.72rem;
  font-weight: 900;
}

.floating-state.is-live {
  color: #dfffe9;
  border-color: rgba(51, 209, 122, 0.42);
}

.start-overlay {
  display: none;
}

.start-overlay.is-hidden {
  display: none;
}

.overlay-status {
  display: none;
}

.viewport-tools {
  position: absolute;
  top: 0.85rem;
  left: 0.85rem;
  z-index: 8;
  display: flex;
  align-items: center;
  gap: 0.42rem;
}

.viewport-tool,
.runtime-log-header button {
  appearance: none;
  border: 1px solid rgba(255, 255, 255, 0.14);
  border-radius: 999px;
  background: rgba(5, 8, 13, 0.82);
  color: #e9eef7;
  min-height: 2rem;
  padding: 0 0.68rem;
  font-size: 0.64rem;
  font-weight: 900;
  letter-spacing: 0.04em;
  cursor: pointer;
}

.viewport-tool {
  display: inline-flex;
  align-items: center;
  gap: 0.38rem;
}

.viewport-tool:hover,
.viewport-tool:focus-visible,
.runtime-log-header button:hover,
.runtime-log-header button:focus-visible {
  border-color: rgba(102, 189, 255, 0.62);
  background: rgba(13, 21, 32, 0.94);
  outline: none;
}

.viewport-tool[aria-pressed="true"] {
  border-color: rgba(72, 167, 255, 0.56);
  background: rgba(17, 54, 82, 0.9);
}

.viewport-tool-icon {
  display: inline-block;
  width: 0.68rem;
  height: 0.58rem;
  border: 1px solid currentColor;
  border-radius: 0.06rem;
  line-height: 1;
}

.live-log-dot {
  flex: 0 0 auto;
  width: 0.44rem;
  height: 0.44rem;
  border-radius: 50%;
  background: #42dc82;
  box-shadow: 0 0 0.72rem rgba(66, 220, 130, 0.9);
}

.log-count {
  min-width: 1.15rem;
  border-radius: 999px;
  padding: 0.08rem 0.3rem;
  background: rgba(255, 255, 255, 0.1);
  color: #cdd5e1;
  font-variant-numeric: tabular-nums;
}

.runtime-log {
  position: absolute;
  top: 3.55rem;
  right: 0.85rem;
  bottom: 0.85rem;
  z-index: 7;
  display: flex;
  flex-direction: column;
  width: min(25rem, 44%);
  min-height: 0;
  overflow: hidden;
  border: 1px solid rgba(255, 255, 255, 0.13);
  border-radius: 0.8rem;
  background: linear-gradient(180deg, rgba(10, 15, 22, 0.96), rgba(4, 7, 11, 0.94));
  box-shadow: 0 1.2rem 3rem rgba(0, 0, 0, 0.42);
  opacity: 0;
  pointer-events: none;
  transform: translate3d(0.65rem, 0, 0);
  transition: opacity 140ms ease, transform 140ms ease;
}

.log-open .runtime-log {
  opacity: 1;
  pointer-events: auto;
  transform: translate3d(0, 0, 0);
}

.runtime-log-header {
  flex: 0 0 auto;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.75rem;
  min-height: 2.6rem;
  padding: 0.45rem 0.55rem 0.45rem 0.75rem;
  border-bottom: 1px solid rgba(255, 255, 255, 0.09);
  color: #eef4fb;
  font-size: 0.62rem;
  letter-spacing: 0.06em;
}

.runtime-log-header > div {
  display: flex;
  align-items: center;
  gap: 0.42rem;
}

.runtime-log-header > div > span:last-child {
  color: #7e8a99;
}

.runtime-log-header button {
  min-height: 1.65rem;
  padding: 0 0.5rem;
  color: #aab4c2;
  font-size: 0.58rem;
}

.runtime-log-list {
  min-height: 0;
  flex: 1 1 auto;
  margin: 0;
  padding: 0.42rem 0;
  overflow: auto;
  overscroll-behavior: contain;
  scrollbar-width: thin;
  scrollbar-color: rgba(255,255,255,0.2) transparent;
  font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
  font-size: 0.66rem;
  line-height: 1.4;
}

.runtime-log-entry {
  display: grid;
  grid-template-columns: 4.6rem minmax(0, 1fr);
  gap: 0.55rem;
  padding: 0.3rem 0.72rem;
  color: #cbd4e0;
}

.runtime-log-entry:hover {
  background: rgba(255, 255, 255, 0.035);
}

.runtime-log-time {
  color: #667282;
  font-variant-numeric: tabular-nums;
}

.runtime-log-message {
  min-width: 0;
  overflow-wrap: anywhere;
}

.runtime-log-detail {
  display: block;
  margin-top: 0.08rem;
  color: #748193;
}

.runtime-log-entry[data-level="live"] .runtime-log-message,
.runtime-log-entry[data-level="metric"] .runtime-log-message {
  color: #8ee8b2;
}

.runtime-log-entry[data-level="warn"] .runtime-log-message {
  color: #ffd479;
}

.runtime-log-entry[data-level="error"] .runtime-log-message {
  color: #ff8d8d;
}

.immersive-hint {
  display: none;
}

.console-strip {
  display: grid;
  grid-template-columns: 1fr auto auto 1fr;
  align-items: center;
  gap: 1rem;
  color: #2f3339;
  font-size: 0.75rem;
  font-weight: 900;
  padding: 0.55rem 0.65rem 0.18rem;
}

.power-light {
  display: flex;
  align-items: center;
  gap: 0.45rem;
}

.power-light span {
  width: 0.54rem;
  height: 0.54rem;
  border-radius: 50%;
  background: #38d978;
  box-shadow: 0 0 0.8rem rgba(56, 217, 120, 0.9);
}

.console-title {
  letter-spacing: 0;
}

.stream-stats {
  color: #4a515b;
  font-size: 0.68rem;
  font-variant-numeric: tabular-nums;
}

.stream-stats.is-live {
  color: #18864c;
}

.timer {
  text-align: right;
  font-variant-numeric: tabular-nums;
}

.rollout-note {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.55rem;
  color: #3d444d;
  font-size: 0.66rem;
  font-weight: 900;
  padding: 0.1rem 0.65rem 0.28rem;
}

.rollout-note span {
  border: 1px solid rgba(45, 52, 60, 0.22);
  border-radius: 999px;
  padding: 0.18rem 0.5rem;
  background: rgba(255, 255, 255, 0.22);
}

.controls-deck {
  display: grid;
  grid-template-columns: 1fr auto 1fr;
  align-items: center;
  gap: 1rem;
  padding: 0.22rem 3.5rem 0.55rem;
}

.controls-deck.controls-centered {
  grid-template-columns: 1fr;
  min-height: 4.1rem;
}

.controls-centered .session-actions {
  justify-self: center;
}

.session-actions {
  display: flex;
  align-items: center;
  justify-content: center;
  min-width: 8rem;
}

.stick-well {
  --stick-offset-x: 0rem;
  --stick-offset-y: 0rem;
  appearance: none;
  border: 0;
  justify-self: start;
  position: relative;
  width: 5.8rem;
  height: 5.8rem;
  border-radius: 50%;
  background:
    radial-gradient(circle at 50% 50%, rgba(255,255,255,0.16) 0 26%, transparent 27%),
    radial-gradient(circle at 48% 44%, #8e9498 0 36%, #687078 37% 52%, #3b424b 53% 100%);
  box-shadow: inset 0 0.35rem 0.8rem rgba(255, 255, 255, 0.42), 0 0.5rem 1.4rem rgba(0,0,0,0.25);
  cursor: grab;
  padding: 0;
  touch-action: none;
}

.stick-well span {
  position: absolute;
  left: 50%;
  top: 50%;
  width: 42%;
  height: 42%;
  border-radius: 50%;
  background:
    radial-gradient(circle at 50% 50%, #f5f7f2 0 34%, #cfd4d2 35% 58%, #7f878d 59% 100%);
  box-shadow:
    inset 0 0.22rem 0.36rem rgba(255,255,255,0.72),
    0 0.45rem 0.9rem rgba(0,0,0,0.28);
  transform: translate(calc(-50% + var(--stick-offset-x)), calc(-50% + var(--stick-offset-y)));
  transition: transform 140ms ease;
}

.stick-well.is-active {
  cursor: grabbing;
}

.stick-well.is-active span {
  transition: none;
}

.stick-right {
  justify-self: end;
}

.start-button,
.reset-button,
.load-button {
  border: 0;
  border-radius: 999px;
  background: #1c2028;
  color: #f5f7fa;
  font-size: 0.78rem;
  font-weight: 900;
  min-height: 2.35rem;
  padding: 0 1.25rem;
  box-shadow: inset 0 0.1rem 0 rgba(255,255,255,0.14), 0 0.4rem 0.9rem rgba(0,0,0,0.2);
}

.start-button {
  color: #071018;
  background: #e9f6ff;
}

.session-actions .is-hidden {
  display: none;
}

.source-dock {
  border-radius: 1.05rem;
  background: linear-gradient(180deg, #191d24, #11141a);
  padding: 0.55rem;
  box-shadow: inset 0 0.12rem 0 rgba(255,255,255,0.08), inset 0 -0.5rem 1.4rem rgba(0,0,0,0.35);
}

.source-toolbar {
  display: grid;
  grid-template-columns: auto auto auto auto 1fr;
  gap: 0.45rem;
  align-items: center;
  margin-bottom: 0.52rem;
}

.segmented {
  display: flex;
  border: 1px solid rgba(255,255,255,0.11);
  border-radius: 999px;
  padding: 0.18rem;
  background: rgba(0,0,0,0.26);
}

.mode-tab {
  border: 0;
  min-height: 2.05rem;
  border-radius: 999px;
  background: transparent;
  color: #aeb7c5;
  padding: 0 0.9rem;
  font-size: 0.72rem;
  font-weight: 900;
}

.mode-tab.is-active {
  color: #071018;
  background: #e9f6ff;
}

.file-chip {
  display: inline-grid;
  place-items: center;
  min-height: 2.12rem;
  border-radius: 999px;
  border: 1px solid rgba(255,255,255,0.13);
  color: #e9eef7;
  padding: 0 0.9rem;
  font-size: 0.72rem;
  font-weight: 900;
  cursor: pointer;
}

.file-chip input {
  position: absolute;
  opacity: 0;
  width: 1px;
  height: 1px;
  pointer-events: none;
}

.file-chip span {
  display: block;
  max-width: 10rem;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.prompt-input {
  width: 100%;
  min-width: 0;
  min-height: 2.12rem;
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: 999px;
  background: rgba(255,255,255,0.06);
  color: #f6f8fb;
  padding: 0 1rem;
  outline: none;
}

.video-mode .image-only,
.image-mode .video-only {
  display: none;
}

.thumb-tray {
  display: grid;
  grid-template-columns: repeat(9, minmax(3.95rem, 1fr));
  gap: 0.5rem;
}

.segment-input-status {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 0.5rem;
  color: #aeb7c5;
  font-size: 0.68rem;
  font-weight: 800;
}

.segment-input-status span {
  min-width: 0;
  border: 1px solid rgba(255,255,255,0.1);
  border-radius: 0.58rem;
  padding: 0.58rem 0.7rem;
  background: rgba(255,255,255,0.045);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.segment-input-status span.is-ready {
  border-color: rgba(51,209,122,0.36);
  color: #dfffe9;
}

.thumb-card {
  position: relative;
  aspect-ratio: 1 / 1;
  border-radius: 0.6rem;
  overflow: hidden;
  border: 0.12rem solid rgba(255,255,255,0.1);
  background: #090b10;
  padding: 0;
}

.thumb-card.is-active {
  border-color: #66bdff;
  box-shadow: 0 0 0 0.15rem rgba(67,165,255,0.22);
}

.thumb-card img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
}

.thumb-card span {
  position: absolute;
  left: 0.45rem;
  bottom: 0.4rem;
  right: 0.45rem;
  color: #fff;
  text-shadow: 0 0.1rem 0.4rem rgba(0,0,0,0.72);
  font-size: 0.68rem;
  font-weight: 900;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.studio-panel {
  position: relative;
  min-width: 0;
}

.status-panel,
.controls-panel {
  margin-top: 0.18rem;
  border: 1px solid rgba(28, 33, 41, 0.12);
  border-radius: 0.82rem;
  background: rgba(255, 255, 255, 0.16);
  overflow: hidden;
}

.studio-panel-header,
.panel-handle {
  user-select: none;
}

.studio-panel-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  min-height: 1.72rem;
  padding: 0.22rem 0.42rem 0.18rem 0.62rem;
  color: #5d6670;
  font-size: 0.58rem;
  font-weight: 950;
  letter-spacing: 0.08em;
  cursor: grab;
  touch-action: none;
}

.source-dock > .studio-panel-header {
  color: #9aa5b5;
  padding: 0.1rem 0.1rem 0.42rem 0.2rem;
}

.panel-handle:active {
  cursor: grabbing;
}

.panel-collapse {
  display: inline-grid;
  place-items: center;
  width: 1.55rem;
  min-height: 1.35rem;
  padding: 0;
  border: 1px solid rgba(128, 140, 154, 0.24);
  border-radius: 0.42rem;
  background: rgba(255, 255, 255, 0.07);
  color: inherit;
  font: 800 0.8rem/1 system-ui, sans-serif;
  cursor: pointer;
}

.studio-panel.is-collapsed > [data-panel-body] {
  display: none !important;
}

.studio-panel.is-collapsed > .studio-panel-header,
.studio-panel.is-collapsed > .runtime-log-header {
  border-bottom: 0;
}

.studio-panel.is-floating {
  position: fixed !important;
  right: auto !important;
  bottom: auto !important;
  z-index: 70;
  width: min(42rem, calc(100vw - 1rem));
  max-height: calc(100vh - 1rem);
  margin: 0 !important;
  overflow: hidden;
  box-shadow: 0 1.25rem 3.8rem rgba(0, 0, 0, 0.42) !important;
  will-change: left, top, transform;
}

.runtime-log.is-collapsed {
  bottom: auto !important;
  min-height: 0;
  height: auto;
  max-height: 2.7rem;
}

.studio-panel.is-floating > [data-panel-body] {
  max-height: calc(100vh - 3.2rem);
  overflow: auto;
  overscroll-behavior: contain;
}

.runtime-log-body {
  display: flex;
  min-height: 0;
  flex: 1 1 auto;
}

.panel-header-actions {
  display: flex;
  align-items: center;
  gap: 0.3rem;
}

.realtime-control-bar {
  display: flex;
  align-items: center;
  justify-content: center;
  flex-wrap: wrap;
  gap: 0.45rem;
  padding: 0 0.65rem 0.55rem;
}

.step-button,
.resolution-control,
.event-trigger,
.event-clear,
.text-event-toolbar button,
.event-remove {
  min-height: 2rem;
  border: 1px solid rgba(31, 37, 45, 0.18);
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.32);
  color: #272d35;
  font-size: 0.66rem;
  font-weight: 900;
}

.step-button,
.event-trigger,
.event-clear,
.text-event-toolbar button,
.event-remove {
  padding: 0 0.8rem;
}

.step-button:disabled,
.event-trigger:disabled {
  opacity: 0.45;
  cursor: not-allowed;
}

.resolution-control {
  display: inline-flex;
  align-items: center;
  gap: 0.38rem;
  padding: 0 0.38rem 0 0.7rem;
}

.resolution-control span {
  color: #5c6570;
  font-size: 0.58rem;
  letter-spacing: 0.05em;
}

.resolution-control select {
  min-height: 1.55rem;
  border: 0;
  border-radius: 999px;
  background: #20262e;
  color: #eef4fb;
  padding: 0 1.6rem 0 0.6rem;
  font: 800 0.62rem/1 system-ui, sans-serif;
}

.control-ack {
  min-width: 5.5rem;
  color: #737d88;
  font-size: 0.58rem;
  font-weight: 900;
  text-align: center;
}

.control-ack.is-ok {
  color: #178249;
}

.control-ack.is-error {
  color: #be3434;
}

.event-trigger-bar {
  display: flex;
  align-items: center;
  justify-content: center;
  flex-wrap: wrap;
  gap: 0.38rem;
  min-height: 2.35rem;
  padding: 0 0.65rem 0.58rem;
}

.event-trigger.is-active {
  border-color: rgba(20, 139, 76, 0.46);
  background: #dff8e9;
  color: #0a6935;
}

.event-trigger.is-pending {
  box-shadow: 0 0 0 0.16rem rgba(52, 145, 255, 0.24);
}

.event-clear {
  color: #66717d;
}

.event-empty {
  color: #737d88;
  font-size: 0.62rem;
  font-weight: 850;
}

.text-event-editor {
  margin-top: 0.55rem;
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 0.78rem;
  background: rgba(0, 0, 0, 0.18);
  overflow: hidden;
}

.text-event-toolbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.75rem;
  padding: 0.48rem 0.55rem;
}

.text-event-toolbar > div {
  display: flex;
  align-items: center;
  gap: 0.42rem;
}

.text-event-toolbar strong {
  color: #e7edf6;
  font-size: 0.64rem;
  letter-spacing: 0.05em;
}

.text-event-toolbar span,
.event-catalog-status {
  color: #778392;
  font-size: 0.6rem;
}

.text-event-toolbar button,
.event-remove {
  min-height: 1.72rem;
  border-color: rgba(255, 255, 255, 0.12);
  background: rgba(255, 255, 255, 0.07);
  color: #dbe3ed;
}

.event-editor-rows {
  display: grid;
  gap: 0.4rem;
  padding: 0 0.55rem;
}

.event-editor-row {
  display: grid;
  grid-template-columns: minmax(5rem, 0.65fr) minmax(6rem, 0.8fr) minmax(12rem, 2.4fr) auto;
  gap: 0.38rem;
  align-items: center;
}

.event-editor-row input {
  min-width: 0;
  min-height: 1.95rem;
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 0.48rem;
  background: rgba(255, 255, 255, 0.055);
  color: #edf3fa;
  padding: 0 0.58rem;
  outline: none;
  font-size: 0.68rem;
}

.event-editor-row input:focus {
  border-color: rgba(86, 172, 255, 0.62);
}

.event-catalog-status {
  padding: 0.45rem 0.62rem 0.5rem;
  text-align: right;
}

.event-catalog-status.is-ok {
  color: #71d99a;
}

.event-catalog-status.is-error {
  color: #ff9292;
}

body.is-immersive {
  overflow: hidden;
  background: #000;
}

.is-immersive .site-chrome,
.is-immersive .device-topline,
.is-immersive .status-panel,
.is-immersive .console-strip,
.is-immersive .rollout-note {
  display: none;
}

.is-immersive .world-shell,
.is-immersive .spatial-device,
.is-immersive .screen-bezel,
.is-immersive .world-viewport {
  position: fixed;
  inset: 0;
  width: 100vw;
  height: 100vh;
  max-width: none;
  min-height: 0;
  margin: 0;
  padding: 0;
  border: 0;
  border-radius: 0;
  background: #000;
  box-shadow: none;
}

.is-immersive .world-shell {
  z-index: 100;
  display: block;
}

.is-immersive .spatial-device {
  isolation: isolate;
}

.is-immersive .world-viewport {
  aspect-ratio: auto;
}

.is-immersive .screen-noise {
  opacity: 0.1;
}

.is-immersive .viewport-tools {
  top: max(0.8rem, env(safe-area-inset-top));
  left: max(0.8rem, env(safe-area-inset-left));
}

.is-immersive .floating-state {
  top: max(0.8rem, env(safe-area-inset-top));
  right: max(0.8rem, env(safe-area-inset-right));
}

.is-immersive .runtime-log {
  position: fixed;
  top: max(3.55rem, calc(env(safe-area-inset-top) + 3rem));
  right: max(0.8rem, env(safe-area-inset-right));
  bottom: max(5.1rem, calc(env(safe-area-inset-bottom) + 4.5rem));
  width: min(26rem, 34vw);
  max-height: 42rem;
}

.is-immersive .controls-deck {
  position: fixed;
  inset: 0;
  z-index: 22;
  display: block;
  min-height: 0;
  padding: 0;
  pointer-events: none;
}

.is-immersive .controls-panel {
  position: fixed !important;
  inset: 0 !important;
  z-index: 22;
  width: 100vw !important;
  max-height: none !important;
  transform: none !important;
  margin: 0;
  border: 0;
  background: transparent;
  overflow: visible;
  pointer-events: none;
}

.is-immersive .controls-panel > .studio-panel-header {
  display: none;
}

.is-immersive .controls-panel > .studio-panel-body {
  display: block !important;
  pointer-events: none;
}

.is-immersive .realtime-control-bar,
.is-immersive .event-trigger-bar {
  position: fixed;
  z-index: 24;
  left: 50%;
  transform: translateX(-50%);
  padding: 0.35rem;
  border: 1px solid rgba(255, 255, 255, 0.12);
  border-radius: 999px;
  background: rgba(8, 12, 18, 0.78);
  backdrop-filter: blur(12px);
  pointer-events: auto;
}

.is-immersive .realtime-control-bar {
  top: max(0.75rem, env(safe-area-inset-top));
}

.is-immersive .event-trigger-bar {
  bottom: max(5rem, calc(env(safe-area-inset-bottom) + 4.5rem));
}

.is-immersive .stick-well {
  position: fixed;
  left: max(1rem, env(safe-area-inset-left));
  bottom: max(1rem, env(safe-area-inset-bottom));
  pointer-events: auto;
  opacity: 0.82;
}

.is-immersive .stick-right {
  right: max(1rem, env(safe-area-inset-right));
  left: auto;
}

.is-immersive .session-actions {
  position: fixed;
  top: max(0.8rem, env(safe-area-inset-top));
  left: 50%;
  transform: translateX(-50%);
  pointer-events: auto;
}

.is-immersive .source-dock {
  position: fixed;
  z-index: 21;
  left: 50%;
  bottom: max(0.8rem, env(safe-area-inset-bottom));
  width: min(58rem, calc(100vw - 15rem));
  padding: 0.45rem;
  transform: translateX(-50%);
  border: 1px solid rgba(255,255,255,0.12);
  background: rgba(8, 12, 18, 0.92);
  box-shadow: 0 1rem 3rem rgba(0,0,0,0.42);
}

.is-immersive .source-dock > .studio-panel-header,
.is-immersive .text-event-editor {
  display: none;
}

.is-immersive .source-dock.is-floating {
  right: auto !important;
  bottom: auto !important;
  transform: none;
}

.is-immersive .controls-centered + .source-dock {
  width: min(62rem, calc(100vw - 2rem));
}

.is-immersive .source-toolbar {
  margin-bottom: 0;
}

.is-immersive .thumb-tray,
.is-immersive .segment-input-status {
  display: none;
}

.is-immersive .immersive-hint {
  position: fixed;
  right: max(0.9rem, env(safe-area-inset-right));
  bottom: max(0.65rem, env(safe-area-inset-bottom));
  z-index: 23;
  display: block;
  color: rgba(225, 232, 241, 0.52);
  font-size: 0.58rem;
  font-weight: 800;
  letter-spacing: 0.05em;
  pointer-events: none;
}

@media (max-width: 820px) {
  .site-chrome {
    padding: 0 1rem;
  }

  .world-shell {
    padding: 4.55rem 0.55rem 0.75rem;
    align-items: start;
  }

  .spatial-device {
    width: 100%;
    border-radius: 1.3rem;
    padding: 0.62rem;
  }

  .screen-bezel {
    padding: 0.52rem;
    border-radius: 1rem;
  }

  .floating-state {
    top: 0.55rem;
    right: 0.55rem;
    padding: 0.34rem 0.58rem;
    font-size: 0.66rem;
  }

  .controls-deck {
    padding: 0.3rem 0.7rem 0.62rem;
  }

  .stick-well {
    width: 5rem;
    height: 5rem;
  }

  .source-toolbar {
    grid-template-columns: 1fr auto auto;
  }

  .segmented {
    grid-column: 1 / -1;
  }

  .prompt-input {
    grid-column: 1 / -1;
  }

  .thumb-tray {
    grid-template-columns: repeat(5, minmax(3.7rem, 1fr));
  }

  .segment-input-status {
    grid-template-columns: 1fr;
  }

  .event-editor-row {
    grid-template-columns: 1fr 1fr auto;
  }

  .event-editor-row .event-prompt {
    grid-column: 1 / -1;
    grid-row: 2;
  }

  .studio-panel-header,
  .panel-handle {
    cursor: default;
    touch-action: pan-y;
  }

  .studio-panel.is-floating:not(.runtime-log) {
    position: relative !important;
    inset: auto !important;
    z-index: auto !important;
    width: auto !important;
    max-height: none;
    transform: none !important;
  }

  .runtime-log.studio-panel.is-floating {
    position: absolute !important;
    top: 3.35rem !important;
    right: 0.55rem !important;
    bottom: 0.55rem !important;
    left: auto !important;
    width: calc(100% - 1.1rem) !important;
    transform: translate3d(0, 0.45rem, 0) !important;
  }

  .log-open .runtime-log.studio-panel.is-floating {
    transform: translate3d(0, 0, 0) !important;
  }

  .viewport-tool > span:not(.viewport-tool-icon):not(.live-log-dot):not(.log-count) {
    display: none;
  }

  .runtime-log,
  .is-immersive .runtime-log {
    top: 3.35rem;
    right: 0.55rem;
    bottom: 0.55rem;
    width: calc(100% - 1.1rem);
    max-height: 48vh;
  }

  .is-immersive .source-dock,
  .is-immersive .controls-centered + .source-dock {
    bottom: max(0.6rem, env(safe-area-inset-bottom));
    width: calc(100vw - 1.2rem);
  }

  .is-immersive .source-toolbar {
    grid-template-columns: auto auto 1fr;
  }

  .is-immersive .segmented,
  .is-immersive .file-chip,
  .is-immersive .load-button {
    display: none;
  }

  .is-immersive .prompt-input {
    grid-column: 1 / -1;
  }

  .is-immersive .runtime-log {
    bottom: 5.2rem;
  }

  .is-immersive .stick-well {
    bottom: max(5.1rem, calc(env(safe-area-inset-bottom) + 4.6rem));
    width: 4.25rem;
    height: 4.25rem;
  }

  .is-immersive .immersive-hint {
    display: none;
  }
}

@media (prefers-reduced-motion: reduce) {
  .runtime-log {
    transition: none;
  }
}
"""


@lru_cache(maxsize=1)
def _legacy_world_frontend_js() -> str:
    """Return the pre-WebRTC RPC client for downstream compatibility tests."""

    return r"""
const state = {
  modelLoaded: false,
  sessionId: null,
  mode: "image",
  seedImageData: null,
  seedImagePath: "",
  seedVideoData: null,
  seedVideoName: "",
  examples: [],
  controls: {
    w: false,
    a: false,
    s: false,
    d: false,
    camera_dx: 0,
    camera_dy: 0,
    l_click: false,
    r_click: false,
  },
  cameraHeld: { up: false, down: false, left: false, right: false },
  socket: null,
  socketReady: false,
  socketSeq: 0,
  socketPending: new Map(),
  reconnectTimer: null,
  lastFrameAt: 0,
  viewerFps: 0,
  stepping: false,
  sessionStartedAt: 0,
  stepTimer: null,
  stepQueued: false,
  pendingAction: null,
  lastStepSentAt: 0,
  imageDecodeSeq: 0,
  lastStatsText: "",
  lastStatsRenderAt: 0,
};

const emptyFrame =
  "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///ywAAAAAAQABAAACAUwAOw==";

const STEP_INTERVAL_MS = 160;
const STATS_INTERVAL_MS = 120;

const el = {
  root: document.body,
  viewport: document.getElementById("viewport"),
  image: document.getElementById("frameImage"),
  video: document.getElementById("frameVideo"),
  startOverlay: document.getElementById("startOverlay"),
  startButton: document.getElementById("startButton"),
  overlayStatus: document.getElementById("overlayStatus"),
  livePill: document.getElementById("livePill"),
  timer: document.getElementById("sessionTimer"),
  streamStats: document.getElementById("streamStats"),
  thumbTray: document.getElementById("thumbTray"),
  imageInput: document.getElementById("imageInput"),
  videoInput: document.getElementById("videoInput"),
  promptInput: document.getElementById("promptInput"),
  loadModelButton: document.getElementById("loadModelButton"),
  resetButton: document.getElementById("resetButton"),
  moveStick: document.getElementById("moveStick"),
  lookStick: document.getElementById("lookStick"),
};

async function api(path, options = {}) {
  const headers = options.body instanceof FormData ? {} : { "Content-Type": "application/json" };
  const res = await fetch(path, { headers, ...options });
  if (!res.ok) {
    let detail = await res.text();
    try {
      detail = JSON.parse(detail).error || detail;
    } catch {
      // Keep raw server text.
    }
    throw new Error(detail || `HTTP ${res.status}`);
  }
  return res.json();
}

function updateStreamStats(meta = {}) {
  const now = performance.now();
  if (state.lastFrameAt > 0) {
    state.viewerFps = 1000 / Math.max(now - state.lastFrameAt, 1);
  }
  if (meta.type === "frame") {
    state.lastFrameAt = now;
  }
  const renderFps = Number(meta.gpu_fps || meta.render_fps || 0);
  const latencyMs = Number(meta.latency_ms || 0);
  const connected = state.socketReady;
  const viewerText = state.viewerFps > 0 ? state.viewerFps.toFixed(1) : "--";
  const renderText = renderFps > 0 ? renderFps.toFixed(1) : "--";
  const latencyText = latencyMs > 0 ? ` / ${latencyMs.toFixed(0)}ms` : "";
  const statsText = `${connected ? "WS" : "HTTP"} VIEW ${viewerText} / GPU ${renderText}${latencyText}`;
  el.streamStats.classList.toggle("is-live", connected);
  if (statsText !== state.lastStatsText || now - state.lastStatsRenderAt > STATS_INTERVAL_MS) {
    el.streamStats.textContent = statsText;
    state.lastStatsText = statsText;
    state.lastStatsRenderAt = now;
  }
}

function connectStream() {
  if (state.socket && [WebSocket.OPEN, WebSocket.CONNECTING].includes(state.socket.readyState)) {
    return;
  }
  const scheme = window.location.protocol === "https:" ? "wss" : "ws";
  const socket = new WebSocket(`${scheme}://${window.location.host}/api/stream`);
  state.socket = socket;
  updateStreamStats();

  socket.addEventListener("open", () => {
    state.socketReady = true;
    updateStreamStats();
  });

  socket.addEventListener("message", (event) => {
    let message;
    try {
      message = JSON.parse(String(event.data || "{}"));
    } catch (err) {
      console.error(err);
      return;
    }
    updateStreamStats(message);
    const pending = state.socketPending.get(message.id);
    if (!pending) {
      return;
    }
    state.socketPending.delete(message.id);
    if (message.type === "error") {
      pending.reject(new Error(message.error || "WebSocket request failed"));
    } else {
      pending.resolve(message.payload || {});
    }
  });

  socket.addEventListener("close", () => {
    state.socketReady = false;
    for (const pending of state.socketPending.values()) {
      pending.reject(new Error("WebSocket closed"));
    }
    state.socketPending.clear();
    updateStreamStats();
    clearTimeout(state.reconnectTimer);
    state.reconnectTimer = setTimeout(connectStream, 1600);
  });

  socket.addEventListener("error", () => {
    state.socketReady = false;
    updateStreamStats();
  });
}

function wsRequest(type, payload) {
  if (!state.socketReady || !state.socket || state.socket.readyState !== WebSocket.OPEN) {
    return Promise.reject(new Error("WebSocket is not ready"));
  }
  const id = `${Date.now()}-${++state.socketSeq}`;
  return new Promise((resolve, reject) => {
    state.socketPending.set(id, { resolve, reject });
    state.socket.send(JSON.stringify({ id, type, payload }));
  });
}

async function interactiveRequest(type, payload, httpFallback) {
  try {
    return await wsRequest(type, payload);
  } catch (err) {
    updateStreamStats();
    return httpFallback();
  }
}

function setStatus(text, live = false, isError = false) {
  el.livePill.textContent = text;
  el.livePill.title = text;
  el.livePill.classList.toggle("is-live", live);
  el.livePill.style.color = isError ? "var(--danger)" : "";
  el.overlayStatus.textContent = text;
  el.overlayStatus.title = text;
}

function errorStatus(label, err) {
  const raw = err && err.message ? String(err.message) : String(err || "");
  const detail = raw.replace(/\s+/g, " ").trim();
  return detail ? `${label}: ${detail}`.slice(0, 220) : label;
}

function setMode(mode) {
  state.mode = mode;
  el.root.classList.toggle("image-mode", mode === "image");
  el.root.classList.toggle("video-mode", mode === "video");
  document.querySelectorAll(".mode-tab").forEach((btn) => {
    btn.classList.toggle("is-active", btn.dataset.mode === mode);
  });
}

function commitImageFrame(src, seq) {
  requestAnimationFrame(() => {
    if (seq !== state.imageDecodeSeq) {
      return;
    }
    el.image.src = src || emptyFrame;
    el.image.classList.add("is-visible");
  });
}

async function decodeImageFrame(src, seq) {
  if (!src || src === emptyFrame) {
    commitImageFrame(src, seq);
    return;
  }
  const img = new Image();
  img.decoding = "async";
  img.src = src;
  try {
    if (typeof img.decode === "function") {
      await img.decode();
    } else if (!img.complete) {
      await new Promise((resolve, reject) => {
        img.onload = resolve;
        img.onerror = reject;
      });
    }
  } catch {
    // If async decode fails, still swap the browser-managed image source.
  }
  commitImageFrame(src, seq);
}

function showImage(src) {
  el.video.pause();
  el.video.removeAttribute("src");
  el.video.load();
  el.video.classList.remove("is-visible");
  const seq = ++state.imageDecodeSeq;
  void decodeImageFrame(src || emptyFrame, seq);
}

function showVideo(src) {
  state.imageDecodeSeq += 1;
  el.image.classList.remove("is-visible");
  el.video.loop = false;
  if (el.video.src !== src) {
    el.video.preload = "auto";
    el.video.src = src;
    el.video.load();
  }
  try {
    el.video.currentTime = 0;
  } catch {
    // Some browsers reject seeking before metadata is ready.
  }
  el.video.classList.add("is-visible");
  el.video.play().catch(() => {});
}

function showPayload(payload) {
  updateStreamStats({ type: "frame", ...(payload || {}) });
  const stamp = `t=${Date.now()}`;
  if (payload.video_url) {
    showVideo(`${payload.video_url}${payload.video_url.includes("?") ? "&" : "?"}${stamp}`);
  } else if (payload.frame_base64) {
    showImage(`data:image/png;base64,${payload.frame_base64}`);
  } else if (payload.image_url) {
    showImage(`${payload.image_url}${payload.image_url.includes("?") ? "&" : "?"}${stamp}`);
  }
}

function updateStartOverlay() {
  const running = !!state.sessionId;
  el.startOverlay.classList.add("is-hidden");
  el.startButton.classList.toggle("is-hidden", running);
  el.resetButton.classList.toggle("is-hidden", !running);
}

function readFileAsDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result || ""));
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

function renderExamples(examples) {
  el.thumbTray.innerHTML = "";
  for (const item of examples.slice(0, 9)) {
    const button = document.createElement("button");
    button.className = "thumb-card";
    button.type = "button";
    button.dataset.path = item.path;
    button.innerHTML = `<img src="${item.url}" alt=""><span>${item.label}</span>`;
    button.addEventListener("click", () => {
      document.querySelectorAll(".thumb-card").forEach((node) => node.classList.remove("is-active"));
      button.classList.add("is-active");
      state.seedImageData = null;
      state.seedImagePath = item.path;
      state.seedVideoData = null;
      state.seedVideoName = "";
      setMode("image");
      state.sessionId = null;
      showImage(item.url);
      updateStartOverlay();
      setStatus("INPUT READY");
    });
    el.thumbTray.appendChild(button);
  }
}

async function loadModel() {
  setStatus("LOADING");
  try {
    const data = await api("/api/models/load", {
      method: "POST",
      body: JSON.stringify({}),
    });
    state.modelLoaded = true;
    el.loadModelButton.textContent = "READY";
    setStatus(data.loaded ? "READY" : "STAGED");
    return data;
  } catch (err) {
    state.modelLoaded = false;
    el.loadModelButton.textContent = "LOAD";
    setStatus(errorStatus("LOAD ERROR", err), false, true);
    throw err;
  }
}

function sessionPayload() {
  return {
    mode: state.mode,
    prompt: el.promptInput.value || "",
    init_image_base64: state.seedImageData,
    init_image_path: state.seedImagePath,
    init_video_base64: state.seedVideoData,
    video_file_name: state.seedVideoName,
  };
}

async function startSession() {
  if (state.mode === "image" && !state.seedImageData && !state.seedImagePath) {
    setStatus("SELECT IMAGE", false, true);
    return;
  }
  if (state.mode === "video" && !state.seedVideoData) {
    setStatus("SELECT VIDEO", false, true);
    return;
  }
  if (!state.modelLoaded) {
    await loadModel();
  }
  setStatus("STARTING", true);
  const payload = sessionPayload();
  const data = await interactiveRequest("start", payload, () =>
    api("/api/sessions/start", {
      method: "POST",
      body: JSON.stringify(payload),
    })
  );
  state.sessionId = data.session_id;
  state.sessionStartedAt = Date.now();
  showPayload(data);
  updateStartOverlay();
  setStatus("RUNNING", true);
  if (hasControls()) {
    scheduleStep();
  }
}

async function resetSession() {
  if (!state.sessionId) {
    return;
  }
  setStatus("RESETTING");
  const payload = { session_id: state.sessionId };
  await interactiveRequest("reset", payload, () =>
    api("/api/sessions/reset", {
      method: "POST",
      body: JSON.stringify(payload),
    })
  );
  state.sessionId = null;
  clearScheduledStep();
  updateStartOverlay();
  setStatus("FROZEN");
}

function controlsSnapshot() {
  return { ...state.controls };
}

function hasControls(action = state.controls) {
  return (
    action.w ||
    action.a ||
    action.s ||
    action.d ||
    action.l_click ||
    action.r_click ||
    Math.abs(action.camera_dx || 0) > 0.08 ||
    Math.abs(action.camera_dy || 0) > 0.08
  );
}

function scheduleStep(delay = 0, action = null) {
  if (action && hasControls(action)) {
    state.pendingAction = action;
  }
  if (!state.sessionId || (!hasControls() && !state.pendingAction)) {
    return;
  }
  state.stepQueued = true;
  if (state.stepTimer) {
    return;
  }
  state.stepTimer = window.setTimeout(() => {
    state.stepTimer = null;
    if (!state.stepQueued) {
      return;
    }
    state.stepQueued = false;
    void stepLoop();
  }, delay);
}

function clearScheduledStep() {
  state.stepQueued = false;
  state.pendingAction = null;
  if (state.stepTimer) {
    window.clearTimeout(state.stepTimer);
    state.stepTimer = null;
  }
}

function noteControlsChanged() {
  if (state.sessionId && hasControls()) {
    scheduleStep(0, controlsSnapshot());
  }
}

async function stepLoop() {
  if (!state.sessionId) {
    state.stepQueued = false;
    return;
  }
  const action = state.pendingAction ? { ...state.pendingAction } : controlsSnapshot();
  if (!hasControls(action)) {
    state.stepQueued = false;
    state.pendingAction = null;
    return;
  }
  if (state.stepping) {
    state.stepQueued = true;
    state.pendingAction = action;
    return;
  }
  const now = performance.now();
  const wait = Math.max(0, STEP_INTERVAL_MS - (now - state.lastStepSentAt));
  if (wait > 0) {
    scheduleStep(wait);
    return;
  }
  state.stepping = true;
  state.pendingAction = null;
  state.lastStepSentAt = now;
  setStatus("INFERRING STEP", true);
  try {
    const payload = {
      session_id: state.sessionId,
      action,
    };
    const data = await interactiveRequest("step", payload, () =>
      api("/api/sessions/step", {
        method: "POST",
        body: JSON.stringify(payload),
      })
    );
    showPayload(data);
    setStatus("STEP READY", true);
  } catch (err) {
    setStatus("STEP ERROR", false, true);
    console.error(err);
  } finally {
    state.stepping = false;
    if (state.sessionId && state.pendingAction) {
      scheduleStep(STEP_INTERVAL_MS);
    } else if (state.sessionId && hasControls()) {
      scheduleStep(STEP_INTERVAL_MS, controlsSnapshot());
    }
  }
}

function resetMoveControls() {
  state.controls.w = false;
  state.controls.a = false;
  state.controls.s = false;
  state.controls.d = false;
  updateControlVisuals();
}

function resetLookControls() {
  state.controls.camera_dx = 0;
  state.controls.camera_dy = 0;
  for (const key of ["up", "down", "left", "right"]) {
    state.cameraHeld[key] = false;
  }
  updateControlVisuals();
}

function moveVectorFromKeys() {
  return {
    dx: (state.controls.d ? 1 : 0) - (state.controls.a ? 1 : 0),
    dy: (state.controls.s ? 1 : 0) - (state.controls.w ? 1 : 0),
  };
}

function syncMoveStickFromKeys() {
  const { dx, dy } = moveVectorFromKeys();
  updateStickVisual(el.moveStick, Math.max(-1, Math.min(1, dx)), Math.max(-1, Math.min(1, dy)));
}

function updateControlVisuals() {
}

function updateCameraFromHeld() {
  const dx = (state.cameraHeld.right ? 1 : 0) + (state.cameraHeld.left ? -1 : 0);
  const dy = (state.cameraHeld.down ? 1 : 0) + (state.cameraHeld.up ? -1 : 0);
  state.controls.camera_dx = Math.max(-1, Math.min(1, dx));
  state.controls.camera_dy = Math.max(-1, Math.min(1, dy));
  updateStickVisual(el.lookStick, state.controls.camera_dx, state.controls.camera_dy);
  updateControlVisuals();
}

function updateStickVisual(stick, dx, dy) {
  if (!stick) {
    return;
  }
  stick.style.setProperty("--stick-offset-x", `${dx * 1.45}rem`);
  stick.style.setProperty("--stick-offset-y", `${dy * 1.45}rem`);
  stick.classList.toggle("is-active", Math.abs(dx) > 0.05 || Math.abs(dy) > 0.05);
}

function applyMoveVector(dx, dy) {
  resetMoveControls();
  const threshold = 0.22;
  if (dy < -threshold) state.controls.w = true;
  if (dy > threshold) state.controls.s = true;
  if (dx < -threshold) state.controls.a = true;
  if (dx > threshold) state.controls.d = true;
  updateStickVisual(el.moveStick, dx, dy);
  updateControlVisuals();
}

function applyLookVector(dx, dy) {
  state.controls.camera_dx = Math.abs(dx) > 0.08 ? dx : 0;
  state.controls.camera_dy = Math.abs(dy) > 0.08 ? dy : 0;
  updateStickVisual(el.lookStick, state.controls.camera_dx, state.controls.camera_dy);
  updateControlVisuals();
}

function bindJoystick(stick, applyVector, releaseVector) {
  if (!stick) {
    return;
  }
  let pointerActive = false;
  let mouseActive = false;
  const updateFromEvent = (event) => {
    const rect = stick.getBoundingClientRect();
    const radius = Math.max(Math.min(rect.width, rect.height) / 2, 1);
    const centerX = rect.left + rect.width / 2;
    const centerY = rect.top + rect.height / 2;
    let dx = (event.clientX - centerX) / radius;
    let dy = (event.clientY - centerY) / radius;
    const length = Math.hypot(dx, dy);
    if (length > 1) {
      dx /= length;
      dy /= length;
    }
    applyVector(dx, dy);
    noteControlsChanged();
  };
  const release = () => {
    const action = controlsSnapshot();
    pointerActive = false;
    mouseActive = false;
    releaseVector();
    updateStickVisual(stick, 0, 0);
    if (hasControls(action)) {
      scheduleStep(0, action);
    }
  };
  stick.addEventListener("pointerdown", (event) => {
    event.preventDefault();
    pointerActive = true;
    stick.setPointerCapture(event.pointerId);
    updateFromEvent(event);
  });
  stick.addEventListener("pointermove", (event) => {
    if (!stick.hasPointerCapture(event.pointerId)) {
      return;
    }
    event.preventDefault();
    updateFromEvent(event);
  });
  for (const type of ["pointerup", "pointercancel", "lostpointercapture"]) {
    stick.addEventListener(type, release);
  }
  stick.addEventListener("mousedown", (event) => {
    if (pointerActive) {
      return;
    }
    event.preventDefault();
    mouseActive = true;
    updateFromEvent(event);
  });
  window.addEventListener("mousemove", (event) => {
    if (!mouseActive || pointerActive) {
      return;
    }
    event.preventDefault();
    updateFromEvent(event);
  });
  window.addEventListener("mouseup", () => {
    if (mouseActive && !pointerActive) {
      release();
    }
  });
  stick.addEventListener("click", (event) => {
    event.preventDefault();
    updateFromEvent(event);
    requestAnimationFrame(release);
  });
}

function shouldIgnoreControlKey(event) {
  const target = event.target;
  if (!(target instanceof Element)) {
    return false;
  }
  return target.isContentEditable || ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName);
}

function bindControls() {
  bindJoystick(el.moveStick, applyMoveVector, resetMoveControls);
  bindJoystick(el.lookStick, applyLookVector, resetLookControls);

  const keyMap = {
    w: ["controls", "w"],
    a: ["controls", "a"],
    s: ["controls", "s"],
    d: ["controls", "d"],
    i: ["cameraHeld", "up"],
    k: ["cameraHeld", "down"],
    j: ["cameraHeld", "left"],
    l: ["cameraHeld", "right"],
    arrowup: ["cameraHeld", "up"],
    arrowdown: ["cameraHeld", "down"],
    arrowleft: ["cameraHeld", "left"],
    arrowright: ["cameraHeld", "right"],
  };

  window.addEventListener("keydown", (event) => {
    if (shouldIgnoreControlKey(event)) {
      return;
    }
    const hit = keyMap[event.key.toLowerCase()];
    if (!hit || event.repeat) {
      return;
    }
    event.preventDefault();
    state[hit[0]][hit[1]] = true;
    syncMoveStickFromKeys();
    updateCameraFromHeld();
    updateControlVisuals();
    noteControlsChanged();
  });

  window.addEventListener("keyup", (event) => {
    if (shouldIgnoreControlKey(event)) {
      return;
    }
    const hit = keyMap[event.key.toLowerCase()];
    if (!hit) {
      return;
    }
    const action = controlsSnapshot();
    event.preventDefault();
    state[hit[0]][hit[1]] = false;
    syncMoveStickFromKeys();
    updateCameraFromHeld();
    updateControlVisuals();
    if (hasControls(action)) {
      scheduleStep(0, action);
    }
  });

  window.addEventListener("blur", () => {
    resetMoveControls();
    resetLookControls();
    updateStickVisual(el.moveStick, 0, 0);
    updateStickVisual(el.lookStick, 0, 0);
    updateControlVisuals();
    clearScheduledStep();
  });
  updateControlVisuals();
}

function bindEvents() {
  document.querySelectorAll(".mode-tab").forEach((btn) => {
    btn.addEventListener("click", () => setMode(btn.dataset.mode));
  });

  el.imageInput.addEventListener("change", async (event) => {
    const [file] = event.target.files || [];
    if (!file) return;
    const dataUrl = await readFileAsDataUrl(file);
    state.seedImageData = dataUrl;
    state.seedImagePath = "";
    state.seedVideoData = null;
    state.seedVideoName = "";
    state.sessionId = null;
    setMode("image");
    showImage(dataUrl);
    updateStartOverlay();
    setStatus("INPUT READY");
  });

  el.videoInput.addEventListener("change", async (event) => {
    const [file] = event.target.files || [];
    if (!file) return;
    const dataUrl = await readFileAsDataUrl(file);
    state.seedVideoData = dataUrl;
    state.seedVideoName = file.name || "input.webm";
    state.seedImageData = null;
    state.seedImagePath = "";
    state.sessionId = null;
    setMode("video");
    showVideo(dataUrl);
    updateStartOverlay();
    setStatus("VIDEO READY");
  });

  el.loadModelButton.addEventListener("click", () => loadModel().catch((err) => {
    setStatus(errorStatus("LOAD ERROR", err), false, true);
    console.error(err);
  }));
  el.startButton.addEventListener("click", () => startSession().catch((err) => {
    setStatus(errorStatus("START ERROR", err), false, true);
    console.error(err);
  }));
  el.resetButton.addEventListener("click", () => resetSession().catch((err) => {
    setStatus(errorStatus("RESET ERROR", err), false, true);
    console.error(err);
  }));
  el.video.addEventListener("ended", () => {
    el.video.pause();
    if (Number.isFinite(el.video.duration) && el.video.duration > 0) {
      try {
        el.video.currentTime = Math.max(0, el.video.duration - 0.04);
      } catch {
        // Keeping the ended frame visible is enough if the seek is denied.
      }
    }
  });
  bindControls();
}

function updateTimer() {
  if (!state.sessionStartedAt) {
    el.timer.textContent = "00:00";
    return;
  }
  const elapsed = Math.floor((Date.now() - state.sessionStartedAt) / 1000);
  const mm = String(Math.floor(elapsed / 60)).padStart(2, "0");
  const ss = String(elapsed % 60).padStart(2, "0");
  el.timer.textContent = `${mm}:${ss}`;
}

async function boot() {
  el.root.classList.add("image-mode");
  showImage(emptyFrame);
  bindEvents();
  connectStream();
  const data = await api("/api/session");
  state.examples = data.examples || [];
  renderExamples(state.examples);
  if (state.examples.length) {
    const first = el.thumbTray.querySelector(".thumb-card");
    if (first) first.click();
  } else {
    setStatus("LOAD INPUT");
  }
  if (hasControls()) {
    scheduleStep();
  }
  setInterval(updateTimer, 500);
}

    boot().catch((err) => {
  setStatus("BOOT ERROR", false, true);
  console.error(err);
});
"""


@lru_cache(maxsize=1)
def world_frontend_js() -> str:
    """Return the persistent WebRTC realtime frontend client."""

    from .world_realtime_client import WORLD_REALTIME_CLIENT_JS

    return WORLD_REALTIME_CLIENT_JS


class WorldFrontendHandler(BaseHTTPRequestHandler):
    """HTTP API and static assets for the standalone world frontend."""

    server_version = "WorldFoundryWorldFrontend/1.0"

    def __init__(self, *args: Any, state: WorldFrontendState, **kwargs: Any) -> None:
        self.state = state
        super().__init__(*args, **kwargs)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        parsed = urlparse(self.path)
        if parsed.path == "/api/stream" and self.headers.get("Upgrade", "").lower() == "websocket":
            self._handle_websocket()
            return
        with self.state.telemetry.track(parsed.path):
            self._handle_get(parsed)

    def do_HEAD(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        parsed = urlparse(self.path)
        with self.state.telemetry.track(parsed.path):
            self._handle_head(parsed)

    def _handle_get(self, parsed) -> None:
        if parsed.path in {"", "/"}:
            self._send_html(world_frontend_html(self.state.entry, self.state.launch_config))
            return
        if parsed.path == "/world.css":
            self._send_text(world_frontend_css(), "text/css; charset=utf-8", cache_control="public, max-age=3600")
            return
        if parsed.path == "/world.js":
            self._send_text(world_frontend_js(), "text/javascript; charset=utf-8", cache_control="public, max-age=3600")
            return
        if parsed.path in {"/favicon.ico", "/favicon.svg"}:
            self._send_text(world_favicon_svg(), "image/svg+xml; charset=utf-8", cache_control="public, max-age=86400")
            return
        if parsed.path == "/healthz":
            self._send_json(self._health_payload())
            return
        if parsed.path == "/api/session":
            self._send_json(self._session_payload())
            return
        if parsed.path == "/api/datasets":
            self._send_json({"datasets": [self._demo_dataset_payload()], "examples": self._example_payloads()})
            return
        if parsed.path == "/api/file":
            query = parse_qs(parsed.query)
            raw_path = (query.get("path") or [""])[0]
            path = Path(unquote(raw_path)).expanduser().resolve()
            if not path.exists() or not path_allowed(path, self.state.allowed_roots):
                self.send_error(HTTPStatus.NOT_FOUND, "File not found or not allowed.")
                return
            self._send_file(path, mimetypes.guess_type(path.name)[0] or "application/octet-stream")
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found.")

    def _handle_head(self, parsed) -> None:
        if parsed.path in {"", "/"}:
            self._send_head_text(
                world_frontend_html(self.state.entry, self.state.launch_config),
                "text/html; charset=utf-8",
            )
            return
        if parsed.path == "/world.css":
            self._send_head_text(
                world_frontend_css(),
                "text/css; charset=utf-8",
                cache_control="public, max-age=3600",
            )
            return
        if parsed.path == "/world.js":
            self._send_head_text(
                world_frontend_js(),
                "text/javascript; charset=utf-8",
                cache_control="public, max-age=3600",
            )
            return
        if parsed.path in {"/favicon.ico", "/favicon.svg"}:
            self._send_head_text(
                world_favicon_svg(),
                "image/svg+xml; charset=utf-8",
                cache_control="public, max-age=86400",
            )
            return
        if parsed.path == "/healthz":
            self._send_head_json(self._health_payload())
            return
        if parsed.path == "/api/session":
            self._send_head_json(self._session_payload())
            return
        if parsed.path == "/api/datasets":
            self._send_head_json(
                {"datasets": [self._demo_dataset_payload()], "examples": self._example_payloads()}
            )
            return
        if parsed.path == "/api/file":
            query = parse_qs(parsed.query)
            raw_path = (query.get("path") or [""])[0]
            path = Path(unquote(raw_path)).expanduser().resolve()
            if not path.exists() or not path_allowed(path, self.state.allowed_roots):
                self.send_error(HTTPStatus.NOT_FOUND, "File not found or not allowed.")
                return
            self._send_head_file(path, mimetypes.guess_type(path.name)[0] or "application/octet-stream")
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found.")

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        parsed = urlparse(self.path)
        with self.state.telemetry.track(parsed.path):
            try:
                payload = self._read_json()
                if parsed.path == "/api/models/load":
                    self._preflight_model_load()
                    with self.state.lock:
                        self.state.model_loaded = True
                    self._send_json(
                        {
                            "loaded": True,
                            "preflighted": True,
                            "model_id": self.state.entry.model_id,
                            "device": self.state.launch_config.device or "cuda",
                        }
                    )
                    return
                if parsed.path == "/api/datasets/random-image":
                    image = random.choice(self.state.demo_images) if self.state.demo_images else None
                    if image is None:
                        raise ValueError("No demo images are available.")
                    self._send_json({"image_base64": _data_url_from_file(image), "file": str(image)})
                    return
                if parsed.path == "/api/sessions/start":
                    self._send_json(self._start_session(payload))
                    return
                if parsed.path == "/api/sessions/step":
                    self._send_json(self._step_session(payload))
                    return
                if parsed.path == "/api/sessions/reset":
                    self._send_json(self._reset_session(payload))
                    return
            except ValueError as exc:
                self._send_json({"error": f"{type(exc).__name__}: {exc}"}, status=HTTPStatus.BAD_REQUEST)
                return
            except (BrokenPipeError, ConnectionResetError):
                return
            except Exception as exc:
                self.state.telemetry.record_error(f"{type(exc).__name__}: {exc}")
                traceback.print_exc()
                self._send_json({"error": f"{type(exc).__name__}: {exc}"}, status=HTTPStatus.BAD_REQUEST)
                return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found.")

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[world] {self.address_string()} - {fmt % args}", flush=True)

    def _handle_websocket(self) -> None:
        key = self.headers.get("Sec-WebSocket-Key", "").strip()
        if not key:
            self.send_error(HTTPStatus.BAD_REQUEST, "Missing WebSocket key.")
            return
        accept = base64.b64encode(hashlib.sha1((key + WEBSOCKET_GUID).encode("ascii")).digest()).decode("ascii")
        self.send_response(HTTPStatus.SWITCHING_PROTOCOLS)
        self.send_header("Upgrade", "websocket")
        self.send_header("Connection", "Upgrade")
        self.send_header("Sec-WebSocket-Accept", accept)
        self.end_headers()
        self.close_connection = True
        try:
            self._send_ws_json(
                {
                    "type": "hello",
                    "payload": {
                        "model_id": self.state.entry.model_id,
                        "display_name": self.state.entry.display_name,
                        "transport": "websocket",
                    },
                }
            )
            while True:
                message = self._read_ws_json()
                if message is None:
                    break
                response = self._handle_ws_request(message)
                if response is not None:
                    self._send_ws_json(response)
        except (BrokenPipeError, ConnectionResetError):
            return

    def _handle_ws_request(self, message: dict[str, Any]) -> dict[str, Any] | None:
        request_id = message.get("id")
        request_type = str(message.get("type") or "")
        payload = message.get("payload") if isinstance(message.get("payload"), dict) else {}
        started = time.perf_counter()
        track_path = f"ws:{request_type or 'unknown'}"
        try:
            with self.state.telemetry.track(track_path):
                try:
                    if request_type == "ping":
                        result = {"ok": True}
                        response_type = "pong"
                    elif request_type == "start":
                        result = self._start_session(payload)
                        result = _embed_frame_payload(result)
                        response_type = "frame"
                    elif request_type == "step":
                        result = self._step_session(payload)
                        result = _embed_frame_payload(result)
                        response_type = "frame"
                    elif request_type == "reset":
                        result = self._reset_session(payload)
                        response_type = "reset"
                    else:
                        raise ValueError(f"Unsupported WebSocket request type: {request_type}")
                except ValueError as exc:
                    return {
                        "id": request_id,
                        "type": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
            latency_ms = (time.perf_counter() - started) * 1000.0
            gpu_fps = 1000.0 / latency_ms if latency_ms > 0 else 0.0
            return {
                "id": request_id,
                "type": response_type,
                "payload": result,
                "latency_ms": latency_ms,
                "gpu_fps": gpu_fps,
            }
        except Exception as exc:
            self.state.telemetry.record_error(f"{type(exc).__name__}: {exc}")
            traceback.print_exc()
            return {
                "id": request_id,
                "type": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }

    def _read_ws_json(self) -> dict[str, Any] | None:
        frame = self._read_ws_frame()
        if frame is None:
            return None
        opcode, payload = frame
        if opcode == 0x8:
            return None
        if opcode == 0x9:
            self._send_ws_frame(payload, opcode=0xA)
            return self._read_ws_json()
        if opcode != 0x1:
            return self._read_ws_json()
        data = json.loads(payload.decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("WebSocket payload must be a JSON object.")
        return data

    def _read_ws_frame(self) -> tuple[int, bytes] | None:
        header = self.rfile.read(2)
        if len(header) < 2:
            return None
        first, second = header
        opcode = first & 0x0F
        masked = bool(second & 0x80)
        length = second & 0x7F
        if length == 126:
            length = struct.unpack("!H", self.rfile.read(2))[0]
        elif length == 127:
            length = struct.unpack("!Q", self.rfile.read(8))[0]
        mask = self.rfile.read(4) if masked else b""
        payload = self.rfile.read(length) if length else b""
        if masked:
            payload = bytes(byte ^ mask[index % 4] for index, byte in enumerate(payload))
        return opcode, payload

    def _send_ws_json(self, payload: dict[str, Any]) -> None:
        self._send_ws_frame(json.dumps(payload, ensure_ascii=False).encode("utf-8"), opcode=0x1)

    def _send_ws_frame(self, payload: bytes, *, opcode: int = 0x1) -> None:
        length = len(payload)
        header = bytearray([0x80 | opcode])
        if length < 126:
            header.append(length)
        elif length < 65536:
            header.append(126)
            header.extend(struct.pack("!H", length))
        else:
            header.append(127)
            header.extend(struct.pack("!Q", length))
        self.wfile.write(bytes(header) + payload)
        self.wfile.flush()

    def _health_payload(self) -> dict[str, object]:
        with self.state.manager.lock:
            cached_model_ids = [
                context.entry.model_id
                for context in self.state.manager.pipeline_cache.values()
            ]
        with self.state.lock:
            model_loaded = self.state.model_loaded
            session_count = len(self.state.sessions)
        runtime_ready = self.state.entry.model_id in cached_model_ids
        return self.state.telemetry.snapshot(
            frontend="world",
            model_id=self.state.entry.model_id,
            display_name=self.state.entry.display_name,
            runtime_ready=runtime_ready,
            model_loaded=model_loaded,
            session_count=session_count,
            max_sessions=self.state.max_sessions,
            demo_image_count=len(self.state.demo_images),
            cached_pipeline_count=len(cached_model_ids),
        )

    def _session_payload(self) -> dict[str, Any]:
        entry = self.state.entry
        return {
            "model": {
                "id": entry.model_id,
                "label": entry.display_name,
                "supports_stream": entry.supports_stream,
                "template": interface_spec_for_entry(entry).template_id,
            },
            "variant": self.state.launch_config.variant_id or "",
            "transport": {
                "websocket_path": "/api/stream",
                "http_fallback": True,
            },
            "examples": self._example_payloads(),
        }

    def _preflight_model_load(self) -> None:
        """Validate the active process can import the selected model runtime."""

        missing_imports = _missing_runtime_validation_imports(self.state.entry)
        if missing_imports:
            raise ValueError(str(_runtime_dependency_error(self.state.entry, missing_imports)))
        try:
            self.state.manager.import_pipeline_class(self.state.entry)
        except ModuleNotFoundError as exc:
            raise ValueError(_model_import_error_message(self.state.entry, exc)) from exc
        except ImportError as exc:
            raise ValueError(_model_import_error_message(self.state.entry, exc)) from exc

    def _demo_dataset_payload(self) -> dict[str, Any]:
        return {
            "id": "studio_demo",
            "label": "Studio Demo",
            "num_images": len(self.state.demo_images),
        }

    def _example_payloads(self) -> list[dict[str, str]]:
        examples: list[dict[str, str]] = []
        for index, path in enumerate(_repeat_to_slots(self.state.demo_images, 9), start=1):
            examples.append(
                {
                    "id": f"demo-{index:02d}",
                    "label": _reference_image_label(path, index),
                    "path": str(path),
                    "url": _file_url(path),
                }
            )
        return examples

    def _start_session(self, payload: dict[str, Any]) -> dict[str, Any]:
        mode = str(payload.get("mode") or "image").strip().lower()
        prompt = str(payload.get("prompt") or self.state.entry.default_prompt or "")
        image = self._image_from_payload(payload)
        video_path = self._video_from_payload(payload) if mode == "video" else ""
        if mode == "image" and image is None:
            raise ValueError("Image world mode expects an uploaded image or selected example.")
        if mode == "video" and not video_path:
            raise ValueError("Video world mode expects an uploaded video.")

        session = WorldSession(
            session_id=uuid.uuid4().hex,
            mode=mode,
            prompt=prompt,
            seed_image=image.copy() if image is not None else None,
            seed_video_path=video_path,
        )
        action = "init" if _uses_state_init(self.state.entry) else "run"
        interactions_text = "" if action == "init" else _initial_start_interactions(self.state.entry)
        record = self._run_model(
            action=action,
            session=session,
            interactions_text=interactions_text,
            image=image,
            video=video_path or None,
        )
        session.last_record = record
        with self.state.lock:
            self.state.sessions[session.session_id] = session
            self._prune_sessions_locked()
        response = _record_payload(record)
        response.update({"session_id": session.session_id, "mode": mode})
        return response

    def _step_session(self, payload: dict[str, Any]) -> dict[str, Any]:
        session_id = str(payload.get("session_id") or "")
        with self.state.lock:
            session = self.state.sessions.get(session_id)
        if session is None:
            raise ValueError("Unknown session.")
        if not self.state.entry.supports_stream:
            raise ValueError(f"{self.state.entry.display_name} does not expose stream controls.")

        interactions_text = controls_to_interactions(payload.get("action") or {})
        if not interactions_text:
            return _record_payload(session.last_record) if session.last_record else {"session_id": session_id}

        # Vary the diffusion seed every step. Reusing a single fixed seed across an
        # autoregressive frame-chained rollout makes the same noise pattern recur each
        # step, which compounds into a locked-in artifact (the scene drifts dark /
        # "crystalline" after a long key-hold). Advancing the seed per step (as the
        # official LingBot interactive runtime does) keeps long rollouts stable.
        with self.state.lock:
            session.step_count += 1
            step_seed = WORLD_STREAM_BASE_SEED + session.step_count

        last_frame = _open_image(session.last_record.preview_image) if session.last_record else None
        record = self._run_model(
            action="stream",
            session=session,
            interactions_text=interactions_text,
            image=session.seed_image,
            video=session.seed_video_path or None,
            last_frame=last_frame,
            seed_override=step_seed,
        )
        with self.state.lock:
            if session_id in self.state.sessions:
                session.last_record = record
        response = _record_payload(record)
        response.update({"session_id": session_id, "interactions": interactions_text})
        return response

    def _reset_session(self, payload: dict[str, Any]) -> dict[str, Any]:
        session_id = str(payload.get("session_id") or "")
        with self.state.lock:
            session = self.state.sessions.pop(session_id, None)
        message = self.state.manager.reset_cached_model(self.state.entry.model_id)
        return {"session_id": session_id, "reset": session is not None, "message": message}

    def _prune_sessions_locked(self) -> None:
        overflow = len(self.state.sessions) - max(1, self.state.max_sessions)
        if overflow <= 0:
            return
        stale_ids = sorted(
            self.state.sessions,
            key=lambda session_id: self.state.sessions[session_id].created_at,
        )[:overflow]
        for session_id in stale_ids:
            self.state.sessions.pop(session_id, None)

    def _run_model(
        self,
        *,
        action: str,
        session: WorldSession,
        interactions_text: str,
        image: Image.Image | None,
        video: str | None,
        last_frame: Image.Image | None = None,
        seed_override: int | None = None,
    ) -> RunRecord:
        load_kwargs_text, call_kwargs_text, model_ref = _launch_runtime_overrides(
            self.state.entry,
            self.state.launch_config,
            interactions_text=interactions_text,
        )
        if seed_override is not None:
            call_kwargs_text = _override_call_kwargs_seed(call_kwargs_text, seed_override)
        return self.state.manager.run(
            model_id=self.state.entry.model_id,
            action=action,
            prompt=session.prompt,
            input_path="",
            image=image,
            video=video,
            last_frame=last_frame,
            reference_files=None,
            interactions_text=interactions_text,
            camera_view_text="",
            task_type=self.state.entry.default_task_type or "",
            intrinsics_text="",
            meta_path="",
            panorama_path="",
            scene_name="",
            fps=DEFAULT_FPS,
            num_frames=0,
            call_kwargs_text=call_kwargs_text,
            load_kwargs_text=load_kwargs_text,
            model_ref=model_ref,
            backend=self.state.launch_config.backend or self.state.entry.default_backend or "auto",
            endpoint=self.state.launch_config.endpoint or self.state.entry.default_endpoint or "",
            api_key=_api_key(),
            device=self.state.launch_config.device or "cuda",
        )

    def _image_from_payload(self, payload: dict[str, Any]) -> Image.Image | None:
        raw_data = payload.get("init_image_base64")
        if isinstance(raw_data, str) and raw_data.strip():
            return Image.open(BytesIO(_decode_data_url(raw_data)[0])).convert("RGB")

        raw_path = str(payload.get("init_image_path") or "").strip()
        if raw_path:
            path = Path(raw_path).expanduser().resolve()
            if not path.exists() or path.suffix.lower() not in IMAGE_EXTS or not path_allowed(path, self.state.allowed_roots):
                raise ValueError("Selected image is not available to the world frontend.")
            with Image.open(path) as image:
                return image.convert("RGB")

        asset_path = Path(self.state.launch_config.asset_path).expanduser() if self.state.launch_config.asset_path else None
        if asset_path and asset_path.exists() and asset_path.suffix.lower() in IMAGE_EXTS:
            with Image.open(asset_path) as image:
                return image.convert("RGB")
        return None

    def _video_from_payload(self, payload: dict[str, Any]) -> str:
        raw_data = payload.get("init_video_base64")
        if isinstance(raw_data, str) and raw_data.strip():
            data, mime = _decode_data_url(raw_data)
            file_name = str(payload.get("video_file_name") or "input.webm")
            suffix = Path(file_name).suffix.lower() or _suffix_for_mime(mime) or ".webm"
            if suffix not in VIDEO_EXTS:
                suffix = ".webm"
            upload_root = Path(self.state.manager.workspace_root) / WORLD_UPLOAD_DIR_NAME
            upload_root.mkdir(parents=True, exist_ok=True)
            path = upload_root / f"{uuid.uuid4().hex}{suffix}"
            path.write_bytes(data)
            return str(path)

        raw_path = str(payload.get("init_video_path") or "").strip()
        if raw_path:
            path = Path(raw_path).expanduser().resolve()
            if not path.exists() or path.suffix.lower() not in VIDEO_EXTS or not path_allowed(path, self.state.allowed_roots):
                raise ValueError("Selected video is not available to the world frontend.")
            return str(path)
        return ""

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or "0")
        raw = self.rfile.read(length) if length else b"{}"
        if not raw:
            return {}
        payload = json.loads(raw.decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Expected a JSON object.")
        return payload

    def _send_html(self, html: str) -> None:
        self._send_text(html, "text/html; charset=utf-8")

    def _send_head_text(
        self,
        text: str,
        content_type: str,
        *,
        status: HTTPStatus = HTTPStatus.OK,
        cache_control: str = "no-store",
    ) -> None:
        self._send_head_bytes(
            len(text.encode("utf-8")),
            content_type,
            status=status,
            cache_control=cache_control,
        )

    def _send_head_json(
        self,
        payload: dict[str, Any],
        *,
        status: HTTPStatus = HTTPStatus.OK,
        cache_control: str = "no-store",
    ) -> None:
        data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self._send_head_bytes(
            len(data),
            "application/json; charset=utf-8",
            status=status,
            cache_control=cache_control,
        )

    def _send_head_file(self, path: Path, content_type: str) -> None:
        stat = path.stat()
        file_size = stat.st_size
        start, end = parse_byte_range(self.headers.get("Range"), file_size)
        status = HTTPStatus.PARTIAL_CONTENT if start is not None else HTTPStatus.OK
        if start is None:
            length = max(file_size, 0)
        else:
            assert end is not None
            length = end - start + 1

        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Last-Modified", self.date_time_string(stat.st_mtime))
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
        self.end_headers()

    def _send_head_bytes(
        self,
        content_length: int,
        content_type: str,
        *,
        status: HTTPStatus = HTTPStatus.OK,
        cache_control: str = "no-store",
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(max(int(content_length), 0)))
        self.send_header("Cache-Control", cache_control)
        self.end_headers()

    def _send_text(self, text: str, content_type: str, *, cache_control: str = "no-store") -> None:
        send_text_response(self, text, content_type, cache_control=cache_control)

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        send_json_response(self, payload, status=status)

    def _send_file(self, path: Path, content_type: str) -> None:
        send_file_response(self, path, content_type, range_header=self.headers.get("Range"))


def _model_import_error_message(entry: CatalogEntry, exc: ImportError) -> str:
    missing_name = getattr(exc, "name", "") if isinstance(exc, ModuleNotFoundError) else ""
    details: list[str] = []
    if missing_name:
        details.append(
            f"{entry.display_name} runtime import failed because Python module '{missing_name}' "
            "is missing in the active process."
        )
    else:
        details.append(f"{entry.display_name} runtime import failed in the active process.")
    details.append("Activate or install the model runtime environment before pressing LOAD or START.")
    details.append(f"Pipeline import: {entry.module_path}:{entry.class_name}.")
    details.append(f"Original error: {type(exc).__name__}: {exc}")
    return " ".join(details)


def controls_to_interactions(action: Any) -> str:
    """Translate held browser controls into Studio interaction tokens."""

    if not isinstance(action, dict):
        return ""
    w = bool(action.get("w"))
    a = bool(action.get("a"))
    s = bool(action.get("s"))
    d = bool(action.get("d"))
    dx = float(action.get("camera_dx") or 0)
    dy = float(action.get("camera_dy") or 0)

    tokens: list[str] = []
    if w and a and not s and not d:
        tokens.append("forward_left")
    elif w and d and not s and not a:
        tokens.append("forward_right")
    elif s and a and not w and not d:
        tokens.append("backward_left")
    elif s and d and not w and not a:
        tokens.append("backward_right")
    elif w and not s:
        tokens.append("forward")
    elif s and not w:
        tokens.append("backward")
    elif a and not d:
        tokens.append("left")
    elif d and not a:
        tokens.append("right")

    camera_x = ""
    camera_y = ""
    if dx > WORLD_ACTION_DEADZONE:
        camera_x = "r"
    elif dx < -WORLD_ACTION_DEADZONE:
        camera_x = "l"
    if dy > WORLD_ACTION_DEADZONE:
        camera_y = "d"
    elif dy < -WORLD_ACTION_DEADZONE:
        camera_y = "u"
    if camera_x and camera_y:
        tokens.append(f"camera_{camera_y}{camera_x}")
    elif camera_y == "u":
        tokens.append("camera_up")
    elif camera_y == "d":
        tokens.append("camera_down")
    elif camera_x == "l":
        tokens.append("camera_l")
    elif camera_x == "r":
        tokens.append("camera_r")

    if not tokens and action.get("l_click"):
        tokens.append("forward")
    if not tokens and action.get("r_click"):
        tokens.append("backward")
    return ", ".join(tokens)


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


def _override_call_kwargs_seed(call_kwargs_text: str, seed: int) -> str:
    """Return call kwargs JSON with ``seed`` set to the given per-step value."""

    try:
        kwargs = json.loads(call_kwargs_text) if call_kwargs_text else {}
        if not isinstance(kwargs, dict):
            kwargs = {}
    except Exception:
        kwargs = {}
    kwargs["seed"] = int(seed)
    return json.dumps(kwargs, ensure_ascii=False)


def _uses_state_init(entry: CatalogEntry) -> bool:
    spec = interface_spec_for_entry(entry)
    return bool(
        entry.supports_stream
        and entry.runtime_kind == "default"
        and spec.template_id == "interactive-world"
        and "state-init" in entry.tags
    )


def _initial_start_interactions(entry: CatalogEntry) -> str:
    for token in entry.default_interactions:
        text = str(token or "").strip()
        if text:
            return text
    return ""


def _record_payload(record: RunRecord | None) -> dict[str, Any]:
    if record is None:
        return {}
    payload: dict[str, Any] = {
        "run_id": record.run_id,
        "status": record.status,
        "mode": record.mode,
        "manifest_url": _file_url(record.manifest_path),
        "image_url": _file_url(record.preview_image),
        "video_url": _file_url(record.preview_video),
        "gallery": [_file_url(path) for path in record.gallery],
        "extra": record.metadata.get("studio_performance", {}),
    }
    return payload


def _embed_frame_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Attach a compact image frame for WebSocket viewer updates when available."""

    if payload.get("video_url") or payload.get("frame_base64"):
        return payload
    image_url = str(payload.get("image_url") or "")
    image_path = _path_from_file_url(image_url)
    if image_path is None or not image_path.exists() or image_path.suffix.lower() not in IMAGE_EXTS:
        return payload
    if image_path.stat().st_size > 8 * 1024 * 1024:
        return payload
    enriched = dict(payload)
    enriched["frame_base64"] = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return enriched


def _path_from_file_url(url: str) -> Path | None:
    if not url:
        return None
    parsed = urlparse(url)
    if parsed.path != "/api/file":
        return None
    raw_path = (parse_qs(parsed.query).get("path") or [""])[0]
    if not raw_path:
        return None
    return Path(unquote(raw_path)).expanduser().resolve()


@lru_cache(maxsize=8)
def _demo_image_files(root: Path | None = None) -> tuple[Path, ...]:
    source_root = (root or WORLD_REFERENCE_IMAGE_ROOT).expanduser()
    if not source_root.exists():
        return ()
    return tuple(
        sorted(
            path.resolve()
            for path in source_root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS
        )
    )


def _repeat_to_slots(paths: tuple[Path, ...], slots: int) -> tuple[Path, ...]:
    if not paths:
        return ()
    return tuple(paths[index % len(paths)] for index in range(slots))


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


def _file_url(path: str | Path | None) -> str:
    if not path:
        return ""
    resolved = Path(path).expanduser().resolve()
    return f"/api/file?path={quote(str(resolved), safe='')}"


def _decode_data_url(value: str) -> tuple[bytes, str]:
    header = ""
    data = value
    if "," in value and value.startswith("data:"):
        header, data = value.split(",", 1)
    mime = "application/octet-stream"
    if header.startswith("data:"):
        mime = header[5:].split(";", 1)[0] or mime
    return base64.b64decode(data), mime


def _data_url_from_file(path: Path) -> str:
    stat = path.stat()
    return _cached_data_url_from_file(str(path), stat.st_mtime_ns, stat.st_size)


@lru_cache(maxsize=64)
def _cached_data_url_from_file(path_text: str, _mtime_ns: int, _size: int) -> str:
    path = Path(path_text)
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def _suffix_for_mime(mime: str) -> str:
    mapping = {
        "video/mp4": ".mp4",
        "video/webm": ".webm",
        "video/quicktime": ".mov",
        "video/x-msvideo": ".avi",
    }
    return mapping.get(mime, "")


def _open_image(path: str | None) -> Image.Image | None:
    if not path:
        return None
    image_path = Path(path)
    if not image_path.exists() or image_path.suffix.lower() not in IMAGE_EXTS:
        return None
    try:
        with Image.open(image_path) as image:
            return image.convert("RGB")
    except Exception:
        return None


def _api_key() -> str:
    return env_first("WORLDFOUNDRY_STUDIO_API_KEY").strip()


def _html_escape(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
