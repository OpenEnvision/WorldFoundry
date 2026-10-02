"""Standalone Studio frontend launchers.

Interactive world models, geometry viewers, simulator-heavy workflows, and 3DGS
assets can each launch a purpose-built browser surface.
"""

from __future__ import annotations

import contextlib
import mimetypes
import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
import threading
from functools import partial
from html import escape
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from worldfoundry.studio.inference.catalog import CatalogEntry
from worldfoundry.studio.inference.config import env_first
from worldfoundry.studio.serving import (
    StudioServiceTelemetry,
    StudioThreadingHTTPServer,
    bind_security_warning,
    path_allowed,
    public_url_for_bind,
    request_token_valid,
    require_auth_token_for_host,
    send_file_response,
    send_json_response,
    send_text_response,
)
from worldfoundry.studio.ui.launch_config import StudioLaunchConfig
from worldfoundry.studio.ui.vendor_assets import (
    SPARK_MODULE_PATH,
    THREE_CORE_MODULE_PATH,
    THREE_MODULE_PATH,
    VENDOR_ASSET_INSTALL_COMMAND,
    VENDOR_ROOT,
    VendorAssetError,
    require_vendor_assets,
)
from worldfoundry.studio.visualization.core.artifacts import (
    StudioVisualizationArtifact,
    infer_visualization_artifact,
)
from worldfoundry.studio.visualization.core.registry import (
    EMBODIED_VISUALIZATION,
    INTERACTIVE_WORLD_VISUALIZATION,
    MEDIA_VISUALIZATION,
    RERUN_VISUALIZATION,
    SPARK_VISUALIZATION,
    VISER_VISUALIZATION,
    BackendCapabilities,
    StudioVisualizationBackend,
    StudioVisualizationLaunch,
    StudioVisualizationRegistry,
    model_visualization_profile,
)

WORLD_FRONTEND = INTERACTIVE_WORLD_VISUALIZATION
POINTS_FRONTEND = VISER_VISUALIZATION
EMBODIED_FRONTEND = EMBODIED_VISUALIZATION
SPARK_FRONTEND = SPARK_VISUALIZATION
MEDIA_FRONTEND = MEDIA_VISUALIZATION
RERUN_FRONTEND = RERUN_VISUALIZATION


def _profile_mode(mode: str):
    return lambda entry, spec: model_visualization_profile(entry, spec).mode == mode


def _world_backend(request) -> StudioVisualizationLaunch | None:
    from worldfoundry.studio.visualization.backends.world import serve_world_frontend

    serve_world_frontend(
        request.entry,
        request.launch_config,
        host=host_for_frontend(request.launch_config),
        port=port_for_frontend(request.launch_config, WORLD_FRONTEND),
        access_printer=print_remote_access,
    )
    return None



def _points_backend(request) -> StudioVisualizationLaunch | None:
    serve_points_frontend(request.entry, request.launch_config)
    return None


def _embodied_backend(request) -> StudioVisualizationLaunch | None:
    serve_embodied_frontend(request.entry, request.launch_config)
    return None


def _spark_backend(request) -> StudioVisualizationLaunch | None:
    serve_spark_frontend(request.entry, request.launch_config)
    return None


def _media_backend(request) -> StudioVisualizationLaunch | None:
    serve_media_frontend(request.entry, request.launch_config)
    return None


def _rerun_backend(request) -> StudioVisualizationLaunch | None:
    serve_rerun_frontend(request.entry, request.launch_config)
    return None


STUDIO_VISUALIZATIONS = StudioVisualizationRegistry(
    (
        StudioVisualizationBackend(
            mode=WORLD_FRONTEND,
            title="Interactive World Model",
            default_port=7868,
            aliases=("interactive-world", "world-model"),
            match=_profile_mode(WORLD_FRONTEND),
            serve=_world_backend,
            capabilities=BackendCapabilities(frozenset({"image", "video"}), score=50),
        ),
        StudioVisualizationBackend(
            mode=POINTS_FRONTEND,
            title="Viser Geometry Viewer",
            default_port=18590,
            aliases=("viser", "geometry", "pointcloud", "point-cloud"),
            match=_profile_mode(POINTS_FRONTEND),
            serve=_points_backend,
            capabilities=BackendCapabilities(frozenset({"point_cloud", "mesh", "camera", "trajectory", "depth"}), score=80),
        ),
        StudioVisualizationBackend(
            mode=SPARK_FRONTEND,
            title="Spark Gaussian Splat Viewer",
            default_port=8765,
            aliases=("3dgs", "splat", "gaussian-splat"),
            match=_profile_mode(SPARK_FRONTEND),
            serve=_spark_backend,
            capabilities=BackendCapabilities(frozenset({"gaussian_splat"}), partial=False, score=90),
        ),
        StudioVisualizationBackend(
            mode=EMBODIED_FRONTEND,
            title="Embodied Simulator Bridge",
            default_port=18610,
            aliases=("sim", "simulator"),
            match=_profile_mode(EMBODIED_FRONTEND),
            serve=_embodied_backend,
            capabilities=BackendCapabilities(frozenset({"action_trace", "trajectory", "video"}), score=75),
        ),
        StudioVisualizationBackend(
            mode=RERUN_FRONTEND,
            title="Rerun Timeline Viewer",
            default_port=9876,
            aliases=("rrd",),
            match=_profile_mode(RERUN_FRONTEND),
            serve=_rerun_backend,
            capabilities=BackendCapabilities(frozenset({"timeline", "image", "video", "point_cloud", "mesh", "camera", "trajectory"}), score=70),
        ),
        StudioVisualizationBackend(
            mode=MEDIA_FRONTEND,
            title="Media Preview",
            default_port=18720,
            aliases=("preview", "video", "image"),
            match=_profile_mode(MEDIA_FRONTEND),
            serve=_media_backend,
            capabilities=BackendCapabilities(frozenset({"image", "video", "audio", "optical_flow"}), score=60),
        ),
    )
)

NATIVE_FRONTENDS = STUDIO_VISUALIZATIONS.native_modes
DEFAULT_FRONTEND_PORTS = STUDIO_VISUALIZATIONS.default_ports
SPARK_ASSET_EXTS = {".ply", ".spz", ".splat", ".ksplat", ".sog"}
MEDIA_ASSET_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".mp4", ".mov", ".webm", ".mkv", ".avi", ".wav", ".mp3", ".flac", ".ogg"}
STUDIO_ASSET_DIR = VENDOR_ROOT.parent
VENDOR_DIR = VENDOR_ROOT
SPARK_VENDOR_DIR = SPARK_MODULE_PATH.parent
THREE_VENDOR_DIR = THREE_MODULE_PATH.parent


def resolve_frontend_mode(
    entry: CatalogEntry,
    requested: str | None,
    artifact: StudioVisualizationArtifact | str | Path | None = None,
) -> str:
    """Resolve ``auto`` using an artifact first and model metadata as fallback."""

    if artifact is not None and not isinstance(artifact, StudioVisualizationArtifact):
        artifact = infer_visualization_artifact(artifact)
    return STUDIO_VISUALIZATIONS.resolve_mode(entry, requested, artifact)


def serve_native_frontend(entry: CatalogEntry, launch_config: StudioLaunchConfig, mode: str) -> None:
    """Launch a standalone Studio frontend and block until interrupted."""
    STUDIO_VISUALIZATIONS.serve(entry=entry, launch_config=launch_config, mode=mode)


def host_for_frontend(launch_config: StudioLaunchConfig) -> str:
    """Return the bind host for standalone frontends."""

    return (
        launch_config.host
        or env_first("WORLDFOUNDRY_STUDIO_HOST")
        or "127.0.0.1"
    ).strip() or "127.0.0.1"


def port_for_frontend(launch_config: StudioLaunchConfig, mode: str) -> int:
    """Return the port for a frontend, honoring mode-specific env overrides."""

    if launch_config.port is not None:
        return int(launch_config.port)
    env_port = (
        env_first(
            f"WORLDFOUNDRY_STUDIO_{mode.upper()}_PORT",
        )
        or ""
    ).strip()
    if env_port:
        return int(env_port)
    return DEFAULT_FRONTEND_PORTS.get(mode, 7868)


def print_remote_access(mode: str, host: str, port: int) -> None:
    """Emit concise SSH-tunnel instructions for local browser access."""

    ssh_host = (
        env_first("WORLDFOUNDRY_STUDIO_SSH_HOST")
        or "remote-host"
    )
    remote_host = "127.0.0.1" if host in {"", "0.0.0.0", "127.0.0.1", "localhost"} else host
    local_url = f"http://127.0.0.1:{port}"
    print(f"WorldFoundry Studio frontend: {mode}", flush=True)
    print(f"Remote bind: http://{host}:{port}", flush=True)
    if host in {"0.0.0.0", "::"}:
        print(f"Network URL: {public_url_for_bind(host, port)}", flush=True)
    warning = bind_security_warning(host)
    if warning:
        print(warning, flush=True)
    print(f"Local tunnel: ssh -N -L {port}:{remote_host}:{port} {ssh_host}", flush=True)
    print(f"Browser URL: {local_url}", flush=True)


def serve_points_frontend(entry: CatalogEntry, launch_config: StudioLaunchConfig) -> None:
    """Launch native Viser for a point cloud or mesh asset."""

    from worldfoundry.studio.visualization.backends.viser import STUDIO_VISER

    asset_path = _required_existing_asset(
        launch_config.asset_path,
        frontend="points",
        hint="pass --asset /path/to/scene.ply or --asset /path/to/points.npz",
    )
    host = host_for_frontend(launch_config)
    requested_port = _explicit_points_port(launch_config)
    presentation = STUDIO_VISER.present_geometry(
        run_id=f"native-{entry.model_id}",
        geometry_path=asset_path,
        model_id=entry.model_id,
        host=host,
        port=requested_port,
        embed=False,
    )
    if not presentation.url:
        raise SystemExit(_strip_html(presentation.caption or presentation.html))
    print_remote_access(POINTS_FRONTEND, host, _port_from_url(presentation.url) or requested_port or 18590)
    print(presentation.caption, flush=True)
    _wait_forever()


def serve_embodied_frontend(entry: CatalogEntry, launch_config: StudioLaunchConfig) -> None:
    """Keep embodied simulator UX native by advertising its own URL."""

    simulator_url = (
        launch_config.simulator_url
        or env_first("WORLDFOUNDRY_STUDIO_SIMULATOR_URL")
    ).strip()
    if not simulator_url:
        model = entry.display_name or entry.model_id
        raise SystemExit(
            "Embodied Studio preserves the simulator frontend instead of embedding it in the game-console UI. "
            f"Start the native simulator for {model}, then launch Studio with --simulator-url http://127.0.0.1:<port>."
        )
    print(f"WorldFoundry Studio frontend: {EMBODIED_FRONTEND}", flush=True)
    print(f"Native simulator URL: {simulator_url}", flush=True)
    _print_tunnel_for_url(simulator_url)
    _wait_forever()


def serve_spark_frontend(entry: CatalogEntry, launch_config: StudioLaunchConfig) -> None:
    """Serve a standalone Spark 3DGS viewer over HTTP."""

    try:
        require_vendor_assets()
    except VendorAssetError as exc:
        raise SystemExit(str(exc)) from None

    host = host_for_frontend(launch_config)
    port = port_for_frontend(launch_config, SPARK_FRONTEND)
    auth_token = require_auth_token_for_host(host, server_name="Spark frontend")
    asset_path = _optional_asset(launch_config.asset_path)
    if asset_path is not None and asset_path.suffix.lower() not in SPARK_ASSET_EXTS:
        raise SystemExit(
            f"Spark frontend expects one of {', '.join(sorted(SPARK_ASSET_EXTS))}; got {asset_path.suffix or 'no suffix'}."
        )

    allowed_roots = _spark_allowed_roots(asset_path)
    handler = partial(
        SparkViewerHandler,
        default_asset=str(asset_path) if asset_path is not None else "",
        title=f"{entry.display_name} · Spark 3DGS",
        allowed_roots=allowed_roots,
        telemetry=StudioServiceTelemetry(f"{SPARK_FRONTEND}:{entry.model_id}"),
        auth_token=auth_token,
    )
    httpd = StudioThreadingHTTPServer((host, port), handler)
    print_remote_access(SPARK_FRONTEND, host, port)
    if asset_path is None:
        print("Spark asset: none. Use the Asset Path box or relaunch with --asset /path/to/world.splat.", flush=True)
    else:
        print(f"Spark asset: {asset_path}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


def serve_media_frontend(entry: CatalogEntry, launch_config: StudioLaunchConfig) -> None:
    """Serve a generic image/video/audio artifact preview over HTTP."""

    host = host_for_frontend(launch_config)
    port = port_for_frontend(launch_config, MEDIA_FRONTEND)
    auth_token = require_auth_token_for_host(host, server_name="Media frontend")
    asset_path = _required_existing_asset(
        launch_config.asset_path,
        frontend="media",
        hint="pass --asset /path/to/preview.mp4, preview.png, or preview.wav",
    )
    if asset_path.suffix.lower() not in MEDIA_ASSET_EXTS:
        raise SystemExit(
            f"Media frontend expects one of {', '.join(sorted(MEDIA_ASSET_EXTS))}; got {asset_path.suffix or 'no suffix'}."
        )
    handler = partial(
        MediaViewerHandler,
        asset_path=asset_path,
        title=f"{entry.display_name} · Media Preview",
        allowed_roots=(asset_path.parent.resolve(),),
        telemetry=StudioServiceTelemetry(f"{MEDIA_FRONTEND}:{entry.model_id}"),
        auth_token=auth_token,
    )
    httpd = StudioThreadingHTTPServer((host, port), handler)
    print_remote_access(MEDIA_FRONTEND, host, port)
    print(f"Media asset: {asset_path}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


def serve_rerun_frontend(entry: CatalogEntry, launch_config: StudioLaunchConfig) -> None:
    """Launch or advertise a Rerun viewer for ``.rrd`` timelines."""

    host = host_for_frontend(launch_config)
    port = port_for_frontend(launch_config, RERUN_FRONTEND)
    rerun_url = (
        launch_config.simulator_url
        or env_first("WORLDFOUNDRY_STUDIO_RERUN_URL")
    ).strip()
    if rerun_url:
        print(f"WorldFoundry Studio frontend: {RERUN_FRONTEND}", flush=True)
        print(f"Rerun URL: {rerun_url}", flush=True)
        _print_tunnel_for_url(rerun_url)
        _wait_forever()
        return

    asset_path = _required_existing_asset(
        launch_config.asset_path,
        frontend="rerun",
        hint="pass --asset /path/to/scene.rrd or set WORLDFOUNDRY_STUDIO_RERUN_URL",
    )
    if asset_path.suffix.lower() != ".rrd":
        raise SystemExit(f"Rerun frontend expects a .rrd asset; got {asset_path.suffix or 'no suffix'}.")

    command_template = env_first("WORLDFOUNDRY_STUDIO_RERUN_COMMAND") or _default_rerun_command_template()
    grpc_port_text = (env_first("WORLDFOUNDRY_STUDIO_RERUN_GRPC_PORT") or "").strip()
    grpc_port = int(grpc_port_text) if grpc_port_text else port + 1
    ws_port_text = (env_first("WORLDFOUNDRY_STUDIO_RERUN_WS_PORT") or "").strip()
    ws_port = int(ws_port_text) if ws_port_text else grpc_port + 1
    command = command_template.format(
        asset=shlex.quote(str(asset_path)),
        host=shlex.quote(host),
        port=port,
        grpc_port=grpc_port,
        ws_port=ws_port,
        model=shlex.quote(entry.model_id),
    )
    print_remote_access(RERUN_FRONTEND, host, port)
    print(f"Rerun asset: {asset_path}", flush=True)
    print(f"Rerun command: {command}", flush=True)
    # Template placeholders are shlex.quote()d above, so splitting the rendered
    # command is equivalent to the previous shell=True invocation without a shell.
    process = subprocess.Popen(shlex.split(command))
    try:
        process.wait()
    except KeyboardInterrupt:
        process.terminate()
        with contextlib.suppress(Exception):
            process.wait(timeout=5)


def _default_rerun_command_template() -> str:
    """Return a Rerun CLI command that works inside virtualenv-launched Studio."""

    sibling = Path(sys.executable).with_name("rerun")
    executable = sibling if sibling.exists() else Path(shutil.which("rerun") or "rerun")
    return (
        f"{shlex.quote(str(executable))} --web-viewer --port {{grpc_port}} "
        "--ws-server-port {ws_port} --web-viewer-port {port} {asset}"
    )


def _handler_request_authorized(handler: BaseHTTPRequestHandler, parsed, auth_token: str) -> bool:
    """Enforce the shared Studio token on an http.server request when set."""

    if not auth_token:
        return True
    query_token = (parse_qs(parsed.query).get("token") or [""])[0]
    if request_token_valid(
        auth_token,
        authorization_header=handler.headers.get("Authorization"),
        query_token=query_token,
    ):
        return True
    handler.send_error(HTTPStatus.UNAUTHORIZED, "Missing or invalid Studio auth token.")
    return False


class MediaViewerHandler(BaseHTTPRequestHandler):
    """Tiny HTTP surface for standalone media previews."""

    server_version = "WorldFoundryMediaFrontend/1.0"

    def __init__(
        self,
        *args,
        asset_path: Path,
        title: str,
        allowed_roots: tuple[Path, ...],
        telemetry: StudioServiceTelemetry,
        auth_token: str = "",
        **kwargs,
    ) -> None:
        self.asset_path = asset_path
        self.title = title
        self.allowed_roots = allowed_roots
        self.telemetry = telemetry
        self.auth_token = auth_token
        super().__init__(*args, **kwargs)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API.
        parsed = urlparse(self.path)
        if not _handler_request_authorized(self, parsed, self.auth_token):
            return
        with self.telemetry.track(parsed.path):
            self._handle_get(parsed.path)

    def _handle_get(self, path: str) -> None:
        if path in {"", "/"}:
            self._send_html(media_viewer_html(title=self.title, asset_path=self.asset_path))
            return
        if path == "/healthz":
            self._send_json(
                self.telemetry.snapshot(
                    frontend=MEDIA_FRONTEND,
                    asset=str(self.asset_path),
                    asset_exists=self.asset_path.exists(),
                )
            )
            return
        if path == "/__worldfoundry__/asset":
            if not self.asset_path.exists() or not path_allowed(self.asset_path, self.allowed_roots):
                self.send_error(HTTPStatus.NOT_FOUND, "Asset not found or not allowed.")
                return
            self._send_file(self.asset_path, mimetypes.guess_type(self.asset_path.name)[0] or "application/octet-stream")
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found.")

    def log_message(self, fmt: str, *args) -> None:
        print(f"[media] {self.address_string()} - {fmt % args}", flush=True)

    def _send_html(self, html: str) -> None:
        send_text_response(self, html, "text/html; charset=utf-8")

    def _send_json(self, payload: dict[str, object]) -> None:
        send_json_response(self, payload)

    def _send_file(self, path: Path, content_type: str) -> None:
        send_file_response(self, path, content_type, range_header=self.headers.get("Range"))


class SparkViewerHandler(BaseHTTPRequestHandler):
    """Tiny HTTP surface for the standalone Spark frontend."""

    server_version = "WorldFoundrySparkFrontend/1.0"

    # Vendored JS modules carry no secrets and import-map fetches cannot
    # attach the token, so they stay reachable when token auth is active.
    AUTH_EXEMPT_PATHS = frozenset(
        {
            "/__worldfoundry__/spark.module.js",
            "/__worldfoundry__/three.module.js",
            "/__worldfoundry__/three.core.js",
        }
    )

    def __init__(
        self,
        *args,
        default_asset: str,
        title: str,
        allowed_roots: tuple[Path, ...],
        telemetry: StudioServiceTelemetry,
        auth_token: str = "",
        **kwargs,
    ) -> None:
        self.default_asset = default_asset
        self.title = title
        self.allowed_roots = allowed_roots
        self.telemetry = telemetry
        self.auth_token = auth_token
        super().__init__(*args, **kwargs)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API.
        parsed = urlparse(self.path)
        if parsed.path not in self.AUTH_EXEMPT_PATHS and not _handler_request_authorized(
            self, parsed, self.auth_token
        ):
            return
        with self.telemetry.track(parsed.path):
            self._handle_get(parsed)

    def _handle_get(self, parsed) -> None:
        if parsed.path in {"", "/"}:
            self._send_html(spark_viewer_html(title=self.title, default_asset=self.default_asset))
            return
        if parsed.path == "/healthz":
            self._send_json(
                self.telemetry.snapshot(
                    frontend=SPARK_FRONTEND,
                    default_asset=self.default_asset,
                    allowed_roots=[str(root) for root in self.allowed_roots],
                )
            )
            return
        if parsed.path == "/__worldfoundry__/spark.module.js":
            if SPARK_MODULE_PATH.exists():
                self._send_file(SPARK_MODULE_PATH, "text/javascript", cache_control="no-store")
                return
            self.send_error(
                HTTPStatus.NOT_FOUND,
                f"Vendored Spark module missing. Run {VENDOR_ASSET_INSTALL_COMMAND}.",
            )
            return
        if parsed.path == "/__worldfoundry__/three.module.js":
            if THREE_MODULE_PATH.exists():
                self._send_file(THREE_MODULE_PATH, "text/javascript", cache_control="no-store")
                return
            self.send_error(
                HTTPStatus.NOT_FOUND,
                f"Vendored Three module missing. Run {VENDOR_ASSET_INSTALL_COMMAND}.",
            )
            return
        if parsed.path == "/__worldfoundry__/three.core.js":
            if THREE_CORE_MODULE_PATH.exists():
                self._send_file(THREE_CORE_MODULE_PATH, "text/javascript", cache_control="no-store")
                return
            self.send_error(
                HTTPStatus.NOT_FOUND,
                f"Vendored Three core module missing. Run {VENDOR_ASSET_INSTALL_COMMAND}.",
            )
            return
        if parsed.path == "/__worldfoundry__/asset":
            query = parse_qs(parsed.query)
            raw_path = (query.get("path") or [""])[0]
            path = Path(unquote(raw_path)).expanduser().resolve()
            if not path.exists() or not path_allowed(path, self.allowed_roots):
                self.send_error(HTTPStatus.NOT_FOUND, "Asset not found or not allowed.")
                return
            self._send_file(path, mimetypes.guess_type(path.name)[0] or "application/octet-stream")
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found.")

    def log_message(self, fmt: str, *args) -> None:
        print(f"[spark] {self.address_string()} - {fmt % args}", flush=True)

    def _send_html(self, html: str) -> None:
        send_text_response(self, html, "text/html; charset=utf-8")

    def _send_json(self, payload: dict[str, object]) -> None:
        send_json_response(self, payload)

    def _send_file(self, path: Path, content_type: str, *, cache_control: str = "no-store") -> None:
        send_file_response(
            self,
            path,
            content_type,
            range_header=self.headers.get("Range"),
            cache_control=cache_control,
        )


def spark_viewer_html(*, title: str, default_asset: str) -> str:
    """Return the standalone Spark viewer HTML."""

    safe_title = escape(title)
    default_asset_js = _js_string(default_asset)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{safe_title}</title>
  <script type="importmap">
  {{
    "imports": {{
      "three": "/__worldfoundry__/three.module.js",
      "@sparkjsdev/spark": "/__worldfoundry__/spark.module.js"
    }}
  }}
  </script>
  <style>
    :root {{ color-scheme: dark; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    * {{ box-sizing: border-box; }}
    html, body {{ width: 100%; height: 100%; margin: 0; overflow: hidden; background: #050505; color: #f4f4f5; }}
    #stage {{ position: fixed; inset: 0; width: 100vw; height: 100vh; display: block; touch-action: none; }}
    .hud {{ position: fixed; left: 16px; right: 16px; top: 14px; display: flex; align-items: start; justify-content: space-between; gap: 12px; pointer-events: none; }}
    .panel {{ pointer-events: auto; max-width: min(680px, calc(100vw - 32px)); border: 1px solid rgba(255,255,255,.12); background: rgba(10,10,10,.78); backdrop-filter: blur(14px); border-radius: 8px; padding: 12px; box-shadow: 0 20px 80px rgba(0,0,0,.4); }}
    h1 {{ margin: 0 0 8px; font-size: 14px; line-height: 1.2; font-weight: 650; letter-spacing: 0; }}
    .status {{ margin: 0; color: #d4d4d8; font: 12px/1.45 ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; white-space: pre-wrap; }}
    .controls {{ display: flex; gap: 8px; margin-top: 10px; }}
    input {{ width: min(520px, calc(100vw - 128px)); min-width: 0; color: #fafafa; background: #111; border: 1px solid #3f3f46; border-radius: 6px; padding: 8px 10px; font: 12px/1.2 ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; }}
    button {{ color: #09090b; background: #f4f4f5; border: 0; border-radius: 6px; padding: 8px 12px; font-weight: 650; cursor: pointer; }}
    .keys {{ min-width: 220px; text-align: right; color: #d4d4d8; font: 11px/1.55 ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; }}
    @media (max-width: 720px) {{ .hud {{ flex-direction: column; }} .keys {{ text-align: left; }} input {{ width: 100%; }} .controls {{ flex-direction: column; }} }}
  </style>
</head>
<body>
  <canvas id="stage"></canvas>
  <div class="hud">
    <section class="panel">
      <h1>{safe_title}</h1>
      <p class="status" id="status">Initializing Spark viewer...</p>
      <div class="controls">
        <input id="asset" aria-label="Asset path" placeholder="$WORLDFOUNDRY_ARTIFACT_DIR/.../scene.splat">
        <button id="load" type="button">Load</button>
      </div>
    </section>
    <section class="panel keys">Drag look · Wheel dolly<br>WASD move · R reset<br>Remote path is resolved on the GPU node</section>
  </div>
  <script type="module">
    import * as THREE from "three";
    import * as Spark from "@sparkjsdev/spark";

    const DEFAULT_ASSET = {default_asset_js};
    const canvas = document.getElementById("stage");
    const status = document.getElementById("status");
    const input = document.getElementById("asset");
    const loadButton = document.getElementById("load");
    let viewer = null;

    const setStatus = (message) => {{ status.textContent = message; }};
    const pixelRatio = () => Math.min(Math.max(1, window.devicePixelRatio || 1), 1.5);
    const AUTH_TOKEN = new URLSearchParams(window.location.search).get("token") || "";
    const assetFromLocation = () => new URLSearchParams(window.location.search).get("asset") || DEFAULT_ASSET || "";
    const assetUrl = (path) => {{
      if (/^https?:\\/\\//i.test(path)) return path;
      const tokenParam = AUTH_TOKEN ? `&token=${{encodeURIComponent(AUTH_TOKEN)}}` : "";
      return `/__worldfoundry__/asset?path=${{encodeURIComponent(path)}}${{tokenParam}}`;
    }};
    const fileName = (path) => (path.split("/").pop() || "world.splat").split("?")[0];

    const disposeViewer = () => {{
      if (!viewer) return;
      viewer.renderer.setAnimationLoop(null);
      viewer.scene.remove(viewer.splat);
      viewer.splat?.dispose?.();
      viewer.spark?.dispose?.();
      viewer.renderer.dispose();
      viewer = null;
    }};

    const pauseViewer = () => {{
      if (!viewer || !viewer.running) return;
      viewer.running = false;
      viewer.renderer.setAnimationLoop(null);
    }};

    const resumeViewer = () => {{
      if (!viewer || viewer.running || document.visibilityState === "hidden") return;
      viewer.running = true;
      viewer.renderer.setAnimationLoop(() => {{
        if (document.visibilityState === "hidden") {{
          pauseViewer();
          return;
        }}
        resize();
        viewer.controls.update(viewer.camera);
        viewer.renderer.render(viewer.scene, viewer.camera);
      }});
    }};

    const resize = () => {{
      if (!viewer) return;
      const width = Math.max(1, Math.round(window.innerWidth));
      const height = Math.max(1, Math.round(window.innerHeight));
      const ratio = pixelRatio();
      if (viewer.pixelRatio !== ratio) {{
        viewer.renderer.setPixelRatio(ratio);
        viewer.pixelRatio = ratio;
      }}
      if (viewer.width !== width || viewer.height !== height) {{
        viewer.renderer.setSize(width, height, false);
        viewer.camera.aspect = width / height;
        viewer.camera.updateProjectionMatrix();
        viewer.width = width;
        viewer.height = height;
      }}
    }};

    const loadAsset = async (path) => {{
      path = (path || "").trim();
      input.value = path;
      if (!path) {{
        disposeViewer();
        setStatus("No asset selected. Enter a remote .ply, .spz, .splat, .ksplat, or .sog path.");
        return;
      }}
      disposeViewer();
      setStatus(`Loading ${{path}}`);
      const scene = new THREE.Scene();
      const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.01, 1000);
      camera.position.set(0, 0, 1);
      const renderer = new THREE.WebGLRenderer({{ canvas, antialias: true, powerPreference: "high-performance" }});
      const spark = new Spark.SparkRenderer({{ renderer }});
      scene.add(spark);
      const controls = new Spark.SparkControls({{ canvas }});
      const splat = new Spark.SplatMesh({{
        url: assetUrl(path),
        fileName: fileName(path),
        onProgress: (event) => {{
          const loaded = Number(event?.loaded || 0);
          const total = Number(event?.total || 0);
          setStatus(total > 0 ? `Loading ${{Math.round((loaded / total) * 100)}}%\\n${{path}}` : `Loading...\\n${{path}}`);
        }},
      }});
      splat.quaternion.set(1, 0, 0, 0);
      scene.add(splat);
      viewer = {{ scene, camera, renderer, spark, controls, splat, running: false, width: 0, height: 0, pixelRatio: 0 }};
      resize();
      await splat.initialized;
      setStatus(`Spark ready\n${{path}}`);
      resumeViewer();
    }};

    loadButton.addEventListener("click", () => {{
      const url = new URL(window.location.href);
      url.searchParams.set("asset", input.value.trim());
      window.history.replaceState(null, "", url.toString());
      void loadAsset(input.value);
    }});
    input.addEventListener("keydown", (event) => {{
      if (event.key === "Enter") loadButton.click();
    }});
    window.addEventListener("resize", resize);
    document.addEventListener("visibilitychange", () => {{
      if (document.visibilityState === "hidden") pauseViewer();
      else resumeViewer();
    }});
    window.addEventListener("keydown", (event) => {{
      if (event.key.toLowerCase() === "r" && viewer) {{
        viewer.camera.position.set(0, 0, 1);
        viewer.camera.lookAt(0, 0, 0);
        setStatus(`Camera reset\\n${{input.value}}`);
      }}
    }});
    void loadAsset(assetFromLocation()).catch((error) => {{
      console.error(error);
      setStatus(error instanceof Error ? error.message : String(error));
    }});
  </script>
</body>
</html>"""


def media_viewer_html(*, title: str, asset_path: Path) -> str:
    """Return the standalone media viewer HTML."""

    safe_title = escape(title)
    safe_name = escape(asset_path.name)
    mime = mimetypes.guess_type(asset_path.name)[0] or "application/octet-stream"
    if mime.startswith("image/"):
        viewer = '<img class="asset asset-image" src="/__worldfoundry__/asset" alt="">'
    elif mime.startswith("audio/"):
        viewer = '<audio class="asset asset-audio" src="/__worldfoundry__/asset" controls autoplay></audio>'
    else:
        viewer = '<video class="asset asset-video" src="/__worldfoundry__/asset" controls autoplay muted playsinline loop></video>'
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{safe_title}</title>
  <style>
    :root {{ color-scheme: dark; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    * {{ box-sizing: border-box; }}
    html, body {{ margin: 0; width: 100%; height: 100%; overflow: hidden; background: #070707; color: #f4f4f5; }}
    main {{ width: 100vw; height: 100vh; display: grid; place-items: center; padding: 52px 16px 16px; }}
    .asset {{ max-width: 100%; max-height: 100%; display: block; background: #000; border: 1px solid rgba(255,255,255,.12); border-radius: 6px; box-shadow: 0 20px 80px rgba(0,0,0,.45); }}
    .asset-video, .asset-audio {{ width: min(1280px, calc(100vw - 32px)); }}
    .asset-audio {{ padding: 18px; background: #111; }}
    .hud {{ position: fixed; inset: 12px 16px auto 16px; display: flex; justify-content: space-between; gap: 12px; align-items: center; pointer-events: none; }}
    .pill {{ pointer-events: auto; border: 1px solid rgba(255,255,255,.12); background: rgba(12,12,12,.8); border-radius: 6px; padding: 8px 10px; font: 12px/1.25 ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; color: #e4e4e7; }}
  </style>
</head>
<body>
  <div class="hud">
    <div class="pill">{safe_title}</div>
    <div class="pill">{safe_name}</div>
  </div>
  <main>{viewer}</main>
  <script>
    (() => {{
      const token = new URLSearchParams(window.location.search).get("token");
      if (!token) return;
      const asset = document.querySelector(".asset");
      if (!asset) return;
      const url = new URL(asset.getAttribute("src"), window.location.origin);
      url.searchParams.set("token", token);
      asset.src = url.toString();
    }})();
  </script>
</body>
</html>"""


def _spark_allowed_roots(asset_path: Path | None) -> tuple[Path, ...]:
    # Only explicit workspace/artifact/model directories and the selected
    # asset's parent are servable. The process cwd (often the repo root) is
    # intentionally not an allowed root.
    roots: list[Path] = []
    for raw in (
        os.getenv("WORLDFOUNDRY_STUDIO_WORKSPACE_DIR"),
        os.getenv("WORLDFOUNDRY_ARTIFACT_DIR"),
        os.getenv("WORLDFOUNDRY_MODEL_DIR"),
    ):
        if raw:
            with contextlib.suppress(OSError):
                path = Path(raw).expanduser().resolve()
                if path.exists():
                    roots.append(path)
    if asset_path is not None:
        roots.append(asset_path.parent.resolve())
    return tuple(dict.fromkeys(roots))


def _explicit_points_port(launch_config: StudioLaunchConfig) -> int | None:
    """Return an explicitly configured Points port, leaving auto mode to the Viser pool."""

    if launch_config.port is not None:
        return int(launch_config.port)
    raw_port = (
        env_first("WORLDFOUNDRY_STUDIO_POINTS_PORT", "WORLDFOUNDRY_STUDIO_VISER_PORT")
        or ""
    ).strip()
    return int(raw_port) if raw_port else None


def _port_from_url(url: str) -> int | None:
    """Extract a port from a localhost frontend URL."""

    parsed = urlparse(url)
    try:
        return int(parsed.port) if parsed.port is not None else None
    except ValueError:
        return None


def _pick_sidecar_port(host: str, preferred: int, *, avoid: set[int] | None = None) -> int:
    """Return a free auxiliary TCP port for viewers with a sidecar service."""

    avoid = avoid or set()
    for offset in range(64):
        port = preferred + offset
        if port in avoid:
            continue
        if _tcp_port_available(host, port):
            return port
    bind_host = "127.0.0.1" if host in {"", "0.0.0.0", "::", "localhost"} else host
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((bind_host, 0))
        return int(sock.getsockname()[1])


def _tcp_port_available(host: str, port: int) -> bool:
    bind_host = "127.0.0.1" if host in {"", "0.0.0.0", "::", "localhost"} else host
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((bind_host, port))
    except OSError:
        return False
    return True


def _required_existing_asset(raw: str, *, frontend: str, hint: str) -> Path:
    path = _optional_asset(raw)
    if path is None:
        raise SystemExit(f"{frontend} frontend requires an asset; {hint}.")
    return path


def _optional_asset(raw: str) -> Path | None:
    value = (raw or "").strip()
    if not value:
        return None
    path = Path(value).expanduser().resolve()
    if not path.exists():
        raise SystemExit(f"Frontend asset does not exist: {path}")
    return path


def _print_tunnel_for_url(url: str) -> None:
    parsed = urlparse(url)
    if not parsed.hostname or not parsed.port:
        return
    ssh_host = (
        env_first("WORLDFOUNDRY_STUDIO_SSH_HOST")
        or "remote-host"
    )
    remote_host = "127.0.0.1" if parsed.hostname in {"localhost", "127.0.0.1", "0.0.0.0"} else parsed.hostname
    print(f"Local tunnel: ssh -N -L {parsed.port}:{remote_host}:{parsed.port} {ssh_host}", flush=True)
    print(f"Browser URL: http://127.0.0.1:{parsed.port}", flush=True)


def _wait_forever() -> None:
    stop = threading.Event()
    try:
        stop.wait()
    except KeyboardInterrupt:
        pass


def _strip_html(value: str) -> str:
    return re.sub(r"<[^>]+>", "", value or "").strip()


def _js_string(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'
