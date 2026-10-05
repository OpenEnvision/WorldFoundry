#!/usr/bin/env python3
"""Serve the 3D reward backend over a constrained JSON HTTP API."""

from __future__ import annotations

from _bootstrap import setup_paths

setup_paths()

import argparse
import hmac
import ipaddress
import json
import os
import signal
import socket
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Mapping, Sequence
from urllib.parse import urlsplit

from model.vlm import resolve_model_name, resolve_vlm_backend

DEFAULT_MODEL_NAME = "worldeval/weights/da3"
DEFAULT_SCORING_MODEL: str | None = None
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8089
DEFAULT_MAX_BODY_BYTES = 1024 * 1024

reward_3d_manager = None


class RequestError(ValueError):
    """An invalid HTTP request that is safe to describe to the client."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


def _is_loopback_host(host: str) -> bool:
    """Return true only when every address for *host* is loopback."""

    normalized = host.strip().removeprefix("[").removesuffix("]")
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        pass

    try:
        infos = socket.getaddrinfo(normalized, None, type=socket.SOCK_STREAM)
    except socket.gaierror:
        return False

    addresses: set[ipaddress.IPv4Address | ipaddress.IPv6Address] = set()
    for _family, _type, _proto, _canon, sockaddr in infos:
        try:
            addresses.add(ipaddress.ip_address(sockaddr[0]))
        except (ValueError, IndexError):
            return False
    # A mixed public/loopback DNS answer must never be classified as local.
    return bool(addresses) and all(address.is_loopback for address in addresses)


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _resolve_roots(values: Sequence[str | os.PathLike[str]]) -> tuple[Path, ...]:
    roots: list[Path] = []
    for value in values:
        root = Path(value).expanduser().resolve(strict=True)
        if not root.is_dir():
            raise ValueError(f"Allowed path root is not a directory: {root}")
        if root not in roots:
            roots.append(root)
    if not roots:
        raise ValueError("At least one allowed input root is required.")
    return tuple(roots)


def _resolve_input_file(value: object, roots: Sequence[Path], *, field: str) -> Path:
    if not isinstance(value, (str, os.PathLike)) or not str(value).strip():
        raise RequestError(400, f"Request JSON must contain a non-empty {field!r} path.")
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = roots[0] / candidate
    try:
        resolved = candidate.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise RequestError(400, f"{field} does not name a readable file.") from exc
    if not resolved.is_file():
        raise RequestError(400, f"{field} does not name a readable file.")
    if not any(_path_is_within(resolved, root) for root in roots):
        raise RequestError(403, f"{field} is outside the configured input roots.")
    return resolved


def _resolve_output_file(value: object, *, default: Path, output_root: Path) -> Path:
    if value is None or value == "":
        candidate = default
    elif isinstance(value, (str, os.PathLike)):
        candidate = Path(value).expanduser()
    else:
        raise RequestError(400, "output_path must be a string path.")
    if not candidate.is_absolute():
        candidate = output_root / candidate
    resolved = candidate.resolve(strict=False)
    if not _path_is_within(resolved, output_root):
        raise RequestError(403, "output_path is outside the configured output root.")
    if resolved.suffix.lower() != ".json":
        raise RequestError(400, "output_path must end in .json.")
    return resolved


def _is_authorized(headers: Mapping[str, str], expected_token: str | None) -> bool:
    if expected_token is None:
        return True
    scheme, separator, supplied = headers.get("Authorization", "").partition(" ")
    return (
        bool(separator)
        and scheme.lower() == "bearer"
        and bool(supplied)
        and hmac.compare_digest(supplied, expected_token)
    )


def resolve_model_arg(value: str) -> str:
    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path)
    repo_root = Path(__file__).resolve().parents[1]
    project_candidate = repo_root.parent / path
    if project_candidate.exists():
        return str(project_candidate.resolve())
    repo_candidate = repo_root / path
    if repo_candidate.exists():
        return str(repo_candidate.resolve())
    return value


def shutdown_server(signum=None, frame=None):
    global reward_3d_manager
    if reward_3d_manager is not None:
        print("Shutting down 3D reward backend...")
        reward_3d_manager.shutdown()
        reward_3d_manager = None


class Reward3DHandler(BaseHTTPRequestHandler):
    def _write_json(
        self,
        status: int,
        payload: dict[str, object],
        *,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)

    def _require_authorization(self) -> bool:
        if _is_authorized(self.headers, getattr(self.server, "auth_token", None)):
            return True
        # Do not drain an unauthenticated request body. Closing the connection
        # prevents it from being parsed as another request on this connection.
        self.close_connection = True
        self._write_json(
            401,
            {"error": "A valid Bearer token is required."},
            headers={"WWW-Authenticate": "Bearer", "Connection": "close"},
        )
        return False

    def do_GET(self):
        if not self._require_authorization():
            return
        path = urlsplit(self.path).path
        if path == "/health":
            self._write_json(
                200,
                {
                    "status": "ok",
                    "model_name": getattr(self.server, "model_name", None),
                    "scorer": getattr(self.server, "scorer", None),
                    "scoring_model": getattr(self.server, "scoring_model", None),
                    "num_workers": getattr(self.server, "num_workers", None),
                    "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES", ""),
                },
            )
            return
        self._write_json(404, {"error": f"Unsupported path: {path}"})

    def do_POST(self):
        if not self._require_authorization():
            return
        path = urlsplit(self.path).path
        if path == "/score_file":
            self._handle_score_file()
            return
        if path == "/extract_trajectory":
            self._handle_extract_trajectory()
            return
        if path != "/":
            self._write_json(404, {"error": f"Unsupported path: {path}"})
            return

        # The former endpoint unpickled attacker-controlled bytes. Never read
        # that body: close the connection after explaining the JSON API.
        self.close_connection = True
        self._write_json(
            415,
            {
                "error": (
                    "Pickle-over-HTTP scoring is disabled. POST JSON to "
                    "/score_file or /extract_trajectory instead."
                )
            },
            headers={"Connection": "close"},
        )

    def _read_json_body(self) -> dict[str, object]:
        content_type = self.headers.get("Content-Type", "").partition(";")[0].strip().lower()
        if content_type != "application/json" and not content_type.endswith("+json"):
            raise RequestError(415, "Content-Type must be application/json.")
        if self.headers.get("Transfer-Encoding"):
            raise RequestError(400, "Transfer-Encoding is not supported.")

        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            raise RequestError(411, "Content-Length is required.")
        try:
            content_length = int(raw_length)
        except ValueError as exc:
            raise RequestError(400, "Content-Length must be an integer.") from exc
        if content_length < 0:
            raise RequestError(400, "Content-Length must not be negative.")
        if content_length > getattr(self.server, "max_body_bytes", DEFAULT_MAX_BODY_BYTES):
            self.close_connection = True
            raise RequestError(413, "Request body is too large.")

        payload = self.rfile.read(content_length)
        if len(payload) != content_length:
            raise RequestError(400, "Request body ended before Content-Length bytes were read.")
        try:
            data = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RequestError(400, "Request body must be a valid UTF-8 JSON object.") from exc
        if not isinstance(data, dict):
            raise RequestError(400, "Request body must be a JSON object.")
        return data

    def _write_request_error(self, exc: Exception) -> None:
        if isinstance(exc, RequestError):
            self._write_json(exc.status, {"error": str(exc)})
            return
        print(f"[serve_reward_3d] request failed: {exc!r}", file=sys.stderr)
        self._write_json(500, {"error": "Internal 3D reward server error."})

    def _handle_score_file(self) -> None:
        try:
            data = self._read_json_body()
            if reward_3d_manager is None:
                raise RuntimeError("3D reward backend is not initialized")

            input_roots = getattr(self.server, "input_roots")
            resolved_video_path = _resolve_input_file(
                data.get("video_path") or data.get("video"), input_roots, field="video_path"
            )

            from reward_3d import load_dynamic_masks, load_video_frames_as_jpeg

            max_frames = data.get("max_frames")
            frames = load_video_frames_as_jpeg(str(resolved_video_path), max_frames=max_frames)

            dynamic_masks = None
            dynamic_mask_video_path = data.get("dynamic_mask_video_path")
            resolved_dynamic_mask_video_path = None
            if dynamic_mask_video_path:
                resolved_dynamic_mask_video_path = _resolve_input_file(
                    dynamic_mask_video_path, input_roots, field="dynamic_mask_video_path"
                )
                dynamic_masks = load_dynamic_masks(
                    str(resolved_dynamic_mask_video_path), max_frames=max_frames
                )
                if len(dynamic_masks) != len(frames):
                    raise RequestError(
                        400,
                        "Dynamic mask frame count does not match sampled video frames: "
                        f"masks={len(dynamic_masks)}, frames={len(frames)}",
                    )

            with self.server.inference_lock:
                reward_3d_manager.compute_batch_scores(
                    [frames],
                    [str(data.get("prompt") or "")],
                    camera_trajectories=[data.get("camera_trajectory")],
                    dynamic_masks=[dynamic_masks],
                    use_lpips=data.get("use_lpips"),
                )

            if not reward_3d_manager.last_results or not reward_3d_manager.last_results[
                "per_video_results"
            ]:
                raise RuntimeError("3D reward backend did not return any per-video result.")

            result = dict(reward_3d_manager.last_results["per_video_results"][0])
            result["video_path"] = str(resolved_video_path)
            result["reconstruction_model_name"] = getattr(self.server, "model_name", None)
            result["scorer_type"] = reward_3d_manager.scorer_type
            result["scoring_model_name"] = reward_3d_manager.scoring_model_name
            result["num_workers"] = reward_3d_manager.num_workers
            result["max_frames"] = len(frames)
            result["dynamic_mask_video_path"] = (
                None
                if resolved_dynamic_mask_video_path is None
                else str(resolved_dynamic_mask_video_path)
            )
            result["batch_dir"] = reward_3d_manager.last_results["batch_dir"]
            result["reward_3d_server"] = {
                "url": f"http://{self.server.server_address[0]}:{self.server.server_address[1]}",
                "model_name": getattr(self.server, "model_name", None),
            }
            self._write_json(200, {"result": result})
        except Exception as exc:
            self._write_request_error(exc)

    def _handle_extract_trajectory(self) -> None:
        try:
            data = self._read_json_body()
            if reward_3d_manager is None:
                raise RuntimeError("3D reward backend is not initialized")

            resolved_video_path = _resolve_input_file(
                data.get("video_path") or data.get("video"),
                getattr(self.server, "input_roots"),
                field="video_path",
            )
            output_path = _resolve_output_file(
                data.get("output_path"),
                default=(
                    resolved_video_path.parent
                    / f"{resolved_video_path.stem}_da3_camera_trajectory.json"
                ),
                output_root=getattr(self.server, "output_root"),
            )

            from reward_3d import load_video_frames_as_jpeg_with_indices

            max_frames = data.get("max_frames")
            process_res = int(data.get("process_res") or 504)
            frames, frame_indices = load_video_frames_as_jpeg_with_indices(
                resolved_video_path, max_frames=max_frames
            )
            with self.server.inference_lock:
                trajectory = reward_3d_manager.extract_camera_trajectory(
                    frames, frame_indices, process_res=process_res
                )

            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(trajectory, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            self._write_json(
                200,
                {
                    "trajectory": trajectory,
                    "path": str(output_path),
                    "frame_indices": frame_indices,
                    "reward_3d_server": {
                        "url": (
                            f"http://{self.server.server_address[0]}:"
                            f"{self.server.server_address[1]}"
                        ),
                        "model_name": getattr(self.server, "model_name", None),
                    },
                },
            )
        except Exception as exc:
            self._write_request_error(exc)

    def log_message(self, format, *args):
        print(f"[serve_reward_3d] {self.address_string()} - {format % args}")


def _env_path_list(name: str) -> list[str]:
    return [part for part in os.getenv(name, "").split(os.pathsep) if part]


def main():
    parser = argparse.ArgumentParser(description="Local 3D reward server")
    parser.add_argument("--host", default=os.getenv("REWARD_3D_HOST", DEFAULT_HOST))
    parser.add_argument("--port", type=int, default=int(os.getenv("REWARD_3D_PORT", DEFAULT_PORT)))
    parser.add_argument(
        "--input-root",
        action="append",
        help=(
            "Directory from which request paths may be read; repeat for multiple roots. "
            "Defaults to REWARD_3D_INPUT_ROOTS or the working directory."
        ),
    )
    parser.add_argument(
        "--output-root",
        default=os.getenv("REWARD_3D_OUTPUT_ROOT") or str(Path.cwd()),
        help="Directory under which trajectory JSON may be written.",
    )
    parser.add_argument(
        "--max-body-bytes",
        type=int,
        default=int(os.getenv("REWARD_3D_MAX_BODY_BYTES", DEFAULT_MAX_BODY_BYTES)),
        help="Maximum JSON request size.",
    )
    parser.add_argument(
        "--vlm-backend",
        "--scorer",
        dest="scorer",
        default=os.getenv("REWARD_3D_SCORER") or os.getenv("VLM_BACKEND"),
        help="VLM backend used for GS/meta scoring: api/openrouter or local/qwenvl.",
    )
    parser.add_argument(
        "--model-name",
        default=os.getenv("REWARD_3D_MODEL_NAME", DEFAULT_MODEL_NAME),
        help="DA3 reconstruction model name or local path",
    )
    parser.add_argument(
        "--vlm-model",
        "--scoring-model",
        dest="scoring_model",
        default=os.getenv("REWARD_3D_SCORING_MODEL") or DEFAULT_SCORING_MODEL,
        help="VLM model name or local path used to score GS/meta renderings",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=int(os.getenv("REWARD_3D_NUM_WORKERS", "1")),
        help="Number of local DA3 worker processes / GPUs to use",
    )
    parser.add_argument(
        "--gpus",
        default=os.getenv("CUDA_VISIBLE_DEVICES"),
        help="Visible GPU ids, e.g. '0' or '0,1'.",
    )
    parser.add_argument("--lpips", dest="lpips", action="store_true")
    parser.add_argument("--no-lpips", dest="lpips", action="store_false")
    parser.set_defaults(
        lpips=os.getenv("REWARD_3D_USE_LPIPS", "1").strip().lower()
        not in {"0", "false", "no"}
    )
    args = parser.parse_args()
    args.scorer = resolve_vlm_backend(args.scorer)
    args.scoring_model = resolve_model_name(args.scoring_model, args.scorer)
    args.model_name = resolve_model_arg(args.model_name)

    if args.max_body_bytes <= 0:
        parser.error("--max-body-bytes must be positive")
    input_roots = _resolve_roots(
        args.input_root or _env_path_list("REWARD_3D_INPUT_ROOTS") or [str(Path.cwd())]
    )
    output_root = _resolve_roots([args.output_root])[0]

    is_loopback = _is_loopback_host(args.host)
    allow_non_loopback = os.getenv("REWARD_3D_ALLOW_NON_LOOPBACK", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    auth_token = os.getenv("REWARD_3D_AUTH_TOKEN", "").strip() or None
    if not is_loopback and not allow_non_loopback:
        raise SystemExit(
            f"Refusing to bind 3D reward server to non-loopback host {args.host!r}. "
            "Set REWARD_3D_ALLOW_NON_LOOPBACK=1 only on a trusted network."
        )
    if not is_loopback and (auth_token is None or len(auth_token) < 16):
        raise SystemExit(
            "A non-loopback 3D reward server requires REWARD_3D_AUTH_TOKEN "
            "with at least 16 characters."
        )

    if args.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    import multiprocessing as mp

    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    from reward_3d import MultiGPUReward3DManager

    global reward_3d_manager
    reward_3d_manager = MultiGPUReward3DManager(
        model_name=args.model_name,
        scorer_type=args.scorer,
        scoring_model_name=args.scoring_model,
        use_lpips=args.lpips,
        num_workers=args.num_workers,
    )
    reward_3d_manager.initialize()

    if not reward_3d_manager.processes:
        raise RuntimeError("3D reward backend failed to initialize. CUDA is required.")

    signal.signal(signal.SIGINT, shutdown_server)
    signal.signal(signal.SIGTERM, shutdown_server)

    server = ThreadingHTTPServer((args.host, args.port), Reward3DHandler)
    server.model_name = args.model_name
    server.scorer = args.scorer
    server.scoring_model = args.scoring_model
    server.num_workers = args.num_workers
    server.inference_lock = threading.Lock()
    server.input_roots = input_roots
    server.output_root = output_root
    server.max_body_bytes = args.max_body_bytes
    server.auth_token = auth_token
    print(f"Serving 3D reward backend on http://{args.host}:{args.port}")

    try:
        server.serve_forever()
    finally:
        server.server_close()
        shutdown_server()


if __name__ == "__main__":
    main()
