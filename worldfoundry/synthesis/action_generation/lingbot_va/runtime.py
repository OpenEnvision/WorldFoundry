"""Utilities for configuring, interacting with, and tracing LingBot-VA models.

This module provides functions for managing LingBot-VA runtime configurations,
selecting appropriate model checkpoints, building server commands, and handling
data normalization for inference, particularly for in-tree official server/client
interactions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from worldfoundry.core.io import file_sha256
from worldfoundry.core.io.paths import project_root, resolve_worldfoundry_path, worldfoundry_path_tokens
from worldfoundry.synthesis.action_generation.wan_va import RUNTIME_ROOT, WAN_VA_PACKAGE, WAN_VA_ROOT


@dataclass(frozen=True)
class LingBotVARuntimeConfig:
    """Runtime settings for in-tree LingBot-VA server/client execution.

    This dataclass encapsulates all necessary configuration parameters to run
    or connect to a LingBot-VA instance, including checkpoint location, network
    details, and distributed training parameters.

    Attributes:
        checkpoint_dir: Local Hugging Face checkpoint directory.
        config_name: Official LingBot-VA config key (e.g., "franka-base").
        host: WebSocket server host address.
        port: WebSocket server port.
        nproc_per_node: Number of processes per node for distributed training (used by torchrun).
        master_port: Port used for distributed training communication (used by torchrun).
    """

    checkpoint_dir: Path
    config_name: str
    host: str
    port: int
    nproc_per_node: int
    master_port: int


def _jsonable(value: Any) -> Any:
    """Recursively converts various Python types into JSON-serializable types.

    Handles mappings, sequences, Path objects, and NumPy arrays (via .tolist() or .item()).
    Non-standard types are converted to strings.

    Args:
        value: The object to convert.

    Returns:
        A JSON-serializable representation of the input value.
    """
    if isinstance(value, Mapping):
        # Convert dictionary keys to strings and recursively process values.
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        # Recursively process items in lists and tuples.
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        # Convert Path objects to their string representation.
        return str(value)
    if hasattr(value, "tolist"):
        # Handle NumPy arrays or similar objects by converting them to a list.
        return _jsonable(value.tolist())
    if hasattr(value, "item"):
        # Handle NumPy scalars or similar objects by extracting their scalar value.
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        # Basic JSON-serializable types and None are returned as is.
        return value
    # Fallback: convert any other type to its string representation.
    return str(value)


def _worldfoundry_repository_root() -> Path:
    """Returns the root path of the WorldFoundry repository."""
    return project_root()


def _expand_checkpoint_path(value: str | Path) -> Path:
    """Expands and resolves a checkpoint path relative to the WorldFoundry repository root.

    Args:
        value: The checkpoint path, which can be relative or absolute.

    Returns:
        The absolute and resolved Path to the checkpoint.
    """
    project_dir = _worldfoundry_repository_root()
    path = resolve_worldfoundry_path(value)
    # If the path is not absolute, treat it as relative to the repository root.
    if not path.is_absolute():
        path = project_dir / path
    return path.resolve()


def _checkpoint_candidate_paths(item: Mapping[str, Any]) -> list[Path]:
    """Generates a list of potential local paths for a given checkpoint item.

    This function attempts to find the actual location of a checkpoint based on
    its 'local_dir' or 'repo_id' keys, considering various conventions and environment variables.

    Args:
        item: A dictionary representing a checkpoint record, potentially containing
              'local_dir' (path) or 'repo_id' (Hugging Face ID).

    Returns:
        A list of unique, resolved Path objects that are potential locations
        for the checkpoint.
    """
    paths: list[Path] = []

    # Handle checkpoints specified by a 'local_dir' path.
    local_dir = str(item.get("local_dir") or "").strip()
    if local_dir:
        local_path = _expand_checkpoint_path(local_dir)
        paths.append(local_path)
        try:
            # Also check within the WORLDFOUNDRY_CKPT_DIR using the local_dir's base name.
            tokens = worldfoundry_path_tokens()
            ckpt_root = Path(tokens["WORLDFOUNDRY_CKPT_DIR"]).expanduser()
            paths.append((ckpt_root / local_path.name).resolve())
        except Exception:
            # Ignore if environment variable or path resolution fails.
            pass

    # Handle checkpoints specified by a 'repo_id' (Hugging Face ID).
    repo_id = str(item.get("repo_id") or "").strip()
    if repo_id:
        # Resolve paths based on common Hugging Face cache and WorldFoundry checkpoint directories.
        tokens = worldfoundry_path_tokens()
        hfd_root = Path(tokens["WORLDFOUNDRY_HFD_ROOT"]).expanduser()
        ckpt_root = Path(tokens["WORLDFOUNDRY_CKPT_DIR"]).expanduser()
        repo_slug = repo_id.replace("/", "--")  # Common pattern for converting repo_id to directory name.
        repo_name = repo_id.rsplit("/", 1)[-1]  # Last part of repo_id.
        paths.extend(
            [
                (hfd_root / repo_slug).resolve(),
                (hfd_root / repo_name).resolve(),
                (ckpt_root / repo_name).resolve(),
                (ckpt_root / repo_slug).resolve(),
            ]
        )

    # Deduplicate paths to avoid redundant checks.
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        text = str(path)
        if text in seen:
            continue
        seen.add(text)
        unique.append(path)
    return unique


def select_lingbot_va_checkpoint(
    *,
    checkpoint_dir: str | Path | None,
    checkpoints: Sequence[Mapping[str, Any]],
    config_name: str | None = None,
    require_exists: bool = True,
) -> Path:
    """Select a local LingBot-VA checkpoint directory based on explicit path or profile records.

    This function prioritizes an explicitly provided `checkpoint_dir`, then tries
    to match `config_name` against checkpoint roles/IDs, and finally iterates
    through all provided checkpoint records.

    Args:
        checkpoint_dir: An optional explicit checkpoint directory provided by the caller.
        checkpoints: A sequence of runtime-profile checkpoint records, each a mapping
                     that might contain 'local_dir', 'repo_id', or 'role'.
        config_name: An optional preferred official config key (e.g., "franka-base")
                     to prioritize matching checkpoints.
        require_exists: If True, only return a path that actually exists as a directory.

    Returns:
        The Path to the selected LingBot-VA checkpoint directory.

    Raises:
        FileNotFoundError: If no suitable checkpoint directory is found, especially
                           when `require_exists` is True.
    """
    if checkpoint_dir:
        explicit_path = _expand_checkpoint_path(checkpoint_dir)
        if not require_exists or explicit_path.is_dir():
            return explicit_path

    candidates: list[Mapping[str, Any]] = []
    # Extend with checkpoints from the runtime profile.
    candidates.extend(dict(item) for item in checkpoints)

    # If a preferred config name is provided, try to match it against checkpoint metadata.
    preferred = (config_name or "").lower()
    if preferred:
        for item in candidates:
            role = str(item.get("role") or "").lower()
            local_dir = str(item.get("local_dir") or "").lower()
            repo_id = str(item.get("repo_id") or "").lower()
            # Check if preferred config name is present in role, local_dir, or repo_id.
            if preferred in role or preferred in local_dir or preferred in repo_id:
                for path in _checkpoint_candidate_paths(item):
                    if not require_exists or path.is_dir():
                        return path

    # If no preferred config name or no match, iterate through all candidates in order.
    for item in candidates:
        for path in _checkpoint_candidate_paths(item):
            if not require_exists or path.is_dir():
                return path
    raise FileNotFoundError("No local LingBot-VA checkpoint directory was found.")


def config_name_for_checkpoint(
    checkpoint: Path,
    checkpoints: Sequence[Mapping[str, Any]],
    *,
    config_by_role: Mapping[str, str],
    fallback: str,
) -> str:
    """Infer the official LingBot-VA config name from profile checkpoint role.

    This function attempts to match a given `checkpoint` path against known
    checkpoint records and then uses a `config_by_role` mapping to determine
    the appropriate LingBot-VA configuration name.

    Args:
        checkpoint: The Path to the selected checkpoint directory.
        checkpoints: A sequence of runtime-profile checkpoint records.
        config_by_role: A dictionary mapping checkpoint roles (from profile) to
                        official LingBot-VA config keys.
        fallback: The config key to use if the checkpoint's role cannot be determined
                  or is not found in `config_by_role`.

    Returns:
        The inferred official LingBot-VA config name.
    """
    checkpoint_text = str(checkpoint.expanduser().resolve())
    for item in checkpoints:
        # Check if the current checkpoint matches any of the candidate paths for this item.
        if any(str(path) == checkpoint_text for path in _checkpoint_candidate_paths(item)):
            # If matched, look up the config name using the item's role.
            return config_by_role.get(str(item.get("role") or ""), fallback)
    # If no match is found, return the fallback config name.
    return fallback


def read_transformer_attn_mode(checkpoint_dir: str | Path) -> str | None:
    """Reads the 'attn_mode' from the LingBot-VA transformer config without importing torch.

    This utility function is useful for inspecting checkpoint properties efficiently.

    Args:
        checkpoint_dir: The local checkpoint directory containing the 'transformer/config.json'.

    Returns:
        The value of the 'attn_mode' key as a string if found, otherwise None.
    """
    config_path = Path(checkpoint_dir).expanduser().resolve() / "transformer" / "config.json"
    if not config_path.is_file():
        return None
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    value = payload.get("attn_mode")
    return str(value) if value is not None else None


def build_server_command(
    *,
    python: str,
    torchrun: str,
    config: LingBotVARuntimeConfig,
    save_root: str | Path,
) -> list[str]:
    """Builds the command-line arguments to launch the official in-tree LingBot-VA server.

    This command uses `torchrun` for distributed execution or falls back to
    `python -m torch.distributed.run` if `torchrun` is not directly available.

    Args:
        python: Path to the Python executable.
        torchrun: Path to the torchrun executable (if available).
        config: The resolved runtime configuration for the server.
        save_root: The directory where the official server will save visual outputs and logs.

    Returns:
        A list of strings representing the command and its arguments suitable for `subprocess.run`.
    """
    runner = torchrun or python
    command = [
        runner,
        "--nproc_per_node",
        str(config.nproc_per_node),
        "--master_port",
        str(config.master_port),
        "-m",
        f"{WAN_VA_PACKAGE}.runtime",
        "--config-name",
        config.config_name,
        "--port",
        str(config.port),
        "--save_root",
        str(save_root),
        "--checkpoint-dir",
        str(config.checkpoint_dir),
    ]
    # If torchrun is not used directly, prepend `python -m torch.distributed.run`.
    if runner == python:
        command[1:1] = ["-m", "torch.distributed.run"]
    return command


def write_action_trace(
    *,
    output_path: str | Path,
    action: Any,
    config: LingBotVARuntimeConfig,
    prompt: str,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Writes a standardized WorldFoundry LingBot-VA action trace artifact to a JSON file.

    This function serializes the inference result, runtime configuration, and other
    metadata into a JSON file, providing a complete record of an action generation event.

    Args:
        output_path: The file path where the JSON artifact will be saved.
        action: The raw action payload returned by the LingBot-VA server.
        config: The LingBotVARuntimeConfig used for the request.
        prompt: The language instruction used for inference.
        metadata: Additional run-specific metadata to include in the trace.

    Returns:
        A dictionary containing status and metadata about the written artifact,
        including its path and SHA256 hash.
    """
    target = Path(output_path).expanduser().resolve()
    # Ensure the parent directory for the output file exists.
    target.parent.mkdir(parents=True, exist_ok=True)

    # Construct the payload dictionary for the JSON artifact.
    payload = {
        "schema_version": "worldfoundry-lingbot-va-action-trace",
        "status": "success",
        "model_id": "lingbot-va",
        "artifact_kind": "action_trace",
        "backend": "worldfoundry.lingbot_va.in_tree_official_websocket_client",
        "backend_quality": "official_server_client",
        "checkpoint_dir": str(config.checkpoint_dir),
        "config_name": config.config_name,
        "host": config.host,
        "port": config.port,
        "prompt": prompt,
        "action": _jsonable(action),
        "metadata": _jsonable(dict(metadata)),
    }
    # Write the payload to the specified output path.
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Return summary information about the created artifact.
    return {
        "status": "success",
        "model_id": "lingbot-va",
        "artifact_kind": "action_trace",
        "artifact_path": str(target),
        "artifact_sha256": file_sha256(target),
        "backend": payload["backend"],
        "backend_quality": payload["backend_quality"],
    }


def _lingbot_va_camera_keys(config_name: str) -> tuple[str, ...]:
    """Determines the relevant camera observation keys for a given LingBot-VA configuration.

    Attempts to load the configuration dynamically from the `wan_va` package.
    If that fails, it falls back to common key sets based on the config name.

    Args:
        config_name: The official LingBot-VA config key (e.g., "franka-base").

    Returns:
        A tuple of strings representing the camera observation keys expected by the model.
    """
    try:
        # Attempt to dynamically load camera keys from the installed WAN-VA package configuration.
        from worldfoundry.synthesis.action_generation.wan_va.configuration import VA_CONFIGS

        config = VA_CONFIGS.get(config_name)
        keys = tuple(str(key) for key in getattr(config, "obs_cam_keys", ()) if key)
        if keys:
            return keys
    except Exception:
        # If dynamic loading fails (e.g., package not fully installed), fall back.
        pass

    # Fallback logic based on common config name patterns.
    normalized = config_name.lower()
    if "robotwin" in normalized:
        return (
            "observation.images.cam_high",
            "observation.images.cam_left_wrist",
            "observation.images.cam_right_wrist",
        )
    if "franka" in normalized:
        return (
            "observation.images.cam_high",
            "observation.images.cam_left_wrist",
            "observation.images.cam_right_wrist",
        )
    if "demo" in normalized:
        return ("observation.images.top", "observation.images.wrist")
    # Default keys if no specific match is found.
    return ("observation.images.agentview_rgb", "observation.images.eye_in_hand_rgb")


def _as_hwc_uint8_image(value: Any) -> Any:
    """Converts various image inputs into a Height x Width x Channel (HWC) uint8 NumPy array.

    Handles file paths, PIL Images, PyTorch tensors, and NumPy arrays.
    Normalizes to 3 channels (RGB) and uint8 data type, scaling if necessary.

    Args:
        value: The image data, which can be a path, PIL Image, tensor, or NumPy array.

    Returns:
        A contiguous NumPy array representing the image in HWC uint8 format.

    Raises:
        ValueError: If the input array has an unexpected number of dimensions.
    """
    import numpy as np
    from PIL import Image

    # Load image from path or convert PIL Image/tensor/array to NumPy.
    if isinstance(value, (str, Path)):
        with Image.open(Path(value).expanduser()) as image:
            array = np.asarray(image.convert("RGB"))
    elif isinstance(value, Image.Image):
        array = np.asarray(value.convert("RGB"))
    elif hasattr(value, "detach"):
        # Handle PyTorch tensors: detach from graph and move to CPU before converting to NumPy.
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)

    # Ensure 3 dimensions (HWC).
    if array.ndim == 4:
        # Handle batched images (remove batch dimension).
        array = array[0]
    if array.ndim == 3 and array.shape[0] in {1, 3, 4} and array.shape[-1] not in {1, 3, 4}:
        # Handle CHW (Channel, Height, Width) format by transposing to HWC.
        array = np.transpose(array, (1, 2, 0))
    if array.ndim == 2:
        # Handle grayscale images by repeating the single channel to make 3 channels.
        array = np.repeat(array[..., None], 3, axis=-1)
    if array.ndim != 3:
        raise ValueError(f"LingBot-VA image input must be 2D, 3D, or batched 4D; got shape {array.shape}.")

    # Ensure 3 channels.
    if array.shape[-1] == 1:
        # If single channel, repeat to make 3 channels (e.g., grayscale to RGB).
        array = np.repeat(array, 3, axis=-1)
    elif array.shape[-1] > 3:
        # If more than 3 channels (e.g., RGBA), keep only the first 3 (RGB).
        array = array[..., :3]

    # Convert to uint8 data type, scaling if necessary.
    if array.dtype != np.uint8:
        array = array.astype(np.float32)
        # If max value is <= 1.0, assume values are normalized [0, 1] and scale to [0, 255].
        if array.size and float(np.nanmax(array)) <= 1.0:
            array = array * 255.0
        # Clip values to [0, 255] and convert to uint8.
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def _normalize_lingbot_va_frame(source: Any, camera_keys: Sequence[str]) -> dict[str, Any]:
    """Normalizes a single observation frame into a dictionary of HWC uint8 images.

    This function intelligently extracts image data from various input structures
    (e.g., nested dictionaries, lists of images) and formats them according to
    the specified `camera_keys`.

    Args:
        source: The raw observation frame, which can be a mapping, a sequence, or a single image.
        camera_keys: A sequence of strings representing the desired camera observation keys.

    Returns:
        A dictionary where keys are `camera_keys` and values are normalized HWC uint8 NumPy arrays.

    Raises:
        ValueError: If the observation mapping or image sequence is empty, or if no images are found.
    """
    if isinstance(source, Mapping):
        # Handle nested observation structures common in some frameworks.
        if "rgb_views" in source and not any(key in source for key in camera_keys):
            return _normalize_lingbot_va_frame(source["rgb_views"], camera_keys)
        if "image" in source and not any(key in source for key in camera_keys):
            return _normalize_lingbot_va_frame(source["image"], camera_keys)
        if "images" in source and not any(key in source for key in camera_keys):
            return _normalize_lingbot_va_frame(source["images"], camera_keys)

        # Extract values corresponding to the camera keys or all values if specific keys not found.
        values: list[Any] = []
        for key in camera_keys:
            if key in source:
                values.append(source[key])
        if not values:
            values = list(source.values()) # Fallback: use all values if no specific camera_keys match
        if not values:
            raise ValueError("LingBot-VA observation mapping is empty.")

        # Normalize each extracted image value. If fewer images than camera_keys, repeat the last.
        frame: dict[str, Any] = {}
        for index, key in enumerate(camera_keys):
            frame[key] = _as_hwc_uint8_image(values[min(index, len(values) - 1)])
        return frame

    if isinstance(source, (list, tuple)):
        if not source:
            raise ValueError("LingBot-VA observation image sequence is empty.")
        # If it's a list of mappings, assume the first mapping contains the frame.
        if all(isinstance(item, Mapping) for item in source):
            return _normalize_lingbot_va_frame(source[0], camera_keys)
        # Otherwise, assume it's a list of images and normalize them directly.
        frame = {}
        for index, key in enumerate(camera_keys):
            frame[key] = _as_hwc_uint8_image(source[min(index, len(source) - 1)])
        return frame

    # If the source is a single image (not a mapping or sequence), normalize it and replicate for all keys.
    image = _as_hwc_uint8_image(source)
    return {key: image for key in camera_keys}


def _normalize_lingbot_va_obs(source: Any, camera_keys: Sequence[str]) -> list[dict[str, Any]]:
    """Normalizes various forms of LingBot-VA observations into a consistent list of frames.

    An observation can be a single frame (mapping or image) or a sequence of frames.
    Each frame is normalized to a dictionary where keys are camera keys and values are
    HWC uint8 NumPy arrays.

    Args:
        source: The raw observation data. Can be a mapping representing a single frame,
                a sequence of mappings (multiple frames), or a single image.
        camera_keys: A sequence of strings representing the desired camera observation keys.

    Returns:
        A list of dictionaries, where each dictionary represents a normalized frame.
    """
    if isinstance(source, Mapping):
        # If a single mapping, normalize it as one frame.
        return [_normalize_lingbot_va_frame(source, camera_keys)]
    if isinstance(source, (list, tuple)) and source and all(isinstance(item, Mapping) for item in source):
        # If a sequence of mappings, normalize each mapping as a separate frame.
        return [_normalize_lingbot_va_frame(item, camera_keys) for item in source]
    # Otherwise, treat the source as a single image or an unstructured sequence of images,
    # normalize it as a single frame.
    return [_normalize_lingbot_va_frame(source, camera_keys)]


def _normalize_lingbot_va_state(value: Any) -> Any:
    """Normalizes a robot state input into a float32 NumPy array.

    Ensures the array is at least 2D (batch dimension, features) and contiguous.

    Args:
        value: The raw robot state, which can be a tensor or NumPy array.

    Returns:
        A contiguous NumPy array representing the normalized robot state.
    """
    import numpy as np

    # Convert PyTorch tensor to NumPy array if applicable.
    array = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    array = array.astype(np.float32)
    # Ensure the array is at least 2D (e.g., [N,] becomes [1, N]).
    if array.ndim == 1:
        array = array[None, :]
    return np.ascontiguousarray(array)


class LingBotVAWebsocketRuntime:
    """A thin in-tree LingBot-VA client for interacting with an already running official server.

    This class provides methods to connect to a LingBot-VA WebSocket server and
    perform action predictions, handling observation and state normalization.
    """

    def __init__(self, config: LingBotVARuntimeConfig) -> None:
        """Create a lazy WebSocket client runtime.

        The client connection is initialized only when the first inference request is made.

        Args:
            config: The resolved server/checkpoint configuration, including host and port.
        """
        self.config = config
        self._client: Any | None = None

    def _client_for_server(self) -> Any:
        """Lazily initializes and returns the WebSocket client instance.

        If the server host is '0.0.0.0' or '::', it's translated to '127.0.0.1' for client connection.

        Returns:
            An instance of WebsocketClientPolicy connected to the LingBot-VA server.
        """
        if self._client is None:
            from worldfoundry.synthesis.action_generation.wan_va.websocket_client import (
                WebsocketClientPolicy,
            )

            # Map generic listen addresses to localhost for client connection.
            host = "127.0.0.1" if self.config.host in {"0.0.0.0", "::"} else self.config.host
            self._client = WebsocketClientPolicy(host=host, port=self.config.port)
        return self._client

    def predict_action(
        self,
        *,
        observation: Mapping[str, Any],
        prompt: str,
        output_path: str | Path,
        extra_metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Runs one inference request through the official LingBot-VA WebSocket client.

        This method prepares the observation and state data, sends the request to the
        LingBot-VA server, and then writes the response as a standardized action trace.

        Args:
            observation: The raw server observation payload, potentially containing image data
                         (e.g., 'obs', 'rgb_views', 'image', 'images') and other environment data.
            prompt: The language instruction for the action prediction.
            output_path: The file path for the JSON action trace artifact.
            extra_metadata: Optional additional WorldFoundry metadata to include in the trace.

        Returns:
            A dictionary containing status and metadata about the written action trace artifact.

        Raises:
            ValueError: If no valid image source is found in the observation for inference.
        """
        client = self._client_for_server()
        request = dict(observation)
        request.setdefault("prompt", prompt)

        # Determine camera keys based on the configured model.
        camera_keys = _lingbot_va_camera_keys(self.config.config_name)

        # Normalize observation images.
        if "obs" in request:
            # If 'obs' key is present, normalize its content.
            request["obs"] = _normalize_lingbot_va_obs(request["obs"], camera_keys)
        else:
            # If 'obs' is not present, search for common image keys and normalize.
            image_source = None
            for key in ("rgb_views", "image", "images", "image_path", "input_path"):
                if key in request and request[key] is not None:
                    image_source = request[key]
                    break
            if image_source is None:
                raise ValueError("LingBot-VA live inference requires obs, rgb_views, image, images, image_path, or input_path.")
            request["obs"] = _normalize_lingbot_va_obs(image_source, camera_keys)

        # Normalize robot state data.
        state = request.pop("robot_state", None) # Pop 'robot_state' if present.
        if "state" in request:
            # If 'state' key is present, normalize its content.
            request["state"] = _normalize_lingbot_va_state(request["state"])
        elif state is not None:
            # If 'robot_state' was popped, use that.
            request["state"] = _normalize_lingbot_va_state(state)

        # Remove redundant image source keys from the request after normalization.
        for key in ("rgb_views", "image", "images", "image_path", "input_path"):
            request.pop(key, None)

        # Send the normalized request to the server and get the response.
        response = client.infer(request)
        # Write the action trace artifact and return its summary.
        return write_action_trace(
            output_path=output_path,
            action=response.get("action", response),  # Use 'action' key if present, otherwise the whole response.
            config=self.config,
            prompt=prompt,
            metadata=extra_metadata or {},
        )


__all__ = [
    "LingBotVARuntimeConfig",
    "LingBotVAWebsocketRuntime",
    "RUNTIME_ROOT",
    "WAN_VA_ROOT",
    "build_server_command",
    "config_name_for_checkpoint",
    "read_transformer_attn_mode",
    "select_lingbot_va_checkpoint",
    "write_action_trace",
]
