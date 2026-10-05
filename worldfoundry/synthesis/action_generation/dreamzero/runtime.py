"""
This module provides utilities for interacting with the DreamZero runtime,
including configuration loading, server command building, client demo execution,
and data processing for the official DreamZero RoboArena WebSocket server.

It defines functions for validating runtime parameters, constructing command-line
arguments for the distributed server, and a client implementation for
simulating a demo interaction with a running DreamZero inference server.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from worldfoundry.core.io import file_sha256
from worldfoundry.core.io.paths import project_root, resolve_worldfoundry_path
from worldfoundry.synthesis.action_generation.dreamzero import runtime_root as dreamzero_runtime_root
from worldfoundry.synthesis.action_generation.runtime_config import load_vla_va_wam_runtime_config

# Root directory for the DreamZero runtime files.
RUNTIME_ROOT = dreamzero_runtime_root()
# Full module path for the optimized DreamZero AR server entrypoint.
OFFICIAL_SERVER_MODULE = (
    "worldfoundry.synthesis.action_generation.dreamzero.server"
)


def _require_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    """Ensures a given value is a mapping (e.g., dict) and returns it.

    Args:
        value: The value to check.
        field_name: The name of the field being checked, for error messages.

    Returns:
        The validated mapping.

    Raises:
        TypeError: If the value is not a mapping.
    """
    if not isinstance(value, Mapping):
        raise TypeError(f"DreamZero runtime config field {field_name!r} must be a mapping.")
    return value


def _require_text(value: Any, field_name: str) -> str:
    """Ensures a given value is non-empty text (string) after stripping whitespace.

    Args:
        value: The value to check and convert to text.
        field_name: The name of the field being checked, for error messages.

    Returns:
        The validated non-empty string.

    Raises:
        ValueError: If the value is empty or becomes empty after stripping.
    """
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"DreamZero runtime config field {field_name!r} is required.")
    return text


def _tuple_of_ints(value: Any, field_name: str) -> tuple[int, ...]:
    """Ensures a given value is a sequence of integers and returns it as a tuple.

    Args:
        value: The value to check and convert.
        field_name: The name of the field being checked, for error messages.

    Returns:
        A tuple of integers.

    Raises:
        TypeError: If the value is not a sequence or its items cannot be converted to int.
    """
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"DreamZero runtime config field {field_name!r} must be a list of integers.")
    return tuple(int(item) for item in value)


def _client_demo_config(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Loads and merges the default client demo configuration with any provided overrides.

    This function retrieves the base client demo settings from the global `DREAMZERO_RUNTIME_CONFIG`,
    applies type validation, and then merges in any additional configuration provided
    by the `config` argument.

    Args:
        config: Optional mapping of configuration overrides for the client demo.

    Returns:
        A dictionary containing the merged and validated client demo configuration.
    """
    # Load default client demo configuration and ensure it's a mapping.
    merged = dict(_require_mapping(DREAMZERO_RUNTIME_CONFIG.get("client_demo"), "client_demo"))
    # Apply any provided configuration overrides.
    if config:
        merged.update(dict(config))
    # Validate and convert specific configuration fields.
    merged["prompt"] = _require_text(merged.get("prompt"), "client_demo.prompt")
    merged["camera_files"] = {
        str(camera_key): str(file_name)
        for camera_key, file_name in _require_mapping(merged.get("camera_files"), "client_demo.camera_files").items()
    }
    merged["relative_offsets"] = _tuple_of_ints(merged.get("relative_offsets"), "client_demo.relative_offsets")
    merged["action_horizon"] = int(merged["action_horizon"])
    merged["num_chunks"] = int(merged["num_chunks"])
    merged["zero_image_height"] = int(merged["zero_image_height"])
    merged["zero_image_width"] = int(merged["zero_image_width"])
    return merged


# Global variable holding the loaded DreamZero runtime configuration.
DREAMZERO_RUNTIME_CONFIG = load_vla_va_wam_runtime_config("dreamzero")


@dataclass(frozen=True)
class DreamZeroRuntimeConfig:
    """Describe an in-tree DreamZero official server runtime.

    Args:
        checkpoint_dir: External DreamZero checkpoint directory.
        host: Server bind host.
        port: WebSocket server port.
        nproc_per_node: Number of distributed worker processes.
        enable_dit_cache: Whether to enable DiT KV/cache mode.
        max_chunk_size: Optional official max_chunk_size override.
        client_demo: Configuration specific to the client demo.
    """

    checkpoint_dir: Path
    host: str
    port: int
    nproc_per_node: int
    enable_dit_cache: bool
    max_chunk_size: int | None
    client_demo: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Converts the runtime configuration to a dictionary of JSON-serializable types."""
        return {
            "checkpoint_dir": str(self.checkpoint_dir),
            "host": self.host,
            "port": self.port,
            "nproc_per_node": self.nproc_per_node,
            "enable_dit_cache": self.enable_dit_cache,
            "max_chunk_size": self.max_chunk_size,
            "client_demo": _jsonable(dict(self.client_demo)),
        }


def _jsonable(value: Any) -> Any:
    """Recursively converts various Python types into JSON-serializable types.

    Handles mappings, sequences, Path objects, numpy arrays (via .tolist()),
    PyTorch tensors (via .detach().cpu().tolist()), and scalar tensors (via .item()).
    Basic types (str, int, float, bool, None) are returned as is. Other types are
    converted to their string representation.

    Args:
        value: The value to convert.

    Returns:
        A JSON-serializable representation of the input value.
    """
    if isinstance(value, Mapping):
        # Recursively convert dictionary items.
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        # Recursively convert list/tuple items.
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        # Convert Path objects to string.
        return str(value)
    if hasattr(value, "tolist"):
        # Handle numpy arrays and similar objects.
        return _jsonable(value.tolist())
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        # Handle PyTorch tensors.
        return _jsonable(value.detach().cpu().tolist())
    if hasattr(value, "item"):
        # Handle scalar tensors.
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        # Basic JSON-serializable types.
        return value
    # Default to string representation for unknown types.
    return str(value)


def _expand_runtime_path(value: str | Path) -> Path:
    """Expands a given path relative to the worldfoundry project root if it's not absolute.

    Args:
        value: The path to expand. Can be a string or Path object.

    Returns:
        An absolute and resolved Path object.
    """
    path = resolve_worldfoundry_path(value)
    if not path.is_absolute():
        # If the path is not absolute, treat it as relative to the project root.
        path = project_root() / path
    return path.resolve()


def _checkpoint_exists(path: str | Path) -> bool:
    """Checks if a given path corresponds to a valid DreamZero checkpoint.

    A valid checkpoint is considered to be a directory containing an
    `experiment_cfg/conf.yaml` file.

    Args:
        path: The path to the potential checkpoint directory.

    Returns:
        True if the checkpoint is valid and exists, False otherwise.
    """
    checkpoint = _expand_runtime_path(path)
    return checkpoint.is_dir() and (checkpoint / "experiment_cfg" / "conf.yaml").is_file()


def select_dreamzero_checkpoint(
    *,
    checkpoint_dir: str | Path | None,
    checkpoints: Sequence[Mapping[str, Any]],
    variant: str | None = None,
    require_exists: bool = False,
) -> Path:
    """Resolves a DreamZero checkpoint path from explicit input or profile metadata.

    This function prioritizes an explicitly provided `checkpoint_dir`. If not provided,
    it attempts to find a suitable checkpoint from a list of candidates based on
    optional variant filtering and local existence.

    Args:
        checkpoint_dir: Explicit external checkpoint path to use, if provided.
        checkpoints: A sequence of checkpoint candidate mappings, typically from a profile.
                      Each mapping should ideally contain 'id', 'repo_id', and 'local_dir'.
        variant: Optional variant ID or repository ID fragment to filter checkpoint candidates.
        require_exists: If True, raises FileNotFoundError if the selected checkpoint path
                        does not exist locally and contain the expected `conf.yaml`.

    Returns:
        An absolute Path object pointing to the selected DreamZero checkpoint directory.

    Raises:
        ValueError: If no checkpoint can be determined from the inputs.
        FileNotFoundError: If `require_exists` is True and the selected checkpoint
                           does not exist or is invalid.
    """
    if checkpoint_dir:
        # If an explicit checkpoint directory is provided, use it directly.
        selected = _expand_runtime_path(checkpoint_dir)
    else:
        checkpoint_items = [
            dict(item)
            for item in checkpoints
            if str(item.get("role") or "").endswith("_checkpoint")
        ]
        candidates = checkpoint_items or [dict(item) for item in checkpoints]
        if not candidates:
            raise ValueError("DreamZero requires checkpoint_dir or profile checkpoint metadata.")

        if variant:
            # Filter candidates by variant if provided. If no matches, fall back to all candidates.
            variant_text = variant.lower()
            candidates = [
                item
                for item in candidates
                if variant_text in str(item.get("id") or "").lower()
                or variant_text in str(item.get("repo_id") or "").lower()
            ] or list(candidates)

        # Prioritize existing local checkpoints if available.
        existing = [item for item in candidates if item.get("local_dir") and _checkpoint_exists(str(item["local_dir"]))]
        # Select the first existing candidate, or the first general candidate if none exist.
        selected_item = (existing or candidates)[0] if (existing or candidates) else {}
        selected_text = str(selected_item.get("local_dir") or "")
        if not selected_text:
            raise ValueError("DreamZero requires checkpoint_dir or profile checkpoint metadata.")
        selected = _expand_runtime_path(selected_text)

    if str(selected) in {"", "."}:
        raise ValueError("DreamZero requires checkpoint_dir or profile checkpoint metadata.")
    if require_exists and not _checkpoint_exists(selected):
        raise FileNotFoundError(f"DreamZero checkpoint is not staged or missing experiment_cfg/conf.yaml: {selected}")
    return selected


def describe_in_tree_runtime(checkpoint_dir: str | Path | None = None) -> dict[str, Any]:
    """Return DreamZero in-tree runtime provenance without importing heavy deps.

    This function provides metadata about the DreamZero runtime environment,
    including paths to key files and known limitations, without loading
    any machine learning or heavy dependencies.

    Args:
        checkpoint_dir: Optional external checkpoint directory to describe.
                        If provided, its existence and configuration file presence will be checked.

    Returns:
        A dictionary containing descriptive metadata about the DreamZero runtime.
    """
    checkpoint = _expand_runtime_path(checkpoint_dir) if checkpoint_dir else None
    return {
        "runtime_root": str(RUNTIME_ROOT),
        "official_server_module": OFFICIAL_SERVER_MODULE,
        "source_evidence": {
            "server_entrypoint": str(RUNTIME_ROOT / "server.py"),
            "official_source_entrypoint": "socket_test_optimized_AR.py",
            "wan22_server_entrypoint": str(RUNTIME_ROOT / "serve_wan22.py"),
            "policy_class": "worldfoundry.synthesis.action_generation.dreamzero.modeling.policy.GrootSimPolicy",
            "model_class": "worldfoundry.synthesis.action_generation.dreamzero.modeling.vla.VLA",
        },
        "checkpoint": {
            "path": "" if checkpoint is None else str(checkpoint),
            "exists": bool(checkpoint and checkpoint.is_dir()),
            "has_experiment_cfg": bool(checkpoint and (checkpoint / "experiment_cfg" / "conf.yaml").is_file()),
        },
        "known_blockers": [
            "Official DreamZero inference still requires external checkpoint assets.",
            "The default 14B runtime requires Python 3.11, torch 2.8/cu129-era CUDA dependencies, and two CUDA ranks; the Wan2.2 5B entrypoint can use one rank.",
            "Flash Attention and TensorRT are optional local accelerators; PyTorch SDPA remains available and TensorRT loads only when LOAD_TRT_ENGINE is explicitly configured.",
            "Live inference requires launching the in-tree distributed WebSocket server before the client demo.",
        ],
    }


def build_official_server_command(
    *,
    checkpoint_dir: str | Path,
    python: str,
    repo_root: str | Path | None = None,
    port: int,
    nproc_per_node: int,
    enable_dit_cache: bool,
    max_chunk_size: int | None,
) -> list[str]:
    """Return DreamZero's in-tree official distributed server command.

    Constructs the command-line arguments needed to launch the DreamZero
    distributed WebSocket server using `torch.distributed.run`.

    Args:
        checkpoint_dir: External DreamZero checkpoint directory.
        python: Python executable used by torch.distributed.run.
        repo_root: Deprecated external checkout root; ignored for in-tree execution.
        port: WebSocket server port.
        nproc_per_node: Number of distributed worker processes.
        enable_dit_cache: Whether to enable DiT cache.
        max_chunk_size: Optional official max_chunk_size override.

    Returns:
        A list of strings representing the command and its arguments.
    """
    del repo_root  # repo_root is deprecated and ignored for in-tree execution.

    command = [
        python,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node",
        str(nproc_per_node),
        "--module",
        OFFICIAL_SERVER_MODULE,
        "--port",
        str(port),
    ]
    if enable_dit_cache:
        command.append("--enable-dit-cache")
    command.extend(["--model-path", str(checkpoint_dir)])
    if max_chunk_size is not None:
        command.extend(["--max-chunk-size", str(max_chunk_size)])
    return command


def build_server_command(
    *,
    python: str,
    config: DreamZeroRuntimeConfig,
) -> list[str]:
    """Return the in-tree DreamZero server command for a resolved config.

    This is a convenience wrapper around `build_official_server_command`
    that takes a `DreamZeroRuntimeConfig` object.

    Args:
        python: Python executable used by torch.distributed.run.
        config: Resolved DreamZero runtime configuration.

    Returns:
        A list of strings representing the command and its arguments.
    """
    return build_official_server_command(
        checkpoint_dir=config.checkpoint_dir,
        python=python,
        port=config.port,
        nproc_per_node=config.nproc_per_node,
        enable_dit_cache=config.enable_dit_cache,
        max_chunk_size=config.max_chunk_size,
    )


def load_all_frames(video_path: str | Path) -> Any:
    """Load official DreamZero debug video frames as RGB uint8 arrays.

    Requires `opencv-python` and `numpy` to be installed.

    Args:
        video_path: Path to the video file.

    Returns:
        A numpy array of shape (num_frames, height, width, 3) with RGB uint8 pixel data.

    Raises:
        RuntimeError: If no frames could be loaded from the video.
    """
    # Imports are intentionally placed here to avoid making cv2/numpy hard dependencies for the module.
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR frame from OpenCV to RGB.
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames loaded from {video_path}")
    return np.stack(frames, axis=0)


def load_camera_frames(video_dir: str | Path, camera_files: Mapping[str, str]) -> dict[str, Any]:
    """Loads video frames for multiple cameras from specified files within a directory.

    Args:
        video_dir: The base directory where camera video files are located.
        camera_files: A mapping where keys are camera identifiers (e.g., 'front_camera')
                      and values are the corresponding video filenames within `video_dir`.

    Returns:
        A dictionary where keys are camera identifiers and values are numpy arrays
        containing the loaded RGB frames for each camera.
    """
    video_dir = Path(video_dir)
    files = dict(camera_files)
    return {
        camera_key: load_all_frames(video_dir / file_name)
        for camera_key, file_name in files.items()
    }


def build_frame_schedule(
    total_frames: int,
    num_chunks: int,
    *,
    relative_offsets: Sequence[int],
    action_horizon: int,
) -> list[list[int]]:
    """Builds a schedule of frame indices for DreamZero inference steps.

    The schedule defines which frames from the debug video should be used
    for each inference chunk, based on a starting frame, relative offsets,
    and an action horizon.

    Args:
        total_frames: The total number of frames available in the video.
        num_chunks: The desired number of inference chunks (steps).
        relative_offsets: A sequence of integer offsets relative to the current
                          frame to select multiple observation frames per chunk.
        action_horizon: The number of frames to advance between chunks.

    Returns:
        A list of lists, where each inner list contains the frame indices
        for a single inference chunk.
    """
    offsets = tuple(int(item) for item in relative_offsets)
    horizon = int(action_horizon)
    chunks: list[list[int]] = []
    current_frame = 23  # DreamZero's official client starts at frame 23.
    for _ in range(num_chunks):
        # Calculate the specific frame indices for the current chunk.
        indices = [max(current_frame + offset, 0) for offset in offsets]
        # Stop if the last required frame for the current chunk exceeds total available frames.
        if indices[-1] >= total_frames:
            break
        chunks.append(indices)
        # Advance the current frame by the action horizon for the next chunk.
        current_frame += horizon
    return chunks


def _make_obs_from_video(
    camera_frames: Mapping[str, Any],
    frame_indices: Sequence[int],
    prompt: str,
    session_id: str,
) -> dict[str, Any]:
    """Constructs an observation dictionary from video frames for a given step.

    Includes camera images, zeroed joint/cartesian/gripper positions,
    the prompt, and session ID, mimicking the DreamZero client's input format.

    Args:
        camera_frames: A mapping from camera key to a numpy array of all frames for that camera.
        frame_indices: A sequence of integer indices indicating which frames to select for this observation.
        prompt: The text prompt for the task.
        session_id: A unique identifier for the current demo session.

    Returns:
        A dictionary representing the observation for a single inference step.
    """
    import numpy as np

    obs: dict[str, Any] = {}
    for camera_key, all_frames in camera_frames.items():
        # Select specific frames for each camera based on the provided indices.
        selected = all_frames[list(frame_indices)]
        if len(frame_indices) == 1:
            # If only one frame is selected, flatten the leading dimension.
            selected = selected[0]
        obs[camera_key] = selected

    # DreamZero observations also include dummy kinematic states.
    obs["observation/joint_position"] = np.zeros(7, dtype=np.float32)
    obs["observation/cartesian_position"] = np.zeros(6, dtype=np.float32)
    obs["observation/gripper_position"] = np.zeros(1, dtype=np.float32)
    obs["prompt"] = prompt
    obs["session_id"] = session_id
    return obs


def _make_zero_observation(
    prompt: str,
    session_id: str,
    *,
    camera_files: Mapping[str, str],
    height: int,
    width: int,
) -> dict[str, Any]:
    """Constructs an observation dictionary with black/zero images.

    This is used when `use_zero_images` is True, simulating a live inference
    without actual video input by providing zeroed-out image arrays.

    Args:
        prompt: The text prompt for the task.
        session_id: A unique identifier for the current demo session.
        camera_files: A mapping of camera keys to filenames (used to determine camera names).
        height: The desired height for the zero images.
        width: The desired width for the zero images.

    Returns:
        A dictionary representing the observation with zeroed images and dummy kinematic states.
    """
    import numpy as np

    obs: dict[str, Any] = {}
    for camera_key in dict(camera_files):
        # Create black images (all zeros) for each camera.
        obs[camera_key] = np.zeros((height, width, 3), dtype=np.uint8)
    # DreamZero observations also include dummy kinematic states.
    obs["observation/joint_position"] = np.zeros(7, dtype=np.float32)
    obs["observation/cartesian_position"] = np.zeros(6, dtype=np.float32)
    obs["observation/gripper_position"] = np.zeros(1, dtype=np.float32)
    obs["prompt"] = prompt
    obs["session_id"] = session_id
    return obs


class DreamZeroWebsocketClient:
    """Minimal DreamZero/RoboArena WebSocket client matching the official test client.

    This client provides basic functionality to connect to a DreamZero WebSocket server,
    retrieve server metadata, perform inference, and send reset commands,
    using `msgpack-numpy` for serialization/deserialization.
    """

    def __init__(self, host: str, port: int) -> None:
        """Initializes the WebSocket client and connects to the server.

        Args:
            host: The hostname or IP address of the DreamZero server.
            port: The port number of the DreamZero WebSocket server.
        """
        # Imports are placed here to avoid making WebSocket transport a module import dependency.
        import websockets.sync.client

        from . import protocol as msgpack_numpy

        self._msgpack_numpy = msgpack_numpy
        self._packer = msgpack_numpy.Packer()
        self._uri = f"ws://{host}:{port}"
        # Establish a synchronous WebSocket connection.
        self._ws = websockets.sync.client.connect(
            self._uri,
            compression=None,  # Compression disabled as per official client.
            max_size=None,     # No limit on message size.
            ping_interval=60,  # Send a ping every 60 seconds.
            ping_timeout=600,  # Close connection if no pong received within 600 seconds.
        )
        # The first message from the server is typically its metadata.
        self._server_metadata = msgpack_numpy.unpackb(self._ws.recv())

    @property
    def server_metadata(self) -> Mapping[str, Any]:
        """Returns the metadata received from the connected DreamZero server."""
        return self._server_metadata

    def infer(self, obs: Mapping[str, Any]) -> Any:
        """Sends an observation to the server for inference and receives actions.

        Args:
            obs: The observation dictionary to send.

        Returns:
            The raw response from the server, typically containing action predictions.

        Raises:
            RuntimeError: If the server returns a string response, indicating an error.
        """
        payload = dict(obs)
        payload["endpoint"] = "infer"
        self._ws.send(self._packer.pack(payload))
        response = self._ws.recv()
        # The official DreamZero server returns error messages as strings.
        if isinstance(response, str):
            raise RuntimeError(f"DreamZero inference server returned an error:\n{response}")
        return self._msgpack_numpy.unpackb(response)

    def reset(self) -> Any:
        """Sends a reset command to the server.

        Returns:
            The raw response from the server to the reset command.
        """
        self._ws.send(self._packer.pack({"endpoint": "reset"}))
        return self._ws.recv()


def _validate_metadata(metadata: Mapping[str, Any]) -> None:
    """Validates essential server metadata for DreamZero's official client demo.

    Checks for expected values like the number of external cameras,
    wrist camera requirement, and action space type.

    Args:
        metadata: The server metadata dictionary to validate.

    Raises:
        RuntimeError: If any of the expected metadata values do not match.
    """
    if int(metadata.get("n_external_cameras", 2)) != 2:
        raise RuntimeError(f"DreamZero server metadata expected 2 external cameras, got {metadata}")
    if metadata.get("needs_wrist_camera") is False:
        raise RuntimeError(f"DreamZero server metadata expected wrist camera, got {metadata}")
    if str(metadata.get("action_space", "joint_position")) != "joint_position":
        raise RuntimeError(f"DreamZero server metadata expected joint_position action space, got {metadata}")


def run_default_client_demo(
    *,
    host: str,
    port: int,
    prompt: str = "",
    output_path: str | Path | None = None,
    model_id: str = "dreamzero",
    artifact_kind: str = "action_trace",
    debug_video_dir: str | Path | None = None,
    num_chunks: int | None = None,
    use_zero_images: bool = False,
    session_id: str | None = None,
    client_demo_config: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run DreamZero's official default WebSocket-client demo against a live server.

    This function simulates a client interacting with a DreamZero inference server.
    It can either process frames from a debug video or send zeroed images.
    The results, including actions and timing, are saved as a JSON artifact.

    Args:
        host: The hostname or IP address of the DreamZero server.
        port: The port number of the DreamZero WebSocket server.
        prompt: The text prompt for the task. Overrides demo config if provided.
        output_path: Optional path to save the generated action trace JSON artifact.
                     Defaults to "dreamzero_action_trace.json".
        model_id: Identifier for the model being evaluated.
        artifact_kind: The type of artifact being generated (e.g., "action_trace").
        debug_video_dir: Directory containing debug video files (e.g., `_episode.mp4`).
                         Required if `use_zero_images` is False.
        num_chunks: Optional number of inference chunks to run. Overrides demo config.
        use_zero_images: If True, sends black images instead of loading from video.
                         Useful for testing server readiness without video assets.
        session_id: Optional unique identifier for the demo session. A UUID is generated if not provided.
        client_demo_config: Optional mapping for overriding default client demo settings.
        extra: Optional mapping for including additional arbitrary data in the output artifact.

    Returns:
        A dictionary summarizing the demo execution, including artifact path and SHA256 digest.

    Raises:
        ValueError: If `debug_video_dir` is not provided when `use_zero_images` is False.
        RuntimeError: If server metadata validation fails.
    """

    started = time.monotonic()
    # Load and merge the client demo configuration.
    demo_config = _client_demo_config(client_demo_config)
    camera_files = dict(demo_config["camera_files"])
    prompt = prompt or str(demo_config["prompt"])
    chunk_count = int(num_chunks if num_chunks is not None else demo_config["num_chunks"])
    session = session_id or str(uuid.uuid4())

    # Initialize DreamZero WebSocket client and retrieve server metadata.
    client = DreamZeroWebsocketClient(host=host, port=port)
    metadata = dict(client.server_metadata)
    # Validate the server metadata against expected DreamZero properties.
    _validate_metadata(metadata)

    steps: list[dict[str, Any]] = []
    if use_zero_images:
        frame_schedule: list[list[int]] = []  # No specific frames when using zero images.
        for index in range(chunk_count):
            # Construct observation with zero images.
            obs = _make_zero_observation(
                prompt,
                session,
                camera_files=camera_files,
                height=int(demo_config["zero_image_height"]),
                width=int(demo_config["zero_image_width"]),
            )
            step_started = time.monotonic()
            actions = client.infer(obs)
            steps.append(
                {
                    "step": index,
                    "frame_indices": [],
                    "actions": _jsonable(actions),
                    "duration_seconds": round(time.monotonic() - step_started, 3),
                }
            )
    else:
        if debug_video_dir is None:
            raise ValueError("debug_video_dir must be provided when use_zero_images is False.")
        # Load all frames from the specified debug videos for each camera.
        camera_frames = load_camera_frames(debug_video_dir, camera_files=camera_files)
        # Determine the minimum total frames across all cameras to ensure no out-of-bounds access.
        total_frames = min(int(value.shape[0]) for value in camera_frames.values())
        # Build the schedule of frame indices for each inference step.
        frame_schedule = build_frame_schedule(
            total_frames,
            chunk_count,
            relative_offsets=demo_config["relative_offsets"],
            action_horizon=int(demo_config["action_horizon"]),
        )
        # Iterate through the frame schedule to perform inference.
        # The first step uses only frame 0 (history-less observation), subsequent steps follow the schedule.
        for step_index, frame_indices in enumerate(([0], *frame_schedule)):
            # Construct observation from selected video frames.
            obs = _make_obs_from_video(camera_frames, frame_indices, prompt, session)
            step_started = time.monotonic()
            actions = client.infer(obs)
            steps.append(
                {
                    "step": step_index,
                    "frame_indices": list(frame_indices),
                    "actions": _jsonable(actions),
                    "duration_seconds": round(time.monotonic() - step_started, 3),
                }
            )

    # Send a reset command to the server after the demo.
    reset_response = client.reset()

    # Prepare and save the output artifact.
    target = Path(output_path or "dreamzero_action_trace.json").expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "worldfoundry-worldfoundry-dreamzero-official-client",
        "status": "success",
        "model_id": model_id,
        "backend": "worldfoundry.dreamzero.official_roboarena_websocket_client",
        "backend_quality": "official_server_client_demo",
        "artifact_kind": artifact_kind,
        "server": {"host": host, "port": port, "metadata": _jsonable(metadata)},
        "prompt": prompt,
        "session_id": session,
        "debug_video_dir": None if debug_video_dir is None else str(debug_video_dir),
        "use_zero_images": use_zero_images,
        "frame_schedule": frame_schedule,
        "steps": steps,
        "reset_response": _jsonable(reset_response),
        "extra": _jsonable(dict(extra or {})),
        "duration_seconds": round(time.monotonic() - started, 3),
    }
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    # Calculate SHA256 digest of the generated artifact.
    digest = file_sha256(target)
    return {
        "status": "success",
        "model_id": model_id,
        "artifact_kind": artifact_kind,
        "artifact_path": str(target),
        "artifact_sha256": digest,
        "runtime": payload["backend"],
        "backend_quality": payload["backend_quality"],
        "duration_seconds": payload["duration_seconds"],
    }
