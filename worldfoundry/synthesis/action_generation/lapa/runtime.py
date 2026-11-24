"""Runtime components for LAPA (Latent Action Prediction Agent) within WorldFoundry.

This module provides classes and functions to configure, load, and run the LAPA model
for generating latent action tokens based on an instruction and an image. It handles
asset path resolution, JAX/Tux compatibility fixes, and model inference.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from worldfoundry.core import jsonable
from worldfoundry.core.io.paths import project_root, resolve_worldfoundry_path


def _install_tux_jax_compat() -> None:
    """Provide JAX symbols expected by tux 0.0.2 on newer JAX wheels.

    This function attempts to import JAX components and, if successful,
    patches `inspect.getargspec` and `pjit.with_sharding_constraint`
    to ensure compatibility with older `tux` versions when running
    with newer JAX distributions. It also aliases `jax.Array` to
    `jax.numpy.DeviceArray` for compatibility.
    """
    try:
        import inspect

        import jax
        import jax.numpy as jnp
        import transformers
        from jax.experimental import pjit
    except Exception:
        # If JAX or its experimental components are not available, skip compatibility patches.
        return

    # Patch inspect.getargspec if it's missing (removed in Python 3.11).
    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = inspect.getfullargspec
    # Patch pjit.with_sharding_constraint if it's missing (moved in newer JAX versions).
    if not hasattr(pjit, "with_sharding_constraint") and hasattr(jax.lax, "with_sharding_constraint"):
        pjit.with_sharding_constraint = jax.lax.with_sharding_constraint
    # Patch jnp.DeviceArray if it's missing (renamed to jax.Array in newer JAX).
    if not hasattr(jnp, "DeviceArray") and hasattr(jax, "Array"):
        jnp.DeviceArray = jax.Array
    if not hasattr(transformers, "FlaxLogitsWarper"):

        class FlaxLogitsWarper:
            """Compatibility base removed from recent Transformers releases."""

        transformers.FlaxLogitsWarper = FlaxLogitsWarper


def _install_tux_checkpoint_compat() -> None:
    """Allow LAPA's large msgpack params file to stream with current tux.

    This function attempts to import `tux.StreamingCheckpointer` and patches
    its `load_checkpoint` method. The patch modifies the `msgpack.Unpacker`
    read and buffer sizes to accommodate very large checkpoint files,
    preventing out-of-memory issues during streaming loading.
    """
    try:
        import tux.checkpoint as tux_checkpoint
        from tux import StreamingCheckpointer
    except Exception:
        # If tux is not available, skip compatibility patches.
        return

    # Prevent re-patching if the buffer fix has already been applied.
    if getattr(StreamingCheckpointer.load_checkpoint, "_worldfoundry_large_buffer", False):
        return

    def load_checkpoint(path, target=None, shard_fns=None, remove_dict_prefix=None):
        # Flatten shard functions dictionary for easier lookup if provided.
        if shard_fns is not None:
            shard_fns = tux_checkpoint.flatten_dict(tux_checkpoint.to_state_dict(shard_fns))
        # Convert prefix to tuple for efficient matching.
        if remove_dict_prefix is not None:
            remove_dict_prefix = tuple(remove_dict_prefix)

        flattened_train_state = {}
        # Open the checkpoint file and initialize the msgpack unpacker with large buffer sizes.
        with tux_checkpoint.open_file(path) as fin:
            unpacker = tux_checkpoint.msgpack.Unpacker(
                fin,
                read_size=83886080,  # Larger read size for performance
                max_buffer_size=32 * 2**30,  # 32 GB max buffer to handle large tensors
            )
            for key, value in unpacker:
                key = tuple(key)
                # Remove specified prefix from keys if applicable.
                if remove_dict_prefix is not None:
                    if key[: len(remove_dict_prefix)] == remove_dict_prefix:
                        key = key[len(remove_dict_prefix) :]
                    else:
                        continue  # Skip keys not matching the prefix
                # Convert byte value to tensor.
                tensor = tux_checkpoint.from_bytes(None, value)
                # Apply sharding functions if provided.
                if shard_fns is not None:
                    tensor = shard_fns[key](tensor)
                flattened_train_state[key] = tensor

        # If a target state dict is provided, merge in any missing empty nodes from the target.
        if target is not None:
            flattened_target = tux_checkpoint.flatten_dict(tux_checkpoint.to_state_dict(target), keep_empty_nodes=True)
            for key, value in flattened_target.items():
                if key not in flattened_train_state and value == tux_checkpoint.empty_node:
                    flattened_train_state[key] = value

        # Unflatten the state dictionary back to its original structure.
        train_state = tux_checkpoint.unflatten_dict(flattened_train_state)
        # If no target, return the unflattened state directly.
        if target is None:
            return train_state
        # Otherwise, load the state into the target structure.
        return tux_checkpoint.from_state_dict(target, train_state)

    # Mark the patched function to prevent duplicate patching.
    load_checkpoint._worldfoundry_large_buffer = True
    # Replace the original load_checkpoint method with the patched version.
    StreamingCheckpointer.load_checkpoint = staticmethod(load_checkpoint)


@dataclass(frozen=True)
class LAPAAssetPaths:
    """Dataclass to hold resolved file paths for LAPA model assets."""

    checkpoint_dir: Path
    params_path: Path
    tokenizer_path: Path
    vqgan_path: Path


@dataclass(frozen=True)
class LAPARuntimeConfig:
    """Configuration for initializing the LAPA runtime model."""

    assets: LAPAAssetPaths
    dtype: str
    image_size: int
    mesh_dim: str
    seed: int
    tokens_per_delta: int


def select_lapa_assets(
    *,
    checkpoint_dir: str | Path | None,
    checkpoints: tuple[Mapping[str, Any], ...],
) -> LAPAAssetPaths:
    """Resolve required LAPA checkpoint files from caller options or profile.

    Args:
        checkpoint_dir: Explicit directory containing params, tokenizer.model, and vqgan.
        checkpoints: Runtime-profile checkpoint records used when no override exists.
            Typically a list of dictionaries from `worldfoundry.config.profile.runtime_config.json`.

    Returns:
        A `LAPAAssetPaths` object containing the absolute paths to the required files.

    Raises:
        ValueError: If `checkpoint_dir` is not provided and no checkpoint metadata is found.
        FileNotFoundError: If the specified checkpoint directory or any required files
                           (params, tokenizer.model, vqgan) do not exist.
    """
    candidate = checkpoint_dir
    # If no explicit directory, try to get it from the first available checkpoint in the profile.
    if candidate is None and checkpoints:
        candidate = checkpoints[0].get("local_dir")
    if candidate is None:
        raise ValueError("LAPA requires checkpoint_dir or profile checkpoint metadata.")

    # Resolve the root path, handling relative paths and making it absolute.
    root = resolve_worldfoundry_path(candidate)
    if not root.is_absolute():
        root = project_root() / root
    root = root.resolve()

    if not root.is_dir():
        raise FileNotFoundError(f"LAPA checkpoint directory not found: {root}")

    # Construct the full paths for all required assets.
    paths = LAPAAssetPaths(
        checkpoint_dir=root,
        params_path=root / "params",
        tokenizer_path=root / "tokenizer.model",
        vqgan_path=root / "vqgan",
    )

    # Check for missing required files within the asset directory.
    missing = [str(path) for path in (paths.params_path, paths.tokenizer_path, paths.vqgan_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"LAPA checkpoint directory is missing required files: {missing}")
    return paths


class LAPARuntime:
    """A lazy-loading runtime for the LAPA (Latent Action Prediction Agent) model.

    This class manages the lifecycle of the LAPA model, including its configuration,
    lazy loading of the underlying JAX model, and execution of inference tasks
    to generate latent action tokens. It incorporates necessary compatibility
    patches for `tux` and JAX.
    """

    def __init__(self, config: LAPARuntimeConfig) -> None:
        """Create a lazy in-tree LAPA latent-action runtime.

        Args:
            config: Checkpoint file paths and generation settings.
        """
        self.config = config
        self._model: Any | None = None

    def _load_model(self) -> Any:
        """Loads and initializes the LAPA model if it hasn't been loaded yet.

        This method performs necessary environment setup, applies compatibility patches,
        and instantiates the `LAPAInference` model. It ensures the model is only
        loaded once.

        Returns:
            The initialized LAPA inference model instance.
        """
        if self._model is not None:
            return self._model

        # Apply JAX compatibility patches for tux.
        _install_tux_jax_compat()
        # Import JAX distributed configuration and random seed setter.
        from tux import JaxDistributedConfig, set_random_seed

        # Apply tux checkpoint compatibility patches for large files.
        _install_tux_checkpoint_compat()
        # Import LAPA-specific model configurations and inference class.
        from .modeling.delta_llama import VideoLLaMAConfig
        from .modeling.inference import LAPAInference

        # Configure JAX distributed environment.
        jax_distributed = JaxDistributedConfig.get_default_config()
        # Configure the tokenizer, pointing to the provided vocabulary file.
        tokenizer = VideoLLaMAConfig.get_tokenizer_config()
        tokenizer.vocab_file = str(self.config.assets.tokenizer_path)
        JaxDistributedConfig.initialize(jax_distributed)
        # Set the random seed for reproducibility.
        set_random_seed(self.config.seed)

        # Initialize the LAPA inference model with detailed configuration.
        self._model = LAPAInference(
            image_size=self.config.image_size,
            tokens_per_delta=self.config.tokens_per_delta,
            vqgan_checkpoint=str(self.config.assets.vqgan_path),
            vocab_file=str(self.config.assets.tokenizer_path),
            multi_image=1,
            jax_distributed=jax_distributed,
            seed=self.config.seed,
            mesh_dim=self.config.mesh_dim,
            dtype=self.config.dtype,
            load_llama_config="7b",  # Specifies the base LLaMA config to load.
            # Extensive string-based configuration for updating LLaMA parameters.
            update_llama_config="dict(delta_vocab_size=8,sample_mode='text',theta=50000000,max_sequence_length=32768,scan_attention=False,scan_query_chunk_size=128,scan_key_chunk_size=128,scan_mlp=False,scan_mlp_chunk_size=8192,scan_layers=True)",
            # Specifies the path to the model parameters checkpoint.
            load_checkpoint=f"params::{self.config.assets.params_path}",
            tokenizer=tokenizer,
            llama=VideoLLaMAConfig.get_default_config(),
        )
        return self._model

    def predict_tokens(
        self,
        *,
        instruction: str,
        image: Any,
        output_path: str | Path,
        extra_metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run LAPA latent-action prediction and write an action-token artifact.

        Args:
            instruction: Natural-language task instruction.
            image: RGB image path, PIL image, or uint8 ndarray accepted by the in-tree runtime.
                   If a list/tuple, it expects a single image frame.
            output_path: JSON artifact path for latent action tokens. The parent directory
                         will be created if it doesn't exist.
            extra_metadata: Additional WorldFoundry metadata to include in the artifact.

        Returns:
            A dictionary containing status, model ID, artifact kind, artifact path,
            runtime identifier, and the generated latent action tokens.

        Raises:
            ValueError: If `image` is a list/tuple with more than one element, or
                        if the image data is not uint8 RGB.
        """
        import numpy as np
        from PIL import Image

        model = self._load_model()

        # Handle various input image types.
        if isinstance(image, (list, tuple)):
            if len(image) != 1:
                raise ValueError("LAPA latent inference expects a single image frame.")
            image = image[0]
        # Open image from path or use directly if it's already a PIL Image or ndarray.
        image_value = Image.open(image) if isinstance(image, (str, Path)) else image
        # Convert image to NumPy array for model input.
        image_array = np.asarray(image_value)
        # Validate image data type.
        if image_array.dtype != np.uint8:
            raise ValueError("LAPA image input must be uint8 RGB data.")

        # Perform inference to get latent action tokens.
        latent_action = model.inference(image_array, instruction)

        # Construct the artifact dictionary, making sure all values are JSON-safe.
        artifact = {
            "model_id": "lapa",
            "artifact_kind": "action_tokens",
            "runtime": "worldfoundry.lapa.in_tree_runtime.predict_tokens",
            "checkpoint_dir": str(self.config.assets.checkpoint_dir),
            "tokens_per_delta": self.config.tokens_per_delta,
            "latent_action_tokens": jsonable(latent_action),
            "metadata": jsonable(dict(extra_metadata or {})),
        }

        # Resolve the output path, create parent directories, and write the JSON artifact.
        target = Path(output_path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        # Return a summary of the prediction result.
        return {
            "status": "success",
            "model_id": "lapa",
            "artifact_kind": "action_tokens",
            "artifact_path": str(target),
            "runtime": artifact["runtime"],
            "latent_action_tokens": artifact["latent_action_tokens"],
        }
