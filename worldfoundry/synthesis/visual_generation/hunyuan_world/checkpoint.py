"""Utility functions for loading model checkpoints, specifically for Hunyuan World models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def load_hunyuan_world_state_dict(
    args: Any,
    model: Any,
    logger: Any,
    pretrained_model_path: str | Path | None = None,
) -> Any:
    """Loads a Hunyuan World model's state dictionary from a checkpoint file.

    This function handles determining the correct checkpoint path, optionally modifying it
    based on context block usage, loading the state dictionary, and then applying it
    to the provided model instance.

    Args:
        args (Any): An object containing configuration arguments, potentially including
                    'model_base' (default checkpoint path), 'use_context_block'
                    (whether to load context-aware states), and 'load_key'
                    (the key under which the state dict is stored in the checkpoint).
        model (Any): The model instance (e.g., a `torch.nn.Module`) to load the
                     state dictionary into.
        logger (Any): A logger object to report loading progress and information.
                      Can be None if no logging is desired.
        pretrained_model_path (str | Path | None, optional): Explicit path to the
                                                             pretrained model checkpoint.
                                                             If provided, it overrides
                                                             `args.model_base`.
                                                             Defaults to None.

    Returns:
        Any: The model instance with the loaded state dictionary.

    Raises:
        ValueError: If neither `pretrained_model_path` nor `args.model_base` is provided,
                    or if the determined checkpoint path does not exist.
        KeyError: If the `load_key` (e.g., "module") is not found in the loaded
                  state dictionary from the checkpoint.
    """
    # Determine the checkpoint path, prioritizing the explicit path over the one from args.
    checkpoint_path = pretrained_model_path or getattr(args, "model_base", None)
    if checkpoint_path is None:
        raise ValueError("pretrained_model_path or args.model_base is required")

    checkpoint_path = Path(checkpoint_path)
    # If context block is used, modify the checkpoint filename to load the context-aware version.
    if getattr(args, "use_context_block", False):
        checkpoint_path = Path(str(checkpoint_path).replace("_model_states.pt", "_model_states_context.pt"))

    if not checkpoint_path.exists():
        raise ValueError(f"model_path not exists: {checkpoint_path}")

    if logger is not None:
        logger.info(f"Loading torch model {checkpoint_path}...")

    # Load the state dictionary from the checkpoint file.
    # The map_location lambda ensures tensors are loaded onto the CPU,
    # preventing device mismatches if the original checkpoint was saved on a GPU.
    state_dict = torch.load(
        checkpoint_path,
        map_location=lambda storage, loc: storage,
        weights_only=True,
    )

    # Identify the key under which the actual model state dictionary is stored within the checkpoint.
    # This handles cases where the checkpoint might contain other metadata or be wrapped.
    load_key = getattr(args, "load_key", "module")
    if load_key not in state_dict:
        raise KeyError(
            f"Missing key: `{load_key}` in the checkpoint: {checkpoint_path}. "
            f"The keys in the checkpoint are: {list(state_dict.keys())}."
        )

    # Load the extracted state dictionary into the model, ensuring all keys match strictly.
    model.load_state_dict(state_dict[load_key], strict=True)
    return model
