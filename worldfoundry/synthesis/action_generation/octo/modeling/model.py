import json
import logging
from enum import Enum
from functools import partial
from pathlib import Path

import flax
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint
from flax import struct
from jax.typing import ArrayLike

from ..preprocessing.text import TextProcessor
from ..spec import ModuleSpec
from ..types import Config, Data, Params, PRNGKey, Sequence
from .action_heads import ActionHead
from .module import OctoModule


class NormalizationType(str, Enum):
    """Action normalization formats stored in released Octo checkpoints."""

    NORMAL = "normal"
    BOUNDS = "bounds"


@struct.dataclass
class OctoModel:
    """Recommended way of interacting with Octo models.

    Usage for inference:

        >>> model = OctoModel.load_pretrained(checkpoint_dir)
        >>> tasks = model.create_tasks(texts=["go to the red room"])
        >>> # or tasks = model.create_tasks(goals={"image_primary": goal_images})
        >>> actions = model.sample_actions(observations, tasks, rng=jax.random.PRNGKey(0))
        >>> # Note: these are normalized actions (processed to mean 0 and std 1). To get correct actions
        >>> # for a particular embodiment, you must additionally specify unnormalization statistics.
        >>> # For example, to get actions for one of Octo's pretraining datasets:
        >>> actions = model.sample_actions(observations, tasks, rng=jax.random.PRNGKey(0),
        >>>     unnormalization_statistics=model.dataset_statistics["DATASET_NAME_HERE"]["action"]
        >>> )

    WorldFoundry packages the OCTO inference runtime only. Build or finetune checkpoints
    in the upstream project, then load them here with ``OctoModel.load_pretrained``.

    """

    module: OctoModule = struct.field(pytree_node=False)
    text_processor: TextProcessor = struct.field(pytree_node=False)
    config: Config = struct.field(pytree_node=False)
    params: Params
    example_batch: Data
    dataset_statistics: Data | None

    def create_tasks(self, goals: Data | None = None, texts: Sequence[str] | None = None):
        """Creates tasks dict from goals and texts.

        Args:
            goals: if not None, dict of arrays with shape (batch_size, *)
            texts: if not None, list of texts of length batch_size

        Omit images to run the language-conditioned model, and omit texts to run the
        goal-conditioned model.
        """
        assert goals is not None or texts is not None
        tasks = {"pad_mask_dict": {}}
        if goals is not None:
            tasks.update(goals)
            tasks["pad_mask_dict"].update({k: np.ones(v.shape[:1], dtype=bool) for k, v in goals.items()})
        else:
            batch_size = len(texts)
            tasks.update(
                {
                    k: np.zeros((batch_size, *v.shape[1:]), dtype=v.dtype)
                    for k, v in self.example_batch["task"].items()
                    if k not in ("pad_mask_dict", "language_instruction")
                }
            )
            tasks["pad_mask_dict"].update({k: np.zeros(batch_size, dtype=bool) for k in tasks if k != "pad_mask_dict"})

        if texts is not None:
            assert self.text_processor is not None
            tasks["language_instruction"] = texts
            tasks["pad_mask_dict"]["language_instruction"] = np.ones(len(texts), dtype=bool)
        else:
            batch_size = jax.tree_util.tree_leaves(goals)[0].shape[0]
            tasks["language_instruction"] = [""] * batch_size
            tasks["pad_mask_dict"]["language_instruction"] = np.zeros(batch_size, dtype=bool)

        if self.text_processor is not None:
            tasks["language_instruction"] = self.text_processor.encode(tasks["language_instruction"])
        else:
            del tasks["language_instruction"]

        _verify_shapes(tasks, "tasks", self.example_batch["task"], starting_dim=1)
        return tasks

    @jax.jit
    def run_transformer(
        self,
        observations: Data,
        tasks: Data,
        timestep_pad_mask: ArrayLike,
    ):
        """Runs the transformer, but does shape checking on the inputs.

        Args:
            observations: dictionary of arrays of shape (batch_size, window_size, *shape).
                Shape must be consistent with self.example_batch["observation"]
            tasks: dict of tasks of shape (batch_size, *shape)
                Shape must be consistent with self.example_batch["task"]
            timestep_pad_mask: (batch_size, window_size) Boolean mask that is False when the timestep corresponds to padding
        """
        _verify_shapes(
            observations,
            "observations",
            self.example_batch["observation"],
            starting_dim=2,
        )
        _verify_shapes(tasks, "tasks", self.example_batch["task"], starting_dim=1)

        return self.module.apply(
            {"params": self.params},
            observations,
            tasks,
            timestep_pad_mask,
            train=False,
            method="octo_transformer",
        )

    @partial(
        jax.jit,
        static_argnames=("normalization_type", "sample_shape", "argmax"),
    )
    def sample_actions(
        self,
        observations: Data,
        tasks: Data,
        unnormalization_statistics: Data | None = None,
        normalization_type: NormalizationType = NormalizationType.NORMAL,
        timestep_pad_mask: ArrayLike | None = None,
        argmax: bool = False,
        sample_shape: tuple[int, ...] = (),
        rng: PRNGKey | None = None,
        temperature: float = 1.0,
    ):
        """Samples actions from the model. See `action_heads.py` for more info.

        Args:
            observations: dictionary of arrays of shape (batch_size, window_size, *)
            tasks: dict of tasks of shape (batch_size, *)
            unnormalization_statistics: dict of statistics for unnormalizing actions (must contain "mean",
                "std", and optionally "mask")
            normalization_type: type of normalization applied to the actions
            timestep_pad_mask: (batch_size, window_size) Boolean mask that is False when the timestep corresponds to padding
            ...see `action_heads.py` for the rest of the kwargs.
        Returns:
            actions: (*sample_shape, batch_size, action_horizon, action_dim)
        """
        if timestep_pad_mask is None:
            timestep_pad_mask = observations["timestep_pad_mask"]

        transformer_outputs = self.run_transformer(observations, tasks, timestep_pad_mask)
        action_head: ActionHead = self.module.bind({"params": self.params}).heads["action"]
        action = action_head.predict_action(
            transformer_outputs,
            train=False,
            argmax=argmax,
            sample_shape=sample_shape,
            rng=rng,
            temperature=temperature,
            embodiment_action_dim=len(unnormalization_statistics["mean"])
            if unnormalization_statistics is not None
            else None,
        )
        if unnormalization_statistics is not None:
            if normalization_type == NormalizationType.NORMAL:
                mask = unnormalization_statistics.get(
                    "mask",
                    jnp.ones_like(unnormalization_statistics["mean"], dtype=bool),
                )
                action = action[..., : len(mask)]
                action = jnp.where(
                    mask,
                    (action * unnormalization_statistics["std"]) + unnormalization_statistics["mean"],
                    action,
                )
            elif normalization_type == NormalizationType.BOUNDS:
                mask = unnormalization_statistics.get(
                    "mask", jnp.ones_like(unnormalization_statistics["p01"], dtype=bool)
                )
                action = action[..., : len(mask)]
                action = jnp.where(
                    mask,
                    (action + 1) * (unnormalization_statistics["p99"] - unnormalization_statistics["p01"]) / 2
                    + unnormalization_statistics["p01"],
                    action,
                )
            else:
                raise ValueError(f"Unknown normalization type: {normalization_type}")
        return action

    @classmethod
    def load_pretrained(
        cls,
        checkpoint_path: str,
        step: int | None = None,
    ) -> "OctoModel":
        """Load an exported Octo checkpoint from a local directory.

        Args:
            checkpoint_path (str): A path to either a directory of checkpoints or a single checkpoint.
            step (int, optional): If multiple checkpoints are present, which one to load. Defaults to the latest.
        """
        checkpoint = Path(checkpoint_path).expanduser().resolve()
        if not checkpoint.is_dir():
            raise FileNotFoundError(f"Octo checkpoint does not exist: {checkpoint}")
        with (checkpoint / "config.json").open("r", encoding="utf-8") as handle:
            config = json.load(handle)

        # shim to support old configs
        if "pred_horizon" in config["model"]["heads"]["action"]["kwargs"]:
            config["model"]["heads"]["action"]["kwargs"]["action_horizon"] = config["model"]["heads"]["action"][
                "kwargs"
            ].pop("pred_horizon")

        with (checkpoint / "example_batch.msgpack").open("rb") as handle:
            example_batch = flax.serialization.msgpack_restore(handle.read())
        # shim for migrating from "tasks" to "task"
        if "tasks" in example_batch:
            example_batch["task"] = example_batch.pop("tasks")

        logging.debug(
            "Model was trained with observations: %s",
            flax.core.pretty_repr(jax.tree_util.tree_map(jnp.shape, example_batch["observation"])),
        )
        logging.debug(
            "Model was trained with tasks: %s",
            flax.core.pretty_repr(jax.tree_util.tree_map(jnp.shape, example_batch["task"])),
        )

        with (checkpoint / "dataset_statistics.json").open("r", encoding="utf-8") as handle:
            dataset_statistics = json.load(handle)
        dataset_statistics = jax.tree_util.tree_map(
            np.array, dataset_statistics, is_leaf=lambda x: not isinstance(x, dict)
        )

        # create model def (an OctoModule)
        module = OctoModule.create(**config["model"])
        # infer params shape without actually doing any computation

        # shim for old checkpoints
        if "timestep_pad_mask" not in example_batch["observation"]:
            example_batch["observation"]["timestep_pad_mask"] = example_batch["observation"]["pad_mask"]

        init_args = (
            example_batch["observation"],
            example_batch["task"],
            example_batch["observation"]["timestep_pad_mask"],
        )
        params_shape = jax.eval_shape(partial(module.init, train=False), jax.random.PRNGKey(0), *init_args)["params"]
        # restore params, checking to make sure the shape matches
        checkpointer = orbax.checkpoint.CheckpointManager(str(checkpoint), orbax.checkpoint.PyTreeCheckpointer())
        step = step if step is not None else checkpointer.latest_step()
        params = checkpointer.restore(step, params_shape)

        if config["text_processor"] is not None:
            text_processor = ModuleSpec.instantiate(config["text_processor"])()
        else:
            text_processor = None

        return cls(
            module=module,
            params=params,
            text_processor=text_processor,
            example_batch=example_batch,
            config=config,
            dataset_statistics=dataset_statistics,
        )


def _verify_shapes(
    pytree,
    name: str,
    example_pytree,
    starting_dim: int = 0,
    strict: bool = False,
    raise_error: bool = True,
    silent: bool = False,
):
    weak_fail, fail = False, False
    pytree_flat = flax.traverse_util.flatten_dict(pytree)
    example_pytree_flat = flax.traverse_util.flatten_dict(example_pytree)

    # Check that all elements are present
    if set(pytree_flat.keys()) != set(example_pytree_flat.keys()):
        if not silent:
            extra = set(pytree_flat.keys()) - set(example_pytree_flat.keys())
            if extra:
                logging.warning(
                    "'%s' contains extra items compared to example_batch: %s",
                    name,
                    {"/".join(x) for x in extra},
                )
            missing = set(example_pytree_flat.keys()) - set(pytree_flat.keys())
            if missing:
                logging.warning(
                    "'%s' is missing items compared to example_batch: %s",
                    name,
                    {"/".join(x) for x in missing},
                )
        weak_fail = True

    mismatched_keys = {
        k: f"{pytree_flat[k].shape} != {example_pytree_flat[k].shape}"
        for k in pytree_flat
        if k in example_pytree_flat
        and pytree_flat[k].shape[starting_dim:] != example_pytree_flat[k].shape[starting_dim:]
    }
    if mismatched_keys:
        if not silent:
            logging.error(
                "'%s' contains mismatched shapes compared to example_batch: %s",
                name,
                flax.core.pretty_repr({"/".join(k): v for k, v in mismatched_keys.items()}),
            )
        fail = True

    if raise_error and (fail or (weak_fail and strict)):
        raise AssertionError(f"{name} does not match example batch.")

    return weak_fail or fail


__all__ = ["NormalizationType", "OctoModel"]
