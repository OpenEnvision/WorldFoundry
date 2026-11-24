"""In-tree MolmoBot checkpoint loading, preprocessing, and action inference."""

from __future__ import annotations

import logging
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

log = logging.getLogger(__name__)


class SynthManipMolmoInferenceWrapper:
    """
    Inference agent for MolmoAct models trained on SynthManip data.

    Loads a checkpoint, handles preprocessing, and provides a simple
    get_action_chunk API for online evaluation.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
        num_flow_steps: int | None = None,
        max_seq_len: int | None = None,
        norm_repo_id: str = "synthmanip",
        use_bfloat16: bool = True,
        compile_model: bool = False,
        states_mode: str | None = None,
    ):
        """
        Initialize the agent with a trained checkpoint.

        Args:
            checkpoint_path: Path to the model checkpoint directory.
            device: Device to run inference on ("cuda" or "cpu").
            num_flow_steps: Number of flow-matching integration steps.
                           Uses checkpoint default if None.
            max_seq_len: Maximum sequence length. Uses checkpoint default if None.
            norm_repo_id: Repository ID for normalization stats lookup.
        """
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device {device!r} was requested, but CUDA is unavailable.")
        self.num_flow_steps = num_flow_steps
        self.norm_repo_id = norm_repo_id
        self.use_bfloat16 = use_bfloat16
        self.compile_model = compile_model
        if not use_bfloat16:
            self.compute_dtype = torch.float32
        elif self.device.type == "cuda" and torch.cuda.is_bf16_supported():
            self.compute_dtype = torch.bfloat16
        elif self.device.type == "cuda":
            self.compute_dtype = torch.float16
            log.warning("BF16 is unavailable on %s; using FP16 inference.", self.device)
        elif self.device.type == "cpu":
            self.compute_dtype = torch.bfloat16
        else:
            self.compute_dtype = torch.float16

        self.states_mode = states_mode

        # Load model and config
        self._load_checkpoint()

        # Override max_seq_len if provided
        if max_seq_len is not None:
            self.max_seq_len = max_seq_len

        # Build preprocessor and collator
        self._build_processors()

        # Build normalization processors
        self._build_normalizers()

    def _load_checkpoint(self) -> None:
        """Load model and config from checkpoint."""
        from .model_config import BaseModelConfig
        from .utils import resource_path

        config_path = resource_path(self.checkpoint_path, "config.yaml")
        log.info("Loading MolmoBot config from %s", config_path)

        # Load only the model config directly, bypassing TrainConfig to avoid
        # importing heavy eval dependencies (scipy, torchmetrics, editdistance, etc.)
        # The key="model" extracts just the model section from the full TrainConfig YAML.
        self.model_config = BaseModelConfig.load(config_path, key="model")

        # Some exported checkpoints bundle the exact tokenizer assets. Prefer
        # those over a network lookup, but never substitute a tokenizer from a
        # different Molmo checkpoint: visual token IDs are tied to embedding
        # rows in model.pt.
        checkpoint_dir = Path(self.checkpoint_path).expanduser()
        if checkpoint_dir.is_dir() and any(
            (checkpoint_dir / filename).is_file()
            for filename in ("tokenizer.json", "tokenizer_config.json", "tokenizer.model")
        ):
            self.model_config.llm.tokenizer.tokenizer_dir = str(checkpoint_dir)

        # Extract useful config values
        self.max_seq_len = self.model_config.llm.max_sequence_length
        self.action_horizon = getattr(self.model_config, "action_horizon", 16)
        self.action_dim = getattr(self.model_config, "action_dim", 7)
        self.n_obs_steps = getattr(self.model_config, "n_obs_steps", 1)

        if self.num_flow_steps is None:
            self.num_flow_steps = getattr(self.model_config, "flow_matching_num_steps", 10)

        # Check if we need to override states_mode for eval configs
        if self.states_mode is not None:
            self.model_config.states_mode = self.states_mode
        selected_states_mode = self.model_config.states_mode

        log.info(
            f"Model config: action_horizon={self.action_horizon}, "
            f"action_dim={self.action_dim}, n_obs_steps={self.n_obs_steps}, "
            f"flow_steps={self.num_flow_steps}, States mode: {selected_states_mode}"
        )

        # Build model
        log.info("Building MolmoBot model on the meta device")
        with torch.device("meta"):
            self.model = self.model_config.build_model()
        model_path = resource_path(self.checkpoint_path, "model.pt")
        state_dict = torch.load(
            model_path,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        self.model.load_state_dict(state_dict, strict=True, assign=True)
        del state_dict
        self.model.to(self.device, dtype=self.compute_dtype)
        self.model.eval()
        log.info("MolmoBot loaded on %s with %s", self.device, self.compute_dtype)

        if self.compile_model:
            log.info("Compiling MolmoBot action inference")
            self.model.generate_actions = torch.compile(self.model.generate_actions, mode="max-autotune")

    def _build_processors(self) -> None:
        """Build preprocessor and collator from model config."""
        self.preprocessor = self.model_config.build_preprocessor(
            for_inference=True,
            is_training=False,
            max_seq_len=self.max_seq_len,
        )

        self.collator = self.model_config.build_collator(
            self.preprocessor.get_output_shapes(),
            pad_mode=None,
            include_metadata=True,
        )

    def _build_normalizers(self) -> None:
        """Build state normalizer and action unnormalizer from config."""
        self.state_preprocessor = None
        self.action_postprocessor = None

        robot_pre = getattr(self.model_config, "robot_preprocessor", None)
        if robot_pre is not None:
            self.state_preprocessor = robot_pre.build_preprocessor()
            log.info("Built state preprocessor from checkpoint config")
        else:
            log.warning("No robot_preprocessor in config - states will not be normalized")

        robot_post = getattr(self.model_config, "robot_postprocessor", None)
        if robot_post is not None:
            self.action_postprocessor = robot_post.build_postprocessor()
            log.info("Built action postprocessor from checkpoint config")
            log.info(f"  action_key: {robot_post.action_key}")
            log.info(f"  action_norm_mode: {robot_post.action_norm_mode}")
            log.info(f"  repos with stats: {list(robot_post.stats_by_repo.keys())}")
            for repo_id, stats in robot_post.stats_by_repo.items():
                for key, feature_stats in stats.items():
                    if isinstance(feature_stats, dict):
                        for stat_name, stat_val in feature_stats.items():
                            if hasattr(stat_val, "__len__"):
                                log.info(f"    {repo_id}/{key}/{stat_name}: len={len(stat_val)}")
        else:
            log.warning("No robot_postprocessor in config - actions will not be unnormalized!")

    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        """Normalize state using checkpoint's normalization stats."""
        if self.state_preprocessor is None:
            return state
        try:
            return self.state_preprocessor.normalize_state(state, self.norm_repo_id)
        except Exception as exc:
            raise RuntimeError(f"State normalization failed for norm_repo_id={self.norm_repo_id!r}.") from exc

    def _unnormalize_action(self, actions: np.ndarray) -> np.ndarray:
        """Unnormalize actions using checkpoint's normalization stats."""
        if self.action_postprocessor is None:
            log.debug(f"Skipping unnormalization (no postprocessor), action shape: {actions.shape}")
            return actions
        try:
            unnormed = self.action_postprocessor.unnormalize_action(actions, self.norm_repo_id)
            log.debug(
                f"Unnormalized actions: shape={actions.shape}, "
                f"input_range=[{actions.min():.3f}, {actions.max():.3f}], "
                f"output_range=[{unnormed.min():.3f}, {unnormed.max():.3f}]"
            )
            return unnormed
        except Exception as exc:
            raise RuntimeError(f"Action unnormalization failed for norm_repo_id={self.norm_repo_id!r}.") from exc

    def _prepare_images(self, images: list[np.ndarray] | list[str] | np.ndarray | str) -> list[np.ndarray]:
        """Convert various image formats to list of numpy arrays."""
        if isinstance(images, (str, Path)):
            images = [images]
        elif isinstance(images, np.ndarray):
            if images.ndim == 3:
                images = [images]
            elif images.ndim == 4:
                images = [images[i] for i in range(images.shape[0])]

        result = []
        for img in images:
            if isinstance(img, (str, Path)):
                pil_img = Image.open(img).convert("RGB")
                result.append(np.array(pil_img))
            elif isinstance(img, np.ndarray):
                if img.ndim != 3 or img.shape[-1] not in (3, 4):
                    raise ValueError(f"Expected an HWC RGB/RGBA image, got shape {img.shape}.")
                if img.shape[-1] == 4:
                    img = img[..., :3]
                if not np.all(np.isfinite(img)):
                    raise ValueError("MolmoBot images must contain only finite values.")
                if img.dtype != np.uint8:
                    image_min = float(img.min())
                    image_max = float(img.max())
                    if image_min >= 0.0 and image_max <= 1.0:
                        img = np.rint(img * 255.0)
                    elif image_min >= -1.0 and image_max <= 1.0:
                        img = np.rint((img + 1.0) * 127.5)
                    img = np.clip(img, 0, 255).astype(np.uint8)
                result.append(img)
            else:
                raise ValueError(f"Unsupported image type: {type(img)}")

        if not result:
            raise ValueError("MolmoBot requires at least one camera image.")
        return result

    def _prepare_state(self, state: np.ndarray | None) -> np.ndarray | None:
        """Prepare and normalize state for model input."""
        if state is None:
            return None

        state = np.asarray(state, dtype=np.float32)

        if state.ndim == 1:
            state = state.reshape(1, -1)
        elif state.ndim != 2:
            raise ValueError(f"Expected state shape [D] or [T, D], got {state.shape}.")
        if not np.all(np.isfinite(state)):
            raise ValueError("MolmoBot state must contain only finite values.")
        return self._normalize_state(state)

    def get_action_chunk(
        self,
        images: list[np.ndarray] | list[str] | np.ndarray,
        task_description: str = "",
        state: np.ndarray | None = None,
        generator: torch.Generator | None = None,
    ) -> np.ndarray:
        """
        Generate an action chunk from observations.

        Args:
            images: Camera observations. Can be:
                - List of numpy arrays (H, W, 3) RGB uint8
                - List of file paths to images
                - Single numpy array (H, W, 3) or (N, H, W, 3)
            task_description: Text prompt / task instruction.
            state: Optional robot state array. Shape (state_dim,) or (n_obs_steps, state_dim).
            generator: Optional torch Generator for reproducible sampling.

        Returns:
            np.ndarray: Unnormalized action chunk of shape (action_horizon, action_dim).
        """
        from .torch_utils import move_to_device

        # Prepare inputs
        images = self._prepare_images(images)
        state = self._prepare_state(state)

        # Build example dict for preprocessor
        example = {
            "style": "demo",
            "question": task_description,
            "image": images if len(images) > 1 else images[0],
        }
        if state is not None:
            example["state"] = state

        # Preprocess and collate
        processed = self.preprocessor(example)
        batch = self.collator([processed])
        batch = move_to_device(batch, self.device)

        # Extract model inputs
        model_inputs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch.get("attention_mask"),
            "position_ids": batch.get("position_ids"),
            "response_mask": batch.get("response_mask"),
            "images": batch.get("images"),
            "image_masks": batch.get("image_masks"),
            "token_pooling": batch.get("token_pooling"),
            "low_res_token_pooling": batch.get("low_res_token_pooling"),
            "states": batch.get("states"),
        }
        model_inputs = {k: v for k, v in model_inputs.items() if v is not None}

        autocast = (
            torch.autocast(device_type=self.device.type, dtype=self.compute_dtype)
            if self.compute_dtype != torch.float32 and self.device.type in {"cpu", "cuda"}
            else nullcontext()
        )
        with torch.inference_mode(), autocast:
            actions = self.model.generate_actions(
                **model_inputs,
                num_steps=self.num_flow_steps,
                generator=generator,
            )

        # Convert to numpy and unnormalize
        actions_np = actions.detach().cpu().numpy()
        actions_np = self._unnormalize_action(actions_np)

        # Return first batch element
        return actions_np[0]

    @property
    def config(self) -> dict[str, Any]:
        """Return agent configuration as a dictionary."""
        return {
            "checkpoint_path": self.checkpoint_path,
            "device": str(self.device),
            "action_horizon": self.action_horizon,
            "action_dim": self.action_dim,
            "n_obs_steps": self.n_obs_steps,
            "num_flow_steps": self.num_flow_steps,
            "max_seq_len": self.max_seq_len,
            "norm_repo_id": self.norm_repo_id,
            "compute_dtype": str(self.compute_dtype),
        }


__all__ = ["SynthManipMolmoInferenceWrapper"]
