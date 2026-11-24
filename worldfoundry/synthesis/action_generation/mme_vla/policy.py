import time
from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ..openpi import nnx_utils
from ..openpi import transforms as _transforms
from .memory import MemoryBuffer, MemoryBufferRecurrent
from .modeling.model import HistoryPi0
from .modeling.observation import HistAugObservation


class MMEVLAPolicy:
    def __init__(
        self,
        model: HistoryPi0,
        *,
        seed: int = 42,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        norm_stats: dict[str, _transforms.NormStats] | None = None,
        use_quantiles: bool = False,
    ):
        self._model = model
        self._seed = seed
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}

        self._sample_actions = nnx_utils.module_jit(model.sample_actions)
        self._vision_encode = nnx_utils.module_jit(model.vision_encode)

        self.config = model.history_config
        self.mem_buffer = None

        if norm_stats is None or "state" not in norm_stats:
            raise ValueError("MME-VLA checkpoint is missing state normalization statistics")
        self.state_norm_stats = norm_stats["state"]
        self.use_quantiles = use_quantiles

        self.reset()

    def _prepare_mem_buffer(self):
        if self.config is None or self.config.representation_type == "symbolic":
            self.mem_buffer = None
        elif self.config.representation_type == "recurrent":
            self.mem_buffer = MemoryBufferRecurrent(
                num_views=self.config.num_views,
                img_emb_dim=self.config.memory_feature.img.input_dim,
                pos_emb_dim=self.config.memory_feature.pos.input_dim,
                state_emb_dim=self.config.memory_feature.state.input_dim,
                input_obs_horizon=self.config.streaming_obs_horizon,
                max_recur_steps=self.config.recurrent_memory.max_recur_steps,
                max_video_steps=self.config.recurrent_memory.max_pretraj_steps,
                vision_enc_fn=self._vision_encode,
            )
        else:
            self.mem_buffer = MemoryBuffer(
                num_views=self.config.num_views,
                img_emb_dim=self.config.memory_feature.img.input_dim,
                pos_emb_dim=self.config.memory_feature.pos.input_dim,
                state_emb_dim=self.config.memory_feature.state.input_dim,
                compute_token_drop_score=self.config.perceptual_memory.type == "token_dropping",
                token_drop_stride=self.config.streaming_obs_horizon // 2,
                vision_enc_fn=self._vision_encode,
            )

    def infer(self, obs: dict) -> dict:
        if self.config is not None and self.config.representation_type != "symbolic":  # noqa: SIM102 - Statement form kept for readability.
            if self.mem_buffer is None or not self.mem_buffer.has_history:
                raise ValueError("MME-VLA requires at least one buffered history observation")

        inputs = self._prepare_history(dict(obs))
        inputs = self._input_transform(inputs)
        observation = HistAugObservation.from_dict(jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs))
        self._rng, sample_rng = jax.random.split(self._rng)

        start_time = time.monotonic()
        actions = self._sample_actions(sample_rng, observation, **self._sample_kwargs)
        jax.block_until_ready(actions)
        model_time = time.monotonic() - start_time
        outputs = {"state": observation.state, "actions": actions}
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        outputs = self._output_transform(outputs)
        outputs["infer_time_ms"] = model_time * 1000

        return outputs

    def reset(self) -> None:
        self.mem_buffer = None
        self._prepare_mem_buffer()
        self.step_idx = -1
        self.exec_start_idx = 0
        self._rng = jax.random.key(self._seed)

    def add_buffer(self, obs: dict) -> None:
        if self.mem_buffer is None:
            return
        images = obs["images"]
        states = obs["state"]
        if obs.get("exec_start_idx", 0) > 0:  # has video
            self.exec_start_idx = obs["exec_start_idx"]

        step_idx_list = list(range(self.step_idx + 1, self.step_idx + len(images) + 1))
        self.mem_buffer.add_buffer(images, states, step_idx_list)
        self.step_idx += len(images)

    def _normalize_state(self, state):
        if self.use_quantiles:
            return (state - self.state_norm_stats.q01) / (
                self.state_norm_stats.q99 - self.state_norm_stats.q01 + 1e-6
            ) * 2.0 - 1.0
        else:
            return (state - self.state_norm_stats.mean) / (self.state_norm_stats.std + 1e-6)

    def _prepare_history(self, inputs: dict) -> dict:
        if self.config is None or self.config.representation_type == "symbolic":
            return inputs

        if self.config.representation_type == "recurrent":
            history_feats_gather_fn = self.mem_buffer.default_history_feats_gather_fn
            recur_image_emb, recur_pos_emb, recur_state_emb, recur_mask = self.mem_buffer.prepare_token_recurrent(
                self.step_idx, self.exec_start_idx, history_feats_gather_fn
            )
            inputs["recur_image_emb"] = recur_image_emb
            inputs["recur_pos_emb"] = recur_pos_emb
            inputs["recur_state_emb"] = self._normalize_state(recur_state_emb)
            inputs["recur_mask"] = recur_mask
        elif self.config.representation_type == "perceptual":
            history_feats_gather_fn = self.mem_buffer.default_history_feats_gather_fn
            token_budget = self.config.budget

            if self.config.perceptual_memory.type == "token_dropping":
                static_image_emb, static_pos_emb, static_state_emb, static_mask = (
                    self.mem_buffer.prepare_token_dropping(self.step_idx, token_budget, history_feats_gather_fn)
                )
            else:
                token_per_image = self.config.token_per_image
                static_image_emb, static_pos_emb, static_state_emb, static_mask = (
                    self.mem_buffer.prepare_frame_sampling(
                        self.step_idx, token_budget, token_per_image, history_feats_gather_fn
                    )
                )

            inputs["static_image_emb"] = static_image_emb
            inputs["static_pos_emb"] = static_pos_emb
            inputs["static_state_emb"] = self._normalize_state(static_state_emb)
            inputs["static_mask"] = static_mask
        else:
            raise ValueError(f"Not supported representation type: {self.config.representation_type}")

        return inputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


# Backward-compatible class name used by released MME-VLA examples.
MME_VLA_Policy = MMEVLAPolicy
