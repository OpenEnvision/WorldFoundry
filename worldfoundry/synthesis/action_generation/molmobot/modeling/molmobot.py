"""Inference-only MolmoBot action policy."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from dataclasses import field
from typing import ClassVar

import torch

from ..preprocessing.robot_processing import RobotProcessorConfig
from .action_expert import ActionExpert, ActionExpertConfig
from .video_olmo import VideoOlmo, VideoOlmoConfig


@dataclasses.dataclass
class MolmoBotConfig(VideoOlmoConfig):
    """Checkpoint-compatible MolmoBot inference configuration."""

    _model_name: ClassVar[str] = "molmobot"

    action_dim: int = 7
    action_horizon: int = 16
    n_action_steps: int = 8
    n_obs_steps: int = 1
    obs_step_delta: int = 8
    action_expert: ActionExpertConfig = field(default_factory=ActionExpertConfig)
    action_expert_layer_mode: str = "per_layer"
    flow_matching_num_steps: int = 10

    # Kept because released checkpoint YAML files contain these fields. They do
    # not enable any training path in this inference-only integration.
    flow_matching_cutoff: float = 0.999
    flow_matching_beta_alpha: float = 1.0
    flow_matching_beta_beta: float = 1.5
    num_flow_timestamps: int = 1
    same_noise_per_time: bool = False

    states_mode: str = "cross_attn"
    robot_preprocessor: RobotProcessorConfig | None = None
    robot_postprocessor: RobotProcessorConfig | None = None

    def build_model(self, device=None) -> MolmoBot:
        return MolmoBot(self, device)


class MolmoBot(VideoOlmo):
    """Molmo vision-language backbone with a flow-matching action head."""

    def __init__(self, config: MolmoBotConfig, device=None):
        super().__init__(config, device)
        valid_modes = {"per_layer", "even", "last", "mean"}
        if config.action_expert_layer_mode not in valid_modes:
            raise ValueError(
                f"Unknown action_expert_layer_mode {config.action_expert_layer_mode!r}; "
                f"expected one of {sorted(valid_modes)}."
            )
        if config.action_expert.action_dim != config.action_dim:
            config.action_expert.action_dim = config.action_dim
        if config.action_expert.max_horizon < config.action_horizon:
            config.action_expert.max_horizon = config.action_horizon
        if (
            config.action_expert_layer_mode == "per_layer"
            and config.action_expert.num_layers != self.config.llm.n_layers
        ):
            raise ValueError(
                f"Action expert depth ({config.action_expert.num_layers}) must match LLM layers "
                f"({self.config.llm.n_layers}) in per_layer mode."
            )
        self.action_expert: ActionExpert = config.action_expert.build(
            llm_dim=self.config.llm.d_model,
            device=device,
        )

    def _adapt_state(self, states: torch.Tensor | None) -> torch.Tensor | None:
        mode = self.config.states_mode
        if mode == "skip":
            return None
        if states is None:
            return None
        if mode == "zero":
            return torch.zeros_like(states)
        if mode == "set_arm_zero":
            states = states.clone()
            states[..., :7] = 0
            return states
        if mode == "set_gripper_zero":
            states = states.clone()
            states[..., -1:] = 0
            return states
        if mode in {"cross_attn", "self_attn"}:
            return states
        raise ValueError(f"Unknown states_mode: {mode}")

    @torch.inference_mode()
    def generate_actions(
        self,
        input_ids: torch.LongTensor,
        input_embeddings: torch.FloatTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        attention_bias: torch.Tensor | None = None,
        response_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        images: torch.Tensor | None = None,
        image_masks: torch.Tensor | None = None,
        token_pooling: torch.Tensor | None = None,
        low_res_token_pooling: torch.Tensor | None = None,
        states: torch.Tensor | None = None,
        num_steps: int | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Generate one action chunk using the released Euler flow solver."""
        if states is None:
            raise ValueError("MolmoBot requires robot state for action generation.")

        encoder_attention_mask = self._get_encoder_attention_mask(input_ids, attention_mask)
        base_output = super().forward(
            input_ids=input_ids,
            input_embeddings=input_embeddings,
            attention_mask=attention_mask,
            attention_bias=attention_bias,
            response_mask=response_mask,
            position_ids=position_ids,
            images=images,
            image_masks=image_masks,
            token_pooling=token_pooling,
            low_res_token_pooling=low_res_token_pooling,
            collect_layer_hidden_states=True,
        )
        layer_states = (base_output.internal or {}).get("layer_hidden_states")
        if layer_states is None:
            raise RuntimeError("MolmoBot backbone did not return layer hidden states.")
        selected = self._select_layer_states(layer_states)
        states = self._adapt_state(states)

        steps = self.config.flow_matching_num_steps if num_steps is None else int(num_steps)
        if steps <= 0:
            raise ValueError(f"num_steps must be positive, got {steps}.")
        batch_size = selected[0].shape[0]
        trajectory = torch.randn(
            (batch_size, self.config.action_horizon, self.config.action_dim),
            device=selected[0].device,
            dtype=torch.float32,
            generator=generator,
        )
        dt = 1.0 / steps
        for step in range(steps):
            t = torch.full(
                (batch_size,),
                step / steps,
                device=trajectory.device,
                dtype=torch.float32,
            )
            velocity = self.action_expert(
                trajectory,
                t,
                selected,
                encoder_attention_mask=encoder_attention_mask,
                state_embeddings=states,
                states_mode=self.config.states_mode,
            )
            trajectory = trajectory + dt * velocity
        return trajectory

    @staticmethod
    def _get_encoder_attention_mask(
        input_ids: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if attention_mask is not None:
            return attention_mask.to(dtype=torch.bool).clone()
        if input_ids is not None:
            return input_ids != -1
        return None

    def _select_layer_states(
        self,
        layer_states: Sequence[torch.Tensor],
    ) -> Sequence[torch.Tensor]:
        if not layer_states:
            raise ValueError("No backbone layer states were returned.")
        mode = self.config.action_expert_layer_mode
        target = len(self.action_expert.blocks)
        available = len(layer_states)
        if mode == "per_layer":
            if available != target:
                raise ValueError(f"Expected {target} layer states, received {available}.")
            return layer_states
        if mode == "last":
            return [layer_states[-1]] * target
        if mode == "mean":
            mean_state = torch.stack(list(layer_states), dim=0).mean(dim=0)
            return [mean_state] * target
        if mode == "even":
            if target <= 0:
                raise ValueError("Action expert must contain at least one block.")
            if available == 1:
                return [layer_states[0]] * target
            if target == 1:
                return [layer_states[-1]]
            scale = (available - 1) / float(target - 1)
            return [layer_states[int(round(i * scale))] for i in range(target)]  # noqa: RUF046 - Keeps the explicit int conversion for non-builtin operands.
        raise ValueError(f"Unsupported action_expert_layer_mode: {mode}")


__all__ = ["MolmoBot", "MolmoBotConfig"]
