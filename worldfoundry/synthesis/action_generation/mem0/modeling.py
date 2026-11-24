"""Checkpoint-compatible inference graph for the Mem-0 execution policy."""

from __future__ import annotations

from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn

from .action_head import FlowmatchingActionHead
from .classifier import SubtaskEndClassifier
from .memory_bank import MemoryBank
from .qwen_encoder import Qwen3VL_Encapsulation


def namespace_config(value: Mapping[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(**dict(value))


class Mem0Policy(nn.Module):
    """Inference subset of the official execution module with identical parameter names."""

    def __init__(
        self,
        *,
        backbone_path: str,
        device: torch.device,
        dtype: torch.dtype,
        attention_implementation: str,
        memory_config: Mapping[str, Any],
        action_config: Mapping[str, Any],
        classifier_config: Mapping[str, Any],
        tokenizer_max_length: int,
    ) -> None:
        super().__init__()
        self.qwen_model = Qwen3VL_Encapsulation(
            backbone_path,
            device=device,
            dtype=dtype,
            attention_implementation=attention_implementation,
        )
        hidden_size = int(self.qwen_model.hidden_size)
        self.memory_bank = MemoryBank(
            hidden_dim=hidden_size,
            window_size=int(memory_config["window_size"]),
            initial_anchor_size=int(memory_config["initial_anchor_size"]),
            num_heads=int(memory_config["num_heads"]),
            dropout=float(memory_config["dropout"]),
            memory_accumulation=int(memory_config["memory_accumulation"]),
        ).to(device=device, dtype=dtype)
        self.action_model = FlowmatchingActionHead(
            namespace_config(action_config),
            hidden_size=hidden_size,
        ).to(device=device, dtype=dtype)
        hidden_sizes = [int(value) for value in classifier_config["hidden_sizes"]]
        self.classifier = SubtaskEndClassifier(
            hidden_sizes=hidden_sizes,
            dropout=float(classifier_config["dropout"]),
            pos_weight=(
                None if classifier_config.get("pos_weight") is None else float(classifier_config["pos_weight"])
            ),
        ).to(device=device, dtype=dtype)
        self.classifier_threshold = float(classifier_config["threshold"])
        self.tokenizer_max_length = tokenizer_max_length
        self.device = device
        self.dtype = dtype

    @torch.inference_mode()
    def predict_action(
        self,
        *,
        image: Any,
        instruction: str,
        state: torch.Tensor,
        episode_id: Any,
    ) -> tuple[torch.Tensor, bool]:
        inputs = self.qwen_model.build_qwenvl_inputs(
            [[image]],
            [instruction],
            max_length=self.tokenizer_max_length,
        )
        outputs = self.qwen_model(
            **inputs,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )
        image_feature, text_feature = self.qwen_model.extract_features(
            inputs.input_ids,
            outputs.hidden_states[-1],
        )
        fused_memory, fused_anchor, subtask_ended = self.memory_bank.update_on_eval(
            image_feature,
            text_feature,
            self.classifier,
            episode_id,
            classifier_threshold=self.classifier_threshold,
        )
        summary = torch.cat([fused_memory, fused_anchor, text_feature], dim=1)
        actions = self.action_model.predict_action(summary, state)
        return actions, subtask_ended

    def reset(self, episode_id: Any | None = None) -> None:
        if episode_id is None:
            self.memory_bank.reset()
        else:
            self.memory_bank.reset([episode_id])


__all__ = ["Mem0Policy", "namespace_config"]
