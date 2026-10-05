from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from worldfoundry.training.post_training.agentic import (
    AgenticAssistantTurn,
    AgentMessage,
    AgentToolCall,
)


class RayQwenTokenizer:
    eos_token_id = 7
    eos_token = "<|im_end|>"
    pad_token_id = 0

    def convert_tokens_to_ids(self, token: str) -> int:
        assert token == "<|im_end|>"
        return 7

    def apply_chat_template(self, messages, **kwargs):
        del kwargs
        token = 2 if any(message["role"] == "tool" for message in messages) else 1
        return {
            "input_ids": torch.tensor([[token]], dtype=torch.int64),
            "attention_mask": torch.ones((1, 1), dtype=torch.int64),
        }

    def decode(self, token_ids, *, skip_special_tokens: bool) -> str:
        del skip_special_tokens
        values = list(token_ids)
        if values == [3, 4, 7]:
            return '<tool_call>{"name":"calculator","arguments":{"expression":"2+3"}}</tool_call>'
        if values == [5, 7]:
            return "<answer>5</answer>"
        if values == [6, 7]:
            return "<answer>0</answer>"
        return ""


class RayQwenPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=4, use_cache=True)
        self.transitions = nn.Parameter(torch.zeros((8, 8), dtype=torch.float32))
        self.embedding = nn.Embedding(8, 4, dtype=torch.float32)
        with torch.no_grad():
            self.embedding.weight.zero_()

    def forward(self, *, input_ids, attention_mask, use_cache, output_hidden_states=False):
        del attention_mask, use_cache
        hidden = self.embedding(input_ids)
        return SimpleNamespace(
            logits=self.transitions[input_ids] + hidden.sum(dim=-1, keepdim=True) * 0.0,
            hidden_states=(hidden,) if output_hidden_states else None,
        )


def ray_qwen_trainer_policy_factory(*, context) -> RayQwenPolicy:
    del context
    return RayQwenPolicy()


def ray_qwen_tokenizer_factory(*, context) -> RayQwenTokenizer:
    del context
    return RayQwenTokenizer()


class RayQwenTurnPolicy:
    def __init__(self) -> None:
        self.module = RayQwenPolicy()

    def _log_probs(self, prefix: int, tokens: tuple[int, ...]) -> torch.Tensor:
        values: list[torch.Tensor] = []
        previous = prefix
        for token in tokens:
            logits = self.module.transitions[previous]
            values.append(torch.log_softmax(logits, dim=-1)[token])
            previous = token
        return torch.stack(values).detach()

    def generate_turn(
        self,
        *,
        sample_id,
        messages,
        policy_revision,
        sampling_temperature,
        rollout_index,
        turn_index,
        conditioning,
        generator,
    ) -> AgenticAssistantTurn:
        del messages, policy_revision, sampling_temperature, rollout_index, conditioning, generator
        if turn_index == 0:
            tokens = (3, 4, 7)
            return AgenticAssistantTurn(
                message=AgentMessage(
                    role="assistant",
                    tool_calls=(
                        AgentToolCall(
                            call_id=f"{sample_id}-calculator",
                            name="calculator",
                            arguments={"expression": "2+3"},
                        ),
                    ),
                ),
                token_ids=torch.tensor(tokens),
                old_log_probs=self._log_probs(1, tokens),
                finish_reason="tool_calls",
            )
        answer_token = 5 if sample_id.endswith("sample-0000") else 6
        tokens = (answer_token, 7)
        return AgenticAssistantTurn(
            message=AgentMessage(
                role="assistant",
                content="<answer>5</answer>" if answer_token == 5 else "<answer>0</answer>",
            ),
            token_ids=torch.tensor(tokens),
            old_log_probs=self._log_probs(2, tokens),
            finish_reason="stop",
        )


def ray_qwen_rollout_policy_factory(*, context) -> RayQwenTurnPolicy:
    del context
    return RayQwenTurnPolicy()


__all__ = [
    "RayQwenPolicy",
    "RayQwenTokenizer",
    "ray_qwen_rollout_policy_factory",
    "ray_qwen_tokenizer_factory",
    "ray_qwen_trainer_policy_factory",
]
