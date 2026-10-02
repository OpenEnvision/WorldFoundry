from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from worldfoundry.training.distributed.ray_runtime import RayWorkerContext
from worldfoundry.training.post_training.agentic import (
    AgenticAssistantTurn,
    AgentMessage,
    AgentToolCall,
    LocalAgentTool,
    LocalToolExecutor,
    agentic_trajectory_from_packed,
)
from worldfoundry.training.post_training.rl.algorithms.token_policy import (
    TokenReplayResult,
)


class RayToyPolicyModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("base", torch.tensor(0.125))
        self.lora_A = nn.Parameter(torch.tensor(0.05))

    def token_log_probs(self, token_ids: torch.Tensor) -> torch.Tensor:
        return -token_ids.to(dtype=torch.float32) * (self.base + self.lora_A)


class RayToyTurnPolicy:
    def __init__(self, *, context: RayWorkerContext) -> None:
        self.context = context
        self.module = RayToyPolicyModule()

    def generate_turn(
        self,
        *,
        sample_id: str,
        messages: tuple[AgentMessage, ...],
        policy_revision: str,
        sampling_temperature: float,
        rollout_index: int,
        turn_index: int,
        conditioning: Mapping[str, object],
        generator: torch.Generator | None,
    ) -> AgenticAssistantTurn:
        del policy_revision, sampling_temperature, rollout_index, generator
        if sample_id == "sample-b":
            raise RuntimeError("isolated sibling failure")
        token = int(conditioning["answer_token"])
        if turn_index == 0:
            token_ids = torch.tensor([token], dtype=torch.int64)
            return AgenticAssistantTurn(
                message=AgentMessage(
                    role="assistant",
                    tool_calls=(
                        AgentToolCall(
                            call_id=f"{sample_id}-add",
                            name="add",
                            arguments={"left": 2, "right": 3},
                        ),
                    ),
                ),
                token_ids=token_ids,
                old_log_probs=self.module.token_log_probs(token_ids).detach(),
                finish_reason="tool_calls",
            )
        if messages[-1].role != "tool" or messages[-1].content != "5":
            raise RuntimeError("tool result was not returned to the policy")
        token_ids = torch.tensor([token + 1], dtype=torch.int64)
        return AgenticAssistantTurn(
            message=AgentMessage(role="assistant", content="The answer is 5."),
            token_ids=token_ids,
            old_log_probs=self.module.token_log_probs(token_ids).detach(),
            finish_reason="stop",
        )


def ray_toy_policy_factory(*, context: RayWorkerContext) -> RayToyTurnPolicy:
    return RayToyTurnPolicy(context=context)


def _add(arguments: Mapping[str, object]) -> int:
    return int(arguments["left"]) + int(arguments["right"])


def ray_tool_executor_factory(*, context: RayWorkerContext) -> LocalToolExecutor:
    del context
    return LocalToolExecutor((LocalAgentTool("add", _add),))


class RayToyReplayAdapter:
    def __init__(self, module: RayToyPolicyModule) -> None:
        self.module = module
        self.sample_batches: list[tuple[str, ...]] = []

    def replay(self, trajectory, *, training: bool) -> TokenReplayResult:
        del training
        agentic = agentic_trajectory_from_packed(trajectory)
        self.sample_batches.append(tuple(sample.request.sample_id for sample in agentic.samples))
        return TokenReplayResult(
            log_probs=self.module.token_log_probs(trajectory.tokens),
            sampling_temperature=trajectory.sampling_temperature,
        )


__all__ = [
    "RayToyPolicyModule",
    "RayToyReplayAdapter",
    "RayToyTurnPolicy",
    "ray_tool_executor_factory",
    "ray_toy_policy_factory",
]
