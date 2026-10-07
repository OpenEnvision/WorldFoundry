# SPDX-License-Identifier: CC-BY-NC-4.0
"""WorldPlay2-Fast's fixed four-evaluation consistency-PDD schedule."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch

from worldfoundry.base_models.diffusion_model.contracts import SamplingConfig, SchedulerStep


@dataclass(frozen=True)
class FixedPDDBlock:
    expert: str
    local_index: int
    start: int
    end: int


class _PDDStateDict(Mapping):
    """Replace interval heads while leaving unrelated checkpoint tensors lazy."""

    def __init__(self, source, replacements, removed):
        self.source = source
        self.replacements = replacements
        self.keys = tuple(key for key in source if key not in removed and key not in replacements) + tuple(replacements)

    def __getitem__(self, key):
        if key in self.replacements:
            return self.replacements[key]
        if key not in self.keys:
            raise KeyError(key)
        return self.source[key]

    def __iter__(self):
        return iter(self.keys)

    def __len__(self):
        return len(self.keys)


class FixedPDD4Scheduler:
    """Two high-noise blocks followed by two low-noise blocks, without CFG."""

    num_intervals = 128
    shift = 7.0
    boundary = 0.9

    def __init__(self):
        raw_boundary = self.boundary / (self.shift - (self.shift - 1.0) * self.boundary)
        high_count = int(round(self.num_intervals * (1.0 - raw_boundary)))
        raw = torch.cat((
            torch.linspace(1.0, raw_boundary, high_count + 1),
            torch.linspace(raw_boundary, 0.0, self.num_intervals - high_count + 1)[1:],
        ))
        self.sigmas = (self.shift * raw / (1.0 + (self.shift - 1.0) * raw)).float()
        self.sigmas[0] = 1.0
        self.sigmas[high_count] = self.boundary
        self.sigmas[-1] = 0.0
        self.timesteps = self.sigmas * 1000.0
        self.boundary_index = high_count
        high_edges = self._partition(0, high_count)
        low_edges = self._partition(high_count, self.num_intervals)
        self._blocks = {
            "high": tuple(FixedPDDBlock("high", i, high_edges[i], high_edges[i + 1]) for i in range(2)),
            "low": tuple(FixedPDDBlock("low", i, low_edges[i], low_edges[i + 1]) for i in range(2)),
        }
        self.ordered_blocks = self._blocks["high"] + self._blocks["low"]

    def schedule(
        self, sampling: SamplingConfig, *, device: torch.device, dtype: torch.dtype,
    ) -> tuple[SchedulerStep, ...]:
        del dtype
        unsupported = set(sampling.scheduler_options) - {"shift"}
        if unsupported:
            raise ValueError(f"unsupported WorldPlay2 PDD scheduler options: {sorted(unsupported)}")
        if sampling.num_inference_steps != 4 or sampling.guidance_scale != 1.0:
            raise ValueError("WorldPlay2-Fast requires four steps and guidance_scale=1")
        if float(sampling.scheduler_options.get("shift", self.shift)) != self.shift:
            raise ValueError("WorldPlay2-Fast's released PDD heads require shift=7")
        return tuple(
            SchedulerStep(
                index=index,
                timestep=self.timesteps[block.start].to(device),
                next_timestep=self.timesteps[block.end].to(device),
            )
            for index, block in enumerate(self.ordered_blocks)
        )

    def scale_model_input(self, latents: torch.Tensor, step: SchedulerStep) -> torch.Tensor:
        del step
        return latents

    def step(
        self, model_output: torch.Tensor, step: SchedulerStep, latents: torch.Tensor,
        *, generator: torch.Generator,
    ) -> torch.Tensor:
        block = self.ordered_blocks[step.index]
        displacement, endpoint_velocity = model_output.chunk(2, dim=1)
        sigma_endpoint = self.sigmas[block.end - 1].to(latents.device)
        state_endpoint = latents.float() + displacement.float()
        x0 = state_endpoint - sigma_endpoint * endpoint_velocity.float()
        if block.end == self.num_intervals:
            return x0.to(latents.dtype)
        sigma_end = self.sigmas[block.end].to(latents.device)
        noise = torch.randn(x0.shape, generator=generator, device=x0.device, dtype=torch.float32)
        return ((1.0 - sigma_end) * x0 + sigma_end * noise).to(latents.dtype)

    @staticmethod
    def _partition(start: int, end: int) -> list[int]:
        return [start, start + (end - start) // 2, end]

    def blocks(self, expert: str) -> tuple[FixedPDDBlock, ...]:
        try:
            return self._blocks[expert]
        except KeyError as error:
            raise ValueError(f"expert must be high/low, got {expert!r}") from error

    def expert_range(self, expert: str) -> tuple[int, int]:
        if expert == "high":
            return 0, self.boundary_index
        if expert == "low":
            return self.boundary_index, self.num_intervals
        raise ValueError(f"expert must be high/low, got {expert!r}")

    def compact_head(self, expert, interval_weight, interval_bias):
        expert_start, expert_end = self.expert_range(expert)
        expected = expert_end - expert_start
        if interval_weight.ndim != 3 or interval_weight.shape[0] != expected:
            raise ValueError(f"{expert} PDD interval weights must have shape [{expected}, O, I]")
        if interval_bias.ndim != 2 or interval_bias.shape[0] != expected:
            raise ValueError(f"{expert} PDD interval biases must have shape [{expected}, O]")
        weights, biases = [], []
        for block in self.blocks(expert):
            indices = list(range(block.start - expert_start, block.end - 1 - expert_start))
            coefficients = (
                self.sigmas[block.start + 1:block.end] - self.sigmas[block.start:block.end - 1]
            ).to(interval_weight)
            displacement_weight = torch.einsum("k,koi->oi", coefficients, interval_weight[indices])
            displacement_bias = torch.einsum("k,ko->o", coefficients, interval_bias[indices])
            endpoint_index = block.end - 1 - expert_start
            weights.append(torch.cat((displacement_weight, interval_weight[endpoint_index]), dim=0))
            biases.append(torch.cat((displacement_bias, interval_bias[endpoint_index]), dim=0))
        return torch.stack(weights), torch.stack(biases)

    def convert_state_dict(self, state: Mapping, *, expert: str) -> Mapping:
        """Compact supported interval-head checkpoints for NativeModuleLoader."""
        compact_weight, compact_bias = "head.block_weight", "head.block_bias"
        removed = {"head.head.weight", "head.head.bias"}
        if compact_weight in state and compact_bias in state:
            if not any(key in state for key in removed):
                return state
            return _PDDStateDict(state, {}, removed)
        weight_key, interval_weight = None, None
        for key in ("head.weight", "head.pdd_weight", "head.head.weight"):
            if key in state:
                tensor = state[key]
                if tensor.ndim == 3:
                    weight_key, interval_weight = key, tensor
                    break
        bias_key, interval_bias = None, None
        for key in ("head.bias", "head.pdd_bias", "head.head.bias"):
            if key in state:
                tensor = state[key]
                if tensor.ndim == 2:
                    bias_key, interval_bias = key, tensor
                    break
        if weight_key is None or bias_key is None:
            raise KeyError("WorldPlay2-Fast checkpoint requires compact or interval PDD heads")
        weight, bias = self.compact_head(expert, interval_weight, interval_bias)
        removed.update((weight_key, bias_key))
        return _PDDStateDict(state, {compact_weight: weight, compact_bias: bias}, removed)
