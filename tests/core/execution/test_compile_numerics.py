# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint and autograd contracts for native normalization boundaries."""

from __future__ import annotations

from copy import deepcopy

import pytest
import torch
from torch.fx import GraphModule

from worldfoundry.core.execution.compile_numerics import preserve_normalization_reductions
from worldfoundry.core.nn.transformer.dit import RMSNorm

pytestmark = pytest.mark.unit


def test_prepared_norms_keep_types_keys_hooks_and_live_values() -> None:
    torch.manual_seed(7)
    reference = torch.nn.Sequential(torch.nn.LayerNorm(16), RMSNorm(16)).eval()
    candidate = deepcopy(reference)
    keys = tuple(candidate.state_dict())
    types = tuple(type(module) for module in candidate.modules())
    preserve_normalization_reductions(candidate)
    assert tuple(candidate.state_dict()) == keys
    assert tuple(type(module) for module in candidate.modules()) == types
    observed: list[torch.Tensor] = []

    def hook(module: torch.nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        observed.append(output.clone())

    handle = candidate[0].register_forward_hook(hook)
    graphs: list[GraphModule] = []

    def backend(graph: GraphModule, inputs: list[torch.Tensor]) -> object:
        graphs.append(graph)
        return graph.forward

    compiled = torch.compile(candidate.forward, backend=backend, fullgraph=True)
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        for _ in range(2):
            value = torch.randn(3, 16, dtype=torch.bfloat16)
            torch.testing.assert_close(compiled(value), reference(value), rtol=0, atol=0)
            torch.testing.assert_close(observed[-1], reference[0](value), rtol=0, atol=0)
    handle.remove()
    with torch.no_grad():
        keyword_call = torch.compile(candidate[0].forward, backend="eager", fullgraph=True)
        torch.testing.assert_close(keyword_call(input=value), reference[0](input=value), rtol=0, atol=0)
    assert len(observed) == 2
    targets = {str(node.target) for graph in graphs for node in graph.graph.nodes}
    assert "worldfoundry.compile_layer_norm.default" in targets
    assert "worldfoundry.compile_rms_norm.default" in targets


def test_prepared_norms_retain_autograd() -> None:
    torch.manual_seed(11)
    reference = torch.nn.Sequential(torch.nn.LayerNorm(16), RMSNorm(16))
    candidate = deepcopy(reference)
    preserve_normalization_reductions(candidate)
    compiled = torch.compile(candidate.forward, backend="eager", fullgraph=True)
    value = torch.randn(3, 16, requires_grad=True)
    other = value.detach().clone().requires_grad_()
    with torch.no_grad():
        torch.testing.assert_close(compiled(other), reference(value), rtol=0, atol=0)
    expected = reference(value)
    actual = compiled(other)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    expected.square().sum().backward()
    actual.square().sum().backward()
    torch.testing.assert_close(other.grad, value.grad, rtol=0, atol=0)
    for original, prepared in zip(reference.parameters(), candidate.parameters(), strict=True):
        torch.testing.assert_close(prepared.grad, original.grad, rtol=0, atol=0)
