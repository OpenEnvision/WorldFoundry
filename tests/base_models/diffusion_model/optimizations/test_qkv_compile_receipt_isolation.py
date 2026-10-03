"""Compiled projection receipts must stay attached to their owning model."""

from __future__ import annotations

import pytest
import torch

from worldfoundry.base_models.diffusion_model.models.networks.wan.variants.causal_action_21 import (
    CausalWanSelfAttention,
)
from worldfoundry.base_models.diffusion_model.optimizations.qkv_fusion import (
    fuse_qkv_projections,
    project_fused_qkv,
    qkv_fusion_report,
)


@pytest.mark.parametrize("dynamic", [False, True])
def test_fullgraph_multiple_models_shapes_and_cache_replay_keep_receipts_isolated(dynamic):
    torch._dynamo.reset()
    graphs, executions = [], []

    def backend(graph, inputs):
        graphs.append(graph)

        def execute(*args):
            executions.append(graph)
            return graph.forward(*args)

        return execute

    def project(projection, hidden):
        return project_fused_qkv(projection, hidden)

    compiled = torch.compile(project, backend=backend, fullgraph=True, dynamic=dynamic)
    owners = []
    with torch.no_grad():
        for strategy in ("packed", "split", "auto"):
            attention = CausalWanSelfAttention(24, 2).eval()
            assert fuse_qkv_projections(attention, strategy=strategy, split_threshold=8) == 1
            owners.append(attention)
            older_receipts = [qkv_fusion_report(owner) for owner in owners[:-1]]
            state = attention._worldfoundry_qkv_fusion
            for tokens in (7, 16):
                hidden = torch.randn(1, tokens, 24)
                expected = project(attention.qkv, hidden)
                state.reset_request_window()
                actual = compiled(attention.qkv, hidden)
                for value, reference in zip(actual, expected):
                    torch.testing.assert_close(value, reference, rtol=0, atol=0)
                report = qkv_fusion_report(attention)
                assert report["eager_projection_calls"] == 0
                assert report["lifetime_compiled_graph_traces"] > 0
                trace_count = report["lifetime_compiled_graph_traces"]
                graph_count = len(graphs)
                for value, reference in zip(compiled(attention.qkv, hidden), expected):
                    torch.testing.assert_close(value, reference, rtol=0, atol=0)
                assert len(graphs) == graph_count
                assert qkv_fusion_report(attention)["lifetime_compiled_graph_traces"] == trace_count
            assert [qkv_fusion_report(owner) for owner in owners[:-1]] == older_receipts
            report = qkv_fusion_report(attention)
            if strategy != "split":
                assert report["lifetime_compiled_packed_graph_traces"] > 0
            if strategy != "packed":
                assert report["lifetime_compiled_split_graph_traces"] > 0
        attention.qkv.weight.add_(0.125)
        hidden = torch.randn(1, 16, 24)
        expected = project(attention.qkv, hidden)
        state.reset_request_window()
        for value, reference in zip(compiled(attention.qkv, hidden), expected):
            torch.testing.assert_close(value, reference, rtol=0, atol=0)
    assert graphs and len(executions) == 13


def test_compiled_auto_strategy_reconfiguration_preserves_weights_and_dispatch():
    torch._dynamo.reset()
    attention = CausalWanSelfAttention(24, 2).eval()
    assert fuse_qkv_projections(attention, strategy="auto", split_threshold=16) == 1
    pointer = attention.qkv.weight.data_ptr()
    compiled = torch.compile(attention.qkv.project_qkv, backend="eager", fullgraph=True, dynamic=True)
    hidden = torch.randn(1, 8, 24)
    with torch.no_grad():
        for threshold, selected in ((16, "packed"), (4, "split"), (16, "packed")):
            assert fuse_qkv_projections(attention, strategy="auto", split_threshold=threshold) == 0
            expected = attention.qkv.project_qkv(hidden)
            attention._worldfoundry_qkv_fusion.reset_request_window()
            for value, reference in zip(compiled(hidden), expected):
                torch.testing.assert_close(value, reference, rtol=0, atol=0)
            report = qkv_fusion_report(attention)
            assert report[f"lifetime_compiled_{selected}_graph_traces"] > 0
            assert report["eager_projection_calls"] == 0
            assert attention.qkv.weight.data_ptr() == pointer
