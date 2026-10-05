"""Behavioral checks for the second core ownership/correctness review."""

import importlib.util
import sys
import types

import numpy as np
import torch


def test_sam_mask_data_keeps_fields_aligned_after_move():
    from worldfoundry.base_models.perception_core.segment.mask_data import MaskData

    data = MaskData(
        scores=torch.tensor([0.25, 0.5, 0.75], dtype=torch.float16),
        boxes=np.array([[0], [1], [2]]),
        labels=["a", "b", "c"],
    )
    data.filter(torch.tensor([True, False, True]))
    added = MaskData(scores=torch.tensor([1.0], dtype=torch.float16), boxes=np.array([[3]]), labels=["d"])
    data.cat(added)
    added["labels"][0] = "changed"
    data.filter(torch.tensor([2, 0]))
    data.to_numpy()
    np.testing.assert_array_equal(data["scores"], [1.0, 0.25])
    assert data["scores"].dtype == np.float32
    np.testing.assert_array_equal(data["boxes"], [[3], [0]])
    assert data["labels"] == ["d", "a"]


def test_xfuser_attention_projects_and_normalizes_query(monkeypatch):
    from worldfoundry.core.io.paths import package_root

    distributed = types.ModuleType("xfuser.core.distributed")
    distributed.get_sequence_parallel_rank = lambda: 0
    distributed.get_sequence_parallel_world_size = lambda: 1
    distributed.get_sp_group = lambda: None
    captured = {}

    class Attention:
        def __init__(self, **kwargs):
            pass

        def __call__(self, _, *, query, key, value):
            captured.update(query=query, key=key, value=value)
            return query + key + value

    attention = types.ModuleType("xfuser.core.long_ctx_attention")
    attention.xFuserLongContextAttention = Attention
    kernels = types.ModuleType("yunchang.kernels")
    kernels.AttnType = types.SimpleNamespace(FA="fa", NPU="npu")
    parallel = types.ModuleType("worldfoundry.core.distributed.sequence_parallel.xfuser_parallel")
    parallel.initialize_usp = lambda: None
    for name, module in [
        (distributed.__name__, distributed),
        (attention.__name__, attention),
        (kernels.__name__, kernels),
        (parallel.__name__, parallel),
    ]:
        monkeypatch.setitem(sys.modules, name, module)
    spec = importlib.util.spec_from_file_location(
        "isolated_scope_xdit", package_root() / "core/attention/parallel/scope_xdit_context_parallel.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "is_npu_available", lambda: False)
    owner = types.SimpleNamespace(
        num_heads=2,
        q=lambda x: x * 2,
        k=lambda x: x * 3,
        v=lambda x: x * 4,
        norm_q=lambda x: x + 1,
        norm_k=lambda x: x + 2,
        o=lambda x: x + 5,
    )
    x = torch.arange(8, dtype=torch.float32).reshape(1, 2, 4)
    result = module.usp_attn_forward(owner, x, torch.ones(2, 1, 1, dtype=torch.complex64))
    torch.testing.assert_close(captured["query"].flatten(2), x * 2 + 1)
    torch.testing.assert_close(captured["key"].flatten(2), x * 3 + 2)
    torch.testing.assert_close(captured["value"].flatten(2), x * 4)
    torch.testing.assert_close(result, x * 9 + 8)
