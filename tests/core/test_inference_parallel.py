"""Real CPU/Gloo inference mesh and checkpoint-shaped projection tests."""

from __future__ import annotations

import copy
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from worldfoundry.core.distributed.model_parallel.inference_parallel import (
    ColumnParallelLinear,
    InferenceParallelContext,
    RowParallelLinear,
    balanced_ranges,
    build_inference_parallel_context,
    validate_parallel_degrees,
)


def _tiny_model():
    from worldfoundry.base_models.diffusion_model.models.networks.cosmos2p5.model import Cosmos25Transformer3DModel

    return Cosmos25Transformer3DModel(
        in_channels=2, out_channels=1, num_attention_heads=4, attention_head_dim=24,
        num_layers=2, mlp_ratio=1.28125, text_in_channels=10, text_embed_dim=12,
        adaln_lora_dim=8, max_size=(8, 12, 12), patch_size=(1, 2, 2),
        rope_enable_fps_modulation=True,
    ).eval()


def _mesh_worker(rank, world_size, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank,
                            world_size=world_size, timeout=timedelta(seconds=60))
    try:
        for tp, cp in ((2, 1), (1, 2)) if world_size == 2 else ((2, 2),):
            parallel = build_inference_parallel_context(tensor_parallel=tp, context_parallel=cp, device="cpu")
            try:
                torch.manual_seed(43)
                model = _tiny_model()
                candidate = copy.deepcopy(model).parallelize(parallel)
                assert not any(parameter.requires_grad for parameter in candidate.parameters())
                hidden = torch.randn(2, 1, 3, 6, 6)
                timestep = torch.tensor([[0.2, 0.4, 0.8], [0.1, 0.3, 0.7]])
                text = torch.randn(2, 5, 10)
                options = {
                    "fps": 16., "condition_mask": torch.rand(2, 1, 3, 6, 6),
                    "padding_mask": torch.rand(2, 1, 6, 6),
                    "attention_mask": torch.tensor([[True, True, False, True, True]] * 2),
                    "control_hidden_states": {"1": torch.randn(2, 27, 96)},
                }
                with torch.inference_mode():
                    expected = model(hidden, timestep, text, **options)
                    output = candidate(hidden, timestep, text, **options)
                    torch.testing.assert_close(output, expected, rtol=3e-5, atol=3e-6)
                    start, end = parallel.local_range(27)
                    local = torch.arange(start, end).reshape(1, -1, 1)
                    gathered = parallel.gather_tokens(local, total=27)
                    torch.testing.assert_close(gathered, torch.arange(27).reshape(1, -1, 1))
                    with torch.inference_mode(False):
                        outputs = [torch.empty_like(output) for _ in range(world_size)]
                        dist.all_gather(outputs, output)
                    for other in outputs:
                        torch.testing.assert_close(output, other, rtol=0, atol=0)

                    linear = torch.nn.Linear(5, 3, bias=True)
                    value = torch.randn(4, 5)
                    first, last = balanced_ranges(5, tp)[parallel.tp_rank]
                    row = RowParallelLinear(linear, start=first, end=last, parallel=parallel)
                    torch.testing.assert_close(row(value[:, first:last]), linear(value), rtol=1e-6, atol=1e-6)

                    cancellation = torch.nn.Linear(5, 1, dtype=torch.bfloat16)
                    cancellation.weight.fill_(1)
                    cancellation.bias.fill_(0.75)
                    terms = torch.tensor([[256., 0.25, 0., -256., 0.]], dtype=torch.bfloat16)
                    row = RowParallelLinear(cancellation, start=first, end=last, parallel=parallel)
                    torch.testing.assert_close(row(terms[:, first:last]), cancellation(terms), rtol=0, atol=0)

                with pytest.raises(ValueError, match="disagree"):
                    parallel.agree(("mismatched-request", rank))
                with pytest.raises(ValueError, match="disagree"), torch.inference_mode():
                    candidate(hidden[:1] if rank == 0 else hidden, timestep, text, **options)
                random_seed = parallel.broadcast_seed(-1)
                parallel.agree(random_seed)
                with pytest.raises(ValueError, match="autocast"), torch.autocast("cpu", dtype=torch.bfloat16):
                    candidate(hidden, timestep, text, **options)
                candidate.train()
                with pytest.raises(RuntimeError, match="eval-mode"), torch.inference_mode():
                    candidate(hidden, timestep, text, **options)
            finally:
                parallel.close()
                parallel.close()
            assert dist.is_initialized()
            dist.barrier()
            with pytest.raises(RuntimeError, match="closed"):
                parallel.agree("closed")

        if world_size == 2:
            with pytest.raises(ValueError, match="disagree"):
                build_inference_parallel_context(
                    tensor_parallel=2 if rank == 0 else 1,
                    context_parallel=1 if rank == 0 else 2, device="cpu",
                )
            assert dist.is_initialized()
            dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [2, 4])
def test_real_mesh_tiny_cosmos_matches_unsharded_inference(tmp_path, world_size):
    mp.spawn(_mesh_worker, args=(world_size, str(tmp_path / "rendezvous")), nprocs=world_size, join=True)


@pytest.mark.parametrize("tp,cp,error", [(True, 1, TypeError), (1, 2., TypeError), (0, 1, ValueError)])
def test_invalid_parallel_degrees(tp, cp, error):
    with pytest.raises(error):
        validate_parallel_degrees(tp, cp)


def test_single_rank_mesh_never_touches_distributed(monkeypatch):
    monkeypatch.setattr(dist, "is_initialized", lambda: pytest.fail("must not inspect world"))
    parallel = build_inference_parallel_context(tensor_parallel=1, context_parallel=1, device="cpu")
    parallel.agree("anything")
    value = torch.ones(1, 3, 1)
    assert parallel.gather_tokens(value, total=3) is value
    assert parallel.sum_tensor_shards(value) is value


def test_missing_mesh_groups_do_not_fall_back_to_default_world(monkeypatch):
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    with pytest.raises(RuntimeError, match="missing"):
        InferenceParallelContext(tp_size=2).sum_tensor_shards(torch.ones(1))


def test_projection_shards_preserve_rng_and_disallow_autograd():
    linear = torch.nn.Linear(5, 7)
    random_state = torch.random.get_rng_state()
    column = ColumnParallelLinear(linear, start=1, end=4)
    row = RowParallelLinear(linear, start=0, end=5, parallel=InferenceParallelContext())
    assert torch.equal(random_state, torch.random.get_rng_state())
    value = torch.randn(2, 5)
    torch.testing.assert_close(column(value), linear(value)[:, 1:4])
    with pytest.raises(RuntimeError, match="autograd"):
        row(value.requires_grad_())


def test_low_precision_cosmos_mesh_is_rejected_before_sharding_or_collectives():
    model = _tiny_model().to(torch.bfloat16)
    original = model.transformer_blocks[0].attn1.to_q
    with pytest.raises(ValueError, match="float32"):
        model.parallelize(InferenceParallelContext(tp_size=2))
    assert model.transformer_blocks[0].attn1.to_q is original
    assert model.parallel_context is None


def test_build_requires_world_and_explicit_device(monkeypatch):
    monkeypatch.setattr(dist, "is_initialized", lambda: False)
    with pytest.raises(RuntimeError, match="initialized"):
        build_inference_parallel_context(tensor_parallel=2, context_parallel=1, device="cpu")
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    with pytest.raises(ValueError, match="explicit"):
        build_inference_parallel_context(tensor_parallel=2, context_parallel=1, device="cuda")
    with pytest.raises(ValueError, match="world size"):
        build_inference_parallel_context(tensor_parallel=4, context_parallel=1, device="cpu")
