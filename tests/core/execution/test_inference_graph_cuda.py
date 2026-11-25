"""Real CUDA capture/replay contracts without model weights or timing thresholds."""

from __future__ import annotations

import gc
from collections import namedtuple
from collections.abc import Iterator

import pytest
import torch
from torch.nn import functional as F

from worldfoundry.core.execution.graphs.inference_graph import InferenceCUDAGraphRunner

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for graph capture/replay"),
]


@torch.inference_mode()
def test_invalidation_recaptures_changed_weight_storage_and_python_policy():
    weight = torch.ones(8, device="cuda")
    policy = {"scale": 2.0}

    def transform(value):
        return value * weight * policy["scale"]

    runner = InferenceCUDAGraphRunner(transform, warmup=1)
    value = torch.arange(8, device="cuda", dtype=torch.float32)
    old = runner(value)
    torch.testing.assert_close(old, value * 2, rtol=0, atol=0)
    weight = torch.full((8,), 3.0, device="cuda")
    policy["scale"] = 0.5
    runner.invalidate()
    assert runner.report()["graphs"] == runner.report()["disabled_signatures"] == 0
    actual = runner(value + 1)
    torch.testing.assert_close(actual, (value + 1) * 1.5, rtol=0, atol=0)
    torch.testing.assert_close(old, value * 2, rtol=0, atol=0)
    assert runner.report()["capture"] == 2
    assert runner.report()["eager"] == runner.report()["capture_failed"] == 0


@pytest.fixture
def cuda_device() -> Iterator[torch.device]:
    device = torch.device("cuda", torch.cuda.current_device())
    yield device
    torch.cuda.synchronize(device)


@torch.inference_mode()
def test_graph_replay_tracks_new_latents_conditioning_and_tensor_timesteps(cuda_device: torch.device) -> None:
    generator = torch.Generator(device=cuda_device).manual_seed(19)
    weight = torch.randn((16, 16), generator=generator, device=cuda_device) * 0.1

    def denoise(latent: torch.Tensor, *, conditioning: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        return torch.tanh(F.linear(latent, weight) + conditioning * timestep) + latent * 0.25

    runner = InferenceCUDAGraphRunner(denoise, warmup=2)
    retained = []
    for index, timestep_value in enumerate((1.0, 0.6, 0.2, 0.0)):
        latent = torch.randn((8, 16), generator=generator, device=cuda_device) + index
        conditioning = torch.randn((8, 16), generator=generator, device=cuda_device)
        timestep = torch.tensor(timestep_value, device=cuda_device)
        expected = denoise(latent, conditioning=conditioning, timestep=timestep)
        actual = runner(latent, conditioning=conditioning, timestep=timestep)
        torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)
        retained.append((actual, expected.clone()))

    for actual, expected in retained:
        torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)
    assert len({actual.data_ptr() for actual, _ in retained}) == len(retained)
    report = runner.report()
    assert report["graphs"] == report["capture"] == 1
    assert report["replay"] == 4
    assert report["eager"] == report["capture_failed"] == 0


@torch.inference_mode()
def test_graph_buckets_shapes_and_constants_and_caps_new_signatures(cuda_device: torch.device) -> None:
    def transform(value: torch.Tensor, *, scale: float) -> torch.Tensor:
        return value.square() * scale + value

    runner = InferenceCUDAGraphRunner(transform, warmup=1, max_graphs=3)
    cases = (((3, 5), 1.0), ((3, 5), -2.0), ((4, 5), 1.0), ((5, 5), 1.0), ((3, 5), 1.0))
    for index, (shape, scale) in enumerate(cases):
        value = torch.arange(shape[0] * shape[1], device=cuda_device, dtype=torch.float32).reshape(shape) * 0.1 + index
        actual = runner(value, scale=scale)
        torch.testing.assert_close(actual, transform(value, scale=scale), rtol=0, atol=0)

    report = runner.report()
    assert report["graphs"] == report["capture"] == 3
    assert report["replay"] == 4
    assert report["eager"] == 1
    assert report["capture_failed"] == 0


@torch.inference_mode()
def test_new_request_replays_existing_graph_with_request_scoped_evidence(cuda_device: torch.device) -> None:
    runner = InferenceCUDAGraphRunner(lambda value: value * 3 + 1, warmup=1)
    first_input = torch.ones((4, 8), device=cuda_device)
    first_result = runner(first_input)
    assert runner.report()["capture"] == runner.report()["replay"] == 1

    runner.begin_request_window()
    empty_window = runner.report()
    assert empty_window["window_id"] == empty_window["graphs"] == 1
    assert empty_window["capture"] == empty_window["replay"] == empty_window["eager"] == 0
    assert empty_window["lifetime"]["capture"] == empty_window["lifetime"]["replay"] == 1

    second_input = torch.full_like(first_input, -2)
    second_result = runner(second_input)
    torch.testing.assert_close(second_result, second_input * 3 + 1, rtol=0, atol=0)
    torch.testing.assert_close(first_result, first_input * 3 + 1, rtol=0, atol=0)
    report = runner.report()
    assert report["capture"] == report["eager"] == report["capture_failed"] == 0
    assert report["replay"] == 1
    assert report["lifetime"]["replay"] == 2


@torch.inference_mode()
def test_graph_nested_outputs_do_not_alias_across_replays(cuda_device: torch.device) -> None:
    Result = namedtuple("Result", ("prediction", "label"))

    def transform(value: torch.Tensor) -> dict:
        return {"nested": ([value + 1], {"result": Result(value.square(), "prediction")}), "metadata": 3}

    runner = InferenceCUDAGraphRunner(transform, warmup=1)
    first_input = torch.arange(12, device=cuda_device, dtype=torch.float32).reshape(3, 4)
    first = runner(first_input)
    second_input = first_input + 10
    second = runner(second_input)

    for result, value in ((first, first_input), (second, second_input)):
        assert isinstance(result["nested"], tuple)
        assert isinstance(result["nested"][0], list)
        assert isinstance(result["nested"][1]["result"], Result)
        assert result["metadata"] == 3
        assert result["nested"][1]["result"].label == "prediction"
        torch.testing.assert_close(result["nested"][0][0], value + 1, rtol=0, atol=0)
        torch.testing.assert_close(result["nested"][1]["result"].prediction, value.square(), rtol=0, atol=0)
    assert first["nested"][0][0].data_ptr() != second["nested"][0][0].data_ptr()
    assert first["nested"][1]["result"].prediction.data_ptr() != second["nested"][1]["result"].prediction.data_ptr()
    report = runner.report()
    assert report["capture"] == report["graphs"] == 1
    assert report["replay"] == 2
    assert report["eager"] == report["capture_failed"] == 0


@torch.inference_mode()
def test_inductor_and_cuda_graph_match_eager_with_changing_inputs(cuda_device: torch.device) -> None:
    from torch._dynamo.utils import counters

    def denoise(latent: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        return torch.tanh(latent * (1 + timestep)) + latent.square() * 0.1

    compilations_before = counters["stats"]["unique_graphs"]
    compiled = torch.compile(denoise, backend="inductor", fullgraph=True, options={"triton.cudagraphs": False})
    runner = InferenceCUDAGraphRunner(compiled, warmup=2)
    for index, timestep_value in enumerate((1.0, 0.5, 0.0)):
        latent = torch.linspace(-2, 2, 128, device=cuda_device).reshape(8, 16) + index * 0.1
        timestep = torch.tensor(timestep_value, device=cuda_device)
        actual = runner(latent, timestep)
        torch.testing.assert_close(actual, denoise(latent, timestep), rtol=2e-5, atol=2e-6)
        if index == 0:
            compilations_after_capture = counters["stats"]["unique_graphs"]
            assert compilations_after_capture > compilations_before

    assert counters["stats"]["unique_graphs"] == compilations_after_capture
    report = runner.report()
    assert report["capture"] == report["graphs"] == 1
    assert report["replay"] == 3
    assert report["eager"] == report["capture_failed"] == 0


def test_compiled_capture_preserves_modes_and_generator_lifecycle(cuda_device: torch.device) -> None:
    def compute(value: torch.Tensor) -> torch.Tensor:
        return value.square()

    compiled = torch.compile(compute, backend="inductor", fullgraph=True, options={"triton.cudagraphs": False})

    def transform(value: torch.Tensor) -> torch.Tensor:
        assert not torch.is_grad_enabled()
        mode = torch.is_inference_mode_enabled()
        # A plain no-grad capture must work after inference-mode compilation,
        # without another compilation restoring the shared generator state.
        return (compiled(value) if mode else compute(value)) + (2 if mode else 1)

    runner = InferenceCUDAGraphRunner(transform, warmup=1)
    for mode in (True, False, True):
        with torch.inference_mode(mode), torch.no_grad():
            value = torch.ones(8, device=cuda_device)
            torch.testing.assert_close(runner(value), value.square() + (2 if mode else 1), rtol=0, atol=0)
    assert runner.report()["capture"] == 2
    assert runner.report()["replay"] == 3
    assert runner.report()["capture_failed"] == 0
    assert torch.isfinite(torch.randn(8, device=cuda_device)).all()


@pytest.mark.parametrize("layout", ("sliced", "expanded"))
@torch.no_grad()
def test_compiled_capture_warms_static_input_layout(cuda_device: torch.device, layout: str) -> None:
    def compute(value: torch.Tensor) -> torch.Tensor:
        return value + 1

    storage = torch.arange(24, device=cuda_device, dtype=torch.float32).reshape(3, 8)
    value = storage[:, ::2] if layout == "sliced" else storage[:1, :4].expand(3, 4)
    compiled = torch.compile(compute, backend="inductor", fullgraph=True, options={"triton.cudagraphs": False})
    runner = InferenceCUDAGraphRunner(compiled, warmup=1)
    assert runner.capture(value)
    runner.freeze()
    storage.add_(7)
    torch.testing.assert_close(runner(value), compute(value), rtol=0, atol=0)
    report = runner.report()
    assert report["capture"] == report["replay"] == 1
    assert report["capture_failed"] == report["eager"] == 0


@torch.no_grad()
def test_reordered_keyword_arguments_preserve_tensor_binding(cuda_device: torch.device) -> None:
    runner = InferenceCUDAGraphRunner(lambda *, left, right: left - right, warmup=1)
    left = torch.ones(3, device=cuda_device)
    right = torch.full_like(left, 5)
    runner(left=left, right=right)
    actual = runner(right=right, left=left)
    torch.testing.assert_close(actual, left - right, rtol=0, atol=0)
    assert runner.report()["graphs"] == 1


@torch.no_grad()
def test_autocast_capture_outlives_temporary_weight_cache(cuda_device: torch.device) -> None:
    model = torch.nn.Linear(128, 128, device=cuda_device).eval()
    value = torch.randn(4, 128, device=cuda_device)
    runner = InferenceCUDAGraphRunner(model, warmup=1)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        expected = model(value).clone()
        assert runner.capture(value)
    # Free cached casts before replay. A graph must own or reproduce all casts,
    # not depend on an enclosing request's autocast-cache lifetime.
    torch.cuda.empty_cache()
    runner.freeze()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        actual = runner(value)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert runner.report()["replay"] == 1


@torch.inference_mode()
def test_nested_kv_inputs_and_explicit_capture_freeze(cuda_device: torch.device) -> None:
    def transform(value: torch.Tensor, *, state: dict) -> torch.Tensor:
        key, val = state["kv"][0]
        return value * key + val * state["scale"]

    runner = InferenceCUDAGraphRunner(transform, warmup=1)
    value = torch.ones((3, 4), device=cuda_device)
    state = {"kv": [(value * 2, value * 3)], "scale": 2}
    assert runner.capture(value, state=state)
    assert runner.report()["replay"] == 0
    runner.freeze()
    runner.begin_request_window()
    retained = []
    for index in range(3):
        # In-place edits of inference tensors have no version counter. Every
        # nested leaf must still be copied, not just pointer-compared.
        state["kv"][0][0].add_(1)
        state["kv"][0][1].fill_(index)
        actual = runner(value, state=state)
        expected = transform(value, state=state)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        retained.append((actual, expected.clone()))
    for actual, expected in retained:
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    state["scale"] = 3
    torch.testing.assert_close(runner(value, state=state), transform(value, state=state), rtol=0, atol=0)
    report = runner.report()
    assert report["graphs"] == 1
    assert report["capture"] == report["capture_failed"] == 0
    assert report["replay"] == 3 and report["eager"] == 1
    assert runner.capture(value, state=state)
    assert not runner.capture_on_miss


@torch.inference_mode()
def test_cuda_graph_capture_pauses_and_restores_cyclic_gc(cuda_device: torch.device) -> None:
    assert gc.isenabled()
    gc_states: list[bool] = []

    def op(value: torch.Tensor) -> torch.Tensor:
        gc_states.append(gc.isenabled())
        return value + 1

    runner = InferenceCUDAGraphRunner(op, warmup=1)
    value = torch.ones((4, 4), device=cuda_device)
    assert runner.capture(value)
    assert gc_states == [True, False]
    assert gc.isenabled()

    fail_states: list[bool] = []
    calls = 0

    def fail_on_capture(tensor: torch.Tensor) -> torch.Tensor:
        nonlocal calls
        calls += 1
        fail_states.append(gc.isenabled())
        if calls == 2:
            raise RuntimeError("boom during capture")
        return tensor * 2

    failing_runner = InferenceCUDAGraphRunner(fail_on_capture, warmup=1)
    assert not failing_runner.capture(value)
    assert fail_states == [True, False]
    assert gc.isenabled()
    assert failing_runner.report()["capture_failed"] == 1


@torch.inference_mode()
def test_fast_key_distinguishes_bool_int_float_signed_zero_and_reuses_nan(cuda_device: torch.device) -> None:
    def op(value: torch.Tensor, tag: object) -> torch.Tensor:
        if type(tag) is bool:
            return value + (10.0 if tag else -10.0)
        if type(tag) is int:
            return value + float(tag) * 100.0
        if type(tag) is float:
            if tag != tag:
                return value + 777.0
            return value + (1.0 if repr(tag) == "-0.0" else 2.0 + tag)
        return value

    runner = InferenceCUDAGraphRunner(op, warmup=1, max_graphs=16)
    x = torch.ones((4, 4), device=cuda_device, dtype=torch.float32)

    # 1, 1.0, True must never collide in Python tuple/dict key hashing.
    torch.testing.assert_close(runner(x, 1), x + 100.0, rtol=0, atol=0)
    torch.testing.assert_close(runner(x, 1.0), x + 3.0, rtol=0, atol=0)
    torch.testing.assert_close(runner(x, True), x + 10.0, rtol=0, atol=0)
    # 0, 0.0, -0.0, False must never collide either.
    torch.testing.assert_close(runner(x, 0), x, rtol=0, atol=0)
    torch.testing.assert_close(runner(x, 0.0), x + 2.0, rtol=0, atol=0)
    torch.testing.assert_close(runner(x, -0.0), x + 1.0, rtol=0, atol=0)
    torch.testing.assert_close(runner(x, False), x - 10.0, rtol=0, atol=0)
    # Repeated NaN must hit the same captured bucket rather than re-capturing.
    torch.testing.assert_close(runner(x, float("nan")), x + 777.0, rtol=0, atol=0)
    torch.testing.assert_close(runner(x, float("nan")), x + 777.0, rtol=0, atol=0)

    report = runner.report()
    assert report["graphs"] == report["capture"] == 8
    assert report["replay"] == 9
    assert report["eager"] == report["capture_failed"] == 0


@torch.inference_mode()
def test_fast_key_distinguishes_strides_views_conj_and_ambient_policy_changes(cuda_device: torch.device) -> None:
    def scale(value: torch.Tensor) -> torch.Tensor:
        return value * 2.0

    runner = InferenceCUDAGraphRunner(scale, warmup=1, max_graphs=16)
    base = torch.arange(12, device=cuda_device, dtype=torch.float32).reshape(3, 4)
    transposed = torch.arange(12, device=cuda_device, dtype=torch.float32).reshape(4, 3).t()
    sliced = torch.arange(24, device=cuda_device, dtype=torch.float32).reshape(3, 8)[:, ::2]

    # All three share shape (3, 4) but have different strides -> 3 distinct graphs.
    for tensor in (base, transposed, sliced):
        torch.testing.assert_close(runner(tensor), scale(tensor), rtol=0, atol=0)
    assert runner.report()["graphs"] == 3

    # Complex tensor vs conjugated view (same shape & stride, different is_conj bit).
    z = torch.complex(base, base + 1.0)
    z_conj = z.conj()
    assert z.shape == z_conj.shape and z.stride() == z_conj.stride() and z_conj.is_conj()
    torch.testing.assert_close(runner(z), scale(z), rtol=0, atol=0)
    torch.testing.assert_close(runner(z_conj), scale(z_conj), rtol=0, atol=0)
    assert runner.report()["graphs"] == 5

    # Ambient policy changes (autocast dtype, TF32, float32 matmul precision) must bucket separately.
    policy_runner = InferenceCUDAGraphRunner(lambda t: t + 1.0, warmup=1, max_graphs=16)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        policy_runner(base)
    with torch.autocast("cuda", dtype=torch.float16):
        policy_runner(base)
    assert policy_runner.report()["graphs"] == 2

    prev_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    prev_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    prev_prec = torch.get_float32_matmul_precision()
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        policy_runner(base)
        torch.backends.cuda.matmul.allow_tf32 = True
        policy_runner(base)
        torch.backends.cudnn.allow_tf32 = not prev_cudnn_tf32
        policy_runner(base)
        torch.set_float32_matmul_precision("highest" if prev_prec != "highest" else "high")
        policy_runner(base)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_matmul_tf32
        torch.backends.cudnn.allow_tf32 = prev_cudnn_tf32
        torch.set_float32_matmul_precision(prev_prec)

    assert policy_runner.report()["graphs"] >= 5
    assert policy_runner.report()["capture_failed"] == 0


@torch.inference_mode()
def test_fast_paths_cover_tuples_nested_containers_namedtuples_and_parameters(cuda_device: torch.device) -> None:
    Pair = namedtuple("Pair", ("left", "right"))

    # Built-in tuple/list/dict outputs & inputs (fast path) + namedtuple (pytree path)
    def add_pair(pair: object) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(pair, Pair):
            return pair.left + pair.right, pair.left - pair.right
        if isinstance(pair, list):
            return pair[0] + pair[1] + 1.0, pair[0] - pair[1]
        if isinstance(pair, tuple):
            return pair[0] + pair[1] + 2.0, pair[0] - pair[1]
        raise TypeError(type(pair))

    runner = InferenceCUDAGraphRunner(add_pair, warmup=1, max_graphs=8)
    a = torch.full((3, 4), 5.0, device=cuda_device)
    b = torch.full((3, 4), 2.0, device=cuda_device)

    out_list = runner([a, b])
    out_tup = runner((a, b))
    out_nt = runner(Pair(a, b))

    torch.testing.assert_close(out_list[0], a + b + 1.0, rtol=0, atol=0)
    torch.testing.assert_close(out_tup[0], a + b + 2.0, rtol=0, atol=0)
    torch.testing.assert_close(out_nt[0], a + b, rtol=0, atol=0)
    assert type(out_list) is tuple and type(out_tup) is tuple and type(out_nt) is tuple
    assert out_list[0].data_ptr() != out_tup[0].data_ptr()
    assert runner.report()["graphs"] == 3

    # Tensor subclass (nn.Parameter with requires_grad=False) runs cleanly via pytree path
    # and does not collide with plain torch.Tensor.
    param_runner = InferenceCUDAGraphRunner(lambda t: t * 3.0, warmup=1, max_graphs=4)
    p = torch.nn.Parameter(a.clone(), requires_grad=False)
    torch.testing.assert_close(param_runner(a), a * 3.0, rtol=0, atol=0)
    torch.testing.assert_close(param_runner(p), p * 3.0, rtol=0, atol=0)
    assert param_runner.report()["graphs"] == 2
    assert param_runner.report()["eager"] == 0


@torch.inference_mode()
def test_shared_pool_between_runners_and_isolated_invalidation(cuda_device: torch.device) -> None:
    from worldfoundry.core.execution.graphs.cuda_graph import graph_pool_handle

    shared_pool = graph_pool_handle()
    runner_a = InferenceCUDAGraphRunner(lambda x: (x * 2.0).sin() + x, warmup=1, pool=shared_pool)
    runner_b = InferenceCUDAGraphRunner(lambda x: (x * 3.0).cos() - x, warmup=1, pool=shared_pool)
    assert runner_a.pool == shared_pool
    assert runner_b.pool == shared_pool

    x = torch.linspace(-1.0, 1.0, 256, device=cuda_device).reshape(16, 16)
    out_a1 = runner_a(x)
    out_b1 = runner_b(out_a1)
    expected_a1 = (x * 2.0).sin() + x
    expected_b1 = (expected_a1 * 3.0).cos() - expected_a1
    torch.testing.assert_close(out_a1, expected_a1, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(out_b1, expected_b1, rtol=1e-6, atol=1e-6)

    # Invalidating runner_a must NOT break runner_b or drop runner_a's shared pool handle.
    runner_a.invalidate()
    assert runner_a.pool == shared_pool
    assert runner_b.pool == shared_pool
    assert runner_a.report()["graphs"] == 0
    assert runner_b.report()["graphs"] == 1

    x2 = x + 0.5
    out_b2 = runner_b(x2)
    torch.testing.assert_close(out_b2, (x2 * 3.0).cos() - x2, rtol=1e-6, atol=1e-6)
    out_a2 = runner_a(x2)
    torch.testing.assert_close(out_a2, (x2 * 2.0).sin() + x2, rtol=1e-6, atol=1e-6)
    # Earlier outputs retained across both replays and invalidation remain intact.
    torch.testing.assert_close(out_a1, expected_a1, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(out_b1, expected_b1, rtol=1e-6, atol=1e-6)
