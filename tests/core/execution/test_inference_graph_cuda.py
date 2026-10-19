"""Real CUDA capture/replay contracts without model weights or timing thresholds."""

from __future__ import annotations

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
    result_type = namedtuple("Result", ("prediction", "label"))

    def transform(value: torch.Tensor) -> dict:
        return {"nested": ([value + 1], {"result": result_type(value.square(), "prediction")}), "metadata": 3}

    runner = InferenceCUDAGraphRunner(transform, warmup=1)
    first_input = torch.arange(12, device=cuda_device, dtype=torch.float32).reshape(3, 4)
    first = runner(first_input)
    second_input = first_input + 10
    second = runner(second_input)

    for result, value in ((first, first_input), (second, second_input)):
        assert isinstance(result["nested"], tuple)
        assert isinstance(result["nested"][0], list)
        assert isinstance(result["nested"][1]["result"], result_type)
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
