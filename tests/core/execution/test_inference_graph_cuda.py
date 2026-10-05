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
