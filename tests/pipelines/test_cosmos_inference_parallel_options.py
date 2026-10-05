from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from worldfoundry.pipelines.cosmos import pipeline_cosmos_predict2p5 as predict
from worldfoundry.pipelines.cosmos import pipeline_cosmos_transfer2p5 as transfer


@pytest.mark.parametrize(
    "options,message",
    [
        ({"tensor_parallel": 2}, "offload_mode"),
        ({"tensor_parallel": 3, "offload_mode": "none", "torch_dtype": "float32"}, "divide"),
        ({"tensor_parallel": 2, "offload_mode": "none", "torch_dtype": "float32", "compile": True}, "compile"),
        ({"tensor_parallel": 2, "offload_mode": "none", "torch_dtype": "float32", "quantization": "fp8"}, "quantization"),
        ({"tensor_parallel": 2, "offload_mode": "none", "torch_dtype": "float32"}, "initialized"),
        ({"context_parallel": 2, "offload_mode": "none", "torch_dtype": "bfloat16"}, "float32"),
        ({"tensor_parallel": 2, "offload_mode": "none", "torch_dtype": "float16"}, "float32"),
    ],
)
def test_invalid_predict_parallel_options_fail_before_weights(monkeypatch, options, message):
    import torch.distributed as dist

    monkeypatch.setattr(dist, "is_initialized", lambda: False)
    monkeypatch.setattr(predict.NativeDiffusionPipeline, "from_pretrained", lambda *a, **k: pytest.fail("weights"))
    with pytest.raises(ValueError if message != "initialized" else RuntimeError, match=message):
        predict.CosmosPredict2p5Pipeline.from_pretrained(device="cpu", **options)


def test_transfer_rejects_parallel_request_before_weights(monkeypatch):
    monkeypatch.setattr(transfer.NativeDiffusionPipeline, "from_pretrained", lambda *a, **k: pytest.fail("weights"))
    with pytest.raises(ValueError, match="not supported"):
        transfer.CosmosTransfer2p5Pipeline.from_pretrained(tensor_parallel=2)


class _Context:
    tp_size, cp_size, is_main = 2, 1, True

    def __init__(self):
        self.signatures = []
        self.closed = False

    def agree(self, value):
        self.signatures.append(value)

    def broadcast_seed(self, value):
        return 987 if value < 0 else value

    def close(self):
        self.closed = True


def test_public_parallel_options_forward_and_loading_failure_closes_groups(monkeypatch):
    captured = []
    context = _Context()
    monkeypatch.setattr(predict, "build_cosmos_parallel_context", lambda *a, **k: context)
    monkeypatch.setattr(predict.NativeDiffusionPipeline, "from_pretrained", lambda *a, **k: captured.append(k))
    pipeline = predict.CosmosPredict2p5Pipeline.from_pretrained(device="cpu", offload_mode="none", tensor_parallel=2)
    assert captured[0]["component_options"]["denoiser:main"] == {
        "tensor_parallel": 2, "context_parallel": 1, "parallel_context": context,
    }
    pipeline.close()
    assert context.closed
    context = _Context()

    def fail(*args, **kwargs):
        raise RuntimeError("load failed")

    monkeypatch.setattr(predict.NativeDiffusionPipeline, "from_pretrained", fail)
    with pytest.raises(RuntimeError, match="load failed"):
        predict.CosmosPredict2p5Pipeline.from_pretrained(device="cpu")
    assert context.closed


def test_non_main_rank_returns_complete_tensors_without_competing_artifact_write(monkeypatch):
    context = _Context()
    context.is_main = False
    requests = []
    sample = torch.ones(1, 3, 1, 16, 16)

    def native(request):
        requests.append(request)
        return SimpleNamespace(sample=sample, latents=sample, metadata={})

    monkeypatch.setattr(predict, "save_image_or_video_tensor", lambda *a, **k: pytest.fail("non-main write"))
    pipeline = predict.CosmosPredict2p5Pipeline(native_pipeline=native, device="cpu", model_id="test",
                                               parallel_context=context)
    result = pipeline("prompt", seed=-1, output_path="test.mp4", return_dict=True)
    assert requests[0].sampling.seed == 987
    assert result["video"] is sample
    assert result["artifact_path"] is None
    assert context.signatures


def test_media_request_signature_includes_array_content():
    from worldfoundry.pipelines.cosmos.parallel_options import agree_cosmos_request

    context = _Context()
    agree_cosmos_request(context, video=torch.zeros(2, 3, 4, 4))
    agree_cosmos_request(context, video=torch.ones(2, 3, 4, 4))
    assert context.signatures[0] != context.signatures[1]
