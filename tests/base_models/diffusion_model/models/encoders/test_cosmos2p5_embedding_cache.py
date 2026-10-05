from __future__ import annotations

import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import torch

from worldfoundry.base_models.diffusion_model.contracts import DiffusionRequest, SamplingConfig
from worldfoundry.base_models.diffusion_model.models.encoders.cosmos2p5.component import (
    Cosmos25PromptConditioner,
)


class _Tokenizer:
    chat_template = "test-template"
    vocab_size = 32
    special_tokens_map = {"pad_token": "<pad>"}

    def __len__(self):
        return self.vocab_size

    def apply_chat_template(self, conversation, *, max_length, **kwargs):
        text = conversation[-1]["content"][0]["text"]
        tokens = [ord(value) % self.vocab_size for value in text][:max_length]
        return tokens + [0] * (max_length - len(tokens))


class _Backbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([0.3, 0.6, 0.9, 1.2]))
        self.batches = []
        self.input_devices = []
        self.fail = False
        self.eval()

    def forward(self, input_ids):
        self.batches.append(int(input_ids.shape[0]))
        self.input_devices.append(input_ids.device)
        if self.fail:
            raise RuntimeError("encoder failed")
        value = input_ids[..., None].to(self.weight.dtype) * self.weight
        return SimpleNamespace(hidden_states=(value, value.sin(), value.cos()))


def _encoder(**kwargs):
    model = _Backbone()
    encoder = Cosmos25PromptConditioner(model, _Tokenizer(), sequence_length=4, **kwargs)
    return encoder, model


def _encode(encoder, prompts=("abc",), *, dtype=torch.float32):
    return encoder._encode(prompts, device=torch.device("cpu"), dtype=dtype)


def test_cache_preserves_full_batch_results_and_does_not_alias_consumers():
    uncached, _ = _encoder()
    cached, model = _encoder(embedding_cache_size=4)
    prompts = ("abc", "def")
    expected = _encode(uncached, prompts)
    first = _encode(cached, prompts)
    torch.testing.assert_close(first, expected, rtol=0, atol=0)
    with torch.inference_mode():
        first.zero_()
    hit = _encode(cached, prompts)
    torch.testing.assert_close(hit, expected, rtol=0, atol=0)
    with torch.inference_mode():
        hit.fill_(123)
    torch.testing.assert_close(_encode(cached, prompts), expected, rtol=0, atol=0)
    _encode(cached, ("abc",))
    assert model.batches == [2, 1]


def test_default_and_training_paths_do_not_reuse_embeddings():
    default, model = _encoder()
    _encode(default)
    _encode(default)
    assert model.batches == [1, 1]
    cached, model = _encoder(embedding_cache_size=2)
    _encode(cached)
    model.train()
    _encode(cached)
    _encode(cached)
    model.eval()
    _encode(cached)
    assert model.batches == [1, 1, 1, 1]


def test_training_child_bypasses_cache_even_with_eval_parent():
    encoder, model = _encoder(embedding_cache_size=2)
    model.child = torch.nn.Dropout(0.5).eval()
    _encode(encoder)
    model.child.train()
    assert not model.training
    _encode(encoder)
    _encode(encoder)
    model.child.eval()
    _encode(encoder)
    _encode(encoder)
    assert model.batches == [1, 1, 1, 1]


def test_cache_invalidates_weight_dtype_and_encoding_configuration_changes():
    encoder, model = _encoder(embedding_cache_size=2)
    initial = _encode(encoder)
    with torch.no_grad():
        model.weight[0].add_(0.4)
    changed = _encode(encoder)
    assert not torch.equal(initial, changed)
    torch.testing.assert_close(
        changed, encoder._encode_uncached(("abc",), device=torch.device("cpu"), dtype=torch.float32)
    )
    model.to(dtype=torch.float64)
    _encode(encoder, dtype=torch.float64)
    encoder.tokenizer.chat_template = "new-template"
    _encode(encoder, dtype=torch.float64)
    encoder.sequence_length = 3
    assert _encode(encoder, dtype=torch.float64).shape[1] == 3
    encoder.embedding_concat_strategy = "mean_pooling"
    assert _encode(encoder, dtype=torch.float64).shape[-1] == 4
    assert model.batches == [1] * 7


def test_cache_evicts_by_entry_and_byte_limits_and_skips_oversized_batches():
    # One [1, 4, 8] FP32 embedding uses 128 bytes.
    encoder, model = _encoder(embedding_cache_size=4, embedding_cache_max_bytes=192)
    _encode(encoder, ("a",))
    _encode(encoder, ("b",))
    _encode(encoder, ("a",))
    assert model.batches == [1, 1, 1]
    assert encoder._embedding_cache_bytes == 128
    _encode(encoder, ("a", "b"))
    _encode(encoder, ("a", "b"))
    assert model.batches == [1, 1, 1, 2, 2]
    assert encoder._embedding_cache_bytes <= 192
    encoder, model = _encoder(embedding_cache_size=1)
    _encode(encoder, ("a",))
    _encode(encoder, ("b",))
    _encode(encoder, ("a",))
    assert model.batches == [1, 1, 1]


def test_cache_serializes_concurrent_hits_and_retries_failed_misses():
    encoder, model = _encoder(embedding_cache_size=2)
    model.fail = True
    with pytest.raises(RuntimeError, match="encoder failed"):
        _encode(encoder)
    assert encoder._embedding_cache_bytes == 0
    model.fail = False
    with ThreadPoolExecutor(max_workers=4) as pool:
        values = list(pool.map(lambda _: _encode(encoder), range(4)))
    assert model.batches == [1, 1]
    for value in values[1:]:
        torch.testing.assert_close(value, values[0], rtol=0, atol=0)
    encoder.clear_embedding_cache()
    _encode(encoder)
    assert model.batches == [1, 1, 1]


def test_cfg_negative_prompts_and_guidance_off_keep_their_original_semantics():
    encoder, model = _encoder(embedding_cache_size=4)
    request = DiffusionRequest(prompt=["abc", "def"], sampling=SamplingConfig(guidance_scale=7))
    first = encoder.encode(request, device=torch.device("cpu"), dtype=torch.float32)
    repeat = encoder.encode(request, device=torch.device("cpu"), dtype=torch.float32)
    for branch in ("positive", "negative"):
        torch.testing.assert_close(
            getattr(first, branch)["context"], getattr(repeat, branch)["context"], rtol=0, atol=0
        )
    assert model.batches == [2, 2]
    unguided = encoder.encode(
        DiffusionRequest(prompt=["abc", "def"], sampling=SamplingConfig(guidance_scale=1)),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert not unguided.negative
    assert model.batches == [2, 2]


def test_cpu_execution_retains_online_and_explicit_output_device_contracts():
    encoder, model = _encoder(run_on_cpu=True, embedding_cache_size=2, output_device="cpu")
    explicit = _encode(encoder)
    online = encoder.compute_text_embeddings_online({"caption": ["abc"]}, "caption")
    torch.testing.assert_close(online, explicit, rtol=0, atol=0)
    assert model.input_devices == [torch.device("cpu")]


def test_unversioned_weights_bypass_cache_and_autocast_changes_invalidate_it():
    with torch.inference_mode():
        model = _Backbone()
    encoder = Cosmos25PromptConditioner(model, _Tokenizer(), sequence_length=4, embedding_cache_size=2)
    _encode(encoder)
    _encode(encoder)
    assert model.batches == [1, 1]
    encoder, model = _encoder(embedding_cache_size=2)
    _encode(encoder)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        _encode(encoder)
        _encode(encoder)
    _encode(encoder)
    assert model.batches == [1, 1, 1]


@pytest.mark.parametrize(
    "options",
    [
        {"embedding_cache_size": -1},
        {"embedding_cache_max_bytes": -1},
        {"embedding_cache_size": True},
        {"run_on_cpu": "false"},
    ],
)
def test_invalid_cache_options_are_rejected(options):
    with pytest.raises((TypeError, ValueError)):
        _encoder(**options)


def test_component_import_defers_optional_qwen_implementation():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import worldfoundry.base_models.diffusion_model.models.encoders.cosmos2p5.component; assert 'transformers.models.qwen2_5_vl.modeling_qwen2_5_vl' not in sys.modules",
        ],
        check=True,
    )


def test_cpu_builder_keeps_model_policy_separate_and_forwards_cache_options(monkeypatch):
    from worldfoundry.base_models.diffusion_model.models.encoders.cosmos2p5 import component
    from worldfoundry.base_models.diffusion_model.optimizations import OffloadPolicy, RuntimePolicy

    policies = []

    def load_backbone(checkpoint, policy):
        policies.append(policy)
        return _Backbone()

    monkeypatch.setattr(component, "_load_reason1_backbone", load_backbone)
    monkeypatch.setattr(component, "_load_reason1_tokenizer", lambda _: _Tokenizer())
    model_policy = RuntimePolicy(device="cuda:1", offload=OffloadPolicy(mode="block"))
    context = SimpleNamespace(
        policy=model_policy,
        component_options={"run_on_cpu": True, "embedding_cache_size": 2, "embedding_cache_max_bytes": 1024},
        require_checkpoint=lambda name: name,
    )
    encoder = component.build_cosmos25_prompt_conditioner(context)
    assert policies[0].device == torch.device("cpu")
    assert policies[0].offload.mode.value == "none"
    assert model_policy.device == torch.device("cuda:1")
    assert model_policy.offload.mode.value == "block"
    assert encoder.output_device == torch.device("cuda:1")
    assert encoder.embedding_cache_size == 2
    assert encoder.embedding_cache_max_bytes == 1024


def test_legacy_configs_without_new_cache_fields_remain_compatible(monkeypatch):
    from worldfoundry.base_models.diffusion_model.models.encoders.cosmos2p5 import component

    options = {}

    def load_encoder(checkpoint, **kwargs):
        options.update(kwargs)
        return SimpleNamespace(model=_Backbone())

    monkeypatch.setattr(component, "load_cosmos_reason1_prompt_encoder", load_encoder)
    component.CosmosReason1TextEncoder(
        SimpleNamespace(ckpt_path="local", embedding_concat_strategy="full_concat", n_layers_per_group=5)
    )
    assert options["embedding_cache_size"] == 0
    assert options["run_on_cpu"] is False


def test_cache_invalidates_backend_precision_changes_and_lowered_capacity():
    encoder, model = _encoder(embedding_cache_size=4)
    _encode(encoder, ("a",))
    _encode(encoder, ("b",))
    assert encoder._embedding_cache_bytes == 256
    encoder.embedding_cache_max_bytes = 128
    _encode(encoder, ("a",))
    assert encoder._embedding_cache_bytes == 128
    assert model.batches == [1, 1, 1]
    before = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("medium" if before != "medium" else "highest")
        _encode(encoder, ("a",))
        assert model.batches == [1, 1, 1, 1]
    finally:
        torch.set_float32_matmul_precision(before)
    encoder.embedding_cache_size = 0
    _encode(encoder, ("a",))
    assert encoder._embedding_cache_bytes == 0
