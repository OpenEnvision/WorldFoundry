from __future__ import annotations

import importlib.util
import json

import pytest

torch = pytest.importorskip("torch")

from worldfoundry.training.tuning import (  # noqa: E402
    SANA_ATTENTION,
    WAN_ATTENTION,
    LoraTargetAudit,
    apply_peft_lora,
    apply_peft_lora_with_audit,
    audit_lora_targets,
    inspect_peft_adapter,
    load_peft_adapter,
    merge_peft_adapter,
    save_peft_adapter,
)


class _SelfAttention(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.qkv = torch.nn.Linear(8, 24)
        self.proj = torch.nn.Linear(8, 8)


class _CrossAttention(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_linear = torch.nn.Linear(8, 8)
        self.kv_linear = torch.nn.Linear(8, 16)
        self.proj = torch.nn.Linear(8, 8)


class _SanaBlock(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = _SelfAttention()
        self.cross_attn = _CrossAttention()
        self.mlp = torch.nn.Linear(8, 8)


class _TinySana(torch.nn.Module):
    def __init__(self, blocks: int = 2) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList([_SanaBlock() for _ in range(blocks)])
        self.x_embedder = torch.nn.Module()
        self.x_embedder.proj = torch.nn.Linear(8, 8)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            self_attention = block.attn.proj(block.attn.qkv(value)[..., :8])
            cross_attention = block.cross_attn.proj(
                block.cross_attn.q_linear(value) + block.cross_attn.kv_linear(value)[..., :8]
            )
            value = self_attention + cross_attention + block.mlp(value)
        return value


class _WanAttention(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q = torch.nn.Linear(24, 24)
        self.k = torch.nn.Linear(24, 24)
        self.v = torch.nn.Linear(24, 24)
        self.o = torch.nn.Linear(24, 24)


class _WanBlock(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _WanAttention()
        self.cross_attn = _WanAttention()
        self.ffn = torch.nn.Linear(24, 24)


class _TinyWan(torch.nn.Module):
    def __init__(self, blocks: int = 2) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList([_WanBlock() for _ in range(blocks)])
        self.head = torch.nn.Linear(24, 24)


def test_sana_lora_preset_targets_only_audited_attention_linears() -> None:
    audit = audit_lora_targets(_TinySana(), SANA_ATTENTION)

    assert audit.block_count == 2
    assert len(audit.module_names) == 10
    assert all(name.startswith("blocks.") for name in audit.module_names)
    assert "x_embedder.proj" not in audit.module_names
    assert "blocks.0.mlp" not in audit.module_names
    assert audit.expected_trainable_parameters(rank=4) == 832


def test_sana_lora_preset_fails_closed_when_a_block_graph_drifts() -> None:
    model = _TinySana()
    del model.blocks[1].cross_attn.kv_linear

    with pytest.raises(ValueError, match="target graph drifted"):
        audit_lora_targets(model, SANA_ATTENTION)


def test_wan_lora_preset_targets_only_self_and_cross_attention_linears() -> None:
    audit = audit_lora_targets(_TinyWan(), WAN_ATTENTION)

    assert audit.block_count == 2
    assert len(audit.module_names) == 16
    assert audit.expected_trainable_parameters(rank=2) == 1536
    assert "blocks.0.self_attn.q" in audit.module_names
    assert "blocks.1.cross_attn.o" in audit.module_names
    assert "blocks.0.ffn" not in audit.module_names
    assert "head" not in audit.module_names


def test_wan_lora_preset_fails_closed_when_a_block_graph_drifts() -> None:
    model = _TinyWan()
    del model.blocks[1].cross_attn.o

    with pytest.raises(ValueError, match="target graph drifted"):
        audit_lora_targets(model, WAN_ATTENTION)


def test_peft_dependency_is_lazy_or_injection_matches_the_audit() -> None:
    model = _TinySana()
    if importlib.util.find_spec("peft") is None:
        with pytest.raises(RuntimeError, match="train-core"):
            apply_peft_lora(model, preset=SANA_ATTENTION, rank=2, alpha=2)
        return

    inputs = torch.randn(2, 3, 8)
    expected = model(inputs).detach()
    application = apply_peft_lora(model, preset=SANA_ATTENTION, rank=2, alpha=2)
    assert len(application.targeted_module_names) == 10
    assert application.trainable_parameter_count == 416
    torch.testing.assert_close(application.model(inputs), expected)
    assert application.trainable_parameter_names
    assert all("lora_" in name for name in application.trainable_parameter_names)

    application.model(inputs).sum().backward()
    trainable_gradients = {
        name: parameter.grad for name, parameter in application.model.named_parameters() if parameter.requires_grad
    }
    assert set(trainable_gradients) == set(application.trainable_parameter_names)
    assert all(gradient is not None for gradient in trainable_gradients.values())
    assert all(parameter.grad is None for parameter in application.model.parameters() if not parameter.requires_grad)


def test_model_family_can_apply_an_exact_precomputed_lora_audit() -> None:
    model = _TinyWan(blocks=1)
    selected = ("blocks.0.self_attn.q", "blocks.0.self_attn.v")
    audit = LoraTargetAudit(
        preset="tiny-attention",
        target_pattern=r"blocks\.0\.self_attn\.(?:q|v)",
        module_names=selected,
        module_shapes={name: (24, 24) for name in selected},
        block_count=1,
        base_parameter_count=sum(parameter.numel() for parameter in model.parameters()),
    )

    application = apply_peft_lora_with_audit(
        model,
        audit=audit,
        rank=2,
        alpha=2,
    )

    assert set(application.targeted_module_names) == set(selected)
    assert application.trainable_parameter_count == audit.expected_trainable_parameters(2)


@pytest.mark.skipif(importlib.util.find_spec("peft") is None, reason="PEFT is not installed")
def test_peft_adapter_save_reload_and_merge(tmp_path) -> None:
    torch.manual_seed(17)
    base_model = _TinySana()
    base_state = {name: value.detach().clone() for name, value in base_model.state_dict().items()}
    application = apply_peft_lora(base_model, preset=SANA_ATTENTION, rank=2, alpha=2)
    with torch.no_grad():
        for name, parameter in application.model.named_parameters():
            if parameter.requires_grad:
                parameter.copy_(torch.randn_like(parameter) * 0.03)

    inputs = torch.randn(2, 3, 8)
    expected = application.model(inputs).detach()
    output_dir = tmp_path / "adapter"
    artifact = save_peft_adapter(
        application,
        output_dir,
    )

    assert artifact.path == output_dir
    assert artifact.file_size_bytes["adapter_model.safetensors"] > 0
    assert not (output_dir / "worldfoundry_adapter.json").exists()
    assert inspect_peft_adapter(output_dir) == artifact

    restored_base = _TinySana()
    restored_base.load_state_dict(base_state)
    restored_base.config = None
    restored = load_peft_adapter(
        restored_base,
        output_dir,
        expected_preset=SANA_ATTENTION,
        expected_base_model_id="sana-test",
    )
    torch.testing.assert_close(restored(inputs), expected)

    merged = merge_peft_adapter(restored)
    torch.testing.assert_close(merged(inputs), expected)
    assert merged.config is None
    assert all("lora_" not in name for name, _ in merged.named_parameters())

    with pytest.raises(ValueError, match="targets are incompatible"):
        load_peft_adapter(
            _TinySana(blocks=1),
            output_dir,
            expected_preset=SANA_ATTENTION,
            expected_base_model_id="sana-test",
        )

    config_path = output_dir / "adapter_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["peft_type"] = "UNSUPPORTED"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported PEFT adapter type"):
        load_peft_adapter(
            _TinySana(),
            output_dir,
            expected_preset=SANA_ATTENTION,
            expected_base_model_id="sana-test",
        )

@pytest.mark.skipif(importlib.util.find_spec("peft") is None, reason="PEFT is not installed")
def test_wan_peft_merge_supports_native_offload_linear_wrappers(tmp_path) -> None:
    from worldfoundry.core.vram import AutoWrappedLinear

    torch.manual_seed(23)
    application = apply_peft_lora(
        _TinyWan(),
        preset=WAN_ATTENTION,
        rank=2,
        alpha=2,
    )
    with torch.no_grad():
        for parameter in application.model.parameters():
            if parameter.requires_grad:
                parameter.copy_(torch.randn_like(parameter) * 0.02)
    output_dir = tmp_path / "wan-adapter"
    save_peft_adapter(application, output_dir)

    inference_base = _TinyWan()
    before = inference_base.blocks[0].self_attn.q.weight.detach().clone()
    wrapper_options = {
        "offload_dtype": torch.float32,
        "offload_device": torch.device("cpu"),
        "onload_dtype": torch.float32,
        "onload_device": torch.device("cpu"),
        "preparing_dtype": torch.float32,
        "preparing_device": torch.device("cpu"),
        "computation_dtype": torch.float32,
        "computation_device": torch.device("cpu"),
    }
    for block in inference_base.blocks:
        for attention in (block.self_attn, block.cross_attn):
            for name in ("q", "k", "v", "o"):
                setattr(
                    attention,
                    name,
                    AutoWrappedLinear(getattr(attention, name), **wrapper_options),
                )

    loaded = load_peft_adapter(
        inference_base,
        output_dir,
        expected_preset=WAN_ATTENTION,
        expected_base_model_id="wan2.1-t2v-1.3b",
    )
    merged = merge_peft_adapter(loaded)

    assert merged is inference_base
    assert isinstance(merged.blocks[0].self_attn.q, AutoWrappedLinear)
    assert not torch.equal(merged.blocks[0].self_attn.q.weight, before)
