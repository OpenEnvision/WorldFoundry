from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from worldfoundry.base_models.diffusion_model import NativeDiffusionPipeline
from worldfoundry.base_models.diffusion_model.components import (
    BuildPurpose,
    validate_runtime_policy_for_purpose,
)
from worldfoundry.core.model_loading.policy import (
    AttentionBackend,
    QuantizationMode,
    RuntimePolicy,
)
from worldfoundry.pipelines import native_diffusion as native_diffusion_module
from worldfoundry.pipelines.wan.pipeline_wan_2p2 import Wan2p2Pipeline


def test_public_native_pipeline_forwards_runtime_optimizations(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_from_pretrained(cls, model_id: str, **kwargs: object) -> SimpleNamespace:
        del cls
        captured.update(kwargs)
        return SimpleNamespace(model_id=model_id)

    monkeypatch.setattr(
        NativeDiffusionPipeline,
        "from_pretrained",
        classmethod(fake_from_pretrained),
    )

    Wan2p2Pipeline.from_pretrained(
        model_path="/checkpoints/wan22",
        device="cpu",
        torch_dtype="bfloat16",
        offload_mode="none",
        attention_backend="flash2",
        quantization={"mode": "fp8", "min_features": 2048, "exclude": ["head"]},
        fuse_qkv=True,
        qkv_strategy="split",
        qkv_split_threshold=4096,
        inplace_residual=True,
        static_cross_kv=True,
        fused_rope=True,
        rope_precision="fp32",
        rms_norm_precision="fp32",
        dit_weight_dtype="bf16",
        vae_weight_dtype="bf16",
        vae_decode_autocast="bf16",
        vae_channels_last_3d=True,
        vae_spatial_tiling=True,
        vae_tile_size=(34, 52),
        vae_tile_stride=(18, 26),
        vae_temporal_chunk_size=4,
        torch_compile=True,
        compile_backend="inductor",
        compile_mode="max-autotune-no-cudagraphs",
        compile_fullgraph=True,
        compile_dynamic=False,
        runtime_options={"request_tag": "release-ab"},
    )

    policy = captured["policy"]
    assert policy.device == torch.device("cpu")
    assert policy.dtype == torch.bfloat16
    assert policy.attention is AttentionBackend.FLASH_ATTENTION_2
    assert policy.quantization.mode is QuantizationMode.FP8
    assert policy.quantization.exclude == ("head",)
    assert policy.quantization.options["min_features"] == 2048
    assert policy.compile is True
    assert dict(policy.options) == {
        "request_tag": "release-ab",
        "compile_backend": "inductor",
        "compile_dynamic": False,
        "compile_fullgraph": True,
        "compile_mode": "max-autotune-no-cudagraphs",
        "dit_weight_dtype": "bf16",
        "fused_rope": True,
        "fuse_qkv": True,
        "inplace_residual": True,
        "qkv_split_threshold": 4096,
        "qkv_strategy": "split",
        "rms_norm_precision": "fp32",
        "rope_precision": "fp32",
        "static_cross_kv": True,
        "vae_channels_last_3d": True,
        "vae_decode_autocast": "bf16",
        "vae_spatial_tiling": True,
        "vae_temporal_chunk_size": 4,
        "vae_tile_size": (34, 52),
        "vae_tile_stride": (18, 26),
        "vae_weight_dtype": "bf16",
    }


def test_public_native_pipeline_forwards_custom_compile_options(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_from_pretrained(cls, model_id: str, **kwargs: object) -> SimpleNamespace:
        del cls
        captured.update(kwargs)
        return SimpleNamespace(model_id=model_id)

    monkeypatch.setattr(
        NativeDiffusionPipeline,
        "from_pretrained",
        classmethod(fake_from_pretrained),
    )

    Wan2p2Pipeline.from_pretrained(
        model_path="/checkpoints/wan22",
        device="cpu",
        torch_compile=True,
        compile_backend="inductor",
        compile_options={"epilogue_fusion": True},
    )

    policy = captured["policy"]
    assert policy.compile is True
    assert dict(policy.options) == {
        "compile_backend": "inductor",
        "compile_options": {"epilogue_fusion": True},
    }


@pytest.mark.parametrize(
    "kwargs",
    ({"fused_residual_adaln": True}, {"runtime_options": {"fused_residual_adaln": True}}),
)
def test_public_native_pipeline_rejects_unimplemented_residual_adaln(monkeypatch, kwargs) -> None:
    monkeypatch.setattr(
        NativeDiffusionPipeline,
        "from_pretrained",
        classmethod(lambda *_args, **_kwargs: pytest.fail("must reject before loading")),
    )
    with pytest.raises(ValueError, match="fused_residual_adaln.*not implemented"):
        Wan2p2Pipeline.from_pretrained(model_path="/unused", device="cpu", **kwargs)


@pytest.mark.parametrize("purpose", (BuildPurpose.INFERENCE, BuildPurpose.TRAINING))
def test_direct_native_policy_rejects_unimplemented_residual_adaln(purpose) -> None:
    with pytest.raises(ValueError, match="fused_residual_adaln.*not implemented"):
        validate_runtime_policy_for_purpose(
            RuntimePolicy(options={"fused_residual_adaln": True}), purpose=purpose,
        )


def test_disabled_residual_adaln_is_allowed(monkeypatch) -> None:
    captured = {}

    def fake_from_pretrained(cls, model_id, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(model_id=model_id)

    monkeypatch.setattr(NativeDiffusionPipeline, "from_pretrained", classmethod(fake_from_pretrained))
    Wan2p2Pipeline.from_pretrained(model_path="/unused", device="cpu", fused_residual_adaln=False)
    assert "fused_residual_adaln" not in captured["policy"].options
    for disabled in (False, None):
        validate_runtime_policy_for_purpose(
            RuntimePolicy(options={"fused_residual_adaln": disabled}), purpose=BuildPurpose.TRAINING,
        )


def test_native_plan_rejects_unimplemented_residual_adaln() -> None:
    with pytest.raises(ValueError, match="fused_residual_adaln.*not implemented"):
        Wan2p2Pipeline.plan(model_path="/unused", fused_residual_adaln=True)


def test_public_native_pipeline_forwards_cfg_gate_aliases(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_from_pretrained(cls, model_id: str, **kwargs: object) -> SimpleNamespace:
        del cls
        captured.update(kwargs)
        return SimpleNamespace(model_id=model_id)

    monkeypatch.setattr(
        NativeDiffusionPipeline,
        "from_pretrained",
        classmethod(fake_from_pretrained),
    )

    Wan2p2Pipeline.from_pretrained(
        model_path="/checkpoints/wan22",
        device="cpu",
        cfg_gate_step=0.5,
        cfg_gate_fraction=0.5,
    )

    policy = captured["policy"]
    assert dict(policy.options) == {
        "cfg_gate_fraction": 0.5,
        "cfg_gate_step": 0.5,
    }


def test_public_native_pipeline_rejects_non_mapping_runtime_options() -> None:
    try:
        Wan2p2Pipeline._runtime_policy_options({"runtime_options": "fuse_qkv"})
    except TypeError as error:
        assert "runtime_options must be a mapping" in str(error)
    else:
        raise AssertionError("invalid runtime_options must fail before model loading")


def test_training_policy_rejects_inplace_residual() -> None:
    with pytest.raises(ValueError, match="inplace_residual"):
        validate_runtime_policy_for_purpose(
            RuntimePolicy(options={"inplace_residual": True}),
            BuildPurpose.TRAINING,
        )


def test_public_native_pipeline_initializes_sequence_parallel_runtime(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_from_pretrained(cls, model_id: str, **kwargs: object) -> SimpleNamespace:
        del cls
        captured.update(kwargs)
        return SimpleNamespace(model_id=model_id)

    monkeypatch.setattr(
        NativeDiffusionPipeline,
        "from_pretrained",
        classmethod(fake_from_pretrained),
    )
    from worldfoundry.core.distributed.sequence_parallel import runtime as sequence_parallel_runtime

    initialized: list[int] = []

    def fake_initialize(degree: int) -> torch.device:
        initialized.append(degree)
        return torch.device("cuda:1")

    monkeypatch.setattr(
        sequence_parallel_runtime,
        "ensure_parallel_runtime",
        lambda sp_degree, cfg_degree: fake_initialize(sp_degree * cfg_degree),
    )
    pipeline = Wan2p2Pipeline.from_pretrained(
        model_path="/checkpoints/wan22",
        device="cuda",
        sequence_parallel=2,
    )
    policy = captured["policy"]
    assert initialized == [2]
    assert policy.device == torch.device("cuda:1")
    assert policy.options["sequence_parallel"] == 2
    assert pipeline.device == "cuda:1"


def test_wan_save_uses_known_output_range_without_device_reduction(
    monkeypatch,
    tmp_path,
) -> None:
    pipeline = Wan2p2Pipeline.__new__(Wan2p2Pipeline)
    pipeline.model_id = Wan2p2Pipeline.MODEL_ID
    pipeline.generation_type = Wan2p2Pipeline.GENERATION_TYPE
    pipeline.process = lambda **_kwargs: {"prompt": "prompt", "images": None}
    sample = torch.zeros(1, 3, 2, 4, 4)
    pipeline.native_pipeline = lambda _request: SimpleNamespace(
        sample=sample,
        latents=torch.zeros(1),
        metadata={},
    )
    captured: dict[str, object] = {}

    def fake_save(tensor, path, **kwargs):
        captured.update(tensor=tensor, path=path, **kwargs)
        return str(path)

    monkeypatch.setattr(
        native_diffusion_module,
        "save_image_or_video_tensor",
        fake_save,
    )
    output_path = tmp_path / "wan.mp4"

    result = pipeline(
        prompt="prompt",
        output_path=output_path,
        num_frames=2,
        num_inference_steps=1,
        guidance_scale=1.0,
        return_dict=True,
    )

    assert captured["value_range"] == "-1,1"
    assert captured["fps"] == 24
    assert captured["path"] == output_path
    assert result["artifact_path"] == str(output_path)
