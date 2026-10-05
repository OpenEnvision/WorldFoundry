"""Wan request-lifecycle wiring for approximate-attention receipts."""

from __future__ import annotations

import pytest
import torch
from safetensors.torch import save_file

from worldfoundry.base_models.diffusion_model.contracts import (
    Conditioning,
    DenoiserInput,
    DenoiserOutput,
    DiffusionRequest,
    SamplingConfig,
    SchedulerStep,
)
from worldfoundry.base_models.diffusion_model.loaders.checkpoints import (
    CheckpointSpec,
)
from worldfoundry.base_models.diffusion_model.loaders.module import (
    ModuleLoadSpec,
    NativeModuleLoader,
)
from worldfoundry.base_models.diffusion_model.models.denoisers.wan import (
    Wan22DualExpertDenoiser,
    WanDenoiser,
)
from worldfoundry.base_models.diffusion_model.models.networks.wan.model import (
    CrossAttention,
    SelfAttention,
)
from worldfoundry.base_models.diffusion_model.optimizations import (
    approximate_attention as approximate_module,
)
from worldfoundry.base_models.diffusion_model.optimizations.approximate_attention import (
    ApproximateAttentionConfig,
    approximate_attention_lifecycle_report,
    approximate_attention_report,
    install_approximate_attention,
)
from worldfoundry.base_models.diffusion_model.optimizations.static_cross_kv import (
    install_static_cross_kv_cache,
)
from worldfoundry.base_models.diffusion_model.runners.base import (
    NativeDiffusionRunner,
    RunnerComponents,
)
from worldfoundry.core.model_loading.policy import AttentionBackend, RuntimePolicy


class _VSAOps:
    def video_sparse_attn(
        self,
        q,
        k,
        v,
        *,
        variable_block_sizes,
        q_variable_block_sizes,
        topk,
        block_size,
        compress_attn_weight,
    ):
        del (
            k,
            v,
            variable_block_sizes,
            q_variable_block_sizes,
            topk,
            block_size,
            compress_attn_weight,
        )
        return q


class _Block(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = SelfAttention(64, 1).eval()
        self.self_attn.gate_compress = torch.nn.Identity()


class _Model(torch.nn.Module):
    per_token_timestep = False
    inject_sample_info = False
    has_image_input = False
    patch_size = (1, 1, 1)

    def __init__(self) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList([_Block()])

    def forward(self, *, x, timestep, context, **kwargs):
        del timestep, context, kwargs
        batch, channels, frames, height, width = x.shape
        hidden = x.permute(0, 2, 3, 4, 1).reshape(batch, -1, channels)
        angles = torch.zeros(hidden.shape[1], 1, channels // 2)
        freqs = torch.polar(torch.ones_like(angles), angles)
        hidden = self.blocks[0].self_attn(
            hidden,
            freqs,
            _worldfoundry_sparse_grid=(frames, height, width),
        )
        return hidden.reshape(batch, frames, height, width, channels).permute(
            0, 4, 1, 2, 3
        )


class _RunnerConditioner:
    def encode(self, request, *, device, dtype) -> Conditioning:
        del request, device, dtype
        return Conditioning(positive={"context": torch.zeros(1, 2, 4)})


class _RunnerInitializer:
    def initialize(self, request, *, generator, device, dtype) -> torch.Tensor:
        del request, generator
        return torch.randn(1, 64, 2, 4, 4, device=device, dtype=dtype)


class _RunnerScheduler:
    def __init__(self, *, fail_step: bool = False) -> None:
        self.fail_step = fail_step

    def schedule(self, sampling, *, device, dtype) -> tuple[SchedulerStep, ...]:
        assert sampling.num_inference_steps == 1
        return (
            SchedulerStep(
                index=0,
                timestep=torch.tensor([10.0], device=device, dtype=dtype),
                next_timestep=torch.tensor([0.0], device=device, dtype=dtype),
            ),
        )

    def scale_model_input(self, latents, step) -> torch.Tensor:
        del step
        return latents

    def step(self, model_output, step, latents, *, generator) -> torch.Tensor:
        del model_output, step, generator
        if self.fail_step:
            raise RuntimeError("scheduler failed after sparse forward")
        return latents


class _RunnerDecoder:
    def decode(self, latents, request) -> torch.Tensor:
        del request
        return latents


def _native_runner(
    denoiser: WanDenoiser,
    *,
    fail_step: bool = False,
) -> NativeDiffusionRunner:
    return NativeDiffusionRunner(
        model_id="approximate-attention-lifecycle-test",
        components=RunnerComponents(
            denoiser=denoiser,
            conditioner=_RunnerConditioner(),
            latent_initializer=_RunnerInitializer(),
            scheduler=_RunnerScheduler(fail_step=fail_step),
            decoder=_RunnerDecoder(),
        ),
    )


def _runner_request() -> DiffusionRequest:
    return DiffusionRequest(
        prompt="test",
        sampling=SamplingConfig(
            num_inference_steps=1,
            guidance_scale=1.0,
            seed=7,
        ),
    )


def _denoiser_for_model(model: torch.nn.Module) -> WanDenoiser:
    denoiser = WanDenoiser.__new__(WanDenoiser)
    denoiser.model = model
    denoiser._teacache_threshold = None
    denoiser._feature_cache_config = None
    denoiser._feature_cache_requests = {}
    denoiser._feature_cache_receipt_snapshots = {}
    denoiser._static_cross_kv_receipt_snapshots = {}
    denoiser._optimization_request_windows = {}
    denoiser.compute_dtype = torch.float32
    denoiser.reference_condition_key = None
    denoiser.channel_condition_key = None
    denoiser.manage_autocast = False
    denoiser._graph_runner = None
    return denoiser


def _denoiser_with_state(monkeypatch):
    monkeypatch.setattr(approximate_module, "_load_sparse_ops", lambda: _VSAOps())
    monkeypatch.setattr(
        approximate_module,
        "_sparse_runtime_ineligibility",
        lambda q, head_dim: None,
    )
    model = _Model()
    state = install_approximate_attention(
        model,
        ApproximateAttentionConfig(kind="vsa"),
    )
    model._worldfoundry_approximate_attention = state
    denoiser = _denoiser_for_model(model)
    return denoiser, state


def test_wan_end_request_finalizes_approximate_receipt(monkeypatch) -> None:
    denoiser, state = _denoiser_with_state(monkeypatch)
    model_input = DenoiserInput(
        latents=torch.randn(1, 64, 2, 4, 4),
        timestep=torch.tensor([10.0]),
        next_timestep=torch.tensor([0.0]),
        conditioning={"context": torch.zeros(1, 2, 4)},
        step_index=0,
        total_steps=1,
        branch="positive",
        request_id="wan-request",
    )

    output = denoiser(model_input)

    assert output.sample.shape == model_input.latents.shape
    active = approximate_attention_report(state, "wan-request")
    assert active["events"][0]["branch"] == "positive"
    assert active["events"][0]["step"] == 0

    denoiser.end_request("wan-request")

    report = approximate_attention_report(state, "wan-request")
    assert report["finalized"] is True
    assert report["release_reason"] == "completed"
    assert report["coverage"]["complete"] is True
    assert approximate_attention_lifecycle_report(state)["live_requests"] == 0


def test_loader_pending_state_is_replaced_by_successful_runtime_receipt(
    monkeypatch,
    tmp_path,
) -> None:
    """Exercise loader -> provider -> finalization -> runtime audit merge."""

    monkeypatch.setattr(approximate_module, "_load_sparse_ops", lambda: _VSAOps())
    monkeypatch.setattr(
        approximate_module,
        "_sparse_runtime_ineligibility",
        lambda q, head_dim: None,
    )
    checkpoint_path = tmp_path / "tiny-wan.safetensors"
    source_model = _Model().eval()
    save_file(
        {
            name: value.detach().contiguous()
            for name, value in source_model.state_dict().items()
        },
        str(checkpoint_path),
    )
    loaded = NativeModuleLoader().load(
        ModuleLoadSpec(
            module_class=_Model,
            supports_approximate_attention=True,
        ),
        CheckpointSpec(source=str(checkpoint_path)),
        RuntimePolicy(
            attention=AttentionBackend.TORCH,
            options={
                "approximate_attention": {"kind": "vsa"},
                "fuse_qkv": True,
            },
        ),
    )
    assert callable(loaded.blocks[0].self_attn.qkv)
    assert not hasattr(loaded.blocks[0].self_attn, "q")
    state = loaded._worldfoundry_approximate_attention
    build_snapshot = (
        loaded._worldfoundry_applied_optimizations.to_optimization_snapshot()
    )
    assert (
        build_snapshot.effective["approximate_attention_kernel"]
        == "vsa-wrapper-installed (runtime-pending)"
    )
    assert not any(
        "approximate_attention" in reason for reason in build_snapshot.fallbacks
    )

    denoiser = _denoiser_for_model(loaded)
    output = _native_runner(denoiser).run(_runner_request())
    assert len(state._receipt_snapshots) == 1
    request_id = next(iter(state._receipt_snapshots))
    report = denoiser.runtime_optimization_report(request_id)
    receipt = report["runtime"]["approximate_attention"]

    assert output.sample.shape == (1, 64, 2, 4, 4)
    assert report["effective"]["approximate_attention_kernel"] == "vsa"
    assert report["fallbacks"] == []
    assert receipt["runtime_effective"] is True
    assert receipt["finalized"] is True
    assert receipt["completed"] is True
    assert receipt["release_reason"] == "completed"
    assert receipt["coverage"]["complete"] is True
    assert receipt["lifecycle"]["live_requests"] == 0


def test_end_request_freezes_static_cross_kv_receipt_before_release() -> None:
    class _CrossAttentionModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.cross_attn = CrossAttention(64, 1).eval()

    model = _CrossAttentionModel()
    cache = install_static_cross_kv_cache(model)
    denoiser = _denoiser_for_model(model)
    hidden = torch.randn(1, 4, 64)
    context = torch.randn(1, 2, 64)

    model.cross_attn(hidden, context)
    model.cross_attn(hidden, context)
    denoiser.end_request("static-kv-request")

    receipt = denoiser.runtime_optimization_report()["runtime"][
        "static_cross_kv"
    ]
    assert receipt["request_id"] == "static-kv-request"
    assert receipt["finalized"] is True
    assert receipt["release_reason"] == "completed"
    assert receipt["effective"] == "kv-reuse"
    assert receipt["misses"] == 1
    assert receipt["hits"] == 1
    assert cache.report()["effective"] == "installed (runtime-pending)"


def test_runner_error_finalizes_and_releases_approximate_receipt(monkeypatch) -> None:
    denoiser, state = _denoiser_with_state(monkeypatch)
    runner = _native_runner(denoiser, fail_step=True)

    with torch.no_grad(), pytest.raises(
        RuntimeError,
        match="scheduler failed after sparse forward",
    ) as caught:
        runner.run(_runner_request())

    assert len(state._receipt_snapshots) == 1
    request_id = next(iter(state._receipt_snapshots))
    receipt = approximate_attention_report(state, request_id)
    assert receipt["finalized"] is True
    assert receipt["completed"] is False
    assert receipt["release_reason"] == "error"
    assert receipt["error_type"] == type(caught.value).__name__
    assert receipt["lifecycle"]["live_requests"] == 0


class _ExpertSpy:
    def __init__(self) -> None:
        self.inputs: list[DenoiserInput] = []

    def __call__(self, model_input: DenoiserInput) -> DenoiserOutput:
        self.inputs.append(model_input)
        return DenoiserOutput(sample=model_input.latents)

    def end_request(self, request_id: str, *, error=None) -> None:
        del request_id, error


def test_dual_expert_marks_global_step_subsets_as_routed() -> None:
    high = _ExpertSpy()
    low = _ExpertSpy()
    denoiser = Wan22DualExpertDenoiser(
        high,
        low,
        boundary_ratio=0.5,
    )
    model_input = DenoiserInput(
        latents=torch.zeros(1, 2, 1, 2, 2),
        timestep=torch.tensor([100.0]),
        next_timestep=torch.tensor([90.0]),
        conditioning={"context": torch.zeros(1, 2, 4)},
        step_index=7,
        total_steps=10,
        branch="negative",
        request_id="dual-request",
    )

    denoiser(model_input)

    assert not high.inputs
    assert low.inputs[0].step_index == 7
    assert low.inputs[0].conditioning[
        "_worldfoundry_approximate_routed_steps"
    ] is True
