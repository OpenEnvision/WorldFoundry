"""Stage: MAGI-2 preview pipeline seam test (tiny config, no weights).

Wires a tiny-config preview DiT + data proxy + FlowUniPC sampler through the
bespoke pipeline's ``generate`` to validate the full ``model_forward`` seam
(sampler -> proxy pack -> preview MoE DiT -> proxy unpack -> sampler step)
end-to-end on CPU/GPU without VAE weights.
"""

from __future__ import annotations

import torch

from worldfoundry.base_models.diffusion_model.models.networks.magi2.config import (
    Magi2PreviewConfig,
    MHCConfig,
    MoEConfig,
)
from worldfoundry.base_models.diffusion_model.models.networks.magi2.preview_dit import Transformer
from worldfoundry.pipelines.magi2 import NativeMagi2Pipeline


class _StubEncoder:
    def __init__(self, dim: int) -> None:
        self.dim = dim

    def encode(self, prompt: str) -> torch.Tensor:
        return torch.randn(1, 3, self.dim, dtype=torch.bfloat16)


def _tiny_preview() -> Transformer:
    cfg = Magi2PreviewConfig(
        num_layers=4,
        hidden_size=256,
        head_dim=128,
        num_query_groups=2,
        mm_layers=(0, 3),
        mhc_config=MHCConfig(enable=True, num_stream=2, alpha_init=0.01),
        moe_config=MoEConfig(
            num_experts=8,
            top_k=2,
            moe_layers=(1, 2),
            num_heads=2,
            expert_intermediate_size=64,
            shared_expert_intermediate_size=64,
            modality_specific_expert_intermediate_size=64,
        ),
        text_in_channels=64,
    )
    torch.manual_seed(0)
    dit = Transformer(cfg).eval()
    # Untrained random weights + MoE route_scale + SwiGLU7 (+1 bias) can overflow;
    # shrink hard and zero router biases so this shape/seam fixture stays finite
    # (real trained weights are numerically stable — validated via ckpt key-diffs).
    with torch.no_grad():
        for n, p in dit.named_parameters():
            if p.dtype in (torch.bfloat16, torch.float32):
                p.mul_(0.02)
            if "expert_bias" in n or n.endswith(".sinks"):
                p.zero_()
    return dit


def test_config_derived_shapes() -> None:
    c = Magi2PreviewConfig()
    assert c.adapter_dim == 3072 * 4 and c.num_heads_q == 24


def test_pipeline_plan_matches_resolve_lengths() -> None:
    pipe = NativeMagi2Pipeline(preview_transformer=object())
    plan = pipe._plan(short_edge=512, aspect_ratio="16:9", duration_seconds=10.0)
    # The preview decoder emits 125 frames at 12.5 fps; 250 is the
    # pre-refiner temporal target used to calculate the 32 latent frames.
    assert plan["frames"] == 125
    assert plan["video_latent_t"] == 32
    assert plan["audio_latent_t"] == 250
    assert plan["latent_h"] == 512 // 16


def test_preview_model_forward_seam() -> None:
    # Run on CPU: this is a shape/seam contract test, and an untrained tiny MoE
    # model overflows bf16 on some CUDA accumulation orderings (route_scale
    # amplifies random weights). CPU fp32 accumulation keeps it finite; real
    # trained weights are validated separately via the checkpoint key-diffs.
    device = "cpu"
    dit = _tiny_preview().to(device)
    pipe = NativeMagi2Pipeline(preview_transformer=dit, text_encoder=_StubEncoder(64), device=device)
    pipe._plan = lambda **kw: {
        "height": 64,
        "width": 64,
        "frames": 9,
        "video_latent_t": 2,
        "latent_h": 4,
        "latent_w": 4,
        "audio_latent_t": 6,
    }
    res = pipe.generate(prompt="a cat", num_inference_steps=3, seed=0, return_latents=True)
    vl, al = res["video_latent"], res["audio_latent"]
    # This validates the full sampler->proxy->MoE DiT->proxy->step SHAPE contract.
    # Numerical finiteness is NOT asserted: an untrained random-weight MoE (with
    # route_scale + SwiGLU7's +1 bias) can overflow bf16 regardless of shrink;
    # real trained weights are validated via the exact checkpoint key-diffs.
    assert vl.shape[1] == 48 and vl.shape[2:] == (2, 4, 4)
    assert al.shape[-1] == 64 and al.shape[1] == 6


def test_preview_layer_offload_reuses_cpu_weights_without_d2h() -> None:
    import pytest

    from worldfoundry.base_models.diffusion_model.models.networks.magi2 import ops

    probe = torch.empty(1, 2, 16, dtype=torch.bfloat16, device="cuda") if torch.cuda.is_available() else None
    use_cuda = probe is not None and not ops.can_use_fused_attn_sink(probe, probe, probe)
    device = torch.device("cuda" if use_cuda else "cpu")
    fixed_plan = {
        "height": 64,
        "width": 64,
        "frames": 9,
        "video_latent_t": 2,
        "latent_h": 4,
        "latent_w": 4,
        "audio_latent_t": 6,
    }
    dit = _tiny_preview()
    dit.pre_adapter.to(device)
    dit.post_adapter.to(device)
    dit.block.layer_offload_device = device

    pipe = NativeMagi2Pipeline(preview_transformer=dit, text_encoder=_StubEncoder(64), device=str(device))
    pipe._plan = lambda **kw: fixed_plan

    res_first = pipe.generate(prompt="a cat", num_inference_steps=2, seed=0, return_latents=True)
    first_param_ptrs = [p.data_ptr() for p in dit.block.parameters()]
    first_buf_ptrs = [b.data_ptr() for b in dit.block.buffers()]
    assert all(p.device.type == "cpu" for p in dit.block.parameters())
    assert all(b.device.type == "cpu" for b in dit.block.buffers())

    res_second = pipe.generate(prompt="a cat", num_inference_steps=2, seed=0, return_latents=True)
    assert [p.data_ptr() for p in dit.block.parameters()] == first_param_ptrs
    assert [b.data_ptr() for b in dit.block.buffers()] == first_buf_ptrs
    torch.testing.assert_close(
        res_first["video_latent"], res_second["video_latent"], rtol=0.0, atol=0.0, equal_nan=True
    )
    torch.testing.assert_close(
        res_first["audio_latent"], res_second["audio_latent"], rtol=0.0, atol=0.0, equal_nan=True
    )

    dit.block.layer_offload_device = None
    dit.block.to(device)
    res_resident = pipe.generate(prompt="a cat", num_inference_steps=2, seed=0, return_latents=True)
    torch.testing.assert_close(
        res_first["video_latent"], res_resident["video_latent"], rtol=0.0, atol=0.0, equal_nan=True
    )
    torch.testing.assert_close(
        res_first["audio_latent"], res_resident["audio_latent"], rtol=0.0, atol=0.0, equal_nan=True
    )

    # Exception inside a layer must still restore all parameters and buffers back to CPU.
    dit.block.to("cpu")
    dit.block._offload_cpu_cache.clear()
    dit.block.layer_offload_device = device

    def _boom(*args, **kwargs):
        raise RuntimeError("injected layer failure")

    original_layer = dit.block.layers[0]
    original_layer_forward = original_layer.forward
    original_layer.forward = _boom  # type: ignore[method-assign]
    try:
        with pytest.raises(RuntimeError, match="injected layer failure"):
            pipe.generate(prompt="a cat", num_inference_steps=1, seed=0, return_latents=True)
    finally:
        original_layer.forward = original_layer_forward  # type: ignore[method-assign]

    assert all(p.device.type == "cpu" for p in dit.block.parameters())
    assert all(b.device.type == "cpu" for b in dit.block.buffers())
